"""LR/HR co-registration audit.

Why this is the deliverable and not the dataloader
--------------------------------------------------
A misaligned LR/HR pair does not fail. It trains. The loss falls, PSNR looks
ordinary, and the model learns the only mapping that is consistent with a
sub-pixel-shifted target: blur. Nothing in a training run says "your pairs are
offset by two pixels" -- the first symptom is a submission that underperforms
bicubic interpolation, by which point the GPU hours are spent.

So the pairs get audited before a single one is trained on, and the audit ends
in a verdict that is stated plainly:

- **PASS**  -- median shift magnitude below ``cfg.alignment.pass_threshold_px``
  AND the LR and HR reflectance ranges overlap.
- **WARN**  -- median shift between the pass and warn thresholds, or the shift is
  small but the two sources are not radiometrically comparable.
- **FAIL**  -- median shift above ``cfg.alignment.warn_threshold_px``. Switch
  datasets. Do not train.

Method
------
For each sampled pair: bicubically upsample LR to the HR grid, take the
configured reference band from both, normalise each to zero mean and unit
variance *for registration only*, window them, and estimate the sub-pixel
translation with :func:`skimage.registration.phase_cross_correlation` at
``upsample_factor=10`` -- i.e. shifts resolved to 0.1 HR pixel. The reported
``(dy, dx)`` is the shift **in HR pixels that must be applied to the
bicubic-upsampled LR to register it onto the HR**.

Trusting the estimator
----------------------
An estimator nobody has checked produces a verdict nobody should believe. So the
audit can inject a known shift (``cfg.alignment.inject_shift_px``) into the HR
images and report whether it recovered it. The ``--smoke`` path sets that to
2 px in both axes: it runs a control audit on the aligned synthetic pairs, then
the injected audit, and states whether the estimate came back at 2 within
``cfg.alignment.inject_tolerance_px``. That self-test is what makes the verdict
on real data mean anything.

Units and conventions
---------------------
- Imagery is ``(C, H, W)`` float32 **surface reflectance**, nominally ``[0, 1]``
  and legitimately above 1.0 over cloud, snow, and specular targets. Nothing in
  this module clips it. Bicubic upsampling overshoots at edges and those
  overshoots are kept: they are what the audit is measuring.
- The zero-mean/unit-variance normalisation applied before phase correlation is
  a **registration preprocessing step on a temporary copy**. It never touches the
  arrays a loss, a metric, or a histogram sees, and it is not an ImageNet-style
  normalisation -- it is derived per image from that image.
- Shifts are in **HR pixels** (2.5 m each). One HR pixel is a quarter of an LR
  pixel.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.eval.baselines import bicubic_upsample as _baseline_bicubic

__all__ = [
    "PASS",
    "WARN",
    "FAIL",
    "bicubic_upsample",
    "block_mean_downsample",
    "reference_band_index",
    "inject_shift",
    "estimate_pair_shift",
    "pair_correlations",
    "audit_alignment",
    "select_audit_indices",
    "describe_selection",
    "format_selection_block",
    "read_manifest_records",
    "manifest_validated_ids",
    "grid_cell_of",
    "collect_pairs",
    "reflectance_range_overlap",
    "verdict",
    "format_summary",
    "format_verdict_block",
    "write_report",
]

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


# -- resampling ------------------------------------------------------------


def bicubic_upsample(lr: np.ndarray, scale: int) -> np.ndarray:
    """Bicubically upsample an LR tile to the HR grid, for registration.

    A thin wrapper over :func:`src.eval.baselines.bicubic_upsample` -- there is
    exactly one bicubic implementation in the repository, so the audit and the
    baseline cannot drift apart.

    It passes ``antialias=False``, which is **not** the baseline's default, and
    the difference is not cosmetic: torch's two bicubic kernels use different
    cubic coefficients (``a = -0.75`` here, ``a = -0.5`` with antialiasing) and
    MEASURED differ by up to 0.083 reflectance on a x4 upsample. This function
    keeps the non-antialiased kernel the alignment thresholds in
    ``cfg.alignment`` were measured against; the SR baseline uses the
    antialiased one because that is the kernel the SR literature reports.
    Neither choice affects phase correlation, which cares about structure rather
    than about a fractional difference in edge sharpening.

    The result is **not clipped**. Bicubic interpolation overshoots at sharp
    edges and can push reflectance slightly below 0 or above the input maximum;
    clipping that would quietly change the radiometry the spectral-consistency
    objective depends on, and would also mask exactly the edge behaviour this
    audit exists to measure.

    Args:
        lr: ``(C, H, W)`` float32 surface reflectance, nominally ``[0, 1]``,
            unclipped.
        scale: Upsampling factor, from ``cfg.sr.scale``.

    Returns:
        ``(C, H*scale, W*scale)`` float32 surface reflectance in the same units,
        unclipped.

    Raises:
        ValueError: ``lr`` is not rank 3.
    """
    array = np.asarray(lr)
    if array.ndim != 3:
        raise ValueError(
            f"bicubic_upsample expects (C, H, W), got shape {array.shape}. The "
            "channel axis is first."
        )
    return _baseline_bicubic(
        np.ascontiguousarray(array, dtype=np.float32), int(scale), antialias=False
    )


def block_mean_downsample(hr: np.ndarray, scale: int) -> np.ndarray:
    """Downsample by an exact block mean -- the degradation the SR task inverts.

    A block mean, not a bicubic or a Gaussian: it is the operator the
    spectral-consistency objective uses, so a correlation computed against it
    measures the same relationship the loss will.

    Args:
        hr: ``(C, H, W)`` float32 surface reflectance, unclipped. ``H`` and ``W``
            must be divisible by ``scale``.
        scale: Downsampling factor.

    Returns:
        ``(C, H//scale, W//scale)`` float32 surface reflectance, same units.

    Raises:
        ValueError: ``hr`` is not rank 3, or its dimensions are not divisible by
            ``scale``.
    """
    array = np.asarray(hr, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(
            f"block_mean_downsample expects (C, H, W), got shape {array.shape}."
        )
    channels, height, width = array.shape
    if height % scale or width % scale:
        raise ValueError(
            f"HR tile {height}x{width} is not divisible by scale={scale}, so an "
            "exact block mean is undefined. Crop to a multiple first rather than "
            "resampling -- resampling here would change the degradation model."
        )
    return array.reshape(
        channels, height // scale, scale, width // scale, scale
    ).mean(axis=(2, 4)).astype(np.float32, copy=False)


def reference_band_index(cfg: Any) -> int:
    """Index of ``cfg.alignment.reference_band`` within ``cfg.dataset.bands``.

    Registration runs on one band. Red carries the most land-surface structure of
    the visible bands and is present in every source we might fall back to.

    Raises:
        KeyError: The configured band is not among the dataset's bands. Not
            defaulted to band 0 -- registering on the wrong band would silently
            change what the verdict describes.
    """
    bands = [str(b) for b in cfg["dataset"]["bands"]]
    name = str(cfg["alignment"]["reference_band"])
    if name not in bands:
        raise KeyError(
            f"cfg.alignment.reference_band={name!r} is not in "
            f"cfg.dataset.bands={bands}."
        )
    return bands.index(name)


# -- the estimator ---------------------------------------------------------


def _prepare_for_registration(
    plane: np.ndarray,
    border_crop_px: int,
    window: Optional[str],
) -> np.ndarray:
    """Crop, zero-mean/unit-variance normalise, and window one 2-D plane.

    The normalisation is local to registration: phase correlation compares
    structure, and an amplitude difference between two sources (which is exactly
    what a cross-sensor pair has) would otherwise bias the peak. It is computed
    from this image alone and applied to a copy.

    Args:
        plane: ``(H, W)`` float, one band's reflectance.
        border_crop_px: Pixels trimmed from each edge before anything else.
        window: ``"hann"`` or None. A window suppresses the edge discontinuity
            that otherwise shows up as a spurious zero-shift peak.

    Returns:
        ``(H', W')`` float64, zero mean and unit variance, windowed.

    Raises:
        ValueError: The crop leaves nothing, or the plane is constant (its
            standard deviation is zero), which makes a shift estimate
            meaningless.
    """
    array = np.asarray(plane, dtype=np.float64)
    if border_crop_px:
        crop = int(border_crop_px)
        if array.shape[0] <= 2 * crop or array.shape[1] <= 2 * crop:
            raise ValueError(
                f"border_crop_px={crop} leaves nothing of a "
                f"{array.shape[0]}x{array.shape[1]} image. Lower "
                "cfg.alignment.border_crop_px."
            )
        array = array[crop:-crop, crop:-crop]

    std = float(array.std())
    if std <= 0.0:
        raise ValueError(
            "Plane is constant (std=0), so a translation estimate is undefined. "
            "This is a degenerate tile -- solid nodata or a flat fill -- not a "
            "registration failure."
        )
    array = (array - float(array.mean())) / std

    if window:
        if window != "hann":
            raise ValueError(
                f"cfg.alignment.window={window!r} is not supported; use 'hann' "
                "or null."
            )
        from skimage.filters import window as sk_window

        array = array * sk_window("hann", array.shape)
    return array


def estimate_pair_shift(
    lr: np.ndarray,
    hr: np.ndarray,
    scale: int,
    band_index: int,
    upsample_factor: int,
    border_crop_px: int = 0,
    window: Optional[str] = "hann",
    normalization: Optional[str] = None,
) -> Dict[str, float]:
    """Estimate the sub-pixel translation between an LR/HR pair.

    Args:
        lr: ``(C, H, W)`` float32 surface reflectance, unclipped.
        hr: ``(C, H*scale, W*scale)`` float32 surface reflectance, same band
            order and units.
        scale: Super-resolution factor.
        band_index: Channel used for registration, from
            :func:`reference_band_index`.
        upsample_factor: Sub-pixel resolution; 10 resolves 0.1 HR pixel.
        border_crop_px: HR pixels trimmed from each edge before registration.
        window: ``"hann"`` or None.
        normalization: None for plain cross-correlation (the default, and the
            measured-correct choice here -- see cfg.alignment.normalization) or
            ``"phase"`` for skimage's phase correlation. Phase normalisation
            weights every frequency equally, including the high frequencies
            where an upsampled LR image has no energy, and on LR/HR pairs that
            produced a 1.4 px phantom shift on imagery aligned by construction.

    Returns:
        ``{"dy", "dx", "shift_magnitude", "phase_error", "phase_diff"}``.
        ``dy`` and ``dx`` are in **HR pixels** and are the shift that must be
        applied to the bicubic-upsampled LR to register it onto the HR: positive
        ``dy`` means the upsampled LR must move **down**, positive ``dx`` means it
        must move **right**. ``shift_magnitude`` is ``hypot(dy, dx)``.

    Raises:
        ValueError: Either image is degenerate (constant) after cropping, or the
            shapes do not satisfy the scale relation.
    """
    from skimage.registration import phase_cross_correlation

    upsampled = bicubic_upsample(lr, scale)
    hr_array = np.asarray(hr, dtype=np.float32)
    if upsampled.shape != hr_array.shape:
        raise ValueError(
            f"Upsampled LR is {upsampled.shape} but HR is {hr_array.shape}. The "
            f"pair does not satisfy scale={scale}."
        )

    moving = _prepare_for_registration(
        upsampled[band_index], border_crop_px, window
    )
    reference = _prepare_for_registration(
        hr_array[band_index], border_crop_px, window
    )

    shift, error, phase_diff = phase_cross_correlation(
        reference_image=reference,
        moving_image=moving,
        upsample_factor=int(upsample_factor),
        normalization=normalization,
    )
    dy, dx = float(shift[0]), float(shift[1])
    return {
        "dy": dy,
        "dx": dx,
        "shift_magnitude": float(np.hypot(dy, dx)),
        "phase_error": float(error),
        "phase_diff": float(phase_diff),
    }


def pair_correlations(
    lr: np.ndarray,
    hr: np.ndarray,
    scale: int,
    band_index: int,
) -> Dict[str, float]:
    """Correlate the bicubic-upsampled LR against the HR, on both grids.

    Args:
        lr: ``(C, H, W)`` float32 surface reflectance.
        hr: ``(C, H*scale, W*scale)`` float32 surface reflectance.
        scale: Super-resolution factor.
        band_index: Channel to correlate.

    Returns:
        ``{"correlation_lr_grid", "correlation_hr_grid"}``, both Pearson r on the
        reference band.

        ``correlation_lr_grid`` compares the upsampled LR and the HR after both
        are block-mean-downsampled by ``scale`` -- i.e. at the LR grid, where
        sub-pixel misregistration is averaged away. It answers "do these two
        images show the same place at all".

        ``correlation_hr_grid`` compares them at full HR resolution, where a
        sub-pixel shift *does* decorrelate. Read together, a high LR-grid
        correlation with a low HR-grid one is the signature of a pair that is the
        right scene but misaligned.
    """
    upsampled = bicubic_upsample(lr, scale)
    hr_array = np.asarray(hr, dtype=np.float32)

    hr_grid = _pearson(upsampled[band_index], hr_array[band_index])
    lr_grid = _pearson(
        block_mean_downsample(upsampled, scale)[band_index],
        block_mean_downsample(hr_array, scale)[band_index],
    )
    return {"correlation_lr_grid": lr_grid, "correlation_hr_grid": hr_grid}


def _normalization_from_cfg(align_cfg: Mapping[str, Any]) -> Optional[str]:
    """Read ``cfg.alignment.normalization`` as skimage expects it.

    ``"none"`` in YAML (and YAML's own ``null``) both mean plain
    cross-correlation, which skimage spells as ``None``. See the config comment
    for why that, not phase normalisation, is the measured default here.

    Raises:
        ValueError: The value is neither ``"none"`` nor ``"phase"``.
    """
    value = align_cfg["normalization"]
    if value is None or str(value).lower() == "none":
        return None
    if str(value).lower() == "phase":
        return "phase"
    raise ValueError(
        f"cfg.alignment.normalization={value!r} must be 'none' or 'phase'."
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equally shaped arrays, as a float.

    Returns NaN when either side is constant -- correlation is genuinely
    undefined there, and NaN propagates into the report where it is counted,
    rather than a 0.0 that would read as "uncorrelated".
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x_std, y_std = x.std(), y.std()
    if x_std <= 0 or y_std <= 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (x_std * y_std))


def inject_shift(hr: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Translate an HR image by a known sub-pixel amount. QA scaffolding only.

    Used to prove the estimator recovers a shift it was given. Never called on
    training data -- the only callers are the self-test path of
    :func:`audit_alignment` and the tests.

    The shift is applied with a cubic spline (``scipy.ndimage.shift``,
    ``order=3``) and ``mode="reflect"`` at the border. Interpolation slightly
    smooths the image; that is why the audit crops
    ``cfg.alignment.border_crop_px`` before registering. Reflectance is not
    clipped.

    Args:
        hr: ``(C, H, W)`` float32 surface reflectance, unclipped.
        dy: Rows to shift by, in HR pixels. Positive moves content **down**.
        dx: Columns to shift by, in HR pixels. Positive moves content **right**.

    Returns:
        ``(C, H, W)`` float32, shifted, same reflectance units.
    """
    from scipy.ndimage import shift as nd_shift

    array = np.asarray(hr, dtype=np.float32)
    out = np.empty_like(array)
    for channel in range(array.shape[0]):
        out[channel] = nd_shift(
            array[channel], shift=(float(dy), float(dx)), order=3, mode="reflect"
        )
    return out


# -- pair selection --------------------------------------------------------


def read_manifest_records(manifest_path: Any) -> List[Dict[str, str]]:
    """Read the rows of a ``build_index`` manifest CSV.

    Args:
        manifest_path: Path to the manifest CSV written by
            :meth:`~src.data.base.SRPairDataset.build_index`.

    Returns:
        One dict per row, carrying at least ``index``, ``sample_id`` and
        ``validation_error``.

    Raises:
        FileNotFoundError: The manifest does not exist. A missing manifest is
            never treated as "no filter" -- that would silently widen the audit
            pool back to samples the index never validated.
        RuntimeError: The CSV is empty, or lacks the columns the audit needs.
    """
    import csv

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest {path} does not exist. Run scripts/prepare_data.py to "
            "build it, or pass --no-manifest-filter to audit every cached "
            "sample regardless of whether the index validated it."
        )

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"Manifest {path} has no rows.")

    required = {"index", "sample_id", "validation_error"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(
            f"Manifest {path} is missing column(s) {sorted(missing)}; it has "
            f"{sorted(rows[0])}. This is not a manifest written by "
            "SRPairDataset.build_index -- check --manifest."
        )
    return rows


def manifest_validated_ids(manifest_path: Any) -> Tuple[set, Dict[int, str]]:
    """Sample ids from a manifest whose ``validation_error`` column is empty.

    Args:
        manifest_path: Path to the manifest CSV.

    Returns:
        ``(validated_ids, id_by_index)`` -- the set of ``sample_id`` values with
        neither a ``validation_error`` nor a truthy ``rejected`` flag, and the full ``index -> sample_id`` map (including
        the failures, so a caller can still name a rejected sample).

    Raises:
        RuntimeError: Every row failed validation, so there is nothing to audit.
    """
    rows = read_manifest_records(manifest_path)

    validated: set = set()
    id_by_index: Dict[int, str] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        index = row.get("index")
        if index not in (None, ""):
            id_by_index[int(index)] = sample_id
        # A row is eligible only if it neither errored nor was rejected by
        # policy. `rejected` arrives from the SEN2NAIPv2 index, where an
        # over-nodata sample is recorded rather than raised -- so a manifest
        # that carries the column must be read through it, or the audit
        # measures shifts on pairs training will never see.
        errored = str(row.get("validation_error") or "").strip()
        rejected = str(row.get("rejected") or "").strip().lower() in ("true", "1")
        if not errored and not rejected:
            validated.add(sample_id)

    if not validated:
        raise RuntimeError(
            f"Every one of the {len(rows)} rows in {manifest_path} carries a "
            "validation_error, so there is no validated sample to audit. Fix "
            "the errors the manifest reports before auditing alignment."
        )
    return validated, id_by_index


# Catalog columns that, when present, name the geographic grouping outright.
# Checked in this order; the first one every selected entry carries becomes the
# primary grouping reported in the audit.
REGION_FIELDS = ("region", "mgrs_tile", "mgrs", "tile", "grid_cell", "scene")


def grid_cell_of(sample_id: Any) -> Optional[str]:
    """The SEN2NAIP grid-cell prefix of a sample id, or None.

    SEN2NAIPv2 ids look like
    ``NA5120_E1183N0757__m_3912321_nw_10_060_20220710``: a grid cell, ``"__"``,
    then the NAIP quarter-quad. The prefix is the geographic unit -- records
    sharing it image the same place, which is exactly what a spread claim needs.

    Args:
        sample_id: The dataset's sample id.

    Returns:
        The substring before ``"__"``, or None when the id has no such
        separator, i.e. this dataset does not encode a cell in its ids.
    """
    text = str(sample_id)
    return text.split("__", 1)[0] if "__" in text else None


def _counts(values: Sequence[str]) -> Dict[str, int]:
    """Value counts, ordered by descending count then by value."""
    from collections import Counter

    counter = Counter(values)
    return {
        str(k): int(v)
        for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    }


def _groupings_for(
    indices: Sequence[int],
    sample_ids: Sequence[str],
    catalog: Optional[Sequence[Mapping[str, Any]]],
) -> Tuple[Optional[str], Dict[str, Dict[str, int]]]:
    """Build ``{grouping name: {value: count}}`` for the selected pairs.

    Args:
        indices: Selected dataset indices.
        sample_ids: Sample id per selected index, in the same order.
        catalog: The dataset's catalog, or None when it has none.

    Returns:
        ``(primary_name, groupings)``. ``primary_name`` is None when nothing in
        the data identifies a region; that is reported, never invented.
    """
    groupings: Dict[str, Dict[str, int]] = {}
    entries = [catalog[int(i)] for i in indices] if catalog is not None else None

    primary: Optional[str] = None
    if entries:
        for field in REGION_FIELDS:
            values = [entry.get(field) for entry in entries]
            if all(v not in (None, "") for v in values):
                groupings[field] = _counts([str(v) for v in values])
                primary = field
                break

    cells = [grid_cell_of(sid) for sid in sample_ids]
    if cells and all(c is not None for c in cells):
        groupings["grid_cell"] = _counts([str(c) for c in cells])
        primary = primary or "grid_cell"

    if entries:
        crs_values = [entry.get("crs") for entry in entries]
        if all(v not in (None, "") for v in crs_values):
            groupings["crs"] = _counts([str(v) for v in crs_values])
            primary = primary or "crs"

    return primary, groupings


def select_audit_indices(
    dataset: Any,
    num_pairs: int,
    seed: int,
    cached_only: bool = True,
    logger: Any = None,
    manifest_path: Any = None,
) -> List[int]:
    """Choose which dataset samples to audit, reproducibly.

    Drawn without replacement, with ``seed``, from the samples that are actually
    available, then sorted -- so the selection is reproducible from the seed and
    the cache is read in order. Sampling rather than taking the first N is the
    whole point: SEN2NAIPv2 records are ordered geographically, so the first 50
    are one corner of one state under one sensor geometry, and an audit of them
    describes that corner rather than the archive.

    Args:
        dataset: An :class:`~src.data.base.SRPairDataset`.
        num_pairs: How many to audit. Capped at what is available.
        seed: From ``cfg.seed``. The same seed over the same available set
            always produces the same selection.
        cached_only: Restrict to samples already in the local cache. The full
            download takes hours; this keeps the audit from triggering fetches.
        logger: Logger for the availability counts.
        manifest_path: Manifest CSV restricting the pool to samples whose
            ``validation_error`` is empty. None audits every available sample
            and the report says so.

    Returns:
        Sorted dataset indices.

    Raises:
        RuntimeError: Nothing is available to audit.
    """
    total = len(dataset)
    is_cached = getattr(dataset, "is_cached", None)
    if cached_only and callable(is_cached):
        available = [i for i in range(total) if is_cached(i)]
    else:
        available = list(range(total))

    if manifest_path is not None:
        validated, id_by_index = manifest_validated_ids(manifest_path)
        catalog = getattr(dataset, "catalog", None)
        before = len(available)
        if catalog is not None:
            available = [
                i for i in available if str(catalog[i]["sample_id"]) in validated
            ]
        else:
            available = [i for i in available if id_by_index.get(i, "") in validated]
        if logger is not None:
            logger.info(
                "Manifest filter %s: %d -> %d samples eligible (rows carrying a "
                "validation_error are excluded from the audit pool).",
                manifest_path,
                before,
                len(available),
            )

    if not available:
        raise RuntimeError(
            f"No samples available to audit ({total} in the catalog, "
            f"cached_only={cached_only}, manifest_path={manifest_path}). Run "
            "scripts/prepare_data.py first, set cfg.loader.cached_only=false to "
            "fetch on demand, or pass --no-manifest-filter."
        )

    if logger is not None and len(available) < total:
        logger.warning(
            "Auditing the AVAILABLE SUBSET ONLY: %d of %d catalog samples are "
            "eligible. The verdict below describes those %d; re-run after the "
            "download completes to audit the rest.",
            len(available),
            total,
            len(available),
        )

    count = min(int(num_pairs), len(available))
    if count < int(num_pairs) and logger is not None:
        logger.warning(
            "Requested %d pairs but only %d are available; auditing %d.",
            int(num_pairs),
            len(available),
            count,
        )

    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(np.asarray(available), size=count, replace=False)
    return sorted(int(i) for i in chosen)


def describe_selection(
    dataset: Any,
    indices: Sequence[int],
    seed: int,
    cached_only: bool = True,
    manifest_path: Any = None,
) -> Dict[str, Any]:
    """Record exactly which pairs the audit covered, and how they spread out.

    The audit's credibility rests on the selection as much as on the median
    shift: "50 pairs" is only defensible if those 50 span the archive. This
    returns the evidence -- the sample ids and their distribution across
    whatever geographic key the data actually carries -- so it goes into the
    report and can be quoted.

    Args:
        dataset: The dataset the indices index into.
        indices: The selected indices, from :func:`select_audit_indices`.
        seed: The seed they were drawn with, recorded for reproducibility.
        cached_only: Recorded, so the report states what the pool was.
        manifest_path: Recorded, or None when no manifest filter was applied.

    Returns:
        ``sampling`` (``"random_without_replacement"``), ``seed``,
        ``cached_only``, ``manifest_path``, ``num_catalog``, ``num_selected``,
        ``indices``, ``sample_ids``, ``primary_grouping``,
        ``num_distinct_regions`` and ``groupings`` (``{name: {value: count}}``).
        ``primary_grouping`` is None when nothing in the data identifies a
        region -- reported rather than invented.
    """
    catalog = getattr(dataset, "catalog", None)

    id_by_index: Dict[int, str] = {}
    if manifest_path is not None:
        _, id_by_index = manifest_validated_ids(manifest_path)

    sample_ids: List[str] = []
    for raw_idx in indices:
        idx = int(raw_idx)
        if catalog is not None:
            sample_ids.append(str(catalog[idx]["sample_id"]))
        elif idx in id_by_index:
            sample_ids.append(str(id_by_index[idx]))
        else:
            sample_ids.append(str(dataset[idx]["meta"]["sample_id"]))

    primary, groupings = _groupings_for(indices, sample_ids, catalog)

    return {
        "sampling": "random_without_replacement",
        "seed": int(seed),
        "cached_only": bool(cached_only),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "num_catalog": int(len(dataset)),
        "num_selected": len(sample_ids),
        "indices": [int(i) for i in indices],
        "sample_ids": sample_ids,
        "primary_grouping": primary,
        "num_distinct_regions": len(groupings[primary]) if primary else None,
        "groupings": groupings,
    }


def format_selection_block(selection: Mapping[str, Any]) -> str:
    """Render :func:`describe_selection` as a printable block.

    Every selected sample id is listed. That list is the audit's provenance: it
    is what lets a reviewer reproduce the selection, and what backs the claim
    that the audit spanned N regions rather than N pairs from one place.
    """
    lines = [
        f"Audit sample -- {selection['num_selected']} of "
        f"{selection['num_catalog']} catalog records, drawn without "
        f"replacement with seed {selection['seed']}",
        f"  pool: cached_only={selection['cached_only']}, manifest filter = "
        f"{selection['manifest_path'] or 'NONE (validation NOT enforced)'}",
    ]

    primary = selection.get("primary_grouping")
    if primary:
        lines.append(
            f"  spans {selection['num_distinct_regions']} distinct {primary} "
            f"value(s) across {selection['num_selected']} pairs"
        )
    else:
        lines.append(
            "  no region / tile / MGRS key is present in this dataset's catalog "
            "or sample ids, so geographic spread CANNOT be stated."
        )

    for name, counts in selection.get("groupings", {}).items():
        marker = " (primary)" if name == primary else ""
        lines.append("")
        lines.append(f"  distribution by {name}{marker} -- {len(counts)} distinct:")
        for value, count in counts.items():
            lines.append(f"    {count:>4}  {value}")

    lines.append("")
    lines.append("  selected sample ids:")
    for idx, sample_id in zip(selection["indices"], selection["sample_ids"]):
        lines.append(f"    [{idx:>5}]  {sample_id}")
    return "\n".join(lines)


def collect_pairs(
    dataset: Any,
    indices: Sequence[int],
    inject_shift_px: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Load the sampled pairs into memory as NumPy reflectance arrays.

    Args:
        dataset: The dataset to read from.
        indices: Sample indices, from :func:`select_audit_indices`.
        inject_shift_px: ``(dy, dx)`` in HR pixels applied to every HR image via
            :func:`inject_shift`, for the estimator self-test. None leaves the
            pairs untouched, which is the only mode used on real data.

    Returns:
        One dict per pair with ``index`` (int), ``sample_id`` (str), ``lr``
        (``(C, H, W)`` float32 reflectance), ``hr`` (``(C, H*scale, W*scale)``
        float32 reflectance), and ``injected_shift`` (``(dy, dx)`` or None).
    """
    pairs: List[Dict[str, Any]] = []
    for idx in indices:
        sample = dataset[int(idx)]
        lr = np.asarray(sample["lr"], dtype=np.float32)
        hr = np.asarray(sample["hr"], dtype=np.float32)
        injected = None
        if inject_shift_px is not None:
            dy, dx = float(inject_shift_px[0]), float(inject_shift_px[1])
            hr = inject_shift(hr, dy, dx)
            injected = (dy, dx)
        pairs.append(
            {
                "index": int(idx),
                "sample_id": str(sample["meta"]["sample_id"]),
                "lr": lr,
                "hr": hr,
                "injected_shift": injected,
            }
        )
    return pairs


# -- radiometry ------------------------------------------------------------


def reflectance_range_overlap(
    pairs: Sequence[Mapping[str, Any]],
    band_names: Sequence[str],
    percentiles: Tuple[float, float],
    min_overlap: float,
) -> Dict[str, Any]:
    """Compare the LR and HR reflectance ranges, band by band.

    Two sources that do not share a radiometric range cannot both be surface
    reflectance of the same scene, and a model trained across the gap learns a
    brightness correction instead of super-resolution. Ranges are compared as the
    interval between the configured percentiles (default p1-p99, so a handful of
    specular pixels do not decide the verdict), scored by intersection over
    union.

    Args:
        pairs: Loaded pairs with ``lr`` and ``hr`` arrays, ``(C, H, W)`` float32
            reflectance.
        band_names: Band names in channel order, from ``cfg.dataset.bands``.
        percentiles: ``(low, high)`` percentiles defining a source's range.
        min_overlap: The per-band IoU below which the sources are judged not
            comparable.

    Returns:
        ``{"bands": {name: {...}}, "min_overlap": float, "ok": bool,
        "min_overlap_band": str}``. Each band entry carries ``lr_low``,
        ``lr_high``, ``lr_median``, the ``hr_`` equivalents (all reflectance),
        and ``overlap``.
    """
    low_pct, high_pct = float(percentiles[0]), float(percentiles[1])
    lr_stack = np.concatenate(
        [np.asarray(p["lr"], dtype=np.float32).reshape(len(band_names), -1) for p in pairs],
        axis=1,
    )
    hr_stack = np.concatenate(
        [np.asarray(p["hr"], dtype=np.float32).reshape(len(band_names), -1) for p in pairs],
        axis=1,
    )

    bands: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(band_names):
        lr_low, lr_high = np.percentile(lr_stack[index], [low_pct, high_pct])
        hr_low, hr_high = np.percentile(hr_stack[index], [low_pct, high_pct])
        inter = max(0.0, min(float(lr_high), float(hr_high)) - max(float(lr_low), float(hr_low)))
        union = max(float(lr_high), float(hr_high)) - min(float(lr_low), float(hr_low))
        bands[str(name)] = {
            "lr_low": float(lr_low),
            "lr_high": float(lr_high),
            "lr_median": float(np.median(lr_stack[index])),
            "hr_low": float(hr_low),
            "hr_high": float(hr_high),
            "hr_median": float(np.median(hr_stack[index])),
            "overlap": float(inter / union) if union > 0 else 0.0,
        }

    worst_band = min(bands, key=lambda b: bands[b]["overlap"])
    worst = bands[worst_band]["overlap"]
    return {
        "bands": bands,
        "percentiles": [low_pct, high_pct],
        "min_overlap": worst,
        "min_overlap_band": worst_band,
        "min_overlap_required": float(min_overlap),
        "ok": bool(worst >= float(min_overlap)),
    }


# -- verdict ---------------------------------------------------------------


def verdict(
    median_shift_px: float,
    ranges_overlap: bool,
    pass_threshold_px: float,
    warn_threshold_px: float,
) -> Dict[str, Any]:
    """Turn the measurements into PASS / WARN / FAIL.

    The rule, stated once and applied without softening:

    - ``median_shift_px > warn_threshold_px`` -> **FAIL**. The pairs are
      misaligned by more than half an LR pixel and the dataset should be
      replaced, not worked around.
    - ``pass_threshold_px <= median_shift_px <= warn_threshold_px`` -> **WARN**.
    - Below ``pass_threshold_px`` -> **PASS**, unless the reflectance ranges do
      not overlap, which downgrades it to **WARN**: geometry is fine but the two
      sources are not radiometrically comparable.

    Args:
        median_shift_px: Median shift magnitude over the audited pairs, in HR
            pixels.
        ranges_overlap: From :func:`reflectance_range_overlap`.
        pass_threshold_px: ``cfg.alignment.pass_threshold_px``.
        warn_threshold_px: ``cfg.alignment.warn_threshold_px``.

    Returns:
        ``{"level", "reasons", "median_shift_px", "thresholds"}``.

    Raises:
        ValueError: The thresholds are not ordered, or the median is NaN --
            which means every pair was degenerate and there is no verdict to
            give.
    """
    if not pass_threshold_px < warn_threshold_px:
        raise ValueError(
            f"pass_threshold_px={pass_threshold_px} must be below "
            f"warn_threshold_px={warn_threshold_px}."
        )
    if not np.isfinite(median_shift_px):
        raise ValueError(
            "Median shift is not finite: no pair produced a usable estimate. "
            "That is a broken audit, not a passing dataset."
        )

    reasons: List[str] = []
    if median_shift_px > warn_threshold_px:
        level = FAIL
        reasons.append(
            f"median shift {median_shift_px:.2f} HR px exceeds the FAIL "
            f"threshold of {warn_threshold_px:.2f} px "
            f"({median_shift_px / 4.0:.2f} LR pixels of misregistration)"
        )
    elif median_shift_px >= pass_threshold_px:
        level = WARN
        reasons.append(
            f"median shift {median_shift_px:.2f} HR px is between the PASS "
            f"({pass_threshold_px:.2f}) and FAIL ({warn_threshold_px:.2f}) "
            "thresholds"
        )
    else:
        level = PASS
        reasons.append(
            f"median shift {median_shift_px:.2f} HR px is below the PASS "
            f"threshold of {pass_threshold_px:.2f} px"
        )

    if ranges_overlap:
        reasons.append("LR and HR reflectance ranges overlap")
    else:
        reasons.append(
            "LR and HR reflectance ranges do NOT overlap: the two sources are "
            "not radiometrically comparable"
        )
        if level == PASS:
            level = WARN

    return {
        "level": level,
        "reasons": reasons,
        "median_shift_px": float(median_shift_px),
        "thresholds": {
            "pass_below_px": float(pass_threshold_px),
            "fail_above_px": float(warn_threshold_px),
        },
    }


# -- the audit -------------------------------------------------------------


def audit_alignment(
    pairs: Sequence[Mapping[str, Any]],
    cfg: Any,
    logger: Any = None,
    label: str = "audit",
) -> Dict[str, Any]:
    """Measure shift and correlation over a set of pairs and reach a verdict.

    Args:
        pairs: Loaded pairs from :func:`collect_pairs`.
        cfg: The loaded config; reads ``alignment``, ``dataset``, and ``sr``.
        logger: Logger for progress and for the degenerate-pair count.
        label: Name for this audit in logs, e.g. ``"control"`` or ``"injected"``.

    Returns:
        A JSON-serialisable report: ``label``, ``created_utc``, ``dataset``,
        ``num_pairs``, ``settings``, ``pairs`` (per-pair records with ``dy``,
        ``dx``, ``shift_magnitude`` in HR pixels and the two correlations),
        ``shift`` (median/p90/mean/max magnitude and median ``dy``/``dx``),
        ``correlation``, ``radiometry`` from
        :func:`reflectance_range_overlap`, ``degenerate`` (pairs excluded, with
        the reason), and ``verdict``.

    Raises:
        ValueError: ``pairs`` is empty, or every pair was degenerate.
    """
    if not pairs:
        raise ValueError("audit_alignment received no pairs.")

    align_cfg = cfg["alignment"]
    scale = int(cfg["sr"]["scale"])
    band_names = [str(b) for b in cfg["dataset"]["bands"]]
    band_index = reference_band_index(cfg)
    upsample_factor = int(align_cfg["upsample_factor"])
    border_crop_px = int(align_cfg["border_crop_px"])
    window = align_cfg["window"]
    window = str(window) if window else None
    normalization = _normalization_from_cfg(align_cfg)

    records: List[Dict[str, Any]] = []
    degenerate: List[Dict[str, str]] = []

    for pair in pairs:
        try:
            shift = estimate_pair_shift(
                pair["lr"],
                pair["hr"],
                scale=scale,
                band_index=band_index,
                upsample_factor=upsample_factor,
                border_crop_px=border_crop_px,
                window=window,
                normalization=normalization,
            )
        except ValueError as exc:
            # A constant tile has no translation to estimate. It is EXCLUDED and
            # RECORDED -- never counted as a zero shift, which would drag the
            # median toward PASS.
            degenerate.append({"sample_id": pair["sample_id"], "reason": str(exc)})
            if logger is not None:
                logger.warning(
                    "Pair %s excluded from the shift statistics: %s",
                    pair["sample_id"],
                    exc,
                )
            continue

        correlations = pair_correlations(
            pair["lr"], pair["hr"], scale=scale, band_index=band_index
        )
        record = {
            "index": pair["index"],
            "sample_id": pair["sample_id"],
            **shift,
            **correlations,
            "injected_shift": list(pair["injected_shift"])
            if pair.get("injected_shift") is not None
            else None,
        }
        records.append(record)

    if not records:
        raise ValueError(
            f"Every one of the {len(pairs)} audited pairs was degenerate "
            "(constant imagery). There is no alignment verdict to give -- check "
            "the cache and the nodata handling."
        )

    magnitudes = np.array([r["shift_magnitude"] for r in records], dtype=np.float64)
    dys = np.array([r["dy"] for r in records], dtype=np.float64)
    dxs = np.array([r["dx"] for r in records], dtype=np.float64)
    lr_grid = np.array([r["correlation_lr_grid"] for r in records], dtype=np.float64)
    hr_grid = np.array([r["correlation_hr_grid"] for r in records], dtype=np.float64)

    radiometry = reflectance_range_overlap(
        pairs,
        band_names,
        percentiles=tuple(float(p) for p in align_cfg["range_percentiles"]),
        min_overlap=float(align_cfg["range_overlap_min"]),
    )

    median_shift = float(np.median(magnitudes))
    result_verdict = verdict(
        median_shift,
        ranges_overlap=bool(radiometry["ok"]),
        pass_threshold_px=float(align_cfg["pass_threshold_px"]),
        warn_threshold_px=float(align_cfg["warn_threshold_px"]),
    )

    report = {
        "label": label,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(cfg["dataset"]["name"]),
        "num_pairs": len(records),
        "num_pairs_requested": len(pairs),
        "settings": {
            "scale": scale,
            "reference_band": str(align_cfg["reference_band"]),
            "bands": band_names,
            "upsample_factor": upsample_factor,
            "border_crop_px": border_crop_px,
            "window": window,
            "normalization": normalization or "none",
            "hr_gsd_m": float(cfg["dataset"]["hr_gsd_m"]),
            "injected_shift_px": records[0]["injected_shift"],
        },
        "shift": {
            "units": "HR pixels",
            "median_magnitude": median_shift,
            "p90_magnitude": float(np.percentile(magnitudes, 90)),
            "mean_magnitude": float(magnitudes.mean()),
            "max_magnitude": float(magnitudes.max()),
            "median_dy": float(np.median(dys)),
            "median_dx": float(np.median(dxs)),
            "mean_dy": float(dys.mean()),
            "mean_dx": float(dxs.mean()),
            "std_dy": float(dys.std()),
            "std_dx": float(dxs.std()),
        },
        "correlation": {
            "median_lr_grid": float(np.nanmedian(lr_grid)),
            "min_lr_grid": float(np.nanmin(lr_grid)),
            "median_hr_grid": float(np.nanmedian(hr_grid)),
            "min_hr_grid": float(np.nanmin(hr_grid)),
            "n_undefined": int(np.isnan(lr_grid).sum() + np.isnan(hr_grid).sum()),
        },
        "radiometry": radiometry,
        "degenerate": degenerate,
        "verdict": result_verdict,
        "pairs": records,
    }

    if logger is not None:
        logger.info(
            "[%s] %d pairs: median shift %.2f HR px (p90 %.2f), median dy=%.2f "
            "dx=%.2f, verdict %s",
            label,
            len(records),
            median_shift,
            report["shift"]["p90_magnitude"],
            report["shift"]["median_dy"],
            report["shift"]["median_dx"],
            result_verdict["level"],
        )
    return report


def self_test_result(
    injected_report: Mapping[str, Any],
    control_report: Optional[Mapping[str, Any]],
    injected_shift_px: Sequence[float],
    tolerance_px: float,
) -> Dict[str, Any]:
    """Score the estimator against a known injected shift.

    Args:
        injected_report: The audit of pairs whose HR was shifted by
            ``injected_shift_px``.
        control_report: The audit of the same pairs unshifted, or None. Used to
            check the estimator does not invent a shift where there is none.
        injected_shift_px: The ``(dy, dx)`` that was injected, in HR pixels.
        tolerance_px: How far the recovered shift may be from the injected one,
            per axis, in HR pixels.

    Returns:
        ``{"passed", "injected", "recovered", "error_dy", "error_dx",
        "tolerance_px", "control_median_shift_px", "notes"}``. ``passed`` is True
        only when both axes are recovered within tolerance and, when a control
        was supplied, the control's median shift is itself within tolerance of
        zero.
    """
    injected_dy, injected_dx = float(injected_shift_px[0]), float(injected_shift_px[1])
    recovered_dy = float(injected_report["shift"]["median_dy"])
    recovered_dx = float(injected_report["shift"]["median_dx"])
    error_dy = recovered_dy - injected_dy
    error_dx = recovered_dx - injected_dx

    notes: List[str] = []
    passed = abs(error_dy) <= tolerance_px and abs(error_dx) <= tolerance_px
    if not passed:
        notes.append(
            f"recovered ({recovered_dy:+.2f}, {recovered_dx:+.2f}) HR px vs "
            f"injected ({injected_dy:+.2f}, {injected_dx:+.2f}); error "
            f"({error_dy:+.2f}, {error_dx:+.2f}) exceeds +/-{tolerance_px} px"
        )

    control_median = None
    if control_report is not None:
        control_median = float(control_report["shift"]["median_magnitude"])
        if control_median > tolerance_px:
            passed = False
            notes.append(
                f"control (unshifted) pairs measured {control_median:.2f} HR px "
                f"of shift, above the {tolerance_px} px tolerance: the estimator "
                "reports a shift where there is none"
            )

    return {
        "passed": bool(passed),
        "injected": [injected_dy, injected_dx],
        "recovered": [recovered_dy, recovered_dx],
        "error_dy": error_dy,
        "error_dx": error_dx,
        "tolerance_px": float(tolerance_px),
        "control_median_shift_px": control_median,
        "notes": notes,
    }


# -- reporting -------------------------------------------------------------


def write_report(report: Mapping[str, Any], path: Any) -> Path:
    """Write an audit report as JSON, creating parent directories.

    Args:
        report: From :func:`audit_alignment`, optionally with a ``self_test``
            entry added.
        path: Destination, e.g. ``outputs/metrics/alignment_report.json``.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
    return out


def format_summary(report: Mapping[str, Any]) -> str:
    """Render the measured numbers as a printable block (no verdict)."""
    shift = report["shift"]
    corr = report["correlation"]
    radiometry = report["radiometry"]
    hr_gsd = float(report["settings"]["hr_gsd_m"])

    lines = [
        f"LR/HR alignment audit -- {report['dataset']} "
        f"({report['num_pairs']} pairs, band "
        f"{report['settings']['reference_band']}, upsample_factor="
        f"{report['settings']['upsample_factor']})",
        "",
        "  shift (HR pixels; 1 px = "
        f"{hr_gsd:g} m, 4 HR px = 1 LR px)",
        f"    median magnitude   {shift['median_magnitude']:.3f} px "
        f"({shift['median_magnitude'] * hr_gsd:.2f} m)",
        f"    p90 magnitude      {shift['p90_magnitude']:.3f} px",
        f"    max magnitude      {shift['max_magnitude']:.3f} px",
        f"    median (dy, dx)    ({shift['median_dy']:+.3f}, {shift['median_dx']:+.3f})",
        f"    std (dy, dx)       ({shift['std_dy']:.3f}, {shift['std_dx']:.3f})",
        "",
        "  correlation (upsampled LR vs HR)",
        f"    at LR grid         median {corr['median_lr_grid']:.4f}  "
        f"min {corr['min_lr_grid']:.4f}",
        f"    at HR grid         median {corr['median_hr_grid']:.4f}  "
        f"min {corr['min_hr_grid']:.4f}",
        "",
        "  reflectance range overlap (p"
        f"{radiometry['percentiles'][0]:g}-p{radiometry['percentiles'][1]:g}, "
        f"required >= {radiometry['min_overlap_required']:.2f})",
    ]
    for name, band in radiometry["bands"].items():
        lines.append(
            f"    {name:<5} LR [{band['lr_low']:.4f}, {band['lr_high']:.4f}]  "
            f"HR [{band['hr_low']:.4f}, {band['hr_high']:.4f}]  "
            f"overlap {band['overlap']:.3f}"
        )
    if report.get("degenerate"):
        lines += [
            "",
            f"  EXCLUDED: {len(report['degenerate'])} degenerate pair(s) had no "
            "estimable shift (constant imagery); they are listed in the JSON "
            "report and are NOT counted as zero shift.",
        ]
    return "\n".join(lines)


def format_verdict_block(report: Mapping[str, Any]) -> str:
    """Render the verdict as an unmissable block.

    A FAIL is not softened, hedged, or buried under caveats: the recommended
    action is printed in the block.
    """
    result = report["verdict"]
    level = result["level"]
    rule = (
        f"PASS < {result['thresholds']['pass_below_px']:.1f} px"
        f"  |  WARN {result['thresholds']['pass_below_px']:.1f}-"
        f"{result['thresholds']['fail_above_px']:.1f} px"
        f"  |  FAIL > {result['thresholds']['fail_above_px']:.1f} px"
    )
    action = {
        PASS: "Pairs are co-registered. Proceed to training.",
        WARN: (
            "Borderline. Train only with the shift documented in the report, and "
            "re-audit after any change to the degradation or cropping path."
        ),
        FAIL: (
            "DO NOT TRAIN ON THIS DATA. The pairs are misaligned by more than "
            "half an LR pixel; a model fitted to them will learn to blur, and no "
            "loss or metric in the training loop will say so. Switch datasets or "
            "co-register the pairs first."
        ),
    }[level]

    width = 74
    lines = [
        "=" * width,
        f"  ALIGNMENT VERDICT: {level}",
        "=" * width,
        f"  rule: {rule}",
    ]
    for reason in result["reasons"]:
        lines.append(f"  - {reason}")
    lines.append("")
    lines.append(f"  {action}")

    self_test = report.get("self_test")
    if self_test is not None:
        status = "PASSED" if self_test["passed"] else "FAILED"
        lines += [
            "",
            f"  estimator self-test: {status}",
            f"    injected  (dy, dx) = ({self_test['injected'][0]:+.2f}, "
            f"{self_test['injected'][1]:+.2f}) HR px",
            f"    recovered (dy, dx) = ({self_test['recovered'][0]:+.2f}, "
            f"{self_test['recovered'][1]:+.2f}) HR px  "
            f"(tolerance +/-{self_test['tolerance_px']:.2f})",
        ]
        if self_test["control_median_shift_px"] is not None:
            lines.append(
                f"    control (unshifted) median shift = "
                f"{self_test['control_median_shift_px']:.2f} HR px"
            )
        for note in self_test["notes"]:
            lines.append(f"    ! {note}")
        if not self_test["passed"]:
            lines.append(
                "    The estimator did not recover a known shift, so the verdict "
                "above is NOT trustworthy."
            )
    lines.append("=" * width)
    return "\n".join(lines)
