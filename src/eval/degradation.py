"""Is the LR a synthetic downsample of the HR, or a genuine second sensor?

This module exists because of a specific observation, and it is worth stating
plainly so nobody has to re-derive it. ``scripts/verify_data_root.py`` printed
per-band reflectance ranges for a cached SEN2NAIPv2 pair in which the LR and HR
minima and maxima agreed to four decimal places on **all four bands**, and the
means differed only in the fourth decimal. A 10 m Sentinel-2 acquisition and a
2.5 m NAIP acquisition cannot agree that closely by chance. Either the reporting
was wrong, or the pair is not two independent observations.

Two hypotheses produce that symptom, and they have opposite consequences:

**H1 -- the LR was synthetically degraded from the HR.** Then the "super-
resolution" task is the inversion of a known analytic operator, the reported
PSNR is inflated, and the result would not survive review.

**H2 -- the HR was radiometrically harmonised to the LR** (histogram matching is
the standard step when NAIP is brought onto Sentinel-2's reflectance scale).
Then the marginal distributions agree *by construction* while the spatial
content of the two images remains genuinely independent, and the SR task is
real.

The tests below separate them, because a single PSNR number cannot:

1. :func:`degrade` applies each candidate kernel to the HR and
   :func:`score_pair` scores it against the stored LR. Under H1 the residual is
   at the level of ``uint16`` rounding and PSNR is enormous. Under H2 there is a
   real, substantial residual.
2. The **variance ratio** ``std(LR) / std(HR)``, per band. This is the decisive
   test and it does not depend on any threshold. *Every averaging kernel reduces
   variance* -- that is what averaging is. So under H1-with-an-averaging-kernel
   the ratio must fall visibly below 1. Under H2 it sits at 1, because matching
   the histogram matches every moment of it.
3. A **mismatched-pair control**: each LR is also scored against the degraded HR
   of a different, randomly chosen sample. Natural imagery over similar terrain
   correlates well, so a PSNR has no meaning until the score for *unrelated*
   tiles is on the same page.

Conventions used throughout, and they are not negotiable inside this module:

- Arrays are surface reflectance, ``float32``, ``(C, H, W)``, nominally
  ``[0, 1]`` but **unclipped** -- bright targets legitimately exceed 1.0 and are
  never clamped here.
- PSNR follows the project convention in :mod:`src.metrics.image_quality` --
  mean of the per-band dB against the fixed ``cfg.metrics.data_range`` (1.0
  reflectance) -- so every number here is directly comparable with the project's
  baseline table. It is computed locally rather than by calling ``psnr()``
  because that function takes no mask, deliberately, and nodata pixels must be
  excluded here.
- Nodata pixels are excluded by an explicit mask that the caller supplies. They
  are never silently zero-filled into a mean.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "DEGRADATION_KERNELS",
    "degrade",
    "best_nearest_offset",
    "block_valid_mask",
    "score_pair",
    "marginal_agreement",
    "summarise",
    "quantisation_ceiling_db",
    "verdict",
]


# -- degradation kernels ---------------------------------------------------
#
# Each maps HR (C, sH, sW) -> (C, H, W) at integer factor ``scale``. Named so a
# config can select them; an unknown name raises in degrade() rather than
# falling back to a default, because a silent fallback would make the audit
# report a kernel it did not actually run.
#
# The two families behave differently under the variance test, which is the
# point of including both:
#   AVERAGING  bicubic_antialias, bilinear_antialias, area, box -- mix
#              neighbouring HR pixels, so they REDUCE variance.
#   SAMPLING   bicubic_naive, nearest -- pass pixels through (naive bicubic
#              samples a continuous interpolant at the output grid without a
#              pre-filter), so they roughly PRESERVE variance.
DEGRADATION_KERNELS: Tuple[str, ...] = (
    "bicubic_antialias",
    "bicubic_naive",
    "bilinear_antialias",
    "area",
    "box",
    "nearest",
)

_AVERAGING_KERNELS = frozenset(
    {"bicubic_antialias", "bilinear_antialias", "area", "box"}
)


def degrade(hr: np.ndarray, kernel: str, scale: int, offset: Tuple[int, int] = (0, 0)):
    """Downsample HR by ``scale`` with one named kernel.

    Args:
        hr: Reference image, ``(C, H*scale, W*scale)`` float32 surface
            reflectance, nominally ``[0, 1]`` and **unclipped**. Nodata must
            already be filled; this function does not mask.
        kernel: One of :data:`DEGRADATION_KERNELS`.
        scale: Integer downsampling factor, ``cfg.sr.scale``.
        offset: ``(dy, dx)`` phase for the ``nearest`` kernel only, each in
            ``[0, scale)``. Ignored by every other kernel, because they are
            phase-free by construction.

    Returns:
        ``(C, H, W)`` float32 surface reflectance in the same units and band
        order as ``hr``. Nothing is clipped: an interpolating kernel may
        overshoot beyond the input's range, and that overshoot is signal about
        which kernel was used, not an error to be hidden.

    Raises:
        ValueError: ``kernel`` is not a known name, ``scale`` is not a positive
            integer, ``hr`` is not 3-D, its spatial dimensions are not divisible
            by ``scale``, or ``offset`` is outside ``[0, scale)``.
    """
    if kernel not in DEGRADATION_KERNELS:
        raise ValueError(
            f"Unknown degradation kernel {kernel!r}. Known kernels: "
            f"{list(DEGRADATION_KERNELS)}. Add it to DEGRADATION_KERNELS and "
            "implement it here rather than letting the audit silently skip it."
        )
    if int(scale) != scale or scale < 1:
        raise ValueError(f"scale must be a positive integer; got {scale!r}.")
    scale = int(scale)

    array = np.asarray(hr)
    if array.ndim != 3:
        raise ValueError(
            f"hr must be (C, H, W); got shape {array.shape}. This module never "
            "guesses an axis order."
        )
    _, height, width = array.shape
    if height % scale or width % scale:
        raise ValueError(
            f"hr spatial dims {(height, width)} are not divisible by scale="
            f"{scale}, so no kernel can produce an exact LR grid."
        )

    dy, dx = (int(offset[0]), int(offset[1]))
    if not (0 <= dy < scale and 0 <= dx < scale):
        raise ValueError(
            f"offset {offset!r} is outside [0, {scale}) in at least one axis."
        )

    if kernel == "nearest":
        # Strided pick, no arithmetic at all. This is the only kernel with a
        # phase, hence the offset argument.
        return np.ascontiguousarray(array[:, dy::scale, dx::scale], dtype=np.float32)

    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None]
    target = (height // scale, width // scale)

    if kernel == "box":
        # Exact block mean. Distinct from "area" only in implementation; at an
        # integer factor the two agree to floating-point noise, and running both
        # is a cheap check that neither has an off-by-one.
        out = F.avg_pool2d(tensor, kernel_size=scale, stride=scale)
    elif kernel == "area":
        out = F.interpolate(tensor, size=target, mode="area")
    elif kernel == "bicubic_antialias":
        out = F.interpolate(
            tensor, size=target, mode="bicubic", align_corners=False, antialias=True
        )
    elif kernel == "bicubic_naive":
        out = F.interpolate(
            tensor, size=target, mode="bicubic", align_corners=False, antialias=False
        )
    else:  # bilinear_antialias
        out = F.interpolate(
            tensor, size=target, mode="bilinear", align_corners=False, antialias=True
        )

    return out[0].numpy().astype(np.float32, copy=False)


def block_valid_mask(
    lr_nodata: np.ndarray, hr_nodata: np.ndarray, scale: int
) -> np.ndarray:
    """Which LR pixels can be compared against a degraded HR at all.

    An LR pixel is comparable only when it is itself valid **and** every one of
    the ``scale**2`` HR pixels that feed it is valid. Anything looser lets a
    nodata fill value (0.0 reflectance) into an averaging kernel, which
    manufactures a dark pixel and a large fake residual.

    Args:
        lr_nodata: ``(C, H, W)`` bool. True where the LR pixel is nodata.
        hr_nodata: ``(C, H*scale, W*scale)`` bool, same convention.
        scale: Integer factor relating the two grids.

    Returns:
        ``(C, H, W)`` bool, True where the pixel is **valid** and may be scored.

    Raises:
        ValueError: The two masks are not consistent with ``scale``.
    """
    lr_mask = np.asarray(lr_nodata, dtype=bool)
    hr_mask = np.asarray(hr_nodata, dtype=bool)
    if lr_mask.ndim != 3 or hr_mask.ndim != 3:
        raise ValueError(
            f"Masks must be (C, H, W); got {lr_mask.shape} and {hr_mask.shape}."
        )
    channels, height, width = lr_mask.shape
    expected = (channels, height * scale, width * scale)
    if hr_mask.shape != expected:
        raise ValueError(
            f"hr_nodata shape {hr_mask.shape} does not match lr_nodata "
            f"{lr_mask.shape} at scale={scale}; expected {expected}."
        )

    # Any nodata anywhere in the scale x scale block poisons the LR pixel.
    blocked = hr_mask.reshape(channels, height, scale, width, scale).any(axis=(2, 4))
    return ~(lr_mask | blocked)


def _masked_psnr(
    candidate: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    data_range: float,
) -> Tuple[float, np.ndarray]:
    """PSNR over valid pixels only, per band and averaged over bands.

    Rather than passing a mask into :func:`src.metrics.image_quality.psnr` --
    which takes none, deliberately, so that its numbers are always over whole
    tiles -- the residual is zeroed on invalid pixels and the mean is taken over
    the valid count per band. That is arithmetically the same masked MSE, and it
    keeps the dB convention (mean of per-band dB) identical to the project's.

    Args:
        candidate: ``(C, H, W)`` float32 reflectance.
        reference: ``(C, H, W)`` float32 reflectance, same units and band order.
        valid: ``(C, H, W)`` bool, True where the pixel may be scored.
        data_range: Peak reflectance, ``cfg.metrics.data_range``.

    Returns:
        ``(mean_db, per_band_db)``. A band with zero valid pixels yields
        ``nan``, never a silently substituted value.

    Raises:
        ValueError: Shapes disagree, or no pixel anywhere is valid.
    """
    if candidate.shape != reference.shape or candidate.shape != valid.shape:
        raise ValueError(
            f"Shape mismatch: candidate {candidate.shape}, reference "
            f"{reference.shape}, valid {valid.shape}."
        )
    if not valid.any():
        raise ValueError(
            "No valid pixels in this pair -- every pixel is nodata in the LR or "
            "in its HR block. This sample should have been rejected by "
            "cfg.dataset.max_nodata_fraction; audit it rather than scoring it."
        )

    residual = np.where(valid, candidate.astype(np.float64) - reference, 0.0)
    counts = valid.reshape(valid.shape[0], -1).sum(axis=1)
    sums = (residual**2).reshape(residual.shape[0], -1).sum(axis=1)

    per_band = np.full(counts.shape, np.nan, dtype=np.float64)
    scorable = counts > 0
    with np.errstate(divide="ignore"):
        mse = sums[scorable] / counts[scorable]
        per_band[scorable] = 10.0 * np.log10((float(data_range) ** 2) / mse)

    return float(np.nanmean(per_band)), per_band


def best_nearest_offset(
    lr: np.ndarray,
    hr: np.ndarray,
    valid: np.ndarray,
    scale: int,
    data_range: float,
) -> Tuple[Tuple[int, int], float, np.ndarray]:
    """Score every nearest-subsampling phase and return the best.

    Nearest subsampling has an unknown phase: HR pixel ``(0, 0)`` may or may not
    be the one that became LR pixel ``(0, 0)``. Scoring a single phase would
    understate the kernel, so all ``scale**2`` offsets are tried.

    The **winning offset is itself evidence**, and is why this returns it rather
    than only the score. A genuine nearest degradation uses one fixed phase, so
    the same offset wins on essentially every sample. If the winning offset
    scatters across samples, the fit is coincidence -- whichever phase happened
    to land on the brighter pixels of that tile.

    Args:
        lr: ``(C, H, W)`` float32 reflectance, the stored LR.
        hr: ``(C, H*scale, W*scale)`` float32 reflectance.
        valid: ``(C, H, W)`` bool from :func:`block_valid_mask`.
        scale: Integer factor.
        data_range: Peak reflectance, ``cfg.metrics.data_range``.

    Returns:
        ``((dy, dx), mean_db, per_band_db)`` for the highest-scoring phase.
    """
    best: Optional[Tuple[Tuple[int, int], float, np.ndarray]] = None
    for dy in range(scale):
        for dx in range(scale):
            candidate = degrade(hr, "nearest", scale, offset=(dy, dx))
            mean_db, per_band = _masked_psnr(candidate, lr, valid, data_range)
            if best is None or mean_db > best[1]:
                best = ((dy, dx), mean_db, per_band)
    assert best is not None  # scale >= 1 guarantees at least one offset
    return best


def score_pair(
    lr: np.ndarray,
    hr: np.ndarray,
    valid: np.ndarray,
    scale: int,
    kernels: Sequence[str],
    data_range: float,
    search_nearest_offsets: bool = True,
) -> Dict[str, Any]:
    """Score every candidate degradation kernel on one pair.

    Args:
        lr: ``(C, H, W)`` float32 surface reflectance, the STORED LR -- the thing
            under suspicion. Nominally ``[0, 1]``, unclipped.
        hr: ``(C, H*scale, W*scale)`` float32 surface reflectance, same units and
            band order.
        valid: ``(C, H, W)`` bool from :func:`block_valid_mask`.
        scale: Integer factor, ``cfg.sr.scale``.
        kernels: Names from :data:`DEGRADATION_KERNELS`.
        data_range: Peak reflectance, ``cfg.metrics.data_range``.
        search_nearest_offsets: Try all ``scale**2`` phases for ``nearest``.
            When False only phase ``(0, 0)`` is scored, which understates it.

    Returns:
        ``{"psnr_db": {kernel: float}, "per_band_db": {kernel: list[float]},
        "nearest_offset": (dy, dx) | None, "valid_fraction": float}``. PSNR is in
        dB in the reflectance domain; ``+inf`` when a kernel reproduces the LR
        exactly, which is the correct reading and is left as ``inf`` rather than
        capped.

    Raises:
        ValueError: A kernel name is unknown, or shapes are inconsistent.
    """
    scores: Dict[str, float] = {}
    per_band: Dict[str, List[float]] = {}
    nearest_offset: Optional[Tuple[int, int]] = None

    for kernel in kernels:
        if kernel == "nearest" and search_nearest_offsets:
            offset, mean_db, bands = best_nearest_offset(
                lr, hr, valid, scale, data_range
            )
            nearest_offset = offset
        else:
            candidate = degrade(hr, kernel, scale)
            mean_db, bands = _masked_psnr(candidate, lr, valid, data_range)
            if kernel == "nearest":
                nearest_offset = (0, 0)
        scores[kernel] = mean_db
        per_band[kernel] = [float(v) for v in bands]

    return {
        "psnr_db": scores,
        "per_band_db": per_band,
        "nearest_offset": nearest_offset,
        "valid_fraction": float(valid.mean()),
    }


def marginal_agreement(
    lr: np.ndarray,
    hr: np.ndarray,
    lr_valid: np.ndarray,
    hr_valid: np.ndarray,
    quantiles: Sequence[float],
) -> Dict[str, Any]:
    """Compare the LR and HR marginal distributions, per band.

    This is the test that separates the two hypotheses in the module docstring,
    and it rests on a fact that needs no threshold to state: **averaging reduces
    variance.** If the LR were produced from the HR by any averaging kernel, the
    LR must be measurably smoother -- ``std(LR) / std(HR)`` well below 1. At x4
    over natural imagery the ratio typically lands near 0.9 or lower. A ratio
    indistinguishable from 1.0, combined with matching extrema, points instead
    at the HR having been histogram-matched onto the LR's radiometry, which
    matches every moment of the distribution while leaving the spatial content
    independent.

    Args:
        lr: ``(C, H, W)`` float32 surface reflectance.
        hr: ``(C, H*scale, W*scale)`` float32 surface reflectance, same units and
            band order.
        lr_valid: ``(C, H, W)`` bool, True where the LR pixel is valid.
        hr_valid: ``(C, H*scale, W*scale)`` bool, same for HR.
        quantiles: Percentages in ``[0, 100]`` to compare.

    Returns:
        ``{"std_ratio": [float per band], "mean_abs_diff": [float per band],
        "quantiles": {"levels": [...], "lr": [[...] per band],
        "hr": [[...] per band], "max_abs_diff": [float per band]}}``. All values
        are in reflectance units. A band with no valid pixels reports ``nan``.
    """
    levels = [float(q) for q in quantiles]
    channels = lr.shape[0]

    std_ratio: List[float] = []
    mean_abs_diff: List[float] = []
    lr_q: List[List[float]] = []
    hr_q: List[List[float]] = []
    max_q_diff: List[float] = []

    for band in range(channels):
        lr_pixels = lr[band][lr_valid[band]].astype(np.float64)
        hr_pixels = hr[band][hr_valid[band]].astype(np.float64)
        if lr_pixels.size == 0 or hr_pixels.size == 0:
            std_ratio.append(float("nan"))
            mean_abs_diff.append(float("nan"))
            lr_q.append([float("nan")] * len(levels))
            hr_q.append([float("nan")] * len(levels))
            max_q_diff.append(float("nan"))
            continue

        hr_std = float(hr_pixels.std())
        # A constant HR band has no variance to reduce, so the ratio is
        # undefined rather than 0 or inf. Reported as nan and excluded from the
        # median upstream.
        std_ratio.append(
            float(lr_pixels.std() / hr_std) if hr_std > 0 else float("nan")
        )
        mean_abs_diff.append(abs(float(lr_pixels.mean()) - float(hr_pixels.mean())))

        lq = np.percentile(lr_pixels, levels)
        hq = np.percentile(hr_pixels, levels)
        lr_q.append([float(v) for v in lq])
        hr_q.append([float(v) for v in hq])
        max_q_diff.append(float(np.abs(lq - hq).max()))

    return {
        "std_ratio": std_ratio,
        "mean_abs_diff": mean_abs_diff,
        "quantiles": {
            "levels": levels,
            "lr": lr_q,
            "hr": hr_q,
            "max_abs_diff": max_q_diff,
        },
    }


def _percentiles(values: Sequence[float], levels: Sequence[float]) -> Dict[str, float]:
    """Percentiles of a sample, keyed ``"p0"``, ``"p50"``, ...

    Non-finite entries are excluded and counted by the caller; they are never
    replaced by a finite stand-in, because ``inf`` here means "reproduced the LR
    exactly" and that is the single most important thing the audit can find.
    """
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {f"p{level:g}": float("nan") for level in levels}
    computed = np.percentile(finite, [float(level) for level in levels])
    return {f"p{level:g}": float(v) for level, v in zip(levels, computed)}


def summarise(
    per_sample: Sequence[Mapping[str, Any]],
    kernels: Sequence[str],
    percentiles: Sequence[float],
) -> Dict[str, Any]:
    """Aggregate per-sample scores into per-kernel distributions.

    The distribution is the deliverable, not the mean. A mean PSNR is exactly
    what let the original anomaly go unnoticed for as long as it did.

    Args:
        per_sample: Records from :func:`score_pair`, one per pair.
        kernels: Kernel names to summarise, in report order.
        percentiles: Percentage levels to report per kernel.

    Returns:
        ``{kernel: {"mean", "std", "num_infinite", "num_scored", "p0", ...}}``
        in dB, plus ``"best_kernel"`` naming the highest median. ``num_infinite``
        counts samples the kernel reproduced exactly.

    Raises:
        ValueError: ``per_sample`` is empty.
    """
    if not per_sample:
        raise ValueError(
            "No samples were scored, so there is nothing to summarise. This is "
            "raised rather than returning an empty summary, which would read as "
            "a clean result."
        )

    summary: Dict[str, Any] = {}
    for kernel in kernels:
        values = [float(record["psnr_db"][kernel]) for record in per_sample]
        finite = [v for v in values if np.isfinite(v)]
        stats: Dict[str, Any] = {
            "num_scored": len(values),
            "num_infinite": len(values) - len(finite),
            "mean": float(np.mean(finite)) if finite else float("inf"),
            "std": float(np.std(finite)) if finite else float("nan"),
        }
        stats.update(_percentiles(values, percentiles))
        summary[kernel] = stats

    # Ranked on the MEDIAN, not the mean: one exactly-reproduced tile makes a
    # mean infinite and would crown a kernel on a single sample.
    def median_of(kernel: str) -> float:
        value = summary[kernel].get("p50", float("nan"))
        return float("-inf") if not np.isfinite(value) else float(value)

    ranked = sorted(kernels, key=median_of, reverse=True)
    best = ranked[0]
    # An all-infinite kernel has median nan by the rule above and would sort
    # last, which is backwards -- it is the strongest possible fit.
    exact = [k for k in kernels if summary[k]["num_infinite"] == summary[k]["num_scored"]]
    if exact:
        best = exact[0]

    summary["best_kernel"] = best
    return summary


def quantisation_ceiling_db(reflectance_scale: float, data_range: float) -> float:
    """The highest PSNR a synthetic degradation could plausibly reach, in dB.

    A degradation computed in floating point and then stored as ``uint16``
    differs from its own kernel's exact output only by that rounding. Uniform
    rounding to the nearest digital number has RMS error
    ``(1 / reflectance_scale) / sqrt(12)`` in reflectance units, so the PSNR
    against ``data_range`` cannot exceed the value returned here by more than a
    little.

    This is the audit's one unconditional ceiling, and it is COMPUTED rather
    than configured on purpose: it follows from the storage format, not from any
    judgement of ours. At ``reflectance_scale=10000`` and ``data_range=1.0`` it
    is 90.8 dB. A candidate kernel scoring tens of dB below it has a residual
    far larger than rounding -- that is real content the kernel does not
    explain, and no amount of threshold-setting can argue it away.

    Args:
        reflectance_scale: ``cfg.dataset.reflectance_scale``, digital numbers
            per unit reflectance.
        data_range: ``cfg.metrics.data_range``, peak reflectance.

    Returns:
        PSNR in dB.

    Raises:
        ValueError: Either argument is not positive and finite.
    """
    scale = float(reflectance_scale)
    peak = float(data_range)
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(peak) or peak <= 0:
        raise ValueError(
            f"reflectance_scale={reflectance_scale!r} and data_range="
            f"{data_range!r} must both be positive and finite."
        )
    rms = (1.0 / scale) / np.sqrt(12.0)
    return float(20.0 * np.log10(peak / rms))


def verdict(
    summary: Mapping[str, Any],
    marginals: Mapping[str, Any],
    control_median_db: Optional[float],
    synthetic_psnr_db: float,
    crosssensor_psnr_db: float,
    variance_ratio_tol: float,
    ceiling_db: float,
    deterministic_spread_db: float,
    quantisation_ceiling_fraction: float,
) -> Dict[str, Any]:
    """Turn the measurements into SYNTHETIC / CROSS_SENSOR / UNRESOLVED.

    THREE INDEPENDENT TESTS, and the reason there are three is worth recording.
    An earlier version of this function checked the PSNR threshold first and
    returned on it, so a run that landed 0.38 dB over the line was reported as
    SYNTHETIC while the variance test in the very same report said no averaging
    kernel could have produced the LR. A verdict decidable by 0.38 dB of a
    threshold we chose ourselves is not a verdict. So the PSNR threshold is now
    the *weakest* of the tests, and two of the others need no threshold at all:

    1. **Variance.** Averaging reduces variance; that is what averaging is. If
       ``std(LR)/std(HR)`` sits at 1.0 then no averaging kernel produced the LR,
       whatever its PSNR. Unconditional.
    2. **Spread.** A deterministic operator applied to N tiles scores nearly the
       same on all of them. A wide spread of per-tile PSNR means the residual
       varies with scene content, which a fixed kernel cannot do.
    3. **The quantisation ceiling.** A synthetic degradation stored as ``uint16``
       sits near :func:`quantisation_ceiling_db`. Far below it, the residual is
       real content. Computed from the storage format, not chosen by us.

    When the PSNR threshold and the physical tests disagree, the disagreement is
    stated in ``reasons`` rather than resolved silently, and the physical tests
    decide -- they rest on properties of the operators involved, the threshold
    rests on our judgement. If the physical tests are themselves split, the
    result is UNRESOLVED and says so.

    Args:
        summary: Output of :func:`summarise`.
        marginals: Aggregated marginal statistics with a ``"median_std_ratio"``
            key.
        control_median_db: Median PSNR of the mismatched-pair control, or None
            when the control was disabled.
        synthetic_psnr_db: Upper threshold, ``cfg.degradation_audit``.
        crosssensor_psnr_db: Lower threshold.
        variance_ratio_tol: How close to 1.0 the variance ratio must sit.
        ceiling_db: From :func:`quantisation_ceiling_db`.
        deterministic_spread_db: Maximum ``p95 - p5`` spread, in dB, still
            consistent with one deterministic kernel.
        quantisation_ceiling_fraction: Fraction of ``ceiling_db`` the best fit
            must reach before the LR counts as that kernel's own output.

    Returns:
        ``{"verdict": str, "best_kernel": str, "best_median_db": float,
        "margin_over_control_db": float|None, "variance_preserved": bool,
        "spread_db": float, "deterministic": bool,
        "reaches_quantisation_ceiling": bool, "evidence": {...},
        "reasons": [str, ...]}``.
    """
    best = str(summary["best_kernel"])
    stats = summary[best]
    best_median = float(stats.get("p50", float("nan")))
    if stats["num_infinite"] == stats["num_scored"]:
        best_median = float("inf")

    # -- test 1: variance, unconditional ----------------------------------
    ratio = float(marginals.get("median_std_ratio", float("nan")))
    variance_preserved = bool(
        np.isfinite(ratio) and abs(ratio - 1.0) <= float(variance_ratio_tol)
    )
    best_is_averaging = best in _AVERAGING_KERNELS

    # -- test 2: spread across tiles --------------------------------------
    low = float(stats.get("p5", float("nan")))
    high = float(stats.get("p95", float("nan")))
    if not np.isfinite(best_median):
        spread = 0.0
    elif np.isfinite(low) and np.isfinite(high):
        spread = high - low
    else:
        spread = float("nan")
    deterministic = bool(
        np.isfinite(spread) and spread <= float(deterministic_spread_db)
    )

    # -- test 3: the quantisation ceiling ---------------------------------
    required = float(quantisation_ceiling_fraction) * float(ceiling_db)
    reaches_ceiling = bool(not np.isfinite(best_median) or best_median >= required)

    # -- test 4, the weakest: the dB threshold ----------------------------
    over_synthetic = bool(best_median > float(synthetic_psnr_db))
    under_crosssensor = bool(best_median < float(crosssensor_psnr_db))

    margin = (
        None if control_median_db is None else best_median - float(control_median_db)
    )

    reasons: List[str] = []
    if np.isfinite(best_median):
        reasons.append(
            f"Best kernel {best!r} reaches a median {best_median:.2f} dB against "
            f"the stored LR, with a p5-p95 spread of {spread:.2f} dB."
        )
    else:
        reasons.append(
            f"Best kernel {best!r} reproduces the stored LR EXACTLY on every "
            "sample scored."
        )
    if margin is not None:
        reasons.append(
            f"Mismatched-pair control sits at {float(control_median_db):.2f} dB, "
            f"so the true-pair advantage is {margin:.2f} dB."
        )

    reasons.append(
        f"Quantisation ceiling for uint16 at this reflectance scale is "
        f"{float(ceiling_db):.1f} dB, and a synthetic degradation must sit near "
        f"it (>= {required:.1f} dB). The best fit "
        + (
            "does."
            if reaches_ceiling
            else f"is {float(ceiling_db) - best_median:.1f} dB below it, so its "
            "residual is real content, not rounding."
        )
    )
    reasons.append(
        f"Per-tile spread is {spread:.2f} dB against a deterministic limit of "
        f"{float(deterministic_spread_db):.2f} dB. One fixed kernel "
        + (
            "could produce this."
            if deterministic
            else "cannot produce a residual that varies this much with scene "
            "content."
        )
    )
    if np.isfinite(ratio):
        if variance_preserved and best_is_averaging:
            tail = (
                f"it is within {float(variance_ratio_tol):g} of 1.0, so no "
                "averaging degradation produced this LR -- and the best-fitting "
                f"kernel {best!r} is itself an averaging kernel."
            )
        elif variance_preserved:
            tail = "it is indistinguishable from 1.0."
        else:
            tail = "it is below 1.0, consistent with an averaging degradation."
        reasons.append(
            f"Median std(LR)/std(HR) is {ratio:.4f}. Averaging kernels must push "
            f"this below 1.0; {tail}"
        )

    # A synthetic LR must satisfy every physical test: it must reach the
    # quantisation ceiling, be deterministic across tiles, and -- when the
    # kernel that fits best is an averaging one -- have lost variance.
    physical_synthetic = bool(
        reaches_ceiling
        and deterministic
        and not (best_is_averaging and variance_preserved)
    )

    if over_synthetic and not physical_synthetic:
        reasons.append(
            f"THE TESTS DISAGREE, and that is reported rather than resolved "
            f"quietly: the {float(synthetic_psnr_db):g} dB threshold is exceeded "
            f"by {best_median - float(synthetic_psnr_db):.2f} dB, but the "
            "physical tests above rule the degradation out. The threshold is a "
            "number we chose; the physical tests are properties of the operators "
            "involved, so they decide."
        )

    if physical_synthetic and over_synthetic:
        name = "SYNTHETIC"
        reasons.append(
            "Every test agrees: the LR is a resampling of the HR, not an "
            "independent acquisition."
        )
    elif physical_synthetic:
        name = "UNRESOLVED"
        reasons.append(
            "The physical tests point at a degradation but the PSNR sits below "
            f"the {float(synthetic_psnr_db):g} dB threshold. Do not report a "
            "headline PSNR until this is resolved."
        )
    elif under_crosssensor or variance_preserved or not deterministic:
        name = "CROSS_SENSOR"
        reasons.append(
            "The LR carries content no candidate kernel reproduces from the HR, "
            "so it is an independent acquisition. Matching LR/HR extrema are "
            "then explained by radiometric harmonisation of the HR onto the LR "
            "scale, which matches the marginal distributions by construction "
            "while leaving the spatial content independent."
        )
    else:
        name = "UNRESOLVED"
        reasons.append(
            "The tests do not agree and none of them is decisive. Do not report "
            "a headline PSNR until this is resolved."
        )

    return {
        "verdict": name,
        "best_kernel": best,
        "best_median_db": best_median,
        "margin_over_control_db": margin,
        "variance_preserved": variance_preserved,
        "variance_ratio": ratio,
        "spread_db": spread,
        "deterministic": deterministic,
        "reaches_quantisation_ceiling": reaches_ceiling,
        "quantisation_ceiling_db": float(ceiling_db),
        "kernel_family": "averaging" if best_is_averaging else "sampling",
        "evidence": {
            "psnr_threshold_says_synthetic": over_synthetic,
            "physical_tests_say_synthetic": physical_synthetic,
            "tests_disagree": bool(over_synthetic and not physical_synthetic),
        },
        "reasons": reasons,
    }
