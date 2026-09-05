"""Patch extraction and rejection rules.

Two jobs, both of which fail silently if they are wrong:

1. **Cutting LR/HR patch pairs.** The HR crop origin must be exactly ``scale``
   times the LR crop origin. A one-pixel error here shifts every training target
   by 0.25 LR pixels, and nothing in the training loop will ever say so: the loss
   still decreases, PSNR still looks plausible, and the model simply learns to
   blur. So the relation is not merely computed -- it is re-derived and asserted
   on every call, in both extraction modes, by :func:`assert_patch_alignment`.

2. **Rejecting patches that teach the model nothing.** Nodata, cloud-like
   brightness, and near-constant content. Every rejection is counted by rule and
   the counts are logged, because a filter that quietly eats most of the training
   set is worse than no filter: it produces a small, biased dataset that still
   trains to completion.

Axis order is ``(C, H, W)`` everywhere in this module -- channel first, then
row, then column. Both NumPy arrays and torch tensors are accepted; slicing
preserves the input type, so an extracted patch is the same type as its source.

Nothing here clips reflectance. Bright targets legitimately exceed 1.0 and the
spectral-consistency objective needs them intact; the cloud rule *counts* bright
pixels, it never rewrites them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "PatchCoords",
    "PatchFilter",
    "PatchExtractionError",
    "patch_grid",
    "crop_pair",
    "centre_crop_pair",
    "extract_patches",
    "assert_patch_alignment",
    "patch_params_from_cfg",
]

GRID = "grid"
RANDOM = "random"
MODES = (GRID, RANDOM)


class PatchExtractionError(ValueError):
    """A patch could not be cut, or the LR/HR geometry does not agree.

    Raised, never logged-and-skipped: every condition that produces it means the
    caller is wrong about the shape of its data, and continuing would train on
    pairs whose correspondence is unknown.
    """


@dataclass(frozen=True)
class PatchCoords:
    """Where one patch was cut from, in both resolutions.

    Attributes:
        lr_row: Top edge of the LR crop, in LR pixels, from the top of the tile.
        lr_col: Left edge of the LR crop, in LR pixels.
        lr_size: LR crop edge length in LR pixels (crops are square).
        hr_row: Top edge of the HR crop, in HR pixels. Always ``lr_row * scale``.
        hr_col: Left edge of the HR crop, in HR pixels. Always ``lr_col * scale``.
        hr_size: HR crop edge length in HR pixels. Always ``lr_size * scale``.
        scale: The super-resolution factor relating the two.
    """

    lr_row: int
    lr_col: int
    lr_size: int
    hr_row: int
    hr_col: int
    hr_size: int
    scale: int

    @classmethod
    def from_lr(cls, lr_row: int, lr_col: int, lr_size: int, scale: int) -> "PatchCoords":
        """Build coordinates from the LR crop alone, deriving the HR crop.

        This is the only constructor :func:`extract_patches` uses. The HR
        coordinates are *derived*, never passed in, so there is no path by which
        a caller can supply an HR origin that disagrees with the LR one.
        """
        return cls(
            lr_row=int(lr_row),
            lr_col=int(lr_col),
            lr_size=int(lr_size),
            hr_row=int(lr_row) * int(scale),
            hr_col=int(lr_col) * int(scale),
            hr_size=int(lr_size) * int(scale),
            scale=int(scale),
        )

    def as_dict(self) -> Dict[str, int]:
        """Plain-dict form, for manifests and JSON reports."""
        return {
            "lr_row": self.lr_row,
            "lr_col": self.lr_col,
            "lr_size": self.lr_size,
            "hr_row": self.hr_row,
            "hr_col": self.hr_col,
            "hr_size": self.hr_size,
            "scale": self.scale,
        }


def assert_patch_alignment(coords: PatchCoords, scale: int) -> None:
    """Assert the HR crop is exactly ``scale`` times the LR crop.

    Called on every patch produced by :func:`extract_patches`, in both modes.
    The check costs six integer comparisons and the failure it guards against is
    invisible at training time, so it is never skipped for speed.

    Args:
        coords: The coordinates to check.
        scale: The expected super-resolution factor.

    Raises:
        PatchExtractionError: Origin, size, or scale disagrees.
    """
    scale = int(scale)
    if coords.scale != scale:
        raise PatchExtractionError(
            f"Patch coords carry scale={coords.scale} but scale={scale} was "
            "requested."
        )
    if coords.hr_row != coords.lr_row * scale or coords.hr_col != coords.lr_col * scale:
        raise PatchExtractionError(
            f"HR patch origin ({coords.hr_row}, {coords.hr_col}) is not "
            f"{scale} x the LR origin ({coords.lr_row}, {coords.lr_col}) = "
            f"({coords.lr_row * scale}, {coords.lr_col * scale}). An off-by-one "
            "here shifts every training target and shows up only as a model "
            "that blurs."
        )
    if coords.hr_size != coords.lr_size * scale:
        raise PatchExtractionError(
            f"HR patch size {coords.hr_size} is not {scale} x the LR patch size "
            f"{coords.lr_size} = {coords.lr_size * scale}."
        )


def patch_grid(
    height: int,
    width: int,
    lr_size: int,
    stride: int,
    drop_incomplete_edge: bool = False,
) -> List[Tuple[int, int]]:
    """Deterministic grid of LR crop origins covering a tile.

    Args:
        height: Tile height in LR pixels.
        width: Tile width in LR pixels.
        lr_size: Crop edge in LR pixels.
        stride: Step between crop origins in LR pixels. Below ``lr_size`` the
            patches overlap.
        drop_incomplete_edge: When the stride does not divide the tile evenly,
            False (the default) appends one extra origin flush against the
            right/bottom edge so the margin is still seen; True discards it.

    Returns:
        ``(row, col)`` origins in row-major order. Deterministic: the same
        arguments always give the same list, which is what makes a validation
        number comparable between runs and between ablations.

    Raises:
        PatchExtractionError: ``lr_size`` or ``stride`` is non-positive, or the
            tile is smaller than one patch.
    """
    if lr_size <= 0:
        raise PatchExtractionError(f"lr_size must be positive, got {lr_size}.")
    if stride <= 0:
        raise PatchExtractionError(f"stride must be positive, got {stride}.")
    if height < lr_size or width < lr_size:
        raise PatchExtractionError(
            f"Tile is {height}x{width} LR pixels, smaller than the requested "
            f"{lr_size}x{lr_size} patch. Lower cfg.patches.lr_size or emit "
            "larger tiles (cfg.sr.lr_patch_size)."
        )

    def axis(extent: int) -> List[int]:
        origins = list(range(0, extent - lr_size + 1, stride))
        last = extent - lr_size
        if origins[-1] != last and not drop_incomplete_edge:
            origins.append(last)
        return origins

    return [(row, col) for row in axis(height) for col in axis(width)]


def patch_params_from_cfg(cfg: Any) -> Dict[str, Any]:
    """Read the patch geometry out of a config.

    There are deliberately no default patch sizes in this module's function
    signatures. A default would be a hyperparameter living in code, and the one
    failure mode this module exists to prevent is code and config disagreeing
    about geometry.

    Args:
        cfg: The loaded config, with ``sr`` and ``patches`` sections.

    Returns:
        ``{"lr_size", "hr_size", "scale", "stride", "mode",
        "random_per_sample", "drop_incomplete_edge"}``.

    Raises:
        KeyError: A required section or key is missing.
        PatchExtractionError: ``cfg.patches.lr_size`` exceeds the LR tile the
            dataset emits (``cfg.sr.lr_patch_size``), or ``mode`` is unknown.
    """
    scale = int(cfg["sr"]["scale"])
    tile_lr_size = int(cfg["sr"]["lr_patch_size"])
    patches = cfg["patches"]
    lr_size = int(patches["lr_size"])
    mode = str(patches["mode"])

    if mode not in MODES:
        raise PatchExtractionError(f"cfg.patches.mode={mode!r} is not one of {MODES}.")
    if lr_size > tile_lr_size:
        raise PatchExtractionError(
            f"cfg.patches.lr_size={lr_size} exceeds cfg.sr.lr_patch_size="
            f"{tile_lr_size}, the LR tile the dataset emits. A patch cannot be "
            "larger than the tile it is cut from."
        )

    return {
        "lr_size": lr_size,
        "hr_size": lr_size * scale,
        "scale": scale,
        "stride": int(patches["stride"]),
        "mode": mode,
        "random_per_sample": int(patches["random_per_sample"]),
        "drop_incomplete_edge": bool(patches["drop_incomplete_edge"]),
    }


def crop_pair(
    lr: Any,
    hr: Any,
    lr_row: int,
    lr_col: int,
    lr_size: int,
    scale: int,
) -> Dict[str, Any]:
    """Cut one aligned patch pair at a given LR origin.

    The single place in the project where an LR/HR pair is sliced. Both
    :func:`extract_patches` and the loader go through it, so the alignment
    assertion cannot be bypassed by a caller that "just needs one patch".

    Args:
        lr: ``(C, H, W)`` float32 surface reflectance, unclipped.
        hr: ``(C, H*scale, W*scale)``, same units and band order.
        lr_row: Top edge of the LR crop, in LR pixels.
        lr_col: Left edge of the LR crop, in LR pixels.
        lr_size: LR crop edge in LR pixels.
        scale: Super-resolution factor.

    Returns:
        ``{"lr", "hr", "coords"}``; ``lr`` and ``hr`` are views into the inputs
        (same dtype, same reflectance units) and ``coords`` is a
        :class:`PatchCoords`.

    Raises:
        PatchExtractionError: The crop falls outside the tile, or the derived HR
            crop is not exactly ``scale`` times the LR crop.
    """
    _, lr_h, lr_w = _check_pair(lr, hr, scale)
    coords = PatchCoords.from_lr(lr_row, lr_col, lr_size, scale)
    assert_patch_alignment(coords, scale)

    if (
        coords.lr_row < 0
        or coords.lr_col < 0
        or coords.lr_row + coords.lr_size > lr_h
        or coords.lr_col + coords.lr_size > lr_w
    ):
        raise PatchExtractionError(
            f"Patch at LR ({coords.lr_row}, {coords.lr_col}) size "
            f"{coords.lr_size} falls outside a {lr_h}x{lr_w} LR tile."
        )

    lr_patch = lr[
        :,
        coords.lr_row : coords.lr_row + coords.lr_size,
        coords.lr_col : coords.lr_col + coords.lr_size,
    ]
    hr_patch = hr[
        :,
        coords.hr_row : coords.hr_row + coords.hr_size,
        coords.hr_col : coords.hr_col + coords.hr_size,
    ]
    return {"lr": lr_patch, "hr": hr_patch, "coords": coords}


def centre_crop_pair(
    lr: Any,
    hr: Any,
    lr_size: int,
    scale: int,
) -> Dict[str, Any]:
    """The deterministic validation transform: centre-crop an LR/HR tile pair.

    THIS IS NOT A DATA-LOADING STEP. It is the transform applied on the
    deterministic (validation) path so that a validation set is byte-for-byte
    the same pixels between runs. It must never be applied inside a dataset's
    read path, and never before caching: a centre crop discards the tile
    periphery, and the periphery is where nodata, tile-edge artefacts and the
    bright targets that reveal a wrong reflectance divisor actually live.
    Training draws random crops from the full tile instead -- see
    :func:`extract_patches` in ``"random"`` mode.

    Args:
        lr: ``(C, H, W)`` float32 surface reflectance, nominally ``[0, 1]`` and
            unclipped -- bright targets legitimately exceed 1.0. ``H``/``W`` are
            the FULL stored tile (130 for SEN2NAIPv2), not a patch.
        hr: ``(C, H*scale, W*scale)``, same dtype, units and band order.
        lr_size: LR edge of the crop, in LR pixels. Must not exceed ``H``/``W``.
        scale: Super-resolution factor.

    Returns:
        ``{"lr", "hr", "coords"}``. ``lr`` is ``(C, lr_size, lr_size)`` and
        ``hr`` is ``(C, lr_size*scale, lr_size*scale)``, both views into the
        inputs carrying the same reflectance units, and ``coords`` is the
        :class:`PatchCoords` of the crop so the discarded margin is recorded
        rather than merely lost.

    Raises:
        PatchExtractionError: ``lr_size`` exceeds the tile, or the pair does not
            satisfy the scale ratio.
    """
    _, lr_h, lr_w = _check_pair(lr, hr, scale)
    size = int(lr_size)

    if size > lr_h or size > lr_w:
        raise PatchExtractionError(
            f"centre_crop_pair: lr_size={size} exceeds the {lr_h}x{lr_w} LR "
            "tile it would be cut from. Lower cfg.sr.lr_patch_size, or check "
            "that the dataset is emitting full tiles."
        )

    # Floor division: with an odd margin the extra row/column is dropped from
    # the bottom/right, deterministically, so the same pixels are validated on
    # every run and on every machine.
    top = (lr_h - size) // 2
    left = (lr_w - size) // 2
    return crop_pair(lr, hr, top, left, size, scale)


def _check_pair(lr: Any, hr: Any, scale: int) -> Tuple[int, int, int]:
    """Validate a tile pair's rank, channel agreement, and scale ratio."""
    if getattr(lr, "ndim", None) != 3 or getattr(hr, "ndim", None) != 3:
        raise PatchExtractionError(
            "lr and hr must be rank 3 (C, H, W); got lr.ndim="
            f"{getattr(lr, 'ndim', None)} hr.ndim={getattr(hr, 'ndim', None)}. "
            "The channel axis is FIRST."
        )
    lr_c, lr_h, lr_w = (int(d) for d in lr.shape)
    hr_c, hr_h, hr_w = (int(d) for d in hr.shape)
    if lr_c != hr_c:
        raise PatchExtractionError(
            f"lr has {lr_c} channels but hr has {hr_c}. Both must carry the "
            "same bands in the same order."
        )
    if hr_h != lr_h * scale or hr_w != lr_w * scale:
        raise PatchExtractionError(
            f"hr is {hr_h}x{hr_w} but lr is {lr_h}x{lr_w} and scale={scale}, "
            f"so hr should be {lr_h * scale}x{lr_w * scale}. The pair is not a "
            "valid LR/HR pair at this scale."
        )
    return lr_c, lr_h, lr_w


def extract_patches(
    lr: Any,
    hr: Any,
    lr_size: int,
    scale: int,
    stride: Optional[int] = None,
    mode: str = GRID,
    num_patches: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    patch_filter: Optional["PatchFilter"] = None,
    nodata_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Cut aligned LR/HR patch pairs out of one tile pair.

    ``mode="grid"`` walks a deterministic stride-based grid and is what
    validation uses: the same tile always yields the same patches in the same
    order. ``mode="random"`` draws uniformly random origins and is what training
    uses.

    In both modes the HR crop is taken at exactly ``scale`` times the LR origin
    and ``scale`` times the size, and that relation is asserted per patch by
    :func:`assert_patch_alignment` before the patch is returned.

    Args:
        lr: Low-resolution tile, ``(C, H, W)``, float32 surface reflectance,
            nominally ``[0, 1]`` and unclipped above it. NumPy array or torch
            tensor.
        hr: High-resolution tile, ``(C, H*scale, W*scale)``, same dtype, same
            band order, same reflectance units as ``lr``.
        lr_size: LR patch edge in LR pixels, from ``cfg.patches.lr_size``. This
            argument deliberately has no default; see
            :func:`patch_params_from_cfg`.
        scale: Super-resolution factor, from ``cfg.sr.scale``.
        stride: Grid step in LR pixels. Required in ``"grid"`` mode, ignored in
            ``"random"``.
        mode: ``"grid"`` or ``"random"``.
        num_patches: Number of random crops in ``"random"`` mode. Ignored in
            ``"grid"`` mode, where the grid decides the count.
        rng: Generator for ``"random"`` mode. Required in that mode -- an
            implicit global RNG would make training unreproducible.
        patch_filter: Applied to each candidate; rejected patches are dropped
            from the result and counted on the filter. None keeps everything.
        nodata_mask: Optional bool array, ``(C, H, W)`` or ``(H, W)`` at **LR**
            resolution, True where the LR pixel has no valid observation. When
            given it is cropped alongside the patch and passed to the filter in
            place of the filter's fill-value heuristic.

    Returns:
        A list of dicts, each with ``"lr"`` (``(C, lr_size, lr_size)``), ``"hr"``
        (``(C, lr_size*scale, lr_size*scale)``) -- both views into the input
        arrays, same dtype and reflectance units -- and ``"coords"``, a
        :class:`PatchCoords`. The list is empty when every candidate was
        rejected; the caller decides whether that is acceptable, and the filter's
        counters say why it happened.

    Raises:
        PatchExtractionError: The tiles disagree in rank, channels, or scale; the
            tile is smaller than one patch; ``mode`` is unknown; or an argument
            the chosen mode requires is missing.
    """
    scale = int(scale)
    lr_size = int(lr_size)
    _, lr_h, lr_w = _check_pair(lr, hr, scale)

    if mode not in MODES:
        raise PatchExtractionError(f"mode must be one of {MODES}, got {mode!r}.")
    if lr_h < lr_size or lr_w < lr_size:
        raise PatchExtractionError(
            f"LR tile is {lr_h}x{lr_w}, smaller than the requested "
            f"{lr_size}x{lr_size} patch."
        )

    if mode == GRID:
        if stride is None:
            raise PatchExtractionError(
                "stride is required in grid mode; pass cfg.patches.stride."
            )
        origins = patch_grid(lr_h, lr_w, lr_size, int(stride))
    else:
        if rng is None:
            raise PatchExtractionError(
                "rng is required in random mode. Pass a seeded "
                "numpy.random.Generator so training is reproducible; there is "
                "deliberately no implicit global RNG here."
            )
        if num_patches is None or int(num_patches) <= 0:
            raise PatchExtractionError(
                "num_patches must be a positive int in random mode, got "
                f"{num_patches!r}. Pass cfg.patches.random_per_sample."
            )
        origins = [
            (
                int(rng.integers(0, lr_h - lr_size + 1)),
                int(rng.integers(0, lr_w - lr_size + 1)),
            )
            for _ in range(int(num_patches))
        ]

    patches: List[Dict[str, Any]] = []
    for row, col in origins:
        patch = crop_pair(lr, hr, row, col, lr_size, scale)
        coords = patch["coords"]

        if patch_filter is not None:
            mask_patch = None
            if nodata_mask is not None:
                if nodata_mask.ndim == 2:
                    mask_patch = nodata_mask[
                        coords.lr_row : coords.lr_row + coords.lr_size,
                        coords.lr_col : coords.lr_col + coords.lr_size,
                    ]
                else:
                    mask_patch = nodata_mask[
                        :,
                        coords.lr_row : coords.lr_row + coords.lr_size,
                        coords.lr_col : coords.lr_col + coords.lr_size,
                    ]
            if patch_filter(patch["lr"], patch["hr"], nodata_mask=mask_patch) is not None:
                continue

        patches.append(patch)

    return patches


def _to_numpy(array: Any) -> np.ndarray:
    """View a NumPy array or a torch tensor as a NumPy array, without copying."""
    if isinstance(array, np.ndarray):
        return array
    detach = getattr(array, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return np.asarray(array)


class PatchFilter:
    """Reject patches that cannot teach the model anything, and count why.

    Three rules, each independently configurable and each counted separately:

    - **nodata**: too large a fraction of the patch has no valid observation.
      When the caller supplies a real mask it is used; otherwise the fallback is
      "every band equals ``cfg.dataset.nodata_fill``", which is exactly what
      ``SRPairDataset.to_reflectance`` writes at masked pixels.
    - **cloud**: too large a fraction of the HR patch is bright in *every* band.
      This is a **proxy, not a cloud mask** -- SEN2NAIPv2 ships none -- and it
      also fires on snow and on specular roofs. It counts bright pixels; it never
      modifies them, because clipping bright reflectance would destroy the
      radiometry the spectral-consistency loss depends on.
    - **constant**: the HR patch's standard deviation is below ``min_std``, i.e.
      there is no high-frequency detail to recover. Training on these teaches the
      model that the safe answer is the local mean.

    Counts are exposed on :attr:`counts` and rendered by :meth:`summary`. Call
    :meth:`log_summary` at the end of a pass -- a rejection rate that is quietly
    climbing is the first sign the data changed.

    Thresholds are never guessed here: construct with :meth:`from_cfg`.
    """

    RULES = ("nodata", "cloud", "constant")

    def __init__(
        self,
        max_nodata_fraction: float,
        min_std: float,
        bright_reflectance: float,
        max_bright_fraction: float,
        nodata_fill: float,
        std_statistic: str = "mean",
        enabled: bool = True,
    ) -> None:
        """
        Args:
            max_nodata_fraction: Reject above this fraction of nodata pixels.
            min_std: Reject below this HR standard deviation, in reflectance
                units.
            bright_reflectance: A pixel above this reflectance in *every* band
                counts as cloud-like.
            max_bright_fraction: Reject above this fraction of cloud-like pixels.
            nodata_fill: The reflectance value written at nodata pixels, used by
                the fallback mask. From ``cfg.dataset.nodata_fill``.
            std_statistic: ``"mean"`` or ``"max"``, how per-band standard
                deviations are combined.
            enabled: When False every patch is accepted, but the statistics are
                still computed and counted under ``would_reject_*``, so an
                ablation can measure what the filter would have removed.

        Raises:
            ValueError: ``std_statistic`` is not one of the two allowed values,
                or a threshold is negative.
        """
        if std_statistic not in ("mean", "max"):
            raise ValueError(
                f"std_statistic must be 'mean' or 'max', got {std_statistic!r}."
            )
        for name, value in (
            ("max_nodata_fraction", max_nodata_fraction),
            ("min_std", min_std),
            ("max_bright_fraction", max_bright_fraction),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}.")

        self.max_nodata_fraction = float(max_nodata_fraction)
        self.min_std = float(min_std)
        self.bright_reflectance = float(bright_reflectance)
        self.max_bright_fraction = float(max_bright_fraction)
        self.nodata_fill = float(nodata_fill)
        self.std_statistic = std_statistic
        self.enabled = bool(enabled)
        self.counts: Counter = Counter()

    @classmethod
    def from_cfg(cls, cfg: Any) -> "PatchFilter":
        """Build from ``cfg.patches.filters`` and ``cfg.dataset.nodata_fill``.

        Raises:
            KeyError: A threshold is missing from the config. Missing thresholds
                are not defaulted -- a filter running on invented numbers is how
                a training set silently loses half its patches.
        """
        filters = cfg["patches"]["filters"]
        return cls(
            max_nodata_fraction=float(filters["max_nodata_fraction"]),
            min_std=float(filters["min_std"]),
            bright_reflectance=float(filters["bright_reflectance"]),
            max_bright_fraction=float(filters["max_bright_fraction"]),
            nodata_fill=float(cfg["dataset"]["nodata_fill"]),
            std_statistic=str(filters["std_statistic"]),
            enabled=bool(filters["enabled"]),
        )

    def statistics(
        self,
        lr_patch: Any,
        hr_patch: Any,
        nodata_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Measure one candidate patch pair.

        Args:
            lr_patch: ``(C, h, w)`` float32 surface reflectance, unclipped.
            hr_patch: ``(C, h*scale, w*scale)``, same units and band order.
            nodata_mask: Optional bool mask at LR resolution, ``(C, h, w)`` or
                ``(h, w)``, True where there is no valid observation. When None,
                nodata is inferred as "all bands equal ``nodata_fill``".

        Returns:
            ``{"nodata_fraction", "bright_fraction", "hr_std", "hr_mean"}``.
            ``nodata_fraction`` is over LR and HR pooled when inferred, or over
            the supplied mask when given; ``bright_fraction`` and ``hr_std`` are
            over the HR patch, in reflectance units.
        """
        lr_arr = _to_numpy(lr_patch)
        hr_arr = _to_numpy(hr_patch)

        if nodata_mask is not None:
            mask = np.asarray(nodata_mask, dtype=bool)
            nodata_fraction = float(mask.mean()) if mask.size else 0.0
        else:
            fill = np.float32(self.nodata_fill)
            lr_bad = np.all(lr_arr == fill, axis=0)
            hr_bad = np.all(hr_arr == fill, axis=0)
            total = lr_bad.size + hr_bad.size
            nodata_fraction = (
                float((int(lr_bad.sum()) + int(hr_bad.sum())) / total) if total else 0.0
            )

        bright = np.all(hr_arr > np.float32(self.bright_reflectance), axis=0)
        bright_fraction = float(bright.mean()) if bright.size else 0.0

        per_band_std = hr_arr.reshape(hr_arr.shape[0], -1).std(axis=1)
        hr_std = float(
            per_band_std.mean() if self.std_statistic == "mean" else per_band_std.max()
        )

        return {
            "nodata_fraction": nodata_fraction,
            "bright_fraction": bright_fraction,
            "hr_std": hr_std,
            "hr_mean": float(hr_arr.mean()),
        }

    def __call__(
        self,
        lr_patch: Any,
        hr_patch: Any,
        nodata_mask: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Decide one patch, updating the counters.

        Args:
            lr_patch: ``(C, h, w)`` float32 reflectance.
            hr_patch: ``(C, h*scale, w*scale)`` float32 reflectance.
            nodata_mask: See :meth:`statistics`.

        Returns:
            None to accept, or the name of the rule that rejected it
            (``"nodata"``, ``"cloud"``, or ``"constant"``). When the filter is
            disabled, statistics are still measured and counted under
            ``would_reject_*`` but None is always returned.
        """
        stats = self.statistics(lr_patch, hr_patch, nodata_mask=nodata_mask)
        self.counts["examined"] += 1

        reason: Optional[str] = None
        if stats["nodata_fraction"] > self.max_nodata_fraction:
            reason = "nodata"
        elif stats["bright_fraction"] > self.max_bright_fraction:
            reason = "cloud"
        elif stats["hr_std"] < self.min_std:
            reason = "constant"

        if reason is None:
            self.counts["accepted"] += 1
            return None

        if not self.enabled:
            self.counts[f"would_reject_{reason}"] += 1
            self.counts["accepted"] += 1
            return None

        self.counts[f"rejected_{reason}"] += 1
        return reason

    def reset(self) -> None:
        """Zero the counters, e.g. at the start of an epoch."""
        self.counts.clear()

    def summary(self) -> str:
        """A rejection report: one line per rule, with rates and thresholds."""
        examined = self.counts.get("examined", 0)
        accepted = self.counts.get("accepted", 0)
        head = (
            f"patch filter (enabled={self.enabled}): examined={examined} "
            f"accepted={accepted}"
        )
        if examined:
            head += f" ({accepted / examined:.1%})"
        lines = [head]

        prefix = "rejected" if self.enabled else "would_reject"
        verb = "rejected" if self.enabled else "would reject"
        thresholds = {
            "nodata": f"> {self.max_nodata_fraction:.1%} nodata",
            "cloud": (
                f"> {self.max_bright_fraction:.1%} of pixels above "
                f"{self.bright_reflectance} reflectance in every band"
            ),
            "constant": f"HR {self.std_statistic} std < {self.min_std}",
        }
        for rule in self.RULES:
            count = self.counts.get(f"{prefix}_{rule}", 0)
            rate = f" ({count / examined:.1%})" if examined else ""
            lines.append(
                f"  {rule:<9} {verb} {count:>7}{rate}   [{thresholds[rule]}]"
            )
        return "\n".join(lines)

    def log_summary(self, logger: Any) -> None:
        """Log :meth:`summary` at INFO, and warn when most patches were dropped."""
        logger.info("%s", self.summary())
        examined = self.counts.get("examined", 0)
        accepted = self.counts.get("accepted", 0)
        if examined and accepted / examined < 0.5:
            logger.warning(
                "Patch filter rejected %.1f%% of %d candidates. That is a "
                "different dataset from the one the config describes -- check "
                "cfg.patches.filters before spending GPU hours on it.",
                100.0 * (1.0 - accepted / examined),
                examined,
            )
