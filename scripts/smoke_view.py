"""Day 2 eyeball test: is the SR output actually sharper, or just smoothed?

Renders two figures from a Sentinel-2 LR scene and the super-resolved GeoTIFF
produced by ``drishtisr.infer.tiled``:

``outputs/day2_smoke.png``
    Three panels over one 256x256 LR crop centred on built-up fabric --
    nearest-neighbour x4, bicubic x4, and the EDSR output read back from the
    SR GeoTIFF. The third panel is READ FROM DISK rather than recomputed, so
    the figure shows the delivered artefact, including anything the tiling,
    the blend and the uint16 write did to it.

``outputs/day2_smoke_full.png``
    The whole scene, LR and SR, decimated to a viewable size with the crop
    footprint marked. Per-pixel sharpness does not survive that decimation and
    is not claimed here; tile seams and global colour shift do, and that is
    what this panel is for.

The one rule that makes the comparison honest
---------------------------------------------
All three panels share ONE percentile stretch, computed once from the LR crop
and applied unchanged to each. This matters more than it sounds. A per-panel
stretch silently hands the blurriest panel its own contrast boost:
nearest-neighbour re-rendered against its own 2nd/98th percentiles looks
crisper than it is, and a comparison drawn that way cannot be read at all. The
LR crop is used as the reference because it is the common ancestor of all
three panels, so no panel is stretched to fit itself.

The stretch is a DISPLAY transform. It is applied to a copy on its way to
matplotlib and never to data that reaches a loss, a metric or a written
raster. The clip into [0, 1] at the end of it is imposed by RGB rendering --
a screen has no colour for reflectance 1.4 -- not by the model, and the
fraction of samples it touches is logged per panel, so a stretch that is
flattening bright targets shows up as a number rather than as a nice figure.

Examples:
    # Offline pre-flight against a fabricated raster pair, seconds, CPU.
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/smoke_view.py --smoke

    # The real thing, after drishtisr.infer.tiled has written the SR raster.
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/smoke_view.py

    # Pin the crop instead of searching for one.
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/smoke_view.py \\
        --set smoke_view.crop_xy=[512,640]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

LOGGER = get_logger("smoke_view")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the Day 2 three-panel comparison (nearest / bicubic / "
            "EDSR) and a full-scene overview from an LR scene and its SR "
            "GeoTIFF."
        ),
    )
    add_standard_args(parser)
    return parser.parse_args(argv)


# ------------------------------------------------------------------ bands
def resolve_band_indices(available: Sequence[str], wanted: Sequence[str]) -> List[int]:
    """Map band NAMES to zero-based positions in the raster's band axis.

    Resolution is by name against ``cfg.dataset.bands`` and not by position.
    ``delhi_*.tif`` is written R, G, B, NIR, so a positional ``[0, 1, 2]``
    happens to be correct today and would render false colour, silently and
    plausibly, the day the band order changes.

    Args:
        available: Band names in raster order, e.g. ``["B04","B03","B02","B08"]``.
        wanted: Band names to locate, e.g. ``["B04","B03","B02"]``.

    Returns:
        Zero-based indices into the band axis, in ``wanted`` order.

    Raises:
        KeyError: A requested band is not present.
    """
    avail = list(available)
    out: List[int] = []
    for name in wanted:
        if name not in avail:
            raise KeyError(
                f"display band {name!r} is not in the raster's bands {avail}. "
                "smoke_view.display_rgb_bands must name bands that "
                "dataset.bands actually carries."
            )
        out.append(avail.index(name))
    return out


# ------------------------------------------------------------------ crop search
def select_builtup_crop(
    lr: np.ndarray,
    red_idx: int,
    nir_idx: int,
    crop: int,
    stride: int,
    margin_frac: float,
    ndvi_weight: float,
    variance_weight: float,
) -> Tuple[int, int, Dict[str, float]]:
    """Locate a ``crop`` x ``crop`` LR window over built-up fabric.

    lr: surface reflectance, float32 ``[C, H, W]`` (channel-first), nominally
        [0, 1] and UNCLIPPED -- bright roofs exceed 1.0 and are exactly what
        this search is looking for.

    Two terms, because either one alone picks the wrong thing. Low NDVI finds
    unvegetated surfaces, but open water is the least vegetated thing in the
    scene and is flat and featureless. High brightness variance finds
    structure, but a field boundary is high-variance farmland. Built-up fabric
    is the window that scores on both, so the terms are z-scored across
    candidates -- NDVI and variance share no unit and cannot be added raw --
    and then weighted.

    Returns:
        ``(x, y, diagnostics)`` with ``x``/``y`` the top-left corner in LR
        pixels and ``diagnostics`` carrying the winning window's mean NDVI and
        brightness standard deviation, both reflectance-domain, for logging.

    Raises:
        ValueError: The scene is smaller than one crop, or no candidate window
            exists at the configured stride and margin.
    """
    _, h, w = lr.shape
    if h < crop or w < crop:
        raise ValueError(
            f"scene is {h}x{w} LR px but smoke_view.crop_lr_px is {crop}; "
            "the crop does not fit. Lower crop_lr_px or use a larger scene."
        )

    margin = int(round(min(h, w) * margin_frac))
    # Keep the margin from eating a small scene entirely.
    margin = min(margin, max(0, (min(h, w) - crop) // 2))
    y_hi, x_hi = h - crop - margin, w - crop - margin
    ys = list(range(margin, max(margin, y_hi) + 1, stride))
    xs = list(range(margin, max(margin, x_hi) + 1, stride))

    red, nir = lr[red_idx], lr[nir_idx]
    # Brightness as the mean over all bands: a single-band proxy would make the
    # variance term depend on which band happened to be indexed.
    bright = lr.mean(axis=0)

    cands: List[Tuple[int, int, float, float]] = []
    for y in ys:
        for x in xs:
            r = red[y:y + crop, x:x + crop]
            n = nir[y:y + crop, x:x + crop]
            b = bright[y:y + crop, x:x + crop]
            denom = n + r
            # NDVI is undefined where the sum vanishes. Guard the DIVISOR
            # rather than the result: forcing the ratio to zero instead would
            # read as "perfectly unvegetated" and win the search outright.
            safe = np.where(np.abs(denom) < 1e-6, np.nan, denom)
            ndvi = float(np.nanmean((n - r) / safe))
            cands.append((x, y, ndvi, float(b.std())))

    if not cands:
        raise ValueError(
            f"no candidate windows in a {h}x{w} scene at crop={crop}, "
            f"stride={stride}, margin_frac={margin_frac}."
        )

    ndvis = np.array([c[2] for c in cands], dtype=np.float64)
    stds = np.array([c[3] for c in cands], dtype=np.float64)

    def z(a: np.ndarray) -> np.ndarray:
        s = a.std()
        # A degenerate spread means the term cannot discriminate; contributing
        # zeros lets the other term decide instead of producing NaNs.
        return np.zeros_like(a) if s < 1e-12 else (a - a.mean()) / s

    score = variance_weight * z(stds) - ndvi_weight * z(ndvis)
    best = int(np.argmax(score))
    x, y, ndvi, std = cands[best]
    LOGGER.info(
        "built-up crop search: %d candidates, winner at (x=%d, y=%d) "
        "mean NDVI %.3f, brightness sd %.4f",
        len(cands), x, y, ndvi, std,
    )
    return x, y, {"mean_ndvi": ndvi, "brightness_sd": std, "candidates": float(len(cands))}


# ------------------------------------------------------------------ display
def stretch_bounds(
    ref: np.ndarray, percentiles: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-band display bounds from a reference array.

    ref: surface reflectance, float32 ``[C, H, W]``, unclipped.

    Returns:
        ``(lo, hi)``, each float32 ``[C, 1, 1]`` so they broadcast over a
        channel-first image. Bands are stretched independently -- one shared
        bound across R, G and B would render the composite with a colour cast
        that has nothing to do with the model.

    Raises:
        ValueError: ``percentiles`` is not an increasing pair.
    """
    if len(percentiles) != 2 or not percentiles[0] < percentiles[1]:
        raise ValueError(
            f"display_percentiles must be an increasing [lo, hi] pair, got "
            f"{list(percentiles)!r}."
        )
    lo = np.percentile(ref, percentiles[0], axis=(1, 2)).astype(np.float32)
    hi = np.percentile(ref, percentiles[1], axis=(1, 2)).astype(np.float32)
    # A constant band gives lo == hi and would divide by zero below.
    hi = np.where(hi - lo < 1e-8, lo + 1e-8, hi).astype(np.float32)
    return lo[:, None, None], hi[:, None, None]


def to_rgb(chw: np.ndarray, lo: np.ndarray, hi: np.ndarray, label: str) -> np.ndarray:
    """Apply a FIXED stretch and return an image matplotlib can draw.

    chw: surface reflectance, float32 ``[3, H, W]`` in display-band order
        (R, G, B), unclipped.
    lo, hi: bounds from :func:`stretch_bounds`, float32 ``[3, 1, 1]``.

    Returns:
        float32 ``[H, W, 3]`` (H, W, C axis order) in [0, 1] -- display
        intensities, NOT reflectance. The clip into [0, 1] belongs to RGB
        rendering, not to the data: a monitor has no colour for reflectance
        1.4. The fraction of samples it moves is logged, so a stretch that is
        flattening bright roofs is visible as a number.
    """
    scaled = (chw - lo) / (hi - lo)
    outside = float(np.count_nonzero((scaled < 0.0) | (scaled > 1.0)) / scaled.size)
    LOGGER.info(
        "panel %-14s display clip touched %5.2f%% of samples "
        "(reflectance range %.4f..%.4f)",
        label, 100.0 * outside, float(chw.min()), float(chw.max()),
    )
    return np.clip(scaled, 0.0, 1.0).transpose(1, 2, 0).astype(np.float32)


def upsample(chw: np.ndarray, scale: int, mode: str) -> np.ndarray:
    """Upsample a reflectance array by an integer factor.

    chw: surface reflectance, float32 ``[C, H, W]``, unclipped.
    mode: ``"nearest"`` or ``"bicubic"``.

    Returns:
        surface reflectance, float32 ``[C, H*scale, W*scale]``, unclipped.
        Bicubic overshoot is left in place: it is a real property of the
        baseline the model is being compared against, and clamping it here
        would quietly flatter bicubic in the figure.
    """
    t = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
    kwargs: Dict[str, Any] = {"scale_factor": float(scale), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    out = F.interpolate(t, **kwargs)
    return out.squeeze(0).numpy().astype(np.float32)


# ------------------------------------------------------------------ io
def read_window(path: Path, idx: Sequence[int], window: Any, div: float) -> np.ndarray:
    """Read selected bands of a window and convert DN to reflectance.

    Returns:
        surface reflectance, float32 ``[len(idx), h, w]``, nominally [0, 1]
        and UNCLIPPED (``DN / cfg.dataset.reflectance_scale``).
    """
    import rasterio

    with rasterio.open(path) as src:
        # rasterio band numbers are 1-based.
        arr = src.read([i + 1 for i in idx], window=window).astype(np.float32)
    return arr / div


def read_overview(
    path: Path, idx: Sequence[int], long_edge: int, div: float
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Read a decimated whole-scene view.

    Decimation happens inside GDAL via ``out_shape`` rather than by striding a
    full-resolution array, so the ~130 MB SR raster never lands in memory.

    Returns:
        ``(reflectance, (height, width))`` with reflectance float32
        ``[len(idx), h, w]``, unclipped, and the ORIGINAL raster shape.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        h, w = src.height, src.width
        factor = max(1.0, max(h, w) / float(long_edge))
        oh, ow = max(1, int(round(h / factor))), max(1, int(round(w / factor)))
        arr = src.read(
            [i + 1 for i in idx],
            out_shape=(len(idx), oh, ow),
            resampling=Resampling.average,
        ).astype(np.float32)
    return arr / div, (h, w)


# ------------------------------------------------------------------ figures
def figure_crop(
    panels: Sequence[Tuple[str, np.ndarray]],
    out_path: Path,
    dpi: int,
    suptitle: str,
) -> None:
    """Write the three-panel comparison.

    panels: ``(title, image)`` pairs, each image float32 ``[H, W, 3]``
        (H, W, C axis order) in [0, 1] display intensities.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    for ax, (title, img) in zip(np.atleast_1d(axes), panels):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(suptitle, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


def figure_full(
    lr_img: np.ndarray,
    sr_img: np.ndarray,
    crop_box_lr: Tuple[int, int, int],
    lr_shape: Tuple[int, int],
    out_path: Path,
    dpi: int,
    suptitle: str,
) -> None:
    """Write the whole-scene overview with the crop footprint marked.

    lr_img, sr_img: float32 ``[H, W, 3]`` (H, W, C) display intensities in
        [0, 1], decimated and stretched identically to each other.
    crop_box_lr: ``(x, y, size)`` in LR pixels; drawn in each panel's own
        decimated coordinates.

    Both panels are decimated, so this figure answers "are there tile seams or
    a global colour shift", not "is it sharper".
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    x, y, crop = crop_box_lr
    lr_h, lr_w = lr_shape
    for ax, img, title in (
        (axes[0], lr_img, "Sentinel-2 LR (10 m)"),
        (axes[1], sr_img, "EDSR SR (2.5 m)"),
    ):
        ax.imshow(img, interpolation="nearest")
        fy, fx = img.shape[0] / lr_h, img.shape[1] / lr_w
        ax.add_patch(
            Rectangle(
                (x * fx, y * fy), crop * fx, crop * fy,
                fill=False, edgecolor="#ff2d55", linewidth=1.4,
            )
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(suptitle, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


# ------------------------------------------------------------------ smoke
def fabricate_pair(
    cfg: Any, lr_path: Path, sr_path: Path, bands: Sequence[str]
) -> None:
    """Write a synthetic LR/SR GeoTIFF pair so --smoke needs no data or network.

    The pair is real on disk -- written and read back through rasterio, with a
    real UTM transform and the project's own reflectance scale -- so the crop
    search, the band-name resolution, the shared stretch and both figures run
    exactly as they do on Delhi. Only the imagery is invented, and the SR half
    is a bicubic upsample, so a --smoke figure shows no super-resolution and
    must never be read as a result.
    """
    import rasterio
    from rasterio.transform import Affine

    n = int(cfg.smoke_view.fabricate_size_lr_px)
    scale = int(cfg.smoke_view.scale)
    div = float(cfg.dataset.reflectance_scale)
    rng = np.random.default_rng(int(cfg.seed))

    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    # Blocky "built" texture in one half, smooth "vegetation" in the other, so
    # the crop search has a real choice to make rather than a coin flip.
    built = 0.25 + 0.12 * (((xx // 4) + (yy // 4)) % 2) + 0.02 * rng.standard_normal((n, n))
    veg = 0.10 + 0.03 * np.sin(xx / 6.0) + 0.01 * rng.standard_normal((n, n))
    half = n // 2
    red = np.where(xx < half, built, veg).astype(np.float32)
    green = red * 0.95
    blue = red * 0.90
    # NIR high over the smooth half so NDVI separates the two, as in a real scene.
    nir = np.where(xx < half, red * 0.9, red * 3.0).astype(np.float32)
    stack = {"B04": red, "B03": green, "B02": blue, "B08": nir}
    missing = [b for b in bands if b not in stack]
    if missing:
        raise KeyError(
            f"the smoke fixture only fabricates {sorted(stack)}, but "
            f"dataset.bands asks for {missing}."
        )
    lr = np.stack([stack[b] for b in bands], axis=0).astype(np.float32)

    sr = upsample(lr, scale, "bicubic")
    origin_x, origin_y = 710110.0, 3172450.0
    for path, arr, px in ((lr_path, lr, 10.0), (sr_path, sr, 10.0 / scale)):
        dn = np.clip(arr * div, 0, 65535).astype(np.uint16)
        path.parent.mkdir(parents=True, exist_ok=True)
        prof = dict(
            driver="GTiff", height=dn.shape[1], width=dn.shape[2], count=dn.shape[0],
            dtype="uint16", crs="EPSG:32643",
            transform=Affine(px, 0.0, origin_x, 0.0, -px, origin_y),
            compress="deflate",
        )
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(dn)
            dst.descriptions = tuple(bands)
    LOGGER.info("fabricated smoke pair: %s, %s", lr_path, sr_path)


# ------------------------------------------------------------------ main
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)
    seed = seed_everything(cfg.seed)
    LOGGER.info("smoke_view starting (seed %d, smoke=%s)", seed, args.smoke)

    sv = cfg.smoke_view
    root = repo_root()
    bands = [str(b) for b in cfg.dataset.bands]
    div = float(cfg.dataset.reflectance_scale)
    scale = int(sv.scale)
    crop = int(sv.crop_lr_px)

    lr_path = root / str(sv.input)
    sr_path = root / str(sv.sr)

    if bool(sv.get("fabricate", False)):
        fabricate_pair(cfg, lr_path, sr_path, bands)

    for label, path in (("input", lr_path), ("sr", sr_path)):
        if not path.is_file():
            raise FileNotFoundError(
                f"smoke_view.{label} does not exist: {path}. Fetch the scene "
                "with scripts/fetch_delhi.py and produce the SR raster with "
                "`python -m drishtisr.infer.tiled` first, or run --smoke."
            )

    rgb_idx = resolve_band_indices(bands, [str(b) for b in sv.display_rgb_bands])
    red_idx = resolve_band_indices(bands, ["B04"])[0]
    nir_idx = resolve_band_indices(bands, ["B08"])[0]

    import rasterio
    from rasterio.windows import Window

    with rasterio.open(lr_path) as src:
        lr_h, lr_w, lr_crs, lr_res = src.height, src.width, src.crs, src.res
        lr_full = src.read().astype(np.float32) / div
    with rasterio.open(sr_path) as src:
        sr_h, sr_w, sr_crs, sr_res = src.height, src.width, src.crs, src.res

    # The SR raster must be exactly `scale` times the LR raster, or the crop
    # windows below address different ground and the figure compares two
    # different places while looking perfectly plausible.
    if (sr_h, sr_w) != (lr_h * scale, lr_w * scale):
        raise ValueError(
            f"SR raster is {sr_h}x{sr_w} but {lr_h}x{lr_w} at scale={scale} "
            f"requires {lr_h * scale}x{lr_w * scale}. {sr_path} was not "
            "produced from this input at this scale."
        )
    if sr_crs != lr_crs:
        raise ValueError(
            f"CRS mismatch: input is {lr_crs}, SR is {sr_crs}. The panels "
            "would not be over the same ground."
        )

    if sv.crop_xy is None:
        x, y, diag = select_builtup_crop(
            lr_full, red_idx, nir_idx, crop,
            int(sv.search_stride_lr_px), float(sv.search_margin_frac),
            float(sv.ndvi_weight), float(sv.variance_weight),
        )
    else:
        x, y = int(sv.crop_xy[0]), int(sv.crop_xy[1])
        if not (0 <= x <= lr_w - crop and 0 <= y <= lr_h - crop):
            raise ValueError(
                f"smoke_view.crop_xy=[{x}, {y}] with crop_lr_px={crop} falls "
                f"outside the {lr_h}x{lr_w} scene."
            )
        diag = {"mean_ndvi": float("nan"), "brightness_sd": float("nan")}
        LOGGER.info("using pinned crop (x=%d, y=%d)", x, y)

    lr_crop = read_window(lr_path, rgb_idx, Window(x, y, crop, crop), div)
    sr_crop = read_window(
        sr_path, rgb_idx,
        Window(x * scale, y * scale, crop * scale, crop * scale), div,
    )

    # ONE stretch for all three panels, from the LR crop. See the module
    # docstring: this is the whole reason the figure can be believed.
    p_lo, p_hi = float(sv.display_percentiles[0]), float(sv.display_percentiles[1])
    lo, hi = stretch_bounds(lr_crop, [p_lo, p_hi])
    LOGGER.info(
        "shared stretch from LR crop, p%.1f-p%.1f reflectance lo=%s hi=%s",
        p_lo, p_hi,
        np.round(lo.ravel(), 4).tolist(), np.round(hi.ravel(), 4).tolist(),
    )

    nn = upsample(lr_crop, scale, "nearest")
    bic = upsample(lr_crop, scale, "bicubic")

    px_lr, px_sr = float(lr_res[0]), float(sr_res[0])
    panels = [
        (f"Nearest x{scale} (from {px_lr:g} m)", to_rgb(nn, lo, hi, "nearest")),
        (f"Bicubic x{scale}", to_rgb(bic, lo, hi, "bicubic")),
        (f"EDSR SR ({px_sr:g} m)", to_rgb(sr_crop, lo, hi, "edsr-sr")),
    ]
    figure_crop(
        panels, root / str(sv.out_crop), int(sv.figure_dpi),
        f"{lr_path.name} - {crop}x{crop} LR crop at (x={x}, y={y}), "
        f"RGB {'/'.join(str(b) for b in sv.display_rgb_bands)}, "
        f"mean NDVI {diag['mean_ndvi']:.3f}. "
        f"Identical p{p_lo:g}-p{p_hi:g} stretch on all three panels, "
        f"computed from the LR crop.",
    )

    lr_ov, _ = read_overview(lr_path, rgb_idx, int(sv.overview_long_edge_px), div)
    sr_ov, _ = read_overview(sr_path, rgb_idx, int(sv.overview_long_edge_px), div)
    # The overview gets its own stretch, shared across its two panels: it
    # covers the whole scene, whose range is not the crop's, but LR and SR are
    # still stretched identically to each other so a colour shift stays visible.
    ov_lo, ov_hi = stretch_bounds(lr_ov, [p_lo, p_hi])
    figure_full(
        to_rgb(lr_ov, ov_lo, ov_hi, "overview-lr"),
        to_rgb(sr_ov, ov_lo, ov_hi, "overview-sr"),
        (x, y, crop), (lr_h, lr_w),
        root / str(sv.out_full), int(sv.figure_dpi),
        f"{lr_path.name} whole scene, decimated to ~{int(sv.overview_long_edge_px)} px "
        f"long edge. LR {lr_h}x{lr_w} @ {px_lr:g} m, SR {sr_h}x{sr_w} @ {px_sr:g} m. "
        f"Red box = the crop above. Shared p{p_lo:g}-p{p_hi:g} stretch from LR.",
    )

    if args.smoke:
        LOGGER.info(
            "--smoke complete. The SR panel is a bicubic upsample of fabricated "
            "imagery; it shows no super-resolution and is not a result."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
