"""The four Day 2 artefact diagnostics, with their definitions written down.

Day 2 section 3 of ``docs/PROGRESS.md`` reported a gradient ratio, a
flat-quartile high-frequency ratio, a ringing percentage and a colour-shift
MAE -- and recorded none of their definitions. When the BOA offset had to be
re-measured on 2026-09-08, only the two whose definitions happened to be
recoverable reproduced; the ringing and HF absolutes could not be compared to
the originals at all. This script exists so that cannot recur: every window
size, quantile and reduction is in ``cfg.artefact_metrics``, and the numbers
come out of one definition applied to every raster in one run.

**None of this is an accuracy measurement.** There is no 2.5 m ground truth
over Delhi. Every number here compares an SR raster against interpolation
baselines and against its own input, on data the model never saw. They are
diagnostics for artefacts, and they must not be quoted as quality.

The four:

1. **Gradient ratio.** Mean gradient magnitude of the SR crop over that of a
   bicubic upsample of the same LR crop. Above 1.0 means more edge energy than
   interpolation produces. It does not distinguish recovered detail from
   overshoot, which is why the next two exist.
2. **Flat-quartile HF energy ratio.** Mean squared high-pass response, SR over
   bicubic, restricted to the flattest quartile of the crop. The mask is
   computed on the LR INPUT, never on an SR output -- scoring a model inside a
   mask it drew itself is circular. A model that amplifies proportionally more
   where there is least real structure is inventing texture.
3. **Ringing.** The fraction of SR pixels falling outside the local min-max
   envelope of the LR source. A real detail recovered by super-resolution still
   lies within the range its neighbouring LR pixels bracket; overshoot does
   not. Bicubic is scored identically in the same run, because the number is
   meaningless without that reference.
4. **Spectral round trip.** The SR raster degraded back to the input GSD by
   area average, differenced against the input. Reported as MAE in reflectance
   AND as the per-band signed mean, because the sign is the finding: a
   consistent per-band bias is a colour shift, noise is not.

**On the DN convention, which is a trap.** Metrics 1 and 2 compare SR against
bicubic, both derived from the same crop, so they are invariant to an additive
offset. Metrics 3 and 4 compare SR against the LR INPUT in absolute terms, and
are NOT: read an SR raster written with ``dn_offset=1000`` against an LR crop
read with 0, and the ringing figure goes from 0.8% to 88% -- the whole raster
sits 0.1 reflectance below an envelope computed from unshifted pixels. So the
LR crop and its baselines are rebuilt for EACH raster in that raster's own
convention; see :func:`reference_crops`. This was MEASURED, not reasoned about:
the first version of this script read the input once with offset 0 and produced
exactly that 88%.

Examples:
    # Offline, seconds, against the pair smoke_view.py --smoke fabricates.
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/artefact_metrics.py --smoke

    # The real rasters listed in cfg.artefact_metrics.rasters.
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/artefact_metrics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

LOGGER = get_logger("artefact_metrics")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure gradient ratio, flat-area high-frequency energy, ringing "
            "and spectral round-trip error for one or more SR rasters."
        ),
    )
    add_standard_args(parser)
    return parser.parse_args(argv)


def read_crop(path: Path, x: int, y: int, width: int, height: int,
              scale: float, offset: float) -> np.ndarray:
    """Read a window and convert digital numbers to surface reflectance.

    Args:
        path: GeoTIFF of uint16 digital numbers.
        x, y, width, height: Window in that raster's OWN pixels.
        scale: ``cfg.dataset.reflectance_scale``.
        offset: DN to SUBTRACT before dividing -- the ``dn_offset`` the
            inference run used, so the result is the reflectance the model
            emitted rather than a re-derived one.

    Returns:
        surface reflectance, float32 ``(C, H, W)`` (channel-first), nominally
        [0, 1] and UNCLIPPED at both ends: bright roofs exceed 1.0 and
        offset-corrected dark water can fall below 0.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        if x < 0 or y < 0 or x + width > src.width or y + height > src.height:
            raise ValueError(
                f"Crop ({x}, {y}) {width}x{height} does not fit inside "
                f"{path.name}, which is {src.width}x{src.height}. Adjust "
                "cfg.artefact_metrics.crop_xy / crop_lr_px."
            )
        arr = src.read(window=Window(x, y, width, height)).astype(np.float32)
    return (arr - np.float32(offset)) / np.float32(scale)


def upsample(chw: np.ndarray, scale: int, mode: str) -> np.ndarray:
    """Upsample reflectance ``(C, H, W)`` by an integer factor.

    Bicubic overshoot is left in place: it is a real property of the baseline
    the model is measured against, and clamping it here would quietly flatter
    bicubic in every ratio below.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
    kwargs: Dict[str, Any] = {"scale_factor": float(scale), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs).squeeze(0).numpy().astype(np.float32)


def box_blur(chw: np.ndarray, window: int) -> np.ndarray:
    """Per-channel box mean over a ``window x window`` neighbourhood.

    Reflect padding, so edge pixels are not pulled toward zero by a pad value
    that is not a reflectance.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
    pad = window // 2
    kernel = torch.ones(chw.shape[0], 1, window, window) / float(window * window)
    padded = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, kernel, groups=chw.shape[0]).squeeze(0).numpy()


def gradient_magnitude(chw: np.ndarray) -> float:
    """Mean |gradient| over ``(C, H, W)`` reflectance.

    Forward differences in both axes, cropped to their common extent so x and
    y refer to the same pixels, combined as a hypotenuse and averaged over
    every channel and pixel.
    """
    gy = np.abs(np.diff(chw, axis=1))[:, :, :-1]
    gx = np.abs(np.diff(chw, axis=2))[:, :-1, :]
    return float(np.mean(np.hypot(gx, gy)))


def hf_energy(chw: np.ndarray, window: int, mask: Optional[np.ndarray] = None) -> float:
    """Mean squared high-pass response.

    ``high_pass = image - box_blur(image, window)``, so ``window`` IS the
    definition of "high frequency" here and changing it changes every HF
    number in the report.

    Args:
        chw: surface reflectance ``(C, H, W)``.
        window: Box-mean window in pixels of this array's own grid.
        mask: Optional ``(H, W)`` bool. True where a pixel counts.
    """
    residual = (chw - box_blur(chw, window)) ** 2
    if mask is None:
        return float(residual.mean())
    if mask.shape != chw.shape[1:]:
        raise ValueError(
            f"Mask {mask.shape} does not match the image grid {chw.shape[1:]}."
        )
    if not mask.any():
        raise ValueError(
            "The flat mask selected no pixels. Check "
            "cfg.artefact_metrics.flat_quantile."
        )
    return float(residual[:, mask].mean())


def flat_mask(lr: np.ndarray, window: int, quantile: float, scale: int) -> np.ndarray:
    """Flattest-fraction mask, computed on the LR INPUT and replicated to HR.

    Local variance of the LR band mean over ``window x window``; the lowest
    ``quantile`` fraction is "flat". Computing this on the input rather than on
    any SR output is deliberate -- a mask drawn from the array being scored
    would make the measurement circular.

    Returns:
        ``(H*scale, W*scale)`` bool on the HR grid, by nearest replication, so
        every HR pixel inherits the flatness of the LR pixel it came from.
    """
    brightness = lr.mean(axis=0, keepdims=True)
    mean = box_blur(brightness, window)
    mean_square = box_blur(brightness ** 2, window)
    variance = (mean_square - mean ** 2)[0]
    threshold = float(np.quantile(variance, quantile))
    flat_lr = variance <= threshold
    return np.repeat(np.repeat(flat_lr, scale, axis=0), scale, axis=1)


def envelope_violation_fraction(hr: np.ndarray, lr: np.ndarray, window: int,
                                scale: int) -> float:
    """Fraction of HR pixels outside the local min-max envelope of the LR source.

    The envelope is taken over ``window x window`` LR pixels (replicate padding
    at the border, so an edge pixel is bracketed by real neighbours rather than
    by zeros) and then nearest-replicated to the HR grid. A detail genuinely
    recovered by super-resolution still lies within the range its neighbouring
    LR pixels bracket; overshoot and ringing do not.

    Returns:
        A fraction in [0, 1], over every channel and pixel.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(lr)).unsqueeze(0)
    pad = window // 2
    padded = F.pad(tensor, (pad, pad, pad, pad), mode="replicate")
    high = F.max_pool2d(padded, window, stride=1).squeeze(0).numpy()
    low = -F.max_pool2d(-padded, window, stride=1).squeeze(0).numpy()

    high_hr = np.repeat(np.repeat(high, scale, axis=1), scale, axis=2)
    low_hr = np.repeat(np.repeat(low, scale, axis=1), scale, axis=2)
    if high_hr.shape != hr.shape:
        raise ValueError(
            f"Envelope grid {high_hr.shape} does not match the SR crop "
            f"{hr.shape}; the crop and scale must agree."
        )
    return float(np.mean((hr > high_hr) | (hr < low_hr)))


def spectral_round_trip(hr: np.ndarray, lr: np.ndarray, scale: int) -> Dict[str, Any]:
    """Degrade SR back to the input GSD by area average and difference it.

    Area average is the correct degradation for a reflectance quantity: the
    mean reflectance of the four-by-four block IS the reflectance of the 10 m
    pixel covering it. Returns MAE plus the per-band SIGNED mean, in
    reflectance, because a consistent per-band bias is a colour shift while
    noise averages away.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(hr)).unsqueeze(0)
    degraded = F.avg_pool2d(tensor, scale).squeeze(0).numpy()
    if degraded.shape != lr.shape:
        raise ValueError(
            f"Degraded SR is {degraded.shape}, input crop is {lr.shape}."
        )
    difference = degraded - lr
    return {
        "mae_reflectance": float(np.mean(np.abs(difference))),
        "per_band_signed_mean": [float(v) for v in difference.mean(axis=(1, 2))],
    }


def reference_crops(cfg: Any, lr_path: Path, dn_offset: float) -> Dict[str, Any]:
    """The LR crop and its baselines, in ONE raster's DN convention.

    Rebuilt per ``dn_offset`` rather than read once, because the ringing and
    spectral metrics compare an SR raster against this input in absolute terms.
    Mixing conventions -- SR offset-corrected, input not -- puts the SR raster a
    uniform 0.1 reflectance below an envelope built from unshifted pixels and
    reports 88% ringing for a raster whose real figure is 0.8%.

    Returns:
        ``{"lr", "bicubic", "nearest", "flat", "ringing_bicubic"}``. The first
        three are surface reflectance, float32, ``(C, H, W)`` for ``lr`` and
        ``(C, H*scale, W*scale)`` for the baselines, unclipped. ``flat`` is
        ``(H*scale, W*scale)`` bool. The flat mask and the bicubic ringing
        figure are both shift-invariant and so identical across offsets; they
        are recomputed here rather than special-cased.
    """
    am = cfg.artefact_metrics
    scale = int(am.scale)
    x, y = int(am.crop_xy[0]), int(am.crop_xy[1])
    crop = int(am.crop_lr_px)

    lr = read_crop(lr_path, x, y, crop, crop,
                   float(cfg.dataset.reflectance_scale), dn_offset)
    bicubic = upsample(lr, scale, "bicubic")
    return {
        "lr": lr,
        "bicubic": bicubic,
        "nearest": upsample(lr, scale, "nearest"),
        "flat": flat_mask(lr, int(am.flat_variance_window_lr_px),
                          float(am.flat_quantile), scale),
        "ringing_bicubic": envelope_violation_fraction(
            bicubic, lr, int(am.envelope_window_lr_px), scale),
    }


def score(cfg: Any, label: str, sr_path: Path, dn_offset: float,
          ref: Dict[str, Any]) -> Dict[str, Any]:
    """Score one SR raster against reference crops in ITS OWN DN convention."""
    am = cfg.artefact_metrics
    scale = int(am.scale)
    x, y = int(am.crop_xy[0]), int(am.crop_xy[1])
    crop = int(am.crop_lr_px)
    lr, bicubic, nearest, flat = (
        ref["lr"], ref["bicubic"], ref["nearest"], ref["flat"]
    )

    sr = read_crop(
        sr_path, x * scale, y * scale, crop * scale, crop * scale,
        float(cfg.dataset.reflectance_scale), dn_offset,
    )
    if sr.shape != bicubic.shape:
        raise ValueError(
            f"{sr_path.name} crop is {sr.shape}, the baselines are "
            f"{bicubic.shape}. cfg.artefact_metrics.scale must match the "
            "raster this was produced with."
        )

    window = int(am.highpass_window_px)
    g_sr = gradient_magnitude(sr)
    result: Dict[str, Any] = {
        "label": label,
        "path": str(sr_path),
        "dn_offset": dn_offset,
        "gradient_ratio_vs_bicubic": g_sr / gradient_magnitude(bicubic),
        "gradient_ratio_vs_nearest": g_sr / gradient_magnitude(nearest),
        "hf_ratio_whole_crop": hf_energy(sr, window) / hf_energy(bicubic, window),
        "hf_ratio_flat_quartile":
            hf_energy(sr, window, flat) / hf_energy(bicubic, window, flat),
        "ringing_frac_sr": envelope_violation_fraction(
            sr, lr, int(am.envelope_window_lr_px), scale),
        "reflectance_min": float(sr.min()),
        "reflectance_max": float(sr.max()),
    }
    result.update(spectral_round_trip(sr, lr, scale))
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)
    seed_everything(cfg.seed)

    am = cfg.artefact_metrics
    bands = [str(b) for b in cfg.dataset.bands]
    scale = int(am.scale)
    x, y = int(am.crop_xy[0]), int(am.crop_xy[1])
    crop = int(am.crop_lr_px)

    LOGGER.info(
        "artefact_metrics starting (seed %s, smoke=%s)", cfg.seed, bool(args.smoke)
    )

    lr_path = repo_root() / str(am.input)
    if not lr_path.is_file():
        raise FileNotFoundError(
            f"Input raster not found at {lr_path}. It is gitignored imagery; "
            "fetch it with scripts/fetch_delhi.py, or use --smoke, which scores "
            "the pair scripts/smoke_view.py --smoke fabricates."
        )

    rows: List[Dict[str, Any]] = []
    references: Dict[float, Dict[str, Any]] = {}
    for entry in am.rasters:
        sr_path = repo_root() / str(entry.path)
        if not sr_path.is_file():
            raise FileNotFoundError(
                f"SR raster not found at {sr_path} (label {str(entry.label)!r}). "
                "Produce it with `python -m drishtisr.infer.tiled`, or remove "
                "the entry from cfg.artefact_metrics.rasters."
            )
        offset = float(entry.dn_offset)
        # Cached by offset: rasters sharing a convention share one reference,
        # and a raster with a different one gets its own rather than being
        # scored against pixels 0.1 reflectance away from it.
        if offset not in references:
            references[offset] = reference_crops(cfg, lr_path, offset)
        rows.append(score(cfg, str(entry.label), sr_path, offset,
                          references[offset]))

    any_ref = references[float(am.rasters[0].dn_offset)]
    LOGGER.info(
        "crop (x=%d, y=%d) %dx%d LR px from %s; flat mask = lowest %.0f%% of "
        "local variance (%d px window), %d of %d HR pixels",
        x, y, crop, crop, lr_path.name, 100.0 * float(am.flat_quantile),
        int(am.flat_variance_window_lr_px), int(any_ref["flat"].sum()),
        any_ref["flat"].size,
    )
    # Shift-invariant, so one figure covers every offset in play. Asserted
    # rather than assumed: the entire point of this script is that an
    # unrecorded definition is worth nothing.
    ringing_values = {round(r["ringing_bicubic"], 9) for r in references.values()}
    if len(ringing_values) != 1:
        raise AssertionError(
            "The bicubic ringing reference differs across DN offsets "
            f"({sorted(ringing_values)}), which it cannot: bicubic and the "
            "envelope derive from the same crop and shift together."
        )
    ringing_bicubic = any_ref["ringing_bicubic"]
    LOGGER.info(
        "bicubic reference: ringing %.2f%% outside the %dx%d LR envelope",
        100.0 * ringing_bicubic, int(am.envelope_window_lr_px),
        int(am.envelope_window_lr_px),
    )

    LOGGER.info("")
    LOGGER.info("%-22s%12s%12s%12s%14s", "raster", "grad/bic", "HF flat", "ring %",
                "spectral MAE")
    for row in rows:
        LOGGER.info(
            "%-22s%11.3fx%11.3fx%11.2f%%%14.4f",
            row["label"][:22], row["gradient_ratio_vs_bicubic"],
            row["hf_ratio_flat_quartile"], 100.0 * row["ringing_frac_sr"],
            row["mae_reflectance"],
        )
    LOGGER.info("%-22s%12s%12s%11.2f%%%14s", "bicubic reference", "1.000x", "1.000x",
                100.0 * ringing_bicubic, "-")

    LOGGER.info("")
    LOGGER.info("per-band signed round-trip bias (reflectance); sign is the finding")
    LOGGER.info("%-22s%s", "raster", "".join(f"{b:>11}" for b in bands))
    for row in rows:
        LOGGER.info(
            "%-22s%s", row["label"][:22],
            "".join(f"{v:>+11.4f}" for v in row["per_band_signed_mean"]),
        )

    payload = {
        "input": str(lr_path),
        "crop_xy": [x, y],
        "crop_lr_px": crop,
        "scale": scale,
        "bands": bands,
        "definitions": {
            "highpass_window_px": int(am.highpass_window_px),
            "flat_quantile": float(am.flat_quantile),
            "flat_variance_window_lr_px": int(am.flat_variance_window_lr_px),
            "envelope_window_lr_px": int(am.envelope_window_lr_px),
        },
        "bicubic_ringing_frac": ringing_bicubic,
        "rasters": rows,
    }
    out = repo_root() / str(am.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("")
    LOGGER.info("wrote %s", out)

    if args.smoke:
        LOGGER.info(
            "--smoke scored a FABRICATED pair whose SR panel is a bicubic "
            "upsample. The numbers describe invented imagery and mean only "
            "that every metric executed."
        )
    else:
        LOGGER.info(
            "These are artefact diagnostics on data with NO ground truth. "
            "They compare an SR raster to interpolation and to its own input; "
            "none of them is an accuracy result."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
