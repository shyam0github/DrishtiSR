"""Tiled SR inference + GeoTIFF writer for DrishtiSR.

Handles rasters far larger than GPU memory: overlapping tiles, raised-cosine
(Hann) blend weights, weight-normalised accumulation -> no seams. Output keeps
CRS and gets an affine with pixel size divided by `scale`.

Lives at src/infer/tiled.py; importable as either ``src.infer.tiled`` or
``drishtisr.infer.tiled`` (see the ``drishtisr`` alias package).

CLI:
    python -m drishtisr.infer.tiled --ckpt runs/runA/best.pt \
        --input delhi_lr.tif --output delhi_sr.tif --tile 256 --overlap 32
"""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np
import torch

try:
    from drishtisr.models.edsr import build_model
    from drishtisr.utils.logging import get_logger
except ImportError:
    from src.models.edsr import build_model  # type: ignore
    from src.utils.logging import get_logger  # type: ignore

LOGGER = get_logger("drishtisr.infer.tiled")


# ---------------------------------------------------------------- windows
def hann_1d(n: int, ramp: int, lo: bool = True, hi: bool = True) -> np.ndarray:
    """1 in the middle, raised-cosine ramp of `ramp` px at each end.

    `lo` / `hi` select which end is ramped. A ramp is only correct where a
    NEIGHBOURING tile supplies the complementary weight; against the image
    border nothing else contributes, so tapering there would divide an almost
    zero numerator by an almost zero accumulator and destroy the border
    pixels. Callers pass `lo=False` / `hi=False` for edges that sit on the
    raster boundary.

    Returns float32 [n], values in [0, 1].
    """
    w = np.ones(n, dtype=np.float32)
    if ramp > 0:
        r = np.arange(ramp, dtype=np.float32)
        edge = 0.5 * (1.0 - np.cos(np.pi * (r + 0.5) / ramp))
        if lo:
            w[:ramp] = edge
        if hi:
            w[n - ramp:] = edge[::-1]
    return w


def blend_window(
    h: int,
    w: int,
    ramp: int,
    top: bool = True,
    bottom: bool = True,
    left: bool = True,
    right: bool = True,
) -> np.ndarray:
    """Separable Hann blend weights, float32 [h, w] (H, W axis order), in [0, 1].

    The four flags disable the ramp on edges that coincide with the raster
    border; see `hann_1d`.
    """
    return np.outer(
        hann_1d(h, ramp, top, bottom), hann_1d(w, ramp, left, right)
    ).astype(np.float32)


# ---------------------------------------------------------------- core
@torch.no_grad()
def sr_array(
    lr: np.ndarray,
    model: torch.nn.Module,
    scale: int = 4,
    tile: int = 256,
    overlap: int = 32,
    device: str = "cuda",
    amp: bool = True,
) -> np.ndarray:
    """Run `model` over overlapping tiles and blend them into one raster.

    lr: surface reflectance, float32 [C, H, W] (channel-first), nominally
        [0, 1] but UNCLIPPED -- cloud, snow and bright roofs exceed 1.0.
    returns: surface reflectance, float32 [C, H*scale, W*scale], same
        unclipped convention. Values are never clamped here; the model's own
        output range is passed through so spectral consistency can be checked
        downstream.

    Tiles are accumulated with Hann weights and divided by the accumulated
    weight, so an interior pixel is a convex combination of every tile that
    covers it. `ramp` spans the FULL overlap, which makes the two facing
    ramps sum to exactly 1 and keeps each tile's own edge pixels -- where a
    convolutional or interpolating model sees padded rather than real
    context -- at near-zero weight wherever a neighbour has real context.
    """
    c, h, w = lr.shape
    step = tile - overlap
    out = np.zeros((c, h * scale, w * scale), dtype=np.float32)
    acc = np.zeros((1, h * scale, w * scale), dtype=np.float32)
    ys = list(range(0, max(1, h - overlap), step))
    xs = list(range(0, max(1, w - overlap), step))
    model.eval()
    for y in ys:
        for x in xs:
            y0, x0 = min(y, max(0, h - tile)), min(x, max(0, w - tile))
            th, tw = min(tile, h - y0), min(tile, w - x0)
            patch = lr[:, y0:y0 + th, x0:x0 + tw]
            ph, pw = patch.shape[1], patch.shape[2]
            # pad to a multiple of `scale` (and to full tile) with reflection
            pad_h, pad_w = tile - ph, tile - pw
            if pad_h or pad_w:
                patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            t = torch.from_numpy(patch).unsqueeze(0).to(device)
            use_amp = amp and str(device).startswith("cuda")
            ctx = torch.autocast("cuda") if use_amp else contextlib.nullcontext()
            with ctx:
                sr = model(t)
            # No clamp: reflectance above 1.0 is physical, not an error.
            sr = sr.float().squeeze(0).cpu().numpy()
            sr = sr[:, : ph * scale, : pw * scale]
            win = blend_window(
                ph * scale, pw * scale, overlap * scale,
                top=y0 > 0, bottom=y0 + ph < h,
                left=x0 > 0, right=x0 + pw < w,
            )
            Y, X = y0 * scale, x0 * scale
            out[:, Y:Y + ph * scale, X:X + pw * scale] += sr * win
            acc[:, Y:Y + ph * scale, X:X + pw * scale] += win
    np.maximum(acc, 1e-6, out=acc)
    return out / acc


# ---------------------------------------------------------------- geotiff
def run_file(
    src_path: str,
    dst_path: str,
    model: torch.nn.Module,
    scale: int = 4,
    tile: int = 256,
    overlap: int = 32,
    device: str = "cuda",
    amp: bool = True,
    reflect_div: float = 10000.0,
    out_dtype: str = "uint16",
    bands: list[int] | None = None,
    dn_offset: float = 0.0,
) -> dict:
    """Super-resolve a GeoTIFF on disk, preserving its georeferencing.

    Reads uint16 digital numbers and converts them to surface reflectance with

        reflectance = (dn - dn_offset) / reflect_div

    giving float32 [C, H, W] (channel-first), nominally [0, 1] but UNCLIPPED --
    bright roofs, cloud and specular water legitimately exceed 1.0, and after
    the offset subtraction dark water may fall slightly below 0. Both are real
    radiometry and are passed to the model untouched. `sr_array` then runs, and
    the result is written back.

    `dn_offset` is Sentinel-2's BOA_ADD_OFFSET, expressed as the value to
    SUBTRACT (ESA publishes it as -1000, i.e. dn_offset=1000 here). It defaults
    to 0.0, which reproduces the previous `dn / reflect_div` behaviour exactly,
    so no existing caller changes. It matters because the training data is
    offset-corrected: MEASURED 2026-09-08 over 250 random SEN2NAIPv2-crosssensor
    LR patches, 82% of B02 pixels sit below DN 1000 with a floor at 0, in BOTH
    the pre-2022 and the 2022+ acquisition cohorts. Uncorrected baseline >= 04.00
    product cannot do that. A raw-DN scene fed in with dn_offset=0 therefore
    reaches the model a uniform +0.1 reflectance too bright, which is a large
    fractional error over water, asphalt and shadow -- most of an urban scene.
    See `cfg.delhi.dn_offset`.

    The output keeps the source CRS and top-left origin exactly; only the
    pixel size is divided by `scale` (10 m -> 2.5 m at scale=4).

    With `out_dtype="uint16"` the reflectance is inverted back through the
    SAME transform -- `dn = reflectance * reflect_div + dn_offset` -- so the
    output raster carries the identical DN convention as its input and stays
    comparable to it band for band. It is then clipped to [0, 65535]. That clip
    is a property of the storage dtype, not of the model: it is the only clip
    in this path, and `out_dtype="float32"` avoids it entirely, writing
    unclipped reflectance, when the true radiometry must survive.
    """
    import rasterio
    from rasterio.transform import Affine

    reflect_off = np.float32(dn_offset)

    with rasterio.open(src_path) as src:
        idx = bands or list(range(1, src.count + 1))
        arr = (src.read(idx).astype(np.float32) - reflect_off) / reflect_div
        prof = src.profile.copy()
        transform, crs = src.transform, src.crs

    sr = sr_array(arr, model, scale, tile, overlap, device, amp)

    if out_dtype == "uint16":
        dn = sr * reflect_div + reflect_off
        n_clipped = int(np.count_nonzero((dn < 0) | (dn > 65535)))
        if n_clipped:
            LOGGER.warning(
                "uint16 output clipped %d of %d samples outside [0, 65535] "
                "(reflectance [%.4f, %.4f]); use --out-dtype float32 to keep them",
                n_clipped, dn.size, float(sr.min()), float(sr.max()),
            )
        data = np.clip(dn, 0, 65535).astype(np.uint16)
    elif out_dtype == "float32":
        data = sr.astype(np.float32)
    else:
        raise ValueError(f"unsupported out_dtype {out_dtype!r}; use uint16 or float32")

    # 10 m -> 2.5 m: same origin, pixel size / scale
    new_tf = transform @ Affine.scale(1.0 / scale, 1.0 / scale)
    prof.update(driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
                dtype=data.dtype, transform=new_tf, crs=crs, compress="deflate",
                predictor=2, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **prof) as dst:
        dst.write(data)
    return {"out": dst_path, "shape": list(data.shape),
            "px": abs(new_tf.a), "crs": str(crs), "dtype": str(data.dtype)}


def load_ckpt(ckpt: str, device: str) -> torch.nn.Module:
    ck = torch.load(ckpt, map_location=device)
    a = ck.get("args", {})
    model = build_model("edsr_baseline", scale=a.get("scale", 4),
                        n_resblocks=a.get("n_resblocks", 16), n_feats=a.get("n_feats", 64),
                        in_ch=a.get("in_ch", 4), out_ch=a.get("in_ch", 4))
    model.load_state_dict(ck["model"])
    return model.to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--reflect-div", type=float, default=10000.0)
    # BOA_ADD_OFFSET as a value to SUBTRACT: pass 1000 for a Sentinel-2 L2A
    # scene at processing baseline >= 04.00 held on disk as raw DN. Default 0
    # keeps the historical behaviour for every other caller. See run_file.
    p.add_argument("--dn-offset", type=float, default=0.0)
    p.add_argument("--out-dtype", default="uint16", choices=["uint16", "float32"])
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    model = load_ckpt(a.ckpt, a.device)
    info = run_file(a.input, a.output, model, a.scale, a.tile, a.overlap, a.device,
                    bool(a.amp), a.reflect_div, a.out_dtype, dn_offset=a.dn_offset)
    print(info)


if __name__ == "__main__":
    main()
