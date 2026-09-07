"""Tiled SR inference + GeoTIFF writer for DrishtiSR.

Handles rasters far larger than GPU memory: overlapping tiles, raised-cosine
(Hann) blend weights, weight-normalised accumulation -> no seams. Output keeps
CRS and gets an affine with pixel size divided by `scale`.

Place at: src/drishtisr/infer/tiled.py

CLI:
    python -m drishtisr.infer.tiled --ckpt runs/runA/best.pt \
        --input delhi_lr.tif --output delhi_sr.tif --tile 256 --overlap 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from drishtisr.models.edsr import build_model
except ImportError:
    from edsr import build_model  # type: ignore


# ---------------------------------------------------------------- windows
def hann_1d(n: int, ramp: int) -> np.ndarray:
    """1 in the middle, raised-cosine ramp of `ramp` px at each end."""
    w = np.ones(n, dtype=np.float32)
    if ramp > 0:
        r = np.arange(ramp, dtype=np.float32)
        edge = 0.5 * (1.0 - np.cos(np.pi * (r + 0.5) / ramp))
        w[:ramp] = edge
        w[n - ramp:] = edge[::-1]
    return w


def blend_window(h: int, w: int, ramp: int) -> np.ndarray:
    return np.outer(hann_1d(h, ramp), hann_1d(w, ramp)).astype(np.float32)


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
    """lr: float32 [C,H,W] in ~[0,1] -> float32 [C,H*scale,W*scale]."""
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
            with torch.cuda.amp.autocast(enabled=amp and device == "cuda"):
                sr = model(t)
            sr = sr.float().squeeze(0).clamp(0, 1).cpu().numpy()
            sr = sr[:, : ph * scale, : pw * scale]
            win = blend_window(ph * scale, pw * scale, overlap * scale // 2)
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
) -> dict:
    import rasterio
    from rasterio.transform import Affine

    with rasterio.open(src_path) as src:
        idx = bands or list(range(1, src.count + 1))
        arr = src.read(idx).astype(np.float32) / reflect_div
        prof = src.profile.copy()
        transform, crs = src.transform, src.crs

    sr = sr_array(arr, model, scale, tile, overlap, device, amp)

    if out_dtype == "uint16":
        data = np.clip(sr * reflect_div, 0, 65535).astype(np.uint16)
    else:
        data = sr.astype(np.float32)

    # 10 m -> 2.5 m: same origin, pixel size / scale
    new_tf = transform * Affine.scale(1.0 / scale, 1.0 / scale)
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
    p.add_argument("--out-dtype", default="uint16", choices=["uint16", "float32"])
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    model = load_ckpt(a.ckpt, a.device)
    info = run_file(a.input, a.output, model, a.scale, a.tile, a.overlap, a.device,
                    bool(a.amp), a.reflect_div, a.out_dtype)
    print(info)


if __name__ == "__main__":
    main()
