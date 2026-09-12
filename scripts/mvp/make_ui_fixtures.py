"""Procedurally generate the PNGs referenced by app/fixtures/*.json (UI mock mode).

Synthetic field-and-road scene, not real imagery. numpy + PIL only; deterministic.

    python scripts/mvp/make_ui_fixtures.py [--out app/fixtures]

Outputs (512x512 unless noted): lr_{rgb,fcc} (blocky nearest x4), bicubic_{rgb,fcc},
sr_{rgb,fcc} (sharper than bicubic), hr_{rgb,fcc}, uncertainty.png and
consistency.png (RGBA overlays, fixed 0..0.02 reflectance scale), thumb_val.png and
thumb_delhi.png (96x96).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

HR = 512
SCALE = 4
DISPLAY_MAX = 0.02  # matches refs.unc_display_max / cons_display_max in sample_response.json

# Colour ramps; the same stops are used by the legend in app/static/styles.css.
INFERNO = [(0.0, "#000004"), (0.25, "#420a68"), (0.5, "#932667"),
           (0.75, "#dd513a"), (0.9, "#fca50a"), (1.0, "#fcffa4")]
VIRIDIS = [(0.0, "#440154"), (0.25, "#3b528b"), (0.5, "#21918c"),
           (0.75, "#5ec962"), (1.0, "#fde725")]

# Band order R, G, B, NIR (inventory §3); reflectance.
COVER = {
    "crop": (0.04, 0.08, 0.035, 0.36),
    "young": (0.07, 0.10, 0.05, 0.28),
    "soil": (0.18, 0.15, 0.11, 0.24),
    "stubble": (0.13, 0.12, 0.08, 0.27),
    "water": (0.02, 0.03, 0.04, 0.02),
}
ASPHALT = (0.10, 0.10, 0.10, 0.13)
DIRT = (0.21, 0.18, 0.14, 0.24)
ROOF = (0.27, 0.25, 0.23, 0.29)
HEDGE = (0.03, 0.05, 0.03, 0.30)


def gaussian_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian over the last two axes, reflect padding."""
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for axis in (-2, -1):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (r, r)
        xp = np.pad(x, pad, mode="reflect")
        n = x.shape[axis]
        x = sum(w * np.take(xp, range(i, i + n), axis=axis) for i, w in enumerate(k))
    return x


def make_scene(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:HR, 0:HR].astype(np.float32)
    img = np.zeros((4, HR, HR), np.float32)

    # Fields: Voronoi parcels with directional crop-row texture.
    n = 34
    seeds = rng.uniform(0, HR, (n, 2))
    d = np.stack([np.hypot(yy - sy, xx - sx) for sy, sx in seeds])
    order = np.argsort(d, axis=0)
    nearest = order[0]
    d_sorted = np.take_along_axis(d, order[:2], axis=0)
    kinds = rng.choice(list(COVER), n, p=[0.36, 0.2, 0.2, 0.2, 0.04])
    for i in range(n):
        m = nearest == i
        base = np.array(COVER[kinds[i]], np.float32)[:, None]
        theta = rng.uniform(0, np.pi)
        period = rng.uniform(5, 11)
        rows = np.sin(2 * np.pi * (xx[m] * np.cos(theta) + yy[m] * np.sin(theta)) / period)
        amp = 0.0 if kinds[i] == "water" else rng.uniform(0.08, 0.2)
        img[:, m] = base * (1 + amp * rows) * rng.uniform(0.9, 1.1)
    hedge = (d_sorted[1] - d_sorted[0]) < 1.6
    img[:, hedge] = np.array(HEDGE, np.float32)[:, None]

    # Roads: two straight asphalt roads and one curved dirt track.
    def paint(mask, colour):
        img[:, mask] = np.array(colour, np.float32)[:, None]

    for _ in range(2):
        a = rng.uniform(0, np.pi)
        c = rng.uniform(0.3, 0.7) * HR
        dist = np.abs((xx - HR / 2) * np.sin(a) - (yy - HR / 2) * np.cos(a) - (c - HR / 2))
        paint(dist < 3.2, ASPHALT)
        paint((dist > 3.2) & (dist < 4.0), (0.16, 0.15, 0.13, 0.2))
    track = np.abs(yy - (0.3 * HR + 40 * np.sin(xx / 70.0 + rng.uniform(0, 6))))
    paint(track < 1.6, DIRT)

    # Buildings: a small cluster of roofs.
    cy, cx = rng.uniform(0.25, 0.75, 2) * HR
    for _ in range(16):
        h, w = rng.integers(5, 13, 2)
        y0 = int(np.clip(cy + rng.normal(0, 35), 0, HR - h))
        x0 = int(np.clip(cx + rng.normal(0, 35), 0, HR - w))
        img[:, y0:y0 + h, x0:x0 + w] = np.array(ROOF, np.float32)[:, None, None] * rng.uniform(0.85, 1.15)
        img[:, y0 + h:y0 + h + 2, x0 + 1:x0 + w + 1] *= 0.6  # shadow

    img += rng.normal(0, 0.004, img.shape).astype(np.float32)
    return np.clip(img, 0.001, None)


def block_mean(x: np.ndarray, s: int) -> np.ndarray:
    c, h, w = x.shape
    return x.reshape(c, h // s, s, w // s, s).mean(axis=(2, 4))


def bicubic_up(lr: np.ndarray, s: int) -> np.ndarray:
    return np.stack([
        np.asarray(Image.fromarray(b.astype(np.float32), mode="F")
                   .resize((b.shape[1] * s, b.shape[0] * s), Image.BICUBIC))
        for b in lr
    ])


def stretch(lr: np.ndarray):
    """Per-band 2-98 % stretch computed on LR, applied identically to every product."""
    lo = np.percentile(lr, 2, axis=(1, 2))[:, None, None]
    hi = np.percentile(lr, 98, axis=(1, 2))[:, None, None]
    return lambda x: (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def to_png(x8: np.ndarray, idx) -> Image.Image:
    return Image.fromarray(np.moveaxis(x8[list(idx)], 0, -1), "RGB")


def ramp(v: np.ndarray, stops) -> np.ndarray:
    pos = [p for p, _ in stops]
    cols = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for _, h in stops], np.float32)
    return np.stack([np.interp(v, pos, cols[:, k]) for k in range(3)], -1)


def overlay(value: np.ndarray, stops, vmax: float) -> Image.Image:
    v = np.clip(value / vmax, 0, 1)
    rgb = ramp(v, stops)
    alpha = 255 * np.clip(v, 0, 1) ** 0.6
    return Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")


def hotspots(rng, n: int, size: int, sig=(20, 60)) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    out = np.zeros((size, size), np.float32)
    for _ in range(n):
        cy, cx = rng.uniform(0, size, 2)
        s = rng.uniform(*sig)
        out += rng.uniform(0.4, 1.0) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=root / "app" / "fixtures")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)

    hr = make_scene(seed=3)
    lr = block_mean(hr, SCALE)
    bic = bicubic_up(lr, SCALE)
    # "SR": bicubic plus most of the lost detail back; sharper than bicubic, softer than HR.
    sr = bic + 0.75 * (gaussian_blur(hr, 0.7) - bic)
    lr_near = np.repeat(np.repeat(lr, SCALE, axis=1), SCALE, axis=2)

    s = stretch(lr)
    rgb, fcc = (0, 1, 2), (3, 0, 1)
    for name, arr in [("lr", lr_near), ("bicubic", bic), ("sr", sr), ("hr", hr)]:
        x8 = s(arr)
        to_png(x8, rgb).save(out / f"{name}_rgb.png", optimize=True)
        to_png(x8, fcc).save(out / f"{name}_fcc.png", optimize=True)

    # Uncertainty: smooth hotspots plus a little edge structure, reflectance units.
    edges = np.abs(gaussian_blur(hr, 1.0) - gaussian_blur(hr, 2.5)).mean(0)
    unc = 0.018 * hotspots(rng, 6, HR) + 0.12 * gaussian_blur(edges, 1.5)
    overlay(unc, INFERNO, DISPLAY_MAX).save(out / "uncertainty.png", optimize=True)

    # Consistency: |degrade(SR) - LR| on the 10 m grid (plus hotspots), nearest x4.
    lr_err = np.abs(block_mean(sr, SCALE) - lr).mean(0)
    lr_err = lr_err + 0.02 * hotspots(rng, 5, HR // SCALE, sig=(8, 22))
    cons = np.repeat(np.repeat(lr_err, SCALE, 0), SCALE, 1)
    overlay(cons, VIRIDIS, DISPLAY_MAX).save(out / "consistency.png", optimize=True)

    x8 = s(hr)
    to_png(x8, rgb).resize((96, 96), Image.LANCZOS).save(out / "thumb_val.png")
    other = make_scene(seed=21)
    to_png(stretch(block_mean(other, SCALE))(other), rgb).resize((96, 96), Image.LANCZOS).save(
        out / "thumb_delhi.png")

    for p in sorted(out.glob("*.png")):
        print(f"{p.name:20s} {Image.open(p).size}")


if __name__ == "__main__":
    main()
