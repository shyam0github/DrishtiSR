"""Tiled-inference correctness: seams, odd sizes, and georeferencing.

These tests use ``nn.Upsample(scale_factor=4, mode='bicubic')`` as an
identity-like stand-in for the SR network. That is the point of them: bicubic
upsampling of an integer-offset crop is *exactly* equal to the corresponding
window of the bicubic upsampling of the whole raster (with
``align_corners=False`` the source coordinate of output pixel ``j`` in a tile
at LR offset ``x0`` is ``(j + 0.5)/4 - 0.5 + x0``, so the four taps line up),
except within ~6 output pixels of a tile edge where the kernel clamps against
padding instead of reading real neighbours.

So the tiler is correct exactly when it reproduces a single full-image forward
pass. No checkpoint, no data and no GPU are needed to say so.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.infer.tiled import run_file, sr_array

SCALE = 4


def _stand_in_model() -> nn.Module:
    """Identity-like SR stand-in: pure bicubic x4, no learned parameters."""
    return nn.Upsample(scale_factor=SCALE, mode="bicubic")


def _full_image_forward(model: nn.Module, lr: np.ndarray) -> np.ndarray:
    """Single-shot reference: model over the whole raster, no tiling.

    lr: float32 [C, H, W] -> float32 [C, H*SCALE, W*SCALE].
    """
    with torch.no_grad():
        return model(torch.from_numpy(lr).unsqueeze(0)).squeeze(0).numpy()


def test_tiled_matches_full_image_forward_no_seams():
    """The seam check: tiled output must equal an untiled forward pass.

    Any seam -- a blend that does not sum to one, a border pixel divided by a
    collapsed accumulator, a silent clamp -- shows up here as a difference
    against the reference.
    """
    rng = np.random.default_rng(0)
    lr = rng.random((4, 300, 220), dtype=np.float32)  # reflectance-like, [0, 1)
    model = _stand_in_model()

    tiled = sr_array(lr, model, scale=SCALE, tile=128, overlap=32,
                     device="cpu", amp=False)

    assert tiled.shape == (4, 1200, 880)
    assert tiled.dtype == np.float32
    assert np.isfinite(tiled).all(), "tiled output contains NaN or inf"

    full = _full_image_forward(model, lr)
    max_abs_diff = float(np.abs(tiled - full).max())
    print(f"\nseam max-abs-diff = {max_abs_diff:.3e}")
    assert max_abs_diff < 1e-3, f"seam artefacts: max|tiled - full| = {max_abs_diff:.3e}"


def test_non_divisible_size_does_not_raise():
    """A raster smaller than one tile, with sides divisible by nothing useful.

    101 x 97 is smaller than tile=128 in both axes and neither side is a
    multiple of the scale, the tile size, or the step. The tiler must pad,
    crop back, and return the exact x4 shape rather than raising or silently
    returning a padded raster.
    """
    rng = np.random.default_rng(1)
    lr = rng.random((4, 101, 97), dtype=np.float32)

    out = sr_array(lr, _stand_in_model(), scale=SCALE, tile=128, overlap=32,
                   device="cpu", amp=False)

    assert out.shape == (4, 404, 388)
    assert np.isfinite(out).all()


def test_run_file_preserves_georeferencing(tmp_path):
    """CRS and origin survive; only the pixel size shrinks by `scale`.

    Writes a 4-band uint16 GeoTIFF on a 10 m EPSG:32643 grid, super-resolves
    it, and checks the output is the same scene on a 2.5 m grid -- not a
    shifted, reprojected, or half-pixel-offset one.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import Affine

    size, px, origin_x, origin_y = 128, 10.0, 300000.0, 3100000.0
    src_transform = Affine.translation(origin_x, origin_y) @ Affine.scale(px, -px)
    rng = np.random.default_rng(2)
    # uint16 digital numbers; /10000 -> reflectance in [0, 1)
    data = rng.integers(0, 10000, size=(4, size, size), dtype=np.uint16)

    src_path = tmp_path / "lr.tif"
    dst_path = tmp_path / "sr.tif"
    profile = dict(driver="GTiff", height=size, width=size, count=4,
                   dtype="uint16", crs="EPSG:32643", transform=src_transform)
    with rasterio.open(src_path, "w", **profile) as dst:
        dst.write(data)

    run_file(str(src_path), str(dst_path), _stand_in_model(), scale=SCALE,
             tile=64, overlap=16, device="cpu", amp=False, out_dtype="uint16")

    with rasterio.open(src_path) as src, rasterio.open(dst_path) as out:
        print(f"\noutput transform = {out.transform}")
        assert out.crs == src.crs
        assert out.transform.a == pytest.approx(px / SCALE)      # 10 m -> 2.5 m
        assert out.transform.e == pytest.approx(-px / SCALE)
        # same top-left corner: no shift, no half-pixel drift
        assert out.transform.c == pytest.approx(src.transform.c)
        assert out.transform.f == pytest.approx(src.transform.f)
        assert (out.width, out.height) == (size * SCALE, size * SCALE)
        assert out.count == 4
        assert out.dtypes == ("uint16",) * 4
