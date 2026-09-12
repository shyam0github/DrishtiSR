"""P5: trust maps (TTA / learned head / spectral consistency) and overlay rendering.

Pure synthetic tests: fake predictors, no checkpoints or data needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.baselines import bicubic_upsample
from src.infer.render import legend_stops, overlay_rgba, stretch_params, to_png_array
from src.infer.trust import D4, TrustResult, apply_transform, compute_trust, invert_transform


class BicubicPredictor:
    """Equivariant fake: bicubic x4 of each LR in the batch."""

    def __init__(self, packed: bool = False) -> None:
        self.packed = packed
        self.info = {"backend": "fake", "checkpoint_id": "fake", "params": 0, "model_bytes": 0,
                     "threads": 1, "interim": False, "has_scale_head": packed}

    def __call__(self, lr: np.ndarray) -> np.ndarray:
        sr = np.asarray(bicubic_upsample(lr, 4, antialias=True), dtype=np.float32)
        if self.packed:
            b = np.full_like(sr, 0.01) + 0.001 * np.arange(4, dtype=np.float32)[None, :, None, None]
            sr = np.concatenate([sr, b], axis=1)
        return sr


def _lr(h=16, w=16, seed=0):
    return np.random.default_rng(seed).uniform(0.02, 0.4, (4, h, w)).astype(np.float32)


@pytest.mark.parametrize("t", D4)
def test_d4_inverse_is_identity(t):
    x = np.random.default_rng(1).normal(size=(2, 4, 5, 7)).astype(np.float32)
    assert np.array_equal(invert_transform(apply_transform(x, t), t), x)


def test_d4_elements_distinct():
    x = np.arange(9, dtype=np.float32).reshape(3, 3)
    assert len({apply_transform(x, t).tobytes() for t in D4}) == 8


def test_tta8_std_zero_for_equivariant_predictor():
    res = compute_trust(_lr(24, 24), BicubicPredictor(), tta=8)
    assert res.method == "tta8"
    assert float(res.unc_bands[:, 8:-8, 8:-8].max()) < 1e-5


@pytest.mark.parametrize("tta,method", [(0, "none"), (4, "tta4"), (8, "tta8")])
@pytest.mark.parametrize("shape", [(16, 16), (12, 20)])
def test_result_shapes(tta, method, shape):
    h, w = shape
    lr = _lr(h, w)
    hr = np.asarray(bicubic_upsample(lr, 4), dtype=np.float32)
    res = compute_trust(lr, BicubicPredictor(), tta=tta, hr=hr)
    assert isinstance(res, TrustResult) and res.method == method
    for a in (res.sr, res.bicubic):
        assert a.shape == (4, 4 * h, 4 * w) and a.dtype == np.float32
    assert res.cons_l1.shape == res.cons_sam_deg.shape == (h, w)
    assert res.cons_l1.dtype == res.cons_sam_deg.dtype == np.float32
    if tta == 0:
        assert res.unc is None and res.unc_bands is None
        assert res.scalars["unc_mean"] is None and res.timings_ms["uncertainty"] is None
    else:
        assert res.unc.shape == (4 * h, 4 * w) and res.unc.dtype == np.float32
        assert res.unc_bands.shape == (4, 4 * h, 4 * w) and res.unc_bands.dtype == np.float32
        assert res.timings_ms["uncertainty"] is not None
    for k in ("spec_l1", "spec_sam_deg", "spec_l1_bicubic", "spec_sam_bicubic_deg",
              "hf_ratio_vs_bicubic", "spec_l1_hr", "spec_sam_hr_deg", "hf_ratio_hr_vs_bicubic"):
        assert np.isfinite(res.scalars[k]), k
    assert res.scalars["hf_ratio_vs_bicubic"] == pytest.approx(1.0)
    assert np.isclose(res.scalars["spec_l1"], float(res.cons_l1.mean()), rtol=1e-4)


def test_learned_laplace_from_packed_channels():
    res = compute_trust(_lr(), BicubicPredictor(packed=True), tta=8)
    assert res.method == "learned_laplace"
    assert res.sr.shape == (4, 64, 64)
    assert np.allclose(res.unc_bands[2], 0.012) and np.allclose(res.unc, 0.0115)


def test_overlay_alpha_and_fixed_scale():
    vmax = 0.02
    a = overlay_rgba(np.array([[0.0, 0.0005, 0.01, 0.02]]), vmax)
    b = overlay_rgba(np.array([[0.0, 0.0005, 0.01, 5.0]]), vmax)
    assert a.shape == (1, 4, 4) and a.dtype == np.uint8
    assert a[0, 0, 3] == 0 and a[0, 1, 3] == 0          # 0 and below 5% of vmax
    assert np.array_equal(a[0, :3], b[0, :3])           # same value -> same RGBA
    assert a[0, 3, 3] == round(0.85 * 255) == b[0, 3, 3]  # clipped at vmax
    assert 0 < a[0, 2, 3] < a[0, 3, 3]


def test_render_helpers():
    lr = _lr()
    p = stretch_params(lr)
    assert p.shape == (4, 2) and (p[:, 1] > p[:, 0]).all()
    for mode in ("rgb", "fcc"):
        img = to_png_array(lr, p, mode)
        assert img.shape == (16, 16, 3) and img.dtype == np.uint8
    assert legend_stops(0.02) == pytest.approx([0, 0.005, 0.01, 0.015, 0.02])
