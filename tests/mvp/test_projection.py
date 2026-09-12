"""P5 add-on: consistency projection sr <- sr + U(lr - D(sr))."""

from __future__ import annotations

import numpy as np

from src.infer.trust import compute_trust, degrade, project_consistency, upsample


class BicubicPredictor:
    info = {"backend": "fake", "checkpoint_id": "fake", "params": 0, "model_bytes": 0,
            "threads": 1, "interim": False, "has_scale_head": False}

    def __call__(self, lr: np.ndarray) -> np.ndarray:
        return upsample(lr)


def test_fixed_point_is_unchanged():
    rng = np.random.default_rng(3)
    # Block-constant SR of dyadic values (k/1024): the float32 4x4 block mean is
    # exact, so lr == D(sr) bit for bit.
    lr = (rng.integers(20, 400, (4, 8, 8)) / 1024.0).astype(np.float32)
    sr = np.repeat(np.repeat(lr, 4, axis=1), 4, axis=2)
    assert np.array_equal(degrade(sr), lr)
    out = project_consistency(sr, lr, 3)
    assert out.dtype == np.float32 and np.array_equal(out, sr)


def test_consistency_error_strictly_decreases():
    rng = np.random.default_rng(4)
    sr = rng.uniform(0.0, 0.5, (4, 64, 64)).astype(np.float32)
    lr = rng.uniform(0.0, 0.5, (4, 16, 16)).astype(np.float32)
    errs = [float(np.abs(degrade(project_consistency(sr, lr, k)) - lr).mean()) for k in (0, 1, 2, 3)]
    assert errs[0] > errs[1] > errs[2] > errs[3], errs


def test_compute_trust_projection_keyword():
    lr = np.random.default_rng(5).uniform(0.02, 0.4, (4, 16, 16)).astype(np.float32)
    base = compute_trust(lr, BicubicPredictor(), tta=4)
    proj = compute_trust(lr, BicubicPredictor(), tta=4, project_iters=2)
    assert proj.scalars["spec_l1"] < base.scalars["spec_l1"]
    assert np.array_equal(proj.bicubic, base.bicubic)
    assert np.array_equal(proj.unc_bands, base.unc_bands)
