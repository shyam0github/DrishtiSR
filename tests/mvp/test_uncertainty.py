"""P6: learned Laplace scale head, NLL, collapse watch, calibration."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.edsr import build_model
from src.uncertainty.calibration import ause, calibration_report
from src.uncertainty.head import EDSRWithScale, total_params
from src.uncertainty.nll import laplace_nll
from src.uncertainty.watch import abort_reasons, watch_stats


def _small(b0: float = 0.02) -> EDSRWithScale:
    torch.manual_seed(0)
    return EDSRWithScale(build_model(n_resblocks=2, n_feats=16), b0=b0)


def test_param_budget():
    m = EDSRWithScale(build_model(n_resblocks=16, n_feats=48))
    assert sum(p.numel() for p in m.backbone.parameters()) == 855_652
    assert total_params(m) < 1_000_000


def test_b_equals_b0_at_init():
    b0 = 0.0137
    _, b = _small(b0)(torch.rand(2, 4, 12, 12))
    assert b.shape == (2, 4, 48, 48)
    assert torch.allclose(b, torch.full_like(b, b0), rtol=1e-3, atol=0)


def test_packed_has_8_channels():
    assert _small().forward_packed(torch.rand(1, 4, 8, 8)).shape == (1, 8, 32, 32)


def _train_steps(m: EDSRWithScale, n: int) -> None:
    opt = torch.optim.Adam(m.head.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(1)
    m.train()
    for _ in range(n):
        lr, hr = torch.rand(2, 4, 8, 8, generator=g), torch.rand(2, 4, 32, 32, generator=g)
        mu, b = m(lr)
        opt.zero_grad()
        laplace_nll(mu, b, hr).backward()
        opt.step()


def test_backbone_grads_none_after_step():
    m = _small()
    _train_steps(m, 1)
    assert all(not p.requires_grad and p.grad is None for p in m.backbone.parameters())
    assert any(p.grad is not None for p in m.head.parameters())


def test_mu_bit_identical_after_training():
    m = _small()
    x = torch.rand(2, 4, 8, 8)
    mu0, b0 = m(x)
    _train_steps(m, 5)
    mu1, b1 = m(x)
    assert torch.equal(mu0, mu1)
    assert not torch.equal(b0, b1)  # the head did train


def test_nll_finite_with_extreme_residuals():
    mu = torch.zeros(1, 4, 4, 4)
    y = torch.tensor([0.0, 1e3, -1e3, 1e-9]).view(1, 4, 1, 1).expand(1, 4, 4, 4)
    for b_val in (1e-4, 1.0, 1e4):
        b = torch.full((1, 4, 4, 4), b_val, requires_grad=True)
        for beta in (0.0, 0.5):
            loss = laplace_nll(mu, b, y, beta=beta)
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(b.grad).all()
            b.grad = None


def _synthetic_err(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(0.0, 0.02, size=(n, 4, 32, 32))) * rng.uniform(0.2, 3.0, size=(n, 1, 32, 32))


def test_watch_aborts_on_constant_b():
    err = _synthetic_err()
    reasons = abort_reasons(watch_stats(np.full_like(err, 0.01), err))
    assert any("rho" in r for r in reasons) and any("spatial_cv" in r for r in reasons)


def test_watch_aborts_on_floor_collapse():
    err = _synthetic_err()
    assert any("frac_floor" in r for r in abort_reasons(watch_stats(np.full_like(err, 1.5e-4) + 1e-6 * err, err)))


def test_watch_passes_on_b_proportional_to_err():
    err = _synthetic_err()
    stats = watch_stats(2.0 * err + 3e-4, err)
    assert stats["rho"] > 0.99
    assert abort_reasons(stats) == []


def test_ause_zero_when_unc_equals_err():
    err = _synthetic_err()
    assert ause(err, err)["ause"] == pytest.approx(0.0, abs=1e-15)
    assert ause(err, -err)["ause"] > 0  # worst ranking is penalised


def test_calibration_report_coverage():
    rng = np.random.default_rng(3)
    b = rng.uniform(0.005, 0.05, size=(4, 4, 64, 64))
    err = rng.laplace(0.0, b)
    rep = calibration_report(err, b, laplace=True)
    assert rep["coverage"]["50"]["empirical"] == pytest.approx(0.5, abs=0.02)
    assert rep["coverage"]["90"]["empirical"] == pytest.approx(0.9, abs=0.02)
    assert calibration_report(err, b, laplace=False)["coverage"] == "N/A"


def test_legacy_tta_names_still_importable():
    from src.uncertainty import TTAResult, sr_output, tta_predict  # noqa: F401
