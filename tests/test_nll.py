"""The Gaussian NLL, its warmup schedule, and the sigma-collapse monitor.

CPU only, no data.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.losses import gaussian_nll, nll_weight_at
from src.metrics.logvar import LOGVAR_STAT_KEYS, LogvarMonitor


def _tensors(seed=0):
    torch.manual_seed(seed)
    sr = torch.rand(2, 4, 8, 8) * 0.3
    hr = sr + 0.01 * torch.randn_like(sr)
    # Well inside the [-10, 10] clamp, so the unclamped reference formulas
    # below describe the same quantity as the (clamping) loss.
    logvar = -8.0 + 0.4 * torch.randn_like(sr)
    assert float(logvar.min()) > -10.0
    return sr, logvar, hr


# -- the formula ----------------------------------------------------------------


def test_matches_the_stated_formula():
    sr, logvar, hr = _tensors()
    expected = (0.5 * (torch.exp(-logvar) * (hr - sr) ** 2 + logvar)).mean()
    assert torch.allclose(gaussian_nll(sr, logvar, hr), expected, rtol=1e-6)


def test_matches_torch_gaussian_nll_loss():
    """An independent implementation of the same likelihood (constant dropped)."""
    sr, logvar, hr = _tensors(1)
    ours = gaussian_nll(sr, logvar, hr)
    theirs = F.gaussian_nll_loss(sr, hr, torch.exp(logvar), full=False, eps=1e-30)
    assert torch.allclose(ours, theirs, rtol=1e-5)


def test_the_minimiser_is_log_squared_residual():
    """d/ds of 0.5 (r^2 e^-s + s) vanishes at s = ln r^2 -- the head learns r^2."""
    r = 0.02
    sr = torch.zeros(1, 1, 1, 1)
    hr = torch.full_like(sr, r)
    s_star = math.log(r * r)
    s = torch.full_like(sr, s_star, requires_grad=True)
    gaussian_nll(sr, s, hr).backward()
    assert abs(float(s.grad)) < 1e-6
    at = lambda v: float(gaussian_nll(sr, torch.full_like(sr, v), hr))  # noqa: E731
    assert at(s_star) < at(s_star - 1.0) and at(s_star) < at(s_star + 1.0)


def test_logvar_is_clamped_inside_the_loss():
    sr, _, hr = _tensors(2)
    low = torch.full_like(sr, -50.0, requires_grad=True)
    assert torch.allclose(gaussian_nll(sr, low, hr), gaussian_nll(sr, torch.full_like(sr, -10.0), hr))
    gaussian_nll(sr, low, hr).backward()
    assert torch.count_nonzero(low.grad) == 0


def test_a_shape_mismatch_is_refused():
    sr, logvar, hr = _tensors()
    with pytest.raises(ValueError, match="identical shapes"):
        gaussian_nll(sr, logvar[:, :1], hr)


# -- the schedule ---------------------------------------------------------------


@pytest.mark.parametrize("it, expected", [
    (0, 0.0), (1999, 0.0),              # pure L1
    (2000, 1.0 / 2000), (2999, 0.5),    # linear ramp
    (3999, 1.0), (4000, 1.0), (40000, 1.0),
])
def test_warmup_then_linear_ramp(it, expected):
    assert nll_weight_at(it, 2000, 2000, 1.0) == pytest.approx(expected)


def test_the_schedule_never_decreases():
    weights = [nll_weight_at(it, 50, 30, 0.7) for it in range(200)]
    assert all(b >= a for a, b in zip(weights, weights[1:]))
    assert weights[49] == 0.0 and weights[-1] == pytest.approx(0.7)


def test_zero_ramp_is_a_step():
    assert nll_weight_at(9, 10, 0, 2.0) == 0.0
    assert nll_weight_at(10, 10, 0, 2.0) == 2.0


def test_negative_settings_are_refused():
    with pytest.raises(ValueError):
        nll_weight_at(0, -1, 10, 1.0)


# -- the sigma-collapse monitor ------------------------------------------------


def test_a_constant_map_reads_as_collapsed():
    monitor = LogvarMonitor(-10.0, 10.0)
    monitor.update(torch.full((3, 4, 16, 16), -7.5))
    stats = monitor.summary()
    assert set(stats) == set(LOGVAR_STAT_KEYS)
    assert stats["spatial_std"] == 0.0
    assert stats["mean"] == stats["min"] == stats["max"] == -7.5
    assert stats["clamp_lo_frac"] == stats["clamp_hi_frac"] == 0.0


def test_spatial_std_is_per_map_then_averaged():
    """Two maps, each constant but different: spatial std 0, not the pooled std."""
    lv = torch.stack([torch.full((1, 4, 4), -8.0), torch.full((1, 4, 4), -4.0)])
    monitor = LogvarMonitor(-10.0, 10.0)
    monitor.update(lv)
    assert monitor.summary()["spatial_std"] == 0.0
    assert monitor.summary()["mean"] == -6.0


def test_clamp_fractions_and_streaming():
    torch.manual_seed(0)
    a = torch.randn(2, 4, 8, 8)
    b = torch.randn(3, 4, 8, 8)
    a[0, 0, :2] = -10.0
    b[1, 2, 0, :4] = 10.0
    split, whole = LogvarMonitor(-10.0, 10.0), LogvarMonitor(-10.0, 10.0)
    split.update(a)
    split.update(b)
    whole.update(torch.cat([a, b]))
    for key in LOGVAR_STAT_KEYS:
        assert split.summary()[key] == pytest.approx(whole.summary()[key], rel=1e-12)
    n = a.numel() + b.numel()
    assert split.summary()["clamp_lo_frac"] == pytest.approx(16 / n)
    assert split.summary()["clamp_hi_frac"] == pytest.approx(4 / n)


def test_a_nan_is_loud_and_an_empty_monitor_is_an_error():
    monitor = LogvarMonitor(-10.0, 10.0)
    with pytest.raises(RuntimeError):
        monitor.summary()
    with pytest.raises(ValueError, match="diverged"):
        monitor.update(torch.full((1, 1, 2, 2), float("nan")))
