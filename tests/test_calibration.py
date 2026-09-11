"""The binning behind scripts/calibrate_uncertainty.py: exact, streaming, honest.

A calibration curve is a claim about millions of pixels computed from a
histogram of them, so the histogram's arithmetic is tested directly: an
informative predictor must read monotone, an uninformative one must not, the
oracle must score AUSE ~0, and streaming must equal one-shot.

CPU only, no data.
"""

import numpy as np
import pytest
import torch

from src.eval.calibration import (
    PredictorHistogram,
    ause,
    controlled_ause_markdown,
    equal_mass_bins,
    gradient_magnitude,
    monotonicity,
    score_against_control,
    sparsification_curve,
)

FR = np.arange(0.0, 0.991, 0.01)


def _hist():
    return PredictorHistogram(n_bands=2, log10_min=-6.0, log10_max=1.0, n_fine=700)


def _informative(seed=0, n=64):
    """err ~ u * |N(0, 1)|: error scales with the predictor."""
    g = torch.Generator().manual_seed(seed)
    u = 10.0 ** (torch.rand(4, 2, n, n, generator=g) * 3.0 - 4.0)
    err = u * torch.randn(4, 2, n, n, generator=g).abs()
    return u, err


def test_every_pixel_is_counted_including_zero_and_overflow():
    hist = _hist()
    u = torch.tensor([0.0, 1e-3, 50.0, 1e-9]).view(1, 1, 2, 2).repeat(1, 2, 1, 1)
    hist.update(u, torch.ones_like(u))
    t = hist.totals()
    assert t["count"].sum() == 8
    assert t["count"][0] == 4          # u == 0 and 1e-9 underflow
    assert t["count"][-1] == 2         # 50 overflows 10**1


def test_an_informative_predictor_reads_monotone():
    hist = _hist()
    hist.update(*_informative())
    bins = equal_mass_bins(hist, 10)
    mono = monotonicity(bins)
    assert mono["monotone"] and mono["spearman"] == pytest.approx(1.0)
    assert mono["mae_ratio_top_bottom"] > 100


def test_an_uninformative_predictor_does_not():
    g = torch.Generator().manual_seed(1)
    u = 10.0 ** (torch.rand(4, 2, 64, 64, generator=g) * 3.0 - 4.0)
    err = 0.01 * torch.randn(4, 2, 64, 64, generator=g).abs()
    hist, oracle = _hist(), _hist()
    hist.update(u, err)
    oracle.update(err, err)
    mono = monotonicity(equal_mass_bins(hist, 10))
    assert mono["mae_ratio_top_bottom"] == pytest.approx(1.0, abs=0.1)
    score = ause(sparsification_curve(hist, FR), sparsification_curve(oracle, FR), FR)
    assert score["ause_ratio"] == pytest.approx(1.0, abs=0.1)


def test_bins_carry_equal_mass():
    hist = _hist()
    hist.update(*_informative(2))
    bins = equal_mass_bins(hist, 10)
    assert len(bins) == 10
    assert all(abs(b["mass"] - 0.1) < 0.01 for b in bins)
    assert sum(b["count"] for b in bins) == 4 * 2 * 64 * 64


def test_streaming_equals_one_shot():
    u, err = _informative(3)
    split, whole = _hist(), _hist()
    split.update(u[:2], err[:2])
    split.update(u[2:], err[2:])
    whole.update(u, err)
    for key in ("count", "sum_abs", "sum_sq", "sum_u"):
        assert torch.allclose(getattr(split, key), getattr(whole, key), rtol=1e-12)


def test_the_oracle_scores_zero_and_the_curve_starts_at_the_mae():
    u, err = _informative(4)
    hist, oracle = _hist(), _hist()
    hist.update(u, err)
    oracle.update(err, err)
    curve = sparsification_curve(oracle, FR)
    assert curve[0] == pytest.approx(float(err.double().mean()), rel=1e-9)
    assert all(b <= a + 1e-12 for a, b in zip(curve, curve[1:]))
    assert ause(curve, curve, FR)["ause"] == 0.0
    informative = ause(sparsification_curve(hist, FR), curve, FR)
    assert 0.0 < informative["ause_ratio"] < 0.5


def test_per_band_selection():
    hist = _hist()
    u, err = _informative(5)
    err[:, 1] *= 10.0
    hist.update(u, err)
    b0 = equal_mass_bins(hist, 5, band=0)
    b1 = equal_mass_bins(hist, 5, band=1)
    assert b1[0]["mae"] > 5 * b0[0]["mae"]


def test_gradient_magnitude_of_flat_and_ramp():
    flat = torch.full((1, 2, 8, 8), 0.3)
    assert float(gradient_magnitude(flat).abs().max()) == 0.0
    ramp = torch.arange(8.0).view(1, 1, 1, 8).expand(1, 2, 8, 8) * 0.01
    assert torch.allclose(gradient_magnitude(ramp), torch.full((1, 2, 8, 8), 0.01))


def _scored(seed=6):
    """An informative predictor, an uninformative control, and the oracle."""
    u, err = _informative(seed)
    noise = 10.0 ** (torch.rand_like(u) * 3.0 - 4.0)
    pred, ctl, oracle = _hist(), _hist(), _hist()
    pred.update(u, err)
    ctl.update(noise, err)
    oracle.update(err, err)
    return pred, ctl, oracle


def test_a_score_always_carries_the_control_and_the_delta():
    score = score_against_control(*_scored(), FR)
    assert set(score) == {"predictor", "control", "delta_ause", "delta_ratio", "beats_control"}
    assert score["delta_ause"] == pytest.approx(
        score["predictor"]["ause"] - score["control"]["ause"]
    )
    assert score["beats_control"] and score["delta_ause"] < 0


def test_a_control_scored_on_other_pixels_is_refused():
    pred, ctl, oracle = _scored()
    u, err = _informative(7)
    ctl.update(u[:1], err[:1])  # one extra batch: different pixel set
    with pytest.raises(ValueError, match="same pixels"):
        score_against_control(pred, ctl, oracle, FR)


def test_the_table_puts_the_control_under_every_predictor():
    score = score_against_control(*_scored(), FR)
    lines = controlled_ause_markdown([
        {"display": "head sigma", "control_display": "SR gradient", "score": score},
        {"display": "TTA std", "control_display": "SR gradient of the mean", "score": score},
    ])
    body = lines[2:]
    assert len(body) == 4
    assert body[0].startswith("| head sigma |") and body[1].startswith("| ↳ control: SR gradient |")
    assert body[2].startswith("| TTA std |") and body[3].startswith("| ↳ control:")
    assert f"{score['delta_ause']:+.4f}" in body[0]


def test_a_predictor_without_its_control_cannot_be_rendered():
    score = score_against_control(*_scored(), FR)
    del score["control"]
    with pytest.raises(KeyError):
        controlled_ause_markdown([{"display": "head", "control_display": "x", "score": score}])


@pytest.mark.parametrize("u, err", [
    (-torch.ones(1, 2, 2, 2), torch.ones(1, 2, 2, 2)),
    (torch.ones(1, 2, 2, 2), torch.full((1, 2, 2, 2), float("nan"))),
    (torch.ones(1, 3, 2, 2), torch.ones(1, 3, 2, 2)),
])
def test_bad_inputs_are_refused(u, err):
    with pytest.raises(ValueError):
        _hist().update(u, err)
