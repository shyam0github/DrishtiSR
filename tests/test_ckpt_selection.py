"""Pin the statistics behind the Day 3 table: pairing, CIs, Wilcoxon, selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.ckpt_selection import (
    bootstrap_mean_ci,
    compare_paired,
    paired_frame,
    select_checkpoint,
    wilcoxon_p,
)


def _frame(values, tiles=None, **extra):
    n = len(values)
    tiles = tiles if tiles is not None else [f"t{i // 4}" for i in range(n)]
    data = {
        "sample_id": tiles,
        "lr_row": [i % 4 for i in range(n)],
        "lr_col": [0] * n,
        "m": list(values),
    }
    data.update(extra)
    return pd.DataFrame(data)


def test_paired_frame_computes_treatment_minus_control():
    out = paired_frame(_frame([3.0, 5.0]), _frame([1.0, 1.0]), "m")
    assert out["delta"].tolist() == [2.0, 4.0]


def test_paired_frame_refuses_mismatched_patches():
    control = _frame([1.0, 1.0])
    control.loc[1, "lr_row"] = 99
    with pytest.raises(ValueError, match="only one of the two"):
        paired_frame(_frame([3.0, 5.0]), control, "m")


def test_paired_frame_refuses_duplicate_keys():
    treatment = _frame([3.0, 5.0])
    treatment.loc[1, "lr_row"] = 0
    with pytest.raises(ValueError, match="duplicated"):
        paired_frame(treatment, _frame([1.0, 1.0]), "m")


def test_bootstrap_ci_covers_the_mean_and_is_seeded():
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 1.0, size=2000)
    a = bootstrap_mean_ci(values, 2000, 0.95, np.random.default_rng(1))
    b = bootstrap_mean_ci(values, 2000, 0.95, np.random.default_rng(1))
    assert a == b
    assert a[0] < values.mean() < a[1]
    assert a[1] - a[0] < 0.2


def test_clustered_ci_is_wider_when_clusters_are_correlated():
    rng = np.random.default_rng(0)
    tile_effect = np.repeat(rng.normal(0.0, 1.0, size=100), 4)
    values = tile_effect + rng.normal(0.0, 0.1, size=400)
    tiles = np.repeat(np.arange(100), 4)
    pair = bootstrap_mean_ci(values, 2000, 0.95, np.random.default_rng(2))
    tile = bootstrap_mean_ci(values, 2000, 0.95, np.random.default_rng(2), clusters=tiles)
    assert (tile[1] - tile[0]) > 1.5 * (pair[1] - pair[0])


def test_bootstrap_refuses_nonfinite():
    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_mean_ci(np.array([1.0, np.nan]), 10, 0.95, np.random.default_rng(0))


def test_wilcoxon_all_zero_is_no_difference():
    assert wilcoxon_p(np.zeros(10)) == 1.0


def test_wilcoxon_detects_a_clear_shift():
    assert wilcoxon_p(np.linspace(0.5, 1.5, 50)) < 1e-6


def test_compare_paired_direction_and_verdicts():
    rng = np.random.default_rng(0)
    control = _frame(rng.normal(1.0, 0.1, size=400))
    better = control.copy()
    better["m"] = control["m"] - 0.2 + rng.normal(0, 0.01, size=400)
    worse = control.copy()
    worse["m"] = control["m"] + 0.2
    noise = control.copy()
    noise["m"] = control["m"] + rng.normal(0, 0.1, size=400)

    kwargs = dict(metric="m", better="lower", n_boot=500, ci=0.95, seed=0)
    assert compare_paired(better, control, **kwargs)["verdict"] == "treatment better"
    assert compare_paired(worse, control, **kwargs)["verdict"] == "control better"
    result = compare_paired(noise, control, **kwargs)
    assert result["verdict"] == "not resolved"
    assert result["straddles_zero_tile"]
    # Same data, direction flipped: lower deltas now favour the control.
    flipped = compare_paired(better, control, **{**kwargs, "better": "higher"})
    assert flipped["verdict"] == "control better"


def test_compare_paired_drops_and_counts_nonfinite_pairs():
    control = _frame([1.0, 1.0, 1.0, 1.0])
    treatment = _frame([0.5, np.nan, 0.5, 0.5])
    result = compare_paired(treatment, control, "m", "lower", 100, 0.95, 0)
    assert result["n"] == 3 and result["n_dropped_nonfinite"] == 1


def _candidates(consistency, tie):
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.01, size=200)
    out = {}
    for label, (c, t) in zip(("it1000", "it2000", "it3000"), zip(consistency, tie)):
        out[label] = _frame(base + c + rng.normal(0, 1e-4, size=200), lp=[t] * 200)
    return out


def test_select_picks_the_clear_minimum_even_with_worse_tie_metric():
    cands = _candidates((0.5, 0.2, 0.9), (0.1, 0.9, 0.1))
    result = select_checkpoint(cands, "m", "lp", "lower", 500, 0.95, 0)
    assert result["selected"] == "it2000"
    assert result["tied_set"] == ["it2000"]
    assert len(result["rows"]) == 3


def test_select_uses_the_tie_break_inside_the_noise():
    cands = _candidates((0.2, 0.2, 0.9), (0.4, 0.1, 0.1))
    result = select_checkpoint(cands, "m", "lp", "lower", 500, 0.95, 0)
    assert set(result["tied_set"]) == {"it1000", "it2000"}
    assert result["selected"] == "it2000"


def test_select_refuses_a_nonfinite_mean():
    cands = _candidates((0.2, 0.3, 0.9), (0.4, 0.1, 0.1))
    cands["it1000"]["m"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        select_checkpoint(cands, "m", "lp", "lower", 100, 0.95, 0)
