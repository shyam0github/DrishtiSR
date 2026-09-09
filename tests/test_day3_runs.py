"""The Day 3 orchestration: the guards, and the arithmetic they depend on.

These are the parts of ``scripts/day3_runs.py`` that must be right BEFORE the
session starts, because each of them is a decision made once, hours in, with no
opportunity to try again:

- the three runs differ in the two lambdas and nothing else -- the assumption
  the entire Day 3 claim rests on;
- three completed runs agree on their config hash and their commit, and two
  runs that both failed to record provenance are NOT treated as agreeing;
- a truncated B1 lands on a checkpoint boundary rather than throwing away the
  iterations since its last one;
- B2 is skipped on measured durations, and the margin is a real margin.

CPU only, no data, no network, no subprocess.
"""

from __future__ import annotations

import pytest

from scripts.day3_runs import (
    RUN_PLAN,
    Day3Error,
    assert_runs_differ_only_in_lambdas,
    assert_same_experiment,
    build_parser,
    build_run_args,
    build_shared_args,
    cut_iters,
    should_run_b2,
)


def _args(**overrides):
    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _planned(args=None, iters=12000):
    args = args or _args()
    shared = build_shared_args(args)
    return {
        spec["name"]: build_run_args(shared, spec, f"/runs/{spec['name']}", iters, 8.0)
        for spec in RUN_PLAN
    }


# -- the run plan itself ----------------------------------------------------


def test_the_plan_is_the_run_spec():
    """A2 control, B1 primary, B2 optional -- in that order, with those weights."""
    assert [spec["name"] for spec in RUN_PLAN] == ["a2", "b1", "b2"]
    assert (RUN_PLAN[0]["lambda1"], RUN_PLAN[0]["lambda2"]) == (0.0, 0.0)
    assert (RUN_PLAN[1]["lambda1"], RUN_PLAN[1]["lambda2"]) == (0.1, 0.02)
    assert (RUN_PLAN[2]["lambda1"], RUN_PLAN[2]["lambda2"]) == (0.3, 0.06)
    assert RUN_PLAN[0]["optional"] is False, "the control is never optional"
    assert RUN_PLAN[1]["optional"] is False
    assert RUN_PLAN[2]["optional"] is True


def test_b2_is_three_times_b1():
    """The point of B2 is to bracket the trend, so the ratio must be exact."""
    assert RUN_PLAN[2]["lambda1"] == pytest.approx(3 * RUN_PLAN[1]["lambda1"])
    assert RUN_PLAN[2]["lambda2"] == pytest.approx(3 * RUN_PLAN[1]["lambda2"])


def test_the_defaults_are_the_day3_run_spec():
    args = _args()
    assert args.iters == 12000
    assert args.ckpt_every == 1000
    assert args.val_every == 500
    assert args.wd == 0.0
    assert args.seed == 1337
    assert args.spectral_downsample == "area"
    assert args.a2_cut_hours == 3.5
    assert args.b2_margin == 2.5
    assert args.require_clean is True, (
        "a dirty tree must be refused by default: the kernel checks out a SHA "
        "and uncommitted work does not exist to it."
    )


def test_the_checkpoint_interval_is_a_multiple_of_the_validation_interval():
    """So every checkpoint carries val metrics measured AT its own iteration."""
    args = _args()
    assert args.ckpt_every % args.val_every == 0


# -- the runs are the same experiment ---------------------------------------


def test_the_planned_runs_differ_only_in_the_lambdas():
    assert_runs_differ_only_in_lambdas(_planned())


def test_every_shared_flag_is_actually_shared():
    """Stated positionally as well, so a flag moved into build_run_args is caught."""
    planned = _planned()
    per_run = {"--spectral-lambda1", "--spectral-lambda2", "--out", "--iters", "--max-hours"}
    shared = build_shared_args(_args())
    shared_flags = {shared[i] for i in range(0, len(shared), 2)}
    assert not (shared_flags & per_run)
    for values in planned.values():
        flags = {values[i] for i in range(0, len(values), 2)}
        assert shared_flags <= flags
        assert per_run <= flags


def test_a_divergent_flag_is_refused():
    planned = _planned()
    planned["b1"] = [
        "--patch-lr" if value == "--n-feats" else value for value in planned["b1"]
    ]
    with pytest.raises(Day3Error, match="not the spectral lambdas"):
        assert_runs_differ_only_in_lambdas(planned)


def test_a_differing_value_on_a_shared_flag_is_refused():
    """The likeliest real mistake: same flags, one different number."""
    planned = _planned()
    index = planned["b1"].index("--n-feats")
    planned["b1"][index + 1] = "64"
    with pytest.raises(Day3Error, match="--n-feats"):
        assert_runs_differ_only_in_lambdas(planned)


def test_a_shortened_run_is_still_allowed():
    """The wall-clock guard may cut --iters; that is a labelled result, not a bug."""
    args = _args()
    shared = build_shared_args(args)
    planned = {
        "a2": build_run_args(shared, RUN_PLAN[0], "/runs/a2", 12000, 8.0),
        "b1": build_run_args(shared, RUN_PLAN[1], "/runs/b1", 4000, 3.0),
    }
    assert_runs_differ_only_in_lambdas(planned)


def test_an_odd_length_argument_list_is_refused():
    """Silent mis-alignment of every flag after the offender, otherwise."""
    with pytest.raises(Day3Error, match="odd length"):
        assert_runs_differ_only_in_lambdas(
            {"a": ["--x", "1"], "b": ["--x", "1", "--bare"]}
        )


# -- provenance across the completed runs -----------------------------------


def _record(name, config_hash="c" * 64, git_sha="a" * 40):
    return {"name": name, "config_hash": config_hash, "git_sha": git_sha}


def test_unanimous_provenance_passes():
    assert_same_experiment([_record("a2"), _record("b1"), _record("b2")])


def test_a_config_hash_disagreement_is_refused():
    with pytest.raises(Day3Error, match="disagree on config_hash"):
        assert_same_experiment([_record("a2"), _record("b1", config_hash="d" * 64)])


def test_a_commit_disagreement_is_refused():
    """config_hash alone was never enough -- the frozen-crop bug proved it."""
    with pytest.raises(Day3Error, match="disagree on git_sha"):
        assert_same_experiment([_record("a2"), _record("b1", git_sha="b" * 40)])


def test_two_unknowns_do_not_count_as_a_match():
    """The sentinel case. "We could not tell" is not "they were the same"."""
    with pytest.raises(Day3Error, match="no provenance"):
        assert_same_experiment(
            [_record("a2", config_hash="unavailable"),
             _record("b1", config_hash="unavailable")]
        )
    with pytest.raises(Day3Error, match="no provenance"):
        assert_same_experiment(
            [_record("a2", git_sha="unavailable"),
             _record("b1", git_sha="unavailable")]
        )


def test_no_completed_runs_is_an_error_not_a_pass():
    with pytest.raises(Day3Error, match="nothing to compare"):
        assert_same_experiment([])


# -- the wall clock ---------------------------------------------------------


def test_cut_iters_scales_by_the_measured_rate():
    """4 h bought 12000 iterations; 2 h buys 6000."""
    assert cut_iters(12000, hours_used=4.0, hours_left=2.0, ckpt_every=1000) == 6000


def test_cut_iters_rounds_down_to_a_checkpoint_boundary():
    """Iterations past the last checkpoint are thrown away, so do not run them."""
    cut = cut_iters(12000, hours_used=4.0, hours_left=2.3, ckpt_every=1000)
    assert cut % 1000 == 0
    assert cut == 6000


def test_cut_iters_never_returns_zero():
    """A zero-iteration run writes an untrained checkpoint that looks like a result."""
    assert cut_iters(12000, hours_used=8.0, hours_left=0.0, ckpt_every=1000) == 1000
    assert cut_iters(12000, hours_used=8.0, hours_left=-1.0, ckpt_every=1000) == 1000


def test_cut_iters_never_lengthens_a_run():
    assert cut_iters(12000, hours_used=1.0, hours_left=99.0, ckpt_every=1000) == 12000


def test_cut_iters_needs_a_rate_to_scale_by():
    with pytest.raises(ValueError, match="positive"):
        cut_iters(12000, hours_used=0.0, hours_left=4.0, ckpt_every=1000)


def test_b2_runs_only_with_the_full_margin():
    """2.5 x the median, and 2.49 x is not enough -- the margin is the guard."""
    durations = [2.0, 2.0]
    assert should_run_b2(durations, hours_left=5.0, margin=2.5)
    assert not should_run_b2(durations, hours_left=4.98, margin=2.5)


def test_b2_uses_the_median_not_the_last_run():
    """One slow run must not license a third; one fast one must not veto it."""
    assert should_run_b2([1.0, 3.0], hours_left=5.0, margin=2.5)  # median 2.0
    assert not should_run_b2([1.0, 9.0], hours_left=5.0, margin=2.5)  # median 5.0


def test_b2_is_refused_when_nothing_has_been_measured():
    """No evidence is not permission."""
    assert not should_run_b2([], hours_left=99.0, margin=2.5)
