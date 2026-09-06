"""Tests for the LR/HR degradation audit.

The audit's job is to answer one question about the dataset -- was the LR made
from the HR? -- so these tests are mostly about it giving the RIGHT answer on
pairs whose provenance is known by construction, in both directions. A detector
that never says SYNTHETIC and a detector that always says SYNTHETIC are equally
useless, and only testing both catches that.

CPU-only, no data required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.eval.degradation import (
    DEGRADATION_KERNELS,
    block_valid_mask,
    degrade,
    marginal_agreement,
    quantisation_ceiling_db,
    score_pair,
    summarise,
    verdict,
)

SCALE = 4
BANDS = 4
LR_SIZE = 32
HR_SIZE = LR_SIZE * SCALE
DATA_RANGE = 1.0
CEILING = quantisation_ceiling_db(10000.0, DATA_RANGE)


def make_hr(seed: int = 0) -> np.ndarray:
    """Fabricate an HR tile with plausible reflectance structure.

    Returns:
        ``(4, 128, 128)`` float32 surface reflectance, nominally ``[0, 1]``,
        deliberately UNCLIPPED so a bright pixel can exceed 1.0 exactly as real
        data does.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.random((BANDS, LR_SIZE // 2, LR_SIZE // 2)).astype(np.float32)
    smooth = F.interpolate(
        torch.from_numpy(coarse)[None],
        size=(HR_SIZE, HR_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0].numpy()
    texture = rng.normal(0.0, 0.05, size=(BANDS, HR_SIZE, HR_SIZE)).astype(np.float32)
    return (0.05 + 0.4 * smooth + texture).astype(np.float32)


def all_valid(shape) -> np.ndarray:
    return np.ones(shape, dtype=bool)


# -- degrade ---------------------------------------------------------------


@pytest.mark.parametrize("kernel", DEGRADATION_KERNELS)
def test_every_kernel_produces_the_lr_grid(kernel):
    out = degrade(make_hr(), kernel, SCALE)
    assert out.shape == (BANDS, LR_SIZE, LR_SIZE)
    assert out.dtype == np.float32


def test_box_and_area_agree_at_integer_scale():
    """Two independent implementations of the same block mean must agree."""
    hr = make_hr()
    np.testing.assert_allclose(
        degrade(hr, "box", SCALE), degrade(hr, "area", SCALE), rtol=0, atol=1e-6
    )


def test_nearest_offset_selects_different_pixels():
    hr = make_hr()
    first = degrade(hr, "nearest", SCALE, offset=(0, 0))
    second = degrade(hr, "nearest", SCALE, offset=(1, 1))
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first[0, 0, 0], hr[0, 0, 0])
    np.testing.assert_array_equal(second[0, 0, 0], hr[0, 1, 1])


def test_unknown_kernel_raises_rather_than_falling_back():
    with pytest.raises(ValueError, match="Unknown degradation kernel"):
        degrade(make_hr(), "lanczos", SCALE)


def test_wrong_axis_order_raises():
    with pytest.raises(ValueError, match=r"must be \(C, H, W\)"):
        degrade(np.zeros((HR_SIZE, HR_SIZE), dtype=np.float32), "box", SCALE)


def test_indivisible_shape_raises():
    with pytest.raises(ValueError, match="not divisible by scale"):
        degrade(np.zeros((BANDS, 130, 130), dtype=np.float32), "box", SCALE)


def test_offset_outside_scale_raises():
    with pytest.raises(ValueError, match="outside"):
        degrade(make_hr(), "nearest", SCALE, offset=(SCALE, 0))


def test_bright_targets_are_not_clipped():
    """Reflectance above 1.0 is legal and must survive every kernel."""
    hr = make_hr()
    hr[:, :8, :8] = 1.6
    for kernel in DEGRADATION_KERNELS:
        assert degrade(hr, kernel, SCALE).max() > 1.0, kernel


# -- valid mask ------------------------------------------------------------


def test_one_bad_hr_pixel_invalidates_its_whole_lr_block():
    lr_nodata = np.zeros((BANDS, LR_SIZE, LR_SIZE), dtype=bool)
    hr_nodata = np.zeros((BANDS, HR_SIZE, HR_SIZE), dtype=bool)
    hr_nodata[0, 5, 7] = True  # inside LR block (1, 1)
    valid = block_valid_mask(lr_nodata, hr_nodata, SCALE)
    assert not valid[0, 1, 1]
    assert valid.sum() == valid.size - 1


def test_mismatched_mask_shapes_raise():
    with pytest.raises(ValueError, match="does not match"):
        block_valid_mask(
            np.zeros((BANDS, LR_SIZE, LR_SIZE), dtype=bool),
            np.zeros((BANDS, HR_SIZE + 4, HR_SIZE), dtype=bool),
            SCALE,
        )


# -- scoring ---------------------------------------------------------------


def test_a_known_degradation_scores_at_the_numerical_ceiling():
    """The detector must recognise its own kernel's output. Positive control."""
    hr = make_hr()
    lr = degrade(hr, "box", SCALE)
    scored = score_pair(
        lr,
        hr,
        all_valid(lr.shape),
        scale=SCALE,
        kernels=["box", "bicubic_antialias", "nearest"],
        data_range=DATA_RANGE,
    )
    assert scored["psnr_db"]["box"] > CEILING
    assert scored["psnr_db"]["box"] > scored["psnr_db"]["bicubic_antialias"]


def test_independent_images_score_far_below_the_ceiling():
    """Negative control: two unrelated scenes must not look like a degradation."""
    lr = degrade(make_hr(seed=1), "box", SCALE)
    hr = make_hr(seed=2)
    scored = score_pair(
        lr, hr, all_valid(lr.shape), scale=SCALE, kernels=["box"], data_range=DATA_RANGE
    )
    assert scored["psnr_db"]["box"] < 0.5 * CEILING


def test_nodata_pixels_are_excluded_not_averaged_in():
    """A masked pixel must not contribute, however wrong its fill value is."""
    hr = make_hr()
    lr = degrade(hr, "box", SCALE)
    valid = all_valid(lr.shape)
    valid[:, 0, 0] = False
    corrupted = lr.copy()
    corrupted[:, 0, 0] = 99.0  # a fill value that would destroy any mean

    clean = score_pair(
        lr, hr, valid, scale=SCALE, kernels=["box"], data_range=DATA_RANGE
    )
    dirty = score_pair(
        corrupted, hr, valid, scale=SCALE, kernels=["box"], data_range=DATA_RANGE
    )
    assert clean["psnr_db"]["box"] == pytest.approx(dirty["psnr_db"]["box"])


def test_a_fully_masked_pair_raises_rather_than_scoring_zero():
    hr = make_hr()
    lr = degrade(hr, "box", SCALE)
    with pytest.raises(ValueError, match="No valid pixels"):
        score_pair(
            lr,
            hr,
            np.zeros(lr.shape, dtype=bool),
            scale=SCALE,
            kernels=["box"],
            data_range=DATA_RANGE,
        )


def test_nearest_offset_search_finds_the_planted_phase():
    hr = make_hr()
    lr = degrade(hr, "nearest", SCALE, offset=(2, 3))
    scored = score_pair(
        lr,
        hr,
        all_valid(lr.shape),
        scale=SCALE,
        kernels=["nearest"],
        data_range=DATA_RANGE,
        search_nearest_offsets=True,
    )
    assert scored["nearest_offset"] == (2, 3)
    assert not np.isfinite(scored["psnr_db"]["nearest"])  # exact, so infinite


# -- marginal agreement ----------------------------------------------------


def test_averaging_reduces_variance_and_the_test_sees_it():
    """The load-bearing physical claim behind the whole audit."""
    hr = make_hr()
    lr = degrade(hr, "box", SCALE)
    result = marginal_agreement(
        lr, hr, all_valid(lr.shape), all_valid(hr.shape), [0, 50, 100]
    )
    assert all(ratio < 0.99 for ratio in result["std_ratio"]), result["std_ratio"]


def test_histogram_matched_pair_preserves_variance():
    """The competing hypothesis, built explicitly so the test can tell them apart.

    An HR whose values are drawn from the LR's own distribution has, by
    construction, the LR's variance and extrema -- while its spatial content is
    unrelated. This is the shape of a radiometrically harmonised pair.
    """
    rng = np.random.default_rng(3)
    lr = make_hr(seed=4)[:, :LR_SIZE, :LR_SIZE]
    # Resample HR pixels from the LR's exact value set: same marginal, different
    # arrangement.
    flat = lr.reshape(BANDS, -1)
    hr = np.stack(
        [rng.choice(flat[b], size=(HR_SIZE, HR_SIZE)) for b in range(BANDS)]
    ).astype(np.float32)

    result = marginal_agreement(
        lr, hr, all_valid(lr.shape), all_valid(hr.shape), [0, 50, 100]
    )
    for ratio in result["std_ratio"]:
        assert ratio == pytest.approx(1.0, abs=0.05), result["std_ratio"]


# -- summarise -------------------------------------------------------------


def test_summarise_reports_the_distribution_not_just_the_mean():
    records = [
        {"psnr_db": {"box": value}} for value in [30.0, 35.0, 40.0, 45.0, 50.0]
    ]
    summary = summarise(records, ["box"], [0, 50, 100])
    assert summary["box"]["p0"] == pytest.approx(30.0)
    assert summary["box"]["p50"] == pytest.approx(40.0)
    assert summary["box"]["p100"] == pytest.approx(50.0)
    assert summary["best_kernel"] == "box"


def test_summarise_counts_exact_reproductions_rather_than_dropping_them():
    records = [{"psnr_db": {"box": float("inf")}}, {"psnr_db": {"box": 40.0}}]
    summary = summarise(records, ["box"], [50])
    assert summary["box"]["num_infinite"] == 1
    assert summary["box"]["num_scored"] == 2


def test_a_kernel_that_is_always_exact_wins_despite_a_nan_median():
    records = [{"psnr_db": {"box": float("inf"), "nearest": 40.0}} for _ in range(3)]
    summary = summarise(records, ["box", "nearest"], [50])
    assert summary["best_kernel"] == "box"


def test_summarising_nothing_raises_rather_than_reading_as_clean():
    with pytest.raises(ValueError, match="nothing to summarise"):
        summarise([], ["box"], [50])


# -- verdict ---------------------------------------------------------------


def _verdict(summary, ratio, control=20.0):
    return verdict(
        summary,
        {"median_std_ratio": ratio},
        control,
        synthetic_psnr_db=45.0,
        crosssensor_psnr_db=35.0,
        variance_ratio_tol=0.02,
        ceiling_db=CEILING,
        deterministic_spread_db=2.0,
        quantisation_ceiling_fraction=0.75,
    )


def _summary(kernel, median, spread=0.1, infinite=0, scored=200):
    return {
        kernel: {
            "p5": median - spread / 2,
            "p50": median,
            "p95": median + spread / 2,
            "num_infinite": infinite,
            "num_scored": scored,
        },
        "best_kernel": kernel,
    }


def test_a_real_degradation_is_called_synthetic():
    result = _verdict(_summary("box", 88.0), ratio=0.95)
    assert result["verdict"] == "SYNTHETIC"


def test_high_psnr_alone_does_not_prove_a_degradation():
    """The case that prompted the three-test structure.

    45.38 dB is over the configured threshold, but it is 45 dB below the uint16
    quantisation ceiling, the per-tile spread is far too wide for one fixed
    kernel, and the best-fitting kernel is an averaging one on a pair that lost
    no variance. The threshold alone would call this SYNTHETIC; the physical
    tests must overrule it.
    """
    result = _verdict(_summary("bilinear_antialias", 45.38, spread=13.15), ratio=1.0004)
    assert result["verdict"] == "CROSS_SENSOR"
    assert result["evidence"]["tests_disagree"]
    assert not result["evidence"]["physical_tests_say_synthetic"]
    assert any("TESTS DISAGREE" in reason for reason in result["reasons"])


def test_the_threshold_is_still_reported_when_it_is_exceeded():
    result = _verdict(_summary("bilinear_antialias", 45.38, spread=13.15), ratio=1.0004)
    assert result["evidence"]["psnr_threshold_says_synthetic"]


def test_a_genuinely_independent_pair_is_called_cross_sensor():
    result = _verdict(_summary("bicubic_antialias", 24.0, spread=8.0), ratio=1.0)
    assert result["verdict"] == "CROSS_SENSOR"


def test_physical_evidence_without_the_threshold_is_unresolved():
    """Deterministic and at the ceiling, but under the dB line: say so."""
    result = _verdict(_summary("nearest", 40.0, spread=0.05), ratio=0.9)
    assert result["verdict"] == "UNRESOLVED"


def test_exact_reproduction_on_every_sample_is_synthetic():
    result = _verdict(
        _summary("box", float("nan"), infinite=200, scored=200), ratio=0.95
    )
    assert result["verdict"] == "SYNTHETIC"
    assert not np.isfinite(result["best_median_db"])


# -- the quantisation ceiling ---------------------------------------------


def test_quantisation_ceiling_matches_the_documented_value():
    assert quantisation_ceiling_db(10000.0, 1.0) == pytest.approx(90.8, abs=0.1)


def test_quantisation_ceiling_rejects_nonsense_inputs():
    with pytest.raises(ValueError, match="positive and finite"):
        quantisation_ceiling_db(0.0, 1.0)
    with pytest.raises(ValueError, match="positive and finite"):
        quantisation_ceiling_db(10000.0, float("nan"))
