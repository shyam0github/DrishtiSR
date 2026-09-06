"""Tests for the opensr-test adapter.

These pin the **input contract** rather than the metric values. The metrics
belong to a third party and their numbers are theirs to change; what must not
change silently is any of the assumptions this project makes when handing them
data. Each assumption below was measured against opensr_test 1.3.3 (see the
module docstring of ``src/eval/opensr_harness.py``) and each has a test here, so
that an upgrade which moves the contract fails in pytest rather than in a report.

The most important test in this file is
:func:`test_digital_numbers_are_refused`. Feeding opensr-test digital numbers
instead of reflectance does NOT raise upstream -- it inflates ``reflectance`` and
``synthesis`` by the scale factor and leaves the other five untouched, so the
result looks entirely normal. That tripwire is the only thing standing between a
divisor mistake and a number in the finals.

Tests that need the library itself are skipped when it is absent, so the suite
still runs on a machine that has not installed it. The adapter tests do not need
it and always run.
"""

import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.eval.baselines import bicubic_upsample
from src.eval.opensr_harness import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    OPENSR_METRICS,
    OpenSRResult,
    OpenSRSampleError,
    build_metrics,
    run_opensr_test,
    score_triplet,
    to_opensr_triplet,
)

opensr_test = pytest.importorskip(
    "opensr_test", reason="opensr-test is optional; the adapter tests run without it."
)

SCALE = 4
BANDS = ["B04", "B03", "B02", "B08"]
# LR 64 -> HR 256 is the real patch geometry (cfg.patches.lr_size * cfg.sr.scale).
# border_mask=16 leaves LR 56 and HR 224, which stay commensurate at x4.
LR_SIZE = 64
HR_SIZE = LR_SIZE * SCALE


def make_cfg(**overrides):
    """A minimal config carrying exactly the keys the harness reads."""
    cfg = OmegaConf.create(
        {
            "sr": {"scale": SCALE},
            "dataset": {"bands": list(BANDS), "reflectance_scale": 10000.0},
            "metrics": {"lpips": {"rgb_bands": ["B04", "B03", "B02"]}},
            "opensr_test": {
                "enabled": True,
                "device": "cpu",
                "agg_method": "pixel",
                "patch_size": None,
                "border_mask": 16,
                "rgb_bands": ["B04", "B03", "B02"],
                "harm_apply_spectral": True,
                "harm_apply_spatial": True,
                "spatial_method": "pcc",
                "spatial_threshold_distance": 5,
                "spatial_max_num_keypoints": 500,
                "reflectance_distance": "l1",
                "spectral_distance": "sad",
                "synthesis_distance": "l1",
                "correctness_distance": "nd",
                "correctness_norm": "softmin",
                "im_score": 0.05,
                "om_score": 0.05,
                "ha_score": 0.05,
                "correctness_temperature": 0.25,
                "gradient_threshold": "auto",
                "n_samples": 4,
                "max_reflectance": 10.0,
                "max_skipped_fraction": 0.10,
                "json_name": "opensr_{method}.json",
                "csv_name": "opensr_{method}.csv",
            },
        }
    )
    for dotted, value in overrides.items():
        OmegaConf.update(cfg, dotted, value, merge=True)
    return cfg


def scene(seed=0, bands=4, size=HR_SIZE, scale_to=1.0):
    """A spatially correlated reflectance scene, (C, H, W) float32 in [0, 1).

    White noise is not a fair stand-in: every opensr-test metric is about
    high-frequency structure, and noise has no structure to find. This upsamples
    a coarse random field so the result has land-cover-like spatial correlation.
    """
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(bands, max(size // 16, 2), max(size // 16, 2), generator=generator)
    fine = torch.nn.functional.interpolate(
        coarse[None], size=(size, size), mode="bicubic", align_corners=False
    )[0]
    return (fine.clamp(0.01, 0.99) * scale_to).to(torch.float32)


def triplet(seed=0, bands=4):
    """A realistic ``(lr, sr, hr)``: sr is the bicubic upsample of lr."""
    hr = scene(seed=seed, bands=bands, size=HR_SIZE)
    lr = torch.nn.functional.interpolate(
        hr[None], scale_factor=1 / SCALE, mode="bicubic", antialias=True, align_corners=False
    )[0].to(torch.float32)
    sr = bicubic_upsample(lr, SCALE)
    return lr, sr, hr


# -- the adapter's assertions ---------------------------------------------


def test_valid_triplet_passes_through_unchanged():
    """The adapter converts containers; it must not touch a single value."""
    cfg = make_cfg()
    lr, sr, hr = triplet()
    lr_o, sr_o, hr_o = to_opensr_triplet(lr, sr, hr, cfg)

    for original, converted in ((lr, lr_o), (sr, sr_o), (hr, hr_o)):
        assert converted.dtype is torch.float32
        assert converted.device.type == "cpu"
        assert not converted.requires_grad
        # Bit-identical: no scaling, no normalisation, no clipping.
        assert torch.equal(original.to(torch.float32), converted)


def test_numpy_is_converted_not_rejected():
    """opensr-test takes only torch, so the adapter is where numpy is handled."""
    cfg = make_cfg()
    lr, sr, hr = triplet()
    lr_o, sr_o, hr_o = to_opensr_triplet(
        lr.numpy(), sr.numpy(), hr.numpy(), cfg
    )
    assert torch.is_tensor(lr_o) and torch.is_tensor(sr_o) and torch.is_tensor(hr_o)
    assert torch.equal(lr_o, lr)


def test_gradients_are_detached():
    """An sr_fn wrapping a model produces a graph; opensr-test raises on one."""
    cfg = make_cfg()
    lr, sr, hr = triplet()
    _, sr_o, _ = to_opensr_triplet(lr, sr.clone().requires_grad_(True), hr, cfg)
    assert not sr_o.requires_grad


@pytest.mark.parametrize(
    "value, expect",
    [(1.4, "passes"), (1.9, "passes")],
)
def test_reflectance_above_one_is_never_clipped(value, expect):
    """Cloud, snow and bright roofs exceed 1.0 legitimately. Do not clip them.

    This is the project's no-silent-clipping rule expressed as a test: the
    tripwire below must catch a factor-of-10000 divisor error WITHOUT touching
    the physically real bright tail.
    """
    cfg = make_cfg()
    lr, sr, hr = triplet()
    hr = hr.clone()
    hr[0, 100, 100] = value
    _, _, hr_o = to_opensr_triplet(lr, sr, hr, cfg)
    assert float(hr_o[0, 100, 100]) == pytest.approx(value)
    assert float(hr_o.max()) > 1.0


def test_negative_reflectance_from_bicubic_overshoot_is_kept():
    """Bicubic rings below zero at sharp edges. That is real output, not an error."""
    cfg = make_cfg()
    lr, sr, hr = triplet()
    sr = sr.clone()
    sr[0, 120, 120] = -0.05
    _, sr_o, _ = to_opensr_triplet(lr, sr, hr, cfg)
    assert float(sr_o[0, 120, 120]) == pytest.approx(-0.05)


def test_digital_numbers_are_refused():
    """THE important one. A wrong divisor is silent upstream; it must not be here.

    MEASURED on opensr_test 1.3.3: the same triplet passed as digital numbers
    returns reflectance=2528.2 / synthesis=2894.1 against the correct 0.0018 /
    0.074, while spectral, ha, om and im are bit-identical because they are
    scale-invariant. Four of seven numbers look completely normal.
    """
    cfg = make_cfg()
    lr, sr, hr = triplet()
    scale = float(cfg.dataset.reflectance_scale)
    with pytest.raises(OpenSRSampleError, match="max_reflectance"):
        to_opensr_triplet(lr * scale, sr * scale, hr * scale, cfg)


def test_tripwire_message_names_the_cause():
    """The error must say 'digital number', not just 'value too large'."""
    cfg = make_cfg()
    lr, sr, hr = triplet()
    with pytest.raises(OpenSRSampleError) as excinfo:
        to_opensr_triplet(lr * 10000, sr * 10000, hr * 10000, cfg)
    message = str(excinfo.value)
    assert "SURFACE REFLECTANCE" in message
    assert "reflectance_scale" in message
    assert "not a clip" in message


def test_batched_input_is_refused():
    """opensr-test has no batch path; a (B, C, H, W) input raises deep inside it."""
    cfg = make_cfg()
    lr, sr, hr = triplet()
    with pytest.raises(OpenSRSampleError, match="batch axis"):
        to_opensr_triplet(
            lr[None].repeat(2, 1, 1, 1),
            sr[None].repeat(2, 1, 1, 1),
            hr[None].repeat(2, 1, 1, 1),
            cfg,
        )


def test_batch_of_one_is_unwrapped():
    """A leading axis of 1 is a convenience, not an error."""
    cfg = make_cfg()
    lr, sr, hr = triplet()
    lr_o, _, _ = to_opensr_triplet(lr[None], sr[None], hr[None], cfg)
    assert lr_o.shape == lr.shape


def test_non_finite_input_is_refused():
    """A NaN does not raise upstream; it silently NaNs the spatial metric."""
    cfg = make_cfg()
    lr, sr, hr = triplet()
    sr = sr.clone()
    sr[0, 128, 128] = float("nan")
    with pytest.raises(OpenSRSampleError, match="non-finite"):
        to_opensr_triplet(lr, sr, hr, cfg)


def test_wrong_channel_count_is_refused():
    cfg = make_cfg()
    lr, sr, hr = triplet(bands=3)
    with pytest.raises(OpenSRSampleError, match="channels"):
        to_opensr_triplet(lr, sr, hr, cfg)


def test_single_channel_is_refused_with_the_measured_reason():
    """C == 1 breaks on a .squeeze() inside apply_upsampling, not on validation."""
    cfg = make_cfg(**{"dataset.bands": ["B04"], "metrics.lpips.rgb_bands": ["B04"]})
    lr, sr, hr = triplet(bands=1)
    with pytest.raises(OpenSRSampleError, match="at least 3 channels|squeeze"):
        to_opensr_triplet(lr, sr, hr, cfg)


def test_scale_mismatch_is_refused():
    cfg = make_cfg()
    lr, sr, hr = triplet()
    lr_small = torch.nn.functional.interpolate(
        lr[None], scale_factor=0.5, mode="bicubic", align_corners=False
    )[0]
    with pytest.raises(OpenSRSampleError, match="cfg.sr.scale"):
        to_opensr_triplet(lr_small, sr, hr, cfg)


def test_sr_hr_shape_mismatch_is_refused():
    cfg = make_cfg()
    lr, sr, hr = triplet()
    with pytest.raises(OpenSRSampleError, match="must match"):
        to_opensr_triplet(lr, sr[:, :128, :128], hr, cfg)


def test_patch_too_small_for_border_mask_is_refused():
    """border_mask=16 on a 32 px HR patch leaves 0x0 and raises inside interpolate."""
    cfg = make_cfg()
    hr = scene(size=32)
    lr = torch.nn.functional.interpolate(
        hr[None], scale_factor=1 / SCALE, mode="bicubic", antialias=True, align_corners=False
    )[0]
    sr = bicubic_upsample(lr, SCALE)
    with pytest.raises(OpenSRSampleError, match="border_mask"):
        to_opensr_triplet(lr, sr, hr, cfg)


def test_non_square_refused_only_under_patch_aggregation():
    """do_square() rejects non-square input; pixel aggregation does not care."""
    cfg_pixel = make_cfg()
    cfg_patch = make_cfg(**{"opensr_test.agg_method": "patch", "opensr_test.patch_size": 16})
    hr = scene(size=HR_SIZE)[:, :, :192]
    lr = torch.nn.functional.interpolate(
        hr[None], scale_factor=1 / SCALE, mode="bicubic", antialias=True, align_corners=False
    )[0]
    sr = bicubic_upsample(lr, SCALE)

    to_opensr_triplet(lr, sr, hr, cfg_pixel)  # must not raise
    with pytest.raises(OpenSRSampleError, match="square"):
        to_opensr_triplet(lr, sr, hr, cfg_patch)


# -- configuration, and the upstream README's trap -------------------------


def test_build_metrics_adopts_the_config():
    cfg = make_cfg(
        **{
            "opensr_test.border_mask": 8,
            "opensr_test.correctness_distance": "l1",
            "opensr_test.agg_method": "image",
        }
    )
    metrics, settings = build_metrics(cfg)
    assert metrics.params.border_mask == 8
    assert metrics.params.correctness_distance == "l1"
    assert metrics.params.agg_method == "image"
    assert settings["opensr_test_version"] == opensr_test.__version__


def test_rgb_bands_are_resolved_by_name_not_assumed():
    """[0,1,2] is right only for the current band order. Resolve, never assume."""
    cfg = make_cfg(
        **{
            "dataset.bands": ["B02", "B03", "B04", "B08"],  # BGR-NIR, reversed
            "metrics.lpips.rgb_bands": ["B04", "B03", "B02"],
        }
    )
    _, settings = build_metrics(cfg)
    # Red is now channel 2, blue channel 0 -- the naive [0,1,2] would be wrong.
    assert settings["rgb_bands"] == [2, 1, 0]


def test_upstream_readme_config_kwarg_would_be_silently_ignored():
    """Documents the trap build_metrics exists to avoid.

    The published example is ``Metrics(config=config)``, but the parameter is
    named ``params``. ``Config`` is a pydantic model that ignores extra fields,
    so the documented call runs on DEFAULTS while appearing to be configured.
    If this test ever fails, upstream fixed it and the guard in build_metrics can
    be relaxed -- until then it must stay.
    """
    params = opensr_test.Config(border_mask=8, correctness_distance="l1")
    ignored = opensr_test.Metrics(config=params)
    honoured = opensr_test.Metrics(params=params)

    assert ignored.params.border_mask == 16, "upstream may have fixed the kwarg name"
    assert ignored.params.correctness_distance == "nd"
    assert honoured.params.border_mask == 8
    assert honoured.params.correctness_distance == "l1"


def test_metric_directions_cover_every_metric_exactly_once():
    assert set(LOWER_IS_BETTER) | set(HIGHER_IS_BETTER) == set(OPENSR_METRICS)
    assert not set(LOWER_IS_BETTER) & set(HIGHER_IS_BETTER)


# -- the library's own contract, pinned ------------------------------------


def test_returned_keys_are_the_documented_ones():
    """Guards against the README's stale 'ha_percent' names coming back."""
    cfg = make_cfg()
    metrics, _ = build_metrics(cfg)
    lr, sr, hr = triplet()
    values = score_triplet(metrics, *to_opensr_triplet(lr, sr, hr, cfg))
    assert set(values) == set(OPENSR_METRICS)


def test_correctness_triple_sums_to_one():
    """im + om + ha is a softmin partition. Free invariant, so it is checked."""
    cfg = make_cfg()
    metrics, _ = build_metrics(cfg)
    lr, sr, hr = triplet(seed=3)
    values = score_triplet(metrics, *to_opensr_triplet(lr, sr, hr, cfg))
    total = values["im_metric"] + values["om_metric"] + values["ha_metric"]
    assert total == pytest.approx(1.0, abs=1e-4)


def test_digital_numbers_would_inflate_only_the_absolute_metrics():
    """Proves the failure mode the tripwire guards against is real and silent.

    Bypasses the adapter deliberately and calls the library directly with digital
    numbers. If this test starts failing because upstream began rejecting them,
    the tripwire is redundant and the caveat in reports/day1_gate.md can be
    dropped -- but not before.
    """
    cfg = make_cfg()
    metrics, _ = build_metrics(cfg)
    lr, sr, hr = triplet(seed=5)
    ref = score_triplet(metrics, lr, sr, hr)
    dn = score_triplet(metrics, lr * 10000, sr * 10000, hr * 10000)

    # Absolute L1 distances move with the scale...
    assert dn["reflectance"] == pytest.approx(ref["reflectance"] * 10000, rel=0.05)
    assert dn["synthesis"] == pytest.approx(ref["synthesis"] * 10000, rel=0.05)
    # ...and the scale-invariant ones do not move at all, which is what makes the
    # mistake look normal.
    assert dn["spectral"] == pytest.approx(ref["spectral"], rel=1e-3)
    for key in ("ha_metric", "om_metric", "im_metric"):
        assert dn[key] == pytest.approx(ref[key], abs=1e-3)


def test_bicubic_hallucinates_less_than_nearest():
    """Direction check: pixel replication must hallucinate MORE than bicubic.

    This is the one ordering between the two baselines that is physically
    motivated. Nearest fabricates hard block edges that exist nowhere in the
    scene, and a hallucination is by definition high-gradient detail in SR that
    is absent from HR; bicubic smooths instead, so it invents less.

    MEASURED over the full 1199-patch validation split (see
    reports/day1_gate.md): bicubic ha_metric 0.0808 vs nearest 0.1692, with
    bicubic lower on **96.3%** of patches individually.

    DO NOT add the corresponding assertion for im_metric. It was in this file
    and it was WRONG. It passed on the synthetic scenes below while the real
    data contradicts it: nearest posts a HIGHER im_metric (0.0900 vs 0.0572) on
    94.7% of real patches, because its block edges partly coincide with genuine
    HR edges and score as improvement. Bicubic and nearest do not order cleanly
    on correctness -- they trade omission against hallucination, which is the
    whole reason the three metrics are reported together and never collapsed
    into one.
    """
    cfg = make_cfg()
    metrics, _ = build_metrics(cfg)
    lr, sr_bicubic, hr = triplet(seed=7)
    sr_nearest = lr.repeat_interleave(SCALE, dim=-2).repeat_interleave(SCALE, dim=-1)

    bic = score_triplet(metrics, *to_opensr_triplet(lr, sr_bicubic, hr, cfg))
    near = score_triplet(metrics, *to_opensr_triplet(lr, sr_nearest, hr, cfg))
    assert bic["ha_metric"] < near["ha_metric"]


# -- the run, and its denominator -----------------------------------------


class _Loader:
    """A minimal stand-in for the validation dataloader."""

    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _batch(n, seed=0, corrupt=None):
    lrs, hrs = [], []
    for i in range(n):
        lr, _, hr = triplet(seed=seed + i)
        lrs.append(lr)
        hrs.append(hr)
    lr_batch = torch.stack(lrs)
    hr_batch = torch.stack(hrs)
    if corrupt is not None:
        lr_batch[corrupt] *= 10000.0  # digital numbers: must be skipped
    return {
        "lr": lr_batch,
        "hr": hr_batch,
        "sample_id": [f"sample_{seed + i}" for i in range(n)],
        "dataset_index": torch.arange(n),
        "split": ["val"] * n,
    }


def test_run_scores_every_sample():
    cfg = make_cfg()
    result = run_opensr_test(
        _Loader([_batch(2, seed=0), _batch(2, seed=10)]),
        lambda lr: bicubic_upsample(lr, SCALE),
        4,
        cfg,
    )
    assert result.num_attempted == 4
    assert result.num_scored == 4
    assert result.num_skipped == 0
    frame = result.to_frame()
    assert len(frame) == 4
    for metric in OPENSR_METRICS:
        assert metric in frame.columns
    assert set(frame["sample_id"]) == {"sample_0", "sample_1", "sample_10", "sample_11"}


def test_rejected_samples_stay_in_the_denominator():
    """A skip must never quietly shrink n. That is the whole point of the list."""
    cfg = make_cfg(**{"opensr_test.max_skipped_fraction": 0.5})
    result = run_opensr_test(
        _Loader([_batch(4, seed=0, corrupt=1)]),
        lambda lr: bicubic_upsample(lr, SCALE),
        4,
        cfg,
    )
    assert result.num_attempted == 4
    assert result.num_scored == 3
    assert result.num_skipped == 1
    assert result.skipped_fraction == pytest.approx(0.25)

    entry = result.skipped[0]
    assert entry["sample_id"] == "sample_1"
    assert entry["stage"] == "adapter"
    assert "max_reflectance" in entry["reason"]
    # Every attempted sample is in exactly one of the two lists.
    ids = {r["sample_id"] for r in result.rows} | {s["sample_id"] for s in result.skipped}
    assert len(ids) == result.num_attempted


def test_too_many_skips_fails_the_run():
    """Refusing to report beats reporting a number from an unknown subset."""
    cfg = make_cfg(**{"opensr_test.max_skipped_fraction": 0.10})
    with pytest.raises(RuntimeError, match="max_skipped_fraction"):
        run_opensr_test(
            _Loader([_batch(4, seed=0, corrupt=1)]),
            lambda lr: bicubic_upsample(lr, SCALE),
            4,
            cfg,
        )


def test_n_samples_counts_attempts_not_successes():
    """Otherwise a run of rejects silently extends the pass over more data."""
    cfg = make_cfg(**{"opensr_test.max_skipped_fraction": 0.9})
    result = run_opensr_test(
        _Loader([_batch(4, seed=0, corrupt=0)]),
        lambda lr: bicubic_upsample(lr, SCALE),
        2,
        cfg,
    )
    assert result.num_attempted == 2
    assert result.num_skipped == 1
    assert result.num_scored == 1


def test_empty_loader_raises():
    cfg = make_cfg()
    with pytest.raises(ValueError, match="no samples"):
        run_opensr_test(_Loader([]), lambda lr: bicubic_upsample(lr, SCALE), 4, cfg)


def test_summary_counts_nonfinite_rather_than_dropping_it():
    """A NaN spatial value must not silently change that column's denominator."""
    result = OpenSRResult(method="test")
    for i in range(4):
        row = {"sample_id": f"s{i}"}
        row.update({m: 0.5 for m in OPENSR_METRICS})
        if i == 0:
            row["spatial"] = float("nan")
        result.rows.append(row)

    summary = result.summary()
    assert summary["spatial"]["count"] == 4
    assert summary["spatial"]["num_finite"] == 3
    assert summary["spatial"]["num_nonfinite"] == 1
    assert summary["reflectance"]["num_nonfinite"] == 0
    assert math.isfinite(summary["spatial"]["mean"])


def test_markdown_reports_both_denominators():
    result = OpenSRResult(method="bicubic")
    row = {"sample_id": "s0"}
    row.update({m: 0.5 for m in OPENSR_METRICS})
    result.rows.append(row)
    result.skipped.append({"sample_id": "s1", "stage": "adapter", "reason": "boom"})

    table = result.to_markdown()
    assert "1 scored of 2 attempted" in table
    assert "2 samples attempted, 1 scored, 1 skipped" in table
    assert "boom" in table
    assert "↓" in table and "↑" in table


def test_json_records_every_skipped_sample(tmp_path):
    """A skip RATE without the ids behind it cannot be investigated later."""
    result = OpenSRResult(method="bicubic", settings={"border_mask": 16})
    row = {"sample_id": "s0"}
    row.update({m: 0.5 for m in OPENSR_METRICS})
    result.rows.append(row)
    result.skipped.append({"sample_id": "s1", "stage": "compute", "reason": "boom"})

    import json

    path = result.to_json(tmp_path / "out.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["num_attempted"] == 2
    assert payload["num_scored"] == 1
    assert payload["num_skipped"] == 1
    assert payload["skipped"][0]["sample_id"] == "s1"
    assert payload["skip_reasons"] == {"boom": 1}
    assert payload["settings"]["border_mask"] == 16
