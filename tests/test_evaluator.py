"""Tests for the evaluation harness.

The Evaluator is the piece that turns metrics into reported numbers, so the
properties tested here are about honesty rather than about arithmetic: that the
percentiles describe the tail they claim to, that a non-finite metric is counted
rather than dropped, that a wrong-sized SR output stops the run instead of being
resampled to fit, and that the LPIPS caveat cannot be separated from the LPIPS
number in the table.

No data and no network: the loader is a list of hand-built batches, so this runs
in a second on CPU.
"""

import json

import numpy as np
import pytest
import torch

from src.eval.baselines import get_baseline
from src.metrics.aggregate import Evaluator, summarise
from src.utils.config import load_config

BANDS = 4
LR_SIZE = 16
SCALE = 4


@pytest.fixture()
def cfg():
    """Base config with LPIPS off -- the unit tests must never hit the network."""
    config = load_config()
    config.metrics.lpips.enabled = False
    return config


def _scene(seed, bands=BANDS, size=LR_SIZE * SCALE):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    levels = (0.058, 0.057, 0.034, 0.249)
    image = np.empty((bands, size, size), dtype=np.float32)
    for band in range(bands):
        field = gaussian_filter(rng.random((size, size)), sigma=2.5)
        field = (field - field.min()) / (field.max() - field.min() + 1e-9)
        image[band] = (levels[band % len(levels)] * (0.5 + field)).astype(np.float32)
    return image


def _batches(num_batches=3, batch_size=2, seed=0):
    """Hand-built batches in the shape src.data.loader.PatchDataset collates to."""
    batches = []
    for b in range(num_batches):
        hr = np.stack([_scene(seed + b * batch_size + i) for i in range(batch_size)])
        lr = (
            hr.reshape(batch_size, BANDS, LR_SIZE, SCALE, LR_SIZE, SCALE)
            .mean(axis=(3, 5))
            .astype(np.float32)
        )
        batches.append(
            {
                "lr": torch.from_numpy(lr),
                "hr": torch.from_numpy(hr),
                "sample_id": [f"tile_{b}_{i}" for i in range(batch_size)],
                "dataset_index": torch.arange(batch_size),
                "lr_row": torch.zeros(batch_size, dtype=torch.long),
                "lr_col": torch.zeros(batch_size, dtype=torch.long),
                "hr_row": torch.zeros(batch_size, dtype=torch.long),
                "hr_col": torch.zeros(batch_size, dtype=torch.long),
                "split": ["val"] * batch_size,
                "filter_reason": [""] * batch_size,
            }
        )
    return batches


def test_evaluator_scores_every_sample_and_names_them(cfg):
    result = Evaluator(cfg).run(
        _batches(), get_baseline("bicubic", SCALE), name="bicubic", split="val"
    )
    assert len(result.per_sample) == 6
    assert list(result.per_sample["sample_id"])[0] == "tile_0_0"
    for column in ("psnr_mean", "ssim_mean", "sam_mean_deg", "ergas", "sr_time_ms"):
        assert column in result.per_sample.columns
    # Per-band columns are named by band, not by index, so a reordered band list
    # cannot silently relabel a column.
    for band in cfg.dataset.bands:
        assert f"psnr_{band}" in result.per_sample.columns
        assert f"ssim_{band}" in result.per_sample.columns


def test_summary_reports_percentiles_not_just_the_mean(cfg):
    result = Evaluator(cfg).run(
        _batches(num_batches=4), get_baseline("bicubic", SCALE), "bicubic", split="val"
    )
    entry = result.summary["psnr_mean"]
    for key in ("mean", "std", "p5", "p95", "min", "max", "num_finite"):
        assert key in entry
    assert entry["min"] <= entry["p5"] <= entry["mean"] <= entry["p95"] <= entry["max"]
    assert entry["worst_sample_id"] in set(result.per_sample["sample_id"])


def test_worst_sample_follows_each_metric_direction(cfg):
    """Worst PSNR is the lowest; worst SAM is the highest. Getting this backwards
    would point a reader at the best tile and call it the failure case."""
    result = Evaluator(cfg).run(
        _batches(num_batches=4), get_baseline("bicubic", SCALE), "bicubic", split="val"
    )
    frame = result.per_sample
    assert result.summary["psnr_mean"]["worst_value"] == pytest.approx(
        frame["psnr_mean"].min()
    )
    assert result.summary["sam_mean_deg"]["worst_value"] == pytest.approx(
        frame["sam_mean_deg"].max()
    )


def test_bicubic_beats_nearest_through_the_full_harness(cfg):
    """The comparison the baseline run exists to produce, end to end."""
    batches = _batches(num_batches=3)
    evaluator = Evaluator(cfg)
    bicubic = evaluator.run(batches, get_baseline("bicubic", SCALE), "bicubic", split="val")
    nearest = evaluator.run(batches, get_baseline("nearest", SCALE), "nearest", split="val")

    assert bicubic.summary["psnr_mean"]["mean"] > nearest.summary["psnr_mean"]["mean"]
    assert bicubic.summary["ssim_mean"]["mean"] > nearest.summary["ssim_mean"]["mean"]
    assert bicubic.summary["ergas"]["mean"] < nearest.summary["ergas"]["mean"]


def test_non_finite_metrics_are_counted_never_dropped(cfg):
    """An exactly reconstructed sample gives +inf PSNR. It must be visible."""
    batches = _batches(num_batches=2)
    result = Evaluator(cfg).run(
        batches, lambda lr: batches[0]["hr"] if lr.shape[0] else lr, "identity",
        split="val",
    )
    entry = result.summary["psnr_mean"]
    assert entry["num_nonfinite"] >= 1
    assert entry["count"] == entry["num_finite"] + entry["num_nonfinite"]


def test_summarise_handles_a_column_with_no_finite_values():
    import pandas as pd

    frame = pd.DataFrame(
        {"sample_id": ["a", "b"], "psnr_mean": [np.inf, np.inf]}
    )
    entry = summarise(frame)["psnr_mean"]
    assert entry["num_finite"] == 0
    assert np.isnan(entry["mean"])
    assert entry["worst_sample_id"] is None


def test_summarise_rejects_an_empty_table():
    import pandas as pd

    with pytest.raises(ValueError, match="empty"):
        summarise(pd.DataFrame({"psnr_mean": []}))


def test_wrong_output_shape_stops_the_run(cfg):
    """Never resample to fit: a size mismatch is a scale-factor bug."""
    with pytest.raises(ValueError, match="does not resample"):
        Evaluator(cfg).run(
            _batches(num_batches=1), get_baseline("bicubic", 2), "wrong-scale",
            split="val",
        )


def test_an_empty_dataloader_is_an_error_not_an_empty_table(cfg):
    with pytest.raises(ValueError, match="no samples"):
        Evaluator(cfg).run([], get_baseline("bicubic", SCALE), "bicubic", split="val")


def test_qualitative_samples_are_collected_with_their_sam_maps(cfg):
    result = Evaluator(cfg).run(
        _batches(num_batches=4),
        get_baseline("bicubic", SCALE),
        "bicubic",
        collect_samples=3,
        split="val",
    )
    assert len(result.qualitative) == 3
    # Spread across the run, not the first three of one batch.
    assert len({s["sample_id"] for s in result.qualitative}) == 3
    sample = result.qualitative[0]
    assert sample["lr"].shape == (BANDS, LR_SIZE, LR_SIZE)
    assert sample["sr"].shape == sample["hr"].shape == (BANDS, LR_SIZE * SCALE, LR_SIZE * SCALE)
    assert sample["sam_map"].shape == (LR_SIZE * SCALE, LR_SIZE * SCALE)


def test_markdown_table_states_metric_direction_and_percentiles(cfg):
    result = Evaluator(cfg).run(
        _batches(), get_baseline("bicubic", SCALE), "bicubic", split="val"
    )
    table = result.to_markdown()
    assert "| psnr_mean | higher |" in table
    assert "| sam_mean_deg | lower |" in table
    assert "p5" in table and "p95" in table


def test_lpips_caveat_is_attached_to_the_table_whenever_lpips_is_in_it(cfg):
    """The caveat cannot be separated from the number by copy-paste."""
    result = Evaluator(cfg).run(
        _batches(), get_baseline("bicubic", SCALE), "bicubic", split="val"
    )
    assert "LPIPS caveat" not in result.to_markdown()  # LPIPS was disabled

    result.summary["lpips"] = {
        "mean": 0.3, "std": 0.1, "min": 0.1, "max": 0.5,
        "p5": 0.15, "p95": 0.45, "num_finite": 6, "num_nonfinite": 0, "count": 6,
    }
    table = result.to_markdown()
    assert "| lpips | lower |" in table
    assert "LPIPS caveat" in table
    assert "perceptual proxy" in table


def test_written_artefacts_carry_the_settings_and_the_caveat(cfg, tmp_path):
    result = Evaluator(cfg).run(
        _batches(), get_baseline("bicubic", SCALE), "bicubic", split="val"
    )
    csv_path = result.to_csv(tmp_path / "metrics" / "baseline_bicubic.csv")
    json_path = result.to_json(tmp_path / "metrics" / "baseline_bicubic.json")

    assert csv_path.is_file() and json_path.is_file()
    assert len(csv_path.read_text(encoding="utf-8").strip().splitlines()) == 7

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["method"] == "bicubic"
    assert payload["settings"]["data_range_reflectance"] == 1.0
    assert payload["settings"]["bands"] == list(cfg.dataset.bands)
    assert payload["settings"]["scale"] == SCALE
    assert payload["settings"]["ssim"]["use_sample_covariance"] is False
    assert "perceptual proxy" in payload["lpips_caveat"]


def test_unknown_metric_in_config_is_rejected_at_construction(cfg):
    cfg.metrics.enabled = ["psnr", "niqe"]
    with pytest.raises(KeyError, match="unknown metric"):
        Evaluator(cfg)


def test_rgb_indices_are_resolved_by_band_name(cfg):
    """A reordered band list must move the LPIPS channels with it."""
    from src.metrics.image_quality import rgb_band_indices

    assert rgb_band_indices(cfg) == (0, 1, 2)
    cfg.dataset.bands = ["B08", "B02", "B03", "B04"]
    assert rgb_band_indices(cfg) == (3, 2, 1)
    cfg.dataset.bands = ["B08", "B11"]
    with pytest.raises(KeyError, match="not in cfg.dataset.bands"):
        rgb_band_indices(cfg)
