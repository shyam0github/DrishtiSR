"""Tests for dataloader construction.

The property that matters here is the split boundary: patches from one scene must
never straddle train and validation. A leak there does not raise, it just makes
every validation number optimistic -- and for super-resolution, where the target
is the high-frequency detail the model may have memorised, optimistic by a lot.

Everything runs on the synthetic stub: no network, no cache, no GPU.
"""

import csv

import pytest
import torch

from src.data.loader import (
    PatchDataset,
    SplitError,
    build_dataloaders,
    resolve_split_assignments,
    select_indices,
    worker_init_fn,
)
from src.data.registry import get_dataset
from src.utils.config import load_config
from src.utils.logging import get_logger


def _cfg(tmp_path, *overrides):
    """A smoke config whose outputs go to a temporary directory.

    ``manifest_dir`` points at ``tmp_path``, so no split file exists and the
    in-process geographic split is exercised. Tests that want the file path write
    one there themselves.
    """
    base = [
        f"paths.manifest_dir={tmp_path.as_posix()}",
        f"paths.log_file={(tmp_path / 'run.log').as_posix()}",
        "loader.cached_only=false",
    ]
    return load_config("configs/base.yaml", smoke=True, overrides=base + list(overrides))


def _logger(tmp_path):
    return get_logger("test.loader", log_file=tmp_path / "run.log")


# -- split assignment ------------------------------------------------------


def test_split_is_recomputed_when_no_file_exists(tmp_path):
    cfg = _cfg(tmp_path)
    dataset = get_dataset(cfg)
    assignments, source = resolve_split_assignments(cfg, dataset, _logger(tmp_path))
    assert source == "computed"
    assert len(assignments) == len(dataset)
    assert set(assignments.values()) <= set(cfg.splits.fractions.keys())


def test_split_file_is_preferred_when_present(tmp_path):
    cfg = _cfg(tmp_path)
    dataset = get_dataset(cfg)
    path = tmp_path / str(cfg.splits.output_name).format(dataset=cfg.dataset.name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "split", "group_id"])
        for idx in range(len(dataset)):
            sample_id = dataset[idx]["meta"]["sample_id"]
            writer.writerow([sample_id, "train" if idx % 2 else "val", idx])

    assignments, source = resolve_split_assignments(cfg, dataset, _logger(tmp_path))
    assert source.startswith("file:")
    assert sum(1 for v in assignments.values() if v == "val") == 6


def test_a_split_file_that_misses_samples_is_an_error_not_a_silent_recompute(tmp_path):
    """A stale split file must fail loudly: the user believes it is in force."""
    cfg = _cfg(tmp_path)
    dataset = get_dataset(cfg)
    path = tmp_path / str(cfg.splits.output_name).format(dataset=cfg.dataset.name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "split"])
        writer.writerow([dataset[0]["meta"]["sample_id"], "train"])

    with pytest.raises(SplitError, match="absent from it"):
        resolve_split_assignments(cfg, dataset, _logger(tmp_path))


def test_a_malformed_split_file_is_an_error(tmp_path):
    cfg = _cfg(tmp_path)
    dataset = get_dataset(cfg)
    path = tmp_path / str(cfg.splits.output_name).format(dataset=cfg.dataset.name)
    path.write_text("sample_id,group\nstub_0000,1\n", encoding="utf-8")
    with pytest.raises(SplitError, match="required column"):
        resolve_split_assignments(cfg, dataset, _logger(tmp_path))


def test_an_empty_split_is_an_error(tmp_path):
    cfg = _cfg(tmp_path)
    dataset = get_dataset(cfg)
    assignments = {
        dataset[i]["meta"]["sample_id"]: "train" for i in range(len(dataset))
    }
    with pytest.raises(SplitError, match="below cfg.loader.min_samples_per_split"):
        select_indices(cfg, dataset, assignments, "val", _logger(tmp_path))


def test_scene_straddling_a_split_boundary_is_rejected(tmp_path):
    """Two records of one NAIP quarter-quad on opposite sides must not pass."""
    from src.data.loader import _assert_scene_disjoint

    assignments = {
        "NA5120_E1183N0757__m_3912321_nw_10_060_20220710": "train",
        "NA5120_E1183N0757__m_3912321_nw_10_060_20200604": "val",
    }
    with pytest.raises(SplitError, match="more than one split"):
        _assert_scene_disjoint(assignments, _logger(tmp_path))


def test_distinct_scenes_in_different_splits_are_fine(tmp_path):
    from src.data.loader import _assert_scene_disjoint

    _assert_scene_disjoint(
        {
            "NA5120_E1183N0757__m_3912321_nw_10_060_20220710": "train",
            "NA5120_E1183N0791__m_4112455_se_10_060_20220620": "val",
        },
        _logger(tmp_path),
    )


# -- patch datasets --------------------------------------------------------


def test_grid_patches_are_deterministic_and_scale_aligned(tmp_path):
    cfg = _cfg(tmp_path)
    dataset = get_dataset(cfg)
    first = PatchDataset(cfg, dataset, [0, 1], "grid", "val", logger=_logger(tmp_path))
    second = PatchDataset(cfg, dataset, [0, 1], "grid", "val", logger=_logger(tmp_path))
    assert len(first) == len(second) > 0
    for item in range(len(first)):
        a, b = first[item], second[item]
        assert a["lr_row"] == b["lr_row"] and a["lr_col"] == b["lr_col"]
        assert a["hr_row"] == a["lr_row"] * cfg.sr.scale
        assert a["hr_col"] == a["lr_col"] * cfg.sr.scale
        assert torch.equal(a["hr"], b["hr"])


def test_patch_shapes_follow_the_config(tmp_path):
    cfg = _cfg(tmp_path)
    dataset = get_dataset(cfg)
    patches = PatchDataset(cfg, dataset, [0], "grid", "val", logger=_logger(tmp_path))
    item = patches[0]
    size = int(cfg.patches.lr_size)
    assert item["lr"].shape == (len(cfg.dataset.bands), size, size)
    assert item["hr"].shape == (
        len(cfg.dataset.bands),
        size * cfg.sr.scale,
        size * cfg.sr.scale,
    )
    assert item["lr"].dtype == torch.float32


def test_random_patches_are_reproducible_and_epoch_dependent(tmp_path):
    cfg = _cfg(tmp_path)
    dataset = get_dataset(cfg)
    patches = PatchDataset(cfg, dataset, [0, 1], "random", "train", logger=_logger(tmp_path))

    first = [(patches[i]["lr_row"], patches[i]["lr_col"]) for i in range(len(patches))]
    again = [(patches[i]["lr_row"], patches[i]["lr_col"]) for i in range(len(patches))]
    assert first == again, "same epoch must give the same crops"

    patches.set_epoch(1)
    later = [(patches[i]["lr_row"], patches[i]["lr_col"]) for i in range(len(patches))]
    assert later != first, "a new epoch must draw new crops"


def test_random_patch_hr_origin_is_scale_times_lr_origin(tmp_path):
    cfg = _cfg(tmp_path)
    dataset = get_dataset(cfg)
    patches = PatchDataset(cfg, dataset, [0, 1], "random", "train", logger=_logger(tmp_path))
    for item in range(len(patches)):
        entry = patches[item]
        assert entry["hr_row"] == entry["lr_row"] * cfg.sr.scale
        assert entry["hr_col"] == entry["lr_col"] * cfg.sr.scale


def test_a_tile_of_unexpected_size_is_rejected(tmp_path):
    """The grid is planned against cfg.sr.lr_patch_size; a mismatch must not pass."""
    cfg = _cfg(tmp_path, "sr.lr_patch_size=48", "sr.hr_patch_size=192")
    dataset = get_dataset(cfg)  # the stub still emits 48 -> but cfg drives both
    patches = PatchDataset(cfg, dataset, [0], "grid", "val", logger=_logger(tmp_path))
    patches.tile_lr_size = 999
    with pytest.raises(SplitError, match="cfg.sr.lr_patch_size"):
        patches[0]


def test_out_of_range_index_raises(tmp_path):
    cfg = _cfg(tmp_path)
    dataset = get_dataset(cfg)
    patches = PatchDataset(cfg, dataset, [0], "grid", "val", logger=_logger(tmp_path))
    with pytest.raises(IndexError):
        patches[len(patches)]


def test_a_split_whose_patches_are_all_rejected_is_an_error(tmp_path):
    """An empty validation set must not be built quietly."""
    cfg = _cfg(tmp_path, "patches.filters.min_std=99.0")
    dataset = get_dataset(cfg)
    with pytest.raises(SplitError, match="rejected every"):
        PatchDataset(cfg, dataset, [0], "grid", "val", logger=_logger(tmp_path))


# -- worker seeding --------------------------------------------------------


def test_worker_init_fn_is_reproducible():
    import numpy as np

    def draw(worker_id, base_seed):
        torch.manual_seed(base_seed)
        worker_init_fn(worker_id)
        return np.random.rand(3).tolist(), torch.rand(3).tolist()

    assert draw(0, 1234) == draw(0, 1234), "same seed and worker must repeat"
    assert draw(0, 1234) != draw(1, 1234), "different workers must diverge"


# -- the entry point -------------------------------------------------------


def test_build_dataloaders_returns_usable_train_and_val_loaders(tmp_path):
    cfg = _cfg(tmp_path)
    built = build_dataloaders(cfg)

    batch = next(iter(built["train"]))
    size = int(cfg.patches.lr_size)
    assert batch["lr"].shape[1:] == (len(cfg.dataset.bands), size, size)
    assert batch["hr"].shape[1:] == (
        len(cfg.dataset.bands),
        size * cfg.sr.scale,
        size * cfg.sr.scale,
    )
    assert batch["lr"].dtype == torch.float32
    assert next(iter(built["val"]))["lr"].shape[0] > 0


def test_train_and_val_never_share_a_sample(tmp_path):
    built = build_dataloaders(_cfg(tmp_path))
    assert not set(built["indices"]["train"]) & set(built["indices"]["val"])


def test_patches_never_straddle_the_split(tmp_path):
    """Every patch's source sample must belong to that loader's split, only."""
    cfg = _cfg(tmp_path)
    built = build_dataloaders(cfg)
    train_ids = {
        built["datasets"]["train"][i]["sample_id"]
        for i in range(len(built["datasets"]["train"]))
    }
    val_ids = {
        built["datasets"]["val"][i]["sample_id"]
        for i in range(len(built["datasets"]["val"]))
    }
    assert train_ids and val_ids
    assert not train_ids & val_ids


def test_build_dataloaders_is_deterministic(tmp_path):
    cfg = _cfg(tmp_path)
    first = build_dataloaders(cfg)
    second = build_dataloaders(cfg)
    assert first["indices"] == second["indices"]

    def val_order(built):
        dataset = built["datasets"]["val"]
        return [(dataset[i]["sample_id"], dataset[i]["lr_row"], dataset[i]["lr_col"]) for i in range(len(dataset))]

    assert val_order(first) == val_order(second)


def test_validation_loader_is_never_shuffled_or_truncated(tmp_path):
    built = build_dataloaders(_cfg(tmp_path))
    loader = built["val"]
    assert not loader.drop_last
    assert loader.batch_size == int(_cfg(tmp_path).train.batch_size)
    seen = sum(batch["lr"].shape[0] for batch in loader)
    assert seen == len(built["datasets"]["val"])
