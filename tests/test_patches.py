"""Tests for patch extraction and the rejection filter.

The property under test is the one that cannot be seen in a loss curve: the HR
crop is exactly ``scale`` times the LR crop, in origin and in size, in every
mode. Everything else here -- grid coverage, filter counting, error messages --
is secondary, because those failures are loud and this one is silent.
"""

import numpy as np
import pytest
import torch

from src.data.patches import (
    PatchCoords,
    PatchExtractionError,
    PatchFilter,
    assert_patch_alignment,
    crop_pair,
    extract_patches,
    patch_grid,
    patch_params_from_cfg,
)
from src.utils.config import load_config

SCALE = 4


def _pair(lr_size=64, channels=4, scale=SCALE, seed=0):
    """A textured LR/HR pair whose HR is exactly the LR block structure x scale."""
    rng = np.random.default_rng(seed)
    hr = rng.random((channels, lr_size * scale, lr_size * scale), dtype=np.float32) * 0.4
    lr = (
        hr.reshape(channels, lr_size, scale, lr_size, scale)
        .mean(axis=(2, 4))
        .astype(np.float32)
    )
    return lr, hr


# -- the alignment invariant ----------------------------------------------


@pytest.mark.parametrize("lr_size,stride", [(16, 16), (16, 8), (32, 7), (8, 3)])
def test_grid_hr_origin_is_exactly_scale_times_lr_origin(lr_size, stride):
    lr, hr = _pair()
    patches = extract_patches(lr, hr, lr_size=lr_size, scale=SCALE, stride=stride)
    assert patches
    for patch in patches:
        coords = patch["coords"]
        assert coords.hr_row == coords.lr_row * SCALE
        assert coords.hr_col == coords.lr_col * SCALE
        assert coords.hr_size == coords.lr_size * SCALE


def test_random_hr_origin_is_exactly_scale_times_lr_origin():
    lr, hr = _pair()
    rng = np.random.default_rng(7)
    patches = extract_patches(
        lr, hr, lr_size=16, scale=SCALE, mode="random", num_patches=40, rng=rng
    )
    assert len(patches) == 40
    for patch in patches:
        coords = patch["coords"]
        assert (coords.hr_row, coords.hr_col) == (coords.lr_row * SCALE, coords.lr_col * SCALE)


def test_extracted_pixels_are_the_pixels_the_coordinates_name():
    """The coordinates and the returned arrays must agree, not merely be consistent."""
    lr, hr = _pair()
    patches = extract_patches(lr, hr, lr_size=16, scale=SCALE, stride=16)
    for patch in patches:
        coords = patch["coords"]
        expected_lr = lr[
            :,
            coords.lr_row : coords.lr_row + coords.lr_size,
            coords.lr_col : coords.lr_col + coords.lr_size,
        ]
        expected_hr = hr[
            :,
            coords.lr_row * SCALE : coords.lr_row * SCALE + coords.lr_size * SCALE,
            coords.lr_col * SCALE : coords.lr_col * SCALE + coords.lr_size * SCALE,
        ]
        assert np.array_equal(patch["lr"], expected_lr)
        assert np.array_equal(patch["hr"], expected_hr)


def test_patch_content_is_scale_consistent():
    """Block-mean the HR patch and it must reproduce the LR patch.

    This is the end-to-end check that the two crops describe the same ground: a
    one-pixel offset in either origin breaks it even though both crops are
    individually valid.
    """
    lr, hr = _pair()
    patches = extract_patches(lr, hr, lr_size=16, scale=SCALE, stride=16)
    for patch in patches:
        hr_patch = patch["hr"]
        channels, height, width = hr_patch.shape
        degraded = hr_patch.reshape(
            channels, height // SCALE, SCALE, width // SCALE, SCALE
        ).mean(axis=(2, 4))
        assert np.allclose(degraded, patch["lr"], atol=1e-6)


def test_assert_patch_alignment_rejects_a_tampered_origin():
    coords = PatchCoords(
        lr_row=4, lr_col=4, lr_size=16, hr_row=17, hr_col=16, hr_size=64, scale=4
    )
    with pytest.raises(PatchExtractionError, match="not 4 x the LR origin"):
        assert_patch_alignment(coords, 4)


def test_assert_patch_alignment_rejects_a_tampered_size():
    coords = PatchCoords(
        lr_row=0, lr_col=0, lr_size=16, hr_row=0, hr_col=0, hr_size=63, scale=4
    )
    with pytest.raises(PatchExtractionError, match="HR patch size"):
        assert_patch_alignment(coords, 4)


def test_crop_pair_rejects_a_crop_outside_the_tile():
    lr, hr = _pair(lr_size=16)
    with pytest.raises(PatchExtractionError, match="outside"):
        crop_pair(lr, hr, lr_row=8, lr_col=0, lr_size=16, scale=SCALE)


# -- grid geometry ---------------------------------------------------------


def test_grid_is_deterministic():
    assert patch_grid(64, 64, 16, 16) == patch_grid(64, 64, 16, 16)


def test_grid_covers_the_edge_when_the_stride_does_not_divide_evenly():
    origins = patch_grid(70, 70, 16, 16)
    rows = sorted({row for row, _ in origins})
    assert rows[-1] == 70 - 16, "the last patch must sit flush against the edge"
    assert rows == [0, 16, 32, 48, 54]


def test_grid_can_drop_the_edge_patch():
    origins = patch_grid(70, 70, 16, 16, drop_incomplete_edge=True)
    rows = sorted({row for row, _ in origins})
    assert rows == [0, 16, 32, 48]


def test_overlapping_stride_yields_more_patches_than_a_full_stride():
    lr, hr = _pair()
    dense = extract_patches(lr, hr, lr_size=16, scale=SCALE, stride=8)
    sparse = extract_patches(lr, hr, lr_size=16, scale=SCALE, stride=16)
    assert len(dense) > len(sparse)


def test_grid_covers_every_pixel_at_stride_equal_to_size():
    lr, hr = _pair(lr_size=64)
    patches = extract_patches(lr, hr, lr_size=16, scale=SCALE, stride=16)
    seen = np.zeros((64, 64), dtype=bool)
    for patch in patches:
        coords = patch["coords"]
        seen[
            coords.lr_row : coords.lr_row + 16, coords.lr_col : coords.lr_col + 16
        ] = True
    assert seen.all()


# -- input validation ------------------------------------------------------


def test_mismatched_scale_is_rejected():
    lr = np.zeros((4, 16, 16), dtype=np.float32)
    hr = np.zeros((4, 48, 48), dtype=np.float32)
    with pytest.raises(PatchExtractionError, match="not a valid LR/HR pair"):
        extract_patches(lr, hr, lr_size=8, scale=SCALE, stride=8)


def test_channel_mismatch_is_rejected():
    lr = np.zeros((4, 16, 16), dtype=np.float32)
    hr = np.zeros((3, 64, 64), dtype=np.float32)
    with pytest.raises(PatchExtractionError, match="channels"):
        extract_patches(lr, hr, lr_size=8, scale=SCALE, stride=8)


def test_rank_2_input_is_rejected():
    with pytest.raises(PatchExtractionError, match="rank 3"):
        extract_patches(
            np.zeros((16, 16), dtype=np.float32),
            np.zeros((64, 64), dtype=np.float32),
            lr_size=8,
            scale=SCALE,
            stride=8,
        )


def test_patch_larger_than_the_tile_is_rejected():
    lr, hr = _pair(lr_size=16)
    with pytest.raises(PatchExtractionError, match="smaller than the requested"):
        extract_patches(lr, hr, lr_size=32, scale=SCALE, stride=8)


def test_grid_mode_requires_a_stride():
    lr, hr = _pair(lr_size=16)
    with pytest.raises(PatchExtractionError, match="stride is required"):
        extract_patches(lr, hr, lr_size=8, scale=SCALE, mode="grid")


def test_random_mode_requires_an_explicit_rng():
    """No implicit global RNG: an unseeded default would make training unrepeatable."""
    lr, hr = _pair(lr_size=16)
    with pytest.raises(PatchExtractionError, match="rng is required"):
        extract_patches(lr, hr, lr_size=8, scale=SCALE, mode="random", num_patches=2)


def test_random_mode_requires_a_patch_count():
    lr, hr = _pair(lr_size=16)
    with pytest.raises(PatchExtractionError, match="num_patches"):
        extract_patches(
            lr,
            hr,
            lr_size=8,
            scale=SCALE,
            mode="random",
            rng=np.random.default_rng(0),
        )


def test_unknown_mode_is_rejected():
    lr, hr = _pair(lr_size=16)
    with pytest.raises(PatchExtractionError, match="mode must be one of"):
        extract_patches(lr, hr, lr_size=8, scale=SCALE, stride=8, mode="sliding")


def test_random_mode_is_reproducible_from_the_seed():
    lr, hr = _pair()
    first = extract_patches(
        lr, hr, lr_size=16, scale=SCALE, mode="random", num_patches=5,
        rng=np.random.default_rng(123),
    )
    second = extract_patches(
        lr, hr, lr_size=16, scale=SCALE, mode="random", num_patches=5,
        rng=np.random.default_rng(123),
    )
    assert [p["coords"] for p in first] == [p["coords"] for p in second]


def test_torch_tensors_are_accepted_and_preserved():
    lr, hr = _pair(lr_size=16)
    patches = extract_patches(
        torch.from_numpy(lr), torch.from_numpy(hr), lr_size=8, scale=SCALE, stride=8
    )
    assert patches
    assert torch.is_tensor(patches[0]["lr"])
    assert patches[0]["lr"].dtype == torch.float32


# -- the filter ------------------------------------------------------------


def _filter(**overrides):
    settings = dict(
        max_nodata_fraction=0.05,
        min_std=0.005,
        bright_reflectance=0.6,
        max_bright_fraction=0.2,
        nodata_fill=0.0,
    )
    settings.update(overrides)
    return PatchFilter(**settings)


def test_filter_accepts_ordinary_reflectance():
    lr, hr = _pair(lr_size=16)
    assert _filter()(lr, hr) is None


def test_filter_rejects_a_near_constant_patch():
    lr = np.full((4, 16, 16), 0.2, dtype=np.float32)
    hr = np.full((4, 64, 64), 0.2, dtype=np.float32)
    assert _filter()(lr, hr) == "constant"


def test_filter_rejects_a_nodata_heavy_patch():
    lr, hr = _pair(lr_size=16)
    lr = lr.copy()
    hr = hr.copy()
    lr[:, :8, :] = 0.0
    hr[:, :32, :] = 0.0
    assert _filter()(lr, hr) == "nodata"


def test_filter_uses_a_supplied_mask_over_the_fill_heuristic():
    """A real mask wins: a legitimately dark scene must not read as nodata."""
    lr, hr = _pair(lr_size=16)
    mask = np.zeros((16, 16), dtype=bool)
    mask[:12] = True
    assert _filter()(lr, hr, nodata_mask=mask) == "nodata"
    assert _filter()(lr, hr, nodata_mask=np.zeros((16, 16), dtype=bool)) is None


def test_filter_rejects_a_cloud_bright_patch():
    lr = np.full((4, 16, 16), 0.9, dtype=np.float32)
    hr = np.full((4, 64, 64), 0.9, dtype=np.float32)
    hr += np.random.default_rng(0).normal(0, 0.02, hr.shape).astype(np.float32)
    assert _filter()(lr, hr) == "cloud"


def test_filter_does_not_modify_bright_reflectance():
    """The cloud rule counts bright pixels; clipping them would break radiometry."""
    lr, hr = _pair(lr_size=16)
    hr = hr.copy()
    hr[:, 0, 0] = 1.4
    before = hr.copy()
    _filter()(lr, hr)
    assert np.array_equal(hr, before)


def test_filter_counts_every_rule_separately():
    patch_filter = _filter()
    lr, hr = _pair(lr_size=16)
    patch_filter(lr, hr)
    patch_filter(np.zeros((4, 16, 16), np.float32), np.zeros((4, 64, 64), np.float32))
    patch_filter(
        np.full((4, 16, 16), 0.2, np.float32), np.full((4, 64, 64), 0.2, np.float32)
    )
    assert patch_filter.counts["examined"] == 3
    assert patch_filter.counts["accepted"] == 1
    assert patch_filter.counts["rejected_nodata"] == 1
    assert patch_filter.counts["rejected_constant"] == 1
    summary = patch_filter.summary()
    assert "nodata" in summary and "rejected" in summary and "examined=3" in summary


def test_disabled_filter_accepts_but_still_counts():
    patch_filter = _filter(enabled=False)
    patch_filter(
        np.full((4, 16, 16), 0.2, np.float32), np.full((4, 64, 64), 0.2, np.float32)
    )
    assert patch_filter.counts["accepted"] == 1
    assert patch_filter.counts["would_reject_constant"] == 1


def test_filter_drops_patches_during_extraction_and_counts_them():
    lr, hr = _pair(lr_size=32)
    lr, hr = lr.copy(), hr.copy()
    # Blank the top half of the tile: those patches must be rejected as nodata.
    lr[:, :16, :] = 0.0
    hr[:, :64, :] = 0.0
    patch_filter = _filter()
    patches = extract_patches(
        lr, hr, lr_size=16, scale=SCALE, stride=16, patch_filter=patch_filter
    )
    assert patch_filter.counts["examined"] == 4
    assert patch_filter.counts["rejected_nodata"] == 2
    assert len(patches) == 2
    for patch in patches:
        assert patch["coords"].lr_row == 16


def test_filter_rejects_an_invalid_std_statistic():
    with pytest.raises(ValueError, match="std_statistic"):
        _filter(std_statistic="median")


def test_filter_from_cfg_reads_the_configured_thresholds():
    cfg = load_config("configs/base.yaml")
    patch_filter = PatchFilter.from_cfg(cfg)
    assert patch_filter.max_nodata_fraction == float(
        cfg.patches.filters.max_nodata_fraction
    )
    assert patch_filter.min_std == float(cfg.patches.filters.min_std)
    assert patch_filter.nodata_fill == float(cfg.dataset.nodata_fill)


# -- config plumbing -------------------------------------------------------


def test_patch_params_match_the_config():
    cfg = load_config("configs/base.yaml")
    params = patch_params_from_cfg(cfg)
    assert params["lr_size"] == int(cfg.patches.lr_size)
    assert params["scale"] == int(cfg.sr.scale)
    assert params["hr_size"] == params["lr_size"] * params["scale"]


def test_patch_larger_than_the_emitted_tile_is_rejected_by_config_check():
    cfg = load_config("configs/base.yaml", overrides=["patches.lr_size=999"])
    with pytest.raises(PatchExtractionError, match="exceeds cfg.sr.lr_patch_size"):
        patch_params_from_cfg(cfg)


def test_unknown_mode_in_config_is_rejected():
    cfg = load_config("configs/base.yaml", overrides=["patches.mode=diagonal"])
    with pytest.raises(PatchExtractionError, match="not one of"):
        patch_params_from_cfg(cfg)
