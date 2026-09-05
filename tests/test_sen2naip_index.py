"""Tests for the SEN2NAIPv2 index (``build_index``) and the crop that moved.

The index is what stands between a 2.3 GB download and a training run, and it
answers exactly two questions the rest of the project cannot recover later:
how much of the archive is unusable, and whether the reflectance divisor is
right. Both answers are destroyed by measuring a centre crop instead of the
full tile -- nodata sits at tile edges and the bright targets that expose a
wrong divisor sit anywhere but the middle -- so the no-crop guarantee is
asserted here rather than assumed.
"""

import json

import numpy as np
import pytest

from src.data.patches import PatchExtractionError, centre_crop_pair
from src.data.sen2naip import (
    NATIVE_BANDS,
    SEN2NAIPv2Dataset,
    format_index_summary,
)
from src.utils.config import load_config

LR_TILE = 130
HR_TILE = 520
SCALE = 4
NODATA = 65535


def _cfg(tmp_path, *overrides):
    return load_config(
        "configs/base.yaml",
        overrides=[
            f"paths.cache_dir={tmp_path.as_posix()}/cache",
            f"paths.log_file={tmp_path.as_posix()}/run.log",
            f"paths.manifest_dir={tmp_path.as_posix()}",
            *overrides,
        ],
    )


class _FakeSEN2NAIP(SEN2NAIPv2Dataset):
    """SEN2NAIPv2 over a hand-written cache. Never touches the network.

    The catalog is supplied rather than fetched, so every code path below the
    catalog -- caching, indexing, statistics -- runs exactly as it does on the
    real dataset.
    """

    def __init__(self, cfg, sample_ids):
        self._fake_ids = list(sample_ids)
        super().__init__(cfg)

    def _load_catalog(self):
        return [
            {
                "row_position": i,
                "sample_id": sample_id,
                "crs": "EPSG:32610",
                "geotransform": (0.0, 2.5, 0.0, 0.0, 0.0, -2.5),
                "data_split": "train",
                "correlation": 0.9,
                "centroid_lonlat": (-122.0, 38.0),
            }
            for i, sample_id in enumerate(self._fake_ids)
        ]

    def write_cache(self, sample_id, lr=None, hr=None):
        """Write one cache entry as raw uint16 digital numbers, uncropped."""
        bands = len(NATIVE_BANDS)
        if lr is None:
            lr = np.full((bands, LR_TILE, LR_TILE), 1500, dtype=np.uint16)
        if hr is None:
            hr = np.full((bands, HR_TILE, HR_TILE), 1500, dtype=np.uint16)
        npz_path, json_path = self._cache_paths(sample_id)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        with npz_path.open("wb") as handle:
            np.savez_compressed(handle, lr=lr, hr=hr)
        json_path.write_text(json.dumps({"sample_id": sample_id}), encoding="utf-8")
        return lr, hr


def _dataset(tmp_path, n=4, overrides=()):
    ids = [
        f"NA5120_E{1100 + i:04d}N0757__m_{3900000 + i}_nw_10_060_20220710"
        for i in range(n)
    ]
    return _FakeSEN2NAIP(_cfg(tmp_path, *overrides), ids)


# -- the crop that moved ---------------------------------------------------


def test_load_sample_returns_the_full_tile_uncropped(tmp_path):
    """The read path must not crop. This is the amendment's core guarantee."""
    dataset = _dataset(tmp_path, n=1)
    dataset.write_cache(dataset.catalog[0]["sample_id"])

    sample = dataset.load_sample(0)

    assert tuple(sample["lr"].shape) == (4, LR_TILE, LR_TILE)
    assert tuple(sample["hr"].shape) == (4, HR_TILE, HR_TILE)


def test_the_dataset_has_no_crop_helper_left(tmp_path):
    """A regression guard: the destructive crop must be gone, not just unused."""
    assert not hasattr(SEN2NAIPv2Dataset, "_crop_pair")


def test_centre_crop_is_deterministic():
    lr = np.random.default_rng(0).random((4, LR_TILE, LR_TILE)).astype(np.float32)
    hr = np.random.default_rng(1).random((4, HR_TILE, HR_TILE)).astype(np.float32)

    first = centre_crop_pair(lr, hr, 128, SCALE)
    again = centre_crop_pair(lr, hr, 128, SCALE)

    assert np.array_equal(first["lr"], again["lr"])
    assert first["coords"].lr_row == first["coords"].lr_col == 1
    assert first["coords"].hr_row == 4


def test_centre_crop_rejects_a_crop_larger_than_the_tile():
    lr = np.zeros((4, 64, 64), dtype=np.float32)
    hr = np.zeros((4, 256, 256), dtype=np.float32)
    with pytest.raises(PatchExtractionError, match="exceeds"):
        centre_crop_pair(lr, hr, 128, SCALE)


# -- nodata ----------------------------------------------------------------


def test_nodata_mask_uses_any_band_not_all(tmp_path):
    """One dead band makes the whole pixel unusable."""
    dataset = _dataset(tmp_path, n=1)
    raw = np.full((4, 10, 10), 1500, dtype=np.uint16)
    raw[0, 0, 0] = NODATA  # one band only

    mask = dataset._nodata_pixel_mask(raw)

    assert mask.shape == (10, 10)
    assert mask[0, 0]
    assert mask.sum() == 1


def test_nodata_fraction_is_measured_on_the_full_tile(tmp_path):
    """Nodata at the tile edge must be counted, not cropped away.

    The nodata is placed in the outer ring only -- exactly the region a
    128 px centre crop of a 130 px tile discards.
    """
    dataset = _dataset(tmp_path, n=1)
    sample_id = dataset.catalog[0]["sample_id"]

    lr = np.full((4, LR_TILE, LR_TILE), 1500, dtype=np.uint16)
    lr[:, 0, :] = NODATA  # top row: 130 of 16900 pixels = 0.77%
    dataset.write_cache(sample_id, lr=lr)

    row = dataset.index_row(0, {})

    assert row["nodata_fraction_lr"] == pytest.approx(LR_TILE / (LR_TILE * LR_TILE), abs=1e-6)
    assert row["nodata_fraction"] > 0, "edge nodata was not counted"


def test_nodata_pixels_are_excluded_from_the_statistics(tmp_path):
    """Not zero-filled and averaged in as if they were black ground."""
    dataset = _dataset(tmp_path, n=1)
    sample_id = dataset.catalog[0]["sample_id"]

    lr = np.full((4, LR_TILE, LR_TILE), 2000, dtype=np.uint16)
    lr[:, 0, 0] = NODATA
    dataset.write_cache(sample_id, lr=lr)

    row = dataset.index_row(0, {})

    # Every valid pixel is 2000 DN = 0.2 reflectance. A zero-fill would drag
    # the mean below 0.2 and put the minimum at 0.0.
    assert row["lr_b0_B04_mean"] == pytest.approx(0.2)
    assert row["lr_b0_B04_min"] == pytest.approx(0.2)
    assert row["lr_b0_B04_std"] == pytest.approx(0.0)


def test_a_sample_over_the_nodata_threshold_is_rejected_but_still_written(tmp_path):
    dataset = _dataset(tmp_path, n=1, overrides=["dataset.max_nodata_fraction=0.02"])
    sample_id = dataset.catalog[0]["sample_id"]

    lr = np.full((4, LR_TILE, LR_TILE), 1500, dtype=np.uint16)
    lr[:, :20, :] = NODATA  # ~15%, well over 2%
    dataset.write_cache(sample_id, lr=lr)

    row = dataset.index_row(0, {})

    assert row["rejected"] is True
    assert "nodata_fraction" in row["rejection_reason"]
    assert row["sample_id"] == sample_id
    assert row["lr_shape"] == f"4x{LR_TILE}x{LR_TILE}", "the row is still described"


def test_an_uncached_sample_is_recorded_rather_than_raised(tmp_path):
    dataset = _dataset(tmp_path, n=1)
    row = dataset.index_row(0, {})

    assert row["rejected"] is True
    assert row["rejection_reason"] == "not_cached"
    assert row["validation_error"]


# -- percentiles -----------------------------------------------------------


def test_percentile_from_histogram_is_exact():
    counts = np.zeros(SEN2NAIPv2Dataset.DN_LEVELS, dtype=np.int64)
    for dn in range(1, 101):
        counts[dn] = 1  # DN 1..100, one pixel each

    p = SEN2NAIPv2Dataset._percentile_from_histogram
    assert p(counts, 0.01) == 1
    assert p(counts, 0.50) == 50
    assert p(counts, 0.99) == 99


def test_percentile_of_an_empty_histogram_raises():
    counts = np.zeros(SEN2NAIPv2Dataset.DN_LEVELS, dtype=np.int64)
    with pytest.raises(ValueError, match="empty histogram"):
        SEN2NAIPv2Dataset._percentile_from_histogram(counts, 0.5)


def test_pooled_percentiles_reach_the_summary(tmp_path):
    dataset = _dataset(tmp_path, n=2)
    for entry in dataset.catalog:
        dataset.write_cache(entry["sample_id"])

    summary = dataset.build_index(tmp_path / "manifest.csv")

    # Every pixel is 1500 DN, so every percentile is 0.15 reflectance.
    assert summary["percentiles"]["lr_B04"]["p50"] == pytest.approx(0.15)
    assert summary["percentiles"]["hr_B08"]["p99"] == pytest.approx(0.15)


# -- incremental / --force -------------------------------------------------


def test_build_index_is_incremental_across_a_growing_download(tmp_path):
    dataset = _dataset(tmp_path, n=4)
    manifest = tmp_path / "manifest.csv"

    dataset.write_cache(dataset.catalog[0]["sample_id"])
    dataset.write_cache(dataset.catalog[1]["sample_id"])
    first = dataset.build_index(manifest)
    assert first["total_indexed"] == 2
    assert first["measured"] == 2
    assert first["reused"] == 0
    assert first["uncached"] == 2

    dataset.write_cache(dataset.catalog[2]["sample_id"])
    second = dataset.build_index(manifest)
    assert second["total_indexed"] == 3
    assert second["measured"] == 1, "already-indexed rows were re-measured"
    assert second["reused"] == 2
    assert second["uncached"] == 1


def test_force_re_measures_everything(tmp_path):
    dataset = _dataset(tmp_path, n=2)
    manifest = tmp_path / "manifest.csv"
    for entry in dataset.catalog:
        dataset.write_cache(entry["sample_id"])

    dataset.build_index(manifest)
    forced = dataset.build_index(manifest, force=True)

    assert forced["measured"] == 2
    assert forced["reused"] == 0


def test_reused_rows_leave_the_percentile_block_empty_rather_than_wrong(tmp_path):
    """A reused row contributes no pixels, so its band stats must not be faked."""
    dataset = _dataset(tmp_path, n=2)
    manifest = tmp_path / "manifest.csv"
    for entry in dataset.catalog:
        dataset.write_cache(entry["sample_id"])

    dataset.build_index(manifest)
    again = dataset.build_index(manifest)

    assert again["reused"] == 2
    assert again["percentiles"] == {}
    assert "re-run with --force" in format_index_summary(again)


def test_indexing_an_empty_cache_raises(tmp_path):
    dataset = _dataset(tmp_path, n=2)
    with pytest.raises(RuntimeError, match="Nothing is cached"):
        dataset.build_index(tmp_path / "manifest.csv")


def test_manifest_carries_the_new_columns(tmp_path):
    import csv

    dataset = _dataset(tmp_path, n=1)
    dataset.write_cache(dataset.catalog[0]["sample_id"])
    manifest = tmp_path / "manifest.csv"
    dataset.build_index(manifest)

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    for column in (
        "nodata_fraction",
        "nodata_fraction_lr",
        "nodata_fraction_hr",
        "rejected",
        "rejection_reason",
        "max_reflectance",
        "lr_b0_B04_mean",
        "lr_b0_B04_std",
    ):
        assert column in rows[0], column
    assert rows[0]["lr_shape"] == f"4x{LR_TILE}x{LR_TILE}"


# -- the divisor verdict ---------------------------------------------------


def test_a_bright_tail_below_the_threshold_does_not_accuse_the_divisor(tmp_path):
    dataset = _dataset(tmp_path, n=20, overrides=["dataset.divisor_suspect_fraction=0.05"])
    for i, entry in enumerate(dataset.catalog):
        lr = np.full((4, LR_TILE, LR_TILE), 1500, dtype=np.uint16)
        dataset.write_cache(entry["sample_id"], lr=lr)

    summary = dataset.build_index(tmp_path / "manifest.csv")

    assert summary["exceeding"] == []
    assert summary["divisor_suspect"] is False
    assert "Divisor looks sound" in format_index_summary(summary)


def test_a_whole_archive_over_the_threshold_accuses_the_divisor(tmp_path):
    """The failure this check exists for: /3000 instead of /10000.

    A forest at 2500 DN is 0.25 reflectance under the correct divisor and 0.83
    under /3000; scaled bright targets then clear 1.2 everywhere at once. The
    summary must say the divisor is wrong, not merely print a percentage.
    """
    dataset = _dataset(tmp_path, n=10, overrides=["dataset.reflectance_scale=3000.0"])
    for entry in dataset.catalog:
        lr = np.full((4, LR_TILE, LR_TILE), 4000, dtype=np.uint16)
        hr = np.full((4, HR_TILE, HR_TILE), 4000, dtype=np.uint16)
        dataset.write_cache(entry["sample_id"], lr=lr, hr=hr)

    summary = dataset.build_index(tmp_path / "manifest.csv")
    block = format_index_summary(summary)

    assert summary["exceed_fraction"] == 1.0
    assert summary["divisor_suspect"] is True
    assert "REFLECTANCE DIVISOR IS SUSPECT" in block
    assert "DO NOT respond by clipping" in block


def test_the_summary_block_names_the_exceeding_samples(tmp_path):
    dataset = _dataset(tmp_path, n=2, overrides=["dataset.reflectance_report_threshold=0.1"])
    for entry in dataset.catalog:
        dataset.write_cache(entry["sample_id"])

    summary = dataset.build_index(tmp_path / "manifest.csv")
    block = format_index_summary(summary)

    assert len(summary["exceeding"]) == 2
    for sample_id in summary["exceeding"]:
        assert sample_id in block


def test_the_summary_reports_the_rejection_tally_by_cause(tmp_path):
    dataset = _dataset(tmp_path, n=3, overrides=["dataset.max_nodata_fraction=0.02"])
    for i, entry in enumerate(dataset.catalog):
        lr = np.full((4, LR_TILE, LR_TILE), 1500, dtype=np.uint16)
        if i < 2:
            lr[:, :20, :] = NODATA
        dataset.write_cache(entry["sample_id"], lr=lr)

    summary = dataset.build_index(tmp_path / "manifest.csv")

    assert summary["accepted"] == 1
    assert summary["rejected"] == 2
    assert summary["rejected_by_reason"] == {"nodata_fraction": 2}
    assert "nodata_fraction" in format_index_summary(summary)
