"""The time-boxed, resumable, atomic cache expansion. Offline, CPU-only.

A 45-minute box that is merely *intended* is a promise. These tests make it
enforceable: the budget is exercised with a fake clock, the atomicity is
exercised by killing a write mid-flight, and the resumability is exercised by
running the loop twice and asserting the second run re-downloads nothing.

The dataset is built with ``__new__`` and only the attributes the expansion path
touches, following tests/test_sen2naip_retry.py -- constructing the real thing
needs network access to fetch the taco catalog.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from src.data.cache_manifest import CacheValidationError, read_manifest
from src.data.sen2naip import NATIVE_BANDS, SEN2NAIPv2Dataset

LR_HW = 16  # small tiles keep the tests fast; the 4x contract is what matters
SCALE = 4


def _dataset(tmp_path, n_samples=10, progress_every=25):
    """A SEN2NAIPv2Dataset carrying only what the expansion path reads."""
    obj = SEN2NAIPv2Dataset.__new__(SEN2NAIPv2Dataset)
    obj.subset = "sen2naipv2-crosssensor"
    obj.cache_dir = tmp_path
    obj.logger = logging.getLogger("test.expand")
    obj.scale = SCALE
    obj.cache_dtype = "uint16"
    obj.reflectance_scale = 10000.0
    obj.nodata_value = 65535
    obj.request_delay_s = 0.0
    obj._manifest_index = None
    obj._sub_cfg = {
        "cache_validation": {
            "expected_dtype": "uint16",
            "dn_p999_range": [50.0, 40000.0],
        },
        "expansion": {"progress_every": progress_every},
    }
    obj._catalog = [
        {
            "sample_id": f"NA_{i:04d}__m_{1000000 + i}_nw_10_060_20220101",
            "row_position": i,
            "crs": "EPSG:32610",
            "geotransform": None,
            "data_split": "train",
            "correlation": 0.95,
            "centroid_lonlat": (-120.0 + i, 38.0),
        }
        for i in range(n_samples)
    ]
    return obj


def _arrays(fill=3000, lr_hw=LR_HW, bands=len(NATIVE_BANDS), scale=SCALE):
    lr = np.full((bands, lr_hw, lr_hw), fill, dtype=np.uint16)
    hr = np.full((bands, lr_hw * scale, lr_hw * scale), fill, dtype=np.uint16)
    return {"lr": lr, "hr": hr}, {"lr": {}, "hr": {}}


def _good_fetch(dataset, **kwargs):
    """Wire a fetch that always returns a valid pair, and count the calls."""
    calls = []

    def fetch(idx):
        calls.append(idx)
        return _arrays(**kwargs)

    dataset._fetch_pair_arrays = fetch
    return calls


# -- the wall clock --------------------------------------------------------


def test_the_budget_stops_the_loop_and_reports_why(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, n_samples=100, progress_every=1000)
    _good_fetch(dataset)

    # A fake clock that advances 10 s per reading, so the 25 s budget expires
    # after a handful of pairs regardless of how fast the machine really is.
    ticks = iter(range(0, 100_000, 10))
    monkeypatch.setattr("src.data.sen2naip.time.monotonic", lambda: next(ticks))

    summary = dataset.expand_cache(time_budget_sec=25)
    assert summary["stopped_because"] == "time_budget"
    assert summary["new"] < 100, "the budget must stop it short of the catalog"
    assert summary["after"] == summary["before"] + summary["new"]


def test_the_pair_in_flight_is_finished_and_committed_before_exiting(tmp_path,
                                                                    monkeypatch):
    """The budget is checked at the TOP of an iteration, so a pair that has
    started always lands whole. A half pair is exactly what must never exist."""
    dataset = _dataset(tmp_path, n_samples=100, progress_every=1000)
    _good_fetch(dataset)
    ticks = iter(range(0, 100_000, 10))
    monkeypatch.setattr("src.data.sen2naip.time.monotonic", lambda: next(ticks))

    summary = dataset.expand_cache(time_budget_sec=25)

    recorded = read_manifest(dataset.manifest_file)
    assert len(recorded) == summary["after"]
    # Every recorded pair has a real file, and no temporaries survive.
    for taco_id in recorded:
        assert (tmp_path / f"{taco_id}.npz").is_file()
        assert (tmp_path / f"{taco_id}.json").is_file()
    assert list(tmp_path.glob("*.tmp")) == []


def test_no_budget_means_the_loop_runs_to_the_end_of_the_catalog(tmp_path):
    dataset = _dataset(tmp_path, n_samples=6)
    _good_fetch(dataset)
    summary = dataset.expand_cache()
    assert summary["stopped_because"] == "catalog_exhausted"
    assert summary["new"] == 6
    assert summary["projected_total_at_rate"] == "unbounded"


def test_target_pairs_is_a_total_not_a_delta(tmp_path):
    """'Get to 4' twice must yield 4, not 8. A resumed run that re-read the
    flag as a delta would overshoot every time."""
    dataset = _dataset(tmp_path, n_samples=10)
    _good_fetch(dataset)
    first = dataset.expand_cache(target_pairs=4)
    assert first["after"] == 4 and first["stopped_because"] == "target"

    second = _dataset(tmp_path, n_samples=10)
    calls = _good_fetch(second)
    result = second.expand_cache(target_pairs=4)
    assert result["after"] == 4
    assert result["new"] == 0
    assert calls == [], "nothing may be re-downloaded to satisfy a met target"


# -- resumability ----------------------------------------------------------


def test_a_second_run_re_downloads_nothing(tmp_path):
    dataset = _dataset(tmp_path, n_samples=5)
    _good_fetch(dataset)
    dataset.expand_cache()

    resumed = _dataset(tmp_path, n_samples=5)
    calls = _good_fetch(resumed)
    summary = resumed.expand_cache()
    assert summary["before"] == 5
    assert summary["new"] == 0
    assert calls == []


def test_a_resumed_run_fetches_only_what_is_missing(tmp_path):
    dataset = _dataset(tmp_path, n_samples=3)
    _good_fetch(dataset)
    dataset.expand_cache()

    grown = _dataset(tmp_path, n_samples=5)
    calls = _good_fetch(grown)
    summary = grown.expand_cache()
    assert summary["before"] == 3
    assert summary["new"] == 2
    assert calls == [3, 4], "only the catalog entries with no cached pair"


def test_pairs_on_disk_without_a_manifest_line_are_adopted_not_refetched(tmp_path):
    """The Day 2 cache has 3002 pairs and no manifest. Re-downloading them to
    obtain one would cost nine hours."""
    dataset = _dataset(tmp_path, n_samples=3)
    _good_fetch(dataset)
    dataset.expand_cache()

    dataset.manifest_file.unlink()

    resumed = _dataset(tmp_path, n_samples=3)
    calls = _good_fetch(resumed)
    summary = resumed.expand_cache()
    assert summary["backfill"]["adopted"] == 3
    assert summary["new"] == 0
    assert calls == []


# -- atomicity -------------------------------------------------------------


def test_a_failure_mid_write_leaves_no_file_under_the_real_name(tmp_path):
    dataset = _dataset(tmp_path, n_samples=1)

    def explode(idx):
        raise OSError("connection reset halfway through")

    dataset._fetch_pair_arrays = explode
    summary = dataset.expand_cache()

    assert summary["new"] == 0
    assert summary["rejects"] == {"fetch_failed": 1}
    assert list(tmp_path.glob("*.npz")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_pair_that_fails_validation_is_never_committed(tmp_path):
    """Written to a temporary, re-read, judged, and discarded. The cache must
    not gain a pair that would have to be filtered out later."""
    dataset = _dataset(tmp_path, n_samples=1)
    dataset._fetch_pair_arrays = lambda idx: _arrays(fill=0)  # all-zero tile

    summary = dataset.expand_cache()
    assert summary["new"] == 0
    assert summary["rejects"] == {"all_zero": 1}
    assert list(tmp_path.glob("*.npz")) == []
    assert list(tmp_path.glob("*.tmp")) == []
    assert read_manifest(dataset.manifest_file) == {}


def test_a_pair_at_the_wrong_scale_is_rejected_by_reason(tmp_path):
    dataset = _dataset(tmp_path, n_samples=1)
    dataset._fetch_pair_arrays = lambda idx: _arrays(scale=2)
    summary = dataset.expand_cache()
    assert summary["rejects"] == {"scale_mismatch": 1}


def test_one_bad_pair_does_not_end_a_time_boxed_run(tmp_path):
    """A single unreachable record must not throw away the rest of the box."""
    dataset = _dataset(tmp_path, n_samples=5)

    def sometimes(idx):
        if idx == 2:
            raise OSError("429 dressed up as a missing file")
        return _arrays()

    dataset._fetch_pair_arrays = sometimes
    summary = dataset.expand_cache()
    assert summary["new"] == 4
    assert summary["rejects"] == {"fetch_failed": 1}
    assert summary["stopped_because"] == "catalog_exhausted"


def test_a_rejected_pair_is_retried_on_the_next_run(tmp_path):
    """Rejections are not remembered. A pair refused because the network
    truncated it must get another chance; only a COMMITTED pair is skipped."""
    dataset = _dataset(tmp_path, n_samples=1)
    dataset._fetch_pair_arrays = lambda idx: _arrays(fill=0)
    assert dataset.expand_cache()["new"] == 0

    retried = _dataset(tmp_path, n_samples=1)
    _good_fetch(retried)
    assert retried.expand_cache()["new"] == 1


# -- the committed artefacts ----------------------------------------------


def test_the_committed_pair_round_trips_with_the_right_shapes(tmp_path):
    dataset = _dataset(tmp_path, n_samples=1)
    _good_fetch(dataset)
    dataset.expand_cache()

    taco_id = dataset._catalog[0]["sample_id"]
    with np.load(tmp_path / f"{taco_id}.npz") as handle:
        assert handle["lr"].shape == (4, LR_HW, LR_HW)
        assert handle["hr"].shape == (4, LR_HW * SCALE, LR_HW * SCALE)
        assert handle["lr"].dtype == np.uint16


def test_the_manifest_record_carries_the_scaling_facts(tmp_path):
    dataset = _dataset(tmp_path, n_samples=1)
    _good_fetch(dataset)
    dataset.expand_cache()

    record = next(iter(read_manifest(dataset.manifest_file).values()))
    assert record["bands"] == list(NATIVE_BANDS)
    assert record["dtype"] == "uint16"
    assert record["reflectance_scale"] == 10000.0
    assert record["nodata_value"] == 65535
    assert record["validated"] == "full"
    assert record["lr_shape"] == [4, LR_HW, LR_HW]
    assert record["hr_shape"] == [4, LR_HW * SCALE, LR_HW * SCALE]


def test_the_sidecar_json_is_written_beside_the_pair(tmp_path):
    dataset = _dataset(tmp_path, n_samples=1)
    _good_fetch(dataset)
    dataset.expand_cache()
    taco_id = dataset._catalog[0]["sample_id"]
    sidecar = json.loads((tmp_path / f"{taco_id}.json").read_text(encoding="utf-8"))
    assert sidecar["sample_id"] == taco_id
    assert "profile" in sidecar


def test_backfilled_records_are_marked_structural_not_full(tmp_path):
    """A backfilled pair was never pixel-checked. Recording it as fully
    validated would overstate what is known about it."""
    dataset = _dataset(tmp_path, n_samples=1)
    _good_fetch(dataset)
    dataset.expand_cache()
    dataset.manifest_file.unlink()

    resumed = _dataset(tmp_path, n_samples=1)
    resumed.backfill_manifest()
    record = next(iter(read_manifest(resumed.manifest_file).values()))
    assert record["validated"] == "structural"
    assert record["lr_dn_p999"] is None


def test_a_truncated_npz_is_not_adopted_and_is_refetched(tmp_path):
    dataset = _dataset(tmp_path, n_samples=1)
    _good_fetch(dataset)
    dataset.expand_cache()
    taco_id = dataset._catalog[0]["sample_id"]
    npz = tmp_path / f"{taco_id}.npz"
    dataset.manifest_file.unlink()
    npz.write_bytes(npz.read_bytes()[: len(npz.read_bytes()) // 2])

    resumed = _dataset(tmp_path, n_samples=1)
    calls = _good_fetch(resumed)
    summary = resumed.expand_cache()
    assert summary["backfill"]["unreadable"] == 1
    assert summary["backfill"]["adopted"] == 0
    assert calls == [0], "the unreadable pair must be re-fetched"
    assert summary["new"] == 1


def test_a_structurally_wrong_cached_pair_is_not_adopted(tmp_path):
    """A pair whose HR is not 4x LR is left on disk, counted, and kept out of
    the manifest -- not deleted on the strength of a header read."""
    dataset = _dataset(tmp_path, n_samples=1)
    taco_id = dataset._catalog[0]["sample_id"]
    with (tmp_path / f"{taco_id}.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            lr=np.full((4, LR_HW, LR_HW), 3000, np.uint16),
            hr=np.full((4, LR_HW * 2, LR_HW * 2), 3000, np.uint16),
        )
    counts = dataset.backfill_manifest()
    assert counts["structural_reject"] == 1
    assert counts["adopted"] == 0
    assert (tmp_path / f"{taco_id}.npz").is_file()


# -- reporting -------------------------------------------------------------


def test_progress_is_printed_every_n_pairs(tmp_path, capsys):
    dataset = _dataset(tmp_path, n_samples=10, progress_every=5)
    _good_fetch(dataset)
    dataset.expand_cache()
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[expand]") and "cached" in line and "start" not in line
    ]
    assert len(lines) == 2, lines
    assert "elapsed" in lines[0] and "pairs/s" in lines[0]
    assert "projected total" in lines[0]


def test_the_summary_counts_add_up(tmp_path):
    dataset = _dataset(tmp_path, n_samples=8)

    def sometimes(idx):
        if idx in (1, 5):
            raise OSError("boom")
        return _arrays()

    dataset._fetch_pair_arrays = sometimes
    summary = dataset.expand_cache()
    assert summary["before"] == 0
    assert summary["new"] == 6
    assert summary["after"] == 6
    assert sum(summary["rejects"].values()) == 2
    assert summary["catalog_size"] == 8


def test_is_cached_uses_the_manifest_and_survives_a_deleted_file(tmp_path):
    dataset = _dataset(tmp_path, n_samples=2)
    _good_fetch(dataset)
    dataset.expand_cache()
    assert dataset.is_cached(0) is True

    (tmp_path / f"{dataset._catalog[0]['sample_id']}.npz").unlink()
    assert dataset.is_cached(0) is False, (
        "a manifest line must not outvote the file being gone"
    )
