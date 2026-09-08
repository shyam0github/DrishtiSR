"""Extending a split without moving the held-out one. CPU-only, no data.

The property under test is the one Day 2's results depend on: adding training
data must not change which tiles are in val or test, and must not put a training
tile in a validation tile's neighbourhood.
"""

from __future__ import annotations

import pytest

from src.data.split_extend import EXCLUDED, extend_split, format_extension_summary
from src.data.splits import verify_separation

# Roughly 111 km per degree of latitude, so 0.1 deg ~ 11 km and 0.01 deg ~ 1.1 km.
# Every fixture below is placed with the 5 km guarantee in mind.
FAR = 1.0
NEAR = 0.01


def _record(sample_id, lon, lat):
    return {"sample_id": sample_id, "centroid_lonlat": (lon, lat)}


def _base():
    """Three well-separated samples, one per split."""
    return [
        _record("m_1000001_nw_train", -120.0, 38.0),
        _record("m_2000002_ne_val", -110.0, 38.0),
        _record("m_3000003_sw_test", -100.0, 38.0),
    ], {
        "m_1000001_nw_train": "train",
        "m_2000002_ne_val": "val",
        "m_3000003_sw_test": "test",
    }


def _extend(records, existing, **kwargs):
    defaults = dict(min_separation_km=5.0, train_split="train",
                    held_out_splits=("val", "test"))
    defaults.update(kwargs)
    return extend_split(records, existing, **defaults)


# -- the guarantee ---------------------------------------------------------


def test_existing_assignments_are_copied_not_recomputed():
    records, existing = _base()
    result = _extend(records, existing)
    assert result["assignments"] == ["train", "val", "test"]
    assert result["n_frozen"] == 3
    assert result["n_new"] == 0


def test_a_distant_new_sample_joins_train():
    records, existing = _base()
    records.append(_record("m_4000004_se_new", -90.0, 38.0))
    result = _extend(records, existing)
    assert result["assignments"][-1] == "train"
    assert result["new_to_train"] == 1
    assert result["n_new"] == 1


def test_a_new_sample_near_val_is_excluded_not_added_to_train():
    """The leak this module exists to prevent. Train on a tile 1 km from a
    validation tile and the validation score is inflated invisibly."""
    records, existing = _base()
    records.append(_record("m_4000004_se_new", -110.0 + NEAR, 38.0))
    result = _extend(records, existing)
    assert result["assignments"][-1] == EXCLUDED
    assert result["new_excluded_near"] == ["m_4000004_se_new"]
    assert result["new_to_train"] == 0


def test_a_new_sample_near_test_is_also_excluded():
    records, existing = _base()
    records.append(_record("m_4000004_se_new", -100.0 + NEAR, 38.0))
    result = _extend(records, existing)
    assert result["assignments"][-1] == EXCLUDED


def test_a_new_sample_near_an_existing_train_sample_is_fine():
    """Proximity within train is not a leak -- it is just correlated data."""
    records, existing = _base()
    records.append(_record("m_4000004_se_new", -120.0 + NEAR, 38.0))
    result = _extend(records, existing)
    assert result["assignments"][-1] == "train"


def test_a_new_sample_sharing_a_naip_quad_with_val_is_excluded_at_any_distance():
    """Crops of one aerial scene overlap however far apart their centroids are,
    so distance alone would not catch this."""
    records, existing = _base()
    records.append(_record("NA_other__m_2000002_ne_20220101", -60.0, 10.0))
    result = _extend(records, existing)
    assert result["assignments"][-1] == EXCLUDED
    assert result["new_excluded_scene"] == ["NA_other__m_2000002_ne_20220101"]
    assert result["new_excluded_near"] == []


def test_a_new_sample_sharing_a_quad_with_train_is_allowed():
    records, existing = _base()
    records.append(_record("NA_other__m_1000001_nw_20220101", -60.0, 10.0))
    result = _extend(records, existing)
    assert result["assignments"][-1] == "train"


def test_the_separation_guarantee_holds_over_the_loaded_splits():
    """Re-derived from coordinates, not restated from the grouping."""
    records, existing = _base()
    records += [
        _record("m_4000004_se_a", -110.0 + NEAR, 38.0),   # near val -> excluded
        _record("m_5000005_se_b", -90.0, 38.0),           # far -> train
    ]
    result = _extend(records, existing)
    live = [
        (r, s) for r, s in zip(records, result["assignments"]) if s != EXCLUDED
    ]
    check = verify_separation(
        [r for r, _ in live], [s for _, s in live], min_separation_km=5.0
    )
    assert check["ok"], check["violations"]


# -- the hard gate ---------------------------------------------------------


def test_a_missing_held_out_sample_raises_naming_the_shrinking_catalog():
    """The split file names a test id the catalog no longer offers. That IS a
    membership change, but the useful diagnosis is the cause -- a catalog that
    got smaller -- so the message must say so rather than blaming the split."""
    records, existing = _base()
    records = [r for r in records if r["sample_id"] != "m_3000003_sw_test"]
    with pytest.raises(ValueError, match="catalog got SMALLER"):
        _extend(records, existing)


def test_a_missing_train_sample_is_reported_rather_than_raised():
    """A vanished TRAINING id costs nothing comparable, so it is returned for
    the caller to judge instead of aborting the extension outright."""
    records, existing = _base()
    records = [r for r in records if r["sample_id"] != "m_1000001_nw_train"]
    result = _extend(records, existing)
    assert result["dropped_from_catalog"] == ["m_1000001_nw_train"]
    assert result["counts"]["val"] == 1


def test_held_out_membership_survives_a_large_extension_id_for_id():
    """The property Day 2's numbers rest on, checked as a set identity rather
    than as a count. 200 new samples scattered around the val tile must leave
    the val set exactly as it was.

    Note the gate inside extend_split that raises on a membership change is
    UNREACHABLE by construction -- existing assignments are copied, never
    recomputed, so nothing in the current code path can move a val id. It is
    kept as a trip-wire for whoever later adds re-balancing, and this test
    verifies the property it guards rather than the raise itself."""
    records, existing = _base()
    for i in range(200):
        records.append(_record(f"m_9{i:06d}_nw_new", -110.0 + 0.001 * i, 38.0))
    result = _extend(records, existing)
    assert result["held_out_ids"]["val"] == ["m_2000002_ne_val"]
    assert result["held_out_ids"]["test"] == ["m_3000003_sw_test"]
    assert result["counts"]["val"] == 1
    assert result["counts"]["test"] == 1
    # And the near ones really were refused, so this is not a vacuous pass.
    assert len(result["new_excluded_near"]) > 0
    assert result["new_to_train"] > 0


def test_an_unknown_split_name_raises():
    records, existing = _base()
    existing["m_1000001_nw_train"] = "holdout"
    with pytest.raises(ValueError, match="not among"):
        _extend(records, existing)


def test_a_record_without_a_centroid_raises():
    records, existing = _base()
    records.append({"sample_id": "m_4000004_se_new", "centroid_lonlat": None})
    with pytest.raises(ValueError, match="no parseable"):
        _extend(records, existing)


def test_excluded_is_not_a_split_any_loader_selects():
    """The name matters: it must not collide with train/val/test, or the
    excluded samples would silently be trained on."""
    assert EXCLUDED not in {"train", "val", "test"}


# -- edge cases ------------------------------------------------------------


def test_extending_with_no_held_out_samples_yet_still_works():
    """A split file that is all train (an early partial cut) must not crash the
    KD-tree construction."""
    records = [_record("m_1000001_nw_a", -120.0, 38.0),
               _record("m_2000002_ne_b", -110.0, 38.0)]
    existing = {"m_1000001_nw_a": "train"}
    result = _extend(records, existing)
    assert result["assignments"] == ["train", "train"]


def test_extending_an_empty_split_file_puts_everything_in_train():
    records, _ = _base()
    result = _extend(records, {})
    assert set(result["assignments"]) == {"train"}
    assert result["n_new"] == 3


def test_counts_cover_every_assigned_split():
    records, existing = _base()
    records += [
        _record("m_4000004_se_a", -110.0 + NEAR, 38.0),
        _record("m_5000005_se_b", -90.0, 38.0),
    ]
    result = _extend(records, existing)
    assert sum(result["counts"].values()) == len(records)
    assert result["counts"]["val"] == 1
    assert result["counts"]["test"] == 1
    assert result["counts"]["train"] == 2
    assert result["counts"][EXCLUDED] == 1


def test_the_summary_reports_zero_exclusions_rather_than_omitting_them():
    records, existing = _base()
    records.append(_record("m_4000004_se_new", -90.0, 38.0))
    block = format_extension_summary(_extend(records, existing))
    assert "new -> train                  1" in block
    assert "shared scene)  0" in block


def test_wkt_centroids_are_accepted_as_well_as_pairs():
    records = [
        {"sample_id": "m_1000001_nw_a", "centroid_lonlat": "POINT (-120.0 38.0)"},
        {"sample_id": "m_2000002_ne_b", "centroid_lonlat": "POINT (-110.0 38.0)"},
    ]
    result = _extend(records, {"m_1000001_nw_a": "val"})
    assert result["assignments"] == ["val", "train"]


def test_separation_threshold_is_honoured_exactly():
    """A sample just outside the radius joins train; just inside does not."""
    # 5 km at this latitude is ~0.045 deg of longitude at 38N
    # (111.32 * cos(38) = 87.7 km/deg), so 0.08 deg ~ 7 km and 0.02 deg ~ 1.8 km.
    records, existing = _base()
    records += [
        _record("m_4000004_se_out", -110.0 + 0.08, 38.0),
        _record("m_5000005_se_in", -110.0 + 0.02, 38.0),
    ]
    result = _extend(records, existing)
    assert result["assignments"][-2] == "train"
    assert result["assignments"][-1] == EXCLUDED
