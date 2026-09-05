"""Tests for the geographic split.

The property under test is the one that matters: no two samples in different
splits are closer than the configured separation. Everything else -- fraction
accuracy, determinism, group counts -- is secondary to that, because a leaky
split does not fail loudly. It quietly makes the model look better than it is.
"""

import numpy as np
import pytest

from src.data.splits import (
    EARTH_RADIUS_KM,
    geographic_split,
    group_samples,
    parse_wkt_point,
    scene_group_key,
    split_summary,
    verify_separation,
)

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _records(points, prefix="s"):
    return [
        {"sample_id": f"{prefix}{i}", "centroid_lonlat": p} for i, p in enumerate(points)
    ]


def _clustered_points(n_clusters=30, per_cluster=10, spread=0.002, seed=0):
    """Tight clusters far apart -- the shape real NAIP-derived data has."""
    rng = np.random.default_rng(seed)
    points = []
    for c in range(n_clusters):
        lon0 = -123.0 + (c % 6) * 2.0
        lat0 = 38.0 + (c // 6) * 2.0
        for _ in range(per_cluster):
            points.append(
                (lon0 + rng.normal(0, spread), lat0 + rng.normal(0, spread))
            )
    return points


# -- WKT parsing ----------------------------------------------------------


def test_parse_wkt_point_is_lon_lat_not_lat_lon():
    """The measured SEN2NAIPv2 centroid format. Swapping these hides the bug."""
    lon, lat = parse_wkt_point("POINT (-123.482558 39.696797)")
    assert lon == pytest.approx(-123.482558)
    assert lat == pytest.approx(39.696797)


@pytest.mark.parametrize(
    "text", ["POINT(-1 2)", "point (-1 2)", "  POINT ( -1.5  2.5 )  "]
)
def test_parse_wkt_point_tolerates_formatting(text):
    assert parse_wkt_point(text) is not None


def test_parse_wkt_point_accepts_pairs():
    assert parse_wkt_point((10.0, 20.0)) == (10.0, 20.0)


def test_parse_wkt_point_returns_none_for_unparseable():
    assert parse_wkt_point(None) is None
    assert parse_wkt_point("not a point") is None


def test_parse_wkt_point_rejects_swapped_coordinates():
    """A latitude of 123 is impossible and must not pass silently."""
    with pytest.raises(ValueError):
        parse_wkt_point("POINT (39.69 -123.48)".replace("39.69", "191.0"))


# -- scene keys -----------------------------------------------------------


def test_scene_group_key_extracts_naip_quad():
    assert (
        scene_group_key("NA5120_E1183N0757__m_3912321_nw_10_060_20220710")
        == "m_3912321_nw"
    )


def test_scene_group_key_none_when_absent():
    assert scene_group_key("stub_0000") is None
    assert scene_group_key(None) is None


# -- grouping -------------------------------------------------------------


def test_nearby_points_share_a_group():
    points = [(0.0, 0.0), (0.001, 0.001), (10.0, 10.0)]
    groups = group_samples(*zip(*points), min_separation_km=5.0)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_grouping_is_transitive():
    """A-B close, B-C close, A-C far: all three must still share a group."""
    # ~3 km apart in latitude each step, threshold 5 km.
    step = 3.0 / 111.32
    points = [(0.0, 0.0), (0.0, step), (0.0, 2 * step)]
    groups = group_samples(*zip(*points), min_separation_km=5.0)
    assert len(set(groups.tolist())) == 1


def test_scene_key_groups_regardless_of_distance():
    points = [(0.0, 0.0), (50.0, 50.0)]
    groups = group_samples(
        *zip(*points), min_separation_km=1.0, scene_keys=["m_1_nw", "m_1_nw"]
    )
    assert groups[0] == groups[1]


def test_zero_separation_makes_every_point_its_own_group():
    points = [(0.0, 0.0), (0.0001, 0.0), (1.0, 1.0)]
    groups = group_samples(*zip(*points), min_separation_km=0.0)
    assert len(set(groups.tolist())) == 3


def test_group_samples_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        group_samples([0.0, 1.0], [0.0], min_separation_km=1.0)


def test_group_samples_rejects_negative_separation():
    with pytest.raises(ValueError):
        group_samples([0.0], [0.0], min_separation_km=-1.0)


# -- the guarantee --------------------------------------------------------


def test_split_guarantees_minimum_separation():
    records = _records(_clustered_points())
    result = geographic_split(records, FRACTIONS, min_separation_km=5.0, seed=1)
    check = verify_separation(records, result["assignments"], 5.0)
    assert check["ok"], check["violations"]
    assert check["min_cross_split_km"] >= 5.0


@pytest.mark.parametrize("separation", [1.0, 5.0, 25.0, 100.0])
def test_guarantee_holds_across_separations(separation):
    records = _records(_clustered_points())
    result = geographic_split(records, FRACTIONS, min_separation_km=separation, seed=3)
    check = verify_separation(records, result["assignments"], separation)
    assert check["ok"]


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
def test_guarantee_holds_across_seeds(seed):
    records = _records(_clustered_points(seed=seed))
    result = geographic_split(records, FRACTIONS, min_separation_km=5.0, seed=seed)
    assert verify_separation(records, result["assignments"], 5.0)["ok"]


def test_same_scene_never_spans_splits():
    points = [(-123.0 + i * 5.0, 39.0) for i in range(20)]
    records = [
        {
            "sample_id": f"NA_E{i}__m_{1000 + i // 2}_nw_10_060_2022",
            "centroid_lonlat": p,
        }
        for i, p in enumerate(points)
    ]
    result = geographic_split(records, FRACTIONS, min_separation_km=1.0, seed=5)
    by_scene = {}
    for record, split in zip(records, result["assignments"]):
        by_scene.setdefault(scene_group_key(record["sample_id"]), set()).add(split)
    assert all(len(v) == 1 for v in by_scene.values())


def test_verify_separation_detects_a_deliberately_leaky_split():
    """The verifier must actually catch leakage, not just agree with itself."""
    points = [(0.0, 0.0), (0.001, 0.001)]  # ~150 m apart
    records = _records(points)
    leaky = ["train", "val"]
    check = verify_separation(records, leaky, min_separation_km=5.0)
    assert not check["ok"]
    assert check["n_violations"] == 1
    assert check["min_cross_split_km"] < 1.0


# -- assignment quality ---------------------------------------------------


def test_all_samples_are_assigned():
    records = _records(_clustered_points())
    result = geographic_split(records, FRACTIONS, min_separation_km=5.0)
    assert len(result["assignments"]) == len(records)
    assert all(a in FRACTIONS for a in result["assignments"])


def test_realised_fractions_are_close_to_targets():
    records = _records(_clustered_points(n_clusters=60, per_cluster=10))
    result = geographic_split(records, FRACTIONS, min_separation_km=5.0, seed=2)
    for name, target in FRACTIONS.items():
        assert abs(result["fractions"][name] - target) < 0.06, result["fractions"]


def test_every_split_is_non_empty():
    records = _records(_clustered_points())
    result = geographic_split(records, FRACTIONS, min_separation_km=5.0)
    assert all(count > 0 for count in result["counts"].values())


def test_split_is_deterministic_for_a_seed():
    records = _records(_clustered_points())
    a = geographic_split(records, FRACTIONS, min_separation_km=5.0, seed=11)
    b = geographic_split(records, FRACTIONS, min_separation_km=5.0, seed=11)
    assert a["assignments"] == b["assignments"]


def test_different_seeds_give_different_splits():
    records = _records(_clustered_points())
    a = geographic_split(records, FRACTIONS, min_separation_km=5.0, seed=1)
    b = geographic_split(records, FRACTIONS, min_separation_km=5.0, seed=99)
    assert a["assignments"] != b["assignments"]


def test_split_does_not_depend_on_record_order():
    """Shuffling the catalog must not change which group a sample belongs to."""
    points = _clustered_points()
    records = _records(points)
    result = geographic_split(records, FRACTIONS, min_separation_km=5.0, seed=4)
    by_id = dict(zip([r["sample_id"] for r in records], result["group_ids"]))

    order = np.random.default_rng(0).permutation(len(records))
    shuffled = [records[i] for i in order]
    shuffled_result = geographic_split(
        shuffled, FRACTIONS, min_separation_km=5.0, seed=4
    )
    groups = {}
    for record, group in zip(shuffled, shuffled_result["group_ids"]):
        groups.setdefault(group, set()).add(record["sample_id"])
    # Same partition, even if group numbering differs.
    original = {}
    for sid, group in by_id.items():
        original.setdefault(group, set()).add(sid)
    assert sorted(map(sorted, groups.values())) == sorted(
        map(sorted, original.values())
    )


# -- input validation -----------------------------------------------------


def test_missing_centroid_raises_rather_than_falling_back():
    records = [{"sample_id": "a", "centroid_lonlat": None}]
    with pytest.raises(ValueError) as excinfo:
        geographic_split(records, FRACTIONS)
    assert "random" in str(excinfo.value).lower()


def test_fractions_must_sum_to_one():
    records = _records([(0.0, 0.0)])
    with pytest.raises(ValueError):
        geographic_split(records, {"train": 0.7, "val": 0.1})


def test_fractions_must_be_positive():
    records = _records([(0.0, 0.0)])
    with pytest.raises(ValueError):
        geographic_split(records, {"train": 1.2, "val": -0.2})


def test_empty_records_raises():
    with pytest.raises(ValueError):
        geographic_split([], FRACTIONS)


def test_verify_separation_rejects_length_mismatch():
    with pytest.raises(ValueError):
        verify_separation(_records([(0.0, 0.0)]), ["train", "val"], 5.0)


# -- distance maths -------------------------------------------------------


def test_one_degree_of_latitude_is_about_111_km():
    """Guards the chord/arc conversion against a factor-of-two slip."""
    groups_close = group_samples([0.0, 0.0], [0.0, 1.0], min_separation_km=112.0)
    groups_far = group_samples([0.0, 0.0], [0.0, 1.0], min_separation_km=110.0)
    assert groups_close[0] == groups_close[1], "1 deg lat should be under 112 km"
    assert groups_far[0] != groups_far[1], "1 deg lat should be over 110 km"


def test_earth_radius_is_sane():
    assert 6350 < EARTH_RADIUS_KM < 6390


def test_split_summary_mentions_each_split():
    records = _records(_clustered_points())
    result = geographic_split(records, FRACTIONS, min_separation_km=5.0)
    text = split_summary(result)
    for name in FRACTIONS:
        assert name in text
