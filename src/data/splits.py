"""Geographic train/val/test splitting.

Why this exists
---------------
SEN2NAIPv2's ``crosssensor`` subset ships **no** validation or test split --
MEASURED: all 8000 rows carry ``tortilla:data_split == "train"``. So we cut our
own, and it must be geographic.

A random split would leak. Adjacent NAIP quarter-quads overlap, and multiple
records are drawn from the same scene, so a random assignment puts near-identical
HR pixels in both train and validation. The model then scores well on validation
by having memorised those pixels during training, and every number in the report
is inflated. Super-resolution is especially vulnerable: the task *is* recovering
high-frequency detail, which is exactly what leaks.

The guarantee
-------------
:func:`geographic_split` guarantees that **no two samples assigned to different
splits are within ``min_separation_km`` of each other**. It achieves this by
grouping samples transitively -- if A is near B and B is near C, all three land
in the same split even when A and C are far apart -- and then assigning whole
groups. The guarantee is re-checked by :func:`verify_separation`, which is
called by the split script and asserted in the tests, so it is enforced rather
than merely intended.

Distances are great-circle on a spherical Earth, computed via a KD-tree over
unit-sphere Cartesian coordinates. That is exact enough at these scales (the
spherical approximation errs by <0.5% against WGS84) and avoids any projection
or UTM-zone handling.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "parse_wkt_point",
    "scene_group_key",
    "group_samples",
    "geographic_split",
    "verify_separation",
    "split_summary",
    "EARTH_RADIUS_KM",
]

EARTH_RADIUS_KM = 6371.0088

# "POINT (-123.482558 39.696797)" -> (lon, lat). MEASURED format of the
# stac:centroid column in SEN2NAIPv2.
_WKT_POINT = re.compile(
    r"^\s*POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)\s*$",
    re.IGNORECASE,
)

# "NA5120_E1183N0757__m_3912321_nw_10_060_20220710" -> "m_3912321_nw".
# The m_XXXXXXX_qq portion is the NAIP quarter-quad identifier; every record
# sharing it is cut from the same source scene and therefore overlaps.
_NAIP_QUAD = re.compile(r"(m_\d+_[a-z]{2})", re.IGNORECASE)


def parse_wkt_point(value: Any) -> Optional[Tuple[float, float]]:
    """Parse a WKT POINT into ``(lon, lat)`` degrees.

    Args:
        value: A WKT string such as ``"POINT (-123.482558 39.696797)"``, or
            something already shaped like a 2-sequence of numbers.

    Returns:
        ``(lon, lat)`` in degrees, or None when ``value`` is None/unparseable.
        WKT is x-then-y, i.e. **longitude first**, which is the opposite of the
        (lat, lon) convention most mapping APIs use -- getting this backwards
        silently places North America in the Indian Ocean, so it is asserted in
        the tests.

    Raises:
        ValueError: The parsed coordinates are outside valid lon/lat ranges.
    """
    if value is None:
        return None

    if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
        lon, lat = float(value[0]), float(value[1])
    else:
        match = _WKT_POINT.match(str(value))
        if match is None:
            return None
        lon, lat = float(match.group(1)), float(match.group(2))

    if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
        raise ValueError(
            f"Centroid ({lon}, {lat}) is out of range. WKT POINT is "
            "(longitude latitude) -- if latitude came first these values are "
            "swapped."
        )
    return lon, lat


def scene_group_key(sample_id: Any) -> Optional[str]:
    """Extract the NAIP quarter-quad id from a sample id.

    Records sharing a quarter-quad are cut from the same aerial scene and
    overlap regardless of how far apart their centroids are, so they must never
    be split apart.

    Args:
        sample_id: e.g. ``"NA5120_E1183N0757__m_3912321_nw_10_060_20220710"``.

    Returns:
        e.g. ``"m_3912321_nw"``, or None when the id does not carry one.
    """
    if sample_id is None:
        return None
    match = _NAIP_QUAD.search(str(sample_id))
    return match.group(1).lower() if match else None


def _unit_sphere_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Convert lon/lat degrees to Cartesian coordinates on the unit sphere."""
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.column_stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)])


def _chord_for_arc(distance_km: float) -> float:
    """Unit-sphere chord length corresponding to a great-circle arc."""
    return 2.0 * np.sin(distance_km / (2.0 * EARTH_RADIUS_KM))


def _arc_for_chord(chord: float) -> float:
    """Great-circle km corresponding to a unit-sphere chord length."""
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


class _UnionFind:
    """Disjoint-set with path compression and union by size."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def group_samples(
    lons: Sequence[float],
    lats: Sequence[float],
    min_separation_km: float,
    scene_keys: Optional[Sequence[Optional[str]]] = None,
) -> np.ndarray:
    """Group samples so that nearby or same-scene samples share a group.

    Grouping is transitive (single-linkage): if A is within
    ``min_separation_km`` of B and B of C, all three share a group even if A and
    C are far apart. That transitivity is what makes the cross-split distance
    guarantee hold.

    Args:
        lons: Longitudes in degrees, length N.
        lats: Latitudes in degrees, length N.
        min_separation_km: Samples closer than this must share a group.
        scene_keys: Optional per-sample scene id (see :func:`scene_group_key`).
            Samples sharing a non-None key are grouped regardless of distance.

    Returns:
        ``int64`` array of length N of group ids, renumbered ``0..n_groups-1``
        in order of first appearance so the output is deterministic.

    Raises:
        ValueError: Inputs have mismatched lengths, or ``min_separation_km`` is
            negative.
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    if lons.shape != lats.shape:
        raise ValueError(
            f"lons and lats must be the same length, got {lons.shape} and "
            f"{lats.shape}."
        )
    if min_separation_km < 0:
        raise ValueError(f"min_separation_km must be >= 0, got {min_separation_km}.")

    n = len(lons)
    uf = _UnionFind(n)

    if scene_keys is not None:
        if len(scene_keys) != n:
            raise ValueError(
                f"scene_keys has length {len(scene_keys)}, expected {n}."
            )
        first_of_key: Dict[str, int] = {}
        for idx, key in enumerate(scene_keys):
            if key is None:
                continue
            if key in first_of_key:
                uf.union(first_of_key[key], idx)
            else:
                first_of_key[key] = idx

    if n > 1 and min_separation_km > 0:
        from scipy.spatial import cKDTree

        tree = cKDTree(_unit_sphere_xyz(lons, lats))
        pairs = tree.query_pairs(_chord_for_arc(min_separation_km), output_type="ndarray")
        for a, b in pairs:
            uf.union(int(a), int(b))

    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    _, group_ids = np.unique(roots, return_inverse=True)
    # Renumber by first appearance for stable, readable ids.
    order = {}
    out = np.empty(n, dtype=np.int64)
    for i, g in enumerate(group_ids):
        if g not in order:
            order[g] = len(order)
        out[i] = order[g]
    return out


def geographic_split(
    records: Sequence[Mapping[str, Any]],
    fractions: Mapping[str, float],
    min_separation_km: float = 5.0,
    seed: int = 42,
    group_by_scene: bool = True,
    centroid_key: str = "centroid_lonlat",
    id_key: str = "sample_id",
) -> Dict[str, Any]:
    """Assign records to splits so that no split shares a neighbourhood.

    Whole groups are assigned, largest first, each to whichever split is
    furthest below its target count. Largest-first matters: assigning a big
    group late would overshoot badly, so the greedy order is what keeps the
    realised fractions close to the requested ones. Ties are broken by a
    seeded shuffle, so the split is reproducible from ``seed`` but not
    correlated with catalog order.

    Args:
        records: Per-sample mappings. Each must carry ``centroid_key`` as a
            ``(lon, lat)`` pair or a WKT POINT string, and ``id_key``.
        fractions: Split name -> target fraction, e.g.
            ``{"train": 0.8, "val": 0.1, "test": 0.1}``. Must sum to ~1.
        min_separation_km: The separation guarantee. Patches are ~1.3 km across
            (130 px at 10 m), so anything above ~3 km comfortably excludes
            overlap; 5 km is the default.
        seed: Reproducibility.
        group_by_scene: Also group by NAIP quarter-quad id parsed from
            ``id_key``.
        centroid_key: Key holding the centroid.
        id_key: Key holding the sample id.

    Returns:
        A dict with ``assignments`` (list of split names, parallel to
        ``records``), ``group_ids`` (list of int), ``counts``, ``fractions``
        realised, and ``n_groups``.

    Raises:
        ValueError: ``fractions`` is empty, has non-positive values, or does not
            sum to 1; or a record is missing a parseable centroid.
    """
    if not fractions:
        raise ValueError("fractions must be a non-empty mapping of split -> fraction.")
    if any(v <= 0 for v in fractions.values()):
        raise ValueError(f"All fractions must be > 0, got {dict(fractions)}.")
    total_fraction = sum(fractions.values())
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError(
            f"fractions must sum to 1.0, got {total_fraction} from {dict(fractions)}."
        )

    n = len(records)
    if n == 0:
        raise ValueError("No records to split.")

    lons, lats = [], []
    for i, record in enumerate(records):
        point = parse_wkt_point(record.get(centroid_key))
        if point is None:
            raise ValueError(
                f"Record {i} ({record.get(id_key)!r}) has no parseable "
                f"{centroid_key!r}: {record.get(centroid_key)!r}. A geographic "
                "split cannot be made without centroids -- do not fall back to "
                "a random split, it leaks."
            )
        lons.append(point[0])
        lats.append(point[1])

    scene_keys = (
        [scene_group_key(r.get(id_key)) for r in records] if group_by_scene else None
    )
    group_ids = group_samples(lons, lats, min_separation_km, scene_keys)

    n_groups = int(group_ids.max()) + 1
    members: List[List[int]] = [[] for _ in range(n_groups)]
    for idx, g in enumerate(group_ids):
        members[int(g)].append(idx)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_groups)
    # Stable sort by descending size, with the shuffle deciding ties.
    ordered = sorted(shuffled, key=lambda g: -len(members[g]))

    names = list(fractions)
    targets = {name: fractions[name] * n for name in names}
    counts = {name: 0 for name in names}
    assignments: List[Optional[str]] = [None] * n

    for g in ordered:
        # Whichever split is furthest below target takes the next group.
        name = max(names, key=lambda s: (targets[s] - counts[s], s))
        for idx in members[g]:
            assignments[idx] = name
        counts[name] += len(members[g])

    return {
        "assignments": assignments,
        "group_ids": [int(g) for g in group_ids],
        "counts": counts,
        "fractions": {k: v / n for k, v in counts.items()},
        "n_groups": n_groups,
        "n_samples": n,
        "min_separation_km": min_separation_km,
    }


def verify_separation(
    records: Sequence[Mapping[str, Any]],
    assignments: Sequence[str],
    min_separation_km: float,
    centroid_key: str = "centroid_lonlat",
) -> Dict[str, Any]:
    """Check that no two differently-split samples are too close.

    This re-derives the property from the coordinates rather than trusting the
    grouping that produced it, so it is a genuine check and not a restatement.

    Args:
        records: As passed to :func:`geographic_split`.
        assignments: Split name per record.
        min_separation_km: The separation that must hold.
        centroid_key: Key holding the centroid.

    Returns:
        ``{"ok": bool, "min_cross_split_km": float, "violations": [...]}``.
        ``min_cross_split_km`` is ``inf`` when there is only one split.

    Raises:
        ValueError: Lengths mismatch.
    """
    if len(records) != len(assignments):
        raise ValueError(
            f"{len(records)} records but {len(assignments)} assignments."
        )

    from scipy.spatial import cKDTree

    points = []
    for record in records:
        point = parse_wkt_point(record.get(centroid_key))
        if point is None:
            raise ValueError(f"Record {record!r} has no parseable centroid.")
        points.append(point)

    lons = np.array([p[0] for p in points])
    lats = np.array([p[1] for p in points])
    xyz = _unit_sphere_xyz(lons, lats)
    labels = np.asarray(assignments)

    tree = cKDTree(xyz)
    radius = _chord_for_arc(min_separation_km)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    violations = [
        (int(a), int(b), float(_arc_for_chord(np.linalg.norm(xyz[a] - xyz[b]))))
        for a, b in pairs
        if labels[a] != labels[b]
    ]

    # Smallest distance between any two different splits, for reporting.
    min_cross = float("inf")
    unique = sorted(set(labels.tolist()))
    for i, left in enumerate(unique):
        for right in unique[i + 1 :]:
            a_pts, b_pts = xyz[labels == left], xyz[labels == right]
            if len(a_pts) == 0 or len(b_pts) == 0:
                continue
            distances, _ = cKDTree(a_pts).query(b_pts, k=1)
            min_cross = min(min_cross, float(_arc_for_chord(distances.min())))

    return {
        "ok": not violations,
        "min_cross_split_km": min_cross,
        "violations": violations[:20],
        "n_violations": len(violations),
    }


def split_summary(result: Mapping[str, Any]) -> str:
    """Render a human-readable summary of a split result."""
    lines = [
        f"samples: {result['n_samples']}   groups: {result['n_groups']}   "
        f"min_separation: {result['min_separation_km']} km",
    ]
    for name, count in result["counts"].items():
        realised = result["fractions"][name]
        lines.append(f"  {name:<8} {count:>6}  ({realised:.1%})")
    return "\n".join(lines)
