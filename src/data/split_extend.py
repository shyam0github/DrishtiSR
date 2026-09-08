"""Grow the training split without moving the held-out one.

The problem
-----------
Day 2's numbers -- the bicubic floor, Run A's PSNR/SSIM -- are all measured on
one specific set of 300 validation tiles. Adding samples to the dataset and
re-running :func:`~src.data.splits.geographic_split` would reassign *everything*:
the greedy largest-first pass sees different group sizes, so tiles move between
splits. Every Day 2 number would then describe a validation set that no longer
exists, and Run A would have to leave the results table.

So the split is not recomputed. It is EXTENDED.

The rule
--------
1. Every sample already in the split file keeps its assignment, verbatim. Not
   recomputed and compared -- copied. The val and test sets are frozen sets of
   ids, and :func:`extend_split` asserts their membership is unchanged before it
   returns.
2. A new sample joins ``train`` only if it is safe to. Otherwise it is assigned
   ``excluded`` and never loaded by anything.

Why new samples cannot simply be dumped into train
--------------------------------------------------
The separation guarantee is symmetric. A new tile 2 km from a *validation* tile
leaks just as badly whether it arrived first or last: train it on and the model
has seen the validation neighbourhood. The leak is invisible in training loss
and shows up only as a validation score that is too good.

A new sample is therefore refused from train when EITHER holds:

- it is within ``min_separation_km`` of any val or test sample, or
- it shares a NAIP quarter-quad (:func:`~src.data.splits.scene_group_key`) with
  any val or test sample -- overlapping crops of one aerial scene, at any
  distance.

Refused samples get the split name ``excluded``. That name is deliberate: it is
not one of ``cfg.loader.train_split`` / ``val_split`` / ``test_split``, so
:func:`~src.data.loader.select_indices` never selects it, while
:func:`~src.data.loader.resolve_split_assignments` -- which raises when a sample
is missing from the split file entirely -- still sees full coverage.

Transitivity, and what this deliberately does NOT do
----------------------------------------------------
:func:`~src.data.splits.geographic_split` groups transitively: if A is near B and
B near C, all three share a split even when A and C are far apart. Extension
cannot offer that, because the existing assignments are frozen -- a new sample
bridging a train group and a val group cannot merge them. It is excluded
instead, which is the conservative resolution and preserves the guarantee that
matters: **no train sample is within ``min_separation_km`` of a val or test
sample.** :func:`~src.data.splits.verify_separation` re-derives exactly that
from the coordinates, and the caller must run it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from src.data.splits import (
    EARTH_RADIUS_KM,
    _unit_sphere_xyz,
    parse_wkt_point,
    scene_group_key,
)

__all__ = ["EXCLUDED", "extend_split", "format_extension_summary"]

# The split name given to a new sample that cannot safely join train. Not a
# split any loader selects; see the module docstring.
EXCLUDED = "excluded"


def _chord_for_arc(distance_km: float) -> float:
    """Unit-sphere chord length corresponding to a great-circle arc."""
    return 2.0 * np.sin(distance_km / (2.0 * EARTH_RADIUS_KM))


def extend_split(
    records: Sequence[Mapping[str, Any]],
    existing: Mapping[str, str],
    *,
    min_separation_km: float,
    train_split: str = "train",
    held_out_splits: Sequence[str] = ("val", "test"),
    centroid_key: str = "centroid_lonlat",
    id_key: str = "sample_id",
) -> Dict[str, Any]:
    """Assign new samples to train, freezing every existing assignment.

    Args:
        records: Every sample the extended split must cover -- the existing ones
            and the new ones together, in catalog order. Each needs ``id_key``
            and a parseable ``centroid_key`` (a ``(lon, lat)`` pair or a WKT
            POINT).
        existing: ``sample_id -> split`` as read from the current split CSV.
            Ids here that are absent from ``records`` are reported in
            ``dropped_from_catalog`` and otherwise ignored.
        min_separation_km: The separation guarantee, ``cfg.splits.min_separation_km``.
        train_split: The split new samples may join.
        held_out_splits: The splits whose membership must not change and whose
            neighbourhoods new samples must stay out of.
        centroid_key: Key holding the centroid.
        id_key: Key holding the sample id.

    Returns:
        ``assignments`` (list of split names parallel to ``records``),
        ``counts`` (split -> count), ``n_new``, ``n_frozen``,
        ``new_to_train``, ``new_excluded_near`` (too close to a held-out
        sample), ``new_excluded_scene`` (shares a NAIP quarter-quad with one),
        ``dropped_from_catalog`` (ids in ``existing`` that the catalog no longer
        offers; only ever training ids, since a missing held-out id raises), and
        ``held_out_ids`` (split -> sorted ids, so the caller can assert identity
        against the previous file).

    Raises:
        ValueError: A record has no parseable centroid; an existing assignment
            names a split not in ``{train_split} | held_out_splits |
            {EXCLUDED}``; the catalog no longer offers a held-out sample; or --
            the check this module exists for -- a held-out split's membership
            would change.
    """
    known_splits = {train_split, EXCLUDED, *held_out_splits}
    unknown = sorted(set(existing.values()) - known_splits)
    if unknown:
        raise ValueError(
            f"The existing split file uses split name(s) {unknown}, which are "
            f"not among {sorted(known_splits)}. Extending it would silently "
            "drop those samples from every split. Fix the names or widen "
            "held_out_splits."
        )

    ids: List[str] = []
    points: List[Tuple[float, float]] = []
    for position, record in enumerate(records):
        point = parse_wkt_point(record.get(centroid_key))
        if point is None:
            raise ValueError(
                f"Record {position} ({record.get(id_key)!r}) has no parseable "
                f"{centroid_key!r}: {record.get(centroid_key)!r}. A sample "
                "whose location is unknown cannot be shown to be far enough "
                "from the validation set, so it cannot be added to train."
            )
        ids.append(str(record[id_key]))
        points.append(point)

    xyz = _unit_sphere_xyz(
        np.array([p[0] for p in points]), np.array([p[1] for p in points])
    )

    # A sample the split file names but the catalog no longer offers. Checked
    # BEFORE the membership gate below, because a held-out id vanishing this way
    # would otherwise surface as "val membership changed" -- true, but it points
    # at the wrong thing. The cause is a catalog that got smaller (num_samples
    # lowered, a filter tightened), and that is what the message must say.
    dropped = sorted(set(existing) - set(ids))
    dropped_held_out = sorted(
        sid for sid in dropped if existing[sid] in set(held_out_splits)
    )
    if dropped_held_out:
        raise ValueError(
            f"{len(dropped_held_out)} held-out sample(s) named by the split "
            f"file are absent from the catalog (e.g. {dropped_held_out[:3]}). "
            "The catalog got SMALLER -- num_samples was lowered or a filter "
            "tightened. Extending would quietly shrink the validation set and "
            "make every metric measured on it incomparable. Restore the "
            "catalog rather than re-cutting the split."
        )

    held_out_positions = [
        i for i, sid in enumerate(ids) if existing.get(sid) in set(held_out_splits)
    ]
    held_out_scenes: Set[str] = set()
    for i in held_out_positions:
        key = scene_group_key(ids[i])
        if key is not None:
            held_out_scenes.add(key)

    # One KD-tree over the held-out samples; every new sample is one query.
    tree = None
    if held_out_positions:
        from scipy.spatial import cKDTree

        tree = cKDTree(xyz[held_out_positions])
    radius = _chord_for_arc(float(min_separation_km))

    assignments: List[str] = []
    n_new = n_frozen = new_to_train = 0
    excluded_near: List[str] = []
    excluded_scene: List[str] = []

    for position, sid in enumerate(ids):
        prior = existing.get(sid)
        if prior is not None:
            assignments.append(prior)
            n_frozen += 1
            continue

        n_new += 1
        scene = scene_group_key(sid)
        if scene is not None and scene in held_out_scenes:
            # Overlapping crops of one aerial scene leak at any distance.
            assignments.append(EXCLUDED)
            excluded_scene.append(sid)
            continue

        if tree is not None:
            neighbours = tree.query_ball_point(xyz[position], radius)
            if neighbours:
                assignments.append(EXCLUDED)
                excluded_near.append(sid)
                continue

        assignments.append(train_split)
        new_to_train += 1

    counts: Dict[str, int] = {}
    for split in assignments:
        counts[split] = counts.get(split, 0) + 1

    held_out_ids = {
        split: sorted(sid for sid, s in zip(ids, assignments) if s == split)
        for split in held_out_splits
    }
    # A held-out split gaining or losing one id makes every previously reported
    # number incomparable, so the property is checked against the file rather
    # than assumed from the loop above.
    #
    # This raise is UNREACHABLE as the code stands: existing assignments are
    # copied verbatim, and a missing held-out id was already rejected above. It
    # is kept deliberately as a trip-wire. The obvious future change here is
    # re-balancing -- "move a few train groups into val now that there is more
    # data" -- and that change would silently invalidate Run A. This makes it
    # fail loudly instead. tests/test_split_extend.py verifies the property;
    # nothing can verify the raise without breaking the loop on purpose.
    for split in held_out_splits:
        was = sorted(sid for sid, s in existing.items() if s == split)
        now = held_out_ids[split]
        if was != now:
            gained = sorted(set(now) - set(was))
            lost = sorted(set(was) - set(now))
            raise ValueError(
                f"Split {split!r} membership CHANGED: {len(gained)} gained, "
                f"{len(lost)} lost (e.g. gained {gained[:3]}, lost "
                f"{lost[:3]}). Every metric measured on the old {split} set "
                "would become incomparable. Nothing was written."
            )

    return {
        "assignments": assignments,
        "counts": counts,
        "n_new": n_new,
        "n_frozen": n_frozen,
        "new_to_train": new_to_train,
        "new_excluded_near": excluded_near,
        "new_excluded_scene": excluded_scene,
        "dropped_from_catalog": dropped,
        "held_out_ids": held_out_ids,
        "n_samples": len(ids),
        "min_separation_km": float(min_separation_km),
    }


def format_extension_summary(result: Mapping[str, Any]) -> str:
    """Render an extension result for the terminal.

    Args:
        result: What :func:`extend_split` returned.

    Returns:
        A multi-line block. Zero counts are printed, not omitted -- "0 new
        samples were excluded for proximity" is a finding.
    """
    counts = result["counts"]
    lines = [
        f"samples: {result['n_samples']}   frozen: {result['n_frozen']}   "
        f"new: {result['n_new']}   min_separation: "
        f"{result['min_separation_km']} km",
    ]
    for name in sorted(counts):
        lines.append(f"  {name:<10} {counts[name]:>6}")
    lines.append("")
    lines.append(f"  new -> train                  {result['new_to_train']}")
    lines.append(
        f"  new -> excluded (within {result['min_separation_km']:g} km)  "
        f"{len(result['new_excluded_near'])}"
    )
    lines.append(
        f"  new -> excluded (shared scene)  "
        f"{len(result['new_excluded_scene'])}"
    )
    if result["dropped_from_catalog"]:
        lines.append(
            f"  in split file but NOT in catalog: "
            f"{len(result['dropped_from_catalog'])} "
            f"(e.g. {result['dropped_from_catalog'][:2]})"
        )
    return "\n".join(lines)
