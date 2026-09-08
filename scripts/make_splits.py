"""Cut a geographic train/val/test split and write it to a CSV.

The split is derived from sample centroids and is verified before it is written:
if any two samples in different splits are closer than
``cfg.splits.min_separation_km``, the script fails and writes nothing. A leaky
split silently inflates every number in the report, so it must never be the
quiet default.

Two modes, and why the second one exists
----------------------------------------
The default mode CUTS a split from scratch. Re-running it after the dataset
grows reassigns everything -- the greedy largest-first pass sees different group
sizes and tiles move between splits -- so every number previously measured on
the old validation set silently becomes a number about a set that no longer
exists.

``--extend`` instead FREEZES every existing assignment and only places the new
samples, into train or into ``excluded``. It refuses to write if val or test
membership changes by a single id. Use it whenever the dataset grows and earlier
results must remain comparable. See src/data/split_extend.py.

Examples:
    python scripts/make_splits.py --config configs/base.yaml --smoke
    python scripts/make_splits.py --config configs/base.yaml
    python scripts/make_splits.py --config configs/base.yaml --set splits.min_separation_km=10

    # After a cache expansion added pairs: keep val/test exactly as they were.
    python scripts/make_splits.py --config configs/base.yaml --extend \\
        --set sen2naipv2.num_samples=4409
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.registry import available_datasets, get_dataset  # noqa: E402
from src.data.split_extend import (  # noqa: E402
    EXCLUDED,
    extend_split,
    format_extension_summary,
)
from src.data.splits import (  # noqa: E402
    geographic_split,
    scene_group_key,
    split_summary,
    verify_separation,
)
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import resolve_output_path  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a verified geographic train/val/test split.",
    )
    add_standard_args(parser)
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"Override cfg.dataset.name. Registered: {available_datasets()}",
    )
    parser.add_argument(
        "--extend",
        action="store_true",
        help="EXTEND the existing split instead of recomputing it. Every sample "
        "already in the split file keeps its assignment verbatim; new samples "
        "join train, or are marked 'excluded' when they sit within "
        "cfg.splits.min_separation_km of a val/test sample or share a NAIP "
        "quarter-quad with one. The script FAILS and writes nothing if val or "
        "test membership would change by even one id -- that is the whole "
        "point, since every metric measured on the old val set would otherwise "
        "become incomparable.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Read the existing split from here instead of the configured "
        "output path. Only meaningful with --extend.",
    )
    return parser.parse_args(argv)


def collect_records(dataset, logger):
    """Pull id + centroid for every sample, without decoding any imagery.

    Uses the dataset's catalog when it has one (SEN2NAIPv2), because reading
    3000 NPZ files just to recover coordinates would take minutes for no reason.
    Falls back to ``meta`` for datasets without a catalog.
    """
    catalog = getattr(dataset, "catalog", None)
    if catalog is not None:
        logger.info("Reading centroids from the %s catalog", type(dataset).__name__)
        return [
            {
                "sample_id": entry["sample_id"],
                "centroid_lonlat": entry.get("centroid_lonlat"),
            }
            for entry in catalog
        ]

    logger.info("Reading centroids from sample metadata (%d samples)", len(dataset))
    records = []
    for idx in range(len(dataset)):
        meta = dataset[idx]["meta"]
        records.append(
            {
                "sample_id": meta["sample_id"],
                "centroid_lonlat": meta.get("centroid_lonlat"),
            }
        )
    return records


def _write_split_csv(path: Path, records, assignments, group_ids) -> None:
    """Write the split CSV. One writer, so both paths produce the same columns.

    Args:
        path: Destination CSV.
        records: The sample records, carrying ``sample_id`` and
            ``centroid_lonlat``.
        assignments: Split name per record.
        group_ids: Group id per record. The extension path has no groups of its
            own and passes empty strings.
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "split", "group_id", "scene_key", "lon", "lat"])
        for record, split, group in zip(records, assignments, group_ids):
            lon, lat = record["centroid_lonlat"] or (None, None)
            writer.writerow(
                [
                    record["sample_id"],
                    split,
                    group,
                    scene_group_key(record["sample_id"]) or "",
                    lon,
                    lat,
                ]
            )


def _extend(cfg, args, records, out_path: Path, logger) -> int:
    """Extend an existing split with new samples. Returns an exit code.

    Refuses to write unless three things hold, each checked rather than assumed:

    1. Every id in the existing file keeps its assignment.
    2. Val and test membership is byte-identical to the file on disk.
    3. :func:`~src.data.splits.verify_separation`, re-derived from the
       coordinates, finds no cross-split pair closer than
       ``cfg.splits.min_separation_km`` among train/val/test.

    Args:
        cfg: The loaded config.
        args: Parsed arguments.
        records: Every sample the extended split must cover.
        out_path: The configured split CSV, also the default input.
        logger: For provenance.

    Returns:
        A process exit code. 0 only when the file was written.
    """
    split_cfg = cfg.splits
    source = Path(args.split_file) if args.split_file else out_path
    if not source.is_file():
        logger.error(
            "--extend needs an existing split to extend, but %s does not "
            "exist. Run this script without --extend to cut one first.",
            source,
        )
        return 1

    with source.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    existing = {row["sample_id"]: row["split"] for row in rows}
    before_counts = Counter(existing.values())
    logger.info(
        "Extending %s: %d existing assignments %s",
        source,
        len(existing),
        dict(before_counts),
    )

    # Held-out = every split in cfg.splits.fractions that is not the training
    # one. Derived rather than listed, because cfg.loader names train_split and
    # val_split but has no test_split key -- hardcoding ("val", "test") here
    # would silently stop protecting a split someone renames or adds.
    train_split = str(cfg.loader.train_split)
    held_out = tuple(
        name for name in split_cfg.fractions if str(name) != train_split
    )
    logger.info(
        "Freezing held-out split(s) %s; new samples may only join %r.",
        list(held_out),
        train_split,
    )

    try:
        result = extend_split(
            records,
            existing,
            min_separation_km=float(split_cfg.min_separation_km),
            train_split=train_split,
            held_out_splits=held_out,
        )
    except ValueError as exc:
        logger.error("Split extension refused: %s", exc)
        return 1

    if result["dropped_from_catalog"]:
        # The catalog shrank under a split file that still names those samples.
        # Writing would silently drop them; the ids are not recoverable later.
        logger.error(
            "%d samples in %s are absent from the catalog (e.g. %s). The "
            "catalog got SMALLER, which --extend cannot express. Nothing was "
            "written.",
            len(result["dropped_from_catalog"]),
            source,
            result["dropped_from_catalog"][:3],
        )
        return 1

    print()
    print(format_extension_summary(result))

    # Re-derive the separation guarantee from the coordinates over the splits
    # that are actually loaded. 'excluded' is deliberately left out: those
    # samples are near a held-out tile BY CONSTRUCTION and nothing reads them,
    # so including them would fail a check they are the answer to.
    live = [
        (record, split)
        for record, split in zip(records, result["assignments"])
        if split != EXCLUDED
    ]
    check = verify_separation(
        [r for r, _ in live],
        [s for _, s in live],
        min_separation_km=float(split_cfg.min_separation_km),
    )
    if not check["ok"]:
        logger.error(
            "EXTENDED SPLIT IS LEAKY: %d cross-split pairs are closer than "
            "%s km. Nothing was written. Examples (index_a, index_b, km): %s",
            check["n_violations"],
            split_cfg.min_separation_km,
            check["violations"][:5],
        )
        return 1

    logger.info(
        "Verified: nearest cross-split pair is %.2f km apart (required >= %.2f).",
        check["min_cross_split_km"],
        float(split_cfg.min_separation_km),
    )

    _write_split_csv(
        out_path, records, result["assignments"], [""] * len(records)
    )
    logger.info("Extended split written to %s", out_path)

    print(f"  nearest cross-split pair: {check['min_cross_split_km']:.2f} km")
    print(f"  val and test membership: UNCHANGED (asserted id by id)")
    print(f"  written to: {out_path}")
    print()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("make_splits", log_file=cfg.paths.log_file)
    seed_everything(cfg.seed)

    name = args.dataset or cfg.dataset.name
    dataset = get_dataset(cfg, name=name, validate=False)
    records = collect_records(dataset, logger)

    split_cfg = cfg.splits
    fractions = {k: float(v) for k, v in split_cfg.fractions.items()}

    out_dir = resolve_output_path(cfg, "manifest_dir")
    out_path = Path(out_dir) / str(split_cfg.output_name).format(dataset=name)

    if args.extend:
        return _extend(cfg, args, records, out_path, logger)

    result = geographic_split(
        records,
        fractions=fractions,
        min_separation_km=float(split_cfg.min_separation_km),
        seed=int(split_cfg.seed),
        group_by_scene=bool(split_cfg.group_by_scene),
    )

    logger.info("Split computed:\n%s", split_summary(result))

    empty = [n for n, count in result["counts"].items() if count == 0]
    if empty:
        logger.warning(
            "Split(s) %s are EMPTY. %d samples fell into only %d separable "
            "groups, so the requested fractions cannot be met. This is expected "
            "under --smoke (4 stub samples); on real data it means the samples "
            "are geographically clustered and min_separation_km is too large.",
            empty,
            result["n_samples"],
            result["n_groups"],
        )

    check = verify_separation(
        records,
        result["assignments"],
        min_separation_km=float(split_cfg.min_separation_km),
    )
    if not check["ok"]:
        logger.error(
            "SPLIT IS LEAKY: %d cross-split pairs are closer than %s km. "
            "Nothing was written. Examples (index_a, index_b, km): %s",
            check["n_violations"],
            split_cfg.min_separation_km,
            check["violations"][:5],
        )
        return 1

    logger.info(
        "Verified: nearest cross-split pair is %.2f km apart (required >= %.2f).",
        check["min_cross_split_km"],
        float(split_cfg.min_separation_km),
    )

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "split", "group_id", "scene_key", "lon", "lat"])
        for record, split, group in zip(
            records, result["assignments"], result["group_ids"]
        ):
            lon, lat = record["centroid_lonlat"] or (None, None)
            writer.writerow(
                [
                    record["sample_id"],
                    split,
                    group,
                    scene_group_key(record["sample_id"]) or "",
                    lon,
                    lat,
                ]
            )

    logger.info("Split written to %s", out_path)
    print()
    print(split_summary(result))
    print(f"  nearest cross-split pair: {check['min_cross_split_km']:.2f} km")
    print(f"  written to: {out_path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
