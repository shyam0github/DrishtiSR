"""Cut a geographic train/val/test split and write it to a CSV.

The split is derived from sample centroids and is verified before it is written:
if any two samples in different splits are closer than
``cfg.splits.min_separation_km``, the script fails and writes nothing. A leaky
split silently inflates every number in the report, so it must never be the
quiet default.

Examples:
    python scripts/make_splits.py --config configs/base.yaml --smoke
    python scripts/make_splits.py --config configs/base.yaml
    python scripts/make_splits.py --config configs/base.yaml --set splits.min_separation_km=10
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.registry import available_datasets, get_dataset  # noqa: E402
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

    out_dir = resolve_output_path(cfg, "manifest_dir")
    out_path = Path(out_dir) / str(split_cfg.output_name).format(dataset=name)
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
