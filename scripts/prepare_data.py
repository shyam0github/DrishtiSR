"""Populate the sample cache and write an inspectable manifest.

Examples:
    # Fast CPU pre-flight, no network, synthetic 4-sample stub:
    python scripts/prepare_data.py --config configs/base.yaml --smoke

    # Real run against whatever cfg.dataset.name selects:
    python scripts/prepare_data.py --config configs/base.yaml

    # Point at a different dataset without editing the config:
    python scripts/prepare_data.py --config configs/base.yaml --dataset worldstrat

    # Index whatever is cached right now, without downloading more, and keep the
    # summary as a comparison baseline:
    python scripts/prepare_data.py --config configs/base.yaml --index-only \\
        --summary-out reports/index_summary_partial.txt

    # Re-index from scratch after changing what the index measures:
    python scripts/prepare_data.py --config configs/base.yaml --index-only --force \\
        --summary-out reports/index_summary_full.txt

Indexing is incremental: a re-run keeps the rows already in the manifest and
measures only what is newly cached. That is what makes it safe to run against a
partial download. It is also the trap -- after changing what the index MEASURES,
pass ``--force``, or the manifest silently becomes two halves written under two
different schemas. The summary block warns whenever it reused rows.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make `src` importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.registry import available_datasets, get_dataset  # noqa: E402
from src.data.sen2naip import format_index_summary  # noqa: E402
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import resolve_output_path  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

# Columns worth showing in the terminal. The CSV keeps every column.
PREVIEW_COLUMNS = (
    "index",
    "sample_id",
    "source_dataset",
    "crs",
    "lr_shape",
    "hr_shape",
    "scale_ok",
    "nodata_fraction",
    "max_reflectance",
    "rejected",
    "rejection_reason",
    "n_nan",
    "validation_error",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache dataset samples and write a manifest CSV.",
    )
    add_standard_args(parser)
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Override cfg.dataset.name for this run. Registered: "
            f"{available_datasets()}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only prepare the first N samples.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Write the manifest but do not print it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-measure every cached sample instead of reusing rows already in "
        "the manifest. REQUIRED after changing what the index measures -- "
        "otherwise the incremental skip preserves rows written under the old "
        "schema and the manifest becomes two incomparable halves.",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Also write the index summary block to this path, verbatim. Use it "
        "to keep a comparison baseline, e.g. reports/index_summary_partial.txt.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Skip the download step and index whatever is already cached.",
    )
    return parser.parse_args(argv)


def print_manifest(path: Path, max_rows: int = 20) -> None:
    """Print a manifest CSV as an aligned table.

    Args:
        path: The CSV written by ``build_index``.
        max_rows: Rows to show before truncating.
    """
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        print("(manifest is empty)")
        return

    columns = [c for c in PREVIEW_COLUMNS if c in rows[0]]
    # Per-band reflectance ranges are the point of the manifest, so show the
    # min/max columns too.
    columns += [
        c
        for c in rows[0]
        if c.endswith(("_min", "_max")) and c not in columns
    ]

    def cell(row, col):
        value = row.get(col, "")
        if col in ("validation_error", "rejection_reason") and value:
            first = value.splitlines()[0]
            return first[:40] + "..." if len(first) > 40 else first
        return str(value)

    widths = {
        c: max(len(c), *(len(cell(r, c)) for r in rows[:max_rows])) for c in columns
    }
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for row in rows[:max_rows]:
        print("  ".join(cell(row, c).ljust(widths[c]) for c in columns))
    if len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} more rows in {path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("prepare_data", log_file=cfg.paths.log_file)
    seed = seed_everything(cfg.seed)
    logger.info("Seeded with %d", seed)

    name = args.dataset or cfg.dataset.name
    logger.info("Preparing dataset %r (smoke=%s)", name, args.smoke)

    dataset = get_dataset(cfg, name=name)

    # Datasets that download implement prepare(); in-memory ones do not.
    prepare = getattr(dataset, "prepare", None)
    fetch_failures = 0
    if args.index_only:
        logger.info("--index-only: skipping the download, indexing the cache as it is.")
    elif callable(prepare):
        counts = prepare(limit=args.limit)
        fetch_failures = int(counts.get("failed", 0))
        logger.info("Cache: %s", counts)
        if fetch_failures:
            logger.error(
                "%d samples could not be fetched after retries. They are listed "
                "with their errors in the manifest. Re-run to retry them -- "
                "caching is resumable, so completed samples are not re-fetched.",
                fetch_failures,
            )
    else:
        logger.info(
            "%s has no prepare() step (nothing to download).",
            type(dataset).__name__,
        )

    manifest_dir = resolve_output_path(cfg, "manifest_dir")
    manifest_path = Path(manifest_dir) / f"manifest_{name}.csv"

    logger.info("Building manifest over %d samples (force=%s)", len(dataset), args.force)
    summary = _build_index(dataset, manifest_path, force=args.force)
    logger.info("Manifest written to %s", manifest_path)

    if not args.no_preview:
        print()
        print_manifest(manifest_path)
        print()

    block = format_index_summary(summary) if summary is not None else None
    if block is not None:
        print(block)
        print()
        if args.summary_out:
            out = Path(args.summary_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(block + "\n", encoding="utf-8")
            logger.info("Index summary written to %s", out)
    elif args.summary_out:
        # Asked for a summary the dataset cannot produce. Say so rather than
        # writing an empty file that later reads as "nothing was wrong".
        logger.error(
            "%s does not produce an index summary, so --summary-out %s was NOT "
            "written.",
            type(dataset).__name__,
            args.summary_out,
        )
        return 1

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # A policy rejection (too much nodata) is an expected, measured outcome and
    # is reported in the summary block above. A validation_error WITHOUT a
    # rejection_reason is a loader bug and still fails the run.
    unexplained = [
        r
        for r in rows
        if r.get("validation_error") and not r.get("rejection_reason")
    ]
    if unexplained:
        logger.error(
            "%d of %d rows carry a validation_error with no rejection reason. "
            "That is a loader fault, not a data-quality rejection. See %s.",
            len(unexplained),
            len(rows),
            manifest_path,
        )
        return 1
    if fetch_failures:
        return 1

    logger.info("Indexed %d rows into %s.", len(rows), manifest_path)
    return 0


def _build_index(dataset, manifest_path: Path, force: bool):
    """Call ``build_index``, passing ``force`` only to datasets that accept it.

    The incremental/``--force`` contract is implemented by SEN2NAIPv2, which is
    the dataset with a multi-hour download to resume. The in-memory stub and the
    fallbacks rebuild their manifest from scratch every time, so ``force`` is
    meaningless to them rather than merely unimplemented.

    Returns:
        The summary dict when the dataset produced one, else None.
    """
    import inspect

    signature = inspect.signature(dataset.build_index)
    if "force" in signature.parameters:
        result = dataset.build_index(manifest_path, force=force)
    else:
        if force:
            get_logger("prepare_data").warning(
                "%s rebuilds its manifest in full on every run, so --force "
                "changes nothing for it.",
                type(dataset).__name__,
            )
        result = dataset.build_index(manifest_path)

    return result if isinstance(result, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
