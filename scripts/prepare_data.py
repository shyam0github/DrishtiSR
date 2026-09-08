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

    # Adopt a cache that predates the manifest (headers only, no decompression):
    python scripts/prepare_data.py --backfill-manifest

    # Time-boxed expansion. Grows the cache toward 4000 pairs but stops after 45
    # minutes whatever it reached, and EXITS 0 either way:
    python scripts/prepare_data.py --target-pairs 4000 --time-budget-sec 2700 \\
        --set sen2naipv2.num_samples=4409

Two caching paths, and which to use
-----------------------------------
``prepare()`` -- the default -- walks the whole catalog to completion. It is the
right thing for an unattended overnight download.

``--target-pairs`` / ``--time-budget-sec`` selects the expansion path instead:
manifest-driven resumption, atomic per-pair commits, per-pair validation before
commit, and a hard wall clock. Use it whenever the run has to end at a known
time. The budget expiring is a SUCCESS -- the script exits 0, because a partial
cache is a smaller training set and not a broken one, and the next run picks up
from the manifest without re-downloading anything.

Indexing is incremental: a re-run keeps the rows already in the manifest and
measures only what is newly cached. That is what makes it safe to run against a
partial download. It is also the trap -- after changing what the index MEASURES,
pass ``--force``, or the manifest silently becomes two halves written under two
different schemas. The summary block warns whenever it reused rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Make `src` importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache_manifest import format_rejection_tally  # noqa: E402
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
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=None,
        help="Expand the cache until it holds this many pairs IN TOTAL "
        "(existing plus new), then stop. Selects the time-boxed, "
        "manifest-driven expansion path instead of the plain prepare() loop. "
        "This is a total, not a delta, so re-running with the same value is "
        "idempotent.",
    )
    parser.add_argument(
        "--time-budget-sec",
        type=float,
        default=None,
        help="Wall-clock seconds the cache expansion may run. Checked at the "
        "top of every pair, so the pair in flight finishes and is committed "
        "before the loop exits. EXPIRY IS A SUCCESS: the script flushes, prints "
        "the summary, and exits 0, because a partial cache is a smaller "
        "training set rather than a broken one. Also selects the expansion "
        "path.",
    )
    parser.add_argument(
        "--backfill-manifest",
        action="store_true",
        help="Adopt pairs already on disk into the cache manifest and exit. "
        "Reads NPZ headers only, so it does not decompress the cache. Run this "
        "once on a cache that predates the manifest.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Write the cache-expansion summary to this path as JSON.",
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

    if args.backfill_manifest:
        return _backfill_only(dataset, logger)

    expanding = args.target_pairs is not None or args.time_budget_sec is not None
    if expanding:
        return _expand(dataset, args, logger)

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


def _require_expansion_support(dataset, what: str):
    """Fail loudly when a dataset cannot do the manifest-driven expansion.

    The synthetic stub fabricates its samples and has no cache to expand. Asking
    it to would otherwise print a summary of zeros and exit 0, which reads as
    "the expansion ran and found nothing left to do" -- the opposite of the
    truth.

    Args:
        dataset: The constructed dataset.
        what: The flag that requested it, for the message.

    Raises:
        SystemExit: The dataset has no expansion path.
    """
    if not hasattr(dataset, "expand_cache"):
        raise SystemExit(
            f"{type(dataset).__name__} has no cache to expand, so {what} means "
            "nothing to it. The manifest-driven expansion is implemented by "
            "SEN2NAIPv2, which is the dataset with a multi-hour download. Pass "
            "--dataset sen2naipv2, or drop the flag."
        )


def _backfill_only(dataset, logger) -> int:
    """Adopt on-disk pairs into the manifest and stop. Returns an exit code."""
    _require_expansion_support(dataset, "--backfill-manifest")
    counts = dataset.backfill_manifest()
    print()
    print(f"Manifest: {dataset.manifest_file}")
    for key in ("adopted", "already_recorded", "unreadable", "structural_reject"):
        print(f"  {key:<20} {counts[key]}")
    print(f"  {'total recorded':<20} {len(dataset.manifest_index)}")
    print()
    # An unreadable or structurally bad pair is a real finding about the cache,
    # not a routine outcome, so it colours the exit code. The manifest itself is
    # still written and usable.
    if counts["unreadable"] or counts["structural_reject"]:
        logger.error(
            "%d unreadable and %d structurally rejected pairs are on disk but "
            "NOT in the manifest. They are listed above at ERROR in the log.",
            counts["unreadable"],
            counts["structural_reject"],
        )
        return 1
    return 0


def _expand(dataset, args, logger) -> int:
    """Run the time-boxed cache expansion and report it. Returns an exit code.

    Exit code policy, which is the point of the whole flag:

    - **0** when the budget expired, the target was reached, or the catalog ran
      out, even with rejects. A partial cache is a success.
    - **1** only when the expansion could not run at all.

    Args:
        dataset: A dataset implementing ``expand_cache``.
        args: Parsed arguments.
        logger: For the provenance lines.

    Returns:
        A process exit code.
    """
    _require_expansion_support(dataset, "--target-pairs / --time-budget-sec")

    summary = dataset.expand_cache(
        target_pairs=args.target_pairs,
        time_budget_sec=args.time_budget_sec,
    )

    rejects = summary["rejects"]
    lines = [
        "",
        "=== cache expansion ===",
        f"manifest            {summary['manifest']}",
        f"catalog size        {summary['catalog_size']}",
        f"pairs before        {summary['before']}",
        f"pairs after         {summary['after']}",
        f"newly cached        {summary['new']}",
        f"elapsed             {summary['elapsed_s']:.1f}s",
        f"rate                {summary['rate_pairs_per_s']:.4f} pairs/s",
        f"projected total     {summary['projected_total_at_rate']}",
        f"stopped because     {summary['stopped_because']}",
        "",
        f"backfill            {summary['backfill']}",
        "",
        f"rejects             {sum(rejects.values())} total",
    ]
    lines += format_rejection_tally(rejects)
    lines.append("")
    block = "\n".join(lines)
    print(block, flush=True)

    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Expansion summary written to %s", out)

    if summary["stopped_because"] == "time_budget":
        logger.info(
            "The wall-clock budget stopped this run. That is the designed "
            "ending, not a failure: the cache grew by %d pairs and the next "
            "run resumes from the manifest without re-downloading any of them.",
            summary["new"],
        )
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
