"""Evaluate a non-learned baseline over the validation split.

This produces the number every later result is quoted against. Bicubic
interpolation is what a Sentinel-2 user already has, for free, with no GPU hours
and no parameter budget; a model that does not beat it convincingly has not
earned its place in the submission. Running it first, through exactly the
Evaluator that will score every model, means the comparison is honest by
construction rather than by care.

What it writes (names come from ``cfg.baseline``, with ``{method}`` filled in):

- ``outputs/metrics/baseline_<method>.csv`` -- one row per validation patch, with
  every metric and the identity of the patch it came from. This is the file that
  answers "where does it fail", which the summary cannot.
- ``outputs/metrics/baseline_<method>.json`` -- the summary (mean, std, p5, p95,
  min, max, worst sample per metric) plus every setting the numbers depend on.
- ``outputs/figures/baseline_qualitative.png`` -- LR / baseline / HR / SAM map
  for ``cfg.baseline.figure_samples`` patches spread across the split.

and prints a markdown table ready to paste into the report.

Threads are pinned to ``cfg.runtime.num_threads`` (6, the deployment target) so
the ``sr_time_ms`` column in the CSV is measured under the conditions the final
INT8 ONNX model will be benchmarked under.

Examples:
    # Pre-flight on the synthetic stub: no network, no data, ~1 minute on CPU.
    python scripts/run_baseline.py --config configs/base.yaml --smoke

    # The real bicubic baseline over the validation split.
    python scripts/run_baseline.py --config configs/base.yaml --baseline bicubic

    # Bicubic and the nearest-neighbour floor, over everything once the
    # download has finished.
    python scripts/run_baseline.py --config configs/base.yaml --baseline all \\
        --set loader.cached_only=false

    # No internet (LPIPS needs to fetch pretrained weights on first use):
    python scripts/run_baseline.py --config configs/base.yaml \\
        --set metrics.lpips.enabled=false
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data.loader import build_dataloaders  # noqa: E402
from src.data.registry import available_datasets, get_dataset  # noqa: E402
from src.eval.baseline_figures import plot_baseline_qualitative  # noqa: E402
from src.eval.baselines import available_baselines, get_baseline  # noqa: E402
from src.metrics.aggregate import Evaluator  # noqa: E402
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import resolve_output_path  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a non-learned super-resolution baseline over the "
            "validation split and write CSV, JSON, a markdown table, and a "
            "qualitative figure."
        ),
    )
    add_standard_args(parser)
    parser.add_argument(
        "--baseline",
        default="bicubic",
        help=(
            "Baseline to evaluate: one of "
            f"{available_baselines()}, or 'all' for every method in "
            "cfg.baseline.methods. Default: bicubic."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"Override cfg.dataset.name. Registered: {available_datasets()}",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Stop after this many validation batches. For debugging only -- the "
            "summary then describes those batches, not the split, and says so."
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help=(
            "Write the CSV and JSON only. The SAM panel is the part of this run "
            "that shows spectral failure; use this only when re-running for the "
            "numbers alone."
        ),
    )
    return parser.parse_args(argv)


def resolve_methods(requested: str, cfg) -> list:
    """Turn ``--baseline`` into a list of method names.

    Args:
        requested: The CLI value: a method name or ``"all"``.
        cfg: The loaded config; ``cfg.baseline.methods`` backs ``"all"``.

    Returns:
        Method names, in the order they will be evaluated.

    Raises:
        SystemExit: An unknown method was requested. Raised here rather than
            deep in the run so a typo costs a second, not a full pass over the
            validation set.
    """
    if str(requested).lower() == "all":
        methods = [str(m) for m in cfg.baseline.methods]
    else:
        methods = [str(requested)]
    unknown = [m for m in methods if m not in available_baselines()]
    if unknown:
        raise SystemExit(
            f"Unknown baseline(s) {unknown}. Available: {available_baselines()}, "
            "or 'all'."
        )
    return methods


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("run_baseline", log_file=cfg.paths.log_file)
    seed = seed_everything(cfg.seed)
    logger.info("Seeded with %d", seed)

    # Pin threads to the deployment target so sr_time_ms is measured under the
    # conditions the final INT8 ONNX model will be benchmarked under.
    torch.set_num_threads(int(cfg.runtime.num_threads))
    logger.info(
        "torch threads set to cfg.runtime.num_threads=%d (deployment target); "
        "CUDA available: %s.",
        int(cfg.runtime.num_threads),
        torch.cuda.is_available(),
    )

    methods = resolve_methods(args.baseline, cfg)
    dataset_name = args.dataset or cfg.dataset.name

    dataset = get_dataset(cfg, name=dataset_name)
    loaders = build_dataloaders(cfg, dataset=dataset, logger=logger)
    val_loader = loaders["val"]
    val_split = str(cfg.loader.val_split)
    logger.info(
        "Validation split %r: %d patches from %d tiles (split source: %s).",
        val_split,
        len(loaders["datasets"]["val"]),
        len(loaders["indices"]["val"]),
        loaders["split_source"],
    )

    evaluator = Evaluator(cfg, logger=logger)
    metric_dir = Path(resolve_output_path(cfg, "metric_dir"))
    figure_dir = Path(resolve_output_path(cfg, "figure_dir"))
    collect = 0 if args.no_figures else int(cfg.baseline.figure_samples)

    # A truncated run must never land on the real result's path. MEASURED the
    # hard way: a --max-batches 8 re-run for a figure tweak overwrote a complete
    # 759-patch baseline_bicubic.csv with 128 rows, and nothing in the CSV itself
    # said so. Same reasoning as the smoke filenames in cfg.smoke.baseline.
    suffix = "" if args.max_batches is None else f"_truncated{args.max_batches}"
    if suffix:
        logger.warning(
            "--max-batches %d: outputs are suffixed %r so this partial run "
            "cannot overwrite a complete baseline.",
            args.max_batches,
            suffix,
        )

    def _named(template: str, method: str) -> str:
        stem = Path(str(template).format(method=method))
        return f"{stem.stem}{suffix}{stem.suffix}"

    tables = []
    for method in methods:
        sr_fn = get_baseline(method, int(cfg.sr.scale))
        result = evaluator.run(
            val_loader,
            sr_fn,
            name=method,
            max_batches=args.max_batches,
            collect_samples=collect,
            split=val_split,
        )

        csv_path = result.to_csv(metric_dir / _named(cfg.baseline.csv_name, method))
        json_path = result.to_json(metric_dir / _named(cfg.baseline.json_name, method))
        logger.info("Per-sample metrics written to %s", csv_path)
        logger.info("Summary written to %s", json_path)

        if collect and result.qualitative:
            # One figure per run, named for the method when more than one is
            # evaluated, so an 'all' run does not overwrite its own output.
            stem = Path(str(cfg.baseline.figure_name))
            method_part = f"_{method}" if len(methods) > 1 else ""
            figure_name = f"{stem.stem}{method_part}{suffix}{stem.suffix}"
            figure_path = plot_baseline_qualitative(
                result.qualitative, cfg, figure_dir / figure_name, method=method
            )
            logger.info("Qualitative figure written to %s", figure_path)

        tables.append(result.to_markdown())

    print()
    print(f"## Baseline results -- {dataset_name}, {val_split} split")
    print()
    for table in tables:
        print(table)
        print()
    if args.smoke:
        print(
            "> **These are SMOKE numbers on the synthetic stub, not results.** "
            "This run exists to prove the evaluation path executes end to end, "
            "and its numbers describe the stub, not super-resolution.\n"
            ">\n"
            "> **Expect nearest to beat bicubic here, and do not read that as a "
            "broken baseline.** MEASURED on the stub: 98.7% of the HR variance "
            "is piecewise constant on 8x8 blocks that align exactly with the "
            "4x4 LR block-mean grid, and the remaining 1.3% is white texture "
            "noise. Pixel replication therefore reproduces almost the entire "
            "signal exactly, while bicubic smooths across block edges that are "
            "genuinely sharp. Real Sentinel-2 land cover has no such structure "
            "and the ordering reverses -- see "
            "tests/test_metrics.py::test_bicubic_beats_nearest_on_a_smooth_scene, "
            "which pins that ordering on a scene with realistic spatial "
            "correlation."
        )
        print()
    if args.max_batches is not None:
        print(
            f"> Truncated to --max-batches {args.max_batches}: the table above "
            "describes those batches, not the whole validation split."
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
