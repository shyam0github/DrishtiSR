"""De-risk opensr-test: score bicubic AND a trained checkpoint on the same pairs.

``scripts/run_opensr_test.py`` benchmarks a *baseline*. This script exists to
answer a narrower and more urgent question, before opensr-test becomes
load-bearing for the submission:

    do the consistency metrics emit real numbers, and do they DISCRIMINATE
    between an interpolation and a trained model?

A metric that returns NaN, or that returns the same value for bicubic and for
Run A, is worthless as evidence no matter how well it installs. So this runs
both methods over the *same* validation pairs, in one process, with one
``opensr_test.Metrics`` object, and prints the full per-metric dict for a single
pair alongside the mean over all of them -- the single-pair dump being the thing
that makes a NaN visible instead of averaged away.

Every array conversion happens in :func:`src.eval.opensr_harness.score_arrays`
and nowhere else. This script never transposes, rescales or casts.

What it writes (under ``cfg.paths.metrics_dir``):

- ``opensr_pair_<method>.json`` -- summary and settings, per method.
- ``opensr_pair_<method>.csv``  -- one row per scored pair, per method.

Examples:
    # Pre-flight on the synthetic stub: no data, no checkpoint, ~1 minute.
    .venv/Scripts/python.exe scripts/eval_opensr.py --smoke

    # The real thing: 50 val pairs, bicubic vs Run A's best.pt.
    .venv/Scripts/python.exe scripts/eval_opensr.py --n-samples 50

    # Bicubic only, when no checkpoint has been fetched yet.
    .venv/Scripts/python.exe scripts/eval_opensr.py --no-model
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data.loader import build_dataloaders  # noqa: E402
from src.data.registry import get_dataset  # noqa: E402
from src.eval.baselines import get_baseline  # noqa: E402
from src.eval.opensr_harness import (  # noqa: E402
    LOWER_IS_BETTER,
    METRIC_MEANING,
    OPENSR_METRICS,
    build_metrics,
    run_opensr_test,
)
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root, resolve_output_path  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score bicubic and a trained checkpoint with opensr-test over the "
            "same validation pairs, and print the full metric dict for one pair "
            "plus the mean over all of them."
        ),
    )
    add_standard_args(parser)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Validation pairs to score per method. Default 50.",
    )
    parser.add_argument(
        "--baseline",
        default="bicubic",
        help="Baseline to score as the floor. Default: bicubic.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Directory holding the checkpoint. Default: cfg.eval_runA.run_dir.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help=(
            "Checkpoint file inside --run-dir. Default: "
            "cfg.eval_runA.checkpoint_name."
        ),
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help=(
            "Score the baseline only. Use when no checkpoint has been fetched "
            "yet -- the run then proves the metrics emit numbers, but says "
            "nothing about whether they discriminate."
        ),
    )
    return parser.parse_args(argv)


def _force_utf8_stdout() -> None:
    """Make the arrows in the printed table survive a cp1252 Windows console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def resolve_checkpoint(
    cfg: Any,
    run_dir: Optional[str],
    checkpoint_name: Optional[str],
) -> Path:
    """Locate the checkpoint to score alongside the baseline.

    Args:
        cfg: Loaded config; reads ``eval_runA.run_dir`` and
            ``eval_runA.checkpoint_name``.
        run_dir: ``--run-dir``, or None for the configured value.
        checkpoint_name: ``--checkpoint-name``, or None for the configured value.

    Returns:
        The ``.pt`` path, resolved against the repository root when relative.

    Raises:
        FileNotFoundError: The checkpoint is absent. Raised rather than silently
            falling back to baseline-only, because a run that quietly scores one
            method answers none of the questions this script was written for.
    """
    block = cfg["eval_runA"]
    raw = Path(str(run_dir if run_dir is not None else block["run_dir"]))
    directory = raw if raw.is_absolute() else repo_root() / raw
    name = str(
        checkpoint_name if checkpoint_name is not None
        else block["checkpoint_name"]
    )
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Fetch the Kaggle output with "
            "'python scripts/kaggle_run.py fetch --job runa', point --run-dir "
            "at it, or pass --no-model to score the baseline alone."
        )
    return path


def print_pair_dump(method: str, result: Any, logger: Any) -> None:
    """Print every metric for ONE scored pair, unrounded and unaggregated.

    The mean over 50 pairs hides a NaN -- ``nanmean`` steps over it and the
    summary still prints a number. One raw pair does not. This is the check that
    the metrics actually emitted values, so it prints the sample id it came from
    and never substitutes anything for a non-finite value.

    Args:
        method: Label for the method, e.g. ``"bicubic"``.
        result: The :class:`~src.eval.opensr_harness.OpenSRResult`.
        logger: Logger.
    """
    if not result.rows:
        logger.warning("%s: nothing scored, no pair to dump.", method)
        return
    row = result.rows[0]
    identity = {
        key: value for key, value in row.items() if key not in OPENSR_METRICS
    }
    print(f"\n### Full metric dict, ONE pair -- {method}")
    print(f"    sample: {identity}")
    for metric in OPENSR_METRICS:
        value = float(row[metric])
        flag = "" if math.isfinite(value) else "   <-- NON-FINITE"
        arrow = (
            "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
        )
        print(f"    {metric:<14} = {value!r:<24} ({arrow}){flag}")


def print_comparison(results: Dict[str, Any], n_requested: int) -> None:
    """Print the per-method means side by side, so discrimination is visible.

    Args:
        results: ``{method: OpenSRResult}``, the baseline first.
        n_requested: The ``--n-samples`` value, for the denominator line.
    """
    methods = list(results)
    summaries = {name: results[name].summary() for name in methods}

    header = " | ".join(methods)
    print(f"\n### opensr-test means over {n_requested} requested pairs")
    print(f"\n| metric | dir | {header} | separation | meaning |")
    print("|---|:--:|" + "---:|" * len(methods) + "---:|---|")

    for metric in OPENSR_METRICS:
        lower = metric in LOWER_IS_BETTER
        arrow = "v" if lower else "^"
        cells = []
        values = []
        for method in methods:
            value = float(summaries[method][metric]["mean"])
            values.append(value)
            cells.append(f"{value:.4f}")
        # Separation is what tells us the metric is doing work at all. Reported
        # as the change from the FIRST method (the baseline) to the last, signed
        # so that positive always means "the later method is better". A
        # separation near zero means the metric does not discriminate, which is
        # the finding this script exists to surface.
        finite_ends = all(math.isfinite(v) for v in (values[0], values[-1]))
        if len(values) >= 2 and finite_ends:
            delta = values[0] - values[-1] if lower else values[-1] - values[0]
            sep = f"{delta:+.4f}"
        else:
            sep = "n/a"
        print(
            f"| `{metric}` | {arrow} | "
            + " | ".join(cells)
            + f" | {sep} | {METRIC_MEANING[metric]} |"
        )

    print()
    for method in methods:
        result = results[method]
        nonfinite = sum(
            1
            for row in result.rows
            if any(not math.isfinite(float(row[m])) for m in OPENSR_METRICS)
        )
        print(
            f"> **{method}**: {result.num_scored} scored, "
            f"{result.num_skipped} skipped, {nonfinite} rows with a "
            "non-finite metric."
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    _force_utf8_stdout()
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("eval_opensr", log_file=cfg.paths.log_file)
    seed = seed_everything(cfg.seed)
    logger.info("Seeded with %d", seed)

    threads = int(cfg.runtime.num_threads)
    torch.set_num_threads(threads)
    logger.info(
        "torch threads set to cfg.runtime.num_threads=%d (deployment target); "
        "CUDA available: %s.",
        threads,
        torch.cuda.is_available(),
    )

    if args.n_samples <= 0:
        raise ValueError(
            f"--n-samples must be positive; got {args.n_samples}. This script "
            "is a bounded de-risking pass, not a full-split benchmark; use "
            "scripts/run_opensr_test.py for that."
        )

    # The checkpoint is located BEFORE any scoring. A missing checkpoint found
    # after the baseline pass has already been paid for wastes the whole
    # baseline run, so this crashes early instead.
    checkpoint: Optional[Path] = None
    if not args.no_model:
        checkpoint = resolve_checkpoint(cfg, args.run_dir, args.checkpoint_name)
        logger.info("Checkpoint to score: %s", checkpoint)

    dataset = get_dataset(cfg)
    loaders = build_dataloaders(cfg, dataset=dataset, logger=logger)
    val_loader = loaders["val"]
    logger.info(
        "Validation split %r: %d patches from %d tiles.",
        str(cfg.loader.val_split),
        len(loaders["datasets"]["val"]),
        len(loaders["indices"]["val"]),
    )

    # One Metrics object, shared by both methods, so the two evaluations differ
    # only in the SR arrays and never in the scoring configuration.
    metrics, settings = build_metrics(cfg, logger=logger)

    methods: Dict[str, Any] = {}
    baseline_label = str(args.baseline)
    methods[baseline_label] = get_baseline(baseline_label, int(cfg.sr.scale))

    if checkpoint is not None:
        # Imported here, not at module scope: the --no-model path must not need
        # the checkpoint-loading code to import cleanly.
        from scripts.eval_runA import load_checkpoint_model, make_model_sr_fn

        # load_checkpoint_model already logs the iteration, the recorded best
        # val PSNR, the architecture and the parameter count, and warns when the
        # count exceeds cfg.runtime.max_parameters. The line below adds only the
        # thing it cannot know: which method label these weights are scored as.
        model, info = load_checkpoint_model(cfg, checkpoint, logger)
        model_label = str(cfg["eval_runA"]["model_label"])
        logger.info(
            "Scoring %r as method %r (%s parameters).",
            info["checkpoint"],
            model_label,
            info["parameters"],
        )
        methods[model_label] = make_model_sr_fn(model)

    results: Dict[str, Any] = {}
    for label, sr_fn in methods.items():
        result = run_opensr_test(
            dataloader=val_loader,
            sr_fn=sr_fn,
            n_samples=args.n_samples,
            cfg=cfg,
            method=label,
            logger=logger,
            metrics=metrics,
            settings=settings,
        )
        results[label] = result

        suffix = "_smoke" if args.smoke else ""
        metric_dir = Path(resolve_output_path(cfg, "metric_dir"))
        json_path = metric_dir / f"opensr_pair_{label}{suffix}.json"
        csv_path = metric_dir / f"opensr_pair_{label}{suffix}.csv"
        result.to_json(json_path)
        result.to_csv(csv_path)
        logger.info("%s: wrote %s and %s", label, json_path, csv_path)

        print_pair_dump(label, result, logger)

    print_comparison(results, args.n_samples)

    if len(results) == 1:
        print(
            "\n> Baseline only (--no-model): this run shows the metrics emit "
            "numbers, but NOT that they discriminate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
