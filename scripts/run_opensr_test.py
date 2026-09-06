"""Benchmark a baseline (or any ``sr_fn``) with the external opensr-test suite.

This is the independent check on the numbers in ``scripts/run_baseline.py``.
Those come from our own metric implementations; these come from
``opensr-test`` (Aybar et al., IEEE JSTARS 2024; ESA OpenSR), which nobody in
this project wrote, and which measures something PSNR cannot -- whether the
detail a super-resolver adds is *actually there in the reference*.

Run the bicubic baseline through it FIRST, before any model exists. Bicubic
invents no detail at all, so its correctness figures are the arithmetic floor:
they say what "added nothing" scores, and no learned model's hallucination
number is legible until that is on paper.

What it writes (names from ``cfg.opensr_test``, with ``{method}`` filled in):

- ``outputs/metrics/opensr_<method>.json`` -- the summary, every setting the
  numbers depend on, and the FULL list of skipped samples with reasons.
- ``outputs/metrics/opensr_<method>.csv`` -- one row per scored sample.

and prints a markdown table of consistency (reflectance, spectral, spatial),
synthesis, and correctness (hallucination / omission / improvement).

Speed: opensr-test scores ONE sample at a time and runs two image registrations
per sample, so it is far slower than ``run_baseline.py``. Budget seconds per
sample on CPU. ``cfg.opensr_test.n_samples`` (default 200) bounds the run;
``--n-samples 0`` means the whole split.

Examples:
    # Pre-flight on the synthetic stub: no network, no data, ~1 minute on CPU.
    .venv/Scripts/python.exe scripts/run_opensr_test.py --smoke

    # The real bicubic benchmark over the validation split.
    .venv/Scripts/python.exe scripts/run_opensr_test.py --config configs/base.yaml

    # The nearest-neighbour floor, for comparison.
    .venv/Scripts/python.exe scripts/run_opensr_test.py --baseline nearest

    # Every validation sample (slow -- check seconds_per_sample first).
    .venv/Scripts/python.exe scripts/run_opensr_test.py --n-samples 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data.loader import build_dataloaders  # noqa: E402
from src.data.registry import available_datasets, get_dataset  # noqa: E402
from src.eval.baselines import available_baselines, get_baseline  # noqa: E402
from src.eval.opensr_harness import build_metrics, run_opensr_test  # noqa: E402
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import resolve_output_path  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a baseline with the external opensr-test benchmark and write "
            "JSON, CSV and a markdown table of the consistency, synthesis and "
            "correctness metrics."
        ),
    )
    add_standard_args(parser)
    parser.add_argument(
        "--baseline",
        default="bicubic",
        help=(
            f"Baseline to benchmark: one of {available_baselines()}. "
            "Default: bicubic, which is the floor every model must clear."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"Override cfg.dataset.name. Registered: {available_datasets()}",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help=(
            "Samples to attempt. Overrides cfg.opensr_test.n_samples. 0 means "
            "the whole split -- check seconds_per_sample from a short run first."
        ),
    )
    return parser.parse_args(argv)


def _force_utf8_stdout() -> None:
    """Make the markdown table printable on a Windows console.

    The table marks each metric's direction with U+2193 / U+2191. Windows
    consoles default to cp1252, which cannot encode either, and the run dies with
    a UnicodeEncodeError *after* the results have been written -- the numbers are
    safe on disk but the script exits non-zero and prints nothing. Reconfiguring
    the stream is the fix; replacing the arrows with ASCII would degrade the
    artefact to work around a terminal setting.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv=None) -> int:
    args = parse_args(argv)
    _force_utf8_stdout()
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("run_opensr_test", log_file=cfg.paths.log_file)
    seed = seed_everything(cfg.seed)
    logger.info("Seeded with %d", seed)

    if not bool(cfg.opensr_test.enabled):
        logger.error(
            "cfg.opensr_test.enabled is false. Set it to true, or pass "
            "--set opensr_test.enabled=true, to run this benchmark."
        )
        return 1

    # Same thread pin as run_baseline.py, so the two runs describe the same
    # machine and opensr_time_ms is measured at the deployment thread count.
    torch.set_num_threads(int(cfg.runtime.num_threads))
    logger.info(
        "torch threads set to cfg.runtime.num_threads=%d (deployment target); "
        "CUDA available: %s.",
        int(cfg.runtime.num_threads),
        torch.cuda.is_available(),
    )

    method = str(args.baseline)
    if method not in available_baselines():
        raise SystemExit(
            f"Unknown baseline {method!r}. Available: {available_baselines()}."
        )

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

    # n_samples: CLI wins, then config. 0 on the CLI means "all", which is
    # expressed as None internally.
    if args.n_samples is None:
        n_samples = cfg.opensr_test.n_samples
        n_samples = None if n_samples is None else int(n_samples)
    else:
        n_samples = None if int(args.n_samples) == 0 else int(args.n_samples)

    metrics, settings = build_metrics(cfg, logger=logger)
    settings = dict(settings)
    settings.update(
        {
            "method": method,
            "dataset": str(dataset_name),
            "split": val_split,
            "scale": int(cfg.sr.scale),
            "reflectance_scale": float(cfg.dataset.reflectance_scale),
            "smoke": bool(args.smoke),
        }
    )

    sr_fn = get_baseline(method, int(cfg.sr.scale))
    result = run_opensr_test(
        val_loader,
        sr_fn,
        n_samples,
        cfg,
        method=method,
        logger=logger,
        metrics=metrics,
        settings=settings,
    )

    metric_dir = Path(resolve_output_path(cfg, "metric_dir"))
    percentiles = [float(p) for p in cfg.metrics.percentiles]
    json_path = result.to_json(
        metric_dir / str(cfg.opensr_test.json_name).format(method=method),
        percentiles=percentiles,
    )
    csv_path = result.to_csv(
        metric_dir / str(cfg.opensr_test.csv_name).format(method=method)
    )
    logger.info("Summary written to %s", json_path)
    logger.info("Per-sample metrics written to %s", csv_path)

    print()
    print(f"## opensr-test -- {dataset_name}, {val_split} split")
    print()
    print(result.to_markdown(percentiles))
    print()
    print(
        f"> Wall time {result.settings['wall_time_s']:.1f} s for "
        f"{result.num_attempted} samples "
        f"({result.settings['seconds_per_sample']:.2f} s/sample) on "
        f"{result.settings['torch_threads']} CPU threads."
    )
    if args.smoke:
        print()
        print(
            "> **These are SMOKE numbers on the synthetic stub, not results.** "
            "The run exists to prove the opensr-test path executes end to end. "
            "The stub's LR is an exact block mean of its own HR, so its "
            "correctness figures describe that construction and not "
            "super-resolution."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
