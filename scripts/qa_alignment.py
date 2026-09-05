"""Audit LR/HR co-registration and print a verdict.

Misaligned pairs are the failure mode that does not announce itself: the model
trains, the loss falls, and it learns to blur. So this runs before training, and
it ends in PASS / WARN / FAIL with the recommended action stated plainly.

What it writes:

- ``outputs/metrics/alignment_report.json`` -- every per-pair measurement, the
  summary statistics, the radiometric comparison, and the verdict.
- ``outputs/figures/alignment_grid.png`` -- per pair: nearest-upsampled LR,
  bicubic LR, HR, and the absolute difference between the last two, annotated
  with the measured shift and correlation.
- ``outputs/figures/alignment_histograms.png`` -- per-band LR vs HR reflectance
  distributions.

The smoke path is the estimator's self-test, not a data audit. It runs on
synthetic pairs that are aligned by construction, shifts their HR by
``cfg.alignment.inject_shift_px`` (2 px), and checks the estimator recovers it --
and also that the same estimator reports ~0 on the unshifted control. When a
shift is injected, the exit code reflects **the self-test**, because the verdict
is then being computed on deliberately corrupted pairs and a FAIL there is the
correct result.

Examples:
    # Prove the estimator works, on synthetic pairs with a known 2 px shift:
    python scripts/qa_alignment.py --config configs/base.yaml --smoke

    # Audit the real dataset (only the samples already cached, by default):
    python scripts/qa_alignment.py --config configs/base.yaml

    # Audit everything once the download has finished:
    python scripts/qa_alignment.py --config configs/base.yaml \\
        --set loader.cached_only=false --n-pairs 200

    # The same audit under a different draw, to show the verdict does not
    # rest on one lucky sample:
    python scripts/qa_alignment.py --config configs/base.yaml --n-pairs 200 --seed 7

Which pairs get audited
-----------------------
The pairs are a random sample WITHOUT REPLACEMENT, drawn with ``cfg.seed``
(overridable with ``--seed``), from the samples that are both cached and
validated by the manifest. It is never "the first N": SEN2NAIPv2 records are
ordered geographically, so the first 50 are one grid cell, one state, one
sensor geometry, and an audit of them describes that corner rather than the
archive. The printed block and the JSON report list every selected sample id
and their distribution across the geographic key the data carries, so the
audit's coverage is a stated, checkable fact rather than a count of pairs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.registry import available_datasets, get_dataset  # noqa: E402
from src.eval.alignment import (  # noqa: E402
    FAIL,
    audit_alignment,
    collect_pairs,
    describe_selection,
    format_selection_block,
    format_summary,
    format_verdict_block,
    select_audit_indices,
    self_test_result,
    write_report,
)
from src.eval.alignment_figures import (  # noqa: E402
    plot_alignment_grid,
    plot_reflectance_histograms,
)
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import resolve_output_path  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit LR/HR co-registration and print a PASS/WARN/FAIL verdict.",
    )
    add_standard_args(parser)
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"Override cfg.dataset.name. Registered: {available_datasets()}",
    )
    parser.add_argument(
        "--n-pairs",
        "--num-pairs",
        dest="num_pairs",
        type=int,
        default=None,
        help="Override cfg.alignment.num_pairs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg.seed. The pair selection is drawn with it, so the "
        "audit is reproducible from the seed printed in the report.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest CSV whose validated rows form the audit pool. Defaults "
        "to paths.manifest_dir/manifest_<dataset>.csv when that exists.",
    )
    parser.add_argument(
        "--no-manifest-filter",
        action="store_true",
        help="Audit every cached sample, including ones the index flagged with "
        "a validation_error. The report records that validation was not "
        "enforced.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write the JSON report only. The figures are the point; use this "
        "only when re-running for the numbers alone.",
    )
    return parser.parse_args(argv)


def _resolve_manifest(args, cfg, name: str, logger):
    """Decide which manifest, if any, restricts the audit pool.

    Precedence: ``--no-manifest-filter`` disables it outright; ``--manifest``
    names one and a missing file there is an error, because the user asked for
    it; otherwise the conventional ``manifest_<dataset>.csv`` is used when it
    exists. A missing conventional manifest is not an error -- a fresh clone and
    the ``--smoke`` path have none -- but it IS logged as a warning and recorded
    in the report, so an unfiltered audit can never be mistaken for a filtered
    one.

    Returns:
        A :class:`~pathlib.Path`, or None when no filter applies.
    """
    if args.no_manifest_filter:
        logger.warning(
            "--no-manifest-filter: auditing every cached sample, including any "
            "the index flagged with a validation_error."
        )
        return None

    if args.manifest is not None:
        return Path(args.manifest)

    default = Path(resolve_output_path(cfg, "manifest_dir")) / f"manifest_{name}.csv"
    if default.is_file():
        return default

    logger.warning(
        "No manifest at %s, so the audit pool is NOT restricted to validated "
        "samples. Run scripts/prepare_data.py --config %s first if you want "
        "the audit to cover only rows the index passed.",
        default,
        args.config,
    )
    return None


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("qa_alignment", log_file=cfg.paths.log_file)
    if args.seed is not None:
        logger.info("--seed %d overrides cfg.seed=%s", args.seed, cfg.seed)
        cfg.seed = int(args.seed)
    seed = seed_everything(cfg.seed)
    logger.info("Seeded with %d", seed)

    name = args.dataset or cfg.dataset.name
    num_pairs = int(args.num_pairs if args.num_pairs is not None else cfg.alignment.num_pairs)
    inject = cfg.alignment.inject_shift_px
    inject = [float(v) for v in inject] if inject is not None else None

    logger.info(
        "Auditing %r: %d pairs, band %s, upsample_factor=%d%s",
        name,
        num_pairs,
        cfg.alignment.reference_band,
        int(cfg.alignment.upsample_factor),
        f", INJECTED SHIFT (dy, dx) = ({inject[0]:+.2f}, {inject[1]:+.2f}) HR px"
        if inject
        else "",
    )

    dataset = get_dataset(cfg, name=name)

    # The audit pool: cached samples that the index validated. Restricting to
    # validated rows keeps the audit from measuring a shift on a pair that will
    # never reach training. When the manifest is absent the audit still runs,
    # but says so -- in the log, and in the report's selection block.
    manifest_path = _resolve_manifest(args, cfg, name, logger)

    indices = select_audit_indices(
        dataset,
        num_pairs=num_pairs,
        seed=int(cfg.seed),
        cached_only=bool(cfg.loader.cached_only),
        logger=logger,
        manifest_path=manifest_path,
    )
    selection = describe_selection(
        dataset,
        indices,
        seed=int(cfg.seed),
        cached_only=bool(cfg.loader.cached_only),
        manifest_path=manifest_path,
    )
    logger.info(
        "Selected %d pairs spanning %s distinct %s value(s).",
        selection["num_selected"],
        selection["num_distinct_regions"],
        selection["primary_grouping"],
    )

    # When a shift is injected, the unshifted pairs are audited first as a
    # control. Recovering the injected shift proves the estimator is sensitive;
    # measuring ~0 on the control proves it is not inventing shifts. Neither
    # alone is enough to trust the verdict.
    control_report = None
    if inject is not None:
        control_pairs = collect_pairs(dataset, indices, inject_shift_px=None)
        control_report = audit_alignment(control_pairs, cfg, logger=logger, label="control")

    pairs = collect_pairs(dataset, indices, inject_shift_px=inject)
    report = audit_alignment(
        pairs, cfg, logger=logger, label="injected" if inject else "audit"
    )
    report["selection"] = selection

    if inject is not None:
        report["self_test"] = self_test_result(
            report,
            control_report,
            injected_shift_px=inject,
            tolerance_px=float(cfg.alignment.inject_tolerance_px),
        )
        report["control"] = {
            "shift": control_report["shift"],
            "correlation": control_report["correlation"],
            "verdict": control_report["verdict"],
        }

    metric_dir = Path(resolve_output_path(cfg, "metric_dir"))
    report_path = write_report(report, metric_dir / str(cfg.alignment.report_name))
    logger.info("Report written to %s", report_path)

    if not args.no_figures:
        figure_dir = Path(resolve_output_path(cfg, "figure_dir"))
        grid_path = plot_alignment_grid(
            pairs, report["pairs"], cfg, figure_dir / str(cfg.alignment.grid_figure_name)
        )
        logger.info("Grid figure written to %s", grid_path)
        histogram_path = plot_reflectance_histograms(
            pairs,
            cfg,
            figure_dir / str(cfg.alignment.histogram_figure_name),
            radiometry=report["radiometry"],
        )
        logger.info("Histogram figure written to %s", histogram_path)

    print()
    print(format_selection_block(selection))
    print()
    print(format_summary(report))
    print()
    if inject is not None:
        print(
            "  NOTE: this run injected a known (dy "
            f"{inject[0]:+.2f}, dx {inject[1]:+.2f}) HR-pixel shift into every HR "
            "image to test the estimator.\n"
            "  The numbers above therefore describe DELIBERATELY MISALIGNED "
            "pairs, and a FAIL verdict is the\n"
            "  expected, correct outcome -- it is the proof that the audit "
            "detects a 2-pixel misregistration.\n"
            "  The control audit on the same pairs unshifted measured "
            f"{report['control']['shift']['median_magnitude']:.2f} HR px.\n"
            "  The exit code below reflects the SELF-TEST, not this verdict."
        )
        print()
    print(format_verdict_block(report))
    print()

    if inject is not None:
        passed = bool(report["self_test"]["passed"])
        if passed:
            logger.info(
                "Estimator self-test PASSED: recovered (%+.2f, %+.2f) HR px "
                "against an injected (%+.2f, %+.2f), control %.2f px.",
                report["self_test"]["recovered"][0],
                report["self_test"]["recovered"][1],
                inject[0],
                inject[1],
                report["control"]["shift"]["median_magnitude"],
            )
        else:
            logger.error(
                "Estimator self-test FAILED: %s. Do not trust a verdict from "
                "this estimator until it is fixed.",
                "; ".join(report["self_test"]["notes"]),
            )
        return 0 if passed else 1

    if report["verdict"]["level"] == FAIL:
        logger.error(
            "ALIGNMENT FAIL: median shift %.2f HR px. Do not train on this data.",
            report["shift"]["median_magnitude"],
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
