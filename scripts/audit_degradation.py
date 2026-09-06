"""Test whether the stored LR is a synthetic downsample of the HR. Report plainly.

WHY THIS EXISTS. ``scripts/verify_data_root.py`` printed per-band reflectance
ranges for a cached SEN2NAIPv2 pair whose LR and HR minima and maxima agreed to
four decimal places on all four bands, with the means differing only in the
fourth decimal. Two sensors at 10 m and 2.5 m do not do that. This script
decides, by measurement, which of two explanations holds:

    H1  the LR was synthetically degraded from the HR -- in which case the
        project's 38 dB bicubic baseline is measuring the inversion of a known
        analytic operator and must not be reported as super-resolution;
    H2  the HR was radiometrically harmonised onto the LR's reflectance scale --
        in which case the marginal distributions agree by construction, the
        spatial content is genuinely independent, and the task is real.

What it does, per pair, over a random sample of ``cfg.degradation_audit.num_pairs``:

1. Degrades the HR to LR size with every kernel in ``cfg.degradation_audit.kernels``
   and scores each against the STORED LR, in the reflectance domain at
   ``cfg.metrics.data_range``. Nearest-subsampling is tried at all 16 phases.
2. Scores the same LR against a DIFFERENT sample's degraded HR -- the control
   that gives the PSNR a scale.
3. Compares the LR and HR marginal distributions. This is the decisive test:
   averaging reduces variance, so any averaging degradation must leave
   ``std(LR)/std(HR)`` visibly below 1.
4. Optionally scores bicubic UPSAMPLING of LR against HR, the x4 baseline number
   this audit exists to justify or discredit.

Everything is reported as a DISTRIBUTION -- p0/p5/p25/p50/p75/p95/p100 -- because
a mean is what let the anomaly go unnoticed.

What it writes:

- ``outputs/metrics/degradation_audit.json`` -- every summary statistic, the
  marginal comparison, the control, the thresholds in force, and the verdict.
- ``outputs/metrics/degradation_audit_per_sample.csv`` -- one row per pair, so
  the distribution can be re-plotted without re-running.

The exit code is 0 when the verdict is CROSS_SENSOR, 1 for SYNTHETIC or
UNRESOLVED. A non-zero exit here means the dataset is not what the config claims
and no reported number should be trusted until that is resolved.

THE SMOKE PATH IS A SELF-TEST, not a rehearsal. The synthetic stub builds its LR
as an exact block-mean downsample of its HR, so ``--smoke`` feeds this audit a
pair that IS synthetically degraded by a kernel in the candidate list. A correct
audit must return SYNTHETIC, name ``box``, and report PSNR at the numerical
ceiling. Until that passes, the verdict on real data means nothing -- so under
``--smoke`` the exit code reflects the SELF-TEST, and 0 means "the detector
works".

Examples:
    # Prove the detector detects a known synthetic degradation:
    python scripts/audit_degradation.py --config configs/base.yaml --smoke

    # Audit the real dataset:
    python scripts/audit_degradation.py --config configs/base.yaml

    # A different draw, to show the verdict does not rest on one lucky sample:
    python scripts/audit_degradation.py --config configs/base.yaml --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.registry import get_dataset  # noqa: E402
from src.eval.alignment import select_audit_indices  # noqa: E402
from src.eval.degradation import (  # noqa: E402
    DEGRADATION_KERNELS,
    block_valid_mask,
    degrade,
    marginal_agreement,
    quantisation_ceiling_db,
    score_pair,
    summarise,
    verdict,
)
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import resolve_output_path  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the stored LR is a synthetic downsample of the HR, "
            "or a genuine second sensor."
        ),
    )
    add_standard_args(parser)
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=None,
        dest="num_pairs",
        help="Override cfg.degradation_audit.num_pairs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg.seed, to redraw the sample.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override cfg.dataset.name.",
    )
    return parser.parse_args(argv)


def load_raw_pair(dataset: Any, idx: int, nodata_value: Any, nodata_fill: float):
    """Read one cached pair as reflectance, with its nodata masks.

    Goes through the dataset's own cache path and its own
    ``to_reflectance``, so this audit measures exactly the arrays training will
    see -- an independent re-implementation of the conversion here could agree
    with the data and disagree with the model.

    No cropping is applied. The full stored tile is used because the tile
    periphery is where nodata and bright targets live, and a centre crop would
    quietly exclude the pixels most able to discriminate the hypotheses.

    Args:
        dataset: An :class:`~src.data.base.SRPairDataset` exposing
            ``ensure_cached``, ``band_indices`` and ``to_reflectance``.
        idx: Index into the dataset catalogue.
        nodata_value: Raw sentinel, ``cfg.dataset.nodata_value``.
        nodata_fill: Reflectance written where a pixel is nodata,
            ``cfg.dataset.nodata_fill``.

    Returns:
        ``(lr, hr, lr_nodata, hr_nodata)``. ``lr`` is ``(C, 130, 130)`` and
        ``hr`` ``(C, 520, 520)``, both float32 surface reflectance -- raw digital
        numbers divided by ``cfg.dataset.reflectance_scale``, nominally
        ``[0, 1]`` and UNCLIPPED above 1.0. The masks are bool in the matching
        shapes, True where the pixel is nodata.

    Raises:
        KeyError: The cached NPZ lacks an ``lr`` or ``hr`` array.
    """
    npz_path = dataset.ensure_cached(idx)
    with np.load(npz_path) as handle:
        if "lr" not in handle or "hr" not in handle:
            raise KeyError(
                f"Cached sample {npz_path} lacks an 'lr' or 'hr' array; it holds "
                f"{list(handle.files)}. The cache entry is corrupt -- delete it "
                "and re-run scripts/prepare_data.py."
            )
        lr_raw = handle["lr"][dataset.band_indices]
        hr_raw = handle["hr"][dataset.band_indices]

    lr, lr_nodata = dataset.to_reflectance(
        lr_raw, nodata_value=nodata_value, nodata_fill=nodata_fill
    )
    hr, hr_nodata = dataset.to_reflectance(
        hr_raw, nodata_value=nodata_value, nodata_fill=nodata_fill
    )
    return (
        np.ascontiguousarray(lr, dtype=np.float32),
        np.ascontiguousarray(hr, dtype=np.float32),
        np.asarray(lr_nodata, dtype=bool),
        np.asarray(hr_nodata, dtype=bool),
    )


def load_stub_pair(dataset: Any, idx: int):
    """Read one fabricated pair from the synthetic stub, with empty nodata masks.

    The stub has no cache and no ``to_reflectance``; it returns reflectance
    tensors directly and contains no nodata by construction.

    Args:
        dataset: A :class:`~src.data.synthetic.SyntheticStubDataset`.
        idx: Index into the stub.

    Returns:
        ``(lr, hr, lr_nodata, hr_nodata)`` in the same convention as
        :func:`load_raw_pair`; both masks are all-False.
    """
    sample = dataset.load_sample(idx)
    lr = np.ascontiguousarray(sample["lr"].numpy(), dtype=np.float32)
    hr = np.ascontiguousarray(sample["hr"].numpy(), dtype=np.float32)
    return lr, hr, np.zeros(lr.shape, dtype=bool), np.zeros(hr.shape, dtype=bool)


def upsample_baseline_db(
    lr: np.ndarray, hr: np.ndarray, hr_valid: np.ndarray, scale: int, data_range: float
) -> float:
    """Bicubic x4 upsampling of LR scored against HR -- the project's baseline.

    Printed beside the degradation scores on purpose. The whole question is
    whether the baseline number describes super-resolution or the inversion of a
    downsample, and the two numbers only mean something next to each other.

    Args:
        lr: ``(C, H, W)`` float32 surface reflectance.
        hr: ``(C, H*scale, W*scale)`` float32 surface reflectance.
        hr_valid: ``(C, H*scale, W*scale)`` bool, True where scorable.
        scale: Integer factor.
        data_range: Peak reflectance, ``cfg.metrics.data_range``.

    Returns:
        Mean-over-bands PSNR in dB.
    """
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(lr)[None]
    up = F.interpolate(
        tensor,
        scale_factor=scale,
        mode="bicubic",
        align_corners=False,
    )[0].numpy()

    residual = np.where(hr_valid, up.astype(np.float64) - hr, 0.0)
    counts = hr_valid.reshape(hr_valid.shape[0], -1).sum(axis=1)
    sums = (residual**2).reshape(residual.shape[0], -1).sum(axis=1)
    with np.errstate(divide="ignore"):
        per_band = 10.0 * np.log10((float(data_range) ** 2) / (sums / counts))
    return float(np.mean(per_band))


def _fmt_db(value: float) -> str:
    """Format a dB figure, preserving ``inf`` rather than printing a large number."""
    if not np.isfinite(value):
        return "     inf" if value > 0 else "    -inf"
    return f"{value:8.2f}"


def print_report(
    summary: Dict[str, Any],
    kernels: List[str],
    percentiles: List[float],
    control: Optional[Dict[str, Any]],
    marginals: Dict[str, Any],
    baseline_db: Optional[float],
    decision: Dict[str, Any],
    bands: List[str],
    num_pairs: int,
    offsets: Dict[str, int],
) -> None:
    """Print the whole audit: distributions first, verdict last."""
    print()
    print("=" * 78)
    print("  IS THE LR A SYNTHETIC DOWNSAMPLE OF THE HR?")
    print("=" * 78)
    print()
    print(f"  {num_pairs} pairs, full stored tiles, reflectance domain.")
    print()

    print("1. DEGRADATION KERNEL FIT  (degraded HR vs stored LR, dB)")
    print()
    header = "  kernel               " + "".join(f"{'p'+f'{p:g}':>9}" for p in percentiles)
    print(header + f"{'mean':>9}{'exact':>7}")
    print("  " + "-" * (len(header) + 14))
    for kernel in kernels:
        stats = summary[kernel]
        cells = "".join(_fmt_db(stats[f"p{p:g}"]) + " " for p in percentiles)
        mark = " <-- best" if kernel == summary["best_kernel"] else ""
        print(
            f"  {kernel:<20} {cells}{_fmt_db(stats['mean'])} "
            f"{stats['num_infinite']:>5}{mark}"
        )
    print()
    print("  'exact' counts pairs the kernel reproduced with zero residual.")

    if offsets:
        print()
        total = sum(offsets.values())
        top = sorted(offsets.items(), key=lambda kv: kv[1], reverse=True)[:4]
        rendered = ", ".join(f"{key} on {count}" for key, count in top)
        print(f"  Best nearest phase per pair: {rendered} (of {total}).")
        print(
            "  A real nearest degradation uses ONE phase, so a scatter here "
            "means the fit is coincidence."
        )

    if control is not None:
        print()
        print("2. CONTROL: the same LR against an UNRELATED sample's degraded HR")
        print()
        cells = "".join(_fmt_db(control[f"p{p:g}"]) + " " for p in percentiles)
        print("  mismatched pairs     " + cells)
        print()
        print(
            "  This is the floor. Natural imagery over similar terrain scores "
            "far above 0 dB,"
        )
        print("  so the true-pair number only means something against this one.")

    print()
    print("3. MARGINAL DISTRIBUTIONS  (the test that does not need a threshold)")
    print()
    print("  Averaging reduces variance. Any averaging degradation MUST push")
    print("  std(LR)/std(HR) visibly below 1.0.")
    print()
    print(f"  {'band':<8}{'std(LR)/std(HR)':>18}{'|mean diff|':>14}{'max |quantile diff|':>22}")
    for index, band in enumerate(bands):
        print(
            f"  {band:<8}{marginals['std_ratio_per_band'][index]:>18.4f}"
            f"{marginals['mean_abs_diff_per_band'][index]:>14.6f}"
            f"{marginals['max_quantile_diff_per_band'][index]:>22.6f}"
        )
    print()
    print(f"  median over all bands and pairs: {marginals['median_std_ratio']:.4f}")

    if baseline_db is not None:
        print()
        print("4. FOR COMPARISON: the x4 bicubic SR baseline")
        print()
        print(f"  bicubic upsampled LR vs HR: {baseline_db:.2f} dB median")
        print(
            "  This is the number the project reports. Read it against section 1."
        )

    print()
    print("5. THE THREE TESTS A SYNTHETIC DEGRADATION MUST PASS")
    print()
    checks = [
        (
            "reaches the uint16 quantisation ceiling "
            f"({decision['quantisation_ceiling_db']:.1f} dB)",
            decision["reaches_quantisation_ceiling"],
        ),
        (
            f"deterministic across tiles (spread {decision['spread_db']:.2f} dB)",
            decision["deterministic"],
        ),
        (
            "lost variance, as an averaging kernel must",
            not (decision["kernel_family"] == "averaging" and decision["variance_preserved"]),
        ),
    ]
    for label, passed in checks:
        print(f"    [{'YES' if passed else ' NO'}]  {label}")
    print()
    if decision["evidence"]["tests_disagree"]:
        print(
            "  NOTE: the dB threshold and the physical tests DISAGREE. See the "
            "verdict below."
        )
        print()

    print("=" * 78)
    print(f"  VERDICT: {decision['verdict']}")
    print("=" * 78)
    print()
    for reason in decision["reasons"]:
        for line in _wrap(reason, 74):
            print(f"  {line}")
        print()


def _wrap(text: str, width: int) -> List[str]:
    """Wrap text to ``width`` columns without importing textwrap for one call."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("audit_degradation", log_file=cfg.paths.log_file)
    if args.seed is not None:
        logger.info("--seed %d overrides cfg.seed=%s", args.seed, cfg.seed)
        cfg.seed = int(args.seed)
    seed = seed_everything(cfg.seed)
    logger.info("Seeded with %d", seed)

    audit_cfg = cfg.degradation_audit
    name = args.dataset or cfg.dataset.name
    num_pairs = int(
        args.num_pairs if args.num_pairs is not None else audit_cfg.num_pairs
    )
    kernels = [str(k) for k in audit_cfg.kernels]
    unknown = [k for k in kernels if k not in DEGRADATION_KERNELS]
    if unknown:
        raise ValueError(
            f"cfg.degradation_audit.kernels names unknown kernels {unknown}. "
            f"Known: {list(DEGRADATION_KERNELS)}."
        )

    scale = int(cfg.sr.scale)
    data_range = float(cfg.metrics.data_range)
    bands = [str(b) for b in cfg.dataset.bands]
    percentiles = [float(p) for p in audit_cfg.percentiles]
    quantiles = [float(q) for q in audit_cfg.quantiles]

    logger.info(
        "Auditing %r: %d pairs, kernels=%s, scale=x%d, data_range=%.4g",
        name,
        num_pairs,
        kernels,
        scale,
        data_range,
    )

    dataset = get_dataset(cfg, name=name)
    is_stub = name == "synthetic_stub"

    indices = select_audit_indices(
        dataset,
        num_pairs=num_pairs,
        seed=int(cfg.seed),
        cached_only=bool(cfg.loader.cached_only),
        logger=logger,
    )

    # The control pairs each LR with a DIFFERENT sample's HR. A derangement --
    # a permutation with no fixed point -- guarantees no sample is silently
    # compared against itself, which would contaminate the floor with true pairs.
    rng = np.random.default_rng(int(cfg.seed))
    partners = list(range(len(indices)))
    if len(partners) > 1:
        for _ in range(64):
            rng.shuffle(partners)
            if all(a != b for a, b in enumerate(partners)):
                break
        else:
            raise RuntimeError(
                "Could not build a derangement for the mismatched-pair control "
                "in 64 attempts, which should be impossible for n > 1. Refusing "
                "to fall back to a permutation with fixed points -- that would "
                "score some samples against themselves and inflate the floor."
            )

    per_sample: List[Dict[str, Any]] = []
    loaded: Dict[int, Any] = {}
    std_ratios: List[List[float]] = []
    mean_diffs: List[List[float]] = []
    quant_diffs: List[List[float]] = []
    baselines: List[float] = []
    offset_counts: Dict[str, int] = {}

    for position, idx in enumerate(indices):
        if is_stub:
            lr, hr, lr_nodata, hr_nodata = load_stub_pair(dataset, idx)
        else:
            lr, hr, lr_nodata, hr_nodata = load_raw_pair(
                dataset,
                idx,
                nodata_value=cfg.dataset.nodata_value,
                nodata_fill=float(cfg.dataset.nodata_fill),
            )
        loaded[position] = (hr, hr_nodata)

        valid = block_valid_mask(lr_nodata, hr_nodata, scale)
        scored = score_pair(
            lr,
            hr,
            valid,
            scale=scale,
            kernels=kernels,
            data_range=data_range,
            search_nearest_offsets=bool(audit_cfg.nearest_search_all_offsets),
        )

        sample_id = (
            dataset.catalog[idx]["sample_id"]
            if getattr(dataset, "catalog", None) is not None
            else f"index_{idx}"
        )
        scored["index"] = int(idx)
        scored["sample_id"] = str(sample_id)

        if scored["nearest_offset"] is not None:
            key = f"({scored['nearest_offset'][0]},{scored['nearest_offset'][1]})"
            offset_counts[key] = offset_counts.get(key, 0) + 1

        marg = marginal_agreement(lr, hr, ~lr_nodata, ~hr_nodata, quantiles)
        std_ratios.append(marg["std_ratio"])
        mean_diffs.append(marg["mean_abs_diff"])
        quant_diffs.append(marg["quantiles"]["max_abs_diff"])

        if bool(audit_cfg.include_upsample_baseline):
            baselines.append(
                upsample_baseline_db(lr, hr, ~hr_nodata, scale, data_range)
            )

        per_sample.append(scored)
        if (position + 1) % 25 == 0:
            logger.info("Scored %d/%d pairs", position + 1, len(indices))

    summary = summarise(per_sample, kernels, percentiles)

    # Control: LR of sample i against the degraded HR of sample partners[i],
    # using the kernel that fitted best, so the floor is measured under the
    # most favourable assumption for the synthetic hypothesis.
    control_stats: Optional[Dict[str, Any]] = None
    control_median: Optional[float] = None
    if bool(audit_cfg.control_mismatched_pairs) and len(indices) > 1:
        best_kernel = str(summary["best_kernel"])
        control_values: List[float] = []
        for position, idx in enumerate(indices):
            if is_stub:
                lr, _, lr_nodata, _ = load_stub_pair(dataset, idx)
            else:
                lr, _, lr_nodata, _ = load_raw_pair(
                    dataset,
                    idx,
                    nodata_value=cfg.dataset.nodata_value,
                    nodata_fill=float(cfg.dataset.nodata_fill),
                )
            other_hr, other_hr_nodata = loaded[partners[position]]
            valid = block_valid_mask(lr_nodata, other_hr_nodata, scale)
            candidate = degrade(other_hr, best_kernel, scale)
            residual = np.where(valid, candidate.astype(np.float64) - lr, 0.0)
            counts = valid.reshape(valid.shape[0], -1).sum(axis=1)
            sums = (residual**2).reshape(residual.shape[0], -1).sum(axis=1)
            with np.errstate(divide="ignore"):
                per_band = 10.0 * np.log10((data_range**2) / (sums / counts))
            control_values.append(float(np.nanmean(per_band)))
        control_stats = {
            f"p{p:g}": float(np.percentile(control_values, p)) for p in percentiles
        }
        control_stats["mean"] = float(np.mean(control_values))
        control_median = float(np.median(control_values))

    std_array = np.asarray(std_ratios, dtype=np.float64)
    marginals = {
        "std_ratio_per_band": [
            float(np.nanmedian(std_array[:, i])) for i in range(len(bands))
        ],
        "mean_abs_diff_per_band": [
            float(np.nanmedian(np.asarray(mean_diffs)[:, i])) for i in range(len(bands))
        ],
        "max_quantile_diff_per_band": [
            float(np.nanmedian(np.asarray(quant_diffs)[:, i])) for i in range(len(bands))
        ],
        "median_std_ratio": float(np.nanmedian(std_array)),
        "quantile_levels": quantiles,
    }

    baseline_median = float(np.median(baselines)) if baselines else None

    # Computed from the storage format and the metric's data range, never
    # configured: it is what uint16 rounding alone would allow.
    ceiling_db = quantisation_ceiling_db(
        float(cfg.dataset.reflectance_scale), data_range
    )

    decision = verdict(
        summary,
        marginals,
        control_median,
        synthetic_psnr_db=float(audit_cfg.synthetic_psnr_db),
        crosssensor_psnr_db=float(audit_cfg.crosssensor_psnr_db),
        variance_ratio_tol=float(audit_cfg.variance_ratio_tol),
        ceiling_db=ceiling_db,
        deterministic_spread_db=float(audit_cfg.deterministic_spread_db),
        quantisation_ceiling_fraction=float(audit_cfg.quantisation_ceiling_fraction),
    )

    print_report(
        summary,
        kernels,
        percentiles,
        control_stats,
        marginals,
        baseline_median,
        decision,
        bands,
        len(indices),
        offset_counts,
    )

    metric_dir = Path(resolve_output_path(cfg, "metric_dir"))
    report_path = metric_dir / str(audit_cfg.report_name)
    csv_path = metric_dir / str(audit_cfg.csv_name)

    report = {
        "dataset": name,
        "subset": str(getattr(dataset, "subset", "")) or None,
        "num_pairs": len(indices),
        "seed": int(cfg.seed),
        "scale": scale,
        "data_range": data_range,
        "bands": bands,
        "kernels": kernels,
        "thresholds": {
            "synthetic_psnr_db": float(audit_cfg.synthetic_psnr_db),
            "crosssensor_psnr_db": float(audit_cfg.crosssensor_psnr_db),
            "variance_ratio_tol": float(audit_cfg.variance_ratio_tol),
            "deterministic_spread_db": float(audit_cfg.deterministic_spread_db),
            "quantisation_ceiling_fraction": float(
                audit_cfg.quantisation_ceiling_fraction
            ),
            "quantisation_ceiling_db": ceiling_db,
        },
        "kernel_summary": summary,
        "nearest_offset_counts": offset_counts,
        "control": control_stats,
        "marginals": marginals,
        "bicubic_upsample_baseline_db": baseline_median,
        "verdict": decision,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("Wrote %s", report_path)

    columns = ["sample_id", "index", "valid_fraction", "nearest_offset"] + [
        f"psnr_{k}" for k in kernels
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in per_sample:
            row = {
                "sample_id": record["sample_id"],
                "index": record["index"],
                "valid_fraction": f"{record['valid_fraction']:.6f}",
                "nearest_offset": record["nearest_offset"],
            }
            for kernel in kernels:
                row[f"psnr_{kernel}"] = f"{record['psnr_db'][kernel]:.4f}"
            writer.writerow(row)
    logger.info("Wrote %s", csv_path)

    print(f"  Report: {report_path}")
    print(f"  Per-sample CSV: {csv_path}")
    print()

    if is_stub:
        # SELF-TEST. The stub's LR is an exact block mean of its HR, so the
        # detector is correct exactly when it says SYNTHETIC and names an
        # averaging kernel. A CROSS_SENSOR verdict here is a broken detector,
        # not clean data.
        passed = decision["verdict"] == "SYNTHETIC" and decision["best_kernel"] in {
            "box",
            "area",
        }
        print("  SELF-TEST (synthetic stub: LR IS an exact block mean of HR)")
        print(
            f"    expected SYNTHETIC via box/area -- got {decision['verdict']} "
            f"via {decision['best_kernel']}: {'PASS' if passed else 'FAIL'}"
        )
        print()
        if passed:
            logger.info("Self-test PASSED: the detector detects a known degradation.")
            return 0
        logger.error(
            "Self-test FAILED: the detector did not identify a degradation that "
            "is synthetic by construction. Its verdict on real data is worthless."
        )
        return 1

    if decision["verdict"] == "CROSS_SENSOR":
        logger.info("Verdict CROSS_SENSOR: %s", decision["reasons"][-1])
        return 0
    logger.error("Verdict %s: %s", decision["verdict"], decision["reasons"][-1])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
