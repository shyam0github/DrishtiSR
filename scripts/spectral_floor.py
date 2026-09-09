"""The irreducible spectral floor: what D(ground-truth HR) vs x_LR already costs.

THE HOSTILE QUESTION THIS ANSWERS. "Your spectral-consistency loss drove
L1(x_LR, D(y_SR)) to 0.019. So what? How close to zero could it possibly have
got?" If the answer is "zero", a run that stops at 0.019 looks like a failure.
It is not, and this script is the evidence.

Our LR is a real Sentinel-2 acquisition. Our HR is NAIP-derived. Two sensors,
two point-spread functions, two acquisition dates, two atmospheric states. So
even a PERFECT super-resolver -- one that returned the ground-truth HR exactly --
would not satisfy the loss, because ``D(y_HR) != x_LR``. There is a floor, it is
a property of the dataset rather than of any model, and it is measurable:
substitute the ground-truth HR for the SR output and compute the identical
terms, through the identical code, over the whole validation split.

``cfg.degradation_audit`` already established the same fact from the other side:
the best-fitting kernel reaches only ~44 dB median against the stored LR, with a
13 dB spread across tiles -- a deterministic resampling would sit near the 90.8
dB quantisation ceiling with a spread under 0.1 dB. These pairs are genuinely
cross-sensor. This script converts that finding into the two numbers the loss is
actually judged in: reflectance L1, and spectral angle.

HOW TO READ THE RESULT.

- A trained run whose ``val_l1_spec`` lands **near the floor** has extracted
  everything the constraint contains. That is the strong claim.
- A run that lands **below the floor** is not doing better than perfect. It is
  reproducing a sensor the HR was never observed by: the spectral term is
  fighting the reconstruction L1, buying agreement with the Sentinel-2 PSF at
  the cost of the high-frequency detail the task exists to produce. Lower
  ``cfg.loss.spectral.lambda1``.
- A run that stays **far above the floor** has not learned the constraint;
  raising lambda1 is justified.

Everything numeric comes from :func:`src.losses.spectral.spectral_terms` with
``cfg.loss.spectral`` settings -- the same function, the same downsampler, the
same eps and clamp the trainer uses -- so the floor and the training curve are
the same quantity and can be plotted on one axis. Nothing here re-implements the
metric.

Examples:
    # Pre-flight on the synthetic stub. Its LR IS an exact block mean of its own
    # HR, so the floor MUST come out at ~0 -- a self-test of the downsampler,
    # the band order and the scale, not a rehearsal.
    .venv/Scripts/python.exe scripts/spectral_floor.py --smoke

    # The real thing: the entire validation split.
    .venv/Scripts/python.exe scripts/spectral_floor.py

    # Quick look at 200 patches while the cache is still filling.
    .venv/Scripts/python.exe scripts/spectral_floor.py --max-patches 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.loader import build_dataloaders  # noqa: E402
from src.losses import spectral_settings_from_cfg, spectral_terms  # noqa: E402
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import resolve_output_path  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the irreducible spectral-consistency floor: L1 and SAM "
            "between the Sentinel-2 LR and the antialiased downsample of the "
            "GROUND-TRUTH HR, over the validation split."
        ),
    )
    add_standard_args(parser)
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help=(
            "Stop after this many validation patches. Default: "
            "cfg.spectral_floor.max_patches (null = the whole split, which is "
            "the number to report)."
        ),
    )
    return parser.parse_args(argv)


def summarise(values: List[float]) -> Dict[str, float]:
    """Mean, std and the percentiles that show whether a mean means anything.

    Args:
        values: One value per validation patch. Must be non-empty and finite.

    Returns:
        ``{"mean", "std", "min", "p5", "p25", "p50", "p75", "p95", "max"}``,
        plain floats. ``std`` is the population standard deviation over the
        patches -- the spread across the split, which is the quantity that says
        whether "the floor" is one number or a range.

    Raises:
        ValueError: ``values`` is empty, or holds a non-finite entry. Neither is
            summarised around: an empty split means the loaders selected
            nothing, and a NaN here would be averaged into a floor that a
            training run is later compared against.
    """
    if not values:
        raise ValueError(
            "no patches were scored, so there is no floor to report. Check "
            "cfg.loader.cached_only and the split file."
        )
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(
            f"{int((~np.isfinite(array)).sum())} of {array.size} patch values "
            "are non-finite. The floor is quoted in the report; it will not be "
            "computed around a NaN."
        )
    percentiles = np.percentile(array, [5, 25, 50, 75, 95])
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p5": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "max": float(array.max()),
    }


def measure_floor(
    val_loader: Any,
    settings: Dict[str, Any],
    max_patches: Optional[int],
    logger: Any,
) -> Dict[str, Any]:
    """Score every validation patch with the HR standing in for the SR output.

    Args:
        val_loader: The validation DataLoader from
            :func:`src.data.loader.build_dataloaders`. Its batches carry ``lr``
            ``(B, C, h, w)`` and ``hr`` ``(B, C, h*scale, w*scale)`` float32
            surface reflectance, nominally ``[0, 1]`` and unclipped, plus
            ``sample_id`` and the patch coordinates.
        settings: Numerics from :func:`src.losses.spectral_settings_from_cfg`.
        max_patches: Stop after this many patches, or None for all of them.
        logger: Logger.

    Returns:
        ``{"rows": [...], "num_patches": int, "truncated": bool}``. Each row is
        one patch: ``sample_id``, ``lr_row``, ``lr_col``, ``l1_spec``
        (reflectance), ``sam_rad``, ``sam_deg``, ``sam_valid_frac``.

    Raises:
        ValueError: Propagated from :func:`src.losses.spectral_terms` -- a
            non-finite tile or a shape mismatch stops the measurement rather
            than being averaged over.
    """
    rows: List[Dict[str, Any]] = []
    truncated = False

    with torch.no_grad():
        for batch in val_loader:
            hr = batch["hr"].float()
            lr = batch["lr"].float()
            # per_sample=True: the floor is reported with a standard deviation,
            # and a batch mean cannot produce one.
            parts = spectral_terms(hr, lr, per_sample=True, **settings)

            ids = batch.get("sample_id", [""] * hr.shape[0])
            lr_rows = batch.get("lr_row", [-1] * hr.shape[0])
            lr_cols = batch.get("lr_col", [-1] * hr.shape[0])
            for i in range(hr.shape[0]):
                sam_rad = float(parts["sam"][i])
                rows.append(
                    {
                        "sample_id": str(ids[i]),
                        "lr_row": int(lr_rows[i]),
                        "lr_col": int(lr_cols[i]),
                        "l1_spec": float(parts["l1_spec"][i]),
                        "sam_rad": sam_rad,
                        "sam_deg": math.degrees(sam_rad),
                        "sam_valid_frac": float(parts["sam_valid_frac"][i]),
                    }
                )
                if max_patches is not None and len(rows) >= max_patches:
                    truncated = True
                    break
            if truncated:
                break

            if len(rows) % 500 == 0:
                logger.info("Scored %d patches.", len(rows))

    logger.info("Scored %d validation patches%s.", len(rows),
                " (truncated by --max-patches)" if truncated else "")
    return {"rows": rows, "num_patches": len(rows), "truncated": truncated}


def print_report(
    result: Dict[str, Any],
    summaries: Dict[str, Dict[str, float]],
    settings: Dict[str, Any],
    lambdas: Dict[str, float],
    reference_l1: float,
    smoke: bool,
) -> None:
    """Print the floor, and what it implies for the lambdas, to stdout.

    Args:
        result: The dict from :func:`measure_floor`.
        summaries: ``{"l1_spec": {...}, "sam_rad": {...}, "sam_deg": {...}}``
            from :func:`summarise`.
        settings: The loss numerics actually used, printed so the number can
            never be quoted without the operator that produced it.
        lambdas: ``{"lambda1": float, "lambda2": float}`` from
            ``cfg.loss.spectral`` -- the candidate weights being sanity-checked
            against the floor, NOT the trainer's defaults (which are 0.0).
        reference_l1: ``cfg.spectral_floor.reference_l1``, the main
            reconstruction L1 in reflectance units. The lambda decision is a
            comparison of two loss magnitudes, so the report states the ratio
            rather than leaving a reader to divide two numbers that live in
            different files.
        smoke: Whether this was a ``--smoke`` run, which must be said in the
            output itself and not only in the filename.
    """
    l1 = summaries["l1_spec"]
    rad = summaries["sam_rad"]
    deg = summaries["sam_deg"]

    print("\n### Irreducible spectral floor")
    print(
        f"    {result['num_patches']} validation patches, "
        f"D = {settings['mode']}, eps = {settings['eps']:g}, "
        f"cos_clamp = {settings['cos_clamp']:g}, "
        f"min_norm = {settings['min_norm']:g}"
    )
    if smoke:
        print(
            "    SMOKE RUN on the synthetic stub. Its LR is an exact block mean\n"
            "    of its own HR, so this floor describes that construction and\n"
            "    says NOTHING about SEN2NAIPv2. A near-zero L1 here is the\n"
            "    self-test passing, not a finding."
        )
    if result["truncated"]:
        print(
            "    TRUNCATED by --max-patches: this is not the floor over the "
            "full split."
        )

    print("\n| term | mean | std | p5 | p50 | p95 | max |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    print(
        f"| `l1_spec` (reflectance) | {l1['mean']:.6f} | {l1['std']:.6f} | "
        f"{l1['p5']:.6f} | {l1['p50']:.6f} | {l1['p95']:.6f} | {l1['max']:.6f} |"
    )
    print(
        f"| `sam` (rad) | {rad['mean']:.6f} | {rad['std']:.6f} | "
        f"{rad['p5']:.6f} | {rad['p50']:.6f} | {rad['p95']:.6f} | {rad['max']:.6f} |"
    )
    print(
        f"| `sam` (deg) | {deg['mean']:.4f} | {deg['std']:.4f} | "
        f"{deg['p5']:.4f} | {deg['p50']:.4f} | {deg['p95']:.4f} | {deg['max']:.4f} |"
    )

    lam1, lam2 = lambdas["lambda1"], lambdas["lambda2"]
    weighted_l1 = lam1 * l1["mean"]
    weighted_sam = lam2 * rad["mean"]
    contribution = weighted_l1 + weighted_sam
    print(
        f"\n> THE FLOOR: **l1_spec {l1['mean']:.6f} +- {l1['std']:.6f}** "
        f"reflectance, **sam {rad['mean']:.6f} +- {rad['std']:.6f}** rad "
        f"({deg['mean']:.4f} deg), over {result['num_patches']} patches."
    )
    print(
        f"> At the candidate weights (lambda1={lam1}, lambda2={lam2}) a PERFECT "
        f"super-resolver still pays {contribution:.6f} of loss it cannot "
        "remove. Judge a trained run's val_l1_spec and val_sam against these "
        "numbers, not against zero."
    )

    # The lambda decision, made arithmetic. The spectral term is a regulariser
    # only while it is small against the objective it regularises; past that it
    # IS the objective, and the objective it displaces is the one that produces
    # detail.
    print(f"\n### Is lambda1={lam1}, lambda2={lam2} the right weight?")
    print(
        f"    reference L1(sr, hr) = {reference_l1:.6f} reflectance "
        "(cfg.spectral_floor.reference_l1)"
    )
    print(
        f"    lambda1 * l1_spec floor = {weighted_l1:.6f} = "
        f"{100.0 * weighted_l1 / reference_l1:.1f}% of it"
    )
    print(
        f"    lambda2 * sam floor     = {weighted_sam:.6f} = "
        f"{100.0 * weighted_sam / reference_l1:.1f}% of it"
    )
    share = 100.0 * contribution / (reference_l1 + contribution)
    print(
        f"    together                = {contribution:.6f}, i.e. {share:.1f}% "
        "of the total loss at the floor is a constant NO model can remove"
    )
    print(
        "\n> THE DEGENERATE SOLUTION, and why that ratio matters. The spectral "
        "term alone is minimised by an output that averages EXACTLY to the LR "
        "-- a blurry, bicubic-like upsample scores near 0 on it, far BELOW the "
        f"{l1['mean']:.6f} floor a perfectly HR-faithful output pays. So the "
        "term does not merely regularise: it pulls towards doing no "
        "super-resolution at all, and lambda1 sets how hard."
    )
    print(
        "> Below the floor is therefore NOT better than perfect. It means the "
        "spectral term has overridden the reconstruction L1 to match a sensor "
        "the HR was never observed by, paid for in the high-frequency detail "
        "the task exists to produce. Watch val_l1_spec against the floor from "
        "the first validation, and lower lambda1 if it goes under."
    )

    valid = min(row["sam_valid_frac"] for row in result["rows"])
    if valid < 1.0:
        print(
            f"\n> Note: the worst patch had only {valid:.4f} of its pixels with "
            "a defined spectral angle (the rest are nodata filled with "
            "cfg.dataset.nodata_fill). Those pixels are excluded from SAM, not "
            "scored as 0 degrees."
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("spectral_floor", log_file=cfg.paths.log_file)
    seed = seed_everything(cfg.seed)
    logger.info("Seeded with %d", seed)

    threads = int(cfg.runtime.num_threads)
    torch.set_num_threads(threads)
    logger.info(
        "torch threads set to cfg.runtime.num_threads=%d; CUDA available: %s. "
        "This measurement is CPU-only by design -- it is a property of the "
        "dataset and must be reproducible on the local machine.",
        threads,
        torch.cuda.is_available(),
    )

    settings = spectral_settings_from_cfg(cfg)
    lambdas = {
        "lambda1": float(cfg.loss.spectral.lambda1),
        "lambda2": float(cfg.loss.spectral.lambda2),
    }
    logger.info(
        "Spectral settings from cfg.loss.spectral: %s; candidate lambdas %s.",
        settings,
        lambdas,
    )

    block = cfg["spectral_floor"]
    reference_l1 = float(block["reference_l1"])
    max_patches = (
        args.max_patches if args.max_patches is not None
        else (None if block["max_patches"] is None else int(block["max_patches"]))
    )
    if max_patches is not None and max_patches <= 0:
        raise ValueError(
            f"--max-patches must be positive; got {max_patches}. Omit it to "
            "score the whole validation split."
        )

    loaders = build_dataloaders(cfg, logger=logger)
    logger.info(
        "Validation split: %d patches from %d samples (%s).",
        len(loaders["datasets"]["val"]),
        len(loaders["indices"]["val"]),
        loaders["split_source"],
    )

    result = measure_floor(loaders["val"], settings, max_patches, logger)
    summaries = {
        "l1_spec": summarise([row["l1_spec"] for row in result["rows"]]),
        "sam_rad": summarise([row["sam_rad"] for row in result["rows"]]),
        "sam_deg": summarise([row["sam_deg"] for row in result["rows"]]),
    }

    metric_dir = Path(resolve_output_path(cfg, "metric_dir"))
    json_path = metric_dir / str(block["json_name"])
    csv_path = metric_dir / str(block["csv_name"])

    payload = {
        "what": (
            "L1 and SAM between the Sentinel-2 LR and the antialiased "
            "downsample of the GROUND-TRUTH HR. The irreducible floor of the "
            "spectral-consistency loss: no super-resolver can go below it "
            "without contradicting the HR."
        ),
        "dataset": str(cfg.dataset.name),
        "bands": list(cfg.dataset.bands),
        "scale": int(cfg.sr.scale),
        "split": str(cfg.loader.val_split),
        "split_source": loaders["split_source"],
        "num_patches": result["num_patches"],
        "num_val_samples": len(loaders["indices"]["val"]),
        "truncated": result["truncated"],
        "max_patches": max_patches,
        "smoke": bool(args.smoke),
        "seed": seed,
        "settings": settings,
        "candidate_lambdas": lambdas,
        "floor": summaries,
        "reference_l1": reference_l1,
        "loss_at_candidate_lambdas": (
            lambdas["lambda1"] * summaries["l1_spec"]["mean"]
            + lambdas["lambda2"] * summaries["sam_rad"]["mean"]
        ),
        "units": {
            "l1_spec": "surface reflectance, unclipped",
            "sam_rad": "radians (what the training loss uses)",
            "sam_deg": "degrees (what src/metrics/image_quality.sam reports)",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    logger.info("Wrote %s and %s", json_path, csv_path)

    print_report(
        result, summaries, settings, lambdas, reference_l1, bool(args.smoke)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
