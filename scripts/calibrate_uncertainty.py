"""Is TTA disagreement monotone against error? Calibration over the full val split.

Runs the 8-way dihedral TTA fallback (``src/uncertainty.py``) on ANY
``src/train.py`` checkpoint, bins every validation pixel by the resulting
per-pixel standard deviation, and reports mean absolute error per bin -- plus
the same for a texture CONTROL (gradient magnitude of the SR output), and the
sparsification curves of both against the oracle. See ``src/eval/calibration.py``
for why the control is there and how the binning stays exact without storing
314 M pixels.

The error is that of the TTA MEAN, because the mean is what ships with the
std raster. The identity member of the TTA stack -- the plain forward pass --
is scored alongside so the PSNR cost or gain of shipping the mean is on record.

Parameterised by checkpoint, so re-running on B1 (or Run C's SR output) is one
command. Outputs are named by ``--tag`` (default ``<run dir>_<checkpoint stem>``):

- ``reports/day3_tta_calibration_<tag>.md``   -- verdict, tables.
- ``reports/day3_tta_calibration_<tag>.json`` -- every number.
- ``reports/figures/day3_tta_calibration_<tag>.png`` -- the curve.

Examples:
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/calibrate_uncertainty.py --smoke
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/calibrate_uncertainty.py --checkpoint runs/runA/best.pt
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/calibrate_uncertainty.py --checkpoint runs/day3/b1/last.pt
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.loader import build_dataloaders  # noqa: E402
from src.data.registry import get_dataset  # noqa: E402
from src.eval.calibration import (  # noqa: E402
    PredictorHistogram,
    ause,
    equal_mass_bins,
    gradient_magnitude,
    monotonicity,
    plot_calibration,
    sparsification_curve,
)
from src.metrics.image_quality import psnr  # noqa: E402
from src.models.edsr import build_model, count_params, model_from_checkpoint  # noqa: E402
from src.uncertainty import tta_predict  # noqa: E402
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TTA-disagreement calibration: MAE per uncertainty bin over the val split."
    )
    add_standard_args(parser)
    parser.add_argument("--checkpoint", default=None,
                        help="src/train.py checkpoint. Overrides cfg.calibrate_uncertainty.checkpoint.")
    parser.add_argument("--tag", default=None,
                        help="Output name tag. Default: <checkpoint parent dir>_<checkpoint stem>.")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="DEBUGGING ONLY: stop after N val batches; outputs get a _truncatedN suffix.")
    return parser.parse_args(argv)


def _path(raw: Any) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root() / path


def fabricate_checkpoint(cfg: Any, path: Path, logger: Any) -> Path:
    """Write a tiny random-weight checkpoint in src/train.py's format (smoke only).

    Args:
        cfg: Smoke-merged config.
        path: Destination ``.pt``.
        logger: Logger.

    Returns:
        ``path``.

    Raises:
        RuntimeError: ``cfg.calibrate_uncertainty.fabricate_checkpoint`` is not
            set -- this writes noise and must never replace a real checkpoint.
    """
    if not bool(cfg["calibrate_uncertainty"].get("fabricate_checkpoint", False)):
        raise RuntimeError("fabricate_checkpoint() without cfg.calibrate_uncertainty.fabricate_checkpoint.")
    channels = len(cfg["dataset"]["bands"])
    args = {"scale": int(cfg["sr"]["scale"]), "n_resblocks": 2, "n_feats": 16,
            "in_ch": channels, "fabricated_by": "scripts/calibrate_uncertainty.py --smoke"}
    model = build_model("edsr_baseline", scale=args["scale"], n_resblocks=2, n_feats=16,
                        in_ch=channels, out_ch=channels)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "it": 0, "best": 0.0, "args": args}, path)
    logger.warning("SMOKE: fabricated a RANDOM-WEIGHT checkpoint at %s.", path)
    return path


def summarise(hist: PredictorHistogram, oracle: PredictorHistogram, n_bins: int,
              fractions: np.ndarray, n_bands: int) -> Dict[str, Any]:
    """Pooled and per-band bins, monotonicity and sparsification for one predictor."""
    pooled = equal_mass_bins(hist, n_bins)
    curve = sparsification_curve(hist, fractions)
    per_band = []
    for band in range(n_bands):
        bins_b = equal_mass_bins(hist, n_bins, band=band)
        per_band.append({
            "monotonicity": monotonicity(bins_b),
            "ause": ause(sparsification_curve(hist, fractions, band=band),
                         sparsification_curve(oracle, fractions, band=band), fractions),
        })
    return {
        "bins": pooled,
        "monotonicity": monotonicity(pooled),
        "sparsification": curve.tolist(),
        "ause": ause(curve, sparsification_curve(oracle, fractions), fractions),
        "per_band": per_band,
    }


def build_report(result: Dict[str, Any], bands: List[str]) -> str:
    """Markdown: verdict first, then the curve as a table, then the controls."""
    tta = result["predictors"]["tta_std"]
    ctl = result["predictors"]["sr_gradient"]
    mono = tta["monotonicity"]
    verdict = (
        f"**TTA disagreement IS monotone against error** on `{result['tag']}`: MAE rises "
        f"in every one of {mono['n_bins'] - 1} steps across {mono['n_bins']} equal-mass "
        f"bins (Spearman {mono['spearman']:.3f}); the most-uncertain bin's MAE is "
        f"{mono['mae_ratio_top_bottom']:.1f}x the least-uncertain's."
        if mono["monotone"] else
        f"**TTA disagreement is NOT strictly monotone against error** on `{result['tag']}`: "
        f"{mono['inversions']} inversion(s) in {mono['n_bins'] - 1} steps "
        f"(Spearman {mono['spearman']:.3f}, top/bottom MAE ratio "
        f"{mono['mae_ratio_top_bottom']:.1f}x)."
    )
    beats = tta["ause"]["ause"] < ctl["ause"]["ause"]
    lines = [
        f"# TTA-disagreement calibration — `{result['tag']}`",
        "",
        verdict,
        "",
        f"Against the texture control (gradient magnitude of the SR): AUSE "
        f"{tta['ause']['ause']:.4f} vs {ctl['ause']['ause']:.4f} "
        f"(random removal {tta['ause']['ause_random']:.4f}; ratio to random "
        f"{tta['ause']['ause_ratio']:.3f} vs {ctl['ause']['ause_ratio']:.3f}). "
        + ("TTA ranks error **better** than the edge-detector control."
           if beats else
           "TTA ranks error **no better** than the edge-detector control."),
        "",
        f"_Written {result['written_utc']}._ Checkpoint `{result['checkpoint']['path']}` "
        f"(iteration {result['checkpoint']['iteration']}, {result['checkpoint']['parameters']:,} "
        f"params, sha256 `{result['checkpoint']['sha256'][:12]}`). Split `{result['split']}`: "
        f"{result['num_patches']} patches, {result['num_pixels']:,} pixel-band values. "
        f"Error is |TTA mean − HR| in reflectance.",
        "",
        f"![calibration]({Path(result['figure']).name})" if not result["smoke"] else "",
        "",
        "## MAE per equal-mass bin of TTA std (pooled bands)",
        "",
        "| bin | mass | TTA std range (refl.) | mean std | MAE | RMSE | RMSE / mean std | implied logvar ln(RMSE²) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, b in enumerate(tta["bins"]):
        hi = "∞" if b["u_hi"] is None else f"{b['u_hi']:.2e}"
        lines.append(
            f"| {i} | {b['mass']:.3f} | {b['u_lo']:.2e} – {hi} | {b['u_mean']:.2e} | "
            f"{b['mae']:.5f} | {b['rmse']:.5f} | {b['rmse'] / b['u_mean'] if b['u_mean'] > 0 else float('nan'):.2f} | "
            f"{2.0 * math.log(b['rmse']) if b['rmse'] > 0 else float('-inf'):.2f} |"
        )
    lines += [
        "",
        "## Per band",
        "",
        "| band | TTA monotone | inversions | Spearman | top/bottom | AUSE ratio TTA | AUSE ratio control |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, name in enumerate(bands):
        m, a = tta["per_band"][i]["monotonicity"], tta["per_band"][i]["ause"]
        c = ctl["per_band"][i]["ause"]
        lines.append(
            f"| {name} | {'yes' if m['monotone'] else 'NO'} | {m['inversions']} | "
            f"{m['spearman']:.3f} | {m['mae_ratio_top_bottom']:.1f}x | "
            f"{a['ause_ratio']:.3f} | {c['ause_ratio']:.3f} |"
        )
    cm = ctl["monotonicity"]
    ps = result["psnr"]
    clamp = result["logvar_clamp_check"]
    lines += [
        "",
        f"Control monotonicity (pooled): {cm['inversions']} inversion(s), Spearman "
        f"{cm['spearman']:.3f}, top/bottom {cm['mae_ratio_top_bottom']:.1f}x.",
        "",
        "## Side numbers",
        "",
        f"- **PSNR** (per-patch mean, all bands, data_range {ps['data_range']}): TTA mean "
        f"{ps['tta_mean']:.4f} dB vs plain forward {ps['identity']:.4f} dB "
        f"({ps['tta_mean'] - ps['identity']:+.4f} dB). Diagnostic only; the headline "
        "table's numbers come from scripts/eval_all_ckpts.py.",
        f"- **Day 4 clamp check.** {100 * clamp['mass_below_min']:.1f}% of pixel mass sits in "
        f"TTA bins whose implied log-variance ln(RMSE²) is below the head's clamp floor "
        f"{clamp['logvar_min']} (sigma {math.exp(clamp['logvar_min'] / 2):.4f} refl.). "
        "Approximate -- within-bin spread makes the true conditional variances wider -- "
        "but it says how much of the raster a well-trained head would pin to the floor.",
        "",
        "## Reproduce",
        "",
        "```",
        f"D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/calibrate_uncertainty.py "
        f"--checkpoint {result['checkpoint']['path']}",
        "```",
    ]
    if result["smoke"]:
        lines.insert(0, "> **SMOKE: synthetic stub, random weights. These numbers mean nothing.**\n")
    if result["max_batches"] is not None:
        lines.append(f"\n> Truncated to {result['max_batches']} batches: NOT the full split.\n")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)
    logger = get_logger("calibrate_uncertainty", log_file=cfg.paths.log_file)
    seed_everything(cfg.seed)
    torch.set_num_threads(int(cfg.runtime.num_threads))
    block = cfg["calibrate_uncertainty"]

    ckpt = _path(args.checkpoint if args.checkpoint is not None else block["checkpoint"])
    if args.smoke and bool(block.get("fabricate_checkpoint", False)):
        fabricate_checkpoint(cfg, ckpt, logger)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint {ckpt} does not exist.")
    tag = args.tag or f"{ckpt.parent.name}_{ckpt.stem}"
    suffix = "" if args.max_batches is None else f"_truncated{args.max_batches}"

    def out_path(key: str) -> Path:
        raw = _path(str(block[key]).format(tag=tag))
        return raw.with_name(f"{raw.stem}{suffix}{raw.suffix}")

    model, payload = model_from_checkpoint(ckpt, device="cpu")
    params = count_params(model)
    if params > int(cfg.runtime.max_parameters):
        logger.warning(
            "%s has %d parameters, over cfg.runtime.max_parameters=%d. Calibrating a "
            "REFERENCE model (Run A is 64 features); it does not ship at this size.",
            ckpt, params, int(cfg.runtime.max_parameters),
        )
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    logger.info("Loaded %s (iteration %s, %d params, sha256 %s).",
                ckpt, payload.get("it"), params, sha[:12])

    dataset = get_dataset(cfg)
    loaders = build_dataloaders(cfg, dataset=dataset, logger=logger)
    val_loader = loaders["val"]
    n_patches = len(loaders["datasets"]["val"])
    bands = [str(b) for b in cfg.dataset.bands]
    data_range = float(cfg.metrics.data_range)

    grid = block["fine_bins"]
    make = lambda: PredictorHistogram(len(bands), float(grid["log10_min"]),  # noqa: E731
                                      float(grid["log10_max"]), int(grid["n"]))
    hist_tta, hist_grad, hist_oracle = make(), make(), make()
    psnr_tta: List[float] = []
    psnr_id: List[float] = []
    chunk = block.get("chunk_size")
    chunk = None if chunk is None else int(chunk)
    n_batches = len(val_loader) if args.max_batches is None else min(len(val_loader), args.max_batches)
    progress_every = int(block["progress_every"])

    logger.info("TTA x8 over %d val patches in %d batches (chunk_size=%s).",
                n_patches, n_batches, chunk)
    t0 = time.time()
    seen = 0
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        lr, hr = batch["lr"].float(), batch["hr"].float()
        res = tta_predict(model, lr, chunk_size=chunk, return_members=True)
        err = res.mean - hr
        hist_tta.update(res.std, err)
        hist_grad.update(gradient_magnitude(res.mean), err)
        hist_oracle.update(err.abs(), err)
        psnr_tta += list(np.atleast_1d(psnr(res.mean, hr, data_range=data_range).mean))
        psnr_id += list(np.atleast_1d(psnr(res.members[0], hr, data_range=data_range).mean))
        seen += lr.shape[0]
        if (i + 1) % progress_every == 0 or i + 1 == n_batches:
            elapsed = time.time() - t0
            logger.info("batch %d/%d, %d patches, %.0fs elapsed, ETA %.0fs.",
                        i + 1, n_batches, seen, elapsed, elapsed / (i + 1) * (n_batches - i - 1))

    n_bins = int(block["n_bins"])
    step = float(block["sparsification_step"])
    fractions = np.arange(0.0, float(block["sparsification_max"]) + 1e-12, step)
    oracle_curve = sparsification_curve(hist_oracle, fractions)
    predictors = {
        "tta_std": summarise(hist_tta, hist_oracle, n_bins, fractions, len(bands)),
        "sr_gradient": summarise(hist_grad, hist_oracle, n_bins, fractions, len(bands)),
    }

    logvar_min = float(cfg.uncertainty.logvar_min)
    below = sum(b["mass"] for b in predictors["tta_std"]["bins"]
                if b["rmse"] > 0 and 2.0 * math.log(b["rmse"]) < logvar_min)

    fig_cfg = block["figure"]
    figure = out_path("figure_path")
    plot_calibration(
        [
            {"display": "TTA std", "color": fig_cfg["colors"]["tta_std"],
             "bins": predictors["tta_std"]["bins"],
             "sparsification": predictors["tta_std"]["sparsification"]},
            {"display": "SR gradient (control)", "color": fig_cfg["colors"]["sr_gradient"],
             "bins": predictors["sr_gradient"]["bins"],
             "sparsification": predictors["sr_gradient"]["sparsification"]},
        ],
        fractions, oracle_curve, figure, fig_cfg,
        title=f"TTA-disagreement calibration — {tag}, {seen} val patches"
              + (" (SMOKE)" if args.smoke else ""),
    )

    result = {
        "written_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "smoke": bool(args.smoke),
        "max_batches": args.max_batches,
        "tag": tag,
        "checkpoint": {
            "path": ckpt.relative_to(repo_root()).as_posix() if ckpt.is_relative_to(repo_root()) else str(ckpt),
            "sha256": sha,
            "iteration": int(payload["it"]) + 1 if "it" in payload else None,
            "args": dict(payload["args"]),
            "lambdas": payload.get("lambdas"),
            "git_sha": payload.get("git_sha"),
            "parameters": int(params),
        },
        "split": str(cfg.loader.val_split),
        "split_source": str(loaders["split_source"]),
        "num_patches": int(seen),
        "num_pixels": int(hist_tta.count.sum().item()),
        "bands": bands,
        "error": "abs(TTA mean - HR), reflectance",
        "tta": {"transforms": 8, "std": "population (correction=0)", "chunk_size": chunk},
        "settings": {"n_bins": n_bins, "fine_bins": dict(grid), "fractions": fractions.tolist()},
        "oracle_sparsification": oracle_curve.tolist(),
        "predictors": predictors,
        "psnr": {"data_range": data_range, "tta_mean": float(np.mean(psnr_tta)),
                 "identity": float(np.mean(psnr_id))},
        "logvar_clamp_check": {"logvar_min": logvar_min, "mass_below_min": float(below)},
        "figure": figure.relative_to(repo_root()).as_posix(),
        "elapsed_s": time.time() - t0,
    }
    report_json, report_md = out_path("report_json"), out_path("report_md")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_md.write_text(build_report(result, bands), encoding="utf-8")

    mono = predictors["tta_std"]["monotonicity"]
    logger.info(
        "TTA std: monotone=%s inversions=%d/%d spearman=%.3f top/bottom=%.2fx AUSE ratio=%.3f "
        "(control %.3f). Figure %s, report %s.",
        mono["monotone"], mono["inversions"], mono["n_bins"] - 1, mono["spearman"],
        mono["mae_ratio_top_bottom"], predictors["tta_std"]["ause"]["ause_ratio"],
        predictors["sr_gradient"]["ause"]["ause_ratio"], figure, report_md,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
