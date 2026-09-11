"""Day 3 A2-vs-B read-out: blur diagnostic on the full val split, verdict, A2
overfitting check, and the validation-curve figure.

REUSES rather than recomputes. ``scripts/eval_all_ckpts.py`` has already scored
every checkpoint on the frozen validation split and cached one per-pair table
per checkpoint: PSNR, SSIM, LPIPS, SAM, ERGAS, opensr-test consistency and
L1_spec. Its selection rule (``cfg.eval_all_ckpts.selection``, written before
any full-split result existed) is re-applied here to those cached tables,
unchanged. This script adds only what that pass never measured: the Sobel
gradient magnitude and the HF energy ratio (``src/metrics/sharpness.py``) per
pair, for the GT HR, bicubic, and each run's selected AND final checkpoint --
so the blur index is read on the same 1199 patches as every other column.

A stale or missing eval_all_ckpts table is an error, not a recompute: this
script must never quietly produce numbers from a different pass.

Examples:
    python scripts/day3_ab_report.py --limit 20   # time 20 patches, project
    python scripts/day3_ab_report.py              # full split, writes reports
    python scripts/eval_all_ckpts.py --smoke && python scripts/day3_ab_report.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from scripts.eval_all_ckpts import (  # noqa: E402
    _item,
    _path,
    cached,
    discover_checkpoints,
    settings_fingerprint,
)
from scripts.eval_runA import load_checkpoint_model  # noqa: E402
from src.data.loader import build_dataloaders  # noqa: E402
from src.data.registry import get_dataset  # noqa: E402
from src.eval.baselines import get_baseline  # noqa: E402
from src.eval.ckpt_selection import PAIR_KEY, select_checkpoint  # noqa: E402
from src.metrics.sharpness import gradient_magnitude, hf_energy_ratio  # noqa: E402
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

# Columns read from the eval_all_ckpts per-pair tables: output name -> column.
TABLE_COLUMNS = {
    "psnr_db": "psnr_mean",
    "ssim": "ssim_mean",
    "lpips": "lpips",
    "sam_deg": "sam_mean_deg",
    "ergas": "ergas",
    "consistency_opensr_l1": "reflectance",
    "consistency_l1_spec_area": "l1_spec",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    parser.add_argument("--limit", type=int, default=None,
                        help="Timing pass: score this many val patches, log the "
                             "projected full-split time, write no report.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute the sharpness pass even on a cache hit.")
    return parser.parse_args(argv)


def load_frames(cache_root: Path, label: str, checkpoints: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """The eval_all_ckpts per-pair table for every checkpoint of one run.

    Raises:
        FileNotFoundError: A checkpoint has no cached table.
        ValueError: The cached table was computed from different weights.
    """
    frames = {}
    for ck in checkpoints:
        folder = cache_root / label / ck["label"]
        key_path, csv_path = folder / "key.json", folder / "per_pair.csv"
        if not (key_path.is_file() and csv_path.is_file()):
            raise FileNotFoundError(
                f"No eval_all_ckpts table for {label}/{ck['label']} at {folder}. "
                "Run scripts/eval_all_ckpts.py (with --smoke if this is --smoke) first."
            )
        stored = json.loads(key_path.read_text(encoding="utf-8"))
        if stored.get("weights_sha256") != ck["weights_sha256"]:
            raise ValueError(
                f"{folder} was scored from weights {stored.get('weights_sha256')} but "
                f"{ck['path']} holds {ck['weights_sha256']}. Re-run eval_all_ckpts."
            )
        frames[ck["label"]] = pd.read_csv(csv_path, dtype={"sample_id": str})
    return frames


def check_metadata(run_dir: Path, label: str, expected: Any, lambdas: Dict[str, float],
                   params: int) -> Dict[str, Any]:
    """Compare a run's recorded provenance with what the comparison assumes.

    Every mismatch is returned AND logged by the caller; nothing is dropped.

    Returns:
        ``{"values": {...}, "flags": [str, ...]}``. An absent metadata file is
        itself a flag.
    """
    flags: List[str] = []
    values: Dict[str, Any] = {"params": params}
    meta_path, summary_path = run_dir / "run_metadata.json", run_dir / "run_summary.json"
    if not meta_path.is_file() or not summary_path.is_file():
        return {"values": values, "flags": [f"{label}: run_metadata.json/run_summary.json absent"]}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    values.update({
        "config_hash": meta["config_hash"], "git_sha": meta["git_sha"],
        "git_dirty": meta["git_dirty"], "lambda1": meta["lambdas"]["spectral_lambda1"],
        "lambda2": meta["lambdas"]["spectral_lambda2"],
        "iters_completed": summary["iters_completed"],
    })
    if not str(meta["config_hash"]).startswith(str(expected.config_hash_prefix)):
        flags.append(f"{label}: config hash {meta['config_hash'][:12]} != {expected.config_hash_prefix}")
    if not str(meta["git_sha"]).startswith(str(expected.git_sha)):
        flags.append(f"{label}: git SHA {meta['git_sha'][:7]} != expected {expected.git_sha}")
    if bool(meta["git_dirty"]):
        flags.append(f"{label}: git tree was dirty")
    want = lambdas
    if (float(values["lambda1"]), float(values["lambda2"])) != (float(want["lambda1"]), float(want["lambda2"])):
        flags.append(f"{label}: lambdas {values['lambda1']}/{values['lambda2']} != "
                     f"{want['lambda1']}/{want['lambda2']}")
    if int(summary["iters_completed"]) != int(expected.iterations):
        flags.append(f"{label}: {summary['iters_completed']} iterations != {expected.iterations}")
    if int(params) != int(expected.parameters):
        flags.append(f"{label}: {params} parameters != {expected.parameters}")
    return {"values": values, "flags": flags}


def sharpness_pass(loader: Any, models: Dict[str, torch.nn.Module], bicubic: Any,
                   cutoff: float, limit: Optional[int], logger: Any) -> pd.DataFrame:
    """Per-pair Sobel gradient magnitude and HF energy ratio, one loader pass.

    Args:
        loader: The validation DataLoader. Batches carry ``lr`` ``(B, C, h, w)``
            and ``hr`` ``(B, C, 4h, 4w)``, float32 surface reflectance,
            nominally [0, 1] and unclipped, plus the ``PAIR_KEY`` fields.
        models: Name -> model in eval mode on CPU, reflectance in and out.
        bicubic: ``get_baseline("bicubic", scale)`` sr_fn.
        cutoff: HF cutoff as a fraction of Nyquist -- the value the runs logged with.
        limit: Stop after this many patches (timing pass), or None for all.

    Returns:
        One row per patch: the PAIR_KEY fields, then ``sobel_<m>`` (reflectance
        per pixel) and ``hf_<m>`` (dimensionless, [0, 1]) for m in
        ``gt``, ``bicubic`` and every model name. Nothing is clipped.
    """
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            lr, hr = batch["lr"].float(), batch["hr"].float()
            outputs = {"gt": hr}
            bic = bicubic(lr)
            outputs["bicubic"] = torch.from_numpy(np.ascontiguousarray(bic)) if isinstance(bic, np.ndarray) else bic
            for name, model in models.items():
                sr = model(lr)
                if not isinstance(sr, torch.Tensor):
                    raise TypeError(f"{name} returned {type(sr).__name__}, expected a tensor.")
                outputs[name] = sr
            terms = {m: (gradient_magnitude(t.float()), hf_energy_ratio(t.float(), cutoff=cutoff))
                     for m, t in outputs.items()}
            for i in range(int(lr.shape[0])):
                row: Dict[str, Any] = {k: _item(batch[k][i]) for k in PAIR_KEY}
                row["sample_id"] = str(row["sample_id"])
                for m, (sob, hf) in terms.items():
                    row[f"sobel_{m}"] = float(sob[i])
                    row[f"hf_{m}"] = float(hf[i])
                rows.append(row)
            if len(rows) % 200 < int(lr.shape[0]):
                logger.info("sharpness pass: %d patches.", len(rows))
            if limit is not None and len(rows) >= limit:
                break
    return pd.DataFrame(rows)


def overfitting(log_csv: Path, window: int) -> Dict[str, Any]:
    """Best-vs-final and last-``window``-iteration trend of the control's val curve.

    "Sustained decline" for a metric = the least-squares slope over the window
    points in the worse direction AND the window's last value is worse than its
    first. Read from the training loop's own validation (log.csv).
    """
    curve = pd.read_csv(log_csv).dropna(subset=["val_psnr"])
    final_it = int(curve["iter"].max())
    out: Dict[str, Any] = {"final_iter": final_it, "window_iters": window}
    for col, better in (("val_psnr", "higher"), ("val_lpips", "lower")):
        best_idx = curve[col].idxmax() if better == "higher" else curve[col].idxmin()
        final = float(curve.loc[curve["iter"] == final_it, col].iloc[0])
        win = curve[curve["iter"] >= final_it - window]
        slope = float(np.polyfit(win["iter"] / 1000.0, win[col], 1)[0])
        first, last = float(win[col].iloc[0]), float(win[col].iloc[-1])
        worse_slope = slope < 0 if better == "higher" else slope > 0
        worse_end = last < first if better == "higher" else last > first
        out[col] = {
            "best_iter": int(curve.loc[best_idx, "iter"]), "best": float(curve.loc[best_idx, col]),
            "final": final, "final_minus_best": final - float(curve.loc[best_idx, col]),
            "window_first_iter": int(win["iter"].iloc[0]), "window_first": first,
            "window_slope_per_1k": slope, "sustained_decline": bool(worse_slope and worse_end),
        }
    out["overfitting_controlled"] = not (out["val_psnr"]["sustained_decline"]
                                         or out["val_lpips"]["sustained_decline"])
    return out


def plot_curves(curves: Dict[str, pd.DataFrame], refs: Dict[str, Dict[str, float]],
                displays: Dict[str, str], style: Any, out: Path) -> None:
    """A2 vs B1 on the training-loop validation: PSNR, LPIPS, HF ratio vs GT.

    The HF panel carries the GT (1.0) and bicubic reference lines from each
    run's sharpness_reference.json, measured on the same val batches as the
    curve. PSNR and LPIPS have no reference line from that subset, so none is
    drawn rather than one borrowed from a different evaluation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = list(style.colors)
    fig, axes = plt.subplots(1, 3, figsize=tuple(style.figsize), facecolor=style.surface)
    panels = (("val_psnr", "PSNR (dB) ↑"), ("val_lpips", "LPIPS ↓"),
              ("hf_ratio", "HF energy ratio vs GT ↑"))
    for ax, (col, title) in zip(axes, panels):
        ax.set_facecolor(style.surface)
        for n, (label, curve) in enumerate(curves.items()):
            y = curve[col] if col != "hf_ratio" else curve["val_hf_energy"] / refs[label]["hr_hf_energy"]
            ax.plot(curve["iter"], y, color=colors[n], linewidth=2, label=displays[label])
        if col == "hf_ratio":
            ref = next(iter(refs.values()))
            bic = ref["bicubic_hf_energy"] / ref["hr_hf_energy"]
            ax.axhline(1.0, color=style.reference, linestyle="--", linewidth=1)
            ax.axhline(bic, color=style.reference, linestyle=":", linewidth=1)
            ax.text(ax.get_xlim()[0], 1.0, " GT = 1.0", va="bottom", color=style.ink, fontsize=8)
            ax.text(ax.get_xlim()[0], bic, f" bicubic = {bic:.3f}", va="bottom", color=style.ink, fontsize=8)
            ax.set_yscale("log")
        ax.set_title(title, color=style.ink, fontsize=10, loc="left")
        ax.set_xlabel("iteration", color=style.ink_muted, fontsize=9)
        ax.tick_params(colors=style.ink_muted, labelsize=8)
        ax.grid(True, color=style.grid, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(style.grid)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=style.ink)
    fig.suptitle(str(style.caption), color=style.ink_muted, fontsize=8, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(style.dpi), facecolor=style.surface)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)
    logger = get_logger("day3_ab_report", log_file=cfg.paths.log_file)
    seed = seed_everything(cfg.seed)
    torch.set_num_threads(int(cfg.runtime.num_threads))
    ev, block = cfg["eval_all_ckpts"], cfg["day3_ab_report"]
    entries = {str(e["label"]): e for e in ev["runs"]}
    cache_root = _path(ev["cache_dir"])
    control, primary = str(block.control), str(block.primary)
    labels = [str(x) for x in block.runs]
    cutoff = float(block.hf_cutoff)

    # -- checkpoints, cached quality tables, the selection rule ----------------
    info: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        run_dir = _path(entries[label]["run_dir"])
        cks = discover_checkpoints(run_dir, str(ev["checkpoint_glob"]))
        frames = load_frames(cache_root, label, cks)
        sel = select_checkpoint(
            frames, metric=str(ev.selection.metric), tie_metric=str(ev.selection.tie_metric),
            tie_better=str(ev.selection.tie_better), n_boot=int(ev.selection.n_boot),
            ci=float(ev.selection.ci), seed=int(cfg.seed))
        final = max(cks, key=lambda c: c["iteration"])["label"]
        by_label = {c["label"]: c for c in cks}
        for ck in cks:
            logged = ck["args"].get("hf_cutoff")
            if logged is not None and float(logged) != cutoff:
                raise ValueError(f"{label}/{ck['label']} logged hf_cutoff {logged}, "
                                 f"cfg.day3_ab_report.hf_cutoff is {cutoff}.")
        info[label] = {"run_dir": run_dir, "frames": frames, "selected": sel["selected"],
                       "final": final, "ckpts": by_label}
        logger.info("%s: selected %s, final %s.", label, sel["selected"], final)

    # -- models: each run's selected and final checkpoint ----------------------
    models: Dict[str, torch.nn.Module] = {}
    for label in labels:
        for which in dict.fromkeys((info[label]["selected"], info[label]["final"])):
            ck = info[label]["ckpts"][which]
            model, model_info = load_checkpoint_model(cfg, ck["path"], logger)
            models[f"{label}_{which}"] = model
            info[label].setdefault("params", model_info["parameters"])

    dataset = get_dataset(cfg)
    loaders = build_dataloaders(cfg, dataset=dataset, logger=logger)
    n_val = len(loaders["datasets"]["val"])
    bicubic = get_baseline("bicubic", int(cfg.sr.scale))

    if args.limit is not None:
        start = time.perf_counter()
        part = sharpness_pass(loaders["val"], models, bicubic, cutoff, args.limit, logger)
        per_patch = (time.perf_counter() - start) / len(part)
        projected = per_patch * n_val / 60.0
        logger.info("TIMING: %d patches, %.3f s/patch over %d model(s); projected full "
                    "split (%d patches) %.1f min; limit %.0f min.", len(part), per_patch,
                    len(models), n_val, projected, float(block.max_local_minutes))
        return 0 if projected <= float(block.max_local_minutes) else 2

    key = {"kind": "day3_sharpness", "cutoff": cutoff,
           "fingerprint": settings_fingerprint(cfg, loaders["split_source"]),
           "weights": {name: info[name.rsplit("_", 1)[0]]["ckpts"][name.rsplit("_", 1)[1]]["weights_sha256"]
                       for name in models}}
    sharp, _ = cached(_path(block.cache_dir), key,
                      lambda: (sharpness_pass(loaders["val"], models, bicubic, cutoff, None, logger),
                               {"n_val": n_val}), args.force, logger)
    if len(sharp) != n_val:
        raise RuntimeError(f"sharpness pass has {len(sharp)} rows for {n_val} val patches.")

    # -- table -----------------------------------------------------------------
    gt_sobel, gt_hf = sharp["sobel_gt"].mean(), sharp["hf_gt"].mean()
    bic_hf = sharp["hf_bicubic"].mean()
    allowed_missing = {str(c) for c in block.allow_missing_columns}
    bic_frame = pd.read_csv(cache_root / "bicubic" / "bicubic" / "per_pair.csv", dtype={"sample_id": str})

    def row(method: str, kind: str, frame: pd.DataFrame, col: str, iteration: Any,
            lambdas: Any) -> Dict[str, Any]:
        merged = frame.merge(sharp[list(PAIR_KEY) + [f"sobel_{col}", f"hf_{col}"]],
                             on=list(PAIR_KEY), how="inner", validate="one_to_one")
        if len(merged) != n_val:
            raise RuntimeError(f"{method}: {len(merged)} of {n_val} pairs matched the sharpness pass.")
        out = {"method": method, "row": kind, "iteration": iteration,
               "lambda1": lambdas[0], "lambda2": lambdas[1], "n_pairs": len(merged)}
        for name, c in TABLE_COLUMNS.items():
            if c in merged:
                out[name] = float(merged[c].mean())
            elif c in allowed_missing:
                logger.warning("%s: column %r absent (allowed by allow_missing_columns); NaN.", method, c)
                out[name] = float("nan")
            else:
                raise KeyError(f"{method}: eval_all_ckpts table has no {c!r} column.")
        out["sobel_ratio_vs_gt"] = float(merged[f"sobel_{col}"].mean() / gt_sobel)
        out["hf_ratio_vs_gt"] = float(merged[f"hf_{col}"].mean() / gt_hf)
        out["hf_abs"] = float(merged[f"hf_{col}"].mean())
        return out

    rows = [row("bicubic", "baseline", bic_frame, "bicubic", "", ("", ""))]
    metadata, flags = {}, []
    for label in labels:
        ck0 = next(iter(info[label]["ckpts"].values()))
        lam = ck0["lambdas"] or {}
        lambdas = (lam.get("spectral_lambda1", ""), lam.get("spectral_lambda2", ""))
        for kind in ("selected", "final"):
            which = info[label][kind]
            it = info[label]["ckpts"][which]["iteration"]
            rows.append(row(label, kind, info[label]["frames"][which], f"{label}_{which}", it, lambdas))
        expected_lam = block.expected.lambdas[label]
        checked = check_metadata(info[label]["run_dir"], label, block.expected, expected_lam,
                                 int(info[label]["params"]))
        metadata[label] = checked["values"]
        flags.extend(checked["flags"])
    shas = {m.get("git_sha") for m in metadata.values()}
    hashes = {m.get("config_hash") for m in metadata.values()}
    unanimous = len(shas) == 1 and len(hashes) == 1
    if not unanimous:
        flags.append(f"runs disagree on provenance: git {sorted(map(str, shas))}, config {sorted(map(str, hashes))}")
    for flag in flags:
        logger.warning("METADATA FLAG: %s", flag)

    table = pd.DataFrame(rows)
    table["blur_index"] = np.nan
    verdicts = {}
    for kind in ("selected", "final"):
        a2 = table[(table.method == control) & (table.row == kind)].iloc[0]
        for label in labels:
            if label == control:
                continue
            idx = table[(table.method == label) & (table.row == kind)].index[0]
            b = table.loc[idx]
            blur = (b.hf_abs - bic_hf) / (a2.hf_abs - bic_hf)
            table.loc[idx, "blur_index"] = blur
            lpips_rel = (b.lpips - a2.lpips) / a2.lpips
            cons_rel = (b.consistency_opensr_l1 - a2.consistency_opensr_l1) / a2.consistency_opensr_l1
            failed = []
            if not b.consistency_opensr_l1 < a2.consistency_opensr_l1:
                failed.append("spectral consistency did not improve")
            if lpips_rel > float(block.lpips_rel_tolerance):
                failed.append(f"LPIPS worse by {lpips_rel:+.1%} (> {float(block.lpips_rel_tolerance):.0%})")
            if blur < float(block.blur_index_min):
                failed.append(f"BLUR HAZARD (blur index {blur:.3f} < {float(block.blur_index_min)})")
            verdicts[f"{label}_{kind}"] = {
                "blur_index": float(blur), "blur_hazard": bool(blur < float(block.blur_index_min)),
                "lpips_rel_change": float(lpips_rel), "consistency_rel_change": float(cons_rel),
                "failed": failed, "spectral_loss_helped": not failed}
    table = table.drop(columns=["hf_abs"])

    over = overfitting(info[control]["run_dir"] / "log.csv", int(block.overfit_window_iters))

    curves, refs, missing = {}, {}, []
    for label in (control, primary):
        curves[label] = pd.read_csv(info[label]["run_dir"] / "log.csv").dropna(subset=["val_psnr"])
        ref_path = info[label]["run_dir"] / "sharpness_reference.json"
        if not ref_path.is_file() or "val_hf_energy" not in curves[label]:
            missing.append(label)
            continue
        refs[label] = json.loads(ref_path.read_text(encoding="utf-8"))
    if missing and bool(block.require_curve_inputs):
        raise FileNotFoundError(f"{missing}: no sharpness_reference.json or val_hf_energy column.")
    if missing:
        logger.warning("FIGURE SKIPPED: %s lack curve inputs (allowed by "
                       "require_curve_inputs=false).", missing)
    else:
        displays = {str(e["label"]): str(e["display"]) for e in ev["runs"]}
        plot_curves(curves, refs, displays, block.figure_style, _path(block.figure_path))

    csv_path, json_path = _path(block.csv_path), _path(block.summary_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, float_format="%.6g")
    summary = {"seed": seed, "n_val": n_val, "hf_cutoff": cutoff, "gt_sobel": float(gt_sobel),
               "gt_hf": float(gt_hf), "bicubic_hf": float(bic_hf), "metadata": metadata,
               "metadata_flags": flags, "provenance_unanimous": unanimous,
               "verdicts": verdicts, "overfitting": over,
               "selection": {k: {"selected": v["selected"], "final": v["final"]} for k, v in info.items()}}
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s, %s, %s", csv_path, json_path, _path(block.figure_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
