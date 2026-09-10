"""Day 3, task 6: every checkpoint, the full val split, post-hoc selection, B - A2.

WHAT THIS PRODUCES. The headline table: bicubic / Run A / Run A2 / Run B, each
learned row represented by ONE checkpoint chosen by a rule fixed in
``cfg.eval_all_ckpts.selection`` before any full-split number existed -- not by
``best.pt``, which the training loop chose on PSNR, the metric the spectral loss
is expected to cost. Then the question a judge will ask: is B - A2 noise?

Per checkpoint, over every validation patch:

- ``src.metrics.aggregate.Evaluator`` -- PSNR, SSIM, LPIPS, SAM, ERGAS against
  the HR, the same object and settings as every earlier table;
- ``src.eval.opensr_harness`` (the task-3 adapter) -- opensr-test's consistency
  (``reflectance``, ``spectral``, ``spatial``), ``synthesis`` and correctness;
- ``src.losses.spectral_terms`` -- the D=area ``l1_spec`` / SAM the spectral
  loss trains on, measured in the same pass, so a row can be read against the
  spectral floor in the floor's own units.

An ``hr_oracle`` pass substitutes the ground-truth HR for the SR output and
gives the spectral floor in both consistency conventions.

CACHING. Each checkpoint's per-pair table is written to
``cfg.eval_all_ckpts.cache_dir/<run>/<iteration>/`` beside a key holding the
hash of the WEIGHTS and a fingerprint of every metric setting and of the split
file. A matching key is a cache hit; a stale one is recomputed and says so. The
key is written last, so a crash mid-write never leaves a false hit.

Independent runs can be scored in parallel processes -- ``--only a2 b1`` in one
terminal, ``--only runA b2 bicubic hr_oracle`` in another -- and a final call
with no ``--only`` reads everything from cache and writes the report.

Examples:
    .venv/Scripts/python.exe scripts/eval_all_ckpts.py --smoke
    .venv/Scripts/python.exe scripts/eval_all_ckpts.py --only a2 b1
    .venv/Scripts/python.exe scripts/eval_all_ckpts.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.data.loader import build_dataloaders  # noqa: E402
from src.data.registry import get_dataset  # noqa: E402
from src.eval.baselines import get_baseline  # noqa: E402
from src.eval.ckpt_selection import PAIR_KEY, compare_paired, select_checkpoint  # noqa: E402
from src.eval.consistency_plot import plot_consistency_vs_iter, read_validation_curve  # noqa: E402
from src.eval.opensr_harness import (  # noqa: E402
    OPENSR_METRICS,
    OpenSRSampleError,
    build_metrics,
    run_opensr_test,
    score_arrays,
)
from src.losses import spectral_settings_from_cfg, spectral_terms  # noqa: E402
from src.metrics.aggregate import Evaluator  # noqa: E402
from src.metrics.image_quality import LPIPS_CAVEAT  # noqa: E402
from src.models.edsr import build_model  # noqa: E402
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root, resolve_output_path  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

# Bumped whenever the per-pair table's columns change, so an old cache misses.
CACHE_SCHEMA = 1

# opensr-test's three consistency metrics: LR vs SR downsampled back to LR.
OPENSR_CONSISTENCY = ("reflectance", "spectral", "spatial")

# Headline table: (column, header, better, format). Display text only -- every
# number comes from the per-pair tables.
TABLE_COLUMNS = (
    ("psnr_mean", "PSNR dB", "higher", "{:.3f}"),
    ("ssim_mean", "SSIM", "higher", "{:.4f}"),
    ("lpips", "LPIPS", "lower", "{:.4f}"),
    ("sam_mean_deg", "SAM° vs HR", "lower", "{:.3f}"),
    ("ergas", "ERGAS", "lower", "{:.3f}"),
    ("reflectance", "opensr refl. L1", "lower", "{:.5f}"),
    ("spectral", "opensr spectral°", "lower", "{:.4f}"),
    ("spatial", "opensr spatial px", "lower", "{:.4f}"),
    ("l1_spec", "L1_spec (D=area)", "lower", "{:.6f}"),
    ("sam_spec_deg", "SAM_spec° (D=area)", "lower", "{:.4f}"),
)


# -- small utilities --------------------------------------------------------


def _path(raw: Any) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root() / path


def _plain(node: Any) -> Any:
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(state: Dict[str, torch.Tensor]) -> str:
    """Hash of the weights alone -- optimiser state and metadata excluded."""
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _item(value: Any) -> Any:
    return value.item() if torch.is_tensor(value) else value


def settings_fingerprint(cfg: Any, split_source: Any) -> str:
    """Hash of everything a cached per-pair number depends on, weights aside.

    Args:
        cfg: Loaded config.
        split_source: ``build_dataloaders(...)["split_source"]``, e.g.
            ``"file:<path>"``. The file's CONTENT is hashed when it exists, so a
            regenerated split misses the cache even at the same path.

    Returns:
        A hex SHA256.
    """
    source = str(split_source)
    split_hash = source
    if source.startswith("file:") and Path(source[5:]).is_file():
        split_hash = _sha256_file(Path(source[5:]))
    opensr = {
        key: value
        for key, value in _plain(cfg["opensr_test"]).items()
        if key not in ("n_samples", "json_name", "csv_name")
    }
    parts = {
        "schema": CACHE_SCHEMA,
        "metrics": _plain(cfg["metrics"]),
        "opensr_test": opensr,
        "spectral": spectral_settings_from_cfg(cfg),
        "dataset": {
            "name": str(cfg["dataset"]["name"]),
            "bands": [str(b) for b in cfg["dataset"]["bands"]],
            "reflectance_scale": float(cfg["dataset"]["reflectance_scale"]),
        },
        "scale": int(cfg["sr"]["scale"]),
        "split": split_hash,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


# -- checkpoints ------------------------------------------------------------


def discover_checkpoints(run_dir: Path, pattern: str) -> List[Dict[str, Any]]:
    """Every distinct checkpoint in a run directory, in iteration order.

    Two files at the same iteration (``best.pt`` taken at the final
    validation, say) are one checkpoint if their weights are identical, and an
    error if not.

    Args:
        run_dir: Directory of ``src/train.py`` checkpoints.
        pattern: Glob, ``cfg.eval_all_ckpts.checkpoint_glob``.

    Returns:
        One dict per checkpoint: ``label`` (``it<iteration>``), ``iteration``
        (1-based, as the log counts), ``path``, ``files``, ``weights_sha256``,
        ``lambdas``, ``git_sha``, ``config_hash`` and the training ``args``.

    Raises:
        FileNotFoundError: No checkpoint matches.
        KeyError: A file lacks ``it``, ``model`` or ``args``.
        ValueError: Two files claim one iteration with different weights, or the
            checkpoints of one run disagree on lambdas, commit or config hash --
            a directory mixing runs would make the selection compare runs, not
            checkpoints.
    """
    files = sorted(run_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No checkpoint matching {pattern!r} in {run_dir}.")

    by_iter: Dict[int, Dict[str, Any]] = {}
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        missing = [key for key in ("it", "model", "args") if key not in payload]
        if missing:
            raise KeyError(f"{path} lacks {missing}; it was not written by src/train.py.")
        iteration = int(payload["it"]) + 1
        weights = _state_dict_sha256(payload["model"])
        if iteration in by_iter:
            prior = by_iter[iteration]
            if prior["weights_sha256"] != weights:
                raise ValueError(
                    f"{prior['files'][0]} and {path.name} both claim iteration "
                    f"{iteration} with different weights."
                )
            prior["files"].append(path.name)
            continue
        by_iter[iteration] = {
            "label": f"it{iteration:05d}",
            "iteration": iteration,
            "path": path,
            "files": [path.name],
            "weights_sha256": weights,
            "lambdas": payload.get("lambdas"),
            "git_sha": payload.get("git_sha"),
            "config_hash": payload.get("config_hash"),
            "args": dict(payload["args"]),
        }

    found = [by_iter[key] for key in sorted(by_iter)]
    for field in ("lambdas", "git_sha", "config_hash"):
        values = {json.dumps(c[field], sort_keys=True, default=str) for c in found}
        if len(values) > 1:
            raise ValueError(
                f"checkpoints in {run_dir} disagree on {field}: {sorted(values)}. "
                "One directory must hold one run."
            )
    return found


# -- scoring ----------------------------------------------------------------


def cached(
    cache_dir: Path,
    key: Dict[str, Any],
    compute: Callable[[], Tuple[pd.DataFrame, Dict[str, Any]]],
    force: bool,
    logger: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return the cached per-pair table for ``key``, computing it on a miss.

    Args:
        cache_dir: This checkpoint's cache directory.
        key: Everything the table depends on. Compared for equality.
        compute: Produces ``(per-pair frame, metadata)``.
        force: Recompute even on a hit.
        logger: Logger.

    Returns:
        ``(frame, meta)``.
    """
    key_path = cache_dir / "key.json"
    csv_path = cache_dir / "per_pair.csv"
    meta_path = cache_dir / "meta.json"
    if not force and key_path.is_file() and csv_path.is_file() and meta_path.is_file():
        stored = json.loads(key_path.read_text(encoding="utf-8"))
        if stored == key:
            logger.info("cache HIT %s", cache_dir)
            frame = pd.read_csv(csv_path, dtype={"sample_id": str})
            return frame, json.loads(meta_path.read_text(encoding="utf-8"))
        stale = sorted(k for k in set(stored) | set(key) if stored.get(k) != key.get(k))
        logger.warning("cache at %s is STALE (differs in %s); recomputing.", cache_dir, stale)

    frame, meta = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    # Last, so an interrupted write can never be mistaken for a complete one.
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    return frame, meta


def score_method(
    label: str,
    sr_fn: Callable[[torch.Tensor], Any],
    ctx: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Score one super-resolver over the whole validation split, per pair.

    Args:
        label: Method label for logs and the result.
        sr_fn: ``(B, C, h, w)`` float32 surface reflectance, nominally [0, 1]
            and unclipped, in ``cfg.dataset.bands`` order -> ``(B, C, 4h, 4w)``
            in the same units. Never normalised or clipped here.
        ctx: Shared state from :func:`main` -- cfg, loader, evaluator, opensr
            metrics/settings, spectral settings, split name, logger.

    Returns:
        ``(frame, meta)``: one row per validation patch with the Evaluator
        columns, ``l1_spec`` (reflectance), ``sam_spec_deg``,
        ``sam_spec_valid_frac`` and every opensr-test metric (NaN where
        opensr-test skipped the patch -- the skips are listed in ``meta``).

    Raises:
        RuntimeError: The in-pass spectral terms and the Evaluator rows do not
            line up, which would mean the loader was not iterated in one order.
    """
    stash: List[Dict[str, np.ndarray]] = []

    def instrumented(lr: torch.Tensor) -> Any:
        sr = sr_fn(lr)
        sr_t = torch.from_numpy(np.ascontiguousarray(sr)) if isinstance(sr, np.ndarray) else sr
        parts = spectral_terms(
            sr_t.detach().float().cpu(), lr.detach().float().cpu(),
            per_sample=True, **ctx["spectral"],
        )
        stash.append({k: v.detach().cpu().numpy().astype(np.float64) for k, v in parts.items()})
        return sr

    quality = ctx["evaluator"].run(ctx["loader"], instrumented, name=label, split=ctx["split"])
    frame = quality.per_sample.copy()
    l1 = np.concatenate([s["l1_spec"] for s in stash])
    if l1.size != len(frame):
        raise RuntimeError(
            f"{label}: {l1.size} spectral rows for {len(frame)} evaluated patches."
        )
    frame["l1_spec"] = l1
    frame["sam_spec_deg"] = np.degrees(np.concatenate([s["sam"] for s in stash]))
    frame["sam_spec_valid_frac"] = np.concatenate([s["sam_valid_frac"] for s in stash])

    opensr = run_opensr_test(
        dataloader=ctx["loader"], sr_fn=sr_fn, n_samples=None, cfg=ctx["cfg"],
        method=label, logger=ctx["logger"], metrics=ctx["metrics"],
        settings=ctx["settings"],
    )
    scored = opensr.to_frame()[list(PAIR_KEY) + list(OPENSR_METRICS)]
    frame["sample_id"] = frame["sample_id"].astype(str)
    scored["sample_id"] = scored["sample_id"].astype(str)
    merged = frame.merge(scored, on=list(PAIR_KEY), how="left", validate="one_to_one")

    meta = {
        "label": label,
        "num_pairs": int(len(merged)),
        "evaluator_settings": quality.settings,
        "opensr_settings": opensr.settings,
        "opensr_skipped": opensr.skipped,
        "spectral_settings": ctx["spectral"],
    }
    return merged, meta


def score_oracle(ctx: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The spectral floor: ground-truth HR scored as though it were the SR.

    Returns:
        ``(frame, meta)``: one row per patch with ``l1_spec``, ``sam_spec_deg``
        and opensr-test's three consistency metrics for ``sr = hr``. A patch
        opensr-test cannot score is NaN in those three and listed in ``meta``.

    Raises:
        RuntimeError: More than ``cfg.opensr_test.max_skipped_fraction`` of the
            patches could not be scored.
    """
    cfg, logger = ctx["cfg"], ctx["logger"]
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in ctx["loader"]:
            lr = batch["lr"].float()
            hr = batch["hr"].float()
            parts = spectral_terms(hr, lr, per_sample=True, **ctx["spectral"])
            for i in range(int(lr.shape[0])):
                row: Dict[str, Any] = {k: _item(batch[k][i]) for k in PAIR_KEY}
                row["sample_id"] = str(row["sample_id"])
                row["l1_spec"] = float(parts["l1_spec"][i])
                row["sam_spec_deg"] = math.degrees(float(parts["sam"][i]))
                try:
                    values = score_arrays(
                        lr[i], hr[i], hr[i], cfg,
                        metrics=ctx["metrics"], settings=ctx["settings"],
                    )
                    row.update({m: values[m] for m in OPENSR_CONSISTENCY})
                except OpenSRSampleError as exc:
                    skipped.append({**{k: row[k] for k in PAIR_KEY}, "reason": str(exc)})
                    logger.warning("hr_oracle: opensr-test SKIP %s: %s", row["sample_id"], exc)
                    row.update({m: float("nan") for m in OPENSR_CONSISTENCY})
                rows.append(row)
            if len(rows) % 200 < int(lr.shape[0]):
                logger.info("hr_oracle: %d patches scored.", len(rows))

    limit = float(cfg["opensr_test"]["max_skipped_fraction"])
    if rows and len(skipped) / len(rows) > limit:
        raise RuntimeError(
            f"hr_oracle: opensr-test skipped {len(skipped)} of {len(rows)} patches, "
            f"above cfg.opensr_test.max_skipped_fraction={limit}."
        )
    return pd.DataFrame(rows), {"label": "hr_oracle", "num_pairs": len(rows),
                                "opensr_skipped": skipped,
                                "spectral_settings": ctx["spectral"]}


# -- smoke ------------------------------------------------------------------


def fabricate_runs(cfg: Any, logger: Any) -> None:
    """Write tiny random-weight run directories for ``--smoke``.

    Each run gets two checkpoints (``best.pt`` at iteration 2, ``last.pt`` at 4)
    in the real ``src/train.py`` format, plus a ``log.csv`` with validation
    rows, so discovery, loading, caching, selection, the tests and the figure
    all run on genuinely written files.

    Raises:
        RuntimeError: ``cfg.eval_all_ckpts.fabricate_runs`` is not set -- this
            writes noise and must never touch a real run directory.
    """
    block = cfg["eval_all_ckpts"]
    if not bool(block["fabricate_runs"]):
        raise RuntimeError("fabricate_runs() called without cfg.eval_all_ckpts.fabricate_runs.")
    scale = int(cfg["sr"]["scale"])
    channels = len(cfg["dataset"]["bands"])
    logger.warning("SMOKE: fabricating RANDOM-weight run directories. Nothing here is a result.")
    for n, entry in enumerate(e for e in block["runs"] if str(e["kind"]) == "run"):
        run_dir = _path(entry["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        args = {"scale": scale, "n_resblocks": 2, "n_feats": 16, "in_ch": channels,
                "iters": 4, "ckpt_every": 2, "wd": 0.0,
                "fabricated_by": "scripts/eval_all_ckpts.py --smoke"}
        for k, (name, iteration) in enumerate((("best.pt", 2), ("last.pt", 4))):
            torch.manual_seed(int(cfg["seed"]) + 10 * n + k)
            model = build_model(str(cfg["eval_runA"]["model_name"]), scale=scale,
                                n_resblocks=2, n_feats=16, in_ch=channels, out_ch=channels)
            torch.save({"model": model.state_dict(), "it": iteration - 1, "best": 0.0,
                        "args": args, "git_sha": "smoke", "config_hash": "smoke",
                        "lambdas": {"spectral_lambda1": 0.1 * n, "spectral_lambda2": 0.02 * n}},
                       run_dir / name)
        lines = ["iter,loss,lr,val_psnr,sec_per_100it,val_l1_spec,val_sam,val_lpips"]
        for step in (1, 2, 3, 4):
            lines.append(f"{step},,,{20 + step},,{0.01 / step:.6f},0.02,0.5")
        (run_dir / "log.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- reporting --------------------------------------------------------------


def _mean(frame: pd.DataFrame, column: str) -> Tuple[float, int]:
    if column not in frame.columns:
        return float("nan"), 0
    series = pd.to_numeric(frame[column], errors="coerce")
    finite = series[np.isfinite(series)]
    return (float(finite.mean()) if len(finite) else float("nan")), int(len(finite))


def _console_safe() -> None:
    """Let the report's Δ, ÷ and ° survive a cp1252 Windows console.

    ``errors="backslashreplace"`` rather than forcing UTF-8: a crash becomes a
    visible ``\\u0394`` instead of mojibake. The report FILES are always UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")


def _fmt(value: float, spec: str) -> str:
    return "n/a" if not math.isfinite(value) else spec.format(value)


def _fmt_ci(interval: List[float]) -> str:
    return f"[{interval[0]:+.3g}, {interval[1]:+.3g}]"


def _p(value: float) -> str:
    return "<1e-300" if value == 0.0 else f"{value:.2g}"


def table_markdown(rows: List[Dict[str, Any]], floor: Dict[str, float],
                   include_floor: bool = True) -> str:
    """The headline table: one row per method, plus the floor as a reference row."""
    headers = ["method", "λ1 / λ2"] + [f"{h} {'↑' if b == 'higher' else '↓'}"
                                        for _, h, b, _ in TABLE_COLUMNS]
    headers += ["L1_spec ÷ floor", "params", "iters (selected / trained)"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        cells = [row["display"], row["lambdas"]]
        cells += [_fmt(row["means"][col], spec) for col, _, _, spec in TABLE_COLUMNS]
        # A zero floor (the smoke stub's LR is an exact area-mean of its HR)
        # makes the ratio meaningless, not large.
        ratio = (f"{row['means']['l1_spec'] / floor['l1_spec']:.2f}×"
                 if floor["l1_spec"] > 0 else "n/a")
        cells += [ratio, row["params"], row["iters"]]
        lines.append("| " + " | ".join(cells) + " |")
    if not include_floor:
        return "\n".join(lines)
    cells = ["*GT HR as the SR — spectral floor*", "—"]
    for col, _, _, spec in TABLE_COLUMNS:
        cells.append(f"*{_fmt(floor[col], spec)}*" if col in floor else "—")
    cells += ["*1.00×*", "—", "—"]
    lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def significance_markdown(results: List[Dict[str, Any]], ci: float) -> str:
    lines = [
        f"| metric | better | A2 mean | B mean | mean Δ (B − A2) | {ci:.0%} CI, pairs "
        f"| {ci:.0%} CI, tiles | Wilcoxon p (pairs) | p (tile means) "
        "| B better on | n | verdict (tile CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        verdict = r["verdict"]
        verdict = f"**{verdict}**" if verdict != "not resolved" else "not resolved — CI straddles 0"
        lines.append(
            f"| `{r['metric']}` | {r['better']} | {r['mean_control']:.5g} | "
            f"{r['mean_treatment']:.5g} | {r['mean_delta']:+.3g} | {_fmt_ci(r['ci_pair'])} | "
            f"{_fmt_ci(r['ci_tile'])} | {_p(r['p_wilcoxon_pair'])} | "
            f"{_p(r['p_wilcoxon_tile'])} | {r['frac_treatment_better']:.1%} | "
            f"{r['n']} | {verdict} |"
        )
    return "\n".join(lines)


def selection_markdown(label: str, display: str, sel: Dict[str, Any],
                       ckpts: Dict[str, Dict[str, Any]]) -> str:
    rule = sel["rule"]
    metric, tie = rule["metric"], rule["tie_metric"]
    lines = [
        f"**{display}** (`{label}`) — selected **{sel['selected']}** "
        f"(files: {', '.join(ckpts[sel['selected']]['files'])}).",
        "",
        f"| checkpoint | files | mean `{metric}` | Δ vs min | CI vs min | tied | "
        f"mean `{tie}` | selected |",
        "|---|---|---:|---:|---:|:--:|---:|:--:|",
    ]
    for r in sel["rows"]:
        lines.append(
            f"| {r['label']} | {', '.join(ckpts[r['label']]['files'])} | "
            f"{r[f'mean_{metric}']:.6f} | {r['delta_vs_min']:+.3g} | "
            f"{'—' if r['is_min'] else _fmt_ci(r['ci_vs_min'])} | "
            f"{'yes' if r['tied_with_min'] else 'no'} | {r[f'mean_{tie}']:.4f} | "
            f"{'**◉**' if r['selected'] else ''} |"
        )
    return "\n".join(lines)


def log_curve_markdown(curves: Dict[str, pd.DataFrame], displays: Dict[str, str]) -> str:
    """The training loop's own validation curves, side by side."""
    labels = list(curves)
    cols = ("val_l1_spec", "val_sam", "val_psnr", "val_lpips")
    header = "| iter | " + " | ".join(f"{displays[l]} {c}" for l in labels for c in cols) + " |"
    lines = [header, "|---:|" + "---:|" * (len(cols) * len(labels))]
    frames = {l: curves[l].set_index("iter") for l in labels}
    for it in sorted(set().union(*[set(f.index) for f in frames.values()])):
        cells = []
        for l in labels:
            for c in cols:
                value = frames[l][c].get(it, float("nan")) if c in frames[l] else float("nan")
                cells.append("" if not math.isfinite(value) else f"{value:.5g}")
        lines.append(f"| {int(it)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def reading(sig: Dict[str, Dict[str, Any]], metric: str) -> List[str]:
    """The honest reading, generated from the numbers so it cannot be spun.

    Returns:
        Two sentences: consistency, then what B costs or gains on PSNR.
    """
    c = sig[metric]
    rel = c["mean_delta"] / c["mean_control"] if c["mean_control"] else float("nan")
    if c["verdict"] == "treatment better":
        clause = "the CI excludes zero, so this consistency gain is resolved, not noise"
    elif c["verdict"] == "control better":
        clause = ("the CI excludes zero in the WRONG direction: **B does not beat A2 on "
                  "consistency; it is worse**")
    else:
        clause = ("the CI straddles zero: **B does not beat A2 on consistency, and this is "
                  "not a headline**")
    first = (
        f"Run B {'lowers' if c['mean_delta'] < 0 else 'raises'} opensr-test's consistency "
        f"error (`{metric}`) from {c['mean_control']:.5f} to {c['mean_treatment']:.5f} "
        f"({rel:+.1%}); the tile-clustered {c['ci']:.0%} CI on the per-pair mean delta is "
        f"{_fmt_ci(c['ci_tile'])}, and {clause}."
    )
    p = sig.get("psnr_mean")
    if p is None:
        return [first]
    resolved = p["verdict"] != "not resolved"
    cost = p["mean_delta"] < 0
    second = (
        f"B {'costs' if cost else 'gains'} {abs(p['mean_delta']):.3f} dB PSNR against A2 "
        f"(tile CI {_fmt_ci(p['ci_tile'])} dB, "
        f"{'resolved' if resolved else 'NOT resolved — indistinguishable from zero'})"
    )
    lp = sig.get("lpips")
    if lp is not None:
        second += (
            f", and moves LPIPS by {lp['mean_delta']:+.4f} "
            f"({'resolved' if lp['verdict'] != 'not resolved' else 'not resolved'})"
        )
    return [first, second + "."]


# -- entry point ------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Score only these labels (run labels, 'bicubic', or the "
                             "oracle label) and exit without a report. For parallel "
                             "processes; a final call without --only reports from cache.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute every table, ignoring the cache.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    _console_safe()
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)
    logger = get_logger("eval_all_ckpts", log_file=cfg.paths.log_file)
    seed = seed_everything(cfg.seed)
    torch.set_num_threads(int(cfg.runtime.num_threads))
    logger.info("Seeded with %d; %d torch threads; CUDA available: %s.",
                seed, torch.get_num_threads(), torch.cuda.is_available())

    block = cfg["eval_all_ckpts"]
    entries = {str(e["label"]): e for e in block["runs"]}
    oracle_label = str(block["oracle_label"])
    known = set(entries) | {oracle_label}
    wanted = set(args.only) if args.only is not None else known
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"--only names unknown label(s) {unknown}; known: {sorted(known)}.")

    if args.smoke and bool(block["fabricate_runs"]):
        fabricate_runs(cfg, logger)

    # Checkpoints are discovered BEFORE any scoring: a missing run directory
    # found after an hour of opensr-test is an hour lost.
    checkpoints: Dict[str, List[Dict[str, Any]]] = {}
    for label, entry in entries.items():
        if str(entry["kind"]) == "run":
            checkpoints[label] = discover_checkpoints(
                _path(entry["run_dir"]), str(block["checkpoint_glob"])
            )
            logger.info("%s: %d checkpoint(s): %s", label, len(checkpoints[label]),
                        ", ".join(f"{c['label']} ({'/'.join(c['files'])})"
                                  for c in checkpoints[label]))

    control, treatment = str(block["control"]), str(block["treatment"])
    for field in ("git_sha", "config_hash"):
        a, b = checkpoints[control][0][field], checkpoints[treatment][0][field]
        if a != b:
            raise ValueError(
                f"control {control!r} and treatment {treatment!r} differ in {field} "
                f"({a} vs {b}); they are not the same experiment."
            )

    dataset = get_dataset(cfg)
    loaders = build_dataloaders(cfg, dataset=dataset, logger=logger)
    split = str(cfg.loader.val_split)
    logger.info("Validation split %r: %d patches.", split, len(loaders["datasets"]["val"]))
    metrics, settings = build_metrics(cfg, logger=logger)
    ctx = {
        "cfg": cfg, "logger": logger, "loader": loaders["val"], "split": split,
        "evaluator": Evaluator(cfg, logger=logger, device=torch.device("cpu")),
        "metrics": metrics, "settings": settings,
        "spectral": spectral_settings_from_cfg(cfg),
    }
    fingerprint = settings_fingerprint(cfg, loaders["split_source"])
    cache_root = _path(block["cache_dir"])

    # -- score (or read from cache) -------------------------------------------
    frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    infos: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if oracle_label in wanted:
        oracle, oracle_meta = cached(
            cache_root / oracle_label, {"kind": "oracle", "fingerprint": fingerprint},
            lambda: score_oracle(ctx), args.force, logger,
        )
    for label, entry in entries.items():
        if label not in wanted:
            continue
        frames[label], infos[label] = {}, {}
        if str(entry["kind"]) == "baseline":
            method = str(entry["method"])
            sr_fn = get_baseline(method, int(cfg.sr.scale))
            frames[label][method], meta = cached(
                cache_root / label / method,
                {"kind": "baseline", "method": method, "fingerprint": fingerprint},
                lambda: score_method(label, sr_fn, ctx), args.force, logger,
            )
            infos[label][method] = {"label": method, "files": [], "params": 0, "meta": meta}
            continue
        from scripts.eval_runA import load_checkpoint_model, make_model_sr_fn

        for ck in checkpoints[label]:
            model, model_info = load_checkpoint_model(cfg, ck["path"], logger)
            name = f"{label}_{ck['label']}"
            frames[label][ck["label"]], meta = cached(
                cache_root / label / ck["label"],
                {"kind": "checkpoint", "weights_sha256": ck["weights_sha256"],
                 "fingerprint": fingerprint},
                lambda: score_method(name, make_model_sr_fn(model), ctx),
                args.force, logger,
            )
            infos[label][ck["label"]] = {**ck, "params": model_info["parameters"],
                                         "parameter_budget": model_info["parameter_budget"],
                                         "meta": meta}

    if args.only is not None:
        logger.info("--only: scored %s. Run again without --only to write the report.",
                    sorted(wanted))
        return 0

    # -- selection ------------------------------------------------------------
    sel_cfg = block["selection"]
    selections: Dict[str, Dict[str, Any]] = {}
    for label, entry in entries.items():
        if str(entry["kind"]) != "run":
            continue
        selections[label] = select_checkpoint(
            frames[label], metric=str(sel_cfg["metric"]),
            tie_metric=str(sel_cfg["tie_metric"]), tie_better=str(sel_cfg["tie_better"]),
            n_boot=int(sel_cfg["n_boot"]), ci=float(sel_cfg["ci"]), seed=int(cfg.seed),
        )
        logger.info("%s: selected %s (minimum %s, tied set %s).", label,
                    selections[label]["selected"], selections[label]["minimum"],
                    selections[label]["tied_set"])

    def chosen(label: str) -> str:
        return selections[label]["selected"] if label in selections else next(iter(frames[label]))

    # -- the floor ------------------------------------------------------------
    floor = {col: _mean(oracle, col)[0] for col in ("l1_spec", "sam_spec_deg", *OPENSR_CONSISTENCY)}
    # The Day 3 floor report's number, beside the one measured in this pass: if
    # they disagree, the split or the operator moved and the table says so.
    floor_json = Path(resolve_output_path(cfg, "metric_dir")) / str(cfg.spectral_floor.json_name)
    stored_floor = (
        json.loads(floor_json.read_text(encoding="utf-8"))["floor"]["l1_spec"]["mean"]
        if floor_json.is_file() else float("nan")
    )
    if not floor_json.is_file():
        logger.warning("No stored spectral floor at %s; cross-check skipped.", floor_json)

    # -- table rows -----------------------------------------------------------
    def row_for(label: str) -> Dict[str, Any]:
        entry, ck = entries[label], chosen(label)
        frame, info = frames[label][ck], infos[label][ck]
        means = {col: _mean(frame, col)[0] for col, *_ in TABLE_COLUMNS}
        if str(entry["kind"]) == "run":
            lam = info["lambdas"] or {}
            l1, l2 = lam.get("spectral_lambda1"), lam.get("spectral_lambda2")
            lambdas = "—" if l1 is None else f"{float(l1):g} / {float(l2):g}"
            iters = f"{info['iteration']:,} / {int(info['args']['iters']):,}"
            params = f"{info['params']:,}"
        else:
            lambdas, iters, params = "—", "—", "0"
        display = str(entry["display"]) + (" ¹" if label == "runA" else "")
        return {"label": label, "display": display, "checkpoint": ck, "means": means,
                "lambdas": lambdas, "iters": iters, "params": params,
                "n_pairs": int(len(frame))}

    main_rows = [row_for(label) for label in block["table_rows"]]
    supp_rows = [row_for(label) for label in block["supplementary_rows"]]

    # -- significance ---------------------------------------------------------
    sig_cfg = block["significance"]
    t_frame = frames[treatment][chosen(treatment)]
    c_frame = frames[control][chosen(control)]
    significance: List[Dict[str, Any]] = []
    for spec in sig_cfg["metrics"]:
        significance.append(compare_paired(
            t_frame, c_frame, metric=str(spec["name"]), better=str(spec["better"]),
            n_boot=int(sig_cfg["n_boot"]), ci=float(sig_cfg["ci"]), seed=int(cfg.seed),
        ))
    sig_by = {r["metric"]: r for r in significance}

    # The same comparison at iterations BOTH runs have, so a selection that
    # picked different iterations cannot hide an iteration effect.
    matched: List[Dict[str, Any]] = []
    for it in sorted(set(frames[control]) & set(frames[treatment])):
        for spec in sig_cfg["metrics"]:
            if str(spec["name"]) not in (str(sel_cfg["metric"]), "psnr_mean"):
                continue
            result = compare_paired(
                frames[treatment][it], frames[control][it], metric=str(spec["name"]),
                better=str(spec["better"]), n_boot=int(sig_cfg["n_boot"]),
                ci=float(sig_cfg["ci"]), seed=int(cfg.seed),
            )
            matched.append({"iteration": it, **result})

    # -- log curves and the figure --------------------------------------------
    fig_cfg = block["figure"]
    displays = {label: str(entries[label]["display"]) for label in entries}
    log_curves = {label: read_validation_curve(_path(entries[label]["run_dir"]) / "log.csv")
                  for label in (control, treatment)}
    series = []
    for label in fig_cfg["runs"]:
        curve = read_validation_curve(_path(entries[label]["run_dir"]) / "log.csv")
        points = pd.DataFrame([
            {"iter": infos[label][ck]["iteration"],
             "value": _mean(frames[label][ck], str(fig_cfg["eval_column"]))[0],
             "selected": ck == chosen(label)}
            for ck in frames[label]
        ])
        series.append({
            "display": displays[label], "color": str(fig_cfg["colors"][label]),
            "curve": pd.DataFrame({"iter": curve["iter"],
                                   "value": curve[str(fig_cfg["log_column"])]}),
            "points": points,
        })
    figure_path = plot_consistency_vs_iter(
        series, floor=floor["l1_spec"], out_path=_path(fig_cfg["path"]), style=fig_cfg,
        title="Spectral-consistency error vs iteration — Run A2 (control) vs Run B",
        ylabel="L1(LR, D_area(SR)), reflectance",
        floor_label=f"spectral floor: GT HR, full val split = {floor['l1_spec']:.5f}",
    )

    # -- runA facts for its footnote (computed, not asserted) -----------------
    ra = infos["runA"][chosen("runA")]
    ra_curve = read_validation_curve(_path(entries["runA"]["run_dir"]) / "log.csv")
    ra_peak = ra_curve.loc[ra_curve["val_psnr"].idxmax()]

    # -- checkpoint availability ---------------------------------------------
    availability = []
    for label in checkpoints:
        found = checkpoints[label]
        args0 = found[0]["args"]
        scheduled = (int(args0["iters"]) // int(args0["ckpt_every"])
                     if args0.get("ckpt_every") else None)
        availability.append({"label": label, "found": len(found),
                             "iterations": [c["iteration"] for c in found],
                             "scheduled": scheduled})

    # -- write ----------------------------------------------------------------
    written = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    ci = float(sig_cfg["ci"])
    metric = str(sel_cfg["metric"])
    sentences = reading(sig_by, metric)
    below_floor = [r["display"] for r in main_rows + supp_rows
                   if r["label"] != "bicubic" and r["means"]["l1_spec"] < floor["l1_spec"]]
    n_pairs = main_rows[0]["n_pairs"]
    report_md = _path(block["report_md"])
    fig_rel = Path(figure_path).resolve().relative_to(report_md.parent.resolve()).as_posix() \
        if report_md.parent.resolve() in Path(figure_path).resolve().parents else str(figure_path)

    md: List[str] = []
    if args.smoke:
        md += ["> **SMOKE NUMBERS on the synthetic stub with RANDOM-weight checkpoints. "
               "Nothing below is a result.**", ""]
    md += [
        "# Day 3 — headline results: post-hoc checkpoint selection, and is B − A2 noise?",
        "",
        f"_Generated {written} by `scripts/eval_all_ckpts.py`._ Dataset "
        f"`{cfg.dataset.name}`, split `{split}`, **{n_pairs} validation pairs**, every "
        "method scored on the identical patches, CPU, float32.",
        "",
        "## The reading",
        "",
        " ".join(sentences),
        "",
    ]
    if below_floor:
        md += [
            f"> **Every learned model is already below the spectral floor** "
            f"(L1_spec {floor['l1_spec']:.5f}, the ground-truth HR's own value): "
            f"{', '.join(below_floor)}. The floor is therefore not a lower bound on "
            "consistency error. An L1-trained network regresses toward the conditional "
            "mean, which is smoother and closer to the Sentinel-2 PSF than the real "
            "NAIP-derived HR. Pushing consistency further down moves the output further "
            "from the HR's own behaviour. `reports/day3_spectral_floor.md` reads "
            "\"below the floor\" as \"the spectral term has overridden reconstruction\", "
            "but the control sits below it with the spectral term at zero. That reading "
            "needs revising.",
            "",
        ]
    md += [
        "## 1. The selection rule (stated before the full-split results)",
        "",
        f"For each run, independently and identically: the checkpoint with the lowest "
        f"mean opensr-test `{metric}` (consistency error: L1 between the LR and the SR "
        f"downsampled back to 10 m) is the minimum. Any checkpoint whose paired "
        f"bootstrap {float(sel_cfg['ci']):.0%} CI on (candidate − minimum) contains 0 "
        f"is **tied** with it. The tied checkpoint with the best "
        f"`{sel_cfg['tie_metric']}` ({sel_cfg['tie_better']} is better) is selected. "
        "`best.pt` was not used as a choice: it is only a filename here, scored like "
        "any other checkpoint. The same rule is applied to Run A.",
        "",
        f"> {str(sel_cfg['provenance']).strip()}",
        "",
        "### Checkpoints available — READ THIS",
        "",
    ]
    for a in availability:
        md.append(
            f"- `{a['label']}`: **{a['found']}** checkpoint(s) on disk, at iterations "
            f"{', '.join(f'{i:,}' for i in a['iterations'])}"
            + (f" — of the {a['scheduled']} its `--ckpt-every` schedule wrote." if a["scheduled"] else ".")
        )
    md += [
        "",
        "> **The 12-checkpoint sweep this task asked for is not possible on these runs.** "
        "`src/train.py` writes every scheduled checkpoint to the same `last.pt`, "
        "overwriting the one before, so a finished run keeps only `last.pt` and the "
        "PSNR-chosen `best.pt`. The rule below therefore chooses between **two** "
        "full-split candidates per run. The dense 24-point curve in §5 comes from the "
        "training loop's own validation (400 patches, AMP). It shows the trajectory, "
        "but it is not the selection data. A true 12-point full-split sweep needs the "
        "runs repeated with per-checkpoint retention. This script scores every `*.pt` "
        "in a run directory, so it would pick those up unchanged.",
        "",
        "## 2. The table",
        "",
        table_markdown(main_rows, floor),
        "",
        "Supplementary (not one of the four rows; B1 was designated \"the primary "
        "result\" in `scripts/day3_runs.py::RUN_PLAN` before training):",
        "",
        table_markdown(supp_rows, floor, include_floor=False),
        "",
        f"¹ **Run A is not a controlled comparison.** It uses a different architecture "
        f"({ra['args']['n_resblocks']} blocks × {ra['args']['n_feats']} features, "
        f"**{ra['params']:,} parameters, {ra['params'] / ra['parameter_budget']:.2f}× the "
        f"{ra['parameter_budget']:,} budget**) and a different data pipeline: no "
        f"augmentation, and weight decay {ra['args'].get('wd', '?')} against A2/B's 0.0 "
        f"(`configs/frozen_day3.yaml`). It trained for {int(ra['args']['iters']):,} "
        f"iterations, but its training-loop validation PSNR peaked at iteration "
        f"{int(ra_peak['iter']):,} ({float(ra_peak['val_psnr']):.3f} dB), so the other "
        f"{int(ra['args']['iters']) - int(ra_peak['iter']):,} iterations bought nothing. "
        "It is the **training-efficiency observation**, the reason Day 3 runs 12k "
        "iterations, and not evidence about the spectral loss.",
        "",
        "Columns: PSNR/SSIM over all four bands (RGBN); LPIPS on RGB only (see caveat); "
        "SAM and ERGAS against the HR. `opensr …` are opensr-test's consistency "
        "metrics (LR vs SR downsampled back to LR by its own antialiased bilinear). "
        "`L1_spec`/`SAM_spec` are the same comparison through the area operator "
        "the spectral loss trains on. `L1_spec ÷ floor` is that value as a multiple "
        "of the ground-truth HR's. `spatial` is NaN where satalign rejects a "
        "translation; those patches are excluded from that column only.",
        "",
        f"Floor cross-check: L1_spec of the GT HR here is {floor['l1_spec']:.6f}; "
        f"`outputs/metrics/spectral_floor.json` holds {stored_floor:.6f}.",
        "",
        "## 3. Is B − A2 noise? Per-pair significance",
        "",
        f"Run B at its selected checkpoint ({chosen(treatment)}) minus Run A2 at its "
        f"selected checkpoint ({chosen(control)}), per validation pair, over all "
        f"{sig_by[metric]['n']} pairs. Bootstrap: {int(sig_cfg['n_boot']):,} replicates, "
        f"percentile {ci:.0%} CI on the mean delta, seed `cfg.seed`. The **pairs** CI "
        "resamples patches independently, as specified. The **tiles** CI resamples the "
        f"{sig_by[metric]['n_clusters']} source tiles and keeps each tile's patches "
        "together. Patches from one tile share a scene and a NAIP flight, so their "
        "deltas are correlated. The tile CI is the honest one, and every verdict uses "
        "it. Wilcoxon signed-rank is two-sided; \"p (tile means)\" runs it on per-tile "
        "mean deltas. No multiple-comparison correction is applied: the headline "
        f"claim rests on one pre-declared metric (`{metric}`), and the rest are context.",
        "",
        significance_markdown(significance, ci),
        "",
    ]
    if matched:
        md += [
            "**Same comparison at matched iterations** (both runs' checkpoints at the "
            "same step). This guards against the selection having picked different "
            "iterations:",
            "",
            "| iteration | metric | mean Δ (B − A2) | CI, tiles | verdict |",
            "|---|---|---:|---:|---|",
        ] + [
            f"| {m['iteration']} | `{m['metric']}` | {m['mean_delta']:+.3g} | "
            f"{_fmt_ci(m['ci_tile'])} | {m['verdict']} |" for m in matched
        ] + [""]
    md += ["## 4. Selection curves (full split, every checkpoint that exists)", ""]
    for label in selections:
        md += [selection_markdown(label, displays[label], selections[label],
                                  infos[label]), ""]
    md += [
        "## 5. Consistency vs iteration",
        "",
        f"![consistency vs iteration]({fig_rel})",
        "",
        "Lines: the training loop's validation (`log.csv`, every 500 iterations, "
        "400-patch subsample, AMP). Markers: the full split at the checkpoints that "
        "exist; the ring marks the selected one. Dashed: the spectral floor, the GT "
        "HR on the full split, in the same D=area units. The full training-loop curve "
        "for both runs, all rows:",
        "",
        log_curve_markdown(log_curves, displays),
        "",
        "## 6. Reproduce",
        "",
        "```",
        ".venv\\Scripts\\python.exe scripts/eval_all_ckpts.py --smoke   # pre-flight, CPU, minutes",
        ".venv\\Scripts\\python.exe scripts/eval_all_ckpts.py          # full; cached per pair",
        "```",
        "",
        f"Per-pair tables: `{Path(str(block['cache_dir'])).as_posix()}/<run>/<checkpoint>/"
        "per_pair.csv`, keyed on the weights hash and a fingerprint of every metric "
        "setting and of the split file.",
        "",
        f"> **LPIPS caveat.** {LPIPS_CAVEAT}",
        "",
    ]
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(md), encoding="utf-8")

    payload = {
        "written_utc": written, "smoke": bool(args.smoke), "n_pairs": n_pairs,
        "selection_rule": _plain(sel_cfg), "selections": selections,
        "checkpoints": {l: [{k: (str(v) if k == "path" else v) for k, v in c.items()
                             if k != "args"} for c in cs] for l, cs in checkpoints.items()},
        "availability": availability, "rows": main_rows, "supplementary_rows": supp_rows,
        "floor": floor, "floor_stored_l1_spec": stored_floor,
        "significance": significance, "matched_iterations": matched,
        "reading": sentences, "below_floor": below_floor,
        "figure": str(figure_path), "fingerprint": fingerprint,
        "oracle_opensr_skipped": len(oracle_meta.get("opensr_skipped", [])),
    }
    report_json = _path(block["report_json"])
    report_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # -- stdout: the auditable part -------------------------------------------
    for label in (control, treatment):
        print(f"\n### Selection curve -- {displays[label]}\n")
        print(selection_markdown(label, displays[label], selections[label], infos[label]))
    print("\n### Training-loop validation curves (subsample)\n")
    print(log_curve_markdown(log_curves, displays))
    print("\n### Headline table\n")
    print(table_markdown(main_rows, floor))
    print("\n### B - A2, per pair\n")
    print(significance_markdown(significance, ci))
    print("\n" + " ".join(sentences))
    print(f"\n  report: {report_md}\n  json:   {report_json}\n  figure: {figure_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
