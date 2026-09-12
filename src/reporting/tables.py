"""Day 6 tables: ablation, deployment, uncertainty, and the headline numbers.

Everything here ASSEMBLES existing evidence; nothing evaluates a model:

* Bicubic / A2 / B1 / B2 VAL numbers: ``reports/day3_ab_metrics.csv`` (full
  1,199-patch VAL, rows ``row == "selected"``), run commit fb8659d. Not
  re-evaluated (RULES.md Day 3 addendum).
* Run A: the ``eval_all_ckpts`` per-pair cache of its selected checkpoint
  (same evaluator, same 1,199 patches) plus ``reports/day2_runA.json`` for the
  architecture. Its Sobel / HF columns were never measured (the Day 3
  sharpness pass covered A2/B1/B2 only) and are shown as such.
* ``A2 + consistency projection``: ``reports/mvp/projection_gate.json`` (200
  VAL patches; flagged not-full-split).
* Deployment: ``reports/mvp/bench_*.json`` + ``quant_gate.json`` (+ ONNX parity).
* Uncertainty: ``reports/mvp/unc_eval_*.json`` (pending when absent).

Units: consistency L1 values are surface reflectance; SAM in degrees; LPIPS on
the RGB bands clipped to [0, 1] on a copy (see ``src.metrics.image_quality``).

Missing inputs render as ``pending`` rows, never as zeros.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

__all__ = [
    "WINNER",
    "WINNER_CHECKPOINT_ID",
    "ReportInputs",
    "TestSplitLocked",
    "guard_test_split",
    "read_ab_csv",
    "build_ablation",
    "render_ablation_md",
    "ablation_csv_rows",
    "build_deployment",
    "render_deployment_md",
    "build_uncertainty",
    "render_uncertainty_md",
    "build_headline",
]

#: Across-run winner, fixed by the Day 3 pre-registered verdict. Not re-selected.
WINNER = "a2"
WINNER_CHECKPOINT_ID = "a2-last-dce224ec"
DAY3_RUN_COMMIT = "fb8659d"

#: Train-arg keys expected to differ between A2 and B runs (lambdas) or that are
#: bookkeeping only; any OTHER difference is footnoted.
_ARGS_ALLOWED_TO_DIFFER = {"spectral_lambda1", "spectral_lambda2", "out", "max_hours"}

PENDING = "pending"


@dataclass
class ReportInputs:
    """Where every source lives. All paths are read only except ``mvp_dir``.

    Attributes:
        runs_root: Main-tree ``runs`` directory (``runA/``, ``day3/{a2,b1,b2}/``).
        eval_cache: ``eval_all_ckpts`` per-pair cache root.
        ab_csv: Day 3 A/B metric CSV for the requested split.
        ab_summary: ``reports/day3_ab_summary.json``.
        day3_results: ``reports/day3_results.json`` (floors, Run A selection).
        run_a_report: ``reports/day2_runA.json``.
        mvp_dir: ``reports/mvp`` (P4/P5/P6 reports in, P7 tables out).
        split: ``"val"`` or ``"test"``.
    """

    runs_root: Path
    eval_cache: Path
    ab_csv: Path
    ab_summary: Path
    day3_results: Path
    run_a_report: Path
    mvp_dir: Path
    split: str = "val"
    blur_index_min: float = 0.90
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- io


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _num(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def read_ab_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read the Day 3 A/B CSV into ``{method: row}``.

    Keeps the ``baseline`` row for bicubic and the ``selected`` row for each
    model run. Numeric fields become floats (``None`` when blank).

    Raises:
        KeyError: A required column is absent.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    required = {
        "method", "row", "iteration", "n_pairs", "psnr_db", "ssim", "lpips", "sam_deg",
        "ergas", "consistency_opensr_l1", "consistency_l1_spec_area",
        "sobel_ratio_vs_gt", "hf_ratio_vs_gt", "blur_index",
    }
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"{path} lacks columns {sorted(missing)}")
        for raw in reader:
            if raw["row"] not in ("baseline", "selected"):
                continue
            row: Dict[str, Any] = {"method": raw["method"], "row": raw["row"]}
            for key, value in raw.items():
                if key in ("method", "row"):
                    continue
                row[key] = _num(value)
            out[raw["method"]] = row
    return out


def _run_dir(runs_root: Path, run: str) -> Path:
    return Path(runs_root) / run if run == "runA" else Path(runs_root) / "day3" / run


def _run_metadata(runs_root: Path, run: str) -> Optional[Dict[str, Any]]:
    return _read_json(_run_dir(runs_root, run) / "run_metadata.json")


def _provenance_cells(meta: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not meta:
        return {"cfg_hash": None, "git_sha": None, "dirty": None}
    git = meta.get("git") or {}
    sha = meta.get("git_sha") or git.get("commit")
    dirty = meta.get("git_dirty", git.get("dirty"))
    cfg_hash = meta.get("config_hash")
    return {
        "cfg_hash": cfg_hash[:8] if cfg_hash else None,
        "git_sha": sha[:7] if sha else None,
        "dirty": None if dirty is None else bool(dirty),
    }


# ----------------------------------------------------------------- test lock


class TestSplitLocked(RuntimeError):
    """Raised when the test split is requested without the required flags."""


def guard_test_split(
    split: str,
    final: bool,
    force: bool,
    lock_path: Path,
    decisions_path: Path,
    git_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Enforce the one-shot test-split discipline.

    * ``val``: always allowed, the lock is not touched.
    * ``test`` without ``final``: refused.
    * ``test`` + ``final``, no lock: allowed; the lock file is created.
    * ``test`` + ``final`` with a lock: refused unless ``force``; a forced
      override appends one line to ``decisions_path``.

    Returns:
        ``{"split", "action"}`` with action ``none | created_lock | forced_override``.

    Raises:
        TestSplitLocked: The request is refused.
    """
    if split == "val":
        return {"split": split, "action": "none"}
    if split != "test":
        raise ValueError(f"split must be 'val' or 'test'; got {split!r}")
    if not final:
        raise TestSplitLocked("--split test requires --final (test is touched once, for the final table).")
    lock_path = Path(lock_path)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if lock_path.exists():
        if not force:
            raise TestSplitLocked(
                f"{lock_path} exists: the test split was already reported. Re-run with --final --force "
                "only if the override is intended (it is logged in docs/mvp/decisions.md)."
            )
        decisions_path = Path(decisions_path)
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(decisions_path, "a", encoding="utf-8") as fh:
            fh.write(
                f"P7 | make_tables --split test --final --force at {stamp} (git {git_sha or 'unknown'}) "
                f"| TEST_EVAL_LOCK overridden | explicit --force by operator\n"
            )
        return {"split": split, "action": "forced_override"}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"created_utc": stamp, "git_sha": git_sha, "by": "scripts/mvp/make_tables.py --final"}, indent=2),
        encoding="utf-8",
    )
    return {"split": split, "action": "created_lock"}


# ------------------------------------------------------------------ ablation

_ROW_SPECS = [
    # key, label, csv method (None = special source)
    ("bicubic", "Bicubic", "bicubic"),
    ("runA", "Run A †", None),
    ("a2", "A2 (control)", "a2"),
    ("b1", "B1 (λ₁=0.1, λ₂=0.02)", "b1"),
    ("b2", "B2 (λ₁=0.3, λ₂=0.06)", "b2"),
    ("best", "Best ★ (A2, Day 3 verdict)", "a2"),
]


def _empty_row(key: str, label: str, split: str) -> Dict[str, Any]:
    return {
        "key": key, "label": label, "status": PENDING, "split": split, "n": None,
        "lpips": None, "lpips_vs_bicubic_pct": None,
        "spec_l1_opensr": None, "spec_l1_train_op": None,
        "sam_deg": None, "hf_ratio_vs_gt": None, "blur_index": None, "blur_flag": False,
        "sobel_ratio_vs_gt": None, "ssim": None, "ergas": None, "psnr_db": None,
        "params": None, "cfg_hash": None, "git_sha": None, "dirty": None,
        "iteration": None, "checkpoint": None,
        "d_lpips_vs_a2_pct": None, "d_spec_l1_vs_a2_pct": None,
        "interim": False, "full_split": True, "notes": [], "source": None,
    }


def _fill_from_csv(row: Dict[str, Any], rec: Mapping[str, Any], source: Path) -> None:
    row.update(
        status="filled",
        n=int(rec["n_pairs"]) if rec.get("n_pairs") is not None else None,
        lpips=rec["lpips"],
        spec_l1_opensr=rec["consistency_opensr_l1"],
        spec_l1_train_op=rec["consistency_l1_spec_area"],
        sam_deg=rec["sam_deg"],
        hf_ratio_vs_gt=rec["hf_ratio_vs_gt"],
        blur_index=rec.get("blur_index"),
        sobel_ratio_vs_gt=rec["sobel_ratio_vs_gt"],
        ssim=rec["ssim"],
        ergas=rec["ergas"],
        psnr_db=rec["psnr_db"],
        iteration=int(rec["iteration"]) if rec.get("iteration") is not None else None,
        source=str(source),
    )


def _run_a_row(inputs: ReportInputs, row: Dict[str, Any]) -> None:
    """Fill the Run A † row from its cached per-pair eval (VAL only)."""
    report = _read_json(inputs.run_a_report)
    sel = _read_json(inputs.mvp_dir / "selection_runA.json")
    if sel is not None:
        label = sel["chosen"]["label"]
        ckpt = sel["chosen"]["path"]
    else:
        results = _read_json(inputs.day3_results) or {}
        label = ((results.get("selections") or {}).get("runA") or {}).get("selected")
        ckpt = None
        row["notes"].append("selection_runA.json absent; label from reports/day3_results.json")
    table = Path(inputs.eval_cache) / "runA" / str(label) / "per_pair.csv" if label else None
    if table is None or not table.is_file():
        row["notes"].append("Run A eval cache not found")
        return
    frame = pd.read_csv(table)
    means = {c: float(pd.to_numeric(frame[c], errors="coerce").mean()) for c in
             ("lpips", "reflectance", "l1_spec", "sam_mean_deg", "ssim_mean", "ergas", "psnr_mean")}
    row.update(
        status="filled", n=int(len(frame)), lpips=means["lpips"],
        spec_l1_opensr=means["reflectance"], spec_l1_train_op=means["l1_spec"],
        sam_deg=means["sam_mean_deg"], ssim=means["ssim_mean"], ergas=means["ergas"],
        psnr_db=means["psnr_mean"], iteration=int(str(label)[2:]), checkpoint=ckpt,
        interim=True, source=str(table),
    )
    row["notes"].append("HF/Sobel ratios not measured for Run A (Day 3 sharpness pass covered A2/B1/B2 only)")
    if report is not None:
        model = report.get("model") or {}
        row["params"] = model.get("parameters")
        if model.get("checkpoint_iteration") is not None and int(model["checkpoint_iteration"]) + 1 != row["iteration"]:
            row["notes"].append(
                f"day2_runA.json evaluated it{int(model['checkpoint_iteration']) + 1}, selection chose {label}"
            )
    row["notes"].append("Run A predates run_metadata.json: no cfg hash / git SHA / dirty flag recorded")


def _projection_row(inputs: ReportInputs, bicubic_lpips: Optional[float]) -> Optional[Dict[str, Any]]:
    gate = _read_json(inputs.mvp_dir / "projection_gate.json")
    if gate is None:
        return None
    row = _empty_row("a2_proj", "A2 + consistency projection", inputs.split)
    metrics = gate.get("metrics") or {}
    variant = f"proj{gate['iters']}" if gate.get("iters") else None
    if variant is None:
        # Gate failed: show the smallest-iteration variant, the one that would be served first.
        proj = sorted(k for k in metrics if k.startswith("proj"))
        variant = proj[0] if proj else None
    row["full_split"] = False
    row["split"] = gate.get("split", "val")
    row["n"] = gate.get("n_samples", gate.get("n"))
    row["notes"].append(f"n={row['n']} VAL patches, NOT the full split; compare only with the A2 row of the same file")
    row["gate_passed"] = bool(gate.get("passed"))
    row["notes"].append("projection gate " + ("PASSED" if gate.get("passed") else "FAILED: not served"))
    if variant is None or variant not in metrics:
        return row
    m = metrics[variant]
    a2 = metrics.get("a2") or {}
    opensr = gate.get("opensr") or {}
    row.update(
        status="filled", variant=variant, lpips=m.get("lpips"), spec_l1_train_op=m.get("spec_l1"),
        sam_deg=m.get("sam_deg"), hf_ratio_vs_gt=m.get("hf_ratio_vs_gt"), ssim=m.get("ssim"),
        ergas=m.get("ergas"), psnr_db=m.get("psnr"), params=855652 if gate.get("checkpoint_id") == WINNER_CHECKPOINT_ID else None,
        checkpoint=gate.get("checkpoint_id"), git_sha=(gate.get("git_sha") or "")[:7] or None,
        dirty=gate.get("git_dirty"), source=str(inputs.mvp_dir / "projection_gate.json"),
    )
    if a2.get("lpips"):
        row["d_lpips_vs_a2_pct"] = 100.0 * (m["lpips"] / a2["lpips"] - 1.0)
    if a2.get("spec_l1"):
        row["d_spec_l1_vs_a2_pct"] = 100.0 * (m["spec_l1"] / a2["spec_l1"] - 1.0)
        row["notes"].append("Δ spec L1 vs A2 uses the training operator (opensr measure not run)")
    if not opensr.get("ran", False):
        row["notes"].append(f"opensr consistency not run ({opensr.get('reason', 'no reason recorded')})")
    row["notes"].append("Sobel ratio not measured by the projection gate")
    return row


def build_ablation(inputs: ReportInputs) -> Dict[str, Any]:
    """Assemble the ablation table rows and footnotes.

    Returns:
        ``{"rows": [...], "footnotes": [...], "floor_train_op": float|None,
        "split": str}``; each row is a dict (see :func:`_empty_row`).
    """
    ab = read_ab_csv(inputs.ab_csv)
    summary = _read_json(inputs.ab_summary) or {}
    results = _read_json(inputs.day3_results) or {}
    floor = (results.get("floor") or {}).get("l1_spec")
    meta_summary = summary.get("metadata") or {}

    rows: List[Dict[str, Any]] = []
    for key, label, method in _ROW_SPECS:
        row = _empty_row(key, label, inputs.split)
        if key == "runA":
            if inputs.split == "val":
                _run_a_row(inputs, row)
            else:
                row["notes"].append("Run A has no test evaluation")
        elif method in ab:
            _fill_from_csv(row, ab[method], inputs.ab_csv)
            if method != "bicubic":
                prov = _provenance_cells(_run_metadata(inputs.runs_root, method))
                if prov["cfg_hash"] is None and method in meta_summary:
                    m = meta_summary[method]
                    prov = {"cfg_hash": (m.get("config_hash") or "")[:8] or None,
                            "git_sha": (m.get("git_sha") or "")[:7] or None,
                            "dirty": m.get("git_dirty")}
                row.update(prov)
                row["params"] = (meta_summary.get(method) or {}).get("params")
                sel = _read_json(inputs.mvp_dir / f"selection_{method}.json")
                if sel is not None:
                    row["checkpoint"] = sel["chosen"]["path"]
                    if row["iteration"] is not None and sel["chosen"]["iteration"] != row["iteration"]:
                        row["notes"].append(
                            f"MISMATCH: CSV row is it{row['iteration']}, selection_{method}.json chose "
                            f"it{sel['chosen']['iteration']}"
                        )
            else:
                row["params"] = 0
        rows.append(row)

    by_key = {r["key"]: r for r in rows}
    bic = by_key["bicubic"]
    a2 = by_key["a2"]
    for row in rows:
        if row["status"] != "filled":
            continue
        if bic["status"] == "filled" and row["lpips"] is not None and row["key"] != "bicubic":
            row["lpips_vs_bicubic_pct"] = 100.0 * (row["lpips"] / bic["lpips"] - 1.0)
        if row["key"] in ("b1", "b2") and a2["status"] == "filled":
            row["d_lpips_vs_a2_pct"] = 100.0 * (row["lpips"] / a2["lpips"] - 1.0)
            row["d_spec_l1_vs_a2_pct"] = 100.0 * (row["spec_l1_opensr"] / a2["spec_l1_opensr"] - 1.0)
        if row["key"] in ("b1", "b2") and row["blur_index"] is not None:
            row["blur_flag"] = row["blur_index"] < inputs.blur_index_min
        else:
            row["blur_index"] = None  # blur index is a B-row statistic only

    if inputs.split == "val":
        proj = _projection_row(inputs, bic.get("lpips"))
        if proj is not None:
            rows.append(proj)

    return {
        "split": inputs.split,
        "rows": rows,
        "footnotes": _footnotes(inputs, rows),
        "floor_train_op": floor,
        "blur_index_min": inputs.blur_index_min,
    }


def _footnotes(inputs: ReportInputs, rows: Sequence[Mapping[str, Any]]) -> List[str]:
    notes = [
        "† Run A is not a valid control. It differs from A2/B on four axes: architecture "
        "(1.52M vs 855,652 params, over the 1M budget), no dihedral augmentation, frozen crop origins "
        "(2,400 fixed crops for all 40k iterations due to infinite(train_dl) with persistent_workers), "
        "and dataset size. Its numbers carry \"interim\": true.",
        f"A2, B1 and B2 executed at commit {DAY3_RUN_COMMIT} (12,000 iterations each); their VAL numbers are "
        "taken from reports/day3_ab_metrics.csv (full 1,199-patch VAL) and not re-evaluated.",
        "★ Best = A2, fixed by the pre-registered Day 3 verdict (training-time spectral loss did not help: "
        "B1 LPIPS +3.9% > 2% limit and blur index 0.847 < 0.90). Not re-selected across runs.",
        f"Blur index = (HF_B − HF_bicubic) / (HF_A2 − HF_bicubic); rows below {inputs.blur_index_min:.2f} are "
        "flagged BLUR HAZARD. The HF energy ratio governs; Sobel is reported alongside.",
        "Spec consistency L1 (opensr) = opensr-test `reflectance` measure (headline). Training-operator column = "
        "down4 area L1 vs LR; the GT floor is the same measure with HR in place of SR.",
    ]
    dirty = [r["label"] for r in rows if r.get("dirty")]
    if dirty:
        notes.append("Dirty working tree at run/eval time: " + ", ".join(dirty) + ".")
    # Config-hash / train-arg differences beyond lambdas.
    a2_meta = _run_metadata(inputs.runs_root, "a2")
    for run in ("b1", "b2"):
        meta = _run_metadata(inputs.runs_root, run)
        if not (a2_meta and meta):
            continue
        diffs = []
        if a2_meta.get("config_hash") != meta.get("config_hash"):
            diffs.append("config_hash")
        a_args, b_args = a2_meta.get("args") or {}, meta.get("args") or {}
        for k in sorted(set(a_args) | set(b_args)):
            if k not in _ARGS_ALLOWED_TO_DIFFER and a_args.get(k) != b_args.get(k):
                diffs.append(f"args.{k} ({a_args.get(k)!r} vs {b_args.get(k)!r})")
        if diffs:
            notes.append(f"{run.upper()} differs from A2 beyond lambdas: " + "; ".join(diffs) + ".")
    for r in rows:
        if not r.get("full_split", True):
            notes.append(f"{r['label']}: " + "; ".join(r["notes"]) + ".")
    return notes


# ------------------------------------------------------------------ render


def _f(value: Optional[float], digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:+.1f}%"


def render_ablation_md(ablation: Mapping[str, Any]) -> str:
    """Render the ablation table as Markdown (one line per row, footnotes below)."""
    floor = ablation.get("floor_train_op")
    floor_txt = f"GT floor {floor:.6f}" if floor is not None else "GT floor n/a"
    head = [
        "Row", "LPIPS↓ (vs bicubic)", "Spec L1 opensr↓", f"Spec L1 training op↓ ({floor_txt})", "SAM°↓",
        "HF ratio vs GT", "Blur index", "Sobel ratio vs GT", "SSIM↑", "ERGAS↓", "PSNR↑ dB",
        "Params", "Split", "n", "cfg hash", "git SHA", "dirty",
    ]
    lines = [
        f"# Ablation ({ablation['split'].upper()})",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for r in ablation["rows"]:
        if r["status"] != "filled":
            cells = [r["label"], PENDING] + ["—"] * (len(head) - 2)
            cells[12] = r["split"]
            lines.append("| " + " | ".join(cells) + " |")
            continue
        lp = _f(r["lpips"])
        if r["lpips_vs_bicubic_pct"] is not None:
            lp += f" ({_pct(r['lpips_vs_bicubic_pct'])})"
        spec = _f(r["spec_l1_opensr"], 6) if r["spec_l1_opensr"] is not None else ("not run" if r["key"] == "a2_proj" else "—")
        if r["key"] in ("b1", "b2", "a2_proj") and r["d_lpips_vs_a2_pct"] is not None:
            lp += f"; Δ vs A2 {_pct(r['d_lpips_vs_a2_pct'])}"
        train = _f(r["spec_l1_train_op"], 6)
        if r["key"] in ("b1", "b2") and r["d_spec_l1_vs_a2_pct"] is not None:
            spec += f" (Δ vs A2 {_pct(r['d_spec_l1_vs_a2_pct'])})"
        if r["key"] == "a2_proj" and r["d_spec_l1_vs_a2_pct"] is not None:
            train += f" (Δ vs A2 {_pct(r['d_spec_l1_vs_a2_pct'])})"
        blur = _f(r["blur_index"], 3)
        if r["blur_flag"]:
            blur += " ⚠ BLUR HAZARD"
        not_measured = "not measured"
        split = r["split"] + ("" if r.get("full_split", True) else " (subset)")
        dirty = "—" if r["dirty"] is None else ("yes" if r["dirty"] else "no")
        cells = [
            r["label"], lp, spec, train, _f(r["sam_deg"], 3),
            _f(r["hf_ratio_vs_gt"]) if r["hf_ratio_vs_gt"] is not None else not_measured,
            blur,
            _f(r["sobel_ratio_vs_gt"]) if r["sobel_ratio_vs_gt"] is not None else not_measured,
            _f(r["ssim"]), _f(r["ergas"]), _f(r["psnr_db"], 3),
            "—" if r["params"] is None else f"{int(r['params']):,}",
            split, "—" if r["n"] is None else str(r["n"]),
            r["cfg_hash"] or "—", r["git_sha"] or "—", dirty,
        ]
        if r["key"] == "bicubic":
            cells[14:17] = ["—", "—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "**Notes**", ""]
    lines += [f"- {n}" for n in ablation["footnotes"]]
    extra = [f"- {r['label']}: {'; '.join(r['notes'])}." for r in ablation["rows"] if r["notes"] and r.get("full_split", True)]
    lines += extra
    return "\n".join(lines) + "\n"


_CSV_FIELDS = [
    "key", "label", "status", "split", "n", "lpips", "lpips_vs_bicubic_pct", "spec_l1_opensr",
    "spec_l1_train_op", "sam_deg", "hf_ratio_vs_gt", "blur_index", "blur_flag", "sobel_ratio_vs_gt",
    "ssim", "ergas", "psnr_db", "params", "cfg_hash", "git_sha", "dirty", "iteration", "checkpoint",
    "d_lpips_vs_a2_pct", "d_spec_l1_vs_a2_pct", "interim", "full_split", "source", "notes",
]


def ablation_csv_rows(ablation: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    """Flat rows for ``ablation.csv`` (notes joined with ``"; "``)."""
    for r in ablation["rows"]:
        out = {k: r.get(k) for k in _CSV_FIELDS}
        out["notes"] = "; ".join(r.get("notes") or [])
        yield out


# --------------------------------------------------------------- deployment


def build_deployment(mvp_dir: Path, checkpoint_id: str = WINNER_CHECKPOINT_ID) -> Dict[str, Any]:
    """Deployment rows (torch-fp32 / onnx-fp32 / onnx-int8) from the P4 reports.

    Δ columns are backend − FP32 on the quant-gate VAL set (int8 − fp32, so a
    negative ΔPSNR is a loss). Missing bench → every row pending.
    """
    mvp_dir = Path(mvp_dir)
    bench_path = mvp_dir / f"bench_{checkpoint_id}.json"
    bench = _read_json(bench_path)
    quant_all = _read_json(mvp_dir / "quant_gate.json") or {}
    quant = quant_all.get(checkpoint_id) or {}
    parity = _read_json(mvp_dir / f"onnx_parity_{checkpoint_id}.json") or {}
    backends = {b["backend"]: b for b in (bench or {}).get("backends", [])}
    rows = []
    for name in ("torch-fp32", "onnx-fp32", "onnx-int8"):
        row: Dict[str, Any] = {"backend": name, "status": PENDING, "params": None, "bytes": None,
                               "median_ms": None, "p90_ms": None, "d_psnr_db": None, "d_sam_deg": None,
                               "d_lpips": None, "provisional": None, "gate_passed": None, "notes": []}
        b = backends.get(name)
        if b is not None:
            row.update(status="filled", params=855652 if checkpoint_id == WINNER_CHECKPOINT_ID else None,
                       bytes=b.get("model_bytes"), median_ms=b.get("median_ms"), p90_ms=b.get("p90_ms"),
                       provisional=bool(bench.get("provisional", True)), gate_passed=b.get("gate_passed"))
            if b.get("note"):
                row["notes"].append(b["note"])
        if name == "torch-fp32" and b is not None:
            row.update(d_psnr_db=0.0, d_sam_deg=0.0, d_lpips=0.0)
            row["notes"].append("reference")
        elif name == "onnx-fp32" and b is not None:
            diff = (parity.get("raw_graph") or {}).get("max_abs_diff")
            row.update(d_psnr_db=0.0, d_sam_deg=0.0, d_lpips=0.0)
            if diff is not None:
                row["notes"].append(f"parity max abs diff {diff:.1e} reflectance vs torch (metric Δ ≈ 0)")
        elif name == "onnx-int8" and b is not None:
            attempt = next((a for a in quant.get("attempts", []) if a.get("bytes") == b.get("model_bytes")), None)
            if attempt is not None:
                row.update(
                    d_psnr_db=attempt["int8"]["psnr_db"] - attempt["fp32"]["psnr_db"],
                    d_sam_deg=attempt["int8"]["sam_deg"] - attempt["fp32"]["sam_deg"],
                    d_lpips=attempt["int8"]["lpips"] - attempt["fp32"]["lpips"],
                    gate_passed=bool(attempt.get("passed")),
                )
                row["notes"].append(f"quant attempt {attempt['attempt']}, n={attempt['int8'].get('n')} VAL")
            else:
                row["notes"].append("no quant_gate attempt matches the benchmarked INT8 file size")
            if not quant.get("passed", False):
                row["notes"].append(f"INT8 gate FAILED; served backend = {quant.get('default_backend', 'n/a')}")
        rows.append(row)
    return {
        "checkpoint_id": checkpoint_id,
        "rows": rows,
        "bench_source": str(bench_path) if bench else None,
        "threads": (bench or {}).get("threads"),
        "input": (bench or {}).get("input"),
        "load_percent_before": (bench or {}).get("load_percent_before"),
        "quant_gate": quant.get("gate"),
        "quant_passed": quant.get("passed"),
        "default_backend": quant.get("default_backend"),
    }


def render_deployment_md(dep: Mapping[str, Any]) -> str:
    head = ["Backend", "Params", "Bytes", "Median ms", "p90 ms", "ΔPSNR dB", "ΔSAM°", "ΔLPIPS", "Provisional", "Notes"]
    lines = [
        f"# Deployment ({dep['checkpoint_id']}, 256² LR, {dep.get('threads') or 6} threads)",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for r in dep["rows"]:
        if r["status"] != "filled":
            lines.append("| " + " | ".join([r["backend"], PENDING] + ["—"] * (len(head) - 2)) + " |")
            continue
        lines.append("| " + " | ".join([
            r["backend"], f"{r['params']:,}" if r["params"] else "—", f"{r['bytes']:,}" if r["bytes"] else "—",
            _f(r["median_ms"], 1), _f(r["p90_ms"], 1),
            f"{r['d_psnr_db']:+.3f}" if r["d_psnr_db"] is not None else "—",
            f"{r['d_sam_deg']:+.3f}" if r["d_sam_deg"] is not None else "—",
            f"{r['d_lpips']:+.4f}" if r["d_lpips"] is not None else "—",
            "yes" if r["provisional"] else "no", "; ".join(r["notes"]) or "—",
        ]) + " |")
    lines += [
        "",
        f"- Input: {dep.get('input') or 'n/a'}; CPU load before benchmark {dep.get('load_percent_before')}%.",
        f"- Δ = backend − FP32 on the quant-gate VAL set; gate {dep.get('quant_gate')}, passed={dep.get('quant_passed')}, "
        f"served backend {dep.get('default_backend')}.",
    ]
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------- uncertainty

_UNC_METHODS = [("tta4", "TTA-4"), ("tta8", "TTA-8"), ("learnedlaplace", "Learned Laplace")]
_UNC_ALIASES = {
    "ause": ("ause", "AUSE", "ause_rmse", "ause_mae"),
    "spearman": ("spearman", "spearman_rho", "rho", "spearman_unc_vs_abs_err"),
    "cov50": ("coverage_50", "cov50", "coverage50", "coverage@50"),
    "cov90": ("coverage_90", "cov90", "coverage90", "coverage@90"),
    "cost": ("cost_forward_passes", "forward_passes", "cost_x", "cost"),
    "collapse": ("collapse_verdict", "collapse", "verdict"),
}


def _norm(name: str) -> str:
    name = re.sub(r"[^a-z0-9]", "", name.lower())
    return "learnedlaplace" if name in ("laplace", "learned", "runc", "laplacehead") else name


def _find(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    cov = d.get("coverage")
    if isinstance(cov, Mapping):
        for k in keys:
            tail = k.split("_")[-1].replace("cov", "").replace("coverage", "").lstrip("@")
            for ck in (tail, f"{tail}%", f"p{tail}", f"{int(tail) / 100:.1f}" if tail.isdigit() else tail):
                if ck in cov:
                    return cov[ck]
    return None


def _unc_entries(doc: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    if isinstance(doc.get("methods"), Mapping):
        return {_norm(k): v for k, v in doc["methods"].items()}
    if "method" in doc:
        return {_norm(str(doc["method"])): doc}
    return {_norm(k): v for k, v in doc.items() if isinstance(v, Mapping)}


def build_uncertainty(mvp_dir: Path) -> Dict[str, Any]:
    """Uncertainty rows from ``reports/mvp/unc_eval_*.json`` (schema-tolerant).

    Recognised layouts: ``{"methods": {name: {...}}}``, one method per file with
    a ``"method"`` key, or ``{name: {...}}``. Unrecognised fields leave cells
    blank; a method with no file stays ``pending``.
    """
    found: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, str] = {}
    for path in sorted(Path(mvp_dir).glob("unc_eval_*.json")):
        doc = _read_json(path) or {}
        for key, entry in _unc_entries(doc).items():
            found[key] = {**{k: doc.get(k) for k in ("checkpoint_id", "split", "n_samples", "interim")}, **entry}
            sources[key] = str(path)
    rows = []
    for key, label in _UNC_METHODS:
        entry = found.get(key)
        row: Dict[str, Any] = {"method": label, "status": PENDING, "source": sources.get(key)}
        for field_, aliases in _UNC_ALIASES.items():
            row[field_] = _find(entry, aliases) if entry else None
        if entry is not None:
            row["status"] = "filled"
            row.update({k: entry.get(k) for k in ("checkpoint_id", "split", "n_samples", "interim")})
        rows.append(row)
    return {"rows": rows, "files": sorted(set(sources.values()))}


def render_uncertainty_md(unc: Mapping[str, Any]) -> str:
    head = ["Method", "AUSE↓", "Spearman ρ↑", "50% coverage", "90% coverage", "Cost (× forward)", "Collapse verdict", "Split / n"]
    lines = ["# Uncertainty (A2 backbone)", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in unc["rows"]:
        if r["status"] != "filled":
            lines.append("| " + " | ".join([r["method"], PENDING] + ["—"] * (len(head) - 2)) + " |")
            continue

        def cell(v: Any, d: int = 3) -> str:
            return _f(float(v), d) if isinstance(v, (int, float)) and not isinstance(v, bool) else ("—" if v is None else str(v))

        lines.append("| " + " | ".join([
            r["method"], cell(r["ause"], 4), cell(r["spearman"]), cell(r["cov50"]), cell(r["cov90"]),
            cell(r["cost"], 1), cell(r["collapse"]), f"{r.get('split') or '—'} / {r.get('n_samples') or '—'}",
        ]) + " |")
    if not unc["files"]:
        lines += ["", "- No reports/mvp/unc_eval_*.json found yet (P6 pending)."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- headline


def build_headline(ablation: Mapping[str, Any], dep: Mapping[str, Any], inputs: ReportInputs) -> Dict[str, Any]:
    """The three lead numbers, each with provenance."""
    rows = {r["key"]: r for r in ablation["rows"]}
    win, bic = rows["best"], rows["bicubic"]
    results = _read_json(inputs.day3_results) or {}
    hr_ref = (results.get("floor") or {}).get("reflectance")
    prov = {"source": str(inputs.ab_csv), "checkpoint_id": WINNER_CHECKPOINT_ID, "split": ablation["split"],
            "n_samples": win.get("n"), "interim": False, "run_commit": DAY3_RUN_COMMIT}
    out: Dict[str, Any] = {"winner": "A2", "winner_checkpoint_id": WINNER_CHECKPOINT_ID}
    if win["status"] == "filled" and bic["status"] == "filled":
        out["lpips"] = {
            "winner": win["lpips"], "bicubic": bic["lpips"],
            "relative_change_vs_bicubic": win["lpips"] / bic["lpips"] - 1.0, "provenance": prov,
        }
        out["spectral_consistency_opensr"] = {
            "measure": "opensr-test reflectance L1 (LR vs SR degraded to 10 m), surface reflectance",
            "winner": win["spec_l1_opensr"], "bicubic": bic["spec_l1_opensr"], "hr_reference": hr_ref,
            "winner_vs_bicubic_ratio": win["spec_l1_opensr"] / bic["spec_l1_opensr"],
            "winner_vs_hr_reference_ratio": (win["spec_l1_opensr"] / hr_ref) if hr_ref else None,
            "provenance": {**prov, "hr_reference_source": f"{inputs.day3_results} floor.reflectance (HR in place of SR)"},
        }
    else:
        out["lpips"] = out["spectral_consistency_opensr"] = {"status": PENDING}
    proj = rows.get("a2_proj")
    if proj is not None and proj.get("gate_passed"):
        out["a2_plus_projection"] = {k: proj[k] for k in ("lpips", "spec_l1_train_op", "n", "variant")}
        out["a2_plus_projection"]["provenance"] = {"source": proj["source"], "checkpoint_id": proj["checkpoint"],
                                                    "split": proj["split"], "full_split": False, "interim": False}
    else:
        out["a2_plus_projection"] = {"included": False, "reason": "projection gate not passed" if proj else "no projection_gate.json"}
    int8 = next((r for r in dep["rows"] if r["backend"] == "onnx-int8"), None)
    fp32 = next((r for r in dep["rows"] if r["backend"] == "onnx-fp32"), None)
    if int8 and int8["status"] == "filled":
        out["int8"] = {
            "bytes": int8["bytes"], "median_ms": int8["median_ms"], "p90_ms": int8["p90_ms"],
            "gate_passed": int8["gate_passed"], "served": bool(dep.get("quant_passed")),
            "served_backend": dep.get("default_backend"),
            "fp32_bytes": fp32["bytes"] if fp32 else None, "fp32_median_ms": fp32["median_ms"] if fp32 else None,
            "provenance": {"source": dep["bench_source"], "quant_gate": str(Path(inputs.mvp_dir) / "quant_gate.json"),
                           "checkpoint_id": dep["checkpoint_id"], "split": None, "input": dep.get("input"),
                           "threads": dep.get("threads"), "provisional": int8["provisional"], "interim": False},
        }
    else:
        out["int8"] = {"status": PENDING}
    return out
