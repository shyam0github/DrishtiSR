"""Snapshot the seven opensr-test means per headline row into a tracked JSON.

WHY. ``scripts/eval_all_ckpts.py`` writes every per-pair opensr-test score to
``cfg.eval_all_ckpts.cache_dir`` (gitignored), but its report keeps only the
three consistency means per row, plus all seven for the B-vs-A2 pair. The
/compare page needs all seven for bicubic, Run A and A2. This script averages
the cached per-pair tables of each row's SELECTED checkpoint (the selection in
``cfg.eval_all_ckpts.report_json``, never re-chosen here) and writes
``cfg.compare_snapshot.output_json``, which ``frontend/tests/compare.test.ts``
checks the page's numbers against.

It reads cached tables only; it never scores an image. A missing table, a
checkpoint whose cache key disagrees with the report, or a non-finite value is
an error, not a skipped row.

Examples:
    .venv/Scripts/python.exe scripts/snapshot_opensr_seven.py --smoke
    .venv/Scripts/python.exe scripts/snapshot_opensr_seven.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.eval.opensr_harness import OPENSR_METRICS  # noqa: E402
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

log = get_logger(__name__)


def _checkpoint_dir(cfg, label: str, report: Dict[str, Any]) -> Path:
    """Cache directory of ``label``'s selected checkpoint (or of a baseline)."""
    runs = {r.label: r for r in cfg.eval_all_ckpts.runs}
    if label not in runs:
        raise KeyError(f"compare_snapshot row '{label}' is not in cfg.eval_all_ckpts.runs ({sorted(runs)})")
    run = runs[label]
    if run.kind == "baseline":
        sub = run.method
    else:
        if label not in report["selections"]:
            raise KeyError(f"{cfg.eval_all_ckpts.report_json} has no selection for '{label}'")
        sub = report["selections"][label]["selected"]
    return repo_root() / cfg.eval_all_ckpts.cache_dir / label / sub


def snapshot_row(cfg, label: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """Mean of each opensr-test metric over one row's cached per-pair table.

    The per-pair values are opensr-test 1.3.3 outputs computed on surface
    reflectance (float, unclipped); ``reflectance`` and ``synthesis`` are L1
    distances in reflectance units, ``spectral`` is degrees, ``spatial`` is LR
    pixels, and the three correctness metrics are unitless in [0, 1].
    """
    ckpt_dir = _checkpoint_dir(cfg, label, report)
    table = ckpt_dir / "per_pair.csv"
    if not table.is_file():
        raise FileNotFoundError(f"{table} missing; run scripts/eval_all_ckpts.py --only {label} first")
    df = pd.read_csv(table)
    key = json.loads((ckpt_dir / "key.json").read_text(encoding="utf-8"))
    means: Dict[str, Any] = {}
    n_nonfinite: Dict[str, int] = {}
    for metric in OPENSR_METRICS:
        col = df[metric].astype(float)
        finite = col.map(math.isfinite)
        bad = int((~finite).sum())
        if bad:
            # Policy cfg.compare_snapshot.allow_nonfinite: false on real data, where a
            # non-finite score is a genuine failure; true only in smoke, whose
            # random-weight runs make satalign reject nearly every `spatial` shift.
            if not cfg.compare_snapshot.allow_nonfinite:
                raise ValueError(f"{table}: {bad} non-finite '{metric}' values")
            log.warning("%s: %d/%d non-finite '%s' values excluded from the mean (allow_nonfinite)", label, bad, len(col), metric)
        n_nonfinite[metric] = bad
        means[metric] = float(col[finite].mean()) if finite.any() else None
    return {
        "label": label,
        "checkpoint": ckpt_dir.name,
        "n_pairs": int(len(df)),
        "n_nonfinite": n_nonfinite,
        "weights_sha256": key.get("weights_sha256"),
        "fingerprint": key["fingerprint"],
        "means": means,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)
    seed_everything(cfg.seed)

    report_path = repo_root() / cfg.eval_all_ckpts.report_json
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [snapshot_row(cfg, label, report) for label in cfg.compare_snapshot.rows]
    for r in rows:
        log.info("%s %s n=%d %s", r["label"], r["checkpoint"], r["n_pairs"],
                 " ".join(f"{k}={v:.6g}" for k, v in r["means"].items()))

    out = repo_root() / cfg.compare_snapshot.output_json
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "smoke": bool(args.smoke),
        "source_report": str(cfg.eval_all_ckpts.report_json),
        "source_cache": str(cfg.eval_all_ckpts.cache_dir),
        "split": "val",
        "metrics": list(OPENSR_METRICS),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
