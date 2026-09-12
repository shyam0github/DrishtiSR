"""P7: Day 6 ablation / deployment / uncertainty tables and headline.json.

Assembles existing evidence only (see src/reporting/tables.py for sources).

    python scripts/mvp/make_tables.py --runs-root D:/SIH/DrishtiSR/runs [--split val|test] [--final] [--force]

--split test requires --final and creates reports/mvp/TEST_EVAL_LOCK; with the
lock present it refuses unless --final --force (logged in docs/mvp/decisions.md).
Test-split model rows are read from reports/mvp/test_ab_metrics.csv (same
columns as reports/day3_ab_metrics.csv) and are pending when it is absent.
Outputs for test carry a ``_test`` suffix so the VAL tables are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))

from src.reporting import tables as T  # noqa: E402
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.gitmeta import git_metadata  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def run(args: argparse.Namespace, reports: Path, decisions: Path) -> dict:
    """Build and write every table; returns a small summary (used by tests)."""
    log = get_logger("p7.tables")
    mvp = reports / "mvp"
    git = git_metadata(WORKTREE)
    guard = T.guard_test_split(args.split, args.final, args.force, mvp / "TEST_EVAL_LOCK", decisions,
                               git_sha=git.get("commit"))
    cfg = load_config(args.config, smoke=args.smoke, overrides=getattr(args, "set", None))
    inputs = T.ReportInputs(
        runs_root=args.runs_root,
        eval_cache=args.eval_cache or args.runs_root.parent / "outputs" / "metrics" / "eval_all_ckpts",
        ab_csv=(reports / "day3_ab_metrics.csv") if args.split == "val" else (mvp / "test_ab_metrics.csv"),
        ab_summary=reports / "day3_ab_summary.json",
        day3_results=reports / "day3_results.json",
        run_a_report=reports / "day2_runA.json",
        mvp_dir=mvp,
        split=args.split,
        blur_index_min=float(cfg.day3_ab_report.blur_index_min),
    )
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prov = {"git_sha": git.get("commit"), "git_dirty": git.get("dirty"), "timestamp_utc": stamp,
            "split": args.split, "test_guard": guard}
    sfx = "" if args.split == "val" else "_test"

    abl = T.build_ablation(inputs)
    (mvp / f"ablation{sfx}.md").write_text(T.render_ablation_md(abl), encoding="utf-8")
    with open(mvp / f"ablation{sfx}.csv", "w", newline="", encoding="utf-8") as fh:
        rows = list(T.ablation_csv_rows(abl))
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (mvp / f"ablation{sfx}.json").write_text(json.dumps({**abl, **prov, "checkpoint_id": T.WINNER_CHECKPOINT_ID,
                                                          "n_samples": next((r["n"] for r in abl["rows"] if r["key"] == "best"), None)},
                                                         indent=2), encoding="utf-8")

    dep = T.build_deployment(mvp)
    (mvp / "deploy_table.md").write_text(T.render_deployment_md(dep), encoding="utf-8")
    unc = T.build_uncertainty(mvp)
    (mvp / "uncertainty_table.md").write_text(T.render_uncertainty_md(unc), encoding="utf-8")
    head = T.build_headline(abl, dep, inputs)
    (mvp / f"headline{sfx}.json").write_text(json.dumps({**head, **prov}, indent=2), encoding="utf-8")

    summary = {
        "ablation_filled": [r["label"] for r in abl["rows"] if r["status"] == "filled"],
        "ablation_pending": [r["label"] for r in abl["rows"] if r["status"] != "filled"],
        "deploy_pending": [r["backend"] for r in dep["rows"] if r["status"] != "filled"],
        "uncertainty_pending": [r["method"] for r in unc["rows"] if r["status"] != "filled"],
        "guard": guard,
    }
    # Counts only: row labels carry non-ASCII glyphs the Windows console cannot encode.
    log.info("tables written to %s: ablation %d filled / %d pending, deploy %d pending, uncertainty %d pending, guard %s",
             mvp, len(summary["ablation_filled"]), len(summary["ablation_pending"]),
             len(summary["deploy_pending"]), len(summary["uncertainty_pending"]), guard["action"])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    parser.add_argument("--runs-root", required=True, type=Path, help="Main-tree runs directory (read only).")
    parser.add_argument("--eval-cache", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reports-dir", type=Path, default=WORKTREE / "reports")
    parser.add_argument("--decisions", type=Path, default=WORKTREE / "docs" / "mvp" / "decisions.md")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args, args.reports_dir, args.decisions)
    except T.TestSplitLocked as exc:
        get_logger("p7.tables").error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
