"""P7: within-run checkpoint selection with the repo's pre-registered rule.

Wraps src.eval.ckpt_selection.select_checkpoint (config eval_all_ckpts.selection:
lowest opensr-test `reflectance`, CI tie, LPIPS tie-break) over the cached
full-VAL per-pair tables of scripts/eval_all_ckpts.py. Never uses test. Writes
reports/mvp/selection_<run>.json per run.

Run:
    python scripts/mvp/select_checkpoint.py --runs-root D:/SIH/DrishtiSR/runs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))

from src.reporting.selection import select_run  # noqa: E402
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.gitmeta import git_metadata  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

RUN_DIRS = {"runA": "runA", "a2": "day3/a2", "b1": "day3/b1", "b2": "day3/b2"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    parser.add_argument("--runs-root", required=True, type=Path, help="Main-tree runs directory (read only).")
    parser.add_argument("--eval-cache", type=Path, default=None,
                        help="eval_all_ckpts cache root (default: <runs-root>/../outputs/metrics/eval_all_ckpts).")
    parser.add_argument("--runs", nargs="+", default=list(RUN_DIRS), choices=list(RUN_DIRS))
    parser.add_argument("--out-dir", type=Path, default=WORKTREE / "reports" / "mvp")
    args = parser.parse_args(argv)

    log = get_logger("p7.select")
    cfg = load_config(args.config, smoke=args.smoke, overrides=getattr(args, "set", None))
    rule_cfg = dict(cfg.eval_all_ckpts.selection)
    cache = args.eval_cache or args.runs_root.parent / "outputs" / "metrics" / "eval_all_ckpts"
    git = git_metadata(WORKTREE)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        record = select_run(run, args.runs_root / RUN_DIRS[run], cache, rule_cfg, seed=int(cfg.seed))
        record.update(git_sha=git.get("commit"), git_dirty=git.get("dirty"),
                      interim=(run == "runA"), timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        out = args.out_dir / f"selection_{run}.json"
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        log.info("%s: chose %s (%s); best.pt it%s coincides=%s -> %s", run, record["chosen"]["label"],
                 record["chosen"]["path"], record["best_pt"]["iteration"], record["best_pt"]["coincides"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
