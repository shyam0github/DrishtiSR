"""P6: score an uncertainty method on VAL patches, method-agnostically.

--method learned : head checkpoint from train_unc.py (--ckpt); unc = Laplace scale b.
--method tta4/8  : A2 (--ckpt, default the config backbone) via TorchPredictor and
                   src.infer.trust (imported lazily; needs the [mvp/P5] commit);
                   unc = per-band std across dihedral TTA.

Patches: P5's VAL pool (scripts/mvp/calibrate_trust_scales.ValPatches), drawn with
seed 26143 EXCLUDING the collapse-watch subset (seed 26142), so the monitoring
patches are never scored. Writes reports/mvp/unc_eval_<method>_<checkpoint_id>.json.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from src.infer.predictor import checkpoint_id  # noqa: E402
from src.uncertainty.calibration import calibration_report  # noqa: E402
from src.utils.gitmeta import git_metadata  # noqa: E402

REPORTS = WORKTREE / "reports" / "mvp"
A2 = Path(r"D:\SIH\DrishtiSR\runs\day3\a2\last.pt")
WATCH_SEED, EVAL_SEED = 26142, 26143


def p5_committed() -> bool:
    r = subprocess.run(["git", "log", "--oneline", "--fixed-strings", "--grep", "[mvp/P5]"],
                       cwd=WORKTREE, capture_output=True, text=True)
    return bool(r.stdout.strip())


def draw_eval_patches(n: int) -> List[Dict[str, Any]]:
    sys.path.insert(0, str(WORKTREE / "scripts" / "mvp"))
    from train_unc import val_pool  # noqa: E402

    vp = val_pool()
    pool = list(range(len(vp.pool)))
    watch_idx = set(np.random.default_rng(WATCH_SEED).choice(len(pool), size=128, replace=False).tolist())
    free = [i for i in pool if i not in watch_idx]
    pick = np.random.default_rng(EVAL_SEED).choice(len(free), size=n, replace=False)
    from src.data.patches import crop_pair

    out, cache = [], {}
    for j in pick:
        sid, (r, c) = vp.pool[free[int(j)]]
        if sid not in cache:
            cache[sid] = vp.load_tile(sid)
        lr, hr = cache[sid]
        p = crop_pair(lr, hr, r, c, vp.size, vp.scale)
        out.append({"sample_id": sid, "origin": [int(r), int(c)],
                    "lr": np.ascontiguousarray(p["lr"], np.float32), "hr": np.ascontiguousarray(p["hr"], np.float32)})
    return out


def _median_time(fn, reps: int = 3) -> float:
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def eval_learned(ckpt: Path, patches) -> Dict[str, Any]:
    from src.uncertainty.head import load_unc_checkpoint

    model, payload = load_unc_checkpoint(ckpt)
    lr = torch.from_numpy(np.stack([p["lr"] for p in patches]))
    hr = np.stack([p["hr"] for p in patches])
    with torch.no_grad():
        outs = [model(lr[s:s + 16]) for s in range(0, len(lr), 16)]
    mu = torch.cat([o[0] for o in outs]).numpy()
    b = torch.cat([o[1] for o in outs]).numpy()
    rep = calibration_report(np.abs(mu - hr), b, laplace=True)
    x = lr[:8]
    with torch.no_grad():
        t_base = _median_time(lambda: model.backbone(x))
        t_meth = _median_time(lambda: model(x))
    summary_path = ckpt.parent / "run_summary.json"
    summ = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return {**rep, "checkpoint_id": checkpoint_id(ckpt), "checkpoint_path": str(ckpt),
            "backbone_checkpoint_id": payload["backbone_checkpoint_id"],
            "watch_verdict": summ.get("watch_verdict", payload.get("watch_verdict")),
            "train_iters_completed": summ.get("iters_completed", payload.get("it")),
            "forward_pass_cost": {"nominal_backbone_passes": 1, "measured_multiplier": t_meth / t_base},
            "unc_definition": "Laplace scale b per band (reflectance)"}


def eval_tta(tta: int, ckpt: Path, patches) -> Dict[str, Any]:
    if not p5_committed():
        raise RuntimeError("TTA path needs src.infer.trust from the [mvp/P5] commit, which is absent.")
    from drishtisr.infer.predictor import TorchPredictor
    from drishtisr.infer.trust import compute_trust

    pred = TorchPredictor(ckpt, threads=6, interim=False)
    errs, uncs = [], []
    for p in patches:
        res = compute_trust(p["lr"], pred, tta=tta)
        errs.append(np.abs(res.sr - p["hr"]))
        uncs.append(res.unc_bands)
    rep = calibration_report(np.stack(errs), np.stack(uncs), laplace=False)
    lr0 = patches[0]["lr"]
    t_base = _median_time(lambda: compute_trust(lr0, pred, tta=0))
    t_meth = _median_time(lambda: compute_trust(lr0, pred, tta=tta))
    return {**rep, "checkpoint_id": pred.info["checkpoint_id"], "checkpoint_path": str(ckpt),
            "forward_pass_cost": {"nominal_backbone_passes": tta, "measured_multiplier": t_meth / t_base},
            "unc_definition": f"per-band std (ddof 0) across TTA-{tta} (reflectance); not a Laplace scale"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", choices=("tta4", "tta8", "learned"), required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--interim", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(6)

    patches = draw_eval_patches(args.n)
    t0 = time.perf_counter()
    if args.method == "learned":
        if not args.ckpt:
            ap.error("--ckpt (head checkpoint) is required for --method learned")
        res = eval_learned(Path(args.ckpt), patches)
    else:
        res = eval_tta(int(args.method[3:]), Path(args.ckpt) if args.ckpt else A2, patches)
    git = git_metadata(WORKTREE)
    report = {"what": f"P6 uncertainty calibration on VAL: {args.method}", "method": args.method, **res,
              "interim": bool(args.interim), "split": "val", "n_samples": len(patches),
              "patch_seed": EVAL_SEED, "excluded_watch_subset_seed": WATCH_SEED,
              "patches": [[p["sample_id"], p["origin"]] for p in patches],
              "eval_seconds": time.perf_counter() - t0, "git_sha": git["commit"], "git_dirty": git["dirty"],
              "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    path = REPORTS / f"unc_eval_{args.method}_{report['checkpoint_id']}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("method", "checkpoint_id", "ause", "ause_rel", "spearman_rho",
                                             "coverage", "forward_pass_cost")}, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
