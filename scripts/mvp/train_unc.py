"""P6: train the post-hoc Laplace scale head on the frozen A2 backbone (CPU).

TRAIN IDs come from the frozen split CSV through the project's own data stack
(the same wiring as ``src.data.adapter.SRPatchDataset``: dataset registry,
``resolve_split_assignments`` with ``loader.split_file`` pinned to the frozen
CSV, ``select_indices``, random-mode ``PatchDataset``, and
``augment.augment_pair`` seeded from (seed, epoch, index, "augment") -- LR/HR
coupled dihedral augmentation as in Day 3). Only the head is optimised.

Run:
  python scripts/mvp/train_unc.py --config configs/frozen_day4_unc.yaml --device cpu \
      --backbone-ckpt D:/SIH/DrishtiSR/runs/day3/a2/last.pt --iters 2000 --run c1
  python scripts/mvp/train_unc.py ... --measure-speed       # -> reports/mvp/unc_cpu_speed.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from src.config import HASH_FIELD, compute_config_hash, load_frozen  # noqa: E402
from src.infer.predictor import checkpoint_id, file_sha256  # noqa: E402
from src.models.edsr import model_from_checkpoint  # noqa: E402
from src.uncertainty import watch as W  # noqa: E402
from src.uncertainty.head import EDSRWithScale, total_params  # noqa: E402
from src.uncertainty.nll import laplace_nll  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.gitmeta import git_metadata  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

REPORTS = WORKTREE / "reports" / "mvp"
LOG = get_logger("drishtisr.unc.train")
# Files whose uncommitted edits would change what training computes.
IMPORT_SURFACE = ("src/models/", "src/data/", "src/losses/", "src/infer/tiled.py",
                  "src/infer/predictor.py", "src/uncertainty", "src/config.py", "src/utils/",
                  "configs/frozen_day4_unc.yaml", "configs/base.yaml", "scripts/mvp/train_unc.py",
                  "scripts/mvp/calibrate_trust_scales.py")


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ data
class TrainPairs(Dataset):
    """Random TRAIN crops + coupled dihedral augmentation (SRPatchDataset wiring)."""

    def __init__(self, cache_root: str, split_file: str, patch_lr: int, scale: int) -> None:
        from src.data.loader import PatchDataset, resolve_split_assignments, select_indices
        from src.data.registry import get_dataset_class

        tile = int(load_config()["sr"]["lr_patch_size"])
        cfg = load_config(overrides=[
            f"paths.cache_dir={Path(cache_root).as_posix()}", f"patches.lr_size={patch_lr}",
            f"sr.scale={scale}", f"sr.hr_patch_size={tile * scale}",
            f"loader.split_file={Path(split_file).as_posix()}"])
        cfg["patches"]["mode"] = "random"
        self.source = get_dataset_class("sen2naipv2")(cfg, validate=True)
        assignments, self.split_source = resolve_split_assignments(cfg, self.source, LOG)
        if not self.split_source.startswith("file:"):
            raise RuntimeError(f"split not read from the frozen CSV: {self.split_source}")
        self.indices = list(select_indices(cfg, self.source, assignments, "train", LOG))
        self.patches = PatchDataset(cfg, self.source, self.indices, mode="random",
                                    split_name="train", logger=LOG)
        self.seed = int(cfg["seed"])

    def set_epoch(self, epoch: int) -> None:
        self.patches.set_epoch(int(epoch))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        from src.data.augment import augment_pair
        from src.utils.seed import torch_generator

        item = self.patches[i]
        lr, hr = augment_pair(item["lr"], item["hr"],
                              generator=torch_generator(self.seed, self.patches.epoch, int(i), "augment"))
        return {"lr": lr, "hr": hr}


def p5_module():
    """scripts/mvp/calibrate_trust_scales.py (P5), loaded once by path."""
    name = "_p5_calibrate_trust_scales"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, WORKTREE / "scripts" / "mvp" / "calibrate_trust_scales.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def val_pool():
    """P5's VAL patch pool (ValPatches)."""
    return p5_module().ValPatches()


def load_val_patches(n: int, seed: int) -> List[Dict[str, Any]]:
    """Seeded draw from P5's VAL patch pool."""
    return val_pool().draw(n, seed=seed)


# ------------------------------------------------------------------ helpers
def dirty_surface(git: Dict[str, Any]) -> List[str]:
    return [f for f in git.get("dirty_files", []) if any(f.strip('"').startswith(s) for s in IMPORT_SURFACE)]


def compute_b0(model: EDSRWithScale, ds: TrainPairs, n: int, seed: int) -> float:
    """Mean |SR - HR| (reflectance) over ``n`` TRAIN patches drawn with ``seed`` (epoch 0)."""
    ds.set_epoch(0)
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    tot, cnt = 0.0, 0
    for s in range(0, len(idx), 16):
        items = [ds[int(i)] for i in idx[s:s + 16]]
        lr = torch.stack([it["lr"] for it in items])
        hr = torch.stack([it["hr"] for it in items])
        mu, _ = model.backbone_forward(lr)
        tot += float((mu - hr).abs().sum())
        cnt += hr.numel()
    return tot / cnt


def build_watch(model: EDSRWithScale, ucfg: Any, run: str, meta: Dict[str, Any]) -> W.CollapseWatch:
    wc = ucfg.watch
    assert (int(wc.every), int(wc.abort_at), int(wc.seed), float(wc.rho_min), float(wc.frac_floor_max),
            float(wc.spatial_cv_min), float(wc.floor)) == (W.CHECK_EVERY, W.ABORT_AT, W.SEED, W.RHO_MIN,
                                                           W.FRAC_FLOOR_MAX, W.SPATIAL_CV_MIN, W.FLOOR), \
        "watch thresholds in config differ from the pre-registered ones in src/uncertainty/watch.py"
    patches = load_val_patches(int(wc.n_patches), int(wc.seed))
    lr = torch.from_numpy(np.stack([p["lr"] for p in patches]))
    hr = torch.from_numpy(np.stack([p["hr"] for p in patches]))
    feats, mus = [], []
    for s in range(0, len(lr), 16):
        mu, f = model.backbone_forward(lr[s:s + 16])
        feats.append(f)
        mus.append(mu)
    return W.CollapseWatch(torch.cat(feats), torch.cat(mus), hr,
                           REPORTS / f"unc_watch_{run}.jsonl", run,
                           meta={**meta, "watch_patches": [[p["sample_id"], p["origin"]] for p in patches]},
                           every=int(wc.every), until=int(wc.until))


def save_head(path: Path, model: EDSRWithScale, it: int, extra: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"head": model.head.state_dict(), "it": int(it), **extra}, tmp)
    tmp.replace(path)


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backbone-ckpt", default=None, help="default: config init_backbone_checkpoint.path")
    ap.add_argument("--config", default="configs/frozen_day4_unc.yaml")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--iters", type=int, default=None, help="default: config unc.iters (4000)")
    ap.add_argument("--batch", type=int, default=None, help="default: config unc.batch_size")
    ap.add_argument("--interim", action="store_true", help="recorded only; A2 is interim=false")
    ap.add_argument("--run", default="c1")
    ap.add_argument("--out", default=None, help="default: runs/mvp/unc_<run>")
    ap.add_argument("--no-watch", action="store_true")
    ap.add_argument("--measure-speed", action="store_true",
                    help="time 5 iters + one watch check, write reports/mvp/unc_cpu_speed.json, exit")
    ap.add_argument("--allow-dirty-surface", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_frozen(args.config)
    if str(cfg.get("status")) != "frozen":
        raise RuntimeError(f"{args.config} has status {cfg.get('status')!r}; training reads frozen configs only.")
    u = cfg.unc
    torch.set_num_threads(int(u.threads))
    torch.manual_seed(int(cfg.seed))
    iters = int(args.iters if args.iters is not None else u.iters)
    batch = int(args.batch if args.batch is not None else u.batch_size)
    dev = torch.device(args.device)
    run = args.run
    out = Path(args.out) if args.out else WORKTREE / "runs" / "mvp" / f"unc_{run}"
    out.mkdir(parents=True, exist_ok=True)

    git = git_metadata(WORKTREE)
    surface = dirty_surface(git)
    if surface and not args.allow_dirty_surface:
        raise RuntimeError(f"uncommitted edits in the training import surface: {surface}")

    bb_path = Path(args.backbone_ckpt or cfg.init_backbone_checkpoint.path)
    bb_sha = file_sha256(bb_path)
    bb_id = checkpoint_id(bb_path, bb_sha)
    if bb_id != str(cfg.init_backbone_checkpoint.checkpoint_id):
        raise RuntimeError(f"backbone {bb_id} != config {cfg.init_backbone_checkpoint.checkpoint_id}")
    backbone, _ = model_from_checkpoint(bb_path, device="cpu")
    model = EDSRWithScale(backbone, hidden=int(u.head_hidden), b_floor=float(u.b_floor))
    n_total = total_params(model)
    assert n_total < int(u.param_budget), n_total
    assert sum(p.numel() for p in model.backbone.parameters()) == int(cfg.model.expected_parameters)
    assert all(not p.requires_grad for p in model.backbone.parameters())

    ds = TrainPairs(u.train_data.cache_root, u.train_data.split_file, int(cfg.data.patch_lr), int(cfg.data.scale))
    t0 = time.perf_counter()
    b0 = compute_b0(model, ds, int(u.init.b0_n_train_patches), int(u.init.b0_seed))
    model.head.set_b0(b0)
    LOG.info("b0 = %.6f reflectance (%d TRAIN patches, %.1f s)", b0, int(u.init.b0_n_train_patches),
             time.perf_counter() - t0)
    model.to(dev).train()
    mu_probe_lr = ds[0]["lr"][None].to(dev)
    mu_probe0, _ = model.backbone_forward(mu_probe_lr)

    opt = torch.optim.Adam(model.head.parameters(), lr=float(u.optim.lr), betas=tuple(u.optim.betas),
                           weight_decay=float(u.optim.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters), eta_min=float(u.optim.min_lr))
    loader_gen = torch.Generator().manual_seed(int(cfg.seed))

    config_hash = str(cfg[HASH_FIELD])
    assert config_hash == compute_config_hash(cfg)
    meta = {
        "run": run, "config": str(args.config), "config_hash": config_hash, "frozen_config": str(args.config),
        "git": git, "git_sha": git["commit"], "git_dirty": git["dirty"], "dirty_files": git["dirty_files"],
        "dirty_files_in_import_surface": surface,
        "backbone_checkpoint_id": bb_id, "backbone_path": str(bb_path), "backbone_sha256": bb_sha,
        "interim": bool(args.interim), "b0": b0, "params_total": n_total,
        "params_head": int(sum(p.numel() for p in model.head.parameters())),
        "iters_requested": iters, "batch": batch, "device": args.device,
        "n_train_samples": len(ds.indices), "n_train_patches_per_epoch": len(ds),
        "split_source": ds.split_source, "seed": int(cfg.seed), "args": vars(args),
        "started_utc": now(), "torch": torch.__version__, "python": sys.version.split()[0],
        "threads": torch.get_num_threads(),
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ckpt_extra = {"backbone_checkpoint_id": bb_id, "backbone_path": str(bb_path), "config_hash": config_hash,
                  "b0": b0, "b_floor": float(u.b_floor), "head_hidden": int(u.head_hidden),
                  "git_sha": git["commit"], "run": run, "interim": bool(args.interim)}

    watch = None if (args.no_watch or (iters < W.CHECK_EVERY and not args.measure_speed)) else \
        build_watch(model, u, run if not args.measure_speed else f"{run}_speedtest", {"config_hash": config_hash,
                                                                                        "backbone_checkpoint_id": bb_id})

    def batches():
        epoch = 0
        while True:
            ds.set_epoch(epoch)
            yield from DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True,
                                  generator=loader_gen)
            epoch += 1

    stream = batches()
    losses: List[float] = []
    it_times: List[float] = []
    log_fh = (out / "log.jsonl").open("a", encoding="utf-8")
    status, it = "completed", 0
    n_run = 6 if args.measure_speed else iters
    t_train = time.perf_counter()
    for it in range(1, n_run + 1):
        ts = time.perf_counter()
        bt = next(stream)
        lr_b, hr_b = bt["lr"].to(dev), bt["hr"].to(dev)
        mu, b = model(lr_b)
        loss = laplace_nll(mu, b, hr_b, beta=float(u.nll.beta))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if any(p.grad is not None for p in model.backbone.parameters()):
            raise RuntimeError("a backbone parameter received a gradient")
        lv = float(loss.detach())
        if not math.isfinite(lv):
            status = "nonfinite_loss"
            break
        losses.append(lv)
        it_times.append(time.perf_counter() - ts)
        if it <= 5 or it % 10 == 0 or it == n_run:
            log_fh.write(json.dumps({"iter": it, "loss": lv, "lr": sched.get_last_lr()[0],
                                     "b_mean": float(b.detach().mean()), "sec": it_times[-1]}) + "\n")
            log_fh.flush()
        if it % 50 == 0:
            print(f"[unc] it {it}/{n_run} loss {lv:.4f} b_mean {float(b.detach().mean()):.5f} "
                  f"{np.mean(it_times[-50:]):.2f}s/it", flush=True)
        if watch is not None and watch.due(it):
            rec, abort = watch.check(model.head, it)
            print(f"[unc] watch it {it}: rho {rec['rho']:.3f} floor {rec['frac_floor']:.3f} "
                  f"cv {rec['spatial_cv']:.3f} verdict {rec['verdict']}", flush=True)
            if abort:
                status = "aborted_collapsed"
                break
        if it % 500 == 0:
            save_head(out / "head_last.pt", model, it, {**ckpt_extra, "watch_verdict": watch.verdict if watch else None})
    log_fh.close()
    elapsed = time.perf_counter() - t_train

    if args.measure_speed:
        timed = it_times[1:6]  # first iter = warm-up
        s_it = float(np.mean(timed))
        rec, _ = watch.check(model.head, W.CHECK_EVERY)
        check_s = float(rec["check_seconds"])
        n_checks = W.ABORT_AT // W.CHECK_EVERY
        proj_min = (2000 * s_it + n_checks * check_s) / 60.0
        (REPORTS / f"unc_watch_{run}_speedtest.jsonl").unlink(missing_ok=True)
        load = p5_module().cpu_load_percent()
        speed = {"what": "P6 head-training CPU speed at the config batch size (5 timed iters after 1 warm-up)",
                 "sec_per_iter": s_it, "iter_seconds": timed, "batch": batch, "threads": torch.get_num_threads(),
                 "watch_check_seconds": check_s, "n_watch_checks": n_checks,
                 "projected_minutes_2000_iters": proj_min, "budget_minutes": 75.0,
                 "needs_kaggle": bool(proj_min > 75.0), "cpu_load_percent_after": load,
                 "provisional": True, "checkpoint_id": bb_id, "git_sha": git["commit"],
                 "config_hash": config_hash, "split": "train", "n_samples": batch * 5, "timestamp_utc": now()}
        (REPORTS / "unc_cpu_speed.json").write_text(json.dumps(speed, indent=2), encoding="utf-8")
        print(json.dumps(speed, indent=2))
        return 0

    mu_probe1, _ = model.backbone_forward(mu_probe_lr)
    mean_unchanged = bool(torch.equal(mu_probe0, mu_probe1))
    verdict = watch.verdict if watch else "not_evaluated"
    save_head(out / "head_last.pt", model, it, {**ckpt_extra, "watch_verdict": verdict})
    head_id = checkpoint_id(out / "head_last.pt")
    summary = {
        "what": "P6 learned Laplace scale head training run", "run": run, "status": status,
        "iters_requested": iters, "iters_completed": len(losses), "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None, "all_losses_finite": status != "nonfinite_loss",
        "sec_per_iter_mean": float(np.mean(it_times)) if it_times else None, "elapsed_min": elapsed / 60.0,
        "watch_verdict": verdict, "watch_history": [{k: r[k] for k in ("iter", "rho", "frac_floor", "spatial_cv")}
                                                    for r in (watch.history if watch else [])],
        "recommendation": "tta_fallback" if verdict == "collapsed" else None,
        "mean_path_bit_identical": mean_unchanged, "b0": b0,
        "checkpoint_id": head_id, "checkpoint_path": str(out / "head_last.pt"),
        "backbone_checkpoint_id": bb_id, "config_hash": config_hash, "git_sha": git["commit"],
        "git_dirty": git["dirty"], "interim": bool(args.interim), "split": "train",
        "n_samples": len(losses) * batch, "timestamp_utc": now(),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORTS / f"unc_train_{run}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "watch_history"}, indent=2))
    return 0 if status == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
