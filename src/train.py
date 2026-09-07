"""DrishtiSR training loop — Run A (L1 only).

Resume-safe (Kaggle sessions die), iteration-based (not epoch-based),
CSV-logged, AMP on by default. Place at: src/drishtisr/train.py

Dataset contract (whatever Day 1 built must satisfy this):
    ds = DatasetClass(root=<data_root>, split="train"|"val", patch_lr=<int>, scale=<int>)
    ds[i] -> {"lr": FloatTensor[C,h,w], "hr": FloatTensor[C,h*scale,w*scale]}, values ~[0,1]
Pass the module/class by name so this file never hardcodes Day 1 naming.

Example:
    python -m drishtisr.train --data-module drishtisr.data.sen2naip \
        --data-class SEN2NAIPPatches --data-root /kaggle/input/sen2naip-cache \
        --iters 40000 --batch 16 --out /kaggle/working/runs/runA
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from drishtisr.models.edsr import build_model, count_params
except ImportError:  # allows running the file directly during smoke tests
    from edsr import build_model, count_params  # type: ignore


# ---------------------------------------------------------------- utils
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(sr.clamp(0, 1), hr.clamp(0, 1)).item()
    return 100.0 if mse == 0 else 10.0 * math.log10(max_val ** 2 / mse)


def lr_at(it: int, args) -> float:
    if it < args.warmup:
        return args.lr * (it + 1) / max(1, args.warmup)
    prog = (it - args.warmup) / max(1, args.iters - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))


def load_dataset_cls(module: str, cls: str):
    return getattr(importlib.import_module(module), cls)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-module", required=True)
    p.add_argument("--data-class", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="runs/runA")
    p.add_argument("--iters", type=int, default=40000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patch-lr", type=int, default=64)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--in-ch", type=int, default=4)
    p.add_argument("--n-resblocks", type=int, default=16)
    p.add_argument("--n-feats", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--val-batches", type=int, default=25)
    p.add_argument("--ckpt-every", type=int, default=2500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--resume", default="auto", help="'auto' | path | 'none'")
    p.add_argument("--max-hours", type=float, default=11.0, help="stop cleanly before session kill")
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    DS = load_dataset_cls(args.data_module, args.data_class)
    train_ds = DS(root=args.data_root, split="train", patch_lr=args.patch_lr, scale=args.scale)
    val_ds = DS(root=args.data_root, split="val", patch_lr=args.patch_lr, scale=args.scale)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = build_model("edsr_baseline", scale=args.scale, n_resblocks=args.n_resblocks,
                        n_feats=args.n_feats, in_ch=args.in_ch, out_ch=args.in_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and dev == "cuda")

    start_it, best = 0, -1.0
    ck_last = out / "last.pt"
    resume_path = ck_last if (args.resume == "auto" and ck_last.exists()) else (
        Path(args.resume) if args.resume not in ("auto", "none") else None)
    if resume_path and resume_path.exists():
        ck = torch.load(resume_path, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); start_it = ck["it"] + 1; best = ck.get("best", -1.0)
        print(f"[resume] {resume_path} @ it={start_it} best_psnr={best:.3f}", flush=True)

    log_path = out / "log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["iter", "loss", "lr", "val_psnr", "sec_per_100it"])
    print(f"[init] dev={dev} params={count_params(model):,} train={len(train_ds)} val={len(val_ds)}", flush=True)

    def save(path: Path, it: int) -> None:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "it": it, "best": best, "args": vars(args)}, path)

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        tot, n = 0.0, 0
        for i, b in enumerate(val_dl):
            if i >= args.val_batches:
                break
            lr_, hr = b["lr"].to(dev, non_blocking=True), b["hr"].to(dev, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp) and dev == "cuda"):
                sr = model(lr_)
            tot += psnr(sr.float(), hr); n += 1
        model.train()
        return tot / max(1, n)

    it_stream = infinite(train_dl)
    model.train()
    t0, tw, run_loss = time.time(), time.time(), 0.0
    for it in range(start_it, args.iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, args)
        b = next(it_stream)
        lr_, hr = b["lr"].to(dev, non_blocking=True), b["hr"].to(dev, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(args.amp) and dev == "cuda"):
            loss = F.l1_loss(model(lr_), hr)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt); scaler.update()
        run_loss += loss.item()

        if (it + 1) % args.log_every == 0:
            sec = time.time() - tw; tw = time.time()
            avg = run_loss / args.log_every; run_loss = 0.0
            print(f"it {it+1}/{args.iters} loss {avg:.5f} lr {lr_at(it,args):.2e} {sec:.1f}s/100it", flush=True)
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([it + 1, f"{avg:.6f}", f"{lr_at(it,args):.3e}", "", f"{sec:.1f}"])

        if (it + 1) % args.val_every == 0 or (it + 1) == args.iters:
            v = validate()
            print(f"[val] it {it+1} psnr {v:.3f} dB (best {max(best, v):.3f})", flush=True)
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([it + 1, "", "", f"{v:.4f}", ""])
            if v > best:
                best = v; save(out / "best.pt", it)

        if (it + 1) % args.ckpt_every == 0 or (it + 1) == args.iters:
            save(ck_last, it)
        if (time.time() - t0) / 3600.0 > args.max_hours:
            save(ck_last, it); print(f"[stop] time budget hit at it={it+1}", flush=True); break

    save(ck_last, args.iters - 1)
    print(f"[done] iters={args.iters} best_val_psnr={best:.3f} elapsed={(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
