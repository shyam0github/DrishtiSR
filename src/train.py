"""DrishtiSR training loop: L1 reconstruction, optionally + spectral consistency.

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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from drishtisr.models.edsr import build_model, count_params, load_backbone_state
except ImportError:  # allows running the file directly during smoke tests
    from edsr import build_model, count_params, load_backbone_state  # type: ignore

from src.config import FROZEN_CONFIG, HASH_FIELD, FrozenConfigError, load_frozen
from src.losses import (
    DEFAULT_DOWNSAMPLE,
    gaussian_nll,
    nll_weight_at,
    spectral_consistency,
    spectral_terms,
)
from src.metrics.logvar import LogvarMonitor
from src.metrics.image_quality import ergas, lpips, rgb_band_indices, ssim
from src.metrics.image_quality import sam as sam_hr
from src.metrics.sharpness import (
    DEFAULT_HF_CUTOFF,
    reference_sharpness,
    sharpness_terms,
)
from src.utils.gitmeta import git_metadata
from src.utils.seed import derive_seed


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


def epoch_stream(
    loader,
    dataset,
    start_epoch: int = 0,
    shuffle_generator=None,
    shuffle_seed=None,
):
    """Yield ``(epoch, batch)`` forever, calling ``set_epoch`` on each new epoch.

    THIS REPLACES A PLAIN ``while True: for batch in loader`` -- which is what
    Run A used, and it is why Run A overfit. That loop never advanced an epoch
    counter, so ``PatchDataset``'s crop RNG stayed seeded at epoch 0 and all
    40k iterations were trained on ONE fixed set of 2400 crops. The loss curve
    looked healthy; the model was memorising.

    Two things are load-bearing here:

    1. ``set_epoch`` is called BEFORE the epoch's iterator is created. Creating
       the iterator is what forks/spawns the workers, and workers receive a
       PICKLED COPY of the dataset. Setting the epoch afterwards would update
       the parent's copy only, and the crops would silently stay frozen -- the
       original bug wearing a fix.
    2. The caller must build the loader with ``persistent_workers=False``.
       Persistent workers are created once and keep their first copy of the
       dataset for the life of the loader, so they would never see any epoch
       but the first. See the DataLoader construction in ``main``.
    3. If a shuffle generator is supplied it is RESEEDED here, per epoch, from
       ``derive_seed(shuffle_seed, epoch, 0, "shuffle")`` -- also before the
       iterator is created, because ``RandomSampler`` draws its whole
       permutation when ``__iter__`` is called. Without it, batch COMPOSITION
       is a function of the global torch RNG, so it depends on how many draws
       the model init and the objective happened to consume. A2 and B1 match
       today because neither loss touches the RNG; seeding explicitly is what
       stops the guarantee from depending on that staying true. See
       ``tests/test_epoch_determinism.py``.

    ``DistributedSampler``, if one is ever introduced, needs the same call on
    the sampler; it is applied here when the loader has one, so adding
    distribution does not silently reintroduce a fixed shuffle.

    Args:
        loader: A DataLoader over ``dataset``.
        dataset: The dataset itself, exposing ``set_epoch(int)``.
        start_epoch: Epoch to resume at, derived by the caller from
            ``start_it // len(loader)`` so a resumed run continues the crop
            sequence instead of restarting it.
        shuffle_generator: The ``torch.Generator`` handed to the DataLoader, or
            None to leave batch order on the global RNG (the pre-Day-3
            behaviour, kept only so this helper stays usable without one).
        shuffle_seed: The DATA seed the shuffle stream is derived from --
            ``cfg.seed``, i.e. the dataset's ``seed`` attribute (42), NOT the
            run's init seed. Deliberate: batch order is part of the data
            stream, and the control and the spectral run must see the same one
            even if their init seeds ever diverge.

    Yields:
        ``(epoch, batch)`` pairs, indefinitely.

    Raises:
        AttributeError: ``dataset`` has no ``set_epoch``. Deliberate -- a
            dataset that cannot be advanced would train on fixed crops, which
            is the exact defect this function exists to remove, and a silent
            ``getattr(..., None)`` here would hide it.
        ValueError: A generator was passed with no seed to derive from.
            Refused rather than defaulted: a generator seeded from something
            unstated looks like determinism while providing none.
    """
    if shuffle_generator is not None and shuffle_seed is None:
        raise ValueError(
            "epoch_stream() was given a shuffle_generator but no shuffle_seed "
            "to derive its per-epoch seeds from. Pass the DATA seed "
            "(dataset.seed / cfg.seed), not the run's init seed."
        )

    epoch = int(start_epoch)
    while True:
        dataset.set_epoch(epoch)
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if shuffle_generator is not None:
            shuffle_generator.manual_seed(
                derive_seed(int(shuffle_seed), epoch, 0, "shuffle")
            )
        for batch in loader:
            yield epoch, batch
        epoch += 1


def uncertainty_record(args) -> dict:
    """The uncertainty settings of a run, as recorded in metadata and checkpoints.

    Args:
        args: Parsed CLI arguments.

    Returns:
        ``{"enabled": bool, ...}``; the head and NLL settings are included only
        when enabled, so a Day-3-style run records ``{"enabled": False}`` and
        nothing that could be misread as having been used.
    """
    if not int(args.uncertainty):
        return {"enabled": False}
    return {
        "enabled": True,
        "var_feats": int(args.var_feats),
        "logvar_min": float(args.logvar_min),
        "logvar_max": float(args.logvar_max),
        "logvar_init": float(args.logvar_init),
        "nll_weight": float(args.nll_weight),
        "nll_warmup": int(args.nll_warmup),
        "nll_ramp": int(args.nll_ramp),
        "init_from": str(args.init_from),
    }


def write_run_metadata(out: Path, args) -> dict:
    """Write ``run_metadata.json``: what config and what CODE this run used.

    Run A2 (control) and Run B (fix) are only a valid comparison if both the
    hyperparameters AND the code matched. ``config_hash`` covers the first;
    the git commit covers the second, and Day 3 is the reason both are needed --
    the frozen-crop bug changed the data stream without touching a single
    config value.

    Args:
        out: The run directory. Created by the caller.
        args: Parsed CLI arguments, recorded verbatim.

    Returns:
        The metadata dict, so the caller can log it without re-reading the file.
        ``config_hash`` is the frozen config's hash, or ``"unavailable"`` when
        the frozen config could not be read -- recorded as that string rather
        than omitted, so a run with no provenance cannot be mistaken for one
        that matched.
    """
    try:
        frozen = load_frozen()
        config_hash = str(frozen[HASH_FIELD])
        frozen_path = FROZEN_CONFIG
    except (FileNotFoundError, FrozenConfigError) as exc:
        # NOT silent, and NOT fatal: a Kaggle kernel unpacked without configs/
        # should still train, but the log and the metadata must both say the
        # run cannot be tied to a frozen config.
        print(f"[provenance] WARNING: no verified frozen config ({exc})", flush=True)
        config_hash, frozen_path = "unavailable", None

    git = git_metadata()
    meta = {
        "config_hash": config_hash,
        "frozen_config": frozen_path,
        "git": git,
        # PROMOTED OUT OF `args`, and this is the whole reason this block was
        # touched. configs/frozen_day3.yaml carries spectral_lambda1/2 = 0.0 --
        # it describes the CONTROL -- so for B1 and B2 the config hash is
        # IDENTICAL to A2's and says nothing about what was optimised. The
        # lambdas are a per-run parameter, and this JSON is their only record.
        # Buried inside `args` they are recoverable but not findable; a reader
        # comparing three runs must be able to see the one difference between
        # them without parsing an argparse dump.
        "lambdas": {
            "spectral_lambda1": float(args.spectral_lambda1),
            "spectral_lambda2": float(args.spectral_lambda2),
        },
        "spectral_downsample": str(args.spectral_downsample),
        "spectral_on": bool(
            args.spectral_lambda1 > 0.0 or args.spectral_lambda2 > 0.0
        ),
        # Promoted for the same reason as the lambdas: frozen_day3.yaml says
        # nothing about the head, so this block is the record of whether a run
        # had one and how its NLL was scheduled.
        "uncertainty": uncertainty_record(args),
        # Flattened next to the config hash because the cross-run assertion in
        # scripts/day3_runs.py reads exactly this pair, and reaching two levels
        # into a nested dict for the value a guard depends on is how a guard
        # ends up comparing None to None.
        "git_sha": git["commit"],
        "git_dirty": git["dirty"],
        "seed": int(args.seed),
        "args": vars(args),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


# The CSV schema, GROWN ONLY BY APPENDING. Each generation is kept as its own
# constant so an older log can be widened in place instead of rejected: a
# resumed Kaggle run finds the log its previous session wrote, and discovering
# a schema mismatch in the figure script hours later is how GPU time gets spent
# twice.
#
#   LEGACY_COLUMNS    -- Run A.
#   SPECTRAL_COLUMNS  -- Day 3, first pass: the spectral scalars.
#   REFERENCE_COLUMNS -- Day 3 (A2/B1/B2): the blur diagnostic and the
#                        reference metrics.
#   LOG_COLUMNS       -- current: the NLL and the sigma-collapse monitor.
#                        Blank on every run without the head.
#
# NOTHING BEFORE AN APPENDED COLUMN EVER MOVES, so a reader that slices the
# first five columns still works against all three.
#
# NAMING TRAP, stated once here because it is the kind of thing that silently
# ruins a comparison: `val_sam` is the SPECTRAL-domain angle -- SAM between the
# LR and the downsampled SR, in RADIANS -- and it has meant that since Day 3's
# first pass. The image-domain angle against the ground-truth HR, in DEGREES,
# which is what src/metrics reports and what the results table quotes, is the
# newly appended `val_sam_hr`. They are different quantities in different units
# and neither is a substitute for the other.
LEGACY_COLUMNS = ["iter", "loss", "lr", "val_psnr", "sec_per_100it"]
SPECTRAL_COLUMNS = LEGACY_COLUMNS + [
    "val_l1_spec",
    "val_sam",
    "val_sam_valid_frac",
]
REFERENCE_COLUMNS = SPECTRAL_COLUMNS + [
    # The blur diagnostic. Logged for EVERY run including the control, because
    # "the spectral run got blurrier" is only a statement if the control's
    # sharpness over the same val patches is on the same plot. See
    # src/metrics/sharpness.py.
    "val_sharpness",
    "val_hf_energy",
    # The reference metric suite, measured on the first --metric-batches of the
    # validation batches rather than all of them; see validate().
    "val_ssim",
    "val_lpips",
    "val_sam_hr",
    "val_ergas",
]
LOG_COLUMNS = REFERENCE_COLUMNS + [
    # The heteroscedastic head (--uncertainty 1). val_nll is the UNWEIGHTED
    # Gaussian NLL over the val batches; nll_weight is the schedule's value at
    # this iteration, so the ramp is readable off the log. The six logvar
    # columns are src.metrics.logvar.LogvarMonitor -- spatial_std collapsing
    # toward 0 is the Day 4 kill criterion.
    "val_nll",
    "nll_weight",
    "val_logvar_mean",
    "val_logvar_min",
    "val_logvar_max",
    "val_logvar_spatial_std",
    "val_logvar_clamp_lo_frac",
    "val_logvar_clamp_hi_frac",
]

# Every accepted header, oldest first, so ensure_log_header can pad forward.
KNOWN_SCHEMAS = (LEGACY_COLUMNS, SPECTRAL_COLUMNS, REFERENCE_COLUMNS, LOG_COLUMNS)


def ensure_log_header(log_path: Path) -> None:
    """Create ``log.csv``, or widen a legacy one, before a row is appended.

    A resumed Kaggle run finds a ``log.csv`` written by an earlier session. If
    that file has a narrower header, appending current rows to it produces a
    CSV that ``pandas`` refuses to read -- and the failure would surface hours
    later, in the figure script, after the GPU time was spent. So the widening
    happens here, at startup, in place, with the old rows padded. Any header in
    :data:`KNOWN_SCHEMAS` is widened; anything else raises.

    Not silent, and not a fallback: the migration prints what it did, and any
    header that is neither the legacy nor the current one raises rather than
    being overwritten, because an unrecognised header means this directory
    belongs to something other than this trainer and its contents are about to
    be interleaved with a different run's.

    Args:
        log_path: ``<out>/log.csv``. Its parent must exist.

    Raises:
        ValueError: The file exists with a header this trainer did not write.
    """
    if not log_path.exists():
        with log_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(LOG_COLUMNS)
        return

    with log_path.open("r", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        with log_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(LOG_COLUMNS)
        return

    header = rows[0]
    if header == LOG_COLUMNS:
        return
    if header not in [list(schema) for schema in KNOWN_SCHEMAS]:
        raise ValueError(
            f"{log_path} has header {header}, which src/train.py did not write "
            f"(expected one of {[list(s) for s in KNOWN_SCHEMAS]}). Refusing to "
            "append: point --out at a fresh directory rather than interleaving "
            "two runs in one file."
        )

    pad = [""] * (len(LOG_COLUMNS) - len(header))
    with log_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LOG_COLUMNS)
        for row in rows[1:]:
            writer.writerow(row + pad)
    print(
        f"[log] widened {log_path.name} from {len(header)} columns to "
        f"{len(LOG_COLUMNS)}; {len(rows) - 1} existing rows padded with empty "
        "cells. The padded columns were never measured in those rows -- they "
        "are blank, not zero.",
        flush=True,
    )


def dataset_config(dataset):
    """The data config a dataset was built from, or a loud failure.

    ``validate()`` needs two values that AGENTS.md forbids hardcoding: the
    reflectance ``data_range`` PSNR and SSIM are measured against, and which
    channel indices are R, G and B for LPIPS. Both live in
    ``configs/base.yaml``, and ``SRPatchDataset`` already resolved that file --
    so the authority is the dataset object, not a second load with possibly
    different overrides.

    Args:
        dataset: The training dataset, expected to expose ``.cfg``.

    Returns:
        The dataset's resolved config mapping.

    Raises:
        AttributeError: The dataset does not carry its config. Raised rather
            than falling back to a freshly loaded ``base.yaml``: a fallback
            would measure against defaults while the run trained against
            overrides, and the two would differ without anything saying so.
    """
    cfg = getattr(dataset, "cfg", None)
    if cfg is None:
        raise AttributeError(
            f"{type(dataset).__name__} does not expose a .cfg, so the "
            "reflectance data_range and the RGB band indices cannot be read "
            "from the config this run was actually built with. Metrics must "
            "not be measured against assumed values -- see AGENTS.md section 2."
        )
    return cfg


# ---------------------------------------------------------------- args
def build_parser() -> argparse.ArgumentParser:
    """The CLI, built separately so its DEFAULTS can be inspected without running.

    Split out of ``main`` so ``tests/test_frozen_config.py`` can assert that
    every default here agrees with ``configs/frozen_day3.yaml``. A default that
    has drifted from the freeze is worse than no default at all: a run launched
    without the flag trains something the recorded config hash does not
    describe, and nothing anywhere would say so.
    """
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
    # 48, not 64. 16/64 is 1,518,724 params -- over cfg.runtime.max_parameters
    # (1,000,000) and the source of the budget-violation warning in the report.
    # 16/48 is 855,652. See src/models/edsr.py.
    p.add_argument("--n-feats", type=int, default=48)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=500)
    # 0, not 1e-4. Run A overfit on ~2400 patches and weight decay was not what
    # was holding it back -- the fix is augmentation and a smaller model, both
    # of which are now defaults here. Decoupled AdamW decay on an EDSR with no
    # normalisation layers mostly shrinks the residual branches, which costs
    # high-frequency detail: exactly the thing being super-resolved.
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--val-batches", type=int, default=25)
    p.add_argument("--ckpt-every", type=int, default=2500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--resume", default="auto", help="'auto' | path | 'none'")

    # -- spectral consistency (technical contribution 1) --------------------
    #
    # BOTH DEFAULT TO 0.0, and that is the point. The control run and the
    # spectral run must differ by exactly these two numbers and nothing else,
    # so the control is this same file with the flags left off -- not a second
    # code path, not an earlier commit. A non-zero default would let a control
    # run acquire a spectral term by omission, and the comparison would be
    # silently between two different objectives.
    #
    # The terms are LOGGED at every validation regardless of the lambdas (see
    # validate() in main), because a control that does not log the quantity it
    # is not optimising cannot demonstrate that the spectral run drove it down.
    #
    # Mirrored by cfg.loss.spectral in configs/base.yaml, which carries the
    # candidate values (0.5 / 0.1) and the reasoning, and by the loss block of
    # configs/frozen_day3.yaml, which records what the frozen run uses.
    # tests/test_frozen_config.py asserts these agree.
    p.add_argument("--spectral-lambda1", type=float, default=0.0,
                   help="weight on L1(lr, down4(sr)), reflectance units; 0.0 = off")
    p.add_argument("--spectral-lambda2", type=float, default=0.0,
                   help="weight on SAM(lr, down4(sr)), radians; 0.0 = off")
    # The degradation operator. MUST stay antialiased -- src/losses/spectral.py
    # rejects decimating modes by name, because with stride-4 slicing the loss
    # constrains one HR pixel in 16 and the consistency claim is vacuous.
    p.add_argument("--spectral-downsample", default=DEFAULT_DOWNSAMPLE,
                   help="antialiased downsampler for the spectral term")
    # 8.5, not 11.0. Kaggle kills a GPU session at 9 hours; 11 meant the guard
    # could never fire and the loop would be killed mid-checkpoint-write instead
    # of exiting cleanly. Matches runtime.max_hours in configs/frozen_day3.yaml.
    p.add_argument("--max-hours", type=float, default=8.5, help="stop cleanly before session kill")

    # -- instrumentation, NOT hyperparameters -------------------------------
    #
    # Neither of these appears in configs/frozen_day3.yaml, and that is on
    # purpose rather than an oversight. The freeze describes what is TRAINED;
    # these describe what is MEASURED. Adding them to the frozen file would
    # change its hash -- e6082c69... -- and invalidate the provenance of every
    # comparison that already cites it, to record two numbers that cannot
    # affect a single weight.
    #
    # They must still be IDENTICAL across the runs being compared, or the
    # curves are not the same quantity. That is enforced structurally rather
    # than by the freeze: scripts/day3_runs.py builds all three runs' argument
    # lists from one shared list and asserts that the lambdas are the only
    # difference.
    p.add_argument(
        "--metric-batches", type=int, default=4,
        help="validation batches scored with the full SSIM/LPIPS/SAM/ERGAS "
             "suite (a SUBSET of --val-batches; the suite runs on CPU in "
             "float64 and LPIPS alone costs ~9 s/batch there, which over 25 "
             "batches x 24 validations x 3 runs would eat 4.5 GPU-hours of a "
             "9-hour session). PSNR, the spectral terms and the blur "
             "diagnostic still run over ALL --val-batches.",
    )
    p.add_argument(
        "--hf-cutoff", type=float, default=DEFAULT_HF_CUTOFF,
        help="fraction of Nyquist above which power counts as high-frequency "
             "in the blur diagnostic. 0.25 is the x4 band edge -- the band the "
             "LR does not contain and the model must invent.",
    )

    # -- heteroscedastic uncertainty head (technical contribution 2) --------
    #
    # OFF BY DEFAULT, for the same reason the spectral lambdas are 0.0: every
    # Day 3 invocation must keep training exactly the model it trained, and
    # tests/test_frozen_config.py pins the Day 3 defaults. Mirrored by
    # cfg.uncertainty in configs/base.yaml; tests/test_uncertainty_head.py
    # asserts the two agree. NOT in configs/frozen_day3.yaml -- Run C gets its
    # own freeze. See src/models/edsr.py (head) and src/losses/nll.py (NLL).
    p.add_argument("--uncertainty", type=int, default=0,
                   help="1 = add the log-variance head and the NLL objective")
    p.add_argument("--var-feats", type=int, default=16)
    p.add_argument("--logvar-min", type=float, default=-10.0)
    p.add_argument("--logvar-max", type=float, default=10.0)
    p.add_argument("--logvar-init", type=float, default=-9.0)
    p.add_argument("--nll-weight", type=float, default=1.0,
                   help="final NLL weight, added to the L1 term (which is kept)")
    p.add_argument("--nll-warmup", type=int, default=2000,
                   help="iterations of pure L1 with the variance branch detached")
    p.add_argument("--nll-ramp", type=int, default=2000,
                   help="iterations over which the NLL weight ramps 0 -> --nll-weight")
    # The strict=False path: start from a checkpoint trained WITHOUT the head
    # (A2/B1/B2). Only var_head.* may be missing; see load_backbone_state.
    # Ignored when --resume finds a checkpoint -- a resumed run continues its
    # own weights, never re-initialises from the parent.
    p.add_argument("--init-from", default="none",
                   help="checkpoint whose weights initialise the backbone; 'none' = scratch")
    return p


# ---------------------------------------------------------------- main
def main() -> None:
    args = build_parser().parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    meta = write_run_metadata(out, args)
    print(f"[provenance] config_hash={meta['config_hash']} "
          f"commit={meta['git']['commit_short']} dirty={meta['git']['dirty']}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    DS = load_dataset_cls(args.data_module, args.data_class)
    train_ds = DS(root=args.data_root, split="train", patch_lr=args.patch_lr, scale=args.scale)
    val_ds = DS(root=args.data_root, split="val", patch_lr=args.patch_lr, scale=args.scale)
    # persistent_workers=False, deliberately. Persistent workers keep the copy
    # of the dataset they were forked with, so set_epoch() in the parent would
    # never reach them and crops would stay frozen at epoch 0 -- the Run A bug,
    # reintroduced through the back door. Fresh workers per epoch re-pickle the
    # dataset and pick up the new epoch. MEASURED cost: 2400 patches / batch 16
    # = 150 iterations per epoch, so ~267 respawns over 40k iterations, which is
    # minutes against an 8.5-hour budget and buys the entire fix.
    #
    # THE SHUFFLE GENERATOR. Without one, `shuffle=True` builds a RandomSampler
    # that draws from the GLOBAL torch RNG, so which patches land in a batch
    # depends on how many random numbers everything before it consumed --
    # weight init, and any objective that samples. A2 and B1 happen to match
    # today because neither L1 nor the spectral term touches the RNG, but that
    # is an accident of the current losses, not a property of the experiment.
    # The heteroscedastic-NLL head is the next contribution and it will not be
    # so polite.
    #
    # Seeded per epoch inside epoch_stream() from
    # derive_seed(train_ds.seed, epoch, 0, "shuffle") -- the DATA seed (42),
    # deliberately not args.seed (1337), because batch order belongs to the
    # data stream. Same stream naming as crops and augmentation, so the three
    # decisions are independent and one cannot shift another.
    shuffle_seed = int(train_ds.seed)
    shuffle_generator = torch.Generator()
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=False,
                          generator=shuffle_generator)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    uncertainty_on = bool(int(args.uncertainty))
    head_kwargs = (
        dict(uncertainty=True, var_feats=args.var_feats, logvar_min=args.logvar_min,
             logvar_max=args.logvar_max, logvar_init=args.logvar_init)
        if uncertainty_on else {}
    )
    model = build_model("edsr_baseline", scale=args.scale, n_resblocks=args.n_resblocks,
                        n_feats=args.n_feats, in_ch=args.in_ch, out_ch=args.in_ch,
                        **head_kwargs).to(dev)
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
    elif args.init_from not in ("none", ""):
        parent = torch.load(args.init_from, map_location=dev, weights_only=False)
        fresh = load_backbone_state(model, parent["model"])
        print(f"[init-from] backbone from {args.init_from} (it={parent.get('it')}, "
              f"lambdas={parent.get('lambdas')}); {len(fresh)} head tensors left at "
              f"their initialisation: {fresh}", flush=True)

    log_path = out / "log.csv"
    ensure_log_header(log_path)
    print(f"[init] dev={dev} params={count_params(model):,} train={len(train_ds)} val={len(val_ds)}", flush=True)

    # One boolean, decided once, so the training step has no per-iteration
    # branch on a float comparison and the log says out loud which run this is.
    spectral_on = args.spectral_lambda1 > 0.0 or args.spectral_lambda2 > 0.0
    print(
        f"[loss] L1 reconstruction"
        + (
            f" + spectral consistency (lambda1={args.spectral_lambda1}, "
            f"lambda2={args.spectral_lambda2}, D={args.spectral_downsample})"
            if spectral_on
            else " ONLY -- spectral lambdas are 0.0, this is the CONTROL. "
            "l1_spec and sam are still measured and logged at every validation."
        ),
        flush=True,
    )

    # Read from the config the DATASET was built with, never assumed. Both are
    # measurement settings that must be identical across A2, B1 and B2, and
    # both come from configs/base.yaml through the one object that resolved it.
    data_cfg = dataset_config(train_ds)
    data_range = float(data_cfg["metrics"]["data_range"])
    rgb_idx = rgb_band_indices(data_cfg)
    # A subset of --val-batches, never more. See --metric-batches' help.
    metric_batches = max(1, min(int(args.metric_batches), int(args.val_batches)))

    # The parameter budget, asserted rather than trusted (AGENTS.md section 1),
    # head included.
    budget = int(data_cfg["runtime"]["max_parameters"])
    if count_params(model) > budget:
        raise ValueError(
            f"model has {count_params(model):,} parameters, over "
            f"cfg.runtime.max_parameters = {budget:,}."
        )
    if uncertainty_on:
        print(
            f"[loss] + heteroscedastic NLL: pure L1 for {args.nll_warmup} iters "
            f"(variance branch detached), weight ramps 0 -> {args.nll_weight} over "
            f"the next {args.nll_ramp}. logvar clamp [{args.logvar_min}, "
            f"{args.logvar_max}], init {args.logvar_init}.",
            flush=True,
        )

    def save(path: Path, it: int, metrics: dict = None, metrics_iter: int = None) -> None:
        """Write a checkpoint that can be identified without its directory.

        WEIGHTS ALONE ARE NOT A RESULT. A .pt file that says only "iteration
        7000" cannot answer the two questions a Day 3 checkpoint exists to
        answer -- which code and config produced it, and what it was optimising
        -- and by the time those are asked the run directory has usually been
        renamed, merged, or fetched into a folder of three sibling runs whose
        only difference is two floats. So every checkpoint carries its own
        provenance.

        ``lambdas`` is the load-bearing entry: the frozen config records 0.0/0.0
        because it describes the control, so ``config_hash`` is IDENTICAL across
        A2, B1 and B2 and cannot distinguish them.

        Args:
            path: Destination ``.pt``.
            it: The iteration this checkpoint was taken at.
            metrics: Validation metrics from :func:`validate`, or None before
                the first validation has run.
            metrics_iter: The iteration ``metrics`` was measured at. Recorded
                separately because a checkpoint is not always taken on a
                validation boundary, and a stale metric labelled with the
                checkpoint's own iteration would be a quiet lie. With
                --ckpt-every a multiple of --val-every the two agree, and the
                field then proves it rather than assuming it.
        """
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "it": it, "best": best, "args": vars(args),
                    "config_hash": meta["config_hash"],
                    "git_sha": meta["git_sha"],
                    "git_dirty": meta["git_dirty"],
                    "lambdas": meta["lambdas"],
                    "spectral_downsample": str(args.spectral_downsample),
                    "val_metrics": metrics,
                    "val_metrics_iter": metrics_iter,
                    "uncertainty": meta["uncertainty"],
                    "sharpness_reference": reference}, path)

    @torch.no_grad()
    def validate() -> dict:
        """Validate: quality, spectral consistency, and the blur diagnostic.

        Three groups of numbers, and the difference between them matters:

        1. **Over ALL ``--val-batches``**, on device: PSNR, the two spectral
           terms, and the two sharpness proxies. Cheap enough to run on the
           full validation subset every time.
        2. **Over the first ``--metric-batches`` only**: SSIM, LPIPS, SAM
           against the HR, and ERGAS, through :mod:`src.metrics.image_quality`
           -- the project's single implementation, so these are the same
           quantities the baseline table and the final evaluation report. They
           run on the CPU in float64 by design, and LPIPS costs ~9 s per batch
           of 16 there; over 25 batches, 24 validations and three runs that is
           4.5 hours of a 9-hour session spent measuring rather than training.
           The subset is the SAME fixed batches at every validation (the val
           loader does not shuffle and val patches are grid-mode, epoch-pinned),
           so the CURVE is self-consistent and comparable across runs. It is
           not a substitute for the full-split evaluation, which
           ``scripts/eval_runA.py`` does afterwards on the finished checkpoint.
        3. **The spectral terms are measured WHETHER OR NOT they are trained.**
           The control (lambdas 0.0) logs ``l1_spec`` and ``sam_spec`` exactly
           as the spectral run does. Without that, "the spectral term went
           down" is unfalsifiable.

        Everything is computed in float32 or better, outside autocast.
        ``arccos`` near +-1 -- where a trained model sits -- loses most of its
        significant digits in fp16, and a validation number that moves with the
        AMP setting is not a measurement.

        Returns:
            A dict of floats:

            - ``psnr`` (dB), ``sharpness`` (reflectance per pixel),
              ``hf_energy`` (dimensionless), ``l1_spec`` (reflectance),
              ``sam_spec`` (RADIANS, spectral domain: LR vs downsampled SR),
              ``sam_valid_frac`` -- means over ``n_val_batches``.
            - ``ssim``, ``lpips``, ``sam_hr`` (DEGREES, image domain: SR vs
              HR), ``ergas`` -- means over ``n_metric_batches``.
            - ``n_val_batches``, ``n_metric_batches`` -- how many batches each
              group actually saw, so a truncated validation is visible.

            The spectral entries are UNWEIGHTED, so they are comparable across
            runs with different lambdas and against the floor from
            ``scripts/spectral_floor.py``.

        Raises:
            ValueError: An aggregated metric came out non-finite. The
                validation batches are fixed, so this fires at the FIRST
                validation -- minutes in -- rather than after hours of
                training, and a NaN silently filling the log for the rest of a
                run is exactly what it exists to prevent.
        """
        model.eval()
        wide = {"psnr": 0.0, "l1_spec": 0.0, "sam_spec": 0.0, "sam_valid_frac": 0.0,
                "sharpness": 0.0, "hf_energy": 0.0}
        narrow = {"ssim": 0.0, "lpips": 0.0, "sam_hr": 0.0, "ergas": 0.0}
        n_wide = n_narrow = 0
        # The sigma-collapse monitor, over the same val batches as PSNR.
        monitor = LogvarMonitor(args.logvar_min, args.logvar_max) if uncertainty_on else None
        nll_sum = 0.0
        for i, b in enumerate(val_dl):
            if i >= args.val_batches:
                break
            lr_, hr = b["lr"].to(dev, non_blocking=True), b["hr"].to(dev, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp) and dev == "cuda"):
                out = model(lr_)
            if uncertainty_on:
                sr, logvar = out[0].float(), out[1].float()
                monitor.update(logvar)
                nll_sum += float(gaussian_nll(sr, logvar, hr.float(),
                                              args.logvar_min, args.logvar_max))
            else:
                sr = out.float()

            wide["psnr"] += psnr(sr, hr)
            parts = spectral_terms(sr, lr_.float(), mode=args.spectral_downsample)
            wide["l1_spec"] += float(parts["l1_spec"])
            wide["sam_spec"] += float(parts["sam"])
            wide["sam_valid_frac"] += float(parts["sam_valid_frac"])
            # On device: the whole point of the diagnostic is that it is cheap
            # enough to run on every validation batch of every run.
            blur = sharpness_terms(sr, cutoff=args.hf_cutoff)
            wide["sharpness"] += blur["sharpness"]
            wide["hf_energy"] += blur["hf_energy"]
            n_wide += 1

            if i < metric_batches:
                # src.metrics works in CPU float64 by contract; the transfer is
                # the cost and it is why this runs on a subset.
                sr_c, hr_c = sr.cpu(), hr.cpu()
                narrow["ssim"] += float(np.mean(ssim(sr_c, hr_c, data_range=data_range).mean))
                narrow["sam_hr"] += float(np.mean(sam_hr(sr_c, hr_c).mean_deg))
                narrow["ergas"] += float(np.mean(ergas(sr_c, hr_c, scale=args.scale)))
                narrow["lpips"] += float(np.mean(
                    lpips(sr_c, hr_c, rgb_indices=rgb_idx, data_range=data_range).distance
                ))
                n_narrow += 1
        model.train()

        vals = {key: value / max(1, n_wide) for key, value in wide.items()}
        vals.update({key: value / max(1, n_narrow) for key, value in narrow.items()})
        if uncertainty_on:
            vals["nll"] = nll_sum / max(1, n_wide)
            vals.update({f"logvar_{k}": v for k, v in monitor.summary().items()})
        bad = sorted(key for key, value in vals.items() if not math.isfinite(value))
        if bad:
            raise ValueError(
                f"validation produced non-finite values for {bad} over "
                f"{n_wide} batches. This is not a transient: the validation "
                "batches are fixed, so every later validation would report the "
                "same thing and log.csv would fill with NaN while the run "
                "continued to look healthy. Check the val split for an all-zero "
                "band (SAM and ERGAS are undefined on one) before restarting."
            )
        vals["n_val_batches"] = float(n_wide)
        vals["n_metric_batches"] = float(n_narrow)
        return vals

    @torch.no_grad()
    def measure_reference() -> dict:
        """The two fixed sharpness lines, measured once, over the val batches.

        These are what make the blur diagnostic readable. A sharpness number on
        its own has no scale; the same number sitting just above the BICUBIC
        line or just below the GROUND TRUTH line is two opposite conclusions.
        Neither line involves the model, so both are identical across A2, B1
        and B2 by construction -- which is exactly what lets "B1 drifted toward
        bicubic" be a statement about B1 rather than about its data.

        Computed over the same ``--val-batches`` the model is scored on, so the
        lines and the curve are measured on the same pixels.

        Returns:
            The dict from :func:`src.metrics.sharpness.reference_sharpness`,
            plus ``n_batches``. Written to ``<out>/sharpness_reference.json``
            and embedded in every checkpoint.
        """
        totals = {"hr_sharpness": 0.0, "hr_hf_energy": 0.0,
                  "bicubic_sharpness": 0.0, "bicubic_hf_energy": 0.0}
        seen = 0
        for i, b in enumerate(val_dl):
            if i >= args.val_batches:
                break
            lr_ = b["lr"].to(dev, non_blocking=True).float()
            hr = b["hr"].to(dev, non_blocking=True).float()
            for key, value in reference_sharpness(
                hr, lr_, scale=args.scale, cutoff=args.hf_cutoff
            ).items():
                totals[key] += value
            seen += 1
        if seen == 0:
            raise RuntimeError(
                "the validation loader yielded no batches, so the sharpness "
                "reference lines cannot be measured and the blur diagnostic "
                "would be unreadable. Check the val split."
            )
        out_ref = {key: value / seen for key, value in totals.items()}
        out_ref["n_batches"] = float(seen)
        out_ref["hf_cutoff"] = float(args.hf_cutoff)
        return out_ref

    reference = measure_reference()
    (out / "sharpness_reference.json").write_text(json.dumps(reference, indent=2))
    print(
        f"[blur] reference lines over {int(reference['n_batches'])} val batches "
        f"(cutoff {reference['hf_cutoff']} x Nyquist): "
        f"sharpness GT {reference['hr_sharpness']:.6f} / bicubic "
        f"{reference['bicubic_sharpness']:.6f}; "
        f"hf_energy GT {reference['hr_hf_energy']:.6f} / bicubic "
        f"{reference['bicubic_hf_energy']:.6f}. A run whose sharpness migrates "
        "toward the bicubic line is buying spectral consistency with blur.",
        flush=True,
    )
    print(
        f"[val] full suite (SSIM/LPIPS/SAM_hr/ERGAS) on the first "
        f"{metric_batches} of {args.val_batches} val batches; PSNR, spectral "
        f"terms and the blur diagnostic on all {args.val_batches}.",
        flush=True,
    )

    # Iterations per epoch, and the epoch a resumed run continues from. Derived
    # rather than stored so an interrupted run rejoins the same crop sequence:
    # the crops for (seed, epoch, index) are a pure function, so recovering the
    # epoch number recovers the exact data stream.
    iters_per_epoch = max(1, len(train_dl))
    start_epoch = start_it // iters_per_epoch
    print(f"[data] {len(train_ds)} train patches, {iters_per_epoch} it/epoch, "
          f"starting at epoch {start_epoch}", flush=True)

    it_stream = epoch_stream(
        train_dl,
        train_ds,
        start_epoch,
        shuffle_generator=shuffle_generator,
        shuffle_seed=shuffle_seed,
    )
    model.train()
    t0, tw, run_loss = time.time(), time.time(), 0.0
    epoch = start_epoch
    # The metrics from the most recent validation, and the iteration they were
    # measured at, so a checkpoint carries the numbers that describe it rather
    # than nothing. None until the first validation has run.
    last_vals, last_vals_it = None, None
    for it in range(start_it, args.iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, args)
        epoch, b = next(it_stream)
        lr_, hr = b["lr"].to(dev, non_blocking=True), b["hr"].to(dev, non_blocking=True)
        # 0.0 throughout for a run without the head; see src/losses/nll.py.
        nll_w = (nll_weight_at(it, args.nll_warmup, args.nll_ramp, args.nll_weight)
                 if uncertainty_on else 0.0)
        with torch.cuda.amp.autocast(enabled=bool(args.amp) and dev == "cuda"):
            if uncertainty_on:
                # detach_var during the warmup: the variance branch gets no
                # path to the trunk, and with nll_w == 0 no loss term either,
                # so the reconstruction trains exactly as a Day 3 run's.
                sr, logvar = model(lr_, detach_var=(nll_w == 0.0))
            else:
                # The control path, unchanged: same graph as A2/B1/B2.
                sr = model(lr_)
            loss = F.l1_loss(sr, hr)
        if spectral_on:
            # Outside autocast, in float32: see validate(). The branch is
            # skipped entirely when both lambdas are 0.0, so the control run
            # executes the same graph Run A did rather than one with a
            # multiply-by-zero grafted onto it.
            with torch.cuda.amp.autocast(enabled=False):
                spec, _ = spectral_consistency(
                    sr.float(), lr_.float(),
                    lam1=args.spectral_lambda1, lam2=args.spectral_lambda2,
                    mode=args.spectral_downsample,
                )
            # NOTE for whoever reads the two runs side by side: from here the
            # `loss` column of log.csv is the FULL objective, so it is not
            # comparable against the control's, which is L1 alone. The
            # cross-run comparisons are val_psnr and the unweighted
            # val_l1_spec / val_sam, which mean the same thing in both.
            loss = loss + spec
        if nll_w > 0.0:
            # float32, outside autocast: squared reflectance residuals of 1e-6
            # sit below fp16's normal range. From here `loss` is L1 + w*NLL,
            # and NLL is negative in reflectance units (logvar ~ -9), so the
            # loss column can go below zero; compare runs on val columns.
            with torch.cuda.amp.autocast(enabled=False):
                nll = gaussian_nll(sr.float(), logvar.float(), hr.float(),
                                   args.logvar_min, args.logvar_max)
            loss = loss + nll_w * nll
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
                # A train row leaves every validation column blank. Blank, not
                # zero: src/eval/curves.py separates the two kinds of row by
                # val_psnr being absent, and a 0.0 would plot as a real
                # measurement of a catastrophically bad model.
                csv.writer(f).writerow(
                    [it + 1, f"{avg:.6f}", f"{lr_at(it,args):.3e}", "", f"{sec:.1f}"]
                    + [""] * (len(LOG_COLUMNS) - len(LEGACY_COLUMNS))
                )

        if (it + 1) % args.val_every == 0 or (it + 1) == args.iters:
            vals = validate()
            last_vals, last_vals_it = vals, it + 1
            v = vals["psnr"]
            print(
                f"[val] it {it+1} psnr {v:.3f} dB (best {max(best, v):.3f}) "
                f"ssim {vals['ssim']:.4f} lpips {vals['lpips']:.4f} "
                f"sam_hr {vals['sam_hr']:.4f} deg ergas {vals['ergas']:.4f} | "
                f"l1_spec {vals['l1_spec']:.6f} sam_spec {vals['sam_spec']:.6f} rad "
                f"({math.degrees(vals['sam_spec']):.4f} deg, "
                f"valid {vals['sam_valid_frac']:.4f})",
                flush=True,
            )
            # The blur diagnostic, printed against its two fixed lines on the
            # same line it is measured on, because the number alone means
            # nothing. `frac` places the run between bicubic (0.0) and ground
            # truth (1.0): falling toward 0 is the failure mode this whole
            # instrument exists to catch, and it has to be legible while
            # tailing a log at hour 5, not after the run.
            span = reference["hr_sharpness"] - reference["bicubic_sharpness"]
            frac = (
                (vals["sharpness"] - reference["bicubic_sharpness"]) / span
                if span != 0
                else float("nan")
            )
            print(
                f"[blur] it {it+1} sharpness {vals['sharpness']:.6f} "
                f"(bicubic {reference['bicubic_sharpness']:.6f}, GT "
                f"{reference['hr_sharpness']:.6f}, frac {frac:+.3f}) "
                f"hf_energy {vals['hf_energy']:.6f} "
                f"(bicubic {reference['bicubic_hf_energy']:.6f}, GT "
                f"{reference['hr_hf_energy']:.6f})",
                flush=True,
            )
            if uncertainty_on:
                # The Day 4 kill-criterion signal, on its own line so it can be
                # grepped out of a live log. spatial_std -> 0 is sigma collapse;
                # clamp_lo is EXPECTED to be non-zero (see src/losses/nll.py).
                print(
                    f"[sigma] it {it+1} nll_w {nll_w:.4f} nll {vals['nll']:.5f} "
                    f"logvar mean {vals['logvar_mean']:.4f} min {vals['logvar_min']:.4f} "
                    f"max {vals['logvar_max']:.4f} spatial_std "
                    f"{vals['logvar_spatial_std']:.5f} clamp lo "
                    f"{vals['logvar_clamp_lo_frac']:.4f} hi "
                    f"{vals['logvar_clamp_hi_frac']:.4f}",
                    flush=True,
                )
                unc_cells = [
                    f"{vals['nll']:.6f}", f"{nll_w:.6f}",
                    f"{vals['logvar_mean']:.6f}", f"{vals['logvar_min']:.6f}",
                    f"{vals['logvar_max']:.6f}", f"{vals['logvar_spatial_std']:.8f}",
                    f"{vals['logvar_clamp_lo_frac']:.6f}",
                    f"{vals['logvar_clamp_hi_frac']:.6f}",
                ]
            else:
                # Blank, not zero: never measured on a run without the head.
                unc_cells = [""] * (len(LOG_COLUMNS) - len(REFERENCE_COLUMNS))
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [it + 1, "", "", f"{v:.4f}", "",
                     f"{vals['l1_spec']:.8f}", f"{vals['sam_spec']:.8f}",
                     f"{vals['sam_valid_frac']:.6f}",
                     f"{vals['sharpness']:.8f}", f"{vals['hf_energy']:.8f}",
                     f"{vals['ssim']:.6f}", f"{vals['lpips']:.6f}",
                     f"{vals['sam_hr']:.6f}", f"{vals['ergas']:.6f}"]
                    + unc_cells
                )
            # Checkpoint selection stays on PSNR alone, unchanged from Run A and
            # identical in both runs. Selecting on the combined objective would
            # make the two runs pick their best.pt by different rules, and the
            # comparison would confound "trained differently" with "chosen
            # differently".
            if v > best:
                best = v; save(out / "best.pt", it, vals, last_vals_it)

        if (it + 1) % args.ckpt_every == 0 or (it + 1) == args.iters:
            save(ck_last, it, last_vals, last_vals_it)
        if (time.time() - t0) / 3600.0 > args.max_hours:
            save(ck_last, it, last_vals, last_vals_it)
            print(f"[stop] time budget hit at it={it+1}", flush=True)
            break

    save(ck_last, args.iters - 1, last_vals, last_vals_it)
    elapsed_min = (time.time() - t0) / 60.0
    print(f"[done] iters={args.iters} best_val_psnr={best:.3f} elapsed={elapsed_min:.1f} min", flush=True)

    # A machine-readable summary next to the checkpoints. scripts/day3_runs.py
    # reads this to enforce the wall-clock guard between runs -- it needs the
    # elapsed time of a finished run without parsing stdout, and it needs the
    # config hash and git SHA to assert that all three runs were the same
    # experiment apart from the lambdas.
    (out / "run_summary.json").write_text(json.dumps({
        "config_hash": meta["config_hash"],
        "git_sha": meta["git_sha"],
        "git_dirty": meta["git_dirty"],
        "lambdas": meta["lambdas"],
        "spectral_downsample": str(args.spectral_downsample),
        "uncertainty": meta["uncertainty"],
        "iters_requested": int(args.iters),
        "iters_completed": int(last_vals_it) if last_vals_it else None,
        "best_val_psnr": float(best),
        "elapsed_min": elapsed_min,
        "final_val": last_vals,
        "sharpness_reference": reference,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))


if __name__ == "__main__":
    main()
