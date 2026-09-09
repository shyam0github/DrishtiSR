"""Day 3: A2, B1 and optionally B2, in one Kaggle session, under a wall clock.

WHAT THIS IS FOR. The Day 3 result is a COMPARISON, and a comparison is only
worth the care taken to make its two halves identical. Three runs differ in
exactly two floats:

======  ==========  ==========  ================================================
run     lambda1     lambda2     role
======  ==========  ==========  ================================================
A2      0.0         0.0         control, load-bearing -- everything is measured
                                against it, including the spectral terms it is
                                not optimising
B1      0.1         0.02        the primary result
B2      0.3         0.06        upside; runs only if the clock allows
======  ==========  ==========  ================================================

Everything else -- the frozen config, the git commit, the seeds, augmentation,
the iteration count, the validation cadence, the degradation operator, the
instrumentation settings -- is built ONCE, here, into a shared argument list
that every run receives. :func:`build_run_args` then adds nothing but the two
lambdas, and :func:`assert_runs_differ_only_in_lambdas` proves it before a
single GPU-second is spent. That assertion is the reason this file exists
instead of three job entries in ``configs/kaggle_jobs.yaml``: three hand-written
argument lists that are supposed to agree in twenty places will eventually not.

WHY ONE KERNEL. Kaggle gives a session at most 9 hours and the weekly budget is
30 GPU-hours. Three runs in three sessions means three queue waits, three pip
installs and -- fatally -- three chances for a different accelerator, since
Kaggle honours ``machine_shape`` silently or ignores it silently. One session
means one card and one set of timings.

THE WALL-CLOCK GUARD, and what it protects. **A2 and B1 complete is the
deliverable. B2 is upside.** So:

- if A2 crashes, B1 does not start and the error is surfaced -- a B1 with no
  control is not a result;
- if A2 alone runs past ``--a2-cut-hours``, B1's iteration count is cut to fit
  the remaining budget and B2 is abandoned. A truncated B1 that finishes and
  checkpoints beats a complete B1 killed at the 9-hour limit mid-write;
- after B1, B2 starts only if at least ``--b2-margin`` times the MEDIAN
  single-run duration remains. The median of the two observed runs, not an
  estimate: by then the session has measured itself.

WHAT EACH RUN LEAVES BEHIND, under ``--runs-root/<name>/``: ``last.pt``,
``best.pt``, ``log.csv``, ``run_metadata.json``, ``sharpness_reference.json``
and ``run_summary.json``. This script adds ``day3_manifest.json`` at the root,
which is the single file that answers "what were these three runs, and were
they the same experiment".

    python scripts/day3_runs.py --config configs/base.yaml
    python scripts/day3_runs.py --config configs/base.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FROZEN_CONFIG, HASH_FIELD, load_frozen  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.gitmeta import NO_GIT, git_metadata  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

__all__ = [
    "RUN_PLAN",
    "TRAINER_MODULE",
    "Day3Error",
    "assert_runs_differ_only_in_lambdas",
    "assert_same_experiment",
    "build_run_args",
    "build_shared_args",
    "cut_iters",
    "should_run_b2",
]

# The trainer, as a MODULE. `python -m` puts the working directory on sys.path,
# which is what makes both `drishtisr.models.edsr` and `src.data.loader` resolve
# from one process -- see the drishtisr alias package's docstring. Running
# src/train.py by path dies on an import instead.
TRAINER_MODULE = "drishtisr.train"

# The experiment. Order is deliberate: the control first, so a crash in the
# expensive shared setup costs the cheapest run, and so B1 never exists without
# something to be compared against.
RUN_PLAN = (
    {
        "name": "a2",
        "lambda1": 0.0,
        "lambda2": 0.0,
        "optional": False,
        "role": "control -- spectral OFF, both terms still MEASURED",
    },
    {
        "name": "b1",
        "lambda1": 0.1,
        "lambda2": 0.02,
        "optional": False,
        "role": "primary result -- spectral consistency at the candidate weights",
    },
    {
        "name": "b2",
        "lambda1": 0.3,
        "lambda2": 0.06,
        "optional": True,
        "role": "upside -- 3x the weights, to bracket the trend",
    },
)


class Day3Error(RuntimeError):
    """The session cannot produce a valid comparison and must not pretend to."""


# -- the argument lists, and the proof that they agree ----------------------


def build_shared_args(args: argparse.Namespace) -> List[str]:
    """Every trainer flag that is IDENTICAL across A2, B1 and B2.

    Built once and handed to all three runs, so the twenty settings that must
    match cannot drift. The two that must NOT match are added by
    :func:`build_run_args` and by nothing else.

    ``--out``, ``--iters`` and ``--max-hours`` are deliberately absent: the
    first is per-run, and the other two are set per-run by the wall-clock guard,
    which is the one place allowed to shorten a run.

    Args:
        args: Parsed CLI arguments of this script.

    Returns:
        The shared argument list, as strings ready for ``subprocess``.
    """
    return [
        "--data-module", str(args.data_module),
        "--data-class", str(args.data_class),
        # "auto", not a mount path. MEASURED 2026-09-06: the live Kaggle mount
        # is /kaggle/input/datasets/<owner>/<slug>. resolve_cache_dir is the
        # only thing in this project allowed to decide where data lives.
        "--data-root", str(args.data_root),
        "--batch", str(args.batch),
        "--patch-lr", str(args.patch_lr),
        "--scale", str(args.scale),
        "--in-ch", str(args.in_ch),
        "--n-resblocks", str(args.n_resblocks),
        "--n-feats", str(args.n_feats),
        "--lr", repr(float(args.lr)),
        "--min-lr", repr(float(args.min_lr)),
        "--warmup", str(args.warmup),
        # 0. Run A overfit WITH decay, so decay was not the missing regulariser;
        # augmentation is. See configs/frozen_day3.yaml.
        "--wd", repr(float(args.wd)),
        "--clip", repr(float(args.clip)),
        "--workers", str(args.workers),
        "--val-every", str(args.val_every),
        "--val-batches", str(args.val_batches),
        "--metric-batches", str(args.metric_batches),
        "--hf-cutoff", repr(float(args.hf_cutoff)),
        "--ckpt-every", str(args.ckpt_every),
        "--log-every", str(args.log_every),
        "--seed", str(args.seed),
        "--amp", str(args.amp),
        "--resume", str(args.resume),
        # The degradation operator D. Shared even though A2's lambdas are 0,
        # because the control MEASURES through it. A control measuring with a
        # different D from the spectral run would not be logging the same
        # quantity, and the comparison would be of two different numbers.
        "--spectral-downsample", str(args.spectral_downsample),
    ]


def build_run_args(
    shared: Sequence[str],
    spec: Dict[str, Any],
    out_dir: Path,
    iters: int,
    max_hours: float,
) -> List[str]:
    """One run's complete argument list: the shared list, plus its two lambdas.

    Args:
        shared: The output of :func:`build_shared_args`.
        spec: One entry of :data:`RUN_PLAN`.
        out_dir: Where this run writes. Created by the caller.
        iters: Iteration budget, possibly cut by the wall-clock guard.
        max_hours: The trainer's own stop-cleanly deadline, set from the time
            actually left in the session rather than from a constant, so the
            child checkpoints and exits instead of being killed mid-write.

    Returns:
        The argument list for ``python -m drishtisr.train``.
    """
    return list(shared) + [
        "--spectral-lambda1", repr(float(spec["lambda1"])),
        "--spectral-lambda2", repr(float(spec["lambda2"])),
        "--iters", str(int(iters)),
        "--max-hours", repr(float(max_hours)),
        "--out", str(out_dir),
    ]


def assert_runs_differ_only_in_lambdas(arg_lists: Dict[str, Sequence[str]]) -> None:
    """Prove the runs are the same experiment before any of them starts.

    THE POINT OF THIS SCRIPT. A comparison between A2 and B1 attributes every
    difference in their curves to the spectral term. That attribution is only
    valid if the spectral term is the only thing that differed, and "I built
    both lists carefully" is not evidence. So the lists are diffed, and anything
    other than the two lambdas -- and the per-run ``--out`` -- is a refusal.

    ``--iters`` and ``--max-hours`` are exempted because the wall-clock guard is
    allowed to shorten a run; a shortened B1 is still a valid, clearly labelled
    result, whereas a B1 with a different patch size silently is not.

    Args:
        arg_lists: ``{run name: argument list}``, at least two entries.

    Raises:
        Day3Error: Two runs differ in a flag that is not one of the exempted
            per-run ones, or a run is missing a flag another one has.
    """
    allowed = {
        "--spectral-lambda1",
        "--spectral-lambda2",
        "--out",
        "--iters",
        "--max-hours",
    }

    def as_map(values: Sequence[str]) -> Dict[str, str]:
        items = list(values)
        if len(items) % 2:
            raise Day3Error(
                f"an argument list has an odd length ({len(items)}), so it "
                "cannot be read as flag/value pairs and the comparison below "
                "would silently mis-align every flag after the offender. Every "
                "trainer flag this script passes takes a value; a bare "
                "store_true flag needs this function taught about it."
            )
        return {items[i]: items[i + 1] for i in range(0, len(items), 2)}

    maps = {name: as_map(values) for name, values in arg_lists.items()}
    names = sorted(maps)
    reference_name = names[0]
    reference = maps[reference_name]

    for name in names[1:]:
        other = maps[name]
        keys = set(reference) | set(other)
        differing = sorted(
            key for key in keys if reference.get(key) != other.get(key)
        )
        unexpected = [key for key in differing if key not in allowed]
        if unexpected:
            detail = "\n".join(
                f"    {key}: {reference_name}={reference.get(key)!r} "
                f"{name}={other.get(key)!r}"
                for key in unexpected
            )
            raise Day3Error(
                f"runs {reference_name!r} and {name!r} differ in flags that are "
                f"not the spectral lambdas:\n{detail}\n"
                "The whole Day 3 claim is that the lambdas are the only "
                "difference between these runs. Refusing to spend GPU-hours "
                "producing a comparison that cannot support it."
            )


def assert_same_experiment(records: Sequence[Dict[str, Any]]) -> None:
    """Every finished run must report the same config hash and the same commit.

    Checked AFTER the runs, from what each one actually wrote, not from what
    this script intended. The two failures it catches are real ones seen on this
    project: a frozen config edited between runs (same script, different
    hyperparameters), and a session that somehow ran different code.

    ``config_hash`` alone is not enough and never was -- Day 3's frozen-crop bug
    changed the training data stream without touching one config value -- which
    is why the commit is checked beside it.

    Args:
        records: One dict per completed run, each with ``name``, ``config_hash``
            and ``git_sha``.

    Raises:
        Day3Error: The hashes or the commits are not unanimous, or a run
            recorded the ``unavailable`` sentinel instead of a real value. The
            sentinel is refused explicitly: two runs that both failed to record
            provenance would otherwise "agree".
    """
    if not records:
        raise Day3Error("no runs completed, so there is nothing to compare.")

    for field, sentinel in (("config_hash", "unavailable"), ("git_sha", NO_GIT)):
        values = {record["name"]: record.get(field) for record in records}
        missing = sorted(
            name for name, value in values.items() if not value or value == sentinel
        )
        if missing:
            raise Day3Error(
                f"runs {missing} recorded {field}={sentinel!r}. A run with no "
                "provenance cannot be compared with one that has it, and two "
                "unknowns must never be taken as a match."
            )
        distinct = sorted(set(values.values()))
        if len(distinct) != 1:
            raise Day3Error(
                f"the completed runs disagree on {field}: {values}. They are "
                "not the same experiment, so any difference between their "
                "curves is unattributable. Do not report these together."
            )


# -- the wall-clock guard ---------------------------------------------------


def cut_iters(full_iters: int, hours_used: float, hours_left: float, ckpt_every: int) -> int:
    """Iterations that fit in ``hours_left``, at the rate the last run measured.

    Rounded DOWN to a multiple of ``ckpt_every`` so the shortened run still ends
    on a checkpoint -- and, since ``--ckpt-every`` is a multiple of
    ``--val-every``, on a validation too. A run that stops 40 iterations after
    its last checkpoint has thrown those 40 away.

    Args:
        full_iters: The iteration count the run would have had.
        hours_used: Wall-clock hours the reference run took.
        hours_left: Hours available for this run.
        ckpt_every: The checkpoint interval, the granularity of the result.

    Returns:
        The cut iteration count, at least ``ckpt_every`` and never more than
        ``full_iters``. Never zero: a run of zero iterations would produce an
        untrained checkpoint that looks like a result.

    Raises:
        ValueError: ``hours_used`` is not positive, so no rate can be derived.
    """
    if hours_used <= 0:
        raise ValueError(
            f"hours_used must be positive to derive an iteration rate, got "
            f"{hours_used!r}."
        )
    scaled = int(full_iters * (max(0.0, hours_left) / hours_used))
    aligned = (scaled // int(ckpt_every)) * int(ckpt_every)
    return max(int(ckpt_every), min(int(full_iters), aligned))


def should_run_b2(durations_h: Sequence[float], hours_left: float, margin: float) -> bool:
    """Whether the optional third run fits, judged on MEASURED durations.

    The rule the session is held to: B2 starts only if at least ``margin`` times
    the median observed single-run duration remains. The median of the runs that
    actually happened, not a prediction -- by this point the session has
    measured its own card, its own data loader and its own contention.

    The margin is above 1.0 on purpose. A run that fits exactly does not fit: it
    leaves nothing for the final checkpoint write, the output inventory and
    Kaggle's own commit step, and a session killed during those loses the runs
    that had already succeeded.

    Args:
        durations_h: Wall-clock hours of the completed runs. Empty means nothing
            has been measured, and the answer is No.
        hours_left: Hours remaining in the session budget.
        margin: Multiple of the median required. ``--b2-margin``, default 2.5.

    Returns:
        True only if the run is affordable on the evidence available.
    """
    if not durations_h:
        return False
    return hours_left >= margin * statistics.median(durations_h)


# -- running one child ------------------------------------------------------


def run_one(
    args: argparse.Namespace,
    spec: Dict[str, Any],
    arg_list: Sequence[str],
    logger: Any,
) -> Dict[str, Any]:
    """Launch one training run as a child process and wait for it.

    A child process rather than an in-process call, deliberately: the trainer
    allocates CUDA memory, builds dataloader workers and installs its own
    signal handling, and three of those in one interpreter share state that none
    of them was written to share. A crashed child is also a non-zero exit code
    instead of a half-initialised third run.

    Output is NOT captured. It streams to the kernel log, because a run being
    watched at hour 5 is the entire reason the blur diagnostic prints every
    validation.

    Args:
        args: This script's parsed arguments.
        spec: The :data:`RUN_PLAN` entry.
        arg_list: The run's complete trainer arguments.
        logger: Project logger.

    Returns:
        A record of the run: name, lambdas, elapsed hours, and the contents of
        the run's ``run_summary.json``.

    Raises:
        Day3Error: The child exited non-zero, or finished without writing
            ``run_summary.json``. Both are surfaced rather than skipped -- a
            missing summary means the run did not reach its end, and treating
            it as a result would put an untrained checkpoint in a results table.
    """
    command = [sys.executable, "-m", TRAINER_MODULE, *arg_list]
    logger.info("=" * 70)
    logger.info("RUN %s -- %s", spec["name"].upper(), spec["role"])
    logger.info(
        "lambda1=%s lambda2=%s", spec["lambda1"], spec["lambda2"]
    )
    logger.info("%s", " ".join(command))
    logger.info("=" * 70)

    started = time.time()
    completed = subprocess.run(command, cwd=str(repo_root()))
    elapsed_h = (time.time() - started) / 3600.0

    if completed.returncode != 0:
        raise Day3Error(
            f"run {spec['name']!r} exited with code {completed.returncode} after "
            f"{elapsed_h * 60:.1f} min. The traceback is above in this log. "
            + (
                "This is the CONTROL: nothing after it can be interpreted "
                "without it, so the session stops here rather than producing a "
                "spectral run with nothing to compare against."
                if not spec["optional"]
                else "This run was optional; the runs before it are unaffected."
            )
        )

    summary_path = Path(args.runs_root) / spec["name"] / "run_summary.json"
    if not summary_path.is_file():
        raise Day3Error(
            f"run {spec['name']!r} exited 0 but wrote no {summary_path.name}. "
            "src/train.py writes that file as its last action, so the run did "
            "not reach the end of its loop and its checkpoints describe an "
            "unfinished run."
        )
    summary = json.loads(summary_path.read_text())

    record = {
        "name": spec["name"],
        "role": spec["role"],
        "lambda1": float(spec["lambda1"]),
        "lambda2": float(spec["lambda2"]),
        "elapsed_h": elapsed_h,
        "config_hash": summary.get("config_hash"),
        "git_sha": summary.get("git_sha"),
        "git_dirty": summary.get("git_dirty"),
        "iters_requested": summary.get("iters_requested"),
        "best_val_psnr": summary.get("best_val_psnr"),
        "final_val": summary.get("final_val"),
        "sharpness_reference": summary.get("sharpness_reference"),
    }
    logger.info(
        "RUN %s complete in %.2f h -- best val PSNR %.3f dB",
        spec["name"].upper(),
        elapsed_h,
        float(record["best_val_psnr"] or float("nan")),
    )
    return record


def report_blur(records: Sequence[Dict[str, Any]], logger: Any) -> None:
    """Print the blur verdict, side by side, in the kernel log.

    The question this session exists to answer, answered in the log rather than
    left for a notebook afterwards: did the spectral runs' sharpness migrate
    toward the bicubic line while the control's held? ``frac`` places a run
    between bicubic (0.0) and ground truth (1.0).

    Args:
        records: Completed run records from :func:`run_one`.
        logger: Project logger.
    """
    logger.info("")
    logger.info("BLUR DIAGNOSTIC -- final validation of each run")
    logger.info(
        "%-5s %-14s %10s %10s %10s %10s",
        "run", "lambdas", "sharpness", "frac", "hf_energy", "psnr",
    )
    for record in records:
        final = record.get("final_val") or {}
        reference = record.get("sharpness_reference") or {}
        span = reference.get("hr_sharpness", 0.0) - reference.get("bicubic_sharpness", 0.0)
        sharp = final.get("sharpness")
        frac = (
            (sharp - reference["bicubic_sharpness"]) / span
            if sharp is not None and span
            else float("nan")
        )
        logger.info(
            "%-5s %-14s %10.6f %10.3f %10.6f %10.3f",
            record["name"],
            f"{record['lambda1']}/{record['lambda2']}",
            sharp if sharp is not None else float("nan"),
            frac,
            final.get("hf_energy", float("nan")),
            record.get("best_val_psnr") or float("nan"),
        )
    reference = (records[0].get("sharpness_reference") or {}) if records else {}
    if reference:
        logger.info(
            "reference lines: bicubic sharpness %.6f (frac 0.000), GT %.6f "
            "(frac 1.000); bicubic hf_energy %.6f, GT %.6f",
            reference.get("bicubic_sharpness", float("nan")),
            reference.get("hr_sharpness", float("nan")),
            reference.get("bicubic_hf_energy", float("nan")),
            reference.get("hr_hf_energy", float("nan")),
        )
    logger.info(
        "A spectral run whose frac falls toward 0.0 while the control's holds "
        "is buying spectral consistency with blur."
    )


# -- entry point ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Defaults are the Day 3 run spec; nothing here is a literal in code."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/base.yaml",
                   help="YAML merged over configs/base.yaml, for seeds and paths.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny iteration counts so every path runs end to end on "
                        "CPU in a couple of minutes, before a GPU-hour is spent.")

    p.add_argument("--runs-root", default="/kaggle/working/runs",
                   help="Parent of the per-run directories. Outside the clone on "
                        "purpose: prune_workdir() clears the clone at the end of "
                        "a successful session and /kaggle/working is what Kaggle "
                        "saves as kernel output.")
    p.add_argument("--data-module", default="src.data.adapter")
    p.add_argument("--data-class", default="SRPatchDataset")
    p.add_argument("--data-root", default="auto")

    p.add_argument("--iters", type=int, default=12000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patch-lr", type=int, default=64)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--in-ch", type=int, default=4)
    p.add_argument("--n-resblocks", type=int, default=16)
    p.add_argument("--n-feats", type=int, default=48)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=25)
    p.add_argument("--metric-batches", type=int, default=4)
    p.add_argument("--hf-cutoff", type=float, default=0.25)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--resume", default="auto")
    p.add_argument("--spectral-downsample", default="area")

    # -- the wall clock ------------------------------------------------------
    p.add_argument("--budget-hours", type=float, default=8.0,
                   help="Wall-clock hours available TO THIS SCRIPT. Kaggle kills "
                        "a GPU session at 9; the difference is the clone, the "
                        "pip installs, the data guard and Kaggle's own output "
                        "commit, none of which this script's clock can see.")
    p.add_argument("--a2-cut-hours", type=float, default=3.5,
                   help="If the CONTROL alone exceeds this, B1's iterations are "
                        "cut to fit and B2 is abandoned. A truncated B1 that "
                        "checkpoints beats a complete B1 killed at the limit.")
    p.add_argument("--b2-margin", type=float, default=2.5,
                   help="B2 starts only if this multiple of the median observed "
                        "run duration still remains.")
    p.add_argument("--skip-b2", action="store_true",
                   help="Abandon B2 regardless of the clock.")
    p.add_argument("--require-clean", dest="require_clean", action="store_true",
                   default=True,
                   help="Refuse to start from a dirty tree (the default). The "
                        "kernel checks out a SHA, so uncommitted work does not "
                        "exist to it and a dirty tree means the recorded commit "
                        "does not describe what ran.")
    p.add_argument("--allow-dirty", dest="require_clean", action="store_false",
                   help="Run from a dirty tree. Local development only; the "
                        "dirty flag is still recorded in every run's metadata.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config, smoke=bool(args.smoke))
    logger = get_logger("day3.runs", log_file=cfg["paths"]["log_file"])
    seed_everything(cfg["seed"])

    if args.smoke:
        # Every path, none of the cost. Small enough that the whole three-run
        # orchestration -- including both guards and the manifest -- completes
        # on a CPU box in a couple of minutes, which is what makes it worth
        # running before a session.
        args.iters, args.val_every, args.ckpt_every = 4, 2, 2
        args.val_batches, args.metric_batches, args.log_every = 1, 1, 2
        args.batch, args.patch_lr, args.workers = 2, 16, 0
        args.amp, args.resume = 0, "none"
        args.budget_hours, args.a2_cut_hours = 0.5, 0.5
        args.require_clean = False
        args.runs_root = str(repo_root() / "outputs" / "day3_smoke")
        logger.info(
            "SMOKE: %d iters per run into %s, and --require-clean is OFF. "
            "Nothing produced here is a result.",
            args.iters, args.runs_root,
        )

    # ---- provenance, before anything expensive ----------------------------
    frozen = load_frozen(FROZEN_CONFIG)
    config_hash = str(frozen[HASH_FIELD])
    git = git_metadata()
    logger.info("frozen config %s verified: %s", FROZEN_CONFIG, config_hash)
    logger.info(
        "code: commit %s branch %s dirty=%s",
        git["commit"], git["branch"], git["dirty"],
    )

    if git["commit"] == NO_GIT:
        raise Day3Error(
            "no git repository, so the commit these runs used cannot be "
            "recorded. The Kaggle kernel clones one; if this is a session "
            "unpacked from a zip, the runs it produces are untraceable."
        )
    if args.require_clean and git["dirty"] is not False:
        raise Day3Error(
            f"the working tree is dirty (dirty={git['dirty']}, "
            f"{len(git['dirty_files'])} file(s): "
            f"{', '.join(git['dirty_files'][:5])}"
            f"{' ...' if len(git['dirty_files']) > 5 else ''}). The Kaggle "
            "kernel checks out a SHA and uncommitted work does not exist to "
            "it, so a dirty tree means the recorded commit does not describe "
            "what ran. Commit and push, then re-push the kernel. Use "
            "--allow-dirty only for local development."
        )

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    # ---- the argument lists, and the proof they agree ---------------------
    shared = build_shared_args(args)
    planned = {
        spec["name"]: build_run_args(
            shared, spec, runs_root / spec["name"], args.iters, args.budget_hours
        )
        for spec in RUN_PLAN
    }
    assert_runs_differ_only_in_lambdas(planned)
    logger.info(
        "argument lists verified: the %d runs differ only in "
        "--spectral-lambda1/--spectral-lambda2 (and --out/--iters/--max-hours).",
        len(planned),
    )
    for name, values in planned.items():
        logger.info("  %-3s %s", name, " ".join(values))

    # ---- run them ---------------------------------------------------------
    session_start = time.time()
    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    iters_for_next = int(args.iters)

    for spec in RUN_PLAN:
        hours_used = (time.time() - session_start) / 3600.0
        hours_left = args.budget_hours - hours_used

        if spec["name"] == "b2":
            if args.skip_b2:
                reason = "--skip-b2 was passed."
            elif iters_for_next < args.iters:
                reason = (
                    f"B1 was already cut to {iters_for_next} iterations because "
                    f"the control overran --a2-cut-hours; a third run cannot fit."
                )
            elif not should_run_b2(
                [record["elapsed_h"] for record in records],
                hours_left,
                args.b2_margin,
            ):
                median = statistics.median(
                    [record["elapsed_h"] for record in records]
                ) if records else 0.0
                reason = (
                    f"{hours_left:.2f} h left is under {args.b2_margin} x the "
                    f"{median:.2f} h median run."
                )
            else:
                reason = ""
            if reason:
                # Loud and unmissable. A quietly absent third run reads as a
                # crash to whoever fetches the outputs.
                logger.warning("")
                logger.warning("#" * 70)
                logger.warning("# B2 SKIPPED -- %s", reason)
                logger.warning("# A2 and B1 are the deliverable; B2 was upside.")
                logger.warning("#" * 70)
                skipped.append({"name": spec["name"], "reason": reason})
                continue

        out_dir = runs_root / spec["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        arg_list = build_run_args(
            shared, spec, out_dir, iters_for_next, max(0.05, hours_left)
        )
        records.append(run_one(args, spec, arg_list, logger))

        if spec["name"] == "a2" and records[-1]["elapsed_h"] > args.a2_cut_hours:
            # The control overran. Cut B1 to what is left rather than starting a
            # full-length run that will be killed before it checkpoints.
            hours_left_after = args.budget_hours - (
                (time.time() - session_start) / 3600.0
            )
            iters_for_next = cut_iters(
                args.iters,
                records[-1]["elapsed_h"],
                hours_left_after,
                args.ckpt_every,
            )
            logger.warning("")
            logger.warning("#" * 70)
            logger.warning(
                "# A2 took %.2f h, over --a2-cut-hours=%.2f. B1 is CUT to %d of "
                "%d iterations to fit the %.2f h left, and B2 is abandoned.",
                records[-1]["elapsed_h"], args.a2_cut_hours,
                iters_for_next, args.iters, hours_left_after,
            )
            logger.warning(
                "# B1's curve is therefore shorter than A2's. Compare them at "
                "the iterations they SHARE, not at their endpoints."
            )
            logger.warning("#" * 70)

    # ---- the runs must be the same experiment -----------------------------
    assert_same_experiment(records)
    logger.info(
        "provenance unanimous across %d run(s): config_hash=%s commit=%s",
        len(records), records[0]["config_hash"], records[0]["git_sha"],
    )

    report_blur(records, logger)

    manifest = {
        "config_hash": config_hash,
        "frozen_config": FROZEN_CONFIG,
        "git": git,
        "budget_hours": float(args.budget_hours),
        "session_elapsed_h": (time.time() - session_start) / 3600.0,
        "iters_requested": int(args.iters),
        "shared_args": shared,
        "runs": records,
        "skipped": skipped,
        "smoke": bool(args.smoke),
    }
    manifest_path = runs_root / "day3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("")
    logger.info("wrote %s", manifest_path)
    for record in records:
        logger.info(
            "  %-3s lambdas %s/%s  %.2f h  best val PSNR %.3f dB",
            record["name"], record["lambda1"], record["lambda2"],
            record["elapsed_h"], record["best_val_psnr"] or float("nan"),
        )
    for entry in skipped:
        logger.info("  %-3s SKIPPED -- %s", entry["name"], entry["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
