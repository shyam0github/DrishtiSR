"""The four things a generated Kaggle notebook is allowed to do.

``scripts/kaggle_run.py`` generates notebooks from a template, and the project
rule is that notebooks contain no logic -- they import and call. This module is
where that logic lives, so a generated cell reads as one function call and the
behaviour behind it is version-controlled, reviewable, and testable on CPU
locally rather than only observable in a Kaggle log.

The four steps, in the order a notebook runs them:

1. :func:`pip_install`       -- add the few packages the Kaggle image lacks.
2. :func:`guard_data_root`   -- prove the session can see its data, or abort.
3. :func:`run_entry`         -- run one ``scripts/*.py`` entry point.
4. :func:`inventory_outputs` -- say what the run actually produced.

Every step streams the child process's output line by line as it arrives, rather
than capturing it and printing at the end. A training run that prints nothing for
two hours is indistinguishable from a hung one, and being able to tell those
apart is the entire point of ``kaggle_run.py status --watch``.

Nothing here is Kaggle-specific beyond the defaults: it runs identically on the
local CPU box, which is how it gets tested without spending a GPU-hour.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

__all__ = [
    "SessionAborted",
    "pip_install",
    "guard_data_root",
    "run_entry",
    "inventory_outputs",
]

# Files a run is expected to produce, grouped so the closing inventory says
# "3 checkpoints, 2 metrics files, 5 figures" instead of listing 400 paths.
_KINDS = (
    ("checkpoints", (".pt", ".pth", ".ckpt", ".onnx", ".safetensors")),
    ("metrics", (".json", ".csv", ".md")),
    ("figures", (".png", ".jpg", ".jpeg", ".pdf", ".svg")),
    ("logs", (".log", ".txt")),
)


class SessionAborted(RuntimeError):
    """A generated notebook stopped on purpose, with the reason already stated.

    Raised rather than returning a status code because a Kaggle notebook that
    keeps executing after a failed step spends the rest of the session budget
    producing nothing. An exception marks the kernel run as failed, which is what
    ``kaggle_run.py status`` reports back.
    """


def _stream(command: Sequence[str], what: str, cwd: Optional[Path] = None) -> int:
    """Run a subprocess, echoing its output live, and return the exit code.

    Args:
        command: Full argv. Always starts with ``sys.executable`` for Python
            tooling -- never a bare console-script name, which on some machines
            resolves to a different interpreter than the one running this code.
        what: Short description used in error messages, e.g. ``"the data guard"``.
        cwd: Working directory, or None for the current one.

    Returns:
        The child's exit code.

    Raises:
        SessionAborted: The child process could not be started at all.
    """
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    try:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise SessionAborted(
            f"Could not start {what}: {exc}\nInterpreter: {sys.executable}"
        ) from exc

    if process.stdout is None:  # pragma: no cover -- guaranteed by stdout=PIPE
        raise SessionAborted(f"No output stream from {what}; cannot follow it.")
    for line in process.stdout:
        print(line.rstrip("\n"), flush=True)
    return process.wait()


def pip_install(packages: Sequence[str], note: str = "") -> None:
    """Install the packages the Kaggle base image does not already carry.

    Deliberately **not** ``pip install -r requirements.txt``. That file pins
    numpy, pandas, scipy and pillow to the versions this project was developed
    against; installing it on Kaggle downgrades packages the preinstalled PyTorch
    build was compiled against, which costs minutes of session time and can leave
    the image in a state where CUDA no longer imports. The per-job
    ``pip_packages`` list in ``configs/kaggle_jobs.yaml`` names only what is
    genuinely missing.

    Args:
        packages: Requirement specifiers, e.g. ``["omegaconf==2.3.0"]``. An empty
            sequence is a valid no-op and says so.
        note: Free text printed alongside, for the reader of the log.

    Raises:
        SessionAborted: pip exited non-zero. The run stops here rather than
            failing later with an ImportError that looks like a code bug.
    """
    if not packages:
        print("pip: nothing to install for this job.", flush=True)
        return
    print(f"pip: installing {len(packages)} package(s) {note}".rstrip(), flush=True)
    code = _stream([sys.executable, "-m", "pip", "install", "--quiet", *packages], "pip")
    if code != 0:
        raise SessionAborted(
            f"pip install failed (exit code {code}) for: {', '.join(packages)}\n"
            "The run is stopping here. Letting it continue would fail later with "
            "an ImportError that reads like a bug in src/ rather than a missing "
            "dependency. Fix the 'pip_packages' list for this job in "
            "configs/kaggle_jobs.yaml."
        )


def guard_data_root(
    script: str,
    args: Sequence[str] = (),
    enabled: bool = True,
    repo_dir: Optional[str] = None,
) -> None:
    """Prove the session can read its data, or abort before anything expensive.

    This guard exists because a Kaggle session that cannot see its mounted
    dataset does not crash -- it resolves an empty tree, reports zero samples,
    "trains" in seconds, and finishes green. On a GPU job that is an hour of the
    weekly 30-hour budget spent on nothing, and the log looks fine.

    ``scripts/verify_data_root.py`` opens real sample files and checks their
    shapes and reflectance ranges, so a non-zero exit here means the data is
    genuinely unreadable, not merely that a directory happened to be missing.

    Args:
        script: Path to the verification entry point, relative to ``repo_dir``.
        args: Extra arguments for it, e.g. ``["--config", "configs/base.yaml"]``.
        enabled: False only for the job whose entry point *is* the verifier,
            where running it twice would be pure duplication. Every other job
            passes True.
        repo_dir: Working directory for the check. None means the current one.

    Raises:
        SessionAborted: The check failed. The message names the likely fixes,
            because it is almost always a dataset that was not attached.
    """
    if not enabled:
        print("Data guard: skipped -- this job's entry point IS the data check.", flush=True)
        return

    print("=" * 78, flush=True)
    print("  DATA GUARD -- nothing expensive runs until this passes", flush=True)
    print("=" * 78, flush=True)
    code = _stream(
        [sys.executable, str(script), *[str(arg) for arg in args]],
        "the data guard",
        cwd=Path(repo_dir) if repo_dir else None,
    )
    if code != 0:
        raise SessionAborted(
            f"DATA GUARD FAILED (exit code {code}). This session cannot read its "
            "data, so the rest of the notebook has been abandoned on purpose.\n"
            "\n"
            "Almost always one of:\n"
            "  - the cache dataset is not attached to this notebook. Add it to "
            "'dataset_sources' for this job in configs/kaggle_jobs.yaml and "
            "push again;\n"
            "  - it is attached under a different name, so the mount path "
            "composed from paths.kaggle_dataset_dir does not exist;\n"
            "  - it is still syncing, or was uploaded empty.\n"
            "\n"
            "The output above lists every path that was checked and why each one "
            "was rejected. No GPU time has been spent on training."
        )
    print("\nData guard passed. Proceeding to the job.", flush=True)


def run_entry(
    script: str,
    args: Sequence[str] = (),
    repo_dir: Optional[str] = None,
) -> None:
    """Run one ``scripts/*.py`` entry point and fail loudly if it fails.

    Args:
        script: Path to the entry point, relative to ``repo_dir``.
        args: Its arguments, already split, e.g. ``["--config", "...", "--smoke"]``.
        repo_dir: Working directory. None means the current one.

    Raises:
        SessionAborted: The entry point exited non-zero. Re-raised as an
            exception so the Kaggle kernel run is marked failed; a notebook that
            swallowed this would report success for a run that produced nothing.
    """
    print("=" * 78, flush=True)
    print(f"  JOB: {script} {' '.join(str(arg) for arg in args)}".rstrip(), flush=True)
    print("=" * 78, flush=True)
    code = _stream(
        [sys.executable, str(script), *[str(arg) for arg in args]],
        f"the job entry point {script}",
        cwd=Path(repo_dir) if repo_dir else None,
    )
    if code != 0:
        raise SessionAborted(
            f"{script} exited with code {code}. The traceback is in the output "
            "above; pull it locally with "
            "`python scripts/kaggle_run.py logs --job <name>`."
        )
    print(f"\n{script} completed successfully.", flush=True)


def inventory_outputs(directories: Iterable[str], repo_dir: Optional[str] = None) -> None:
    """Print what the run produced, grouped by kind.

    Runs last, and never raises: by this point the job has already succeeded, and
    an inventory that could fail the run would turn a good result into a red
    kernel. An empty inventory is reported as a warning in the text, which is the
    signal worth seeing -- a job that "succeeded" and wrote nothing usually wrote
    to a path outside ``/kaggle/working``, and those results are gone when the
    session ends.

    Args:
        directories: Paths to inventory, relative to ``repo_dir``.
        repo_dir: Base directory. None means the current one.
    """
    base = Path(repo_dir) if repo_dir else Path.cwd()
    print("\n" + "=" * 78, flush=True)
    print("  OUTPUTS PRODUCED BY THIS RUN", flush=True)
    print("=" * 78, flush=True)

    total_files = 0
    total_bytes = 0
    for entry in directories:
        directory = base / entry
        if not directory.is_dir():
            print(f"  {entry}: not created by this run", flush=True)
            continue
        files = [path for path in directory.rglob("*") if path.is_file()]
        size = sum(path.stat().st_size for path in files)
        total_files += len(files)
        total_bytes += size
        print(f"  {entry}: {len(files)} file(s), {size / 1024 / 1024:.2f} MB", flush=True)
        for label, suffixes in _KINDS:
            matched = [path for path in files if path.suffix.lower() in suffixes]
            if not matched:
                continue
            names = ", ".join(sorted(path.name for path in matched)[:6])
            more = "" if len(matched) <= 6 else f", +{len(matched) - 6} more"
            print(f"      {label:12s} {len(matched):4d}  {names}{more}", flush=True)

    print(f"\n  TOTAL: {total_files} file(s), {total_bytes / 1024 / 1024:.2f} MB", flush=True)
    if total_files == 0:
        print(
            "\n  WARNING: this run produced no output files. If the job was "
            "supposed to write checkpoints or metrics, they went somewhere "
            "outside /kaggle/working and are lost when the session ends.",
            flush=True,
        )
    else:
        print(
            "\n  Download them with: python scripts/kaggle_run.py fetch --job <name>",
            flush=True,
        )
