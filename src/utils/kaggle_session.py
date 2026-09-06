"""Everything a Kaggle notebook is allowed to do.

Notebooks in this project contain no logic -- they import and call (see
``AGENTS.md``). This module is where that logic lives, so a cell reads as one
function call and the behaviour behind it is version-controlled, reviewable, and
testable on CPU locally rather than only observable in a Kaggle log.

Two notebooks use it, and they are different shapes:

**The headless job**, generated from ``notebooks/templates/kaggle_job.py`` by
``scripts/kaggle_run.py``. Six steps, in order:

1. :func:`pip_install`            -- add the few packages the Kaggle image lacks.
2. :func:`stage_supporting_files` -- copy the manifest and split CSVs out of the
   mounted dataset, so the split is READ rather than recomputed.
3. :func:`guard_data_root`        -- prove the session can see its data, or abort.
4. :func:`run_entry`              -- run one ``scripts/*.py`` entry point.
5. :func:`inventory_outputs`      -- say what the run actually produced.
6. :func:`prune_workdir`          -- leave results, not a copy of the source tree.

Step 2 comes before step 3 on purpose: the guard asserts those two files are in
place, so staging has to have happened for the guard to be able to check it.

**The interactive Day 1 notebook**, ``notebooks/01_day1_baseline.ipynb``, run in
a browser tab with a human watching. It runs several entry points in sequence
rather than one, so it adds:

6. :func:`environment_report`     -- the go/no-go facts about this session.
7. :func:`run_stage`              -- one named stage, timed, with a failure hint.
8. :func:`dataset_fallback_hint`  -- the one-edit recipe for switching datasets.
9. :func:`stage_args`             -- the arguments every stage shares.
10. :func:`show_markdown`         -- put the Day 1 gate on screen.
11. :func:`archive_outputs`       -- one zip to download at the end.

Every step streams the child process's output line by line as it arrives, rather
than capturing it and printing at the end. A training run that prints nothing for
two hours is indistinguishable from a hung one, and being able to tell those
apart is the entire point of ``kaggle_run.py status --watch``.

Nothing here is Kaggle-specific beyond the defaults: it runs identically on the
local CPU box, which is how it gets tested without spending a GPU-hour.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SessionAborted",
    "pip_install",
    "supporting_file_names",
    "stage_supporting_files",
    "guard_data_root",
    "run_entry",
    "inventory_outputs",
    "prune_workdir",
    "environment_report",
    "run_stage",
    "dataset_fallback_hint",
    "archive_outputs",
    "stage_args",
    "show_markdown",
]

# Files a run is expected to produce, grouped so the closing inventory says
# "3 checkpoints, 2 metrics files, 5 figures" instead of listing 400 paths.
_KINDS = (
    ("checkpoints", (".pt", ".pth", ".ckpt", ".onnx", ".safetensors")),
    ("metrics", (".json", ".csv", ".md")),
    ("figures", (".png", ".jpg", ".jpeg", ".pdf", ".svg")),
    ("logs", (".log", ".txt")),
)

# Ceiling on the recursive entry count in :func:`environment_report`. The mounted
# cache is ~6000 files; an environment check exists to be read in seconds, not to
# walk an arbitrarily large tree. A count that reaches this is printed with a "+".
_ENTRY_COUNT_CAP = 20000

# Package versions recorded by :func:`environment_report`. This is not a
# dependency list -- it is the list of packages whose version has already broken
# something on this project, so that the number is in the saved output of every
# run rather than reconstructed afterwards from a pip log nobody kept.
#
#   numpy, opencv-python -- installing satalign pulled opencv 5.x, which requires
#       numpy >= 2, which broke scipy. See "Environment, pinned by hand" in
#       reports/day1_gate.md.
#   opensr-test, satalign -- the stage-4 benchmark and its registration backend.
#   torch is reported separately, from the imported module.
_REPORTED_PACKAGES = (
    "numpy",
    "scipy",
    "opencv-python",
    "opensr-test",
    "satalign",
    "omegaconf",
    "tacoreader",
    "lpips",
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


def pip_install(
    packages: Sequence[str],
    note: str = "",
    extra_args: Sequence[str] = (),
) -> None:
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
        extra_args: Extra pip flags for this call, applied to every package in
            it. ``["--no-deps"]`` is the one this project actually needs:
            ``opensr-test`` pulls ``open-clip-torch`` and ``openai-clip`` for a
            correctness distance we do not use, and installing them moves numpy
            past what the pinned torch and scipy accept (MEASURED locally; see
            the "Environment, pinned by hand" section of ``reports/day1_gate.md``).
            Split such packages into their own call rather than weakening the
            flags for everything.

    Raises:
        SessionAborted: pip exited non-zero. The run stops here rather than
            failing later with an ImportError that looks like a code bug.
    """
    if not packages:
        print("pip: nothing to install for this job.", flush=True)
        return
    flags = " ".join(extra_args)
    print(
        f"pip: installing {len(packages)} package(s) {flags} {note}".replace("  ", " ").rstrip(),
        flush=True,
    )
    code = _stream(
        [sys.executable, "-m", "pip", "install", "--quiet", *extra_args, *packages],
        "pip",
    )
    if code != 0:
        raise SessionAborted(
            f"pip install failed (exit code {code}) for: {', '.join(packages)}\n"
            "The run is stopping here. Letting it continue would fail later with "
            "an ImportError that reads like a bug in src/ rather than a missing "
            "dependency. Fix the 'pip_packages' list for this job in "
            "configs/kaggle_jobs.yaml."
        )


def supporting_file_names(cfg: Any) -> List[str]:
    """The manifest and split CSV file names for the configured dataset.

    Derived from config, never spelled out: the dataset name comes from
    ``cfg.dataset.name`` and the split file from ``cfg.splits.output_name``, so
    switching datasets moves these with it. ``scripts/kaggle_upload.py``
    (``copy_supporting_files``) stages the same two names into the Kaggle
    Dataset, and the two must agree -- ``tests/test_kaggle_session.py`` pins that.

    Args:
        cfg: The loaded config.

    Returns:
        ``["manifest_<dataset>.csv", "splits_<dataset>.csv"]``, in that order.

    Raises:
        KeyError: ``cfg`` lacks ``dataset.name`` or ``splits.output_name``.
    """
    dataset_name = str(cfg["dataset"]["name"])
    return [
        f"manifest_{dataset_name}.csv",
        str(cfg["splits"]["output_name"]).format(dataset=dataset_name),
    ]


def stage_supporting_files(
    config_path: str,
    repo_dir: Optional[str] = None,
) -> List[str]:
    """Copy the manifest and split CSVs out of the mounted dataset into outputs/.

    THE FAILURE THIS PREVENTS. The Kaggle Dataset carries the pixels *and* the
    two small CSVs that say which samples were validated and which are
    train/val/test. Nothing used to copy them into the working directory, so a
    job ran with the imagery present and the split file absent -- and the loader
    responds to a missing split file by recomputing one in-process. That is
    reproducible and completely wrong: adjacent NAIP tiles overlap, so a
    recomputed split is not the geographic split the baseline was measured on,
    and the numbers stop being comparable without anything announcing it.

    The mount path is resolved through :func:`src.utils.paths.kaggle_mount_path`,
    never composed here. That layout has already been wrong once in this project
    -- the real mount is ``/kaggle/input/datasets/<owner>/<slug>``, not the
    ``/kaggle/input/<slug>`` the code originally assumed -- and a second copy of
    that assumption is exactly how the fix would come undone.

    Missing files abort the run. A job that proceeds without them produces
    numbers that look fine and mean something different, which is worse than not
    running at all.

    Args:
        config_path: Path to the YAML config, relative to ``repo_dir``. The same
            config the job's entry point is given, so both agree on the dataset
            name and therefore on the file names.
        repo_dir: Repository root. None means the current working directory,
            which is where the generated notebook has already ``chdir``'d.

    Returns:
        The names copied, in the order of :func:`supporting_file_names`. An
        already-present file is left alone and still listed.

    Raises:
        SessionAborted: No mount is configured or present, or either CSV is
            absent from it. The message names the mount that was searched, the
            files that were expected, and what is actually in the mount's top
            level, because "not found" without those three is not diagnosable
            from a Kaggle log.
    """
    root = Path(repo_dir) if repo_dir else Path.cwd()
    # Imported here rather than at module scope: this module is imported by the
    # notebook's FIRST cell, before the repo's own dependencies are pip
    # installed, and omegaconf is one of the things being installed.
    sys.path.insert(0, str(root))
    from src.utils.config import load_config
    from src.utils.paths import kaggle_mount_path

    cfg = load_config(root / config_path)
    wanted = supporting_file_names(cfg)

    # The destination is cfg.paths.manifest_dir anchored at ``root``, NOT at
    # resolve_output_path()'s repo_root(). The two are the same thing on Kaggle,
    # where the notebook has chdir'd into the clone that this module was
    # imported from -- but repo_root() is derived from this file's own location,
    # so it ignores ``repo_dir`` entirely. That difference is not academic: it
    # let a test that passed a temporary repo overwrite the real
    # outputs/manifest_sen2naipv2.csv. Writing where the caller said to write is
    # both safer and the only version that can be tested.
    configured = Path(str(cfg["paths"]["manifest_dir"]))
    destination = configured if configured.is_absolute() else root / configured
    destination.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("  STAGING the manifest and split CSVs from the mounted dataset", flush=True)
    print("=" * 78, flush=True)

    mount = kaggle_mount_path(cfg)
    if mount is None:
        raise SessionAborted(
            "No Kaggle mount is configured, so the manifest and split CSVs "
            "cannot be staged.\n"
            "\n"
            "paths.kaggle_mount_root and paths.kaggle_dataset_dir must both be "
            "set in the config for a job that reads data. Without the split "
            "file the loader recomputes a split in-process, and a recomputed "
            "split is not the split the baseline numbers were measured on."
        )

    print(f"  mount:       {mount}", flush=True)
    print(f"  destination: {destination}", flush=True)

    if not mount.is_dir():
        raise SessionAborted(
            f"The Kaggle mount {mount} does not exist, so the manifest and "
            "split CSVs cannot be staged.\n"
            "\n"
            "The dataset is not attached to this notebook, or is attached under "
            "a name that differs from paths.kaggle_dataset_dir. Add it to this "
            "job's 'dataset_sources' in configs/kaggle_jobs.yaml and push again."
        )

    staged: List[str] = []
    missing: List[str] = []
    for name in wanted:
        source = mount / name
        if not source.is_file():
            missing.append(name)
            print(f"  ! MISSING from the mount: {name}", flush=True)
            continue
        target = destination / name
        shutil.copy2(source, target)
        staged.append(name)
        size = target.stat().st_size
        print(f"  + {name}  ({size:,} bytes)", flush=True)

    if missing:
        present = sorted(item.name for item in mount.iterdir() if item.is_file())
        listing = "\n".join(f"    {item}" for item in present[:20]) or "    (no files)"
        if len(present) > 20:
            listing += f"\n    ... and {len(present) - 20} more"
        raise SessionAborted(
            f"Cannot stage {len(missing)} required file(s) from the mounted "
            f"dataset: {', '.join(missing)}.\n"
            "\n"
            f"Searched: {mount}\n"
            f"Top-level files actually there:\n{listing}\n"
            "\n"
            "The run stops here on purpose. Without the split CSV the loader "
            "recomputes a split in-process; that split is reproducible but is "
            "NOT the geographic split the existing baseline was measured on, so "
            "every number the job produced would be quietly incomparable.\n"
            "\n"
            "Fix it at the source, by re-uploading the dataset with the CSVs in "
            "it:\n"
            "  python scripts/make_splits.py --config configs/base.yaml\n"
            "  python scripts/kaggle_upload.py stage\n"
            "  python scripts/kaggle_upload.py push\n"
            "then re-run this job."
        )

    print(f"\n  Staged {len(staged)} file(s). The split will be READ, not recomputed.", flush=True)
    return staged


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


def prune_workdir(keep: Sequence[str], repo_dir: Optional[str] = None) -> None:
    """Delete everything in the working directory except the output folders.

    Kaggle saves ``/kaggle/working`` as the kernel's output, and the job clones
    this repository into it -- so without this step every ``fetch`` drags a full
    copy of the source tree back alongside the checkpoints, and a local ``pytest``
    then finds two copies of every test module. MEASURED: exactly that happened
    on the first real run.

    Runs last, after the job has succeeded and the inventory has been printed.
    Only paths inside ``repo_dir`` are touched, and only its top-level entries;
    the code is already committed in git, so nothing unique is being deleted --
    the clone is a disposable copy of a pushed commit.

    Args:
        keep: Top-level names to preserve, normally the job's ``output_dirs``.
        repo_dir: The cloned working directory. None means the current one.

    Raises:
        SessionAborted: A path escaped ``repo_dir``, or a deletion failed. This
            fails the run rather than leaving output in an unknown state; the
            job's results are still saved and fetchable either way.
    """
    base = (Path(repo_dir) if repo_dir else Path.cwd()).resolve()
    keep_names = {Path(name).parts[0] for name in keep if str(name).strip()}
    print(f"\nPruning {base}, keeping: {', '.join(sorted(keep_names)) or '(nothing)'}", flush=True)

    removed = 0
    for child in sorted(base.iterdir()):
        if child.name in keep_names:
            continue
        resolved = child.resolve()
        if base not in resolved.parents:
            raise SessionAborted(
                f"Refusing to delete {resolved}: it is not inside {base}. "
                "This is a bug in the generated notebook, not in your job."
            )
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            raise SessionAborted(
                f"Could not remove {child} while pruning the working directory: "
                f"{exc}\nThe job itself succeeded and its outputs are saved -- "
                "fetch them and ignore this failure."
            ) from exc
        removed += 1

    print(f"Removed {removed} top-level entries; only outputs remain.", flush=True)


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


# ---------------------------------------------------------------------------
# The interactive Day 1 notebook (notebooks/01_day1_baseline.ipynb) adds four
# more calls on top of the five above. Same rule, same reason: the notebook is a
# shell, and the behaviour behind each call lives here, where it is
# version-controlled and testable on the local CPU box.
# ---------------------------------------------------------------------------


def _free_space(path: Path) -> Tuple[Optional[Path], Optional[int], Optional[int]]:
    """Free and total bytes on the filesystem holding ``path``.

    Walks up to the first ancestor that exists, because the interesting question
    on Kaggle is "how much room is there in /kaggle/temp" and the cache directory
    under it has usually not been created yet. The ancestor actually measured is
    returned, so the report can say which path the numbers describe rather than
    implying they came from one that does not exist.

    **The filesystem root is never accepted as an answer.** On the local Windows
    box, ``/kaggle/temp/...`` walks all the way up to ``\\``, whose free space is
    the D: drive's -- a real number, about a directory that is not there, printed
    under the label ``/kaggle/temp``. Excluding the anchor turns that into
    "absent", which is what it is.

    Args:
        path: Any path, existing or not.

    Returns:
        ``(measured_path, free_bytes, total_bytes)``, or ``(None, None, None)``
        when no ancestor below the root exists -- the honest answer on a machine
        with no ``/kaggle``. The caller reports it as "absent" rather than as
        zero free space.
    """
    anchor = Path(path.anchor) if path.anchor else None
    for candidate in [path, *path.parents]:
        if anchor is not None and candidate == anchor:
            break
        if candidate.exists():
            usage = shutil.disk_usage(candidate)
            return candidate, usage.free, usage.total
    return None, None, None


def _count_entries(path: Path, limit: int = _ENTRY_COUNT_CAP) -> int:
    """Number of entries under ``path``, recursively, stopping at ``limit``.

    The cap exists because the mounted cache is ~6000 files and a future dataset
    could be far larger; an environment check must not spend a minute walking a
    tree to answer "is there anything there". A count that reached the cap is
    printed with a trailing ``+`` by the caller.
    """
    seen = 0
    for _ in path.rglob("*"):
        seen += 1
        if seen >= limit:
            break
    return seen


def _package_version(name: str) -> Optional[str]:
    """Installed version of ``name``, or None when it is not installed.

    ``importlib.metadata`` reads the installed distribution's metadata, so it
    reports the version of what pip actually resolved -- which is the question
    worth asking on Kaggle, where the base image's numpy is not the one
    ``requirements.txt`` pins. Absence is a normal, reportable answer here, not
    an error, which is why ``PackageNotFoundError`` is caught and only that.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_report(
    config: Optional[str] = None,
    disk_paths: Optional[Sequence[str]] = None,
    packages: Sequence[str] = _REPORTED_PACKAGES,
) -> Dict[str, Any]:
    """Print everything about this session that could invalidate the run.

    This is the first cell of the Day 1 notebook and the only one whose product
    is text rather than a file. It answers, in one block: which interpreter is
    this, which torch, is there a GPU and which one, is there room on the scratch
    disk, and -- the question that has already cost this project a session --
    **which directory did the data root actually resolve to**.

    Nothing here is wrapped in a try/except that would let a surprising answer
    read as a normal one. A config that will not load, or a data root that
    resolves nowhere, raises out of this call and stops the notebook at cell one,
    which is much cheaper than discovering it four stages later.

    ``torch`` and ``omegaconf`` are imported inside the function rather than at
    module scope. The notebook's bootstrap cell imports this module *before* it
    pip installs omegaconf -- that ordering is what lets :func:`pip_install` be
    the thing that installs it -- so a module-level import here would break the
    very cell that exists to fix the gap.

    Args:
        config: Path to the YAML config, relative to the repository root. None
            means ``src.utils.config.DEFAULT_CONFIG``.
        disk_paths: Paths to report free space for. None means the two that
            matter on Kaggle, read from config rather than hardcoded:
            ``paths.kaggle_cache_dir`` (normally under ``/kaggle/temp``, which is
            fast and does **not** count against the output quota) and the
            resolved output root (under ``/kaggle/working``, which does).
        packages: Distribution names whose installed versions are recorded.
            Defaults to :data:`_REPORTED_PACKAGES` -- the ones that have already
            broken something here. A package that is not installed is reported as
            such rather than omitted.

    Returns:
        A dict of the facts that were printed -- ``python``, ``torch``,
        ``cuda_available``, ``cuda_device``, ``data_root``, ``data_root_entries``,
        ``cache_dir``, ``disks``, ... -- so a test can assert on values without
        parsing stdout and a caller can record them beside the run's metrics.

    Raises:
        FileNotFoundError: The config file, or the data root, does not resolve.
            ``resolve_data_root``'s message lists every path that was checked and
            why each one was rejected.
        ImportError: torch or omegaconf is missing, i.e. the bootstrap above this
            cell did not do its job.
    """
    # Imported here, not at module scope -- see the ordering note above.
    import torch

    from src.utils.config import DEFAULT_CONFIG, load_config
    from src.utils.paths import repo_root, resolve_cache_dir, resolve_data_root

    config_path = config or DEFAULT_CONFIG
    cfg = load_config(config_path)

    cuda_available = bool(torch.cuda.is_available())
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    capability = (
        ".".join(str(part) for part in torch.cuda.get_device_capability(0))
        if cuda_available
        else None
    )
    gpu_total_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if cuda_available
        else None
    )

    data_root = resolve_data_root(cfg)
    cache_dir = resolve_cache_dir(cfg)

    if disk_paths is None:
        disk_paths = [
            str(cfg.paths.kaggle_cache_dir),
            str(repo_root() / str(cfg.paths.output_root)),
        ]

    disks: List[Dict[str, Any]] = []
    for raw in disk_paths:
        measured, free, total = _free_space(Path(str(raw)))
        disks.append(
            {
                "requested": str(raw),
                "measured": None if measured is None else str(measured),
                "free_bytes": free,
                "total_bytes": total,
            }
        )

    entries = _count_entries(data_root)
    versions = {name: _package_version(name) for name in packages}

    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "cuda_capability": capability,
        "gpu_total_gb": gpu_total_gb,
        "config": str(config_path),
        "dataset": str(cfg.dataset.name),
        "data_root": str(data_root),
        "data_root_entries": entries,
        "cache_dir": str(cache_dir),
        "repo_root": str(repo_root()),
        "disks": disks,
        "packages": versions,
    }

    print("=" * 78, flush=True)
    print("  ENVIRONMENT -- if anything below is a surprise, stop here", flush=True)
    print("=" * 78, flush=True)
    print(f"  python         {report['python']}  ({report['executable']})", flush=True)
    print(f"  torch          {report['torch']}", flush=True)
    if cuda_available:
        sm = (capability or "").replace(".", "")
        print(
            f"  cuda           YES -- {cuda_device} (sm_{sm}, {gpu_total_gb:.1f} GB)",
            flush=True,
        )
        print(
            "                 Day 1 is CPU work. A GPU session here spends the "
            "weekly budget on interpolation.",
            flush=True,
        )
    else:
        print("  cuda           no -- CPU only, which is correct for Day 1.", flush=True)
    for disk in disks:
        if disk["measured"] is None:
            print(
                f"  disk           {disk['requested']}: ABSENT "
                "(no such path -- expected off Kaggle)",
                flush=True,
            )
            continue
        free_gb = disk["free_bytes"] / 1024**3
        total_gb = disk["total_bytes"] / 1024**3
        note = (
            ""
            if disk["measured"] == disk["requested"]
            else f"  [filesystem measured at {disk['measured']}]"
        )
        print(
            f"  disk           {disk['requested']}: {free_gb:.1f} GB free of "
            f"{total_gb:.1f} GB{note}",
            flush=True,
        )
    installed = ", ".join(
        f"{name} {version}" if version else f"{name} NOT INSTALLED"
        for name, version in versions.items()
    )
    print(f"  packages       {installed}", flush=True)
    print(f"  config         {report['config']}", flush=True)
    print(f"  dataset.name   {report['dataset']}", flush=True)
    capped = "+" if entries >= _ENTRY_COUNT_CAP else ""
    print(
        f"  data root      {report['data_root']}  ({entries}{capped} entries)",
        flush=True,
    )
    print(f"  cache dir      {report['cache_dir']}", flush=True)
    print(f"  repo root      {report['repo_root']}", flush=True)
    print("=" * 78, flush=True)

    if entries == 0:
        print(
            "  WARNING: the data root resolved but is empty. Every stage below "
            "would report zero samples and finish green. Attach the cache "
            "dataset before continuing.",
            flush=True,
        )
    return report


def _elapsed(seconds: float) -> str:
    """Format a duration as ``H:MM:SS``, or ``MM:SS`` when under an hour."""
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def run_stage(
    name: str,
    steps: Sequence[Tuple[str, Sequence[str]]],
    on_failure: str = "",
    repo_dir: Optional[str] = None,
) -> float:
    """Run one named stage -- one or more entry points -- with timings.

    :func:`run_entry` runs a single script and says whether it worked.
    ``run_stage`` is what the interactive notebook calls instead: it stamps the
    wall-clock time at the start, times each step, and prints the stage total. In
    a session with a hard time limit, "the baseline took 38 minutes" is the
    number that decides whether the next stage fits, and it has to be in the
    saved output rather than inferred from when somebody happened to look at the
    screen.

    Args:
        name: Human label for the stage, e.g. ``"2/5  alignment QA"``.
        steps: ``(script, args)`` pairs, run in order; the first failure stops
            the stage. A stage holds more than one step only when the steps are
            not separately re-runnable -- indexing and splitting the same cache,
            say.
        on_failure: Extra guidance printed **before** the exception when a step
            fails. This is where the notebook's fallback recipe goes, so the
            instruction appears in the failure output itself rather than in a
            markdown cell that has scrolled off screen.
        repo_dir: Working directory for the steps. None means the current one.

    Returns:
        Total wall-clock seconds for the stage.

    Raises:
        SessionAborted: A step exited non-zero. Raised rather than returned, so
            the notebook stops at the failing stage instead of running four more
            against data that was never prepared.
    """
    started = time.monotonic()
    stamp = _dt.datetime.now(_dt.timezone.utc)
    print("=" * 78, flush=True)
    print(f"  STAGE: {name}", flush=True)
    print(
        f"  started {stamp.isoformat(timespec='seconds')}, {len(steps)} step(s)",
        flush=True,
    )
    print("=" * 78, flush=True)

    for index, (script, args) in enumerate(steps, start=1):
        step_started = time.monotonic()
        code = _stream(
            [sys.executable, str(script), *[str(arg) for arg in args]],
            f"step {index} of stage {name!r}",
            cwd=Path(repo_dir) if repo_dir else None,
        )
        step_elapsed = time.monotonic() - step_started
        if code != 0:
            print(
                f"\n  step {index}/{len(steps)} FAILED after {_elapsed(step_elapsed)}",
                flush=True,
            )
            if on_failure:
                print("\n" + "!" * 78, flush=True)
                print(on_failure.rstrip("\n"), flush=True)
                print("!" * 78, flush=True)
            raise SessionAborted(
                f"Stage {name!r} failed: {script} exited with code {code} after "
                f"{_elapsed(step_elapsed)}. Its traceback is immediately above."
            )
        print(
            f"\n  step {index}/{len(steps)} ok -- {script} in {_elapsed(step_elapsed)}",
            flush=True,
        )

    total = time.monotonic() - started
    finished = _dt.datetime.now(_dt.timezone.utc)
    print(
        f"\n  STAGE {name!r} COMPLETE in {_elapsed(total)} "
        f"(finished {finished.isoformat(timespec='seconds')})",
        flush=True,
    )
    return total


def dataset_fallback_hint(
    dataset: str,
    variable: str = "DATASET",
    stage: str = "prepare data",
) -> str:
    """The recipe for switching datasets, printed when data preparation fails.

    Day 1 has exactly one fallback and it has to be one edit. SEN2NAIPv2 reaches
    its catalog over the network and its cache arrives as an attached Kaggle
    Dataset; both can be unavailable in ways that are nobody's fault and that no
    amount of retrying inside the session will fix. WorldStrat is read from a
    mounted public dataset instead, so it fails differently -- which is the whole
    value of keeping it.

    The message names the variable to change and the cell to restart from,
    because the alternative, "switch to the fallback dataset", is a message that
    requires its reader to already know what this function knows.

    Args:
        dataset: The fallback ``dataset.name`` value, e.g. ``"worldstrat"``.
        variable: The notebook variable carrying the override.
        stage: Name of the cell to re-run from.

    Returns:
        The message, ready to pass to :func:`run_stage` as ``on_failure``.
    """
    return (
        "  DATA PREPARATION FAILED. This is the one failure with a planned\n"
        "  fallback, and taking it is a single edit:\n"
        "\n"
        "    1. Scroll up to the CONFIGURATION cell.\n"
        f'    2. Change   {variable} = None   to   {variable} = "{dataset}"\n'
        "       Every stage below passes it through as\n"
        f"         --set dataset.name={dataset}\n"
        "       so nothing else in the notebook changes.\n"
        f"    3. Re-run the configuration cell, then this {stage!r} cell, then\n"
        "       each cell below it in order.\n"
        "\n"
        "  Read the traceback above first. If it names a missing mount or an\n"
        "  empty data root, attaching the right dataset is the real fix and\n"
        f"  {dataset!r} will fail the same way -- it needs a mount too. Switch\n"
        "  when the failure is specific to the current dataset: its catalog is\n"
        "  unreachable, its download is rate-limited, or its cache came back\n"
        "  short."
    )


def archive_outputs(
    directories: Sequence[str],
    dest: str,
    repo_dir: Optional[str] = None,
    skip: Sequence[str] = (),
) -> Path:
    """Zip the run's artefacts into one file to download from the session.

    Kaggle's file browser downloads one file at a time, and a Day 1 run leaves a
    few hundred files across ``outputs/metrics``, ``outputs/figures`` and
    ``reports``. One archive is one click.

    The archive is written **outside** every directory it packs, and that is
    checked rather than assumed: a zip that contains itself grows until the disk
    does not, and the failure lands at the end of a long session.

    Args:
        directories: Directories to pack, relative to ``repo_dir``. A missing one
            is named in the output and skipped -- for this notebook a stage that
            was not run is a normal state, unlike an empty archive.
        dest: Path for the ``.zip``, relative to ``repo_dir``. On Kaggle it
            belongs directly under ``/kaggle/working``.
        repo_dir: Base directory. None means the current one.
        skip: Sub-paths to leave out, relative to ``repo_dir``, each excluding
            itself and everything under it. ``outputs/`` holds the sample cache
            and the upload staging folders as well as the results, and those are
            gigabytes: MEASURED locally, archiving ``outputs`` wholesale packs the
            ~3.9 GB SEN2NAIPv2 cache and a second staged copy of it. On Kaggle
            the cache lives in ``/kaggle/temp`` or on a read-only mount, so this
            usually excludes nothing there -- which is exactly why it has to be
            explicit rather than inferred from what happens to be present.

    Returns:
        The absolute path to the archive.

    Raises:
        SessionAborted: ``dest`` sits inside one of ``directories``, or every
            directory was missing or empty. An archive with nothing in it is
            worse than an error: it downloads, opens, and looks like a result.
    """
    base = (Path(repo_dir) if repo_dir else Path.cwd()).resolve()
    excluded = [(base / entry).resolve() for entry in skip]
    archive = Path(dest)
    if not archive.is_absolute():
        archive = base / archive
    archive = archive.resolve()

    present: List[Tuple[str, Path]] = []
    for entry in directories:
        path = (base / entry).resolve()
        if archive == path or path in archive.parents:
            raise SessionAborted(
                f"Refusing to write {archive} inside {path}, which is one of the "
                "directories being archived: the zip would contain itself. Point "
                "`dest` at a path outside every entry of `directories` -- on "
                "Kaggle, directly under /kaggle/working."
            )
        if not path.is_dir():
            print(f"  skipping {entry}: not present (stage not run?)", flush=True)
            continue
        present.append((entry, path))

    print(f"\nArchiving {len(present)} directory(ies) into {archive}", flush=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    raw_bytes = 0
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for entry, path in present:
            packed_here = 0
            excluded_here = 0
            for item in sorted(path.rglob("*")):
                if not item.is_file():
                    continue
                resolved = item.resolve()
                if any(ex == resolved or ex in resolved.parents for ex in excluded):
                    excluded_here += 1
                    continue
                bundle.write(item, arcname=str(Path(entry) / item.relative_to(path)))
                packed_here += 1
                raw_bytes += item.stat().st_size
            written += packed_here
            note = f", {excluded_here} excluded by `skip`" if excluded_here else ""
            print(f"  {entry}: {packed_here} file(s){note}", flush=True)

    if written == 0:
        archive.unlink()
        raise SessionAborted(
            f"Nothing to archive: {', '.join(directories)} held no files, so "
            f"{archive} was deleted rather than left as an empty download. The "
            "stages above either did not run or wrote somewhere else."
        )

    packed = archive.stat().st_size
    print(
        f"\n  {archive}\n"
        f"  {written} file(s), {raw_bytes / 1024 / 1024:.2f} MB raw, "
        f"{packed / 1024 / 1024:.2f} MB compressed",
        flush=True,
    )
    print(
        "\n  Download it from the session's Output panel on the right, or with "
        "`python scripts/kaggle_run.py fetch` if this ran as a headless job.",
        flush=True,
    )
    return archive


def stage_args(
    config: str,
    dataset: Optional[str] = None,
    smoke: bool = False,
) -> List[str]:
    """Build the arguments every entry point in the Day 1 notebook shares.

    Three lines of ``if`` in a notebook cell is three lines of logic in a
    notebook, and this particular assembly carries the notebook's one fallback:
    ``dataset`` becomes ``--set dataset.name=<name>``, which is what makes
    switching to WorldStrat a single edit in a single cell rather than an edit
    per stage.

    ``--set dataset.name=`` is used rather than each script's own ``--dataset``
    flag because not every entry point has one -- ``scripts/make_day1_gate.py``
    reads ``cfg.dataset.name`` to find the split file and takes no ``--dataset``.
    One mechanism that works everywhere beats two that each work somewhere.

    Args:
        config: Path to the YAML config, relative to the repository root.
        dataset: Override for ``cfg.dataset.name``, or None to leave the config's
            own value alone.
        smoke: Add ``--smoke``, which merges the config's smoke block and puts
            every script on the synthetic stub -- a couple of minutes on CPU with
            no network and no mounted data. Use it to prove the notebook's wiring
            before spending a session on the real thing.

    Returns:
        The argument list, e.g. ``["--config", "configs/base.yaml", "--set",
        "dataset.name=worldstrat"]``. Printed as well as returned, because the
        exact arguments a stage ran under are the first thing anyone reading the
        saved output wants to confirm.
    """
    args = ["--config", str(config)]
    if smoke:
        args += ["--smoke"]
    if dataset:
        args += ["--set", f"dataset.name={dataset}"]

    print(f"Every stage below runs with: {' '.join(args)}", flush=True)
    if smoke:
        print(
            "  --smoke is ON: this is a wiring check on the synthetic stub. The "
            "numbers it produces are NOT results and must not reach the report.",
            flush=True,
        )
    if dataset:
        print(f"  dataset.name is overridden to {dataset!r} (the fallback path).", flush=True)
    return args


def show_markdown(path: str, repo_dir: Optional[str] = None) -> str:
    """Display a markdown artefact in the notebook, and return its text.

    ``scripts/make_day1_gate.py`` writes ``reports/day1_gate.md`` and prints only
    where it put it. The gate is the one thing this notebook exists to produce
    and the one thing the reader needs on screen before deciding whether the
    problem statement is viable, so the notebook shows it rather than asking
    somebody to download a zip to read the verdict.

    Rendered through IPython when it is available, which it is inside a notebook,
    and printed as plain text otherwise -- the presence of IPython is checked
    with :func:`importlib.util.find_spec` rather than by catching an ImportError,
    so a genuinely broken IPython still raises instead of quietly degrading.

    Args:
        path: Path to the markdown file, relative to ``repo_dir``.
        repo_dir: Base directory. None means the current one.

    Returns:
        The file's text, so a caller can assert on it.

    Raises:
        SessionAborted: The file does not exist. Naming the stage that writes it
            is more use than a bare FileNotFoundError, because the usual cause is
            a stage that was skipped rather than a wrong path.
    """
    base = Path(repo_dir) if repo_dir else Path.cwd()
    target = base / path
    if not target.is_file():
        raise SessionAborted(
            f"{target} does not exist. It is written by the final stage, "
            "scripts/make_day1_gate.py -- run that cell before this one."
        )

    text = target.read_text(encoding="utf-8")

    if importlib.util.find_spec("IPython") is not None:
        from IPython.display import Markdown, display

        display(Markdown(text))
        return text

    # Plain-text fallback, used off Kaggle. The report contains en dashes and
    # arrows, and a Windows console defaults to cp1252, which cannot encode them
    # -- MEASURED: printing the gate straight to a local terminal raised
    # UnicodeEncodeError and lost the whole document to a display problem. The
    # unrepresentable characters are escaped visibly rather than dropped, and the
    # file on disk is untouched and correct either way.
    encoding = sys.stdout.encoding or "utf-8"
    safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    if safe != text:
        print(
            f"  (console encoding is {encoding}; characters it cannot represent "
            f"are shown escaped. {target} itself is UTF-8 and intact.)",
            flush=True,
        )
    print(safe, flush=True)
    return text
