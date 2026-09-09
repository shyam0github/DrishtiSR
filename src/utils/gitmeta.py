"""The code identity of a run, recorded next to its config identity.

WHY. ``configs/frozen_day3.yaml`` and its ``config_hash`` pin the
HYPERPARAMETERS of a run. They say nothing about the code, and Day 3 proved
that gap matters: the frozen-crop bug lived entirely in ``src/train.py``, so two
runs could carry the identical config hash and still see completely different
data streams. Run A2 (control) and Run B (fix) are only a valid comparison if
they ran the same code, and the only way to know that afterwards is to have
written it down at the time.

So every run records the commit SHA it ran from and whether the tree was dirty.
A dirty tree is not an error here -- it is normal mid-development -- but it IS
recorded, because "A2 and B were the same code" cannot be verified from a SHA
that had uncommitted edits behind it.

Fails soft ONLY on the absence of git itself (a Kaggle kernel unpacked from a
zip has no ``.git``), and says so explicitly in the recorded value rather than
returning something that looks like a SHA.
"""

from __future__ import annotations

import subprocess
from typing import Any, Dict, Optional

from src.utils.paths import repo_root

__all__ = ["NO_GIT", "git_metadata"]

# The recorded `commit` when no git repository is available. A sentinel, not a
# SHA: a comparison of two runs must be able to say "unknown", never quietly
# match two unknowns against each other.
NO_GIT = "unavailable"


def _git(*args: str, cwd: Any) -> Optional[str]:
    """Run one git command, returning stripped stdout or None if git cannot answer.

    Args:
        *args: Arguments after ``git``.
        cwd: Directory to run in.

    Returns:
        Stdout with TRAILING whitespace removed, or None when git is absent,
        the directory is not a repository, or the command failed. The caller
        turns None into an explicit recorded sentinel; nothing here guesses.

        Trailing only, not surrounding, and that is load-bearing for
        ``git status --porcelain``: its format is two status columns, a space,
        then the path, so the first line of an unstaged change begins with a
        SPACE (" M configs/base.yaml"). A leading strip removed it, and the
        caller's fixed ``line[3:]`` slice then ate the first character of the
        first filename -- "onfigs/base.yaml". Harmless-looking, and it landed in
        the provenance record of every run started from a dirty tree.
    """
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        # git not installed, or not on PATH. Distinguished from a git error by
        # the caller only in that both record NO_GIT; there is nothing to
        # recover, and raising would stop a Kaggle run over bookkeeping.
        return None
    if done.returncode != 0:
        return None
    return done.stdout.rstrip()


def git_metadata(root: Any = None) -> Dict[str, Any]:
    """Describe the working tree a run is starting from.

    Args:
        root: Repository directory. Defaults to
            :func:`src.utils.paths.repo_root`.

    Returns:
        A JSON-serialisable dict:

        - ``commit``: 40-character SHA, or :data:`NO_GIT`.
        - ``commit_short``: first 12 characters, or :data:`NO_GIT`.
        - ``dirty``: True when tracked files have uncommitted changes; None
          when there is no repository to ask. Never defaulted to False -- "we
          could not tell" and "it was clean" are different facts and the
          comparison of two runs depends on the difference.
        - ``branch``: branch name, ``"HEAD"`` when detached, or None.
        - ``dirty_files``: sorted paths reported by ``git status --porcelain``,
          capped at 50 so a run log stays readable. Empty when clean or
          unknown.
    """
    cwd = repo_root() if root is None else root

    commit = _git("rev-parse", "HEAD", cwd=cwd)
    if commit is None:
        return {
            "commit": NO_GIT,
            "commit_short": NO_GIT,
            "dirty": None,
            "branch": None,
            "dirty_files": [],
        }

    status = _git("status", "--porcelain", cwd=cwd)
    # Porcelain v1 is "XY<space><path>", so the path starts at column 3 for
    # every line -- including the leading-space form of an unstaged change.
    # This depends on _git preserving leading whitespace; see its docstring.
    changed = (
        sorted(line[3:] for line in status.splitlines() if line.strip())
        if status
        else []
    )
    return {
        "commit": commit,
        "commit_short": commit[:12],
        "dirty": bool(changed) if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd),
        "dirty_files": changed[:50],
    }
