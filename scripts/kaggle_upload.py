"""Publish the local SEN2NAIPv2 sample cache to Kaggle as a Dataset.

Why this exists
---------------
The cache is ~3000 LR/HR pairs that took roughly 8.7 hours to download
(MEASURED at ~10.4 s/sample). A Kaggle GPU session that re-downloaded them would
spend more than a quarter of the weekly 30-hour budget fetching data it could
have read from a mount in seconds. So the cache is uploaded once as a Kaggle
Dataset and attached to every notebook thereafter.

Everything here is a Python subcommand with plain-English output. There are no
shell one-liners to assemble, no JSON to hand-edit, and no raw stack traces: a
Kaggle CLI failure is caught and translated into what went wrong and what to do
about it. The ``kaggle`` CLI is still what talks to Kaggle -- it is the supported
client -- but you never have to type it.

The workflow, in order
----------------------
1. ``survey``  -- look at the local cache and say whether its size is sane.
2. ``stage``   -- build the upload folder (data + manifest + splits + README +
                  checksums), checking free disk space first.
3. ``dryrun``  -- upload 5 samples to a throwaway slug to prove the metadata,
                  the slug, and the folder handling all work.
4. ``upload``  -- the real thing, behind a typed YES confirmation.
5. ``version`` -- push an updated version of an existing dataset (re-uploads).

Every subcommand ends with a numbered "what happens next" block, so there is
always exactly one obvious next command.

Requires the Kaggle API token. Get it once from
https://www.kaggle.com/settings/account -> API -> "Create New Token", which
downloads ``kaggle.json``; put it at the path in ``cfg.kaggle.credentials_file``
(``~/.kaggle/kaggle.json``). Every subcommand that needs it checks for it first
and explains this if it is missing.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import (  # noqa: E402
    repo_root,
    resolve_cache_dir,
    resolve_output_path,
)
from src.utils.seed import seed_everything  # noqa: E402

GIB = 1024**3
MB = 1000**2

# Kaggle's own rules. Checked locally BEFORE any transfer starts, because
# discovering a 6-character title requirement after pushing 3.9 GB is the
# expensive way to learn it.
OK = "  [OK]  "
INFO = "         "
BAD = "  [!!]  "

SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SLUG_MIN, SLUG_MAX = 6, 50
TITLE_MIN, TITLE_MAX = 6, 50


class UploadError(RuntimeError):
    """A failure already translated into plain English.

    Raised with a message that says what happened, why, and what to do. ``main``
    prints it without a traceback -- a stack trace is the correct output for a
    bug in this script, and the wrong output for a missing API token.
    """


# -- formatting helpers ----------------------------------------------------


def human_bytes(count: float) -> str:
    """Format a byte count as a human-readable string (KB/MB/GB, base 1024)."""
    size = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or unit == "TB":
            return f"{size:,.2f} {unit}" if unit != "B" else f"{size:,.0f} B"
        size /= 1024.0
    return f"{size:,.2f} TB"


def human_duration(seconds: float) -> str:
    """Format a duration as ``1 h 23 min`` / ``4 min 5 s`` / ``12 s``."""
    total = int(max(0.0, seconds))
    if total < 60:
        return f"{total} s"
    if total < 3600:
        return f"{total // 60} min {total % 60} s"
    return f"{total // 3600} h {(total % 3600) // 60} min"


def next_steps(*lines: str) -> None:
    """Print the numbered 'what happens next' block that ends every subcommand."""
    print()
    print("WHAT HAPPENS NEXT")
    for index, line in enumerate(lines, start=1):
        print(f"  {index}. {line}")
    print()


def rule(title: str = "") -> None:
    """Print a section divider, optionally labelled."""
    print()
    print(f"--- {title} ".ljust(78, "-") if title else "-" * 78)


# -- credentials and slug validation ---------------------------------------


def credentials_path(cfg: Any) -> Path:
    """Absolute path to ``kaggle.json``, with ``~`` expanded.

    Args:
        cfg: The loaded config; reads ``kaggle.credentials_file``.

    Returns:
        The expanded path. Existence is not checked here -- see
        :func:`read_kaggle_username`.
    """
    return Path(str(cfg.kaggle.credentials_file)).expanduser()


def read_kaggle_username(cfg: Any) -> str:
    """Read the Kaggle username out of ``kaggle.json``.

    This is what removes ``dataset-metadata.json`` from the list of things you
    have to edit by hand: the owner slug is derived from the credentials that are
    already on the machine.

    Args:
        cfg: The loaded config.

    Returns:
        The ``username`` field from the credentials file.

    Raises:
        UploadError: The file is missing, unreadable, not JSON, or has no
            ``username`` -- each with the specific fix, since this is the single
            most common first-time failure and the raw symptom is an opaque
            ``401 - Unauthorized`` from the API.
    """
    path = credentials_path(cfg)
    if not path.is_file():
        raise UploadError(
            f"Kaggle API token not found at {path}.\n"
            "\n"
            "This is what causes the '401 - Unauthorized' error from Kaggle.\n"
            "To fix it, once:\n"
            "  1. Sign in at https://www.kaggle.com and open "
            "https://www.kaggle.com/settings/account\n"
            "  2. Scroll to the 'API' section and click 'Create New Token'.\n"
            "     Your browser downloads a small file called kaggle.json.\n"
            f"  3. Move that file to exactly this path:\n     {path}\n"
            f"     (create the {path.parent} folder if it does not exist)\n"
            "  4. Re-run this command.\n"
            "\n"
            "The file contains your API key. Do not commit it to the repository."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UploadError(
            f"Could not read {path}: {exc}\n"
            "The file should be the kaggle.json downloaded from your Kaggle "
            "account page, containing a single JSON object with 'username' and "
            "'key'. If you edited it, download a fresh one."
        ) from exc

    username = payload.get("username")
    if not username:
        raise UploadError(
            f"{path} has no 'username' field. It should look like:\n"
            '  {"username":"yourname","key":"0123456789abcdef..."}\n'
            "Download a fresh token from "
            "https://www.kaggle.com/settings/account -> API -> Create New Token."
        )
    return str(username)


def validate_slug(slug: str, key: str) -> str:
    """Check a dataset slug against Kaggle's rules, locally.

    Args:
        slug: The bare dataset name, with no owner prefix.
        key: The config key it came from, for the error message.

    Returns:
        The slug unchanged.

    Raises:
        UploadError: The slug contains an owner prefix, is the wrong length, or
            uses characters Kaggle rejects.
    """
    if "/" in slug:
        raise UploadError(
            f"cfg.{key} is {slug!r}, but it must be the dataset NAME only, with "
            "no 'username/' prefix. The owner is added automatically from your "
            "kaggle.json. Set it to just the name part, e.g. "
            f"{suggest_slug(slug.split('/')[-1])!r}."
        )
    if not SLUG_MIN <= len(slug) <= SLUG_MAX:
        raise UploadError(
            f"cfg.{key} is {slug!r} ({len(slug)} characters). Kaggle requires "
            f"{SLUG_MIN}-{SLUG_MAX} characters."
        )
    if not SLUG_PATTERN.match(slug):
        raise UploadError(
            f"cfg.{key} is {slug!r}, which Kaggle will reject. Use lowercase "
            "letters, digits, and single hyphens between them -- no spaces, "
            "underscores, capitals, or leading/trailing hyphens. "
            f"Suggested: {suggest_slug(slug)!r}."
        )
    return slug


def suggest_slug(text: str) -> str:
    """Turn arbitrary text into something Kaggle would accept, for error hints."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    cleaned = cleaned or "drishtisr-dataset"
    return (cleaned + "-dataset")[:SLUG_MAX] if len(cleaned) < SLUG_MIN else cleaned[:SLUG_MAX]


def validate_title(title: str, key: str) -> str:
    """Check a dataset title against Kaggle's length rules, locally.

    Raises:
        UploadError: Kaggle rejects titles shorter than 6 characters, and this
            is checked here so the failure costs a second rather than an upload.
    """
    if not TITLE_MIN <= len(title) <= TITLE_MAX:
        raise UploadError(
            f"cfg.{key} is {title!r} ({len(title)} characters). Kaggle requires "
            f"a title of {TITLE_MIN}-{TITLE_MAX} characters. Edit that value in "
            "configs/base.yaml."
        )
    return title


# -- the kaggle CLI --------------------------------------------------------


def kaggle_command() -> List[str]:
    """The command prefix that invokes the Kaggle CLI.

    **Always** ``[sys.executable, "-m", "kaggle"]`` -- never the bare ``kaggle``
    executable.

    MEASURED on this machine: the ``kaggle`` console script is not on the Windows
    PATH, so a bare invocation dies with "The term 'kaggle' is not recognized as
    the name of a cmdlet, function, script file, or operable program". Going
    through ``-m`` resolves the package inside the *same interpreter that is
    running this script*, which is the only interpreter guaranteed to have both
    the Kaggle client and this project's dependencies. It also removes a whole
    class of confusion where a ``kaggle.exe`` from a different Python is found
    first and reports a different set of installed packages.

    Returns:
        The argv prefix. Whether the package is actually importable is checked
        once, up front, by :func:`preflight` -- not here.
    """
    return [sys.executable, "-m", "kaggle"]


def run_kaggle(
    args: Sequence[str], timeout_s: int, logger: Any, what: str
) -> subprocess.CompletedProcess:
    """Run a Kaggle CLI command, translating any failure into plain English.

    Args:
        args: Arguments after the ``kaggle`` prefix, e.g.
            ``["datasets", "create", "-p", "..."]``.
        timeout_s: Seconds before the call is treated as hung.
        logger: Logger for the command line actually executed.
        what: Short description used in error messages, e.g. ``"the upload"``.

    Returns:
        The completed process, on success.

    Raises:
        UploadError: Anything went wrong, already translated by
            :func:`translate_kaggle_failure`.
    """
    command = kaggle_command() + list(args)
    logger.info("Running: %s", " ".join(str(part) for part in command))
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        raise UploadError(
            f"{what.capitalize()} timed out after {human_duration(timeout_s)} "
            "with no response from Kaggle.\n"
            "Most likely causes, in order:\n"
            "  - the connection dropped mid-transfer (the upload is NOT "
            "resumable; it has to start over);\n"
            "  - your upload speed is slower than the timeout allows -- raise "
            "cfg.kaggle.cli_timeout_s and try again;\n"
            "  - Kaggle is down; check https://status.kaggle.com\n"
            "Nothing on Kaggle was changed by a timed-out create."
        ) from exc
    except OSError as exc:
        raise UploadError(
            f"Could not start the Kaggle client for {what}: {exc}\n"
            f"Install it with `{sys.executable} -m pip install kaggle`."
        ) from exc

    if completed.returncode != 0:
        raise UploadError(translate_kaggle_failure(completed, what))
    return completed


def translate_kaggle_failure(
    completed: subprocess.CompletedProcess, what: str
) -> str:
    """Turn a Kaggle CLI failure into an explanation and a fix.

    The Kaggle client reports most problems as an HTTP status inside a long
    traceback. Handled specifically: 401 (bad or missing token), 403 (token
    valid but not accepted), 409 / "already exists" (duplicate slug), the
    title-length rejection, the slug-ownership rejection, and network errors.

    Args:
        completed: The failed process, with ``stdout``/``stderr`` captured.
        what: Short description of the operation, for the first line.

    Returns:
        A multi-line, plain-English message. Always ends with the raw output, so
        an unrecognised error is still fully reported rather than swallowed.
    """
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}".strip()
    lowered = output.lower()

    if "401" in lowered or "unauthorized" in lowered:
        explanation = (
            "Kaggle rejected your credentials (401 Unauthorized).\n"
            "  - Your kaggle.json is missing, expired, or from a different "
            "account.\n"
            "  - Fix: go to https://www.kaggle.com/settings/account -> API -> "
            "'Create New Token', and replace your kaggle.json with the new "
            "download. Tokens are invalidated when you create a new one, so an "
            "old copy on another machine will stop working."
        )
    elif "403" in lowered or "forbidden" in lowered:
        explanation = (
            "Kaggle accepted your token but refused the action (403 Forbidden).\n"
            "  - Most often this means you have not accepted the Kaggle terms, "
            "or your account is not yet phone-verified. Dataset creation "
            "requires a verified account.\n"
            "  - Fix: sign in at kaggle.com, open your profile settings, and "
            "complete phone verification."
        )
    elif "409" in lowered or "already exists" in lowered or "duplicate" in lowered:
        explanation = (
            "A dataset with this slug already exists on your account "
            "(409 Conflict).\n"
            "  - To REPLACE its contents with a new version, use:\n"
            "        python scripts/kaggle_upload.py version -m \"your message\"\n"
            "  - To create a separate dataset instead, change "
            "cfg.kaggle.dataset_slug in configs/base.yaml to an unused name."
        )
    elif "title" in lowered and ("length" in lowered or "characters" in lowered):
        explanation = (
            "Kaggle rejected the dataset title for its length.\n"
            f"  - Titles must be {TITLE_MIN}-{TITLE_MAX} characters.\n"
            "  - Fix: edit cfg.kaggle.dataset_title in configs/base.yaml."
        )
    elif "slug" in lowered or "url" in lowered and "invalid" in lowered:
        explanation = (
            "Kaggle rejected the dataset slug.\n"
            "  - It must be lowercase letters, digits and hyphens, "
            f"{SLUG_MIN}-{SLUG_MAX} characters, and must not include your "
            "username (that is added automatically).\n"
            "  - Fix: edit cfg.kaggle.dataset_slug in configs/base.yaml."
        )
    elif any(
        token in lowered
        for token in ("connectionerror", "timed out", "timeout", "temporary failure",
                      "name resolution", "ssl", "max retries", "connection aborted")
    ):
        explanation = (
            "The connection to Kaggle failed partway through.\n"
            "  - The upload is NOT resumable: it has to start again from the "
            "beginning.\n"
            "  - Check your internet connection, then re-run the same command. "
            "If it keeps failing on a large payload, run `dryrun` first to "
            "confirm everything else works on a small one."
        )
    elif "404" in lowered or "not found" in lowered:
        explanation = (
            "Kaggle could not find that dataset (404).\n"
            "  - `version` updates a dataset that already exists. If you have "
            "not created it yet, run `upload` first.\n"
            "  - Also check that cfg.kaggle.dataset_slug matches the dataset on "
            "your Kaggle account exactly."
        )
    else:
        explanation = (
            f"The Kaggle client failed during {what} and the error was not one "
            "this script recognises. The full output is below -- the useful "
            "line is usually the last one."
        )

    return (
        f"{what.capitalize()} failed (exit code {completed.returncode}).\n\n"
        f"{explanation}\n\n"
        "----- raw output from the Kaggle client -----\n"
        f"{output or '(no output)'}"
    )


# -- preflight -------------------------------------------------------------


def preflight(cfg: Any, logger: Any) -> str:
    """Check everything needed to talk to Kaggle, before anything expensive.

    Runs at the top of **every** subcommand. It exists because the alternative is
    finding out that the API token is missing after staging 3.9 GB and starting a
    two-hour transfer. Three checks, all local, all fast:

    1. the ``kaggle`` package is importable by *this* interpreter;
    2. ``kaggle.json`` exists and parses as JSON;
    3. it carries a username, which is printed.

    Deliberately uses :func:`importlib.util.find_spec` rather than ``import
    kaggle``: the Kaggle client authenticates against the network at import time,
    which is exactly what a two-second local preflight must not do.

    Args:
        cfg: The loaded config.
        logger: Logger.

    Returns:
        The Kaggle username.

    Raises:
        UploadError: Any check failed, with the specific remedy.
    """
    started = time.perf_counter()
    print("PREFLIGHT")

    spec = importlib.util.find_spec("kaggle")
    if spec is None:
        raise UploadError(
            "The 'kaggle' package is not installed in the Python that is running "
            "this script.\n"
            f"  Interpreter: {sys.executable}\n"
            f"  Version:     {sys.version.split()[0]}\n"
            "\n"
            "Install it into THIS interpreter -- not a different one -- with:\n"
            f"  {sys.executable} -m pip install kaggle\n"
            "\n"
            "Note: having `kaggle` installed under some other Python on the "
            "machine does not help. This script invokes it as "
            "`python -m kaggle` through its own interpreter, so the package and "
            "the project's dependencies must live together."
        )
    print(f"{OK} kaggle package importable")
    print(f"{INFO}interpreter: {sys.executable}")
    print(f"{INFO}python:      {sys.version.split()[0]}")

    path = credentials_path(cfg)
    if not path.is_file():
        raise UploadError(
            f"Kaggle API token not found at {path}.\n"
            "\n"
            "This is what causes the '401 - Unauthorized' error from Kaggle.\n"
            "To fix it, once:\n"
            "  1. Sign in at https://www.kaggle.com and open "
            "https://www.kaggle.com/settings/account\n"
            "  2. Scroll to the 'API' section and click 'Create New Token'.\n"
            "     Your browser downloads a small file called kaggle.json.\n"
            f"  3. Move that file to exactly this path:\n     {path}\n"
            f"     (create the {path.parent} folder if it does not exist)\n"
            "  4. Re-run this command.\n"
            "\n"
            "The file contains your API key. Do not commit it to the repository."
        )
    print(f"{OK} credentials file present: {path}")

    username = read_kaggle_username(cfg)
    print(f"{OK} Kaggle username: {username}")

    elapsed = time.perf_counter() - started
    print(f"{INFO}preflight completed in {elapsed:.2f} s")
    if elapsed > 2.0:
        logger.warning(
            "Preflight took %.2f s, which is longer than it should be (<2 s). "
            "Something is doing I/O it should not.",
            elapsed,
        )
    logger.info("Preflight OK for Kaggle user %s in %.2f s", username, elapsed)
    return username


# -- cache inspection ------------------------------------------------------


def cache_subset_dir(cfg: Any) -> Path:
    """The directory holding the cached samples for the active dataset subset.

    Mirrors what :class:`src.data.sen2naip.SEN2NAIPv2Dataset` computes, so the
    folder that is uploaded is exactly the folder the loader will look for on the
    other side.

    Args:
        cfg: The loaded config.

    Returns:
        ``resolve_cache_dir(cfg) / <subset>``.

    Raises:
        UploadError: The active dataset has no configured subset.
    """
    name = str(cfg.dataset.name)
    section = cfg.get(name) if hasattr(cfg, "get") else None
    subset = None if section is None else section.get("subset")
    if subset is None:
        # Fall back to the dataset name itself, which is what the synthetic stub
        # and any future subset-less dataset would use.
        subset = name
    return Path(resolve_cache_dir(cfg)) / str(subset)


def require_cache(cfg: Any) -> Path:
    """Return the cache directory, refusing to continue if it is not there.

    Raises:
        UploadError: The directory is missing or empty, naming the exact path
            that was checked and how to populate it.
    """
    path = cache_subset_dir(cfg)
    if not path.is_dir():
        raise UploadError(
            f"The sample cache does not exist.\n"
            f"  Looked for: {path}\n"
            "\n"
            "Nothing can be surveyed or uploaded until it is populated. Build it "
            "with:\n"
            "  python scripts/prepare_data.py --config configs/base.yaml\n"
            "\n"
            "That downloads the SEN2NAIPv2 pairs, which takes several hours. If "
            "you believe the cache exists somewhere else, check "
            "cfg.paths.cache_dir in configs/base.yaml."
        )
    if not any(path.iterdir()):
        raise UploadError(
            f"The sample cache directory exists but is empty:\n  {path}\n"
            "Run `python scripts/prepare_data.py --config configs/base.yaml` to "
            "populate it."
        )
    return path


def scan_cache(path: Path) -> Dict[str, Any]:
    """Collect sizes and counts for every file under ``path``.

    Args:
        path: The cache directory.

    Returns:
        ``{"files": [(relative_path, size_bytes), ...], "total_bytes": int,
        "by_suffix": {".npz": (count, bytes), ...}}``, with ``files`` sorted by
        relative path so two runs enumerate identically.
    """
    files: List[Tuple[Path, int]] = []
    by_suffix: Dict[str, List[int]] = {}
    for entry in sorted(path.rglob("*")):
        if not entry.is_file():
            continue
        size = entry.stat().st_size
        files.append((entry.relative_to(path), size))
        by_suffix.setdefault(entry.suffix.lower() or "(no suffix)", []).append(size)
    return {
        "files": files,
        "total_bytes": sum(size for _, size in files),
        "by_suffix": {
            suffix: (len(sizes), sum(sizes)) for suffix, sizes in by_suffix.items()
        },
    }


def sample_array_geometry(path: Path) -> Optional[Dict[str, Any]]:
    """Read one ``.npz`` and report the arrays it holds.

    The expected-size verdict is computed from the data itself rather than from
    hardcoded tile dimensions, so it stays correct if the tile size or dtype ever
    changes.

    Args:
        path: The cache directory.

    Returns:
        ``{"sample": name, "arrays": {key: {"shape", "dtype", "nbytes"}},
        "raw_bytes_per_pair": int}``, or ``None`` when there is no readable
        ``.npz`` (the caller then skips the size verdict rather than guessing).
    """
    import numpy as np

    for candidate in sorted(path.rglob("*.npz")):
        with np.load(candidate) as payload:
            arrays = {
                key: {
                    "shape": tuple(int(d) for d in payload[key].shape),
                    "dtype": str(payload[key].dtype),
                    "nbytes": int(payload[key].nbytes),
                }
                for key in payload.files
            }
        return {
            "sample": candidate.name,
            "arrays": arrays,
            "raw_bytes_per_pair": sum(a["nbytes"] for a in arrays.values()),
        }
    return None


def directory_tree(path: Path, max_depth: int = 2) -> List[str]:
    """Render a directory tree ``max_depth`` levels deep, with per-level counts.

    Args:
        path: Root to describe.
        max_depth: Levels below ``path`` to expand.

    Returns:
        Lines ready to print. Directories show how many files they contain
        directly, so a 3000-file folder is one line rather than 3000.
    """
    lines = [f"{path}"]

    def walk(current: Path, depth: int, prefix: str) -> None:
        children = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
        directories = [c for c in children if c.is_dir()]
        files = [c for c in children if c.is_file()]
        for directory in directories:
            contained = [p for p in directory.rglob("*") if p.is_file()]
            size = sum(p.stat().st_size for p in contained)
            lines.append(
                f"{prefix}+- {directory.name}/  "
                f"({len(contained):,} files, {human_bytes(size)})"
            )
            if depth < max_depth:
                walk(directory, depth + 1, prefix + "|  ")
        if files:
            shown = files[:3]
            for entry in shown:
                lines.append(
                    f"{prefix}+- {entry.name}  ({human_bytes(entry.stat().st_size)})"
                )
            if len(files) > len(shown):
                lines.append(f"{prefix}+- ... and {len(files) - len(shown):,} more files")

    walk(path, 1, "  ")
    return lines


def size_distribution(sizes: Sequence[int]) -> List[str]:
    """Describe a set of file sizes: percentiles and a coarse histogram.

    Args:
        sizes: File sizes in bytes.

    Returns:
        Lines ready to print. Empty when ``sizes`` is empty.
    """
    if not sizes:
        return []
    import numpy as np

    array = np.asarray(sizes, dtype=np.float64)
    lines = [
        f"  min    {human_bytes(array.min())}",
        f"  p05    {human_bytes(np.percentile(array, 5))}",
        f"  median {human_bytes(np.median(array))}",
        f"  mean   {human_bytes(array.mean())}",
        f"  p95    {human_bytes(np.percentile(array, 95))}",
        f"  max    {human_bytes(array.max())}",
    ]
    counts, edges = np.histogram(array, bins=8)
    widest = max(int(c) for c in counts) or 1
    lines.append("")
    lines.append("  distribution:")
    for count, low, high in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(40 * int(count) / widest))
        lines.append(
            f"    {human_bytes(low):>10} - {human_bytes(high):>10} | "
            f"{bar:<40} {int(count):,}"
        )
    return lines


def size_verdict(
    total_bytes: int, pair_count: int, geometry: Optional[Dict[str, Any]]
) -> List[str]:
    """Say, in plain English, whether the cache is a sane size.

    The expectation is computed from the arrays actually stored: raw bytes per
    pair times the number of pairs. NPZ compresses, so a healthy cache lands
    *below* that figure. Above 2x it, something is structurally wrong.

    Args:
        total_bytes: Total size of the cache on disk.
        pair_count: Number of ``.npz`` sample files.
        geometry: :func:`sample_array_geometry` output, or None.

    Returns:
        Lines ready to print.
    """
    if geometry is None or pair_count == 0:
        return [
            "  Cannot judge the size: no readable .npz sample was found, so "
            "there is nothing to compare against.",
        ]

    raw_per_pair = geometry["raw_bytes_per_pair"]
    expected = raw_per_pair * pair_count
    ratio = total_bytes / expected if expected else float("inf")
    shapes = ", ".join(
        f"{key} {info['shape']} {info['dtype']}"
        for key, info in geometry["arrays"].items()
    )

    lines = [
        f"  Each pair stores: {shapes}",
        f"  That is {human_bytes(raw_per_pair)} of raw array data per pair, so "
        f"{pair_count:,} pairs hold {human_bytes(expected)} uncompressed.",
        f"  On disk the cache is {human_bytes(total_bytes)}, which is "
        f"{ratio:.2f}x the uncompressed size.",
        "",
    ]

    if ratio > 2.0:
        lines += [
            "  VERDICT: TOO BIG. This is more than twice what the arrays "
            "themselves need.",
            "  Likely causes, most common first:",
            "    - the cache contains leftovers from an interrupted run "
            "(part-files, duplicates, or an older subset in the same folder);",
            "    - samples were saved as float32/float64 instead of the raw "
            "uint16 digital numbers, which quadruples the size for no gain;",
            "    - a second dataset subset is sharing the directory.",
            "  Look at the tree above: if there is more than one subset folder, "
            "or files that are not .npz/.json, that is the cause.",
        ]
    elif ratio > 1.05:
        lines += [
            "  VERDICT: LARGER THAN EXPECTED, but not alarmingly so. The arrays "
            "appear to be stored uncompressed, plus sidecar overhead.",
            "  This is fine to upload; it just costs more transfer time than it "
            "needs to.",
        ]
    elif ratio < 0.15:
        lines += [
            "  VERDICT: SUSPICIOUSLY SMALL -- under 15% of the uncompressed "
            "size.",
            "  Check that the sample count is what you expect and that the "
            "files are not truncated. Run:",
            "    python scripts/verify_data_root.py",
            "  which opens real samples and prints their shapes and reflectance "
            "ranges.",
        ]
    else:
        lines += [
            f"  VERDICT: SANE. {human_bytes(total_bytes)} for {pair_count:,} "
            "pairs is exactly what compressed NPZ storage of this geometry "
            "should look like",
            f"  (NPZ typically lands at 0.5-0.7x uncompressed; this is "
            f"{ratio:.2f}x).",
        ]
    return lines


# -- staging ---------------------------------------------------------------


def check_free_space(target: Path, needed_bytes: int, margin_gb: float) -> None:
    """Refuse to stage unless the copy fits with room to spare.

    Args:
        target: Where the staging folder will be written. The nearest existing
            ancestor is measured, since the folder itself may not exist yet.
        needed_bytes: Bytes about to be written.
        margin_gb: Free space that must remain afterwards.

    Raises:
        UploadError: There is not enough room, with the numbers spelled out.
    """
    probe = target
    while not probe.exists():
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    margin_bytes = int(margin_gb * GIB)
    required = needed_bytes + margin_bytes

    print(f"  Drive holding {probe}:")
    print(f"    total {human_bytes(usage.total)}, free {human_bytes(usage.free)}")
    print(
        f"    staging needs {human_bytes(needed_bytes)} "
        f"+ {margin_gb:g} GB safety margin = {human_bytes(required)}"
    )

    if usage.free < required:
        raise UploadError(
            "Not enough free disk space to stage the upload.\n"
            f"  Free now:  {human_bytes(usage.free)}\n"
            f"  Needed:    {human_bytes(needed_bytes)} for the copy, plus a "
            f"{margin_gb:g} GB margin = {human_bytes(required)}\n"
            f"  Short by:  {human_bytes(required - usage.free)}\n"
            "\n"
            "Staging COPIES the cache rather than moving it, deliberately: a "
            "failed upload must never be able to damage the only copy of an "
            "8.7-hour download.\n"
            "Options:\n"
            "  - free up space on that drive;\n"
            "  - point cfg.kaggle.staging_dir at a drive with more room;\n"
            "  - lower cfg.kaggle.free_space_margin_gb if you are confident "
            f"(currently {margin_gb:g} GB)."
        )
    print(f"    OK -- {human_bytes(usage.free - required)} would remain spare.")


def hash_file(path: Path, algorithm: str, digest_bytes: int, chunk_bytes: int) -> str:
    """Hash one file for ``checksums.csv``.

    Args:
        path: File to hash.
        algorithm: ``hashlib`` algorithm name, from
            ``cfg.kaggle.checksum_algorithm``.
        digest_bytes: Digest length for algorithms that take one (blake2b).
        chunk_bytes: Read buffer size.

    Returns:
        The hex digest.

    Raises:
        UploadError: The algorithm is not available in this Python build.
    """
    try:
        if algorithm == "blake2b":
            digest = hashlib.blake2b(digest_size=digest_bytes)
        else:
            digest = hashlib.new(algorithm)
    except (ValueError, TypeError) as exc:
        raise UploadError(
            f"cfg.kaggle.checksum_algorithm={algorithm!r} is not a hash this "
            f"Python supports. Available: {sorted(hashlib.algorithms_available)}"
        ) from exc

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(
    destination: Path,
    cfg: Any,
    pair_count: int,
    total_bytes: int,
    subset_name: str,
    extras: Sequence[str],
) -> Path:
    """Generate the dataset README.

    Everything a future reader needs to interpret the numbers, stated rather than
    implied: where the data came from, the band order, the reflectance divisor,
    and the nodata sentinel. A cache of uint16 arrays with no divisor recorded is
    an archive of meaningless integers.

    Args:
        destination: Staging folder.
        cfg: The loaded config.
        pair_count: Number of sample pairs staged.
        total_bytes: Size of the staged sample data.
        subset_name: The subset folder name inside the dataset.
        extras: Names of the non-sample files also staged.

    Returns:
        The path written.
    """
    bands = list(cfg.dataset.bands)
    today = _dt.date.today().isoformat()
    mount = f"/kaggle/input/{cfg.kaggle.dataset_slug}"

    text = f"""# DrishtiSR -- SEN2NAIPv2 sample cache

Pre-downloaded Sentinel-2 / NAIP super-resolution pairs for
**{cfg.project.name}** ({cfg.project.problem_statement}):
{cfg.project.description}

Uploaded **{today}**.

## What this is, and why it exists

Fetching these pairs from the source archive takes about **10.4 seconds per
sample**, i.e. roughly **8.7 hours** for the full set. A Kaggle GPU session that
re-downloaded them would burn more than a quarter of the weekly 30-hour budget
on data transfer. This dataset is that download, done once.

## Contents

| item | description |
|---|---|
| `{subset_name}/` | {pair_count:,} sample pairs ({human_bytes(total_bytes)}), one `.npz` + one `.json` sidecar each |
{chr(10).join(f"| `{name}` | see below |" for name in extras)}

## Data format

Each `.npz` holds **raw uint16 digital numbers**, not reflectance:

- `lr` -- low-resolution tile, shape `(C, H, W)`, channel axis first
- `hr` -- high-resolution tile, shape `(C, H*4, W*4)`

| property | value |
|---|---|
| source dataset | `{cfg.dataset.name}` / `{subset_name}` |
| sample pairs | {pair_count:,} |
| band order (channel axis) | `{', '.join(str(b) for b in bands)}` |
| reflectance divisor | `{cfg.dataset.reflectance_scale:g}` |
| nodata value (raw DN) | `{cfg.dataset.nodata_value}` |
| LR ground sample distance | {cfg.dataset.lr_gsd_m:g} m |
| HR ground sample distance | {cfg.dataset.hr_gsd_m:g} m |
| super-resolution factor | x{cfg.sr.scale} |

**Band order is load-bearing.** `{bands[0]}` is red and `{bands[-1]}` is NIR --
this is RGBN, not BGR. Reading it in the wrong order silently transposes red and
blue and every metric still looks plausible.

To convert to surface reflectance:

```python
reflectance = digital_number.astype("float32") / {cfg.dataset.reflectance_scale:g}
```

Mask `{cfg.dataset.nodata_value}` **before** dividing --
{cfg.dataset.nodata_value} / {cfg.dataset.reflectance_scale:g} =
{cfg.dataset.nodata_value / float(cfg.dataset.reflectance_scale):.2f} reflectance
would otherwise fail every validity check.

Reflectance is nominally `[0, 1]` but is **not clipped**: cloud, snow, specular
water and bright roofs legitimately exceed 1.0, and the spectral-consistency
objective this dataset supports depends on that tail being intact.

## Using it in a Kaggle notebook

Attach this dataset (sidebar -> `+ Add Input`), then:

```python
!python scripts/verify_data_root.py
```

That resolves the mount, loads the manifest, opens real samples, and prints
their shapes and per-band reflectance ranges. It is the one-command proof that
the session can actually see the data.

The repository resolves the mount path from config, so nothing needs editing:
`paths.kaggle_mount_root` + `paths.kaggle_dataset_dir` compose to `{mount}`.

The manifest and split CSVs live at the top level of this dataset. Copy them
into the repository's `outputs/` folder so the loader finds them:

```python
!mkdir -p outputs && cp {mount}/*.csv outputs/
```

## Integrity

`checksums.csv` lists every file with its size and a
`{cfg.kaggle.checksum_algorithm}` digest, so a truncated or corrupted upload is
detectable rather than merely suspected.

## Provenance and licence

Derived from SEN2NAIPv2 (Sentinel-2 L2A + NAIP aerial imagery). Redistributed
here for reproducibility of the DrishtiSR experiments. Consult the upstream
SEN2NAIPv2 dataset card for the terms attached to the source imagery.
"""
    path = destination / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_checksums(
    destination: Path, cfg: Any, logger: Any
) -> Tuple[Path, int]:
    """Hash every staged file into ``checksums.csv``.

    Args:
        destination: Staging folder. Everything under it is hashed except the
            checksum file itself.
        cfg: The loaded config.
        logger: Logger for progress.

    Returns:
        ``(path, file_count)``.
    """
    algorithm = str(cfg.kaggle.checksum_algorithm)
    digest_bytes = int(cfg.kaggle.checksum_digest_bytes)
    chunk_bytes = int(cfg.kaggle.checksum_chunk_bytes)
    path = destination / "checksums.csv"

    targets = [
        entry
        for entry in sorted(destination.rglob("*"))
        if entry.is_file() and entry != path
    ]
    total = len(targets)
    print(f"  Hashing {total:,} files with {algorithm}...")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "size_bytes", algorithm])
        for index, entry in enumerate(targets, start=1):
            writer.writerow(
                [
                    entry.relative_to(destination).as_posix(),
                    entry.stat().st_size,
                    hash_file(entry, algorithm, digest_bytes, chunk_bytes),
                ]
            )
            if index % 500 == 0 or index == total:
                print(f"    {index:,}/{total:,}")
    logger.info("Wrote %s covering %d files.", path, total)
    return path, total


def write_metadata(destination: Path, username: str, slug: str, title: str) -> Path:
    """Generate ``dataset-metadata.json`` -- the file you never edit by hand.

    Args:
        destination: Staging folder.
        username: Kaggle username, read from ``kaggle.json``.
        slug: Bare dataset name.
        title: Dataset title.

    Returns:
        The path written.
    """
    payload = {
        "title": title,
        "id": f"{username}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    path = destination / "dataset-metadata.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def copy_supporting_files(cfg: Any, destination: Path, logger: Any) -> List[str]:
    """Copy the manifest and split CSVs into the staging folder.

    These are small and are what let a Kaggle session reproduce the exact local
    view of the data -- which samples were validated, and which are train/val/test.
    Uploading pixels without them means re-deriving the split in the notebook,
    and a re-derived split is a different split.

    Args:
        cfg: The loaded config.
        destination: Staging folder.
        logger: Logger.

    Returns:
        The file names copied. Missing files are reported and skipped, not
        fabricated -- but the caller warns loudly about the split file, because
        training without it is the thing that silently leaks.
    """
    manifest_dir = Path(resolve_output_path(cfg, "manifest_dir"))
    dataset_name = str(cfg.dataset.name)
    wanted = [
        manifest_dir / f"manifest_{dataset_name}.csv",
        manifest_dir / str(cfg.splits.output_name).format(dataset=dataset_name),
    ]

    copied = []
    for source in wanted:
        if source.is_file():
            shutil.copy2(source, destination / source.name)
            copied.append(source.name)
            print(f"  + {source.name}  ({human_bytes(source.stat().st_size)})")
        else:
            print(f"  ! MISSING, not staged: {source}")
            logger.warning("Supporting file absent, not staged: %s", source)
    return copied


def build_staging(
    cfg: Any,
    logger: Any,
    staging_dir: Path,
    slug: str,
    title: str,
    username: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble the folder that gets uploaded.

    Args:
        cfg: The loaded config.
        logger: Logger.
        staging_dir: Destination. Cleared first if it already exists.
        slug: Bare dataset name for ``dataset-metadata.json``.
        title: Dataset title.
        username: Kaggle username.
        limit: Stage only this many sample pairs (used by ``dryrun``). ``None``
            stages everything.

    Returns:
        ``{"dir", "file_count", "total_bytes", "pair_count", "subset"}``.

    Raises:
        UploadError: The cache is missing, or there is not enough disk space.
    """
    cache = require_cache(cfg)
    subset = cache.name
    scan = scan_cache(cache)

    npz_files = sorted(p for p, _ in scan["files"] if p.suffix.lower() == ".npz")
    if limit is not None:
        npz_files = npz_files[: int(limit)]
        selected = set()
        for relative in npz_files:
            selected.add(relative)
            sidecar = relative.with_suffix(".json")
            if (cache / sidecar).is_file():
                selected.add(sidecar)
        chosen = [(p, s) for p, s in scan["files"] if p in selected]
    else:
        chosen = list(scan["files"])

    payload_bytes = sum(size for _, size in chosen)

    rule("disk space")
    check_free_space(staging_dir, payload_bytes, float(cfg.kaggle.free_space_margin_gb))

    rule("staging")
    if staging_dir.exists():
        print(f"  Clearing previous staging folder {staging_dir}")
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    subset_dir = staging_dir / subset
    subset_dir.mkdir(parents=True)
    print(f"  Copying {len(chosen):,} files into {subset_dir} ...")
    for index, (relative, _size) in enumerate(chosen, start=1):
        target = subset_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache / relative, target)
        if index % 1000 == 0 or index == len(chosen):
            print(f"    {index:,}/{len(chosen):,}")

    print()
    print("  Supporting files:")
    extras = copy_supporting_files(cfg, staging_dir, logger)

    pair_count = len(npz_files)
    readme = write_readme(
        staging_dir, cfg, pair_count, payload_bytes, subset, extras
    )
    print(f"  + {readme.name}  (generated)")

    metadata = write_metadata(staging_dir, username, slug, title)
    print(f"  + {metadata.name}  (generated: id = {username}/{slug})")

    print()
    checksums, hashed = write_checksums(staging_dir, cfg, logger)
    print(f"  + {checksums.name}  ({hashed:,} files covered)")

    staged = [p for p in staging_dir.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in staged)

    return {
        "dir": staging_dir,
        "file_count": len(staged),
        "total_bytes": total_bytes,
        "pair_count": pair_count,
        "subset": subset,
    }


def public_flag(cfg: Any) -> List[str]:
    """``["-u"]`` when the dataset should be public, else ``[]``.

    ``kaggle datasets create`` defaults to private and takes ``-u/--public`` to
    publish. Private is the default here: a Sentinel-2/NAIP derivative is
    redistributable, but publishing it should be a deliberate decision on the
    dataset page, not a side effect of running a script.
    """
    return [] if bool(cfg.kaggle.is_private) else ["-u"]


def upload_estimates(cfg: Any, total_bytes: int) -> List[str]:
    """Estimated transfer times at the configured link speeds."""
    lines = []
    for mbps in cfg.kaggle.upload_speeds_mbps:
        seconds = total_bytes * 8.0 / (float(mbps) * 1e6)
        lines.append(f"    at {float(mbps):g} Mbps upload:  {human_duration(seconds)}")
    return lines


# -- smoke support ---------------------------------------------------------


def ensure_smoke_cache(cfg: Any, logger: Any) -> None:
    """Fabricate a tiny cache so ``--smoke`` exercises the whole script offline.

    ``--smoke`` switches ``dataset.name`` to the synthetic stub, which has no
    cache on disk because it fabricates samples in memory. Without this, every
    subcommand would stop at "cache not found" and the pre-flight would prove
    nothing.

    So under ``--smoke`` only, a handful of stub ``.npz``/``.json`` pairs are
    written into the stub's cache path. Staging, README generation, checksums,
    metadata generation, and the size verdict then all run for real, on CPU, in
    seconds, with no network access whatsoever.

    Args:
        cfg: The loaded config, already smoke-merged.
        logger: Logger.
    """
    import numpy as np

    path = cache_subset_dir(cfg)
    if path.is_dir() and any(path.glob("*.npz")):
        return

    path.mkdir(parents=True, exist_ok=True)
    bands = len(cfg.dataset.bands)
    lr_size = int(cfg.sr.lr_patch_size)
    scale = int(cfg.sr.scale)
    rng = np.random.default_rng(int(cfg.seed))
    count = int(cfg.synthetic_stub.num_samples)

    logger.warning(
        "--smoke: fabricating %d stub cache entries in %s so the upload "
        "pipeline can be exercised without real data. These are NOT real "
        "samples and must never be uploaded.",
        count,
        path,
    )
    for index in range(count):
        stem = f"smoke_{index:04d}"
        np.savez_compressed(
            path / f"{stem}.npz",
            lr=rng.integers(200, 3000, (bands, lr_size, lr_size), dtype=np.uint16),
            hr=rng.integers(
                200, 3000, (bands, lr_size * scale, lr_size * scale), dtype=np.uint16
            ),
        )
        (path / f"{stem}.json").write_text(
            json.dumps({"sample_id": stem, "synthetic": True}), encoding="utf-8"
        )


def refuse_network_in_smoke(command: str) -> None:
    """Stop a network subcommand under ``--smoke``.

    ``--smoke`` must never touch the network. ``dryrun``, ``upload``, and
    ``version`` therefore stage and validate everything, then stop before the
    transfer and say so.
    """
    print()
    print("=" * 78)
    print(f"  --smoke: STOPPING BEFORE THE NETWORK CALL for `{command}`.")
    print("=" * 78)
    print(
        "\n  Everything above ran for real: the cache was scanned, the staging\n"
        "  folder was built, the README and checksums were generated, and the\n"
        "  metadata was written with your slug and title validated against\n"
        "  Kaggle's rules.\n"
        "\n"
        "  The only thing skipped is the transfer itself, because --smoke is a\n"
        "  pre-flight and must never contact the network.\n"
        "\n"
        f"  Re-run without --smoke to actually perform `{command}`."
    )


# -- subcommands -----------------------------------------------------------


def cmd_survey(args, cfg, logger) -> int:
    """Inspect the local cache and report whether its size makes sense."""
    preflight(cfg, logger)
    cache = require_cache(cfg)

    print()
    print("=" * 78)
    print("  LOCAL CACHE SURVEY")
    print("=" * 78)
    print(f"\n  Cache directory: {cache}")

    scan = scan_cache(cache)
    files, total_bytes = scan["files"], scan["total_bytes"]
    npz_sizes = [s for p, s in files if p.suffix.lower() == ".npz"]

    rule("totals")
    print(f"  Total size:  {human_bytes(total_bytes)}")
    print(f"  File count:  {len(files):,}")
    for suffix, (count, size) in sorted(scan["by_suffix"].items()):
        print(f"    {suffix:<14} {count:>7,} files   {human_bytes(size):>14}")

    rule("directory tree (2 levels)")
    for line in directory_tree(cache, max_depth=2):
        print("  " + line)

    rule("per-file size distribution (.npz only)")
    for line in size_distribution(npz_sizes):
        print(line)

    rule("verdict")
    geometry = sample_array_geometry(cache)
    if geometry:
        print(f"  Read geometry from a real sample: {geometry['sample']}")
    for line in size_verdict(total_bytes, len(npz_sizes), geometry):
        print(line)

    estimated = upload_estimates(cfg, total_bytes)
    rule("upload time, if you push this as-is")
    for line in estimated:
        print(line)

    next_steps(
        "Build the upload folder:  python scripts/kaggle_upload.py stage",
        "Test the settings on 5 samples:  python scripts/kaggle_upload.py dryrun",
        "Do the real upload:  python scripts/kaggle_upload.py upload",
    )
    return 0


def cmd_stage(args, cfg, logger) -> int:
    """Build the staging folder, without contacting Kaggle."""
    username = preflight(cfg, logger)
    slug = validate_slug(str(cfg.kaggle.dataset_slug), "kaggle.dataset_slug")
    title = validate_title(str(cfg.kaggle.dataset_title), "kaggle.dataset_title")
    staging_dir = repo_root() / str(cfg.kaggle.staging_dir)

    print()
    print("=" * 78)
    print("  BUILDING THE UPLOAD FOLDER")
    print("=" * 78)
    print(f"\n  Kaggle account: {username}")
    print(f"  Dataset will be: {username}/{slug}")
    print(f"  Title:           {title}")

    result = build_staging(cfg, logger, staging_dir, slug, title, username)

    rule("staged")
    print(f"  Folder:     {result['dir']}")
    print(f"  Files:      {result['file_count']:,}")
    print(f"  Total size: {human_bytes(result['total_bytes'])}")
    print(f"  Pairs:      {result['pair_count']:,} in {result['subset']}/")

    next_steps(
        f"Look inside {result['dir']} if you want to check it by eye.",
        "Test everything on a 5-sample payload first:  "
        "python scripts/kaggle_upload.py dryrun",
        "Then upload for real:  python scripts/kaggle_upload.py upload",
    )
    return 0


def cmd_dryrun(args, cfg, logger) -> int:
    """Upload a 5-sample test dataset to prove the settings work."""
    username = preflight(cfg, logger)
    slug = validate_slug(str(cfg.kaggle.dryrun_slug), "kaggle.dryrun_slug")
    title = validate_title(str(cfg.kaggle.dryrun_title), "kaggle.dryrun_title")
    staging_dir = repo_root() / str(cfg.kaggle.dryrun_staging_dir)
    samples = int(cfg.kaggle.dryrun_samples)

    print()
    print("=" * 78)
    print("  DRY RUN -- A TINY TEST UPLOAD")
    print("=" * 78)
    print(
        f"\n  WHY THIS STEP EXISTS\n"
        f"\n"
        f"  The real cache is several GB and the upload is NOT resumable. If the\n"
        f"  slug is malformed, the title is too short, the token is stale, or the\n"
        f"  folder structure does not survive the round trip, you would find out\n"
        f"  only after a long transfer -- and then have to redo all of it.\n"
        f"\n"
        f"  So this pushes {samples} samples plus the manifest to a SEPARATE throwaway\n"
        f"  slug. It exercises exactly the same code path as the real upload:\n"
        f"  same metadata generation, same directory handling, same --dir-mode zip.\n"
        f"  If it works, the only thing left to vary is the size of the payload.\n"
        f"  If it fails, you have lost seconds instead of hours.\n"
    )
    print(f"  Kaggle account:  {username}")
    print(f"  Test dataset:    {username}/{slug}   (throwaway -- delete it later)")

    result = build_staging(
        cfg, logger, staging_dir, slug, title, username, limit=samples
    )

    rule("staged for the test upload")
    print(f"  Folder: {result['dir']}")
    print(f"  Files:  {result['file_count']:,}  "
          f"({human_bytes(result['total_bytes'])})")

    if args.smoke:
        refuse_network_in_smoke("dryrun")
        next_steps(
            "Re-run without --smoke to perform the real dry run.",
        )
        return 0

    rule("uploading the test dataset")
    run_kaggle(
        [
            "datasets", "create",
            "-p", str(result["dir"]),
            "--dir-mode", "zip",
            "--keep-tabular",
        ] + public_flag(cfg),
        timeout_s=int(cfg.kaggle.metadata_timeout_s),
        logger=logger,
        what="the test upload",
    )
    print("  Test upload accepted by Kaggle.")

    url = f"https://www.kaggle.com/datasets/{username}/{slug}"
    print(f"\n  Dataset page: {url}")

    passed = verify_remote_layout(cfg, logger, f"{username}/{slug}", result["subset"])

    if passed:
        next_steps(
            "The structure is correct, so the real upload will inherit it.",
            "Upload the real dataset (multi-hour, run it in its own window):  "
            "python scripts/kaggle_upload.py upload --yes",
            f"Afterwards, delete the throwaway test dataset at {url}/settings",
        )
        return 0

    next_steps(
        "DO NOT run the real upload yet -- it would reproduce the same flaw at "
        "3.9 GB instead of 6 MB.",
        f"Inspect what actually landed: {url}",
        "Fix the staging layout or the --dir-mode handling, then re-run "
        "`dryrun`.",
        f"Delete the failed test dataset at {url}/settings before retrying.",
    )
    return 1


# -- programmatic verification of what actually landed on Kaggle -----------


def _files_page(
    cfg: Any, logger: Any, full_slug: str, token: Optional[str]
) -> Optional[Tuple[List[Dict[str, Any]], Optional[str]]]:
    """Fetch one page of ``kaggle datasets files``.

    Args:
        cfg: The loaded config.
        logger: Logger.
        full_slug: ``"<username>/<dataset-name>"``.
        token: Page token from a previous call, or ``None`` for the first page.

    Returns:
        ``(rows, next_token)`` where each row has ``name`` and ``size``, or
        ``None`` when the dataset is not queryable yet.
    """
    command = kaggle_command() + [
        "datasets", "files", full_slug, "--format", "csv", "--page-size", "200"
    ]
    if token:
        command += ["--page-token", token]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if completed.returncode != 0:
        logger.info(
            "datasets files not ready yet (exit %d): %s",
            completed.returncode,
            (completed.stderr or completed.stdout or "").strip()[:200],
        )
        return None

    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None

    next_token = None
    for line in lines:
        if line.startswith("Next Page Token = "):
            next_token = line.split("= ", 1)[1].strip()

    # Kaggle prints a human note before the CSV while a dataset is still being
    # processed; treat anything without a header row as "not ready".
    header_index = next(
        (i for i, line in enumerate(lines) if line.lower().startswith("name,")), None
    )
    if header_index is None:
        logger.info("datasets files returned no CSV header yet: %s", lines[:2])
        return None

    rows = [
        {"name": str(row["name"]).strip(), "size": int(row.get("size") or 0)}
        for row in csv.DictReader(lines[header_index:])
        if row.get("name")
    ]
    return rows, next_token


def list_remote_files(
    cfg: Any, logger: Any, full_slug: str, with_sizes: bool = False
) -> Optional[Any]:
    """Ask Kaggle what files a dataset contains, following every page.

    Kaggle pages this endpoint at 200 entries. The full cache is 6008 files, so
    reading only the first page would report a complete upload as 3% complete --
    or, worse, would let a genuinely truncated upload look fine because the
    missing files happen to sort past the first page.

    Args:
        cfg: The loaded config.
        logger: Logger.
        full_slug: ``"<username>/<dataset-name>"``.
        with_sizes: Return ``{name: size}`` instead of a list of names.

    Returns:
        File names (or ``{name: size}`` when ``with_sizes``), or ``None`` when
        the dataset is not queryable yet. ``None`` means "ask again"; an empty
        result means "Kaggle says this dataset has no files", which is a real
        answer and a real failure.
    """
    collected: Dict[str, int] = {}
    token: Optional[str] = None
    pages = 0
    while True:
        page = _files_page(cfg, logger, full_slug, token)
        if page is None:
            return None if pages == 0 else _shape(collected, with_sizes)
        rows, token = page
        for row in rows:
            collected[row["name"]] = row["size"]
        pages += 1
        if not token or not rows:
            break
        if pages >= int(cfg.kaggle.max_listing_pages):
            logger.warning(
                "Stopped paging %s after %d pages (cfg.kaggle.max_listing_pages). "
                "The listing is INCOMPLETE.",
                full_slug,
                pages,
            )
            break
    logger.info("Listed %d file(s) from %s over %d page(s).",
                len(collected), full_slug, pages)
    return _shape(collected, with_sizes)


def _shape(collected: Dict[str, int], with_sizes: bool) -> Any:
    """Return the listing as a size map or a plain name list."""
    return dict(collected) if with_sizes else list(collected)


def verify_remote_layout(
    cfg: Any, logger: Any, full_slug: str, subset: str
) -> bool:
    """Poll Kaggle until the dataset is processed, then check the file layout.

    This is the whole point of the dry run: proving, without opening a browser,
    that ``--dir-mode zip`` round-tripped the directory structure. Kaggle
    *unpacks* an uploaded zip, so the sample files must come back reported inside
    the ``<subset>/`` folder. If they come back flat at the top level, the loader
    on the other side will look in ``<mount>/<subset>/`` and find nothing.

    Args:
        cfg: The loaded config.
        logger: Logger.
        full_slug: ``"<username>/<dataset-name>"``.
        subset: The directory the sample files must be nested inside -- the
            dataset subset name, which is exactly what
            ``resolve_cache_dir(cfg) / subset`` will look for.

    Returns:
        True when the layout is correct.
    """
    timeout_s = int(cfg.kaggle.verify_timeout_s)
    interval_s = int(cfg.kaggle.verify_poll_interval_s)
    deadline = time.monotonic() + timeout_s

    rule("verifying what actually landed on Kaggle")
    print(
        f"  Kaggle unpacks the uploaded zip server-side, which takes a minute or\n"
        f"  two. Polling `kaggle datasets files {full_slug}` every {interval_s} s\n"
        f"  (giving up after {human_duration(timeout_s)}).\n"
    )

    names: Optional[List[str]] = None
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        names = list_remote_files(cfg, logger, full_slug)
        if names:
            print(f"  Attempt {attempt}: Kaggle reports {len(names)} file(s). Ready.")
            break
        remaining = int(deadline - time.monotonic())
        print(f"  Attempt {attempt}: not processed yet, {remaining} s left...")
        time.sleep(interval_s)

    if not names:
        print()
        print(f"{BAD}VERIFICATION INCONCLUSIVE -- TIMED OUT")
        print(
            f"\n  Kaggle never reported any files for {full_slug} within "
            f"{human_duration(timeout_s)}.\n"
            "\n"
            "  This is NOT necessarily a failed upload. It usually means Kaggle's\n"
            "  server-side processing is slow or backed up. What it does mean is\n"
            "  that the layout is unproven, so the real upload is not yet safe.\n"
            "\n"
            "  What to do:\n"
            f"    - wait a few minutes, then re-check with:\n"
            f"        {sys.executable} -m kaggle datasets files {full_slug}\n"
            "    - or raise cfg.kaggle.verify_timeout_s and re-run `dryrun`;\n"
            f"    - or look at https://www.kaggle.com/datasets/{full_slug}"
        )
        return False

    print()
    print("  Files Kaggle reports:")
    for name in sorted(names)[:20]:
        print(f"    {name}")
    if len(names) > 20:
        print(f"    ... and {len(names) - 20} more")

    prefix = f"{subset}/"
    npz_names = [n for n in names if n.lower().endswith(".npz")]
    nested = [n for n in npz_names if n.replace("\\", "/").startswith(prefix)]
    flat = [n for n in npz_names if not n.replace("\\", "/").startswith(prefix)]
    zips_left = [n for n in names if n.lower().endswith(".zip")]

    print()
    print(f"  .npz files reported:            {len(npz_names)}")
    print(f"  nested under '{prefix}':  {len(nested)}")
    print(f"  flat at the top level:          {len(flat)}")
    if zips_left:
        print(f"  still packed in a .zip:         {zips_left}")

    print()
    if npz_names and not flat and not zips_left:
        print("=" * 78)
        print("  PASS -- the directory structure survived the round trip.")
        print("=" * 78)
        print(
            f"\n  Every one of the {len(nested)} sample files came back nested "
            f"inside '{prefix}',\n"
            "  which is exactly where the loader looks: resolve_cache_dir(cfg) "
            "returns the\n"
            f"  mount, and the dataset appends '{subset}' to it.\n"
            "\n"
            "  --dir-mode zip worked. The real upload will produce the same "
            "layout at full size."
        )
        logger.info("Dry-run layout verification PASSED for %s", full_slug)
        return True

    print("=" * 78)
    print("  FAIL -- the directory structure did NOT survive the round trip.")
    print("=" * 78)
    # Order matters: a still-zipped upload reports NO .npz entries at all, so the
    # zip diagnosis has to be checked before the generic "nothing arrived" one --
    # otherwise the most informative case prints the least informative message.
    if zips_left:
        print(
            f"\n  The upload is still sitting as a .zip ({zips_left}) rather than "
            "being unpacked.\n"
            "  Kaggle did not expand the archive, so the loader would see a zip "
            "file where it\n"
            "  expects a folder of samples."
        )
    elif not npz_names:
        print(
            "\n  Kaggle reports no .npz files at all. The sample data did not "
            "arrive."
        )
    else:
        print(
            f"\n  {len(flat)} .npz file(s) came back FLAT at the top level "
            f"instead of inside '{prefix}'.\n"
            f"  Examples: {flat[:3]}"
        )
    print(
        "\n  WHAT THIS MEANS\n"
        "\n"
        "  --dir-mode zip did not apply, so the folder structure was flattened on\n"
        "  the way to Kaggle. The real upload would inherit exactly the same "
        "flaw --\n"
        "  at 3.9 GB and a multi-hour transfer instead of a few megabytes.\n"
        "\n"
        f"  Downstream, resolve_cache_dir(cfg) returns the mount and the loader\n"
        f"  appends '{subset}'. With a flattened layout that folder does not "
        "exist, so\n"
        "  every sample reads as uncached and the training run sees zero data.\n"
        "\n"
        "  Fix the staging layout or the --dir-mode handling and re-run `dryrun`\n"
        "  BEFORE spending the real upload."
    )
    logger.error("Dry-run layout verification FAILED for %s", full_slug)
    return False


def cmd_upload(args, cfg, logger) -> int:
    """The real upload, behind a typed confirmation."""
    username = preflight(cfg, logger)
    slug = validate_slug(str(cfg.kaggle.dataset_slug), "kaggle.dataset_slug")
    title = validate_title(str(cfg.kaggle.dataset_title), "kaggle.dataset_title")
    staging_dir = repo_root() / str(cfg.kaggle.staging_dir)

    print()
    print("=" * 78)
    print("  REAL UPLOAD")
    print("=" * 78)
    print(f"\n  Kaggle account: {username}")
    print(f"  Dataset:        {username}/{slug}")
    print(f"  Title:          {title}")
    print(f"  Visibility:     {'private' if cfg.kaggle.is_private else 'PUBLIC'}")

    result = build_staging(cfg, logger, staging_dir, slug, title, username)

    print()
    print("=" * 78)
    print("  CONFIRM BEFORE UPLOADING")
    print("=" * 78)
    print(f"\n  About to upload:")
    print(f"    {result['file_count']:,} files")
    print(f"    {human_bytes(result['total_bytes'])}")
    print(f"    {result['pair_count']:,} sample pairs")
    print(f"    to {username}/{slug}")
    print("\n  Estimated transfer time:")
    for line in upload_estimates(cfg, result["total_bytes"]):
        print(line)
    print(
        "\n  !! THE UPLOAD IS NOT RESUMABLE.\n"
        "  !! If you close this window, lose your connection, or press Ctrl+C,\n"
        "  !! the transfer starts again from zero. Nothing is kept.\n"
        "  !! Leave this running, on a stable connection, and do not sleep the\n"
        "  !! machine. Disable any 'sleep on idle' setting first.\n"
        "\n"
        "  Your local cache is not touched by this: the staging folder is a copy."
    )

    if args.smoke:
        refuse_network_in_smoke("upload")
        next_steps("Re-run without --smoke to perform the real upload.")
        return 0

    if args.yes:
        print(
            "\n  --yes given: skipping the confirmation prompt and starting now.\n"
            "  (This is the unattended mode -- intended for a dedicated window.)"
        )
    else:
        print()
        try:
            answer = input("  Type YES (in capitals) to start the upload: ").strip()
        except EOFError:
            raise UploadError(
                "No answer received -- this command needs an interactive terminal "
                "so it can ask for confirmation before a multi-GB, non-resumable "
                "transfer.\n"
                "Run it directly in your terminal, or pass --yes to skip the "
                "prompt for an unattended run."
            )
        if answer != "YES":
            print(f"\n  You typed {answer!r}, not 'YES'. Nothing was uploaded.")
            next_steps(
                "Nothing changed on Kaggle, and the staging folder is still at "
                f"{result['dir']}.",
                "Re-run `python scripts/kaggle_upload.py upload` when you are "
                "ready, or add --yes to skip this prompt.",
            )
            return 1

    upload_log = Path(resolve_output_path(cfg, "upload_log"))
    progress = get_logger("upload_progress", log_file=upload_log)

    rule("uploading -- this will take a while")
    print(f"  Progress is being written to {upload_log}")
    print("  Watch it from another window with:")
    print(f"    Get-Content -Path {upload_log} -Wait -Tail 20")
    print()

    progress.info("=" * 70)
    progress.info("UPLOAD STARTED")
    progress.info("  dataset:    %s/%s", username, slug)
    progress.info("  staging:    %s", result["dir"])
    progress.info("  files:      %d", result["file_count"])
    progress.info("  size:       %s", human_bytes(result["total_bytes"]))
    progress.info("  pairs:      %d in %s/", result["pair_count"], result["subset"])
    progress.info("  visibility: %s", "private" if cfg.kaggle.is_private else "PUBLIC")
    for line in upload_estimates(cfg, result["total_bytes"]):
        progress.info("  estimate %s", line.strip())
    progress.info(
        "  NOT RESUMABLE -- an interruption means starting over from zero."
    )
    progress.info("Handing off to the Kaggle client; this is the long part.")

    started = time.perf_counter()
    try:
        run_kaggle(
            [
                "datasets", "create",
                "-p", str(result["dir"]),
                "--dir-mode", "zip",
                # Kaggle CONVERTS tabular files to CSV by default, which would
                # rewrite the manifest and split CSVs and invalidate every hash
                # in checksums.csv. Keep them byte-for-byte.
                "--keep-tabular",
            ] + public_flag(cfg),
            timeout_s=int(cfg.kaggle.cli_timeout_s),
            logger=logger,
            what="the upload",
        )
    except UploadError as exc:
        elapsed = time.perf_counter() - started
        progress.error("UPLOAD FAILED after %s", human_duration(elapsed))
        for line in str(exc).splitlines():
            progress.error("  %s", line)
        raise

    elapsed = time.perf_counter() - started
    url = f"https://www.kaggle.com/datasets/{username}/{slug}"
    progress.info("UPLOAD FINISHED in %s", human_duration(elapsed))
    progress.info("  dataset page: %s", url)
    progress.info(
        "  effective rate: %.2f Mbps",
        (result["total_bytes"] * 8.0 / 1e6) / max(elapsed, 1e-6),
    )
    progress.info("=" * 70)

    print(f"\n  Upload accepted in {human_duration(elapsed)}. Dataset page: {url}")
    next_steps(
        f"Open {url} and wait for Kaggle to finish unpacking (a few minutes for "
        f"{human_bytes(result['total_bytes'])}).",
        "Confirm the layout without a browser:  "
        f"{sys.executable} -m kaggle datasets files {username}/{slug}",
        "In a notebook, sidebar -> '+ Add Input' -> search for "
        f"\"{slug}\" -> Add.",
        "Run `python scripts/verify_data_root.py` in the notebook to prove the "
        "session can read the data.",
        "For any later re-upload use `python scripts/kaggle_upload.py version "
        "-m \"what changed\"` -- never `upload` again, which would fail as a "
        "duplicate slug.",
    )
    return 0


def cmd_verify_remote(args, cfg, logger) -> int:
    """Check the uploaded dataset against the local staging folder, file by file.

    Answers "did all of it actually arrive, intact?" without a browser and
    without downloading 3.6 GB back. Run it after ``upload`` and after every
    ``version`` push.

    Compares the Kaggle file listing against ``checksums.csv`` in the staging
    folder on three axes: every local file is present remotely, no remote file is
    unaccounted for, and every size matches. Sizes are what catch a truncated
    transfer -- a half-written ``.npz`` keeps its name and would otherwise pass
    a file-count check.

    Two differences are expected and are reported as such rather than as
    failures:

    - ``dataset-metadata.json`` is consumed by Kaggle as metadata and is not
      stored as a data file;
    - ``checksums.csv`` cannot appear in its own listing.
    """
    username = preflight(cfg, logger)
    slug = validate_slug(str(cfg.kaggle.dataset_slug), "kaggle.dataset_slug")
    full_slug = f"{username}/{slug}"
    staging_dir = repo_root() / str(cfg.kaggle.staging_dir)
    checksums = staging_dir / "checksums.csv"

    print()
    print("=" * 78)
    print("  VERIFYING THE UPLOADED DATASET")
    print("=" * 78)
    print(f"\n  Dataset: {full_slug}")
    print(f"  Against: {checksums}")

    if not checksums.is_file():
        raise UploadError(
            f"No local manifest to compare against: {checksums} does not exist.\n"
            "\n"
            "`verify-remote` compares Kaggle's file listing against the "
            "checksums.csv written by `stage`. If the staging folder has been "
            "deleted, re-create it with:\n"
            "  python scripts/kaggle_upload.py stage\n"
            "That rebuilds the same folder from the same cache and does not "
            "upload anything."
        )

    with checksums.open("r", newline="", encoding="utf-8") as handle:
        local = {
            row["relative_path"]: int(row["size_bytes"])
            for row in csv.DictReader(handle)
        }
    print(f"  Local manifest lists {len(local):,} file(s).")

    rule("asking Kaggle what it has")
    remote = list_remote_files(cfg, logger, full_slug, with_sizes=True)
    if remote is None:
        raise UploadError(
            f"Kaggle did not return a file listing for {full_slug}.\n"
            "\n"
            "Either the dataset is still being processed (wait a minute and "
            "retry), or it does not exist under that name. Check "
            "cfg.kaggle.dataset_slug, and confirm the dataset at "
            f"https://www.kaggle.com/datasets/{full_slug}"
        )
    print(f"  Kaggle reports {len(remote):,} file(s).")

    # Expected asymmetries, named rather than silently filtered.
    expected_absent = {"dataset-metadata.json"}
    expected_extra = {"checksums.csv"}

    missing = sorted(set(local) - set(remote) - expected_absent)
    extra = sorted(set(remote) - set(local) - expected_extra)
    shared = set(local) & set(remote)
    mismatched = sorted(n for n in shared if local[n] != remote[n])

    subset = str(cfg.sen2naipv2.subset) if cfg.dataset.name == "sen2naipv2" else str(
        cfg.dataset.name
    )
    prefix = f"{subset}/"
    remote_npz = [n for n in remote if n.lower().endswith(".npz")]
    flat_npz = [n for n in remote_npz if not n.replace("\\", "/").startswith(prefix)]

    rule("comparison")
    print(f"  .npz files on Kaggle:        {len(remote_npz):,}")
    print(f"  nested under '{prefix}': {len(remote_npz) - len(flat_npz):,}")
    print(f"  flat at the top level:       {len(flat_npz):,}")
    print(f"  missing on Kaggle:           {len(missing):,}")
    print(f"  unaccounted for on Kaggle:   {len(extra):,}")
    print(f"  size mismatches (truncated): {len(mismatched):,}")
    print(
        f"  total bytes local / remote:  {human_bytes(sum(local.values()))} / "
        f"{human_bytes(sum(remote.values()))}"
    )

    for label, entries in (
        ("MISSING", missing),
        ("UNACCOUNTED FOR", extra),
        ("SIZE MISMATCH", mismatched),
    ):
        if entries:
            print(f"\n  {label} ({len(entries):,}):")
            for name in entries[:10]:
                if label == "SIZE MISMATCH":
                    print(f"    {name}: local {local[name]:,} B, "
                          f"remote {remote[name]:,} B")
                else:
                    print(f"    {name}")
            if len(entries) > 10:
                print(f"    ... and {len(entries) - 10:,} more")

    print()
    if not missing and not extra and not mismatched and not flat_npz:
        print("=" * 78)
        print("  PASS -- the upload is complete and intact.")
        print("=" * 78)
        print(
            f"\n  All {len(shared):,} shared files match on size, every .npz is "
            f"nested under\n  '{prefix}', and nothing is missing.\n"
            "\n"
            "  The two expected differences were accounted for:\n"
            "    - dataset-metadata.json: consumed by Kaggle as metadata, not "
            "stored as data;\n"
            "    - checksums.csv: cannot list itself."
        )
        logger.info("Remote verification PASSED for %s", full_slug)
        next_steps(
            "Attach it in a notebook: sidebar -> '+ Add Input' -> search for "
            f"\"{slug}\" -> Add.",
            "In the notebook, run:  python scripts/verify_data_root.py",
            "Then run the baseline to confirm the numbers reproduce on Kaggle.",
        )
        return 0

    print("=" * 78)
    print("  FAIL -- the uploaded dataset does not match the staging folder.")
    print("=" * 78)
    if flat_npz:
        print(
            f"\n  {len(flat_npz):,} .npz file(s) are flat instead of nested under "
            f"'{prefix}'.\n"
            "  The loader appends the subset name to the mount, so it would find "
            "nothing."
        )
    if missing or mismatched:
        print(
            "\n  Files are missing or truncated. The transfer did not complete "
            "cleanly.\n"
            "  Push the staging folder again as a new version:\n"
            '    python scripts/kaggle_upload.py version -m "re-upload after '
            'incomplete transfer"'
        )
    logger.error("Remote verification FAILED for %s", full_slug)
    return 1


def cmd_version(args, cfg, logger) -> int:
    """Push a new version of an existing dataset."""
    username = preflight(cfg, logger)
    slug = validate_slug(str(cfg.kaggle.dataset_slug), "kaggle.dataset_slug")
    title = validate_title(str(cfg.kaggle.dataset_title), "kaggle.dataset_title")
    staging_dir = repo_root() / str(cfg.kaggle.staging_dir)

    print()
    print("=" * 78)
    print("  NEW VERSION OF AN EXISTING DATASET")
    print("=" * 78)
    print(f"\n  Dataset: {username}/{slug}")
    print(f"  Message: {args.message}")
    print(
        "\n  This adds a version to the dataset that is already on Kaggle rather\n"
        "  than creating a new one, so the slug, the URL, and every notebook\n"
        "  that already attaches it keep working. Notebooks pick up the new\n"
        "  version the next time they are run."
    )

    result = build_staging(cfg, logger, staging_dir, slug, title, username)

    rule("confirm")
    print(f"  {result['file_count']:,} files, "
          f"{human_bytes(result['total_bytes'])}")
    for line in upload_estimates(cfg, result["total_bytes"]):
        print(line)
    print("\n  !! Like the first upload, this is NOT resumable.")

    if args.smoke:
        refuse_network_in_smoke("version")
        next_steps("Re-run without --smoke to push the new version.")
        return 0

    answer = "YES"
    if args.yes:
        print("\n  --yes given: skipping the confirmation prompt.")
    else:
        print()
        try:
            answer = input("  Type YES (in capitals) to push this version: ").strip()
        except EOFError:
            raise UploadError(
                "No answer received -- this command needs an interactive "
                "terminal. Run it directly in your terminal, or pass --yes for "
                "an unattended run."
            )
    if answer != "YES":
        print(f"\n  You typed {answer!r}, not 'YES'. Nothing was uploaded.")
        next_steps("Nothing changed on Kaggle. Re-run when ready.")
        return 1

    upload_log = Path(resolve_output_path(cfg, "upload_log"))
    progress = get_logger("upload_progress", log_file=upload_log)

    rule("uploading the new version")
    print(f"  Progress is being written to {upload_log}")
    print(f"  Watch it with:  Get-Content -Path {upload_log} -Wait -Tail 20")
    print()
    progress.info("=" * 70)
    progress.info("VERSION UPLOAD STARTED")
    progress.info("  dataset: %s/%s", username, slug)
    progress.info("  message: %s", args.message)
    progress.info("  files:   %d (%s)", result["file_count"],
                  human_bytes(result["total_bytes"]))

    started = time.perf_counter()
    try:
        run_kaggle(
            [
                "datasets", "version",
                "-p", str(result["dir"]),
                "-m", str(args.message),
                "--dir-mode", "zip",
                "--keep-tabular",
            ],
            timeout_s=int(cfg.kaggle.cli_timeout_s),
            logger=logger,
            what="the version upload",
        )
    except UploadError as exc:
        progress.error("VERSION UPLOAD FAILED after %s",
                       human_duration(time.perf_counter() - started))
        for line in str(exc).splitlines():
            progress.error("  %s", line)
        raise
    progress.info("VERSION UPLOAD FINISHED in %s",
                  human_duration(time.perf_counter() - started))
    progress.info("=" * 70)

    url = f"https://www.kaggle.com/datasets/{username}/{slug}"
    print(f"\n  New version accepted. Dataset page: {url}")
    next_steps(
        f"Open {url} -> 'Data Explorer' and confirm the version number went up.",
        "Any notebook that already attaches this dataset picks up the new "
        "version on its next run -- no need to re-attach.",
        "Run `python scripts/verify_data_root.py` in a notebook to confirm the "
        "new contents are readable.",
    )
    return 0


# -- entry point -----------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish the local SEN2NAIPv2 sample cache to Kaggle as a Dataset, "
            "so GPU notebooks read it from a mount instead of re-downloading it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical order:\n"
            "  python scripts/kaggle_upload.py survey\n"
            "  python scripts/kaggle_upload.py stage\n"
            "  python scripts/kaggle_upload.py dryrun\n"
            "  python scripts/kaggle_upload.py upload --yes\n"
            "  python scripts/kaggle_upload.py verify-remote\n"
            "\n"
            "Later re-uploads:\n"
            "  python scripts/kaggle_upload.py version -m \"added 500 tiles\"\n"
            "  python scripts/kaggle_upload.py verify-remote\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    definitions = [
        ("survey", "Inspect the local cache and judge whether its size is sane."),
        ("stage", "Build the upload folder. Does not contact Kaggle."),
        ("dryrun", "Upload a 5-sample test dataset to validate the settings."),
        ("upload", "The real upload, behind a typed YES confirmation."),
        ("version", "Push a new version of the existing dataset."),
        (
            "verify-remote",
            "Check the uploaded dataset against the local staging folder "
            "(file list and sizes). Run after upload and after every version.",
        ),
    ]
    for name, help_text in definitions:
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        add_standard_args(sub)
        if name == "version":
            sub.add_argument(
                "-m",
                "--message",
                required=True,
                help="What changed in this version. Shown in the dataset's "
                "version history.",
            )
        if name in ("upload", "version"):
            sub.add_argument(
                "--yes",
                action="store_true",
                help="Skip the interactive YES confirmation. For unattended "
                "runs in a dedicated window. The transfer is still NOT "
                "resumable -- progress goes to outputs/upload.log so it can be "
                "tailed from elsewhere.",
            )
    return parser.parse_args(argv)


HANDLERS = {
    "survey": cmd_survey,
    "stage": cmd_stage,
    "dryrun": cmd_dryrun,
    "upload": cmd_upload,
    "version": cmd_version,
    "verify-remote": cmd_verify_remote,
}


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("kaggle_upload", log_file=cfg.paths.log_file)
    seed_everything(cfg.seed)

    if args.smoke:
        ensure_smoke_cache(cfg, logger)

    try:
        return HANDLERS[args.command](args, cfg, logger)
    except UploadError as exc:
        print()
        print("=" * 78)
        print("  STOPPED")
        print("=" * 78)
        print()
        for line in str(exc).splitlines():
            print(f"  {line}" if line else "")
        print()
        logger.error("%s failed: %s", args.command, str(exc).splitlines()[0])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
