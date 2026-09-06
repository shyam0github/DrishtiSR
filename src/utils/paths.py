"""Filesystem root resolution.

The repository must behave identically whether it runs from a local Windows
checkout (``D:/SIH/DrishtiSR``) or from a Kaggle notebook (``/kaggle/working``
with inputs mounted read-only under ``/kaggle/input``). Every path in the project
is derived from one of the two functions here; no module may hardcode a root.
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "resolve_data_root",
    "repo_root",
    "resolve_cache_dir",
    "resolve_output_path",
    "kaggle_mount_path",
    "kaggle_mount_candidates",
]

# Order matters: Kaggle wins, so a notebook never silently trains off a stale
# local mirror that happens to be visible through a mounted drive.
_CANDIDATE_KEYS = ("kaggle_data_root", "local_data_root")
_OVERRIDE_KEY = "data_root"


def repo_root() -> Path:
    """Absolute path to the repository root.

    Derived from this file's location (``<repo>/src/utils/paths.py``), so it is
    correct under any checkout location without configuration. Relative paths in
    ``configs/*.yaml`` (``outputs``, ``outputs/cache``, ...) are interpreted
    against this directory.
    """
    return Path(__file__).resolve().parents[2]


def resolve_data_root(cfg: Any) -> Path:
    """Resolve the dataset root directory for the current machine.

    Resolution order:

    1. ``cfg.paths.data_root`` -- an explicit override. If set (non-null) it wins
       unconditionally and must exist.
    2. **The mounted Kaggle Dataset**, at any of the layouts
       ``paths.kaggle_mount_patterns`` describes -- the cache published by
       ``scripts/kaggle_upload.py``. MEASURED: the live layout is
       ``/kaggle/input/datasets/<owner>/<slug>``, not the ``/kaggle/input/<slug>``
       this project originally assumed, so both are tried. Skipped when the mount
       keys are absent, and **rejected when a directory exists but is empty**.
    3. ``cfg.paths.kaggle_data_root`` -- normally ``/kaggle/input``.
    4. ``cfg.paths.local_data_root`` -- the local Windows mirror.

    The first candidate that is an existing, non-empty directory is returned.
    There is no fallback beyond the list and no directory is created: a missing
    data root is a hard error, because every downstream failure mode it produces
    (empty dataset, zero-length loader, silently skipped split) is far harder to
    debug than an exception here.

    Why the emptiness check on candidate 2 specifically
    ---------------------------------------------------
    ``/kaggle/input`` exists on **every** Kaggle notebook, whether or not a
    dataset is attached. Resolving to it unconditionally is the single most
    dangerous thing this function could do: the run starts, the loader reports
    zero samples, training "succeeds" in nine seconds, and nothing anywhere says
    the data was never mounted. Composing the full mount path and requiring it to
    contain something turns that silent no-op into an exception naming the
    dataset that was expected -- which is the whole reason the slug lives in
    config rather than being assumed.

    Args:
        cfg: The loaded config. Any mapping with a ``paths`` section works --
            an ``omegaconf.DictConfig`` or a plain ``dict`` -- which keeps this
            function testable without OmegaConf.

    Returns:
        An absolute, resolved :class:`~pathlib.Path` to an existing directory.

    Raises:
        KeyError: ``cfg`` has no ``paths`` section, or a required candidate key
            (``kaggle_data_root``, ``local_data_root``) is absent.
        ValueError: A candidate is present but empty/blank.
        NotADirectoryError: The explicit ``data_root`` override does not exist.
        FileNotFoundError: No candidate resolved. The message lists **every**
            path that was checked and why each one was rejected -- missing, not a
            directory, or present but empty -- so the failure is diagnosable from
            the traceback alone without re-running anything.
    """
    paths = _require_section(cfg, "paths")

    override = _get(paths, _OVERRIDE_KEY, required=False)
    if override is not None:
        candidate = _as_path(override, f"paths.{_OVERRIDE_KEY}")
        if not candidate.is_dir():
            raise NotADirectoryError(
                f"paths.{_OVERRIDE_KEY} is set to {str(candidate)!r} but that is not "
                "an existing directory. Clear the override to fall back to "
                "automatic Kaggle/local detection."
            )
        return candidate.resolve()

    candidates: list = []
    for mount in kaggle_mount_candidates(cfg):
        candidates.append(("paths.kaggle_mount_root/kaggle_dataset_dir", mount))
    candidates += [
        (f"paths.{key}", _as_path(_get(paths, key, required=True), f"paths.{key}"))
        for key in _CANDIDATE_KEYS
    ]

    rejections = []
    for label, candidate in candidates:
        if not candidate.exists():
            rejections.append(f"{label}={str(candidate)!r} (does not exist)")
            continue
        if not candidate.is_dir():
            rejections.append(f"{label}={str(candidate)!r} (exists but is not a directory)")
            continue
        if _is_empty_dir(candidate):
            # Only the composed Kaggle mount is rejected for being empty. The
            # other candidates keep their historical "exists is enough" contract:
            # a local data root is legitimately empty before the first download.
            if label.startswith("paths.kaggle_mount_root"):
                rejections.append(
                    f"{label}={str(candidate)!r} (mounted but EMPTY -- the "
                    "dataset is attached under the wrong name, is still syncing, "
                    "or was uploaded empty)"
                )
                continue
        return candidate.resolve()

    tried = "\n  - ".join(rejections)
    raise FileNotFoundError(
        "No usable data root found. Checked, in order:\n  - "
        + tried
        + "\n\nOn Kaggle: open the notebook sidebar, click '+ Add Input', and "
        "attach the dataset whose name matches paths.kaggle_dataset_dir "
        f"({_get(paths, 'kaggle_dataset_dir', required=False)!r}); build and push "
        "it with `python scripts/kaggle_upload.py upload`.\n"
        "Locally: create the directory, or point paths.local_data_root at it.\n"
        "Either way, run `python scripts/verify_data_root.py` to confirm the "
        "data is visible before starting a training run."
    )


def kaggle_mount_candidates(cfg: Any) -> list:
    """Every path a Kaggle Dataset might be mounted at, in preference order.

    There is more than one layout, which is the whole reason this function
    replaced a single composed path.

    MEASURED on Kaggle, 2026-09-06: a dataset attached to a notebook through
    ``kernel-metadata.json``'s ``dataset_sources`` appears at
    ``/kaggle/input/datasets/<owner>/<slug>`` -- **with** the owner segment. The
    older, widely documented layout is ``/kaggle/input/<slug>``, with no owner.
    This project assumed the second, and its own source comment asserted that
    "the mount uses the dataset name only -- there is no owner prefix". That was
    wrong, and the cost of being wrong is a session that finds no data.

    So both layouts are tried. The patterns live in
    ``paths.kaggle_mount_patterns`` and may contain ``{root}``, ``{name}`` and
    shell globs; the owner segment is a ``*`` so that no Kaggle username is
    written into version control.

    Args:
        cfg: The loaded config, or any mapping with a ``paths`` section.

    Returns:
        Candidate paths, most-preferred first. Glob patterns contribute only
        paths that currently exist; literal patterns are always included so a
        failure message can name what was looked for. Empty when
        ``kaggle_mount_root`` or ``kaggle_dataset_dir`` is absent.

    Raises:
        KeyError: ``cfg`` has no ``paths`` section.
        ValueError: A key is present but blank.
    """
    paths = _require_section(cfg, "paths")
    root = _get(paths, "kaggle_mount_root", required=False)
    name = _get(paths, "kaggle_dataset_dir", required=False)
    if root is None or name is None:
        return []

    root_text = str(_as_path(root, "paths.kaggle_mount_root")).replace("\\", "/").rstrip("/")
    name_text = str(_as_path(name, "paths.kaggle_dataset_dir"))

    patterns = _get(paths, "kaggle_mount_patterns", required=False) or ["{root}/{name}"]
    candidates: list = []
    for pattern in patterns:
        text = str(pattern).format(root=root_text, name=name_text)
        if any(character in text for character in "*?["):
            candidates.extend(Path(match) for match in sorted(_glob.glob(text)))
        else:
            candidates.append(Path(text))
    return candidates


def kaggle_mount_path(cfg: Any) -> Optional[Path]:
    """The Kaggle Dataset mount, or the path it was expected at.

    Prefers a candidate that exists and is non-empty; failing that, returns the
    first literal candidate so callers can say what they looked for. See
    :func:`kaggle_mount_candidates` for why there is more than one candidate.

    Args:
        cfg: The loaded config, or any mapping with a ``paths`` section.

    Returns:
        The mounted path when one is usable, else the first expected path, else
        ``None`` when the config names no mount at all -- the correct state for a
        config that predates the upload tooling.

    Raises:
        KeyError: ``cfg`` has no ``paths`` section.
        ValueError: A key is present but blank.
    """
    candidates = kaggle_mount_candidates(cfg)
    for candidate in candidates:
        if candidate.is_dir() and not _is_empty_dir(candidate):
            return candidate
    return candidates[0] if candidates else None


def _is_empty_dir(path: Path) -> bool:
    """True when ``path`` is a directory containing nothing.

    Errors are not swallowed: an unreadable directory raises, because "I could
    not tell" must never be reported as "it is fine".
    """
    return not any(path.iterdir())


def resolve_output_path(cfg: Any, key: str) -> Path:
    """Resolve a writable path from ``cfg.paths[key]``, creating its directory.

    Relative values resolve against :func:`repo_root`, so ``outputs/metrics``
    means the same thing on ``D:`` and under ``/kaggle/working``.

    Args:
        cfg: The loaded config.
        key: A key in the ``paths`` section, e.g. ``"manifest_dir"``.

    Returns:
        An absolute path. If the value looks like a directory (no suffix) the
        directory is created; otherwise the parent directory is created.

    Raises:
        KeyError: ``paths[key]`` is absent.
        ValueError: The value is empty.
    """
    paths = _require_section(cfg, "paths")
    path = _as_path(_get(paths, key, required=True), f"paths.{key}")
    if not path.is_absolute():
        path = repo_root() / path
    target = path if not path.suffix else path.parent
    target.mkdir(parents=True, exist_ok=True)
    return path


def resolve_cache_dir(cfg: Any) -> Path:
    """Resolve the sample cache directory.

    Resolution order:

    1. **The mounted Kaggle Dataset**, when any layout in
       ``paths.kaggle_mount_patterns`` exists and is non-empty. This is the whole point of ``scripts/kaggle_upload.py``: the
       ~3000 SEN2NAIPv2 pairs are already there, so ``is_cached()`` returns True
       for every sample and the notebook never re-downloads (MEASURED at
       ~10.4 s/sample, i.e. ~8.7 hours of session time). The mount is
       **read-only** and is returned without any attempt to create it.
    2. ``cfg.paths.kaggle_cache_dir`` when ``/kaggle`` exists -- normally
       ``/kaggle/temp``, which is fast and does not count against the 20 GB
       ``/kaggle/working`` output quota. This is the path a *download* run on
       Kaggle uses, i.e. when no dataset has been attached yet.
    3. ``cfg.paths.cache_dir`` -- the local cache. Relative values resolve
       against :func:`repo_root`.

    Detection keys off directory existence, not an environment variable, because
    the Kaggle notebook environment does not set a reliable one.

    Cases 2 and 3 create the directory if missing -- unlike the data root, a
    writable cache is ours to create. Case 1 never creates anything.

    Args:
        cfg: The loaded config.

    Returns:
        An absolute path to an existing directory. Writable in cases 2 and 3;
        **read-only** in case 1, which is correct -- everything is already
        cached, so nothing needs to write.
    """
    paths = _require_section(cfg, "paths")

    for mount in kaggle_mount_candidates(cfg):
        if mount.is_dir() and not _is_empty_dir(mount):
            return mount.resolve()

    if Path("/kaggle").is_dir():
        kaggle_cache = _get(paths, "kaggle_cache_dir", required=False)
        if kaggle_cache is not None:
            path = _as_path(kaggle_cache, "paths.kaggle_cache_dir")
            path.mkdir(parents=True, exist_ok=True)
            return path.resolve()

    path = _as_path(_get(paths, "cache_dir", required=True), "paths.cache_dir")
    if not path.is_absolute():
        path = repo_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _require_section(cfg: Any, name: str) -> Any:
    try:
        section = cfg[name]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"Config is missing the {name!r} section. Load configs/base.yaml "
            "(or a config that merges over it) rather than building one by hand."
        ) from exc
    if section is None:
        raise KeyError(f"Config section {name!r} is null; it must be a mapping.")
    return section


def _get(section: Any, key: str, *, required: bool) -> Any:
    try:
        return section[key]
    except (KeyError, TypeError) as exc:
        if required:
            raise KeyError(
                f"Config is missing 'paths.{key}'. It is required by "
                "resolve_data_root(); see configs/base.yaml."
            ) from exc
        return None


def _as_path(value: Any, label: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is empty. Set it to a directory or leave it null.")
    return Path(text)
