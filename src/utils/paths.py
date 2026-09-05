"""Filesystem root resolution.

The repository must behave identically whether it runs from a local Windows
checkout (``D:/SIH/DrishtiSR``) or from a Kaggle notebook (``/kaggle/working``
with inputs mounted read-only under ``/kaggle/input``). Every path in the project
is derived from one of the two functions here; no module may hardcode a root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["resolve_data_root", "repo_root", "resolve_cache_dir", "resolve_output_path"]

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
    2. ``cfg.paths.kaggle_data_root`` -- normally ``/kaggle/input``.
    3. ``cfg.paths.local_data_root`` -- the local Windows mirror.

    The first candidate that is an existing directory is returned. There is no
    fallback beyond the list and no directory is created: a missing data root is
    a hard error, because every downstream failure mode it produces (empty
    dataset, zero-length loader, silently skipped split) is far harder to debug
    than an exception here.

    Args:
        cfg: The loaded config. Any mapping with a ``paths`` section works --
            an ``omegaconf.DictConfig`` or a plain ``dict`` -- which keeps this
            function testable without OmegaConf.

    Returns:
        An absolute, resolved :class:`~pathlib.Path` to an existing directory.

    Raises:
        KeyError: ``cfg`` has no ``paths`` section, or a candidate key is absent.
        ValueError: A candidate is present but empty/blank.
        NotADirectoryError: The explicit ``data_root`` override does not exist.
        FileNotFoundError: Neither candidate root exists on this machine.
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

    candidates = [
        (key, _as_path(_get(paths, key, required=True), f"paths.{key}"))
        for key in _CANDIDATE_KEYS
    ]

    for _key, candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    tried = ", ".join(f"paths.{key}={str(path)!r}" for key, path in candidates)
    raise FileNotFoundError(
        f"No data root found. Tried, in order: {tried}. "
        "On Kaggle, attach the dataset so it mounts under /kaggle/input. "
        "Locally, create the directory or point paths.local_data_root at it."
    )


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
    """Resolve the sample cache directory, preferring Kaggle's scratch space.

    On Kaggle, ``/kaggle/temp`` is fast and does not count against the 20 GB
    ``/kaggle/working`` output quota, so a multi-GB image cache belongs there and
    not in the repository. Detection keys off the existence of ``/kaggle``, not
    off an environment variable, because the notebook environment does not set a
    reliable one.

    Falls back to ``cfg.paths.cache_dir`` (relative values resolve against
    :func:`repo_root`). The directory is created if missing -- unlike the data
    root, a cache is ours to create.

    Args:
        cfg: The loaded config.

    Returns:
        An absolute path to an existing, writable directory.
    """
    paths = _require_section(cfg, "paths")

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
