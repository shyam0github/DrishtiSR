"""The frozen training config and its hash.

WHY THIS EXISTS. ``configs/frozen_day3.yaml`` is the complete definition of a
training run. A file is only a freeze if something checks it, so this module is
that check: :func:`load_frozen` loads the file, recomputes the SHA256 over its
contents, and RAISES rather than returning a config whose ``config_hash`` no
longer matches. Runs log the hash into their metadata JSON, which is what lets a
checkpoint six weeks old be tied to the exact settings that made it.

This is deliberately separate from :mod:`src.utils.config`, which composes
``--config`` and ``--smoke`` over ``configs/base.yaml`` for the data pipeline.
That layer is meant to be overridden from the command line; this one is meant to
be impossible to override by accident.

CANONICALISATION. The hash is taken over compact, sorted-key JSON of the parsed
config with ``config_hash`` removed -- not over the file bytes. So reflowing a
comment, reordering keys, or changing quote style does not invalidate a run's
provenance, while changing ``weight_decay`` from 0.0 to 1e-4 does. YAML scalars
are parsed before hashing, so ``2.0e-4`` and ``0.0002`` hash identically: the
hash tracks the VALUES, which is the property that matters when the question is
"was this checkpoint trained with the settings I think it was".

Run as a module to inspect or update the recorded hash::

    .venv\\Scripts\\python.exe -m src.config           # print, verify
    .venv\\Scripts\\python.exe -m src.config --write   # rewrite config_hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from src.utils.paths import repo_root

__all__ = [
    "FROZEN_CONFIG",
    "HASH_FIELD",
    "FrozenConfigError",
    "canonical_bytes",
    "compute_config_hash",
    "load_frozen",
]

FROZEN_CONFIG = "configs/frozen_day3.yaml"

# The one field excluded from its own hash. Named once so the loader, the
# writer and the error messages cannot disagree about which key that is.
HASH_FIELD = "config_hash"


class FrozenConfigError(RuntimeError):
    """The frozen config is missing, unhashed, or does not match its hash."""


def _resolve(path: Any) -> Path:
    """Resolve a config path against the repository root.

    Args:
        path: Absolute path, or one relative to the repository root.

    Returns:
        An absolute :class:`~pathlib.Path`. Existence is NOT checked here; the
        caller reports it with its own message.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = repo_root() / resolved
    return resolved


def canonical_bytes(cfg: Any) -> bytes:
    """Serialise a config to the exact bytes the hash is taken over.

    Canonical form is UTF-8 compact JSON with sorted keys and the
    :data:`HASH_FIELD` entry removed from the top level. JSON rather than YAML
    because YAML has several spellings for the same value and this must have
    one; sorted keys so a reordered block is not a different config; the field
    removed so the hash can be stored inside the thing it hashes.

    Args:
        cfg: A ``DictConfig``, a plain mapping, or anything
            ``OmegaConf.to_container`` accepts. Interpolations are resolved
            first, so a config whose value is computed hashes as the value it
            actually resolves to.

    Returns:
        The canonical byte string. Deterministic across runs, platforms and
        Python versions -- no ``hash()``, no ``id()``, no set iteration.

    Raises:
        TypeError: The config holds a value JSON cannot represent, which for
            YAML input means a date or a binary scalar. Quote it in the YAML.
    """
    container = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    if not isinstance(container, dict):
        raise TypeError(
            f"a frozen config must be a mapping at the top level, got "
            f"{type(container).__name__}."
        )
    payload = {k: v for k, v in container.items() if k != HASH_FIELD}
    try:
        text = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    except TypeError as exc:
        raise TypeError(
            f"{FROZEN_CONFIG} holds a value that is not JSON-serialisable and "
            f"so cannot be hashed: {exc}. Quote it in the YAML (dates and "
            f"binary scalars are the usual culprits)."
        ) from exc
    return text.encode("utf-8")


def compute_config_hash(cfg: Any) -> str:
    """SHA256 of :func:`canonical_bytes`, as 64 lowercase hex characters.

    Args:
        cfg: As :func:`canonical_bytes`.

    Returns:
        The hex digest.

    Raises:
        TypeError: Propagated from :func:`canonical_bytes`.
    """
    return hashlib.sha256(canonical_bytes(cfg)).hexdigest()


def load_frozen(
    path: Any = FROZEN_CONFIG,
    verify: bool = True,
) -> DictConfig:
    """Load the frozen training config, verifying its hash.

    Args:
        path: Path to the frozen YAML, absolute or relative to the repository
            root. Defaults to :data:`FROZEN_CONFIG`.
        verify: Check the recorded hash against the recomputed one. Leave this
            True. It exists only so the ``--write`` path in ``__main__`` can
            read a file whose hash is deliberately stale; a training run that
            passes ``verify=False`` has opted out of the entire point of the
            file.

    Returns:
        The config as a :class:`~omegaconf.DictConfig`, ``config_hash``
        INCLUDED so the caller can write ``cfg.config_hash`` straight into its
        run-metadata JSON without recomputing it.

    Raises:
        FileNotFoundError: No file at ``path``.
        FrozenConfigError: The file has no ``config_hash``, its value is empty
            or not a 64-character hex digest, or it does not match the hash
            recomputed from the contents -- i.e. the config was edited without
            being re-frozen.
    """
    config_path = _resolve(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Frozen config not found: {config_path}. Every training run reads "
            f"one; if this is a new experiment, copy {FROZEN_CONFIG} and freeze "
            f"the copy with 'python -m src.config --write'."
        )

    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise FrozenConfigError(
            f"{config_path} must contain a YAML mapping at the top level."
        )

    if not verify:
        return cfg

    recorded = cfg.get(HASH_FIELD, None)
    if recorded is None:
        raise FrozenConfigError(
            f"{config_path} has no '{HASH_FIELD}' field, so it is not frozen. "
            f"Add it with 'python -m src.config --write'."
        )
    recorded = str(recorded).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", recorded):
        raise FrozenConfigError(
            f"{config_path}: '{HASH_FIELD}' is {recorded!r}, which is not a "
            f"64-character SHA256 hex digest. Recompute it with "
            f"'python -m src.config --write'."
        )

    actual = compute_config_hash(cfg)
    if actual != recorded:
        raise FrozenConfigError(
            f"{config_path} does not match its recorded hash -- the config was "
            f"edited without being re-frozen.\n"
            f"  recorded:   {recorded}\n"
            f"  recomputed: {actual}\n"
            f"Every run that cited the recorded hash was trained under "
            f"DIFFERENT settings from the ones now in this file. If the edit "
            f"was intended, re-freeze with 'python -m src.config --write' and "
            f"say in the commit message which reported numbers it invalidates. "
            f"If it was not, 'git checkout' the file."
        )
    return cfg


def _write_hash(path: Path) -> str:
    """Rewrite the ``config_hash`` line in place and return the new digest.

    A line-level substitution rather than a YAML round-trip: OmegaConf's dumper
    would discard every comment in the file, and in this repository the comments
    are the record of WHY each frozen value is what it is.

    Args:
        path: Absolute path to the frozen YAML.

    Returns:
        The newly computed 64-character hex digest.

    Raises:
        FrozenConfigError: The file has no top-level ``config_hash:`` line to
            replace, or has more than one.
    """
    cfg = load_frozen(path, verify=False)
    digest = compute_config_hash(cfg)

    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{HASH_FIELD}:.*$", re.MULTILINE)
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise FrozenConfigError(
            f"expected exactly one top-level '{HASH_FIELD}:' line in {path}, "
            f"found {len(matches)}."
        )
    path.write_text(
        pattern.sub(f'{HASH_FIELD}: "{digest}"', text), encoding="utf-8"
    )
    return digest


def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print, verify, or rewrite a frozen config's SHA256."
    )
    parser.add_argument(
        "--config",
        default=FROZEN_CONFIG,
        help=f"Frozen config to act on (default: {FROZEN_CONFIG}).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the config_hash field from the current contents.",
    )
    # Accepted and ignored: every entry point in this repo takes --smoke, and
    # hashing a file is already a two-millisecond CPU operation, so smoke mode
    # is the same code path. Rejecting the flag would be the surprise.
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="No-op here; accepted so this behaves like every other entry point.",
    )
    args = parser.parse_args(argv)

    path = _resolve(args.config)
    if args.write:
        digest = _write_hash(path)
        print(f"{path.name}: config_hash written")
        print(digest)
        return 0

    cfg = load_frozen(path)
    print(f"{path.name}: hash verified")
    print(str(cfg[HASH_FIELD]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
