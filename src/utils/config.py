"""Config loading: one place that knows how ``--config`` and ``--smoke`` compose."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from src.utils.paths import repo_root

__all__ = ["load_config", "add_standard_args"]

DEFAULT_CONFIG = "configs/base.yaml"


def load_config(
    path: Any = DEFAULT_CONFIG,
    smoke: bool = False,
    overrides: Optional[Sequence[str]] = None,
) -> DictConfig:
    """Load a config, optionally applying smoke mode and CLI overrides.

    Composition order, later winning: the file, then ``cfg.smoke`` when
    ``smoke`` is set, then ``overrides``.

    Smoke mode is a config merge rather than a set of ``if`` branches scattered
    through the scripts. That way ``--smoke`` exercises exactly the same code
    path as a real run, which is the only way it can serve as a pre-flight check
    before spending GPU hours.

    Args:
        path: Path to the YAML config. Relative paths resolve against the
            repository root.
        smoke: Merge the ``smoke`` block over the config.
        overrides: OmegaConf dotlist entries, e.g. ``["train.epochs=2"]``.

    Returns:
        The composed :class:`~omegaconf.DictConfig`.

    Raises:
        FileNotFoundError: The config file does not exist.
        KeyError: ``smoke`` was requested but the config has no ``smoke`` block.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config not found: {config_path}. Pass --config with a path "
            f"relative to the repository root, e.g. {DEFAULT_CONFIG}."
        )

    cfg = OmegaConf.load(config_path)

    if smoke:
        if "smoke" not in cfg:
            raise KeyError(
                f"--smoke was requested but {config_path} has no 'smoke' block. "
                "Add one, or merge over configs/base.yaml which does."
            )
        cfg = OmegaConf.merge(cfg, cfg.smoke)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    return cfg


def add_standard_args(parser) -> None:
    """Attach the ``--config`` / ``--smoke`` / ``--set`` flags every script takes.

    Args:
        parser: An :class:`argparse.ArgumentParser`.
    """
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to the YAML config (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast end-to-end check on CPU with no network access: merges the "
            "config's 'smoke' block, which selects the synthetic stub dataset."
        ),
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        # action="extend" so REPEATED --set flags accumulate. With plain "store"
        # (argparse's default) a second --set silently replaces the first, so
        #     --set a.b=1 --set c.d=2
        # would apply only c.d=2 and drop a.b=1 with no warning -- the caller
        # sees a successful run configured differently from what they typed.
        action="extend",
        default=None,
        metavar="KEY=VALUE",
        help="OmegaConf dotlist overrides, e.g. --set train.epochs=2. May be "
        "given more than once; all values are applied, later ones winning.",
    )
