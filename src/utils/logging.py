"""Run logging: timestamped output to stdout and to a persistent log file.

Kaggle discards notebook cell output once a session ends, so every run also
writes to ``outputs/run.log`` inside the repository, which survives as a notebook
output artefact.

Note: Python 3 uses absolute imports, so this module shadowing the stdlib
``logging`` name inside ``src.utils`` is safe -- ``import logging`` below still
resolves to the standard library.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

from src.utils.paths import repo_root

__all__ = ["get_logger"]

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LOG_FILE = "outputs/run.log"


def get_logger(
    name: str = "drishtisr",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger writing timestamped records to stdout and a file.

    Handlers are attached once per logger name; calling this repeatedly with the
    same name returns the same logger without duplicating output. Propagation to
    the root logger is disabled so records are not emitted twice under pytest or
    inside a Jupyter kernel.

    Args:
        name: Logger name. Use the module or script name so log lines are
            attributable, e.g. ``get_logger("train")``.
        log_file: Destination file. Relative paths resolve against the repository
            root, so the same config works on ``D:`` and under ``/kaggle/working``.
            Defaults to ``outputs/run.log``. Pass ``cfg.paths.log_file`` from
            entry-point scripts. The parent directory is created if needed.
        level: Threshold for both handlers.

    Returns:
        The configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    path = Path(log_file) if log_file is not None else Path(_DEFAULT_LOG_FILE)
    if not path.is_absolute():
        path = repo_root() / path
    path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
