"""Uncertainty package (P6): learned Laplace scale head, NLL, collapse watch, calibration.

COMPATIBILITY. This package shadows the older module ``src/uncertainty.py``
(TTA helpers: ``tta_predict``, ``sr_output``, ``TTAResult``), which
``tests/test_tta.py`` and ``scripts/calibrate_uncertainty.py`` import as
``src.uncertainty``. That file is loaded here by path and its public names are
re-exported unchanged, so those imports keep working.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_LEGACY_NAME = "src._uncertainty_legacy"
if _LEGACY_NAME in sys.modules:
    _legacy = sys.modules[_LEGACY_NAME]
else:
    _spec = importlib.util.spec_from_file_location(
        _LEGACY_NAME, Path(__file__).resolve().parent.parent / "uncertainty.py")
    _legacy = importlib.util.module_from_spec(_spec)
    sys.modules[_LEGACY_NAME] = _legacy
    _spec.loader.exec_module(_legacy)

TTAResult = _legacy.TTAResult
tta_predict = _legacy.tta_predict
sr_output = _legacy.sr_output

__all__ = ["TTAResult", "tta_predict", "sr_output"]
