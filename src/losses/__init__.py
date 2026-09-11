"""Reconstruction, spectral-consistency, and heteroscedastic NLL objectives.

``from src.losses import spectral_consistency`` is the import every caller uses.
The implementation lives in :mod:`src.losses.spectral` rather than in a flat
``src/losses.py`` because a module and a package of the same name cannot coexist
-- the package wins on import and the module becomes dead code that still passes
review. Re-exporting here keeps the one-line import the design called for.
"""

from src.losses.nll import gaussian_nll, nll_objective, nll_weight_at, sr_grad_ratio
from src.losses.spectral import (
    ANTIALIASED_MODES,
    DEFAULT_COS_CLAMP,
    DEFAULT_DOWNSAMPLE,
    DEFAULT_EPS,
    DEFAULT_LAMBDA1,
    DEFAULT_LAMBDA2,
    DEFAULT_MIN_NORM,
    REJECTED_MODES,
    down4,
    spectral_consistency,
    spectral_settings_from_cfg,
    spectral_terms,
)

__all__ = [
    "ANTIALIASED_MODES",
    "REJECTED_MODES",
    "DEFAULT_EPS",
    "DEFAULT_COS_CLAMP",
    "DEFAULT_MIN_NORM",
    "DEFAULT_DOWNSAMPLE",
    "DEFAULT_LAMBDA1",
    "DEFAULT_LAMBDA2",
    "down4",
    "spectral_terms",
    "spectral_consistency",
    "spectral_settings_from_cfg",
    "gaussian_nll",
    "nll_objective",
    "nll_weight_at",
    "sr_grad_ratio",
]
