"""The sigma-collapse monitor: summary statistics of the log-variance raster.

Sigma collapse is the Day 4 kill criterion. A collapsed head emits (nearly) the
same log-variance everywhere -- a constant map carries no information about
where the model is wrong, however good its mean. The statistics below are
logged at every validation so the collapse is visible while the run is live:

- ``mean``, ``min``, ``max`` over every validation pixel and band;
- ``spatial_std``: the standard deviation over ``(H, W)`` of each
  ``(image, band)`` map, averaged over maps. THE collapse signal -- a constant
  map scores 0 no matter where the constant sits;
- ``clamp_lo_frac`` / ``clamp_hi_frac``: the fraction of values pinned at the
  clamp bounds. Pinned values have zero gradient; a head migrating wholesale
  onto a bound is collapsing onto the clamp rather than onto a constant.

Accumulated in float64 so a 25-batch validation does not lose the small
spatial std of a nearly collapsed head to rounding.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

__all__ = ["LogvarMonitor", "LOGVAR_STAT_KEYS"]

LOGVAR_STAT_KEYS = ("mean", "min", "max", "spatial_std", "clamp_lo_frac", "clamp_hi_frac")


class LogvarMonitor:
    """Streaming log-variance statistics over validation batches.

    Args:
        logvar_min: The lower clamp the model applies.
        logvar_max: The upper clamp the model applies.
    """

    def __init__(self, logvar_min: float, logvar_max: float) -> None:
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self._sum = 0.0
        self._count = 0
        self._min = math.inf
        self._max = -math.inf
        self._std_sum = 0.0
        self._maps = 0
        self._lo = 0
        self._hi = 0

    @torch.no_grad()
    def update(self, logvar: torch.Tensor) -> None:
        """Fold in one batch.

        Args:
            logvar: ``(B, C, H, W)``, any float dtype, any device. Natural-log
                variance of the reflectance, as the model emits it (clamped).

        Raises:
            ValueError: ``logvar`` is not 4-D, or holds a non-finite value --
                a NaN here means the head diverged, and averaging it away
                would hide the one event this monitor exists to catch.
        """
        if logvar.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(logvar.shape)}")
        lv = logvar.detach().double()
        if not torch.isfinite(lv).all():
            raise ValueError("log-variance contains non-finite values; the head diverged.")
        self._sum += float(lv.sum())
        self._count += lv.numel()
        self._min = min(self._min, float(lv.min()))
        self._max = max(self._max, float(lv.max()))
        per_map = lv.flatten(2).std(dim=2, correction=0)
        self._std_sum += float(per_map.sum())
        self._maps += per_map.numel()
        self._lo += int((lv <= self.logvar_min).sum())
        self._hi += int((lv >= self.logvar_max).sum())

    def summary(self) -> Dict[str, float]:
        """The statistics named in :data:`LOGVAR_STAT_KEYS`.

        Raises:
            RuntimeError: Nothing was accumulated; an empty monitor reporting
                zeros would read as a perfectly collapsed head.
        """
        if self._count == 0:
            raise RuntimeError("LogvarMonitor.summary() called before any update().")
        return {
            "mean": self._sum / self._count,
            "min": self._min,
            "max": self._max,
            "spatial_std": self._std_sum / self._maps,
            "clamp_lo_frac": self._lo / self._count,
            "clamp_hi_frac": self._hi / self._count,
        }
