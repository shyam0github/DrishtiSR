"""Method-agnostic uncertainty calibration (P6): any (err, unc) pair scores the same way.

- AUSE: sparsification curve on band-mean absolute error -- remove the fraction
  ``f`` of pixels with the highest uncertainty, mean error of the rest -- minus
  the oracle curve (remove by error itself), integrated over ``N_FRACTIONS``
  removal fractions ``f = k / N_FRACTIONS``, k = 0..N-1 (trapezoid). 0 = perfect
  ranking. Also reported relative to the mean error.
- Spearman rho of band-mean unc vs band-mean |err| (sampled pixels).
- Laplace interval coverage: |err| <= b ln 2 (nominal 50 %) and b ln 10
  (nominal 90 %), per band and pixel. Only meaningful when ``unc`` IS a
  Laplace scale; TTA std is not, so its coverage is reported as "N/A".
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

__all__ = ["N_FRACTIONS", "sparsification_curve", "ause", "spearman", "laplace_coverage",
           "calibration_report"]

N_FRACTIONS = 20


def _band_mean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x.mean(axis=-3) if x.ndim >= 3 else x).ravel()


def sparsification_curve(err: np.ndarray, score: np.ndarray, n_fracs: int = N_FRACTIONS) -> np.ndarray:
    """Mean err of the pixels kept after removing the top-``f`` by ``score``, per fraction."""
    err, score = np.asarray(err, np.float64).ravel(), np.asarray(score, np.float64).ravel()
    order = np.argsort(-score, kind="stable")
    # suffix means: mean of err[order[k:]] for every k
    tail = np.cumsum(err[order][::-1])[::-1]
    n = err.size
    out = []
    for k in range(n_fracs):
        start = int(math.floor(k / n_fracs * n))
        out.append(tail[start] / (n - start))
    return np.asarray(out)


def ause(err: np.ndarray, unc: np.ndarray, n_fracs: int = N_FRACTIONS) -> Dict[str, Any]:
    """AUSE on band-mean |err| (inputs ``(..., C, H, W)`` or flat)."""
    e, u = _band_mean(err), _band_mean(unc)
    fr = np.arange(n_fracs) / n_fracs
    c_unc, c_orc = sparsification_curve(e, u, n_fracs), sparsification_curve(e, e, n_fracs)
    area = float(np.trapz(c_unc - c_orc, fr))
    return {"ause": area, "ause_rel": area / float(e.mean()) if e.mean() > 0 else float("nan"),
            "fractions": fr.tolist(), "curve_unc": c_unc.tolist(), "curve_oracle": c_orc.tolist()}


def spearman(err: np.ndarray, unc: np.ndarray, n_pixels: int = 200_000, seed: int = 26142) -> float:
    from scipy.stats import spearmanr

    e, u = _band_mean(err), _band_mean(unc)
    k = min(int(n_pixels), e.size)
    pick = np.random.default_rng(seed).choice(e.size, size=k, replace=False)
    return float(spearmanr(u[pick], e[pick]).statistic)


def laplace_coverage(err: np.ndarray, b: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Empirical vs nominal coverage of central Laplace intervals (per band, per pixel)."""
    e, b = np.abs(np.asarray(err, np.float64)), np.asarray(b, np.float64)
    return {"50": {"nominal": 0.5, "empirical": float((e <= b * math.log(2.0)).mean())},
            "90": {"nominal": 0.9, "empirical": float((e <= b * math.log(10.0)).mean())}}


def calibration_report(err: np.ndarray, unc: np.ndarray, laplace: bool) -> Dict[str, Any]:
    """AUSE, rho and (Laplace only) coverage for per-band ``err``/``unc`` of equal shape."""
    err, unc = np.abs(np.asarray(err)), np.asarray(unc)
    if err.shape != unc.shape:
        raise ValueError(f"err {err.shape} and unc {unc.shape} differ.")
    a = ause(err, unc)
    return {**a, "spearman_rho": spearman(err, unc),
            "coverage": laplace_coverage(err, unc) if laplace else "N/A",
            "mean_abs_err": float(err.mean()), "mean_unc": float(unc.mean())}
