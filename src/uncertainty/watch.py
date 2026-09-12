"""Collapse watch for the learned scale head (P6). Thresholds are pre-registered.

Every ``CHECK_EVERY`` iterations up to ``ABORT_AT``, on a fixed 128-patch VAL
subset (seed 26142, monitoring only -- nothing is selected on it):

- ``rho``: Spearman rank correlation of band-mean ``b`` vs band-mean ``|err|``
  on ``N_PIXELS`` pixels sampled with a fixed seed;
- ``frac_floor``: share of per-band pixel values with ``b < FLOOR``
  (reflectance units);
- ``spatial_cv``: per-image std / mean of the band-mean ``b`` map, averaged.

At ``ABORT_AT`` the run is declared "collapsed" (and stopped; TTA fallback
recommended) if ``rho < RHO_MIN`` or ``frac_floor > FRAC_FLOOR_MAX`` or
``spatial_cv < SPATIAL_CV_MIN``. These values are fixed here and never tuned.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["CHECK_EVERY", "ABORT_AT", "N_PIXELS", "SEED", "FLOOR", "RHO_MIN", "FRAC_FLOOR_MAX",
           "SPATIAL_CV_MIN", "watch_stats", "abort_reasons", "CollapseWatch"]

CHECK_EVERY = 250
ABORT_AT = 2000
N_PIXELS = 200_000
SEED = 26142
FLOOR = 2e-4
RHO_MIN = 0.10
FRAC_FLOOR_MAX = 0.20
SPATIAL_CV_MIN = 0.05


def watch_stats(b: np.ndarray, err: np.ndarray, n_pixels: int = N_PIXELS, seed: int = SEED,
                floor: float = FLOOR) -> Dict[str, float]:
    """Collapse statistics. ``b``, ``err``: ``(N, C, H, W)``, reflectance; ``err`` = |mu - y|."""
    from scipy.stats import spearmanr

    b = np.asarray(b, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    if b.shape != err.shape or b.ndim != 4:
        raise ValueError(f"b and err must share an (N, C, H, W) shape; got {b.shape}, {err.shape}.")
    bm, em = b.mean(axis=1).ravel(), err.mean(axis=1).ravel()
    k = min(int(n_pixels), bm.size)
    pick = np.random.default_rng(seed).choice(bm.size, size=k, replace=False)
    if np.ptp(bm[pick]) == 0.0 or np.ptp(em[pick]) == 0.0:
        rho = 0.0  # a constant map has no rank information
    else:
        rho = float(spearmanr(bm[pick], em[pick]).statistic)
    per_img = b.mean(axis=1).reshape(b.shape[0], -1)
    cv = per_img.std(axis=1) / np.maximum(per_img.mean(axis=1), 1e-12)
    return {"rho": rho, "frac_floor": float((b < floor).mean()), "spatial_cv": float(cv.mean()),
            "b_mean": float(b.mean()), "err_mean": float(err.mean()), "n_pixels": k}


def abort_reasons(stats: Dict[str, float]) -> List[str]:
    """Pre-registered abort conditions that ``stats`` meets (empty = pass)."""
    out = []
    if not stats["rho"] >= RHO_MIN:
        out.append(f"rho {stats['rho']:.4f} < {RHO_MIN}")
    if stats["frac_floor"] > FRAC_FLOOR_MAX:
        out.append(f"frac_floor {stats['frac_floor']:.4f} > {FRAC_FLOOR_MAX}")
    if not stats["spatial_cv"] >= SPATIAL_CV_MIN:
        out.append(f"spatial_cv {stats['spatial_cv']:.4f} < {SPATIAL_CV_MIN}")
    return out


class CollapseWatch:
    """Runs the checks on cached backbone outputs and logs each to a JSONL file.

    ``feats`` / ``mu`` are the frozen backbone's outputs on the VAL subset,
    computed once (the backbone cannot change), so a check only re-runs the head.
    """

    def __init__(self, feats: Any, mu: Any, hr: Any, log_path: Path, run: str,
                 meta: Optional[Dict[str, Any]] = None, every: int = CHECK_EVERY,
                 until: int = ABORT_AT) -> None:
        self.feats, self.mu, self.hr = feats, mu, hr
        self.err = (mu - hr).abs().numpy()
        self.log_path, self.run, self.meta = Path(log_path), run, dict(meta or {})
        self.every, self.until = int(every), int(until)
        self.history: List[Dict[str, Any]] = []
        self.verdict: str = "not_evaluated"

    def due(self, it: int) -> bool:
        return 0 < it <= self.until and it % self.every == 0

    def check(self, head: Any, it: int, batch: int = 16) -> Tuple[Dict[str, Any], bool]:
        """Evaluate at 1-based iteration ``it``; returns ``(record, abort)``."""
        import torch

        t0 = time.perf_counter()
        was_training = head.training
        head.eval()
        with torch.no_grad():
            b = torch.cat([head(self.feats[i:i + batch]) for i in range(0, len(self.feats), batch)])
        head.train(was_training)
        stats = watch_stats(b.numpy(), self.err)
        final = it >= self.until
        reasons = abort_reasons(stats) if final else []
        if final:
            self.verdict = "collapsed" if reasons else "passed"
        rec = {"run": self.run, "iter": int(it), **stats, "final_check": final,
               "abort_reasons": reasons, "verdict": self.verdict if final else None,
               "check_seconds": time.perf_counter() - t0,
               "thresholds": {"rho_min": RHO_MIN, "frac_floor_max": FRAC_FLOOR_MAX,
                              "spatial_cv_min": SPATIAL_CV_MIN, "floor": FLOOR},
               **self.meta}
        self.history.append(rec)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec, bool(reasons)
