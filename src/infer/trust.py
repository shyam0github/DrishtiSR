"""Reference-free trust signals for one SR prediction (P5).

Given an LR patch and any :class:`src.infer.predictor.Predictor`, compute the
single-pass SR, the project bicubic baseline, a per-pixel uncertainty map and a
per-pixel spectral-consistency map, plus the scalar summaries the API reports
under ``metrics.reference_free`` (docs/mvp/api_contract.md).

Everything is SURFACE REFLECTANCE, float32, unclipped, band order
``cfg.dataset.bands`` (B04, B03, B02, B08). Nothing here re-implements a
project operator: the degradation operator is ``src.losses.spectral.down4``
(exact 4x4 block mean), the bicubic is ``src.eval.baselines.bicubic_upsample``
(antialias on, as eval), the sharpness proxy is
``src.metrics.sharpness.hf_energy_ratio`` (cutoff 0.25 x Nyquist).

Uncertainty, in order of preference:
- ``predictor.info["has_scale_head"]``: the packed Laplace scale channels
  (method ``"learned_laplace"``).
- ``tta > 0``: per-band std (ddof 0) across dihedral test-time augmentations
  (``"tta4"``: identity, hflip, vflip, rot180; ``"tta8"``: full D4). The
  identity member IS the single-pass SR, reused rather than recomputed. TTA
  outputs are used only for the uncertainty map; the SR shown is always the
  single identity pass.
- otherwise none (``"none"``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from drishtisr.eval.baselines import bicubic_upsample
    from drishtisr.losses.spectral import (DEFAULT_COS_CLAMP, DEFAULT_DOWNSAMPLE, DEFAULT_EPS,
                                           DEFAULT_MIN_NORM, down4, spectral_terms)
    from drishtisr.metrics.sharpness import DEFAULT_HF_CUTOFF, hf_energy_ratio
except ImportError:
    from src.eval.baselines import bicubic_upsample  # type: ignore
    from src.losses.spectral import (DEFAULT_COS_CLAMP, DEFAULT_DOWNSAMPLE, DEFAULT_EPS,  # type: ignore
                                     DEFAULT_MIN_NORM, down4, spectral_terms)
    from src.metrics.sharpness import DEFAULT_HF_CUTOFF, hf_energy_ratio  # type: ignore

SCALE = 4
N_BANDS = 4
# Eval's bicubic (cfg.baseline.antialias: true).
BICUBIC_ANTIALIAS = True

# A D4 element as (k, flip): rotate by k*90 deg over (H, W), then mirror the W axis.
Transform = Tuple[int, bool]
D4: Tuple[Transform, ...] = tuple((k, f) for k in range(4) for f in (False, True))
# identity, hflip, rot180, rot180+hflip (= vflip)
TTA4: Tuple[Transform, ...] = ((0, False), (0, True), (2, False), (2, True))
TTA_SETS = {4: TTA4, 8: D4}


def apply_transform(x: np.ndarray, t: Transform) -> np.ndarray:
    """Apply a D4 element to the last two axes of ``x``."""
    k, flip = t
    y = np.rot90(x, k, axes=(-2, -1))
    if flip:
        y = y[..., ::-1]
    return np.ascontiguousarray(y)


def invert_transform(y: np.ndarray, t: Transform) -> np.ndarray:
    """Exact inverse of :func:`apply_transform` (undo the flip, then the rotation)."""
    k, flip = t
    if flip:
        y = y[..., ::-1]
    return np.ascontiguousarray(np.rot90(y, -k, axes=(-2, -1)))


@dataclass
class TrustResult:
    sr: np.ndarray                   # (4, H, W) float32 reflectance, single pass (projected if asked)
    bicubic: np.ndarray              # (4, H, W) float32 reflectance
    unc: Optional[np.ndarray]        # (H, W) float32, band mean of unc_bands
    unc_bands: Optional[np.ndarray]  # (4, H, W) float32
    cons_l1: np.ndarray              # (h, w) float32, band mean |D(SR) - LR|
    cons_sam_deg: np.ndarray         # (h, w) float32, spectral angle D(SR) vs LR, degrees
    scalars: Dict[str, Any] = field(default_factory=dict)
    timings_ms: Dict[str, Any] = field(default_factory=dict)
    method: str = "none"


# ------------------------------------------------------------ operators
def degrade(x: np.ndarray) -> np.ndarray:
    """Project degradation operator D (down4, area) on ``(C, H, W)`` or ``(N, C, H, W)``."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    batched = t.ndim == 4
    out = down4(t if batched else t[None], scale=SCALE, mode=DEFAULT_DOWNSAMPLE)
    return (out if batched else out[0]).numpy()


def upsample(x: np.ndarray) -> np.ndarray:
    """Project bicubic x4 U (eval settings) on ``(C, h, w)`` or ``(N, C, h, w)``."""
    return np.asarray(bicubic_upsample(np.ascontiguousarray(x, dtype=np.float32), SCALE,
                                       antialias=BICUBIC_ANTIALIAS), dtype=np.float32)


def project_consistency(sr: np.ndarray, lr: np.ndarray, iters: int) -> np.ndarray:
    """Back-project SR onto the LR measurement: ``sr <- sr + U(lr - D(sr))``, ``iters`` times.

    ``sr`` ``(C, H, W)`` and ``lr`` ``(C, H/4, W/4)``, reflectance, float32.
    ``iters == 0`` returns a float32 copy of ``sr``.
    """
    out = np.array(sr, dtype=np.float32, copy=True)
    lr = np.asarray(lr, dtype=np.float32)
    for _ in range(int(iters)):
        out = (out + upsample(lr - degrade(out))).astype(np.float32, copy=False)
    return out


def spectral_maps(sr: np.ndarray, lr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-LR-pixel ``(|D(sr) - lr| band mean, spectral angle in degrees)``.

    The angle uses spectral_terms' numerics (eps inside the norm, cosine clamp)
    and reports 0 where either spectrum is below ``DEFAULT_MIN_NORM`` (nodata).
    """
    down = degrade(sr).astype(np.float64)
    ref = np.asarray(lr, dtype=np.float64)
    l1 = np.abs(down - ref).mean(axis=0)
    eps_sq = DEFAULT_EPS ** 2
    dot = (down * ref).sum(axis=0)
    nd = np.sqrt((down ** 2).sum(axis=0) + eps_sq)
    nr = np.sqrt((ref ** 2).sum(axis=0) + eps_sq)
    lim = 1.0 - DEFAULT_COS_CLAMP
    ang = np.degrees(np.arccos(np.clip(dot / (nd * nr), -lim, lim)))
    ang = np.where((nd > DEFAULT_MIN_NORM) & (nr > DEFAULT_MIN_NORM), ang, 0.0)
    return l1.astype(np.float32), ang.astype(np.float32)


def spectral_scalars(sr: np.ndarray, lr: np.ndarray) -> Tuple[float, float]:
    """``(l1_spec, sam_deg)`` from the training operator's spectral_terms."""
    terms = spectral_terms(torch.from_numpy(np.ascontiguousarray(sr, dtype=np.float32))[None],
                           torch.from_numpy(np.ascontiguousarray(lr, dtype=np.float32))[None],
                           scale=SCALE)
    return float(terms["l1_spec"]), float(terms["sam"]) * 180.0 / math.pi


def hf_energy(x: np.ndarray) -> float:
    """Project HF energy ratio (cutoff 0.25 x Nyquist) of one ``(C, H, W)`` image."""
    return float(hf_energy_ratio(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                                 cutoff=DEFAULT_HF_CUTOFF)[0])


# ------------------------------------------------------------ uncertainty
def _tta_outputs(lr: np.ndarray, predictor: Any, transforms: Tuple[Transform, ...],
                 identity_sr: np.ndarray) -> List[np.ndarray]:
    """SR (first 4 channels) for each transform, mapped back to the identity frame.

    The identity member reuses ``identity_sr``. Remaining transforms are grouped
    by transformed shape (rot90/270 swap H and W on non-square inputs) and each
    group is one predictor call.
    """
    outs: Dict[Transform, np.ndarray] = {}
    groups: Dict[Tuple[int, ...], List[Transform]] = {}
    for t in transforms:
        if t == (0, False):
            outs[t] = identity_sr
            continue
        groups.setdefault(apply_transform(lr, t).shape, []).append(t)
    for ts in groups.values():
        batch = np.stack([apply_transform(lr, t) for t in ts]).astype(np.float32, copy=False)
        pred = np.asarray(predictor(batch), dtype=np.float32)[:, :N_BANDS]
        for t, y in zip(ts, pred):
            outs[t] = invert_transform(y, t)
    return [outs[t] for t in transforms]


# ------------------------------------------------------------ main entry
def compute_trust(lr: np.ndarray, predictor: Any, tta: int, hr: Optional[np.ndarray] = None,
                  project_iters: int = 0) -> TrustResult:
    """Trust maps and scalars for one LR patch.

    Args:
        lr: ``(4, h, w)`` float32 reflectance.
        predictor: any :class:`src.infer.predictor.Predictor`.
        tta: 0, 4 or 8. Ignored when the predictor has a scale head.
        hr: optional ``(4, 4h, 4w)`` reference; adds ``spec_l1_hr``,
            ``spec_sam_hr_deg``, ``hf_ratio_hr_vs_bicubic``.
        project_iters: >0 applies :func:`project_consistency` to the single-pass
            SR before the consistency, sharpness and GT metrics; bicubic and the
            TTA uncertainty are unaffected. Its time is counted in ``sr``.
    """
    if tta not in (0, 4, 8):
        raise ValueError(f"tta must be 0, 4 or 8; got {tta!r}.")
    lr = np.ascontiguousarray(lr, dtype=np.float32)
    if lr.ndim != 3 or lr.shape[0] != N_BANDS:
        raise ValueError(f"lr must be ({N_BANDS}, h, w); got {lr.shape}.")
    t_start = time.perf_counter()

    out = np.asarray(predictor(lr[None]), dtype=np.float32)[0]
    sr = np.ascontiguousarray(out[:N_BANDS])
    has_head = bool(predictor.info.get("has_scale_head", False))
    if project_iters > 0:
        sr_measured = project_consistency(sr, lr, project_iters)
    else:
        sr_measured = sr
    t_sr = time.perf_counter()

    unc_bands: Optional[np.ndarray] = None
    t_unc: Optional[float] = None
    if has_head and out.shape[0] >= 2 * N_BANDS:
        unc_bands = np.ascontiguousarray(out[N_BANDS:2 * N_BANDS])
        method = "learned_laplace"
        t_unc = 0.0  # produced by the same forward pass
    elif tta > 0:
        t0 = time.perf_counter()
        stack = np.stack(_tta_outputs(lr, predictor, TTA_SETS[tta], sr))
        unc_bands = stack.std(axis=0, ddof=0).astype(np.float32)
        method = f"tta{tta}"
        t_unc = (time.perf_counter() - t0) * 1e3
    else:
        method = "none"
    unc = unc_bands.mean(axis=0).astype(np.float32) if unc_bands is not None else None

    bicubic = upsample(lr)
    cons_l1, cons_sam = spectral_maps(sr_measured, lr)
    spec_l1, spec_sam = spectral_scalars(sr_measured, lr)
    spec_l1_b, spec_sam_b = spectral_scalars(bicubic, lr)
    hf_bic = hf_energy(bicubic)
    scalars: Dict[str, Any] = {
        "spec_l1": spec_l1,
        "spec_sam_deg": spec_sam,
        "spec_l1_bicubic": spec_l1_b,
        "spec_sam_bicubic_deg": spec_sam_b,
        "hf_ratio_vs_bicubic": hf_energy(sr_measured) / hf_bic if hf_bic > 0 else float("nan"),
        "unc_mean": float(unc.mean()) if unc is not None else None,
        "unc_p95": float(np.percentile(unc, 95)) if unc is not None else None,
    }
    if hr is not None:
        hr = np.ascontiguousarray(hr, dtype=np.float32)
        l1_hr, sam_hr = spectral_scalars(hr, lr)
        scalars.update(spec_l1_hr=l1_hr, spec_sam_hr_deg=sam_hr,
                       hf_ratio_hr_vs_bicubic=hf_energy(hr) / hf_bic if hf_bic > 0 else float("nan"))

    timings = {"sr": (t_sr - t_start) * 1e3, "uncertainty": t_unc,
               "total": (time.perf_counter() - t_start) * 1e3}
    return TrustResult(sr=sr_measured, bicubic=bicubic, unc=unc, unc_bands=unc_bands,
                       cons_l1=cons_l1, cons_sam_deg=cons_sam, scalars=scalars,
                       timings_ms=timings, method=method)
