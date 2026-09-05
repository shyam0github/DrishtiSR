"""Reflectance-domain image quality and spectral-fidelity metrics.

``image_quality`` holds the metric functions (PSNR, SSIM, LPIPS, SAM, ERGAS), all
operating on float32 surface reflectance in ``(C, H, W)`` or ``(B, C, H, W)``.
``aggregate`` holds :class:`~src.metrics.aggregate.Evaluator`, which runs any
``sr_fn(lr) -> sr`` over a dataloader and produces the per-sample table and the
summary. Every reported number in the project comes through these two modules, so
a baseline, an ablation, and the quantised ONNX model are measured identically.
"""

from src.metrics.aggregate import EvaluationResult, Evaluator, summarise
from src.metrics.image_quality import (
    LPIPS_CAVEAT,
    BandMetric,
    LPIPSResult,
    SAMResult,
    ergas,
    lpips,
    psnr,
    rgb_band_indices,
    sam,
    ssim,
)

__all__ = [
    "BandMetric",
    "EvaluationResult",
    "Evaluator",
    "LPIPSResult",
    "LPIPS_CAVEAT",
    "SAMResult",
    "ergas",
    "lpips",
    "psnr",
    "rgb_band_indices",
    "sam",
    "ssim",
    "summarise",
]
