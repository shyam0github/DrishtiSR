"""Reflectance-domain image quality and spectral-fidelity metrics.

``image_quality`` holds the metric functions (PSNR, SSIM, LPIPS, SAM, ERGAS), all
operating on float32 surface reflectance in ``(C, H, W)`` or ``(B, C, H, W)``.
``sharpness`` holds the two blur proxies -- mean Sobel gradient magnitude and
high-frequency energy ratio -- logged at every validation so a spectral run that
buys consistency with blur is visible while it is still training rather than
afterwards. ``aggregate`` holds :class:`~src.metrics.aggregate.Evaluator`, which runs any
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
from src.metrics.sharpness import (
    DEFAULT_HF_CUTOFF,
    gradient_magnitude,
    hf_energy_ratio,
    reference_sharpness,
    sharpness_terms,
)

__all__ = [
    "BandMetric",
    "DEFAULT_HF_CUTOFF",
    "EvaluationResult",
    "Evaluator",
    "LPIPSResult",
    "LPIPS_CAVEAT",
    "SAMResult",
    "ergas",
    "gradient_magnitude",
    "hf_energy_ratio",
    "lpips",
    "psnr",
    "reference_sharpness",
    "rgb_band_indices",
    "sam",
    "sharpness_terms",
    "ssim",
    "summarise",
]
