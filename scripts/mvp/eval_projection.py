"""P5 add-on (Novelty 1): does consistency projection pass its pre-registered gate?

200 VAL patches (seed 26142, same pool as calibrate_trust_scales.py). A2
single pass vs projected SR (sr <- sr + U(lr - D(sr)), iters 1, 2, 3).
Per variant: LPIPS, PSNR, SSIM, SAM deg, spec_l1 (training operator), HF
energy ratio vs GT, and opensr-test ``reflectance`` consistency if the
wrapper's projected runtime is under 10 minutes.

Gate (pre-registered, never tuned), on VAL means:
    LPIPS_proj <= 1.01 * LPIPS_A2  AND  HF_proj >= 0.95 * HF_A2
    AND  spec_l1_proj <= 0.5 * spec_l1_A2
Smallest passing iters is chosen. Writes reports/mvp/projection_gate.json.

Run: python scripts/mvp/eval_projection.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_trust_scales import A2_CKPT, SEED, WORKTREE, ValPatches, provenance  # noqa: E402

from src.eval.opensr_harness import build_metrics, score_arrays  # noqa: E402
from src.infer.predictor import TorchPredictor  # noqa: E402
from src.infer.trust import hf_energy, project_consistency, spectral_scalars  # noqa: E402
from src.metrics.image_quality import ergas, lpips, psnr, sam, ssim  # noqa: E402

OUT = WORKTREE / "reports" / "mvp" / "projection_gate.json"
N = 200
ITERS = (1, 2, 3)
OPENSR_BUDGET_S = 600.0
GATE = {"lpips_max_ratio": 1.01, "hf_min_ratio": 0.95, "spec_l1_max_ratio": 0.5}


def variants(sr: np.ndarray, lr: np.ndarray) -> Dict[str, np.ndarray]:
    out = {"a2": sr}
    for k in ITERS:
        out[f"proj{k}"] = project_consistency(sr, lr, k)
    return out


def main() -> None:
    torch.set_num_threads(6)
    pred = TorchPredictor(A2_CKPT, threads=6, interim=False)
    val = ValPatches()
    cfg = val.cfg
    scale = int(cfg.sr.scale)
    patches = val.draw(N)
    names = ["a2"] + [f"proj{k}" for k in ITERS]
    rows: Dict[str, List[Dict[str, float]]] = {n: [] for n in names}
    srs: Dict[str, List[np.ndarray]] = {n: [] for n in names}
    hrs = []

    t0 = time.perf_counter()
    for p in patches:
        lr, hr = p["lr"], p["hr"]
        sr = np.asarray(pred(lr[None]), dtype=np.float32)[0, :4]
        hf_hr = hf_energy(hr)
        hrs.append(hr)
        for name, x in variants(sr, lr).items():
            l1, sam_spec = spectral_scalars(x, lr)
            rows[name].append({
                "psnr": float(psnr(x, hr, data_range=1.0).mean),
                "ssim": float(ssim(x, hr, data_range=1.0).mean),
                "sam_deg": float(sam(x, hr).mean_deg),
                "ergas": float(ergas(x, hr, scale=scale)),
                "spec_l1": l1, "spec_sam_deg": sam_spec,
                "hf_ratio_vs_gt": hf_energy(x) / hf_hr if hf_hr > 0 else float("nan"),
            })
            srs[name].append(x)
    hr_b = np.stack(hrs)
    for name in names:
        d = lpips(np.stack(srs[name]), hr_b, rgb_indices=(0, 1, 2)).distance
        for r, v in zip(rows[name], np.asarray(d).ravel()):
            r["lpips"] = float(v)
    core_s = time.perf_counter() - t0

    # opensr-test consistency, only if it fits the 10-minute budget.
    opensr: Dict[str, Any] = {"ran": False}
    try:
        metrics, settings = build_metrics(cfg)
        probe = 3
        t1 = time.perf_counter()
        for p in patches[:probe]:
            score_arrays(p["lr"], srs["a2"][0], p["hr"], cfg, metrics=metrics, settings=settings)
        projected = (time.perf_counter() - t1) / probe * N * len(names)
        opensr["projected_seconds"] = projected
        if projected < OPENSR_BUDGET_S:
            for name in names:
                vals = [score_arrays(p["lr"], x, p["hr"], cfg, metrics=metrics,
                                     settings=settings)["reflectance"]
                        for p, x in zip(patches, srs[name])]
                for r, v in zip(rows[name], vals):
                    r["opensr_reflectance"] = float(v)
            opensr["ran"] = True
        else:
            opensr["reason"] = f"projected {projected:.0f}s > {OPENSR_BUDGET_S:.0f}s budget"
    except Exception as exc:  # noqa: BLE001 -- optional measure; failure is recorded, not fatal
        opensr["reason"] = f"{type(exc).__name__}: {exc}"

    summary = {n: {k: float(np.nanmean([r[k] for r in rows[n]])) for k in rows[n][0]}
               for n in names}
    base = summary["a2"]
    checks: Dict[str, Any] = {}
    chosen = None
    for k in ITERS:
        s = summary[f"proj{k}"]
        c = {"lpips_ratio": s["lpips"] / base["lpips"],
             "hf_ratio": s["hf_ratio_vs_gt"] / base["hf_ratio_vs_gt"],
             "spec_l1_ratio": s["spec_l1"] / base["spec_l1"]}
        c["passed"] = bool(c["lpips_ratio"] <= GATE["lpips_max_ratio"]
                           and c["hf_ratio"] >= GATE["hf_min_ratio"]
                           and c["spec_l1_ratio"] <= GATE["spec_l1_max_ratio"])
        checks[f"proj{k}"] = c
        if c["passed"] and chosen is None:
            chosen = k

    report = {
        "what": "P5 Novelty 1 consistency projection gate: A2 single pass vs "
                "sr + U(lr - D(sr)) iterated, D = down4 area, U = eval bicubic",
        "passed": chosen is not None,
        "iters": chosen,
        "gate": {"rule": "LPIPS_proj <= 1.01*LPIPS_A2 AND HF_proj >= 0.95*HF_A2 AND "
                         "spec_l1_proj <= 0.5*spec_l1_A2; smallest passing iters; pre-registered",
                 **GATE, "per_variant": checks},
        "metrics": summary,
        "metric_notes": {"hf_ratio_vs_gt": "per-patch hf_energy_ratio(x)/hf_energy_ratio(HR), mean",
                         "spec_l1": "training operator (down4 area) vs LR, reflectance",
                         "opensr_reflectance": "opensr-test reflectance consistency (headline measure)"},
        "opensr": opensr,
        "n": N, "n_samples": N,
        "core_seconds": core_s,
        **provenance(pred),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
