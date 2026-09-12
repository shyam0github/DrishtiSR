"""Inference pipeline behind POST /api/upscale (docs/mvp/api_contract.md v1).

:class:`Engine` loads the predictor once (backend chosen from env and the P4
quant gate), then :meth:`Engine.run` does one job: ``compute_trust`` (P5) ->
PNG renders (``src.infer.render``) -> float32 GeoTIFFs -> GT metrics via the
project Evaluator (identical LPIPS band handling to eval) -> the contract
response. Job files live in ``app/_jobs/<job_id>/``; only the newest 50 jobs
are kept.

Environment:
    DRISHTI_CKPT      checkpoint path (default: A2 last.pt, inventory section 2)
    DRISHTI_BACKEND   auto | onnx-int8 | onnx-fp32 | torch   (default auto)
    DRISHTI_THREADS   CPU threads (default 6)
    DRISHTI_PROJECT   "0" disables the consistency projection even if its gate passed
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from app.io_geotiff import Profile, profile_from_json, write_sr_tif, write_unc_tif
from src.infer import render, trust
from src.infer.predictor import TorchPredictor, checkpoint_id
from src.metrics.aggregate import Evaluator
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.paths import repo_root

LOGGER = get_logger("drishtisr.app.pipeline")

ROOT = repo_root()
APP_DIR = ROOT / "app"
JOBS_DIR = APP_DIR / "_jobs"
SAMPLES_DIR = APP_DIR / "samples"
REPORTS = ROOT / "reports" / "mvp"
DEFAULT_CKPT = Path(r"D:\SIH\DrishtiSR\runs\day3\a2\last.pt")
BACKENDS = ("auto", "onnx-int8", "onnx-fp32", "torch")
KEEP_JOBS = 50
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

# Fixed refs (contract): HR-in-place-of-SR floors on full VAL and the blur guard.
SPEC_L1_GT_FLOOR = 0.005733
SPEC_SAM_GT_FLOOR_DEG = 1.2748
SHARPNESS_WARN_BELOW = 1.05


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _finite(x: Any) -> Optional[float]:
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


class Engine:
    """Predictor + display scales + projection setting, loaded once per process."""

    def __init__(self) -> None:
        self.ckpt = Path(os.environ.get("DRISHTI_CKPT", str(DEFAULT_CKPT)))
        requested = os.environ.get("DRISHTI_BACKEND", "auto").strip().lower()
        if requested not in BACKENDS:
            raise ValueError(f"DRISHTI_BACKEND must be one of {BACKENDS}; got {requested!r}.")
        self.threads = int(os.environ.get("DRISHTI_THREADS", "6"))
        self.interim = self.ckpt.parent.name == "runA"  # Day 3 addendum: A2 is not interim
        self.startup_warnings: List[str] = []
        self.checkpoint_id = checkpoint_id(self.ckpt)
        self.predictor = self._load_predictor(requested)
        self.backend_reason = self._backend_reason
        scales = _read_json(REPORTS / "trust_scales.json")
        if "unc_display_max" not in scales or "cons_display_max" not in scales:
            raise FileNotFoundError(f"display scales missing from {REPORTS / 'trust_scales.json'}")
        self.refs = {
            "spec_l1_gt_floor": SPEC_L1_GT_FLOOR,
            "spec_sam_gt_floor_deg": SPEC_SAM_GT_FLOOR_DEG,
            "unc_display_max": float(scales["unc_display_max"]),
            "cons_display_max": float(scales["cons_display_max"]),
            "sharpness_warn_below": SHARPNESS_WARN_BELOW,
        }
        self.project_iters = self._projection_iters()
        self.cfg = load_config()
        self._evaluator: Optional[Evaluator] = None
        self.lock = threading.Lock()
        LOGGER.info("engine ready: %s (%s); projection iters %d", self.predictor.info["backend"],
                    self.backend_reason, self.project_iters)

    # -------------------------------------------------------------- startup
    def _load_predictor(self, requested: str):
        kw = dict(threads=self.threads, interim=self.interim)
        if requested == "torch":
            self._backend_reason = "DRISHTI_BACKEND=torch"
            return TorchPredictor(self.ckpt, **kw)
        if requested == "auto":
            gate = _read_json(REPORTS / "quant_gate.json").get(self.checkpoint_id, {})
            precision = "int8" if gate.get("passed") is True else "fp32"
            self._backend_reason = (f"auto: quant_gate.json passed={gate.get('passed')} for "
                                    f"{self.checkpoint_id} -> onnx-{precision}")
        else:
            precision = requested.split("-", 1)[1]
            self._backend_reason = f"DRISHTI_BACKEND={requested}"
        try:
            from src.infer.onnx_predictor import OnnxPredictor
            return OnnxPredictor(checkpoint_path=self.ckpt, precision=precision, **kw)
        except (FileNotFoundError, ImportError) as exc:
            msg = (f"ONNX {precision} model unavailable ({exc}); serving torch-fp32 instead, "
                   "which is slower.")
            LOGGER.warning(msg)
            self.startup_warnings.append(msg)
            self._backend_reason += " -> ONNX missing, torch-fp32 fallback"
            return TorchPredictor(self.ckpt, **kw)

    def _projection_iters(self) -> int:
        gate = _read_json(REPORTS / "projection_gate.json")
        if (hasattr(trust, "project_consistency") and gate.get("passed") is True
                and gate.get("iters") and os.environ.get("DRISHTI_PROJECT") != "0"):
            return int(gate["iters"])
        return 0

    @property
    def evaluator(self) -> Evaluator:
        if self._evaluator is None:
            self._evaluator = Evaluator(self.cfg, device="cpu")
        return self._evaluator

    def model_info(self) -> Dict[str, Any]:
        i = self.predictor.info
        return {"backend": str(i["backend"]), "checkpoint_id": str(i["checkpoint_id"]),
                "params": int(i["params"]), "model_bytes": int(i["model_bytes"]),
                "threads": int(i["threads"]), "interim": bool(i["interim"]),
                "has_scale_head": bool(i["has_scale_head"])}

    # -------------------------------------------------------------- samples
    def list_samples(self) -> List[Dict[str, Any]]:
        out = []
        for mf in sorted(SAMPLES_DIR.glob("*/manifest.json")):
            m = _read_json(mf)
            if not (mf.parent / "lr.npy").is_file():
                continue
            out.append(m)
        out.sort(key=lambda m: (m.get("order", 1_000), m["id"]))
        return [{"id": m["id"], "label": m["label"], "thumb_url": f"/api/samples/{m['id']}/thumb.png",
                 "has_gt": bool(m["has_gt"]), "lr_size": [int(v) for v in m["lr_size"]]} for m in out]

    def sample_dir(self, sample_id: str) -> Optional[Path]:
        if not SAFE_ID.match(sample_id or ""):
            return None
        d = SAMPLES_DIR / sample_id
        return d if (d / "manifest.json").is_file() and (d / "lr.npy").is_file() else None

    def load_sample(self, sample_id: str):
        """``(lr, hr|None, profile|None, manifest)`` or ``None`` when unknown."""
        d = self.sample_dir(sample_id)
        if d is None:
            return None
        m = _read_json(d / "manifest.json")
        lr = np.load(d / "lr.npy").astype(np.float32, copy=False)
        hr = np.load(d / "hr.npy").astype(np.float32, copy=False) if m.get("has_gt") else None
        return lr, hr, profile_from_json(m.get("profile")), m

    # -------------------------------------------------------------- one job
    def run(self, lr: np.ndarray, hr: Optional[np.ndarray], tta: int, profile: Profile,
            source: str, sample_id: Optional[str], dn_mode_applied: str,
            warnings: Optional[List[str]] = None) -> Dict[str, Any]:
        with self.lock:  # one job at a time: the predictor already uses every thread
            return self._run(lr, hr, tta, profile, source, sample_id, dn_mode_applied,
                             list(warnings or []))

    def _gt_block(self, x: np.ndarray, hr: np.ndarray) -> Dict[str, Optional[float]]:
        cols = self.evaluator.metrics_for_batch(torch.from_numpy(np.ascontiguousarray(x))[None],
                                                torch.from_numpy(np.ascontiguousarray(hr))[None])
        return {"lpips": _finite(cols["lpips"][0]), "ssim": _finite(cols["ssim_mean"][0]),
                "psnr": _finite(cols["psnr_mean"][0]), "sam_deg": _finite(cols["sam_mean_deg"][0]),
                "ergas": _finite(cols["ergas"][0])}

    def _run(self, lr, hr, tta, profile, source, sample_id, dn_mode_applied, warns):
        job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        jd = JOBS_DIR / job_id
        jd.mkdir(parents=True, exist_ok=False)
        url = lambda name: f"/files/{job_id}/{name}"  # noqa: E731

        tr = trust.compute_trust(lr, self.predictor, tta, hr=hr, project_iters=self.project_iters)
        _, h, w = lr.shape
        if self.project_iters > 0:
            warns.append(f"Spectral-consistency projection applied ({self.project_iters} iterations)")

        # --- renders (shared LR stretch; FCC = NIR, R, G)
        params = render.stretch_params(lr)
        layers = {"lr": render.lr_display(lr), "bicubic": tr.bicubic, "sr": tr.sr}
        if hr is not None:
            layers["hr"] = hr
        images: Dict[str, Optional[str]] = {}
        for key in ("lr", "bicubic", "sr", "hr"):
            for mode in ("rgb", "fcc"):
                name = f"{key}_{mode}"
                if key in layers:
                    render.save_png(render.to_png_array(layers[key], params, mode), jd / f"{name}.png")
                    images[name] = url(f"{name}.png")
                else:
                    images[name] = None
        if tr.unc is not None:
            render.save_png(render.overlay_rgba(tr.unc, self.refs["unc_display_max"], "magma"),
                            jd / "uncertainty.png")
            images["uncertainty"] = url("uncertainty.png")
        else:
            images["uncertainty"] = None
        render.save_png(render.overlay_rgba(render.upsample_map(tr.cons_l1, trust.SCALE),
                                            self.refs["cons_display_max"], "viridis"),
                        jd / "consistency.png")
        images["consistency"] = url("consistency.png")

        # --- GeoTIFF downloads
        write_sr_tif(jd / "sr.tif", tr.sr, profile)
        downloads = {"sr_tif": url("sr.tif"), "uncertainty_tif": None}
        if tr.unc is not None:
            write_unc_tif(jd / "uncertainty.tif", tr.unc, profile)
            downloads["uncertainty_tif"] = url("uncertainty.tif")

        # --- metrics
        s = tr.scalars
        t = tr.timings_ms
        reference_free = {
            "spec_l1": _finite(s["spec_l1"]), "spec_sam_deg": _finite(s["spec_sam_deg"]),
            "spec_l1_bicubic": _finite(s["spec_l1_bicubic"]),
            "spec_sam_bicubic_deg": _finite(s["spec_sam_bicubic_deg"]),
            "hf_ratio_vs_bicubic": _finite(s["hf_ratio_vs_bicubic"]),
            "unc_mean": _finite(s["unc_mean"]), "unc_p95": _finite(s["unc_p95"]),
            "runtime_ms": {"sr": float(t["sr"]), "uncertainty": _finite(t["uncertainty"]),
                           "total": float(t["total"])},
        }
        with_gt = None
        if hr is not None:
            with_gt = {"sr": self._gt_block(tr.sr, hr), "bicubic": self._gt_block(tr.bicubic, hr),
                       "spec_l1_hr": _finite(s["spec_l1_hr"]),
                       "spec_sam_hr_deg": _finite(s["spec_sam_hr_deg"]),
                       "hf_ratio_hr_vs_bicubic": _finite(s["hf_ratio_hr_vs_bicubic"])}
        metrics = {"reference_free": reference_free, "with_gt": with_gt}
        undefined = [k for k, v in reference_free.items() if v is None
                     and k not in ("unc_mean", "unc_p95")]
        if undefined:
            warns.append(f"Metrics undefined for this input (constant image?): {undefined}.")
        (jd / "metrics.json").write_text(json.dumps({
            "job_id": job_id, "checkpoint_id": self.checkpoint_id, "source": source,
            "sample_id": sample_id, "tta": tta, "uncertainty_method": tr.method,
            "project_iters": self.project_iters, "metrics": metrics,
        }, indent=2), encoding="utf-8")

        response = {
            "job_id": job_id,
            "input": {"source": source, "sample_id": sample_id if source == "sample" else None,
                      "lr_size": [int(h), int(w)], "sr_size": [int(trust.SCALE * h), int(trust.SCALE * w)],
                      "georeferenced": bool(profile is not None and profile.get("crs") is not None),
                      "dn_mode_applied": dn_mode_applied},
            "model": self.model_info(),
            "uncertainty_method": tr.method,
            "images": images,
            "downloads": downloads,
            "metrics": metrics,
            "refs": dict(self.refs),
            "warnings": list(self.startup_warnings) + warns,
        }
        self._prune()
        return response

    @staticmethod
    def _prune() -> None:
        dirs = sorted((d for d in JOBS_DIR.iterdir() if d.is_dir()),
                      key=lambda d: d.stat().st_mtime, reverse=True)
        for d in dirs[KEEP_JOBS:]:
            shutil.rmtree(d, ignore_errors=True)
