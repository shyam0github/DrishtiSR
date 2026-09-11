"""CPU latency of the full predictor for torch-fp32, onnx-fp32 and onnx-int8.

Input: a 256x256 LR window of the Day 2 Delhi scene (raw L2A DN), converted to
reflectance with the project's offset ``(dn - 1000) / 10000`` inside the timed
region, then the predictor (tiled.py path). The project has no normalisation,
so that conversion is the whole pre-processing. 3 warm-ups, 20 timed runs.

Latency carries ``provisional: true`` unless CPU load was < 20% beforehand
(RULES). Usage:
    python scripts/mvp/bench_cpu.py --checkpoint <path> [--interim]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.deploy.export_onnx import artifact_dir
from src.deploy.quantize import main_tree, write_json
from src.infer.onnx_predictor import OnnxPredictor
from src.infer.predictor import TorchPredictor, checkpoint_id
from src.metrics.image_quality import psnr
from src.utils.gitmeta import git_metadata
from src.utils.paths import repo_root

SIZE, WARMUP, RUNS = 256, 3, 20
DELHI = Path("data") / "delhi" / "delhi_A_20241030.tif"
DN_OFFSET, REFLECT_DIV = 1000.0, 10000.0  # configs/base.yaml delhi.dn_offset, reflectance_scale
WINDOW_ORIGIN = (400, 400)  # row, col of the 256x256 window inside the 1243x1294 scene


def _ps(expr: str) -> str:
    return subprocess.run(["powershell", "-NoProfile", "-Command", expr],
                          capture_output=True, text=True, timeout=60).stdout.strip()


def cpu_state(samples: int = 3) -> Dict[str, Any]:
    """CPU model and mean LoadPercentage over ``samples`` readings (Windows WMI)."""
    name = _ps("(Get-CimInstance Win32_Processor).Name") or platform.processor()
    loads: List[float] = []
    for _ in range(samples):
        out = _ps("(Get-CimInstance Win32_Processor).LoadPercentage")
        try:
            loads.append(float(out))
        except ValueError:
            pass
        time.sleep(1.0)
    load = float(np.mean(loads)) if loads else None
    return {"cpu_model": name, "logical_cpus": os.cpu_count(),
            "load_percent_samples": loads, "load_percent_before": load}


def delhi_window() -> np.ndarray:
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(main_tree() / DELHI) as src:
        r, c = WINDOW_ORIGIN
        return src.read(window=Window(c, r, SIZE, SIZE))  # (4, 256, 256) uint16 DN


def _peak_rss() -> Optional[int]:
    if importlib.util.find_spec("psutil") is None:
        return None
    import psutil

    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def time_backend(pred: Any, dn: np.ndarray) -> Dict[str, Any]:
    def once() -> np.ndarray:
        lr = ((dn.astype(np.float32) - DN_OFFSET) / REFLECT_DIV)[None]
        return pred(lr)

    for _ in range(WARMUP):
        out = once()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = once()
        times.append((time.perf_counter() - t0) * 1e3)
    t = np.asarray(times)
    return {"backend": pred.info["backend"], "model_bytes": pred.info["model_bytes"],
            "median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)),
            "min_ms": float(t.min()), "times_ms": [round(v, 3) for v in times],
            "peak_rss_bytes": _peak_rss(), "_out": out}


def run(checkpoint: Path, interim: bool = False, threads: int = 6) -> Dict[str, Any]:
    state = cpu_state()
    torch.set_num_threads(threads)
    dn = delhi_window()
    cid = checkpoint_id(checkpoint)
    preds = [TorchPredictor(checkpoint, threads=threads, interim=interim),
             OnnxPredictor(checkpoint, precision="fp32", threads=threads, interim=interim)]
    int8_note = None
    if (artifact_dir(cid) / "model_int8.onnx").is_file():
        preds.append(OnnxPredictor(checkpoint, precision="int8", threads=threads, interim=interim))
        int8_note = "gate-passing INT8 graph (model_int8.onnx)"
    else:
        # No attempt passed: time the one closest to the gate (smallest worst
        # delta/limit) for information only. It is not the served backend.
        gate_file = repo_root() / "reports" / "mvp" / "quant_gate.json"
        entry = json.loads(gate_file.read_text(encoding="utf-8")).get(cid) if gate_file.is_file() else None
        if entry and entry.get("attempts"):
            best = min(entry["attempts"], key=lambda r: max(r[k] / v for k, v in entry["gate"].items()))
            preds.append(OnnxPredictor(model_path=artifact_dir(cid) / best["file"], threads=threads,
                                       interim=interim))
            int8_note = f"attempt {best['attempt']} FAILED the quality gate; timed for information, not served"
    rows = [time_backend(p, dn) for p in preds]
    for row in rows:
        row["gate_passed"] = None if row["backend"] != "onnx-int8" else "FAILED" not in (int8_note or "")
        row["note"] = int8_note if row["backend"] == "onnx-int8" else None
    ref = rows[0]["_out"][0]
    for row in rows:
        row["psnr_vs_torch_db"] = float(psnr(row.pop("_out")[0], ref, data_range=1.0).mean)
    load = state["load_percent_before"]
    report = {
        "what": "P4 CPU latency: 256x256 LR through the full predictor (tiled.py + DN->reflectance)",
        "checkpoint_id": cid, "checkpoint_path": str(checkpoint),
        "git_sha": git_metadata(repo_root()).get("commit"),
        "split": None, "input": f"{DELHI.as_posix()} window rows/cols {WINDOW_ORIGIN} {SIZE}x{SIZE}, (dn-{DN_OFFSET:g})/{REFLECT_DIV:g}",
        "n_samples": 1, "warmup": WARMUP, "runs": RUNS, "threads": threads,
        **state,
        "provisional": load is None or load >= 20.0,
        "peak_rss_note": None if _peak_rss() is not None else "psutil not installed; peak RSS not recorded",
        "interim": bool(interim),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "onnxruntime": __import__("onnxruntime").__version__,
        "backends": rows,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(repo_root() / "reports" / "mvp" / f"bench_{cid}.json", report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--interim", action="store_true")
    a = ap.parse_args()
    rep = run(a.checkpoint, a.interim)
    for r in rep["backends"]:
        print(f"{r['backend']:>11}: median {r['median_ms']:.1f} ms  p90 {r['p90_ms']:.1f} ms")
    print(f"load before {rep['load_percent_before']}% provisional={rep['provisional']}")


if __name__ == "__main__":
    main()
