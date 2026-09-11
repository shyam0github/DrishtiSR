"""P4 end to end: export -> parity -> INT8 ladder + gate -> CPU benchmark.

    python scripts/mvp/export_and_quantize.py --checkpoint <path> [--interim] [--skip-bench]

Writes artifacts/onnx/<checkpoint_id>/{model_fp32.onnx, export_meta.json,
model_int8*.onnx}, reports/mvp/onnx_parity_<checkpoint_id>.json,
reports/mvp/quant_gate.json (merged, keyed by checkpoint_id) and
reports/mvp/bench_<checkpoint_id>.json. Exits 1 if parity fails (no INT8 or
benchmark is attempted on a graph that does not match torch).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.deploy.export_onnx import EXPORT_LR, export_onnx  # noqa: E402
from src.deploy.quantize import (CALIB_N, CALIB_SEED, PairSource, run_ladder, score_vs_hr,  # noqa: E402
                                 train_tiles, val_patches, write_json)
from src.infer.onnx_predictor import OnnxPredictor, make_session  # noqa: E402
from src.infer.predictor import TorchPredictor, build_model_from_checkpoint  # noqa: E402
from src.deploy.export_onnx import exportable_module  # noqa: E402
from src.metrics.image_quality import psnr  # noqa: E402
from src.utils.gitmeta import git_metadata  # noqa: E402
from src.utils.paths import repo_root  # noqa: E402

THREADS = 6
RAW_TOL = 1e-4
TILED_MIN_DB = 60.0
N_RAW, N_TILED, N_GATE = 8, 2, 64
REPORTS = repo_root() / "reports" / "mvp"


def _stamp(**kw: Any) -> Dict[str, Any]:
    return {**kw, "git_sha": git_metadata(repo_root()).get("commit"),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}


def parity(ckpt: Path, fp32: Path, src: PairSource, interim: bool) -> Dict[str, Any]:
    """(a) raw graph vs torch on TRAIN tiles; (b) tiled.py predictors on full VAL LR tiles."""
    model, meta = build_model_from_checkpoint(ckpt)
    module = exportable_module(model, meta["has_scale_head"])
    sess = make_session(fp32, THREADS)
    tiles, tile_ids = train_tiles(src, N_RAW, EXPORT_LR, seed=CALIB_SEED)
    raw = []
    with torch.no_grad():
        for t in tiles:
            ref = module(torch.from_numpy(t)).numpy()
            raw.append(float(np.abs(ref - sess.run(None, {"lr": t})[0]).max()))

    cfg = src.cfg
    val_ids = src.ids(str(cfg.loader.val_split))[:N_TILED]
    tiled = []
    for label, kw in (("default tiling (cfg.frontend.tile)", {}), ("forced tiling tile=64 overlap=16",
                                                                   {"tile": 64, "overlap": 16})):
        tp = TorchPredictor(ckpt, threads=THREADS, interim=interim, **kw)
        op = OnnxPredictor(model_path=fp32, threads=THREADS, interim=interim, **kw)
        for sid in val_ids:
            lr = src.lr(sid)[None]
            a, b = tp(lr), op(lr)
            tiled.append({"sample_id": sid, "tiling": label, "lr_shape": list(lr.shape),
                          "psnr_torch_vs_onnx_db": float(psnr(b[0], a[0], data_range=1.0).mean),
                          "max_abs_diff": float(np.abs(a - b).max())})
    ok_a = max(raw) <= RAW_TOL
    ok_b = all(r["psnr_torch_vs_onnx_db"] >= TILED_MIN_DB for r in tiled)
    return {"raw_graph": {"split": "train", "n_samples": len(tiles), "sample_ids": tile_ids,
                          "tile_lr": EXPORT_LR, "max_abs_diff": max(raw), "per_tile": raw,
                          "tolerance": RAW_TOL, "passed": ok_a},
            "tiled": {"split": "val", "n_samples": len(val_ids), "rows": tiled,
                      "min_psnr_db": min(r["psnr_torch_vs_onnx_db"] for r in tiled),
                      "threshold_db": TILED_MIN_DB, "passed": ok_b},
            "passed": ok_a and ok_b}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--interim", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(THREADS)
    ckpt = a.checkpoint.resolve()

    fp32, meta = export_onnx(ckpt)
    cid, work = meta["checkpoint_id"], fp32.parent
    src = PairSource()

    par = parity(ckpt, fp32, src, a.interim)
    write_json(REPORTS / f"onnx_parity_{cid}.json",
               _stamp(what="P4 ONNX FP32 parity", checkpoint_id=cid, split="train+val",
                      n_samples=par["raw_graph"]["n_samples"] + par["tiled"]["n_samples"],
                      interim=a.interim, export=meta, **par))
    print(f"parity (a) max|torch-onnx| {par['raw_graph']['max_abs_diff']:.3g}  "
          f"(b) min PSNR {par['tiled']['min_psnr_db']:.2f} dB  passed={par['passed']}")
    if not par["passed"]:
        return 1

    cfg = src.cfg
    calib, calib_ids = train_tiles(src, CALIB_N, EXPORT_LR, seed=CALIB_SEED)
    patches, gate_ids = val_patches(src, N_GATE)

    def score(path: Path) -> Dict[str, float]:
        s = make_session(path, THREADS)
        return score_vs_hr(lambda x: s.run(None, {"lr": x})[0], patches, cfg)

    ladder = run_ladder(fp32, work, calib, score, score(fp32))
    ladder = _stamp(**ladder, checkpoint_id=cid, interim=a.interim, split="val",
                    n_samples=len(patches), gate_sample_ids=gate_ids,
                    calibration={"split": "train", "n": len(calib), "seed": CALIB_SEED,
                                 "tile_lr": EXPORT_LR, "sample_ids": calib_ids},
                    fp32_bytes=meta["bytes"])
    gate_path = REPORTS / "quant_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    gate[cid] = ladder
    write_json(gate_path, gate)
    for r in ladder["attempts"]:
        print(f"INT8 {r['attempt']}: dPSNR {r['d_psnr_db']:+.4f} dSAM {r['d_sam_deg']:+.4f} "
              f"dLPIPS {r['d_lpips']:+.5f} bytes {r['bytes']} passed={r['passed']}")
    print(f"default backend: {ladder['default_backend']}")

    if not a.skip_bench:
        import bench_cpu

        rep = bench_cpu.run(ckpt, a.interim, THREADS)
        for r in rep["backends"]:
            print(f"bench {r['backend']:>11}: median {r['median_ms']:.1f} ms p90 {r['p90_ms']:.1f} ms")
        print(f"bench load {rep['load_percent_before']}% provisional={rep['provisional']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
