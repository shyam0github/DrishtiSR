"""Static INT8 (QDQ) quantisation of the exported FP32 graph, with a quality gate.

Recipe: ``quant_pre_process`` then ``quantize_static``, QDQ format,
per-channel weights QInt8, activations QUInt8. Calibration uses TRAIN tiles
only (RULES: anything fitted uses TRAIN); the gate measures on VAL.

Attempt ladder, stop at the first pass:
    A  MinMax
    B  Percentile 99.99
    C  B + first conv and final output conv kept in float
    D  B + the whole upsampler kept in float
Gate: INT8 vs FP32 ONNX, each scored against HR with src.metrics.image_quality
on VAL patches: dPSNR <= 0.10 dB, dSAM <= 0.05 deg, dLPIPS <= 0.005.

Also hosts the offline TRAIN/VAL pair reader used by calibration, parity and
the gate (npz cache read by ID; the dataset catalog needs the network).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import pandas as pd
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quant_pre_process, quantize_static)

from ..data.patches import centre_crop_pair
from ..data.sen2naip import SEN2NAIPv2Dataset
from ..metrics.image_quality import lpips, psnr, rgb_band_indices, sam
from ..utils.config import load_config
from ..utils.logging import get_logger
from ..utils.paths import repo_root

LOGGER = get_logger("drishtisr.deploy.quantize")

CALIB_SEED = 26142
CALIB_N = 128
GATE = {"d_psnr_db": 0.10, "d_sam_deg": 0.05, "d_lpips": 0.005}
ATTEMPTS = (
    ("A", CalibrationMethod.MinMax, {}, None),
    ("B", CalibrationMethod.Percentile, {"CalibPercentile": 99.99}, None),
    ("C", CalibrationMethod.Percentile, {"CalibPercentile": 99.99}, "first_and_output_conv"),
    ("D", CalibrationMethod.Percentile, {"CalibPercentile": 99.99}, "upsampler"),
)
# Bounds calibration memory: ORT's CalibStridedMinMax runs collect_data once per
# slice of N tiles (ranges / histograms merge across calls; despite the name it
# applies to every calibrator). Histogram calibration otherwise holds every
# activation of all 128 tiles (~11 GB). CalibMaxIntermediateOutputs is not used:
# in ORT 1.20 MinMax raises "No data is collected" after its last flush.
CALIB_CHUNK = 8


# ------------------------------------------------------------------ data
def main_tree() -> Path:
    """The main checkout (parent of the shared git dir); holds runs/ and outputs/."""
    common = subprocess.run(
        ["git", "-C", str(repo_root()), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True, capture_output=True, text=True).stdout.strip()
    return Path(common).parent


class PairSource:
    """Offline SEN2NAIPv2 pairs by sample ID, float32 reflectance, unclipped.

    Reads the main tree's npz cache directly and converts with the dataset's
    own ``to_reflectance`` and nodata policy (as tests/mvp/test_predictor.py).
    """

    def __init__(self, cfg: Any = None) -> None:
        self.cfg = cfg if cfg is not None else load_config()
        main = main_tree()
        split_csv = main / "outputs" / str(self.cfg.splits.output_name).format(dataset=self.cfg.dataset.name)
        self.cfg.paths.cache_dir = str(main / "outputs" / "cache")
        self.ds = SEN2NAIPv2Dataset(self.cfg, validate=False)
        self.splits = pd.read_csv(split_csv, dtype={"sample_id": str})
        self.split_csv = split_csv

    def ids(self, split: str) -> List[str]:
        """Cached IDs of ``split`` in frozen split-CSV order."""
        ids = self.splits.loc[self.splits["split"] == split, "sample_id"]
        return [s for s in ids if self.ds._cache_paths(s)[0].is_file()]

    def lr(self, sid: str) -> np.ndarray:
        return self._read(sid, "lr")

    def pair(self, sid: str) -> Tuple[np.ndarray, np.ndarray]:
        return self._read(sid, "lr"), self._read(sid, "hr")

    def _read(self, sid: str, member: str) -> np.ndarray:
        npz, _ = self.ds._cache_paths(sid)
        with np.load(npz) as handle:
            raw = handle[member][self.ds.band_indices]
        refl, _ = self.ds.to_reflectance(raw, self.cfg.dataset.nodata_value,
                                         float(self.cfg.dataset.nodata_fill))
        return np.ascontiguousarray(refl, dtype=np.float32)


def train_tiles(src: PairSource, n: int, size: int, seed: int = CALIB_SEED) -> Tuple[List[np.ndarray], List[str]]:
    """``n`` random ``size``-px LR crops (1, C, size, size) from distinct TRAIN tiles."""
    rng = np.random.default_rng(seed)
    ids = src.ids(str(src.cfg.loader.train_split))
    pick = [ids[i] for i in rng.choice(len(ids), size=n, replace=False)]
    tiles = []
    for sid in pick:
        lr = src.lr(sid)
        y = int(rng.integers(0, lr.shape[1] - size + 1))
        x = int(rng.integers(0, lr.shape[2] - size + 1))
        tiles.append(np.ascontiguousarray(lr[None, :, y:y + size, x:x + size]))
    return tiles, pick


def val_patches(src: PairSource, n: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[str]]:
    """First ``n`` cached VAL IDs, centre-cropped to cfg.patches.lr_size: [(lr, hr)] (C,h,w)."""
    ids = src.ids(str(src.cfg.loader.val_split))[:n]
    out = []
    for sid in ids:
        lr, hr = src.pair(sid)
        crop = centre_crop_pair(lr, hr, lr_size=int(src.cfg.patches.lr_size), scale=int(src.cfg.sr.scale))
        out.append((np.ascontiguousarray(crop["lr"]), np.ascontiguousarray(crop["hr"])))
    return out, ids


class TileReader(CalibrationDataReader):
    """Feeds calibration tiles one at a time under the graph's input name."""

    def __init__(self, tiles: Sequence[np.ndarray], input_name: str = "lr") -> None:
        self.tiles, self.input_name = list(tiles), input_name
        self.set_range(0, len(self.tiles))

    def __len__(self) -> int:
        return len(self.tiles)

    def set_range(self, start_index: int, end_index: int) -> None:
        """Used by ORT's ``CalibStridedMinMax``: calibrate one slice per call."""
        self._it = iter(self.tiles[start_index:end_index])

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        tile = next(self._it, None)
        return None if tile is None else {self.input_name: tile}

    def rewind(self) -> None:
        self.set_range(0, len(self.tiles))


# ------------------------------------------------------------------ graph
def excluded_nodes(model_path: Union[str, Path], which: Optional[str]) -> List[str]:
    """Node names kept in float for a ladder rung.

    ``first_and_output_conv``: the first Conv (``head``) and the output conv
    (``tail.1``). ``upsampler``: every node under ``tail.0`` (convs and pixel
    shuffle). Names come from the torch exporter's module scopes.

    Raises:
        LookupError: the expected nodes are not in the graph.
    """
    if which is None:
        return []
    nodes = onnx.load(str(model_path)).graph.node
    convs = [n.name for n in nodes if n.op_type == "Conv"]
    if which == "first_and_output_conv":
        out_conv = [n for n in convs if "tail.1" in n]
        if not convs or len(out_conv) != 1:
            raise LookupError(f"cannot find head/output conv among {convs}")
        return [convs[0], out_conv[0]]
    if which == "upsampler":
        # torch exporter scopes: "/tail.0/tail.0.0/Conv" (or "/model/tail.0/..." when packed)
        names = [n.name for n in nodes if "/tail.0/" in n.name]
        if not names:
            raise LookupError("no upsampler (tail.0) nodes in graph")
        return names
    raise ValueError(f"unknown exclusion {which!r}")


def preprocess(fp32_path: Union[str, Path], out_path: Union[str, Path]) -> Path:
    quant_pre_process(str(fp32_path), str(out_path))
    return Path(out_path)


def quantize(pre_path: Union[str, Path], out_path: Union[str, Path], tiles: Sequence[np.ndarray],
             method: CalibrationMethod, extra: Dict[str, Any], exclude: Sequence[str]) -> Path:
    quantize_static(
        str(pre_path), str(out_path), TileReader(tiles),
        quant_format=QuantFormat.QDQ, per_channel=True,
        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
        calibrate_method=method, nodes_to_exclude=list(exclude),
        extra_options={"CalibStridedMinMax": CALIB_CHUNK, **extra})
    return Path(out_path)


# ------------------------------------------------------------------ gate
def score_vs_hr(run: Callable[[np.ndarray], np.ndarray], patches: Iterable[Tuple[np.ndarray, np.ndarray]],
                cfg: Any) -> Dict[str, float]:
    """Mean PSNR / SAM / LPIPS of ``run(lr[None])[0, :C]`` against HR, per patch then averaged."""
    dr = float(cfg.metrics.data_range)
    rgb = rgb_band_indices(cfg)
    net = str(cfg.metrics.lpips.get("net", "alex"))
    rows = []
    for lr, hr in patches:
        sr = run(lr[None])[0, : hr.shape[0]]
        rows.append((float(psnr(sr, hr, data_range=dr).mean),
                     float(sam(sr, hr).mean_deg),
                     float(np.mean(lpips(sr, hr, rgb_indices=rgb, net=net, data_range=dr).distance))))
    arr = np.asarray(rows, dtype=np.float64)
    return {"psnr_db": float(arr[:, 0].mean()), "sam_deg": float(np.nanmean(arr[:, 1])),
            "lpips": float(arr[:, 2].mean()), "n": int(len(rows))}


def gate_deltas(fp32: Dict[str, float], int8: Dict[str, float]) -> Dict[str, Any]:
    """Degradations (positive = INT8 worse) and pass flag against :data:`GATE`."""
    d = {"d_psnr_db": fp32["psnr_db"] - int8["psnr_db"],
         "d_sam_deg": int8["sam_deg"] - fp32["sam_deg"],
         "d_lpips": int8["lpips"] - fp32["lpips"]}
    d["passed"] = all(d[k] <= GATE[k] for k in GATE)
    return d


def run_ladder(fp32_path: Path, work_dir: Path, tiles: Sequence[np.ndarray],
               score: Callable[[Path], Dict[str, float]], fp32_scores: Dict[str, float]) -> Dict[str, Any]:
    """Try A..D; copy the first passing graph to ``model_int8.onnx``.

    Returns ``{"passed", "attempt", "default_backend", "attempts": [...]}``.
    """
    pre = preprocess(fp32_path, work_dir / "model_fp32_pre.onnx")
    attempts: List[Dict[str, Any]] = []
    winner = None
    for tag, method, extra, which in ATTEMPTS:
        exclude = excluded_nodes(pre, which)
        path = quantize(pre, work_dir / f"model_int8_{tag}.onnx", tiles, method, extra, exclude)
        s = score(path)
        d = gate_deltas(fp32_scores, s)
        rec = {"attempt": tag, "calibration": method.name, "extra_options": extra,
               "exclusion": which, "excluded_nodes": exclude, "bytes": path.stat().st_size,
               "file": path.name, "int8": s, "fp32": fp32_scores, **d}
        attempts.append(rec)
        LOGGER.info("INT8 attempt %s: %s", tag, {k: rec[k] for k in ("d_psnr_db", "d_sam_deg", "d_lpips", "passed")})
        if d["passed"]:
            winner = tag
            shutil.copyfile(path, work_dir / "model_int8.onnx")
            break
    return {"passed": winner is not None, "attempt": winner,
            "default_backend": "onnx-int8" if winner else "onnx-fp32",
            "gate": GATE, "attempts": attempts}


def json_safe(obj: Any) -> Any:
    """Non-finite floats -> strings, so reports stay strict JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def write_json(path: Union[str, Path], obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_safe(obj), indent=2), encoding="utf-8")
