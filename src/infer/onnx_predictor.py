"""ONNX Runtime backend, served through the existing ``tiled.sr_array``.

There is ONE inference implementation. :class:`OnnxModule` makes an
``InferenceSession`` look like a ``torch.nn.Module`` so tiled.py can drive it
unchanged, and :class:`OnnxPredictor` reuses ``TorchPredictor``'s own
``_tiling`` and ``__call__`` -- validation, tile choice and tiled.py call are
literally the same functions. Units as in :mod:`src.infer.predictor`: surface
reflectance in and out, float32, unclipped, no normalisation.

Registered as ``"onnx"``. ``precision="auto"`` serves INT8 when
reports/mvp/quant_gate.json says the INT8 graph passed for this checkpoint and
FP32 otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ..deploy.export_onnx import artifact_dir
from ..utils.config import load_config
from ..utils.logging import get_logger
from ..utils.paths import repo_root
from .predictor import TorchPredictor, checkpoint_id, register_predictor

LOGGER = get_logger("drishtisr.infer.onnx_predictor")

GATE_PATH = Path("reports") / "mvp" / "quant_gate.json"


def make_session(model_path: Union[str, Path], threads: int = 6) -> ort.InferenceSession:
    """CPU session: graph optimisation ALL, intra_op ``threads``, inter_op 1, sequential."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


class OnnxModule(torch.nn.Module):
    """A session behind a ``torch.nn.Module`` face, for ``tiled.sr_array``.

    ``forward`` converts the tensor to float32 numpy, runs the session and
    returns a tensor with the input's dtype and device. ``channels`` optionally
    selects an output slice (the halves of a packed 8-channel graph, because
    tiled.py sizes its buffer from the input channel count).
    """

    def __init__(self, session: ort.InferenceSession, channels: Optional[slice] = None) -> None:
        super().__init__()
        self.session, self.channels = session, channels
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed = x.detach().to("cpu", torch.float32).contiguous().numpy()
        out = self.session.run([self.output_name], {self.input_name: feed})[0]
        if self.channels is not None:
            out = out[:, self.channels]
        return torch.from_numpy(np.ascontiguousarray(out)).to(dtype=x.dtype, device=x.device)


def _is_quantized(path: Path) -> bool:
    graph = onnx.load(str(path), load_external_data=False).graph
    return any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in graph.node)


def default_precision(ckpt_id: str, gate_path: Optional[Union[str, Path]] = None) -> str:
    """``"int8"`` iff the quant gate recorded ``passed: true`` for ``ckpt_id``."""
    path = Path(gate_path) if gate_path is not None else repo_root() / GATE_PATH
    if not path.is_file():
        return "fp32"
    entry = json.loads(path.read_text(encoding="utf-8")).get(ckpt_id, {})
    return "int8" if entry.get("default_backend") == "onnx-int8" else "fp32"


class OnnxPredictor:
    """ONNX Runtime backend; see :class:`src.infer.predictor.Predictor`.

    Args:
        checkpoint_path: The source checkpoint; locates
            ``artifacts/onnx/<checkpoint_id>/model_{fp32,int8}.onnx``.
        model_path: An explicit ``.onnx`` file instead (its directory must hold
            the ``export_meta.json`` written with it).
        precision: ``"auto"`` (quant gate decides), ``"int8"`` or ``"fp32"``.
            Ignored when ``model_path`` is given (read off the graph).
        threads, interim, tile, overlap, config: as :class:`TorchPredictor`.

    Raises:
        ValueError: neither path given, or band/scale mismatch with config.
        FileNotFoundError: the graph or its export_meta.json is missing.
    """

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None,
                 model_path: Optional[Union[str, Path]] = None, precision: str = "auto",
                 threads: int = 6, interim: bool = False, tile: Optional[int] = None,
                 overlap: Optional[int] = None, config: Optional[Union[str, Path]] = None,
                 gate_path: Optional[Union[str, Path]] = None) -> None:
        cfg = load_config(config) if config is not None else load_config()
        self.scale = int(cfg["sr"]["scale"])
        self.n_bands = len(cfg["dataset"]["bands"])
        self.tile = int(tile if tile is not None else cfg["frontend"]["tile"]["lr_px"])
        self.overlap = int(overlap if overlap is not None else cfg["frontend"]["tile"]["overlap_lr_px"])
        if not 0 <= self.overlap < self.tile:
            raise ValueError(f"overlap {self.overlap} must be in [0, tile={self.tile}).")

        if model_path is None:
            if checkpoint_path is None:
                raise ValueError("give checkpoint_path or model_path.")
            cid = checkpoint_id(checkpoint_path)
            if precision == "auto":
                precision = default_precision(cid, gate_path)
            if precision not in ("int8", "fp32"):
                raise ValueError(f"precision must be auto|int8|fp32, got {precision!r}.")
            model_path = artifact_dir(cid) / f"model_{precision}.onnx"
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX graph not found: {self.model_path}")
        meta_path = self.model_path.parent / "export_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"export_meta.json not found next to {self.model_path}")
        self.meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(self.meta["arch"]["scale"]) != self.scale:
            raise ValueError(f"graph scale {self.meta['arch']['scale']} != cfg.sr.scale {self.scale}.")

        torch.set_num_threads(int(threads))
        self.session = make_session(self.model_path, threads)
        out_ch = int(self.session.get_outputs()[0].shape[1])
        if out_ch not in (self.n_bands, 2 * self.n_bands):
            raise ValueError(f"graph emits {out_ch} channels; expected {self.n_bands} or {2 * self.n_bands}.")
        has_scale_head = out_ch == 2 * self.n_bands
        self.model = OnnxModule(self.session)
        self._modules = ({"mean": OnnxModule(self.session, slice(0, self.n_bands)),
                          "scale": OnnxModule(self.session, slice(self.n_bands, out_ch))}
                         if has_scale_head else {"mean": self.model})
        self.info: Dict[str, Any] = {
            "backend": "onnx-int8" if _is_quantized(self.model_path) else "onnx-fp32",
            "checkpoint_id": self.meta["checkpoint_id"],
            "params": int(self.meta["params"]),
            "model_bytes": int(self.model_path.stat().st_size),
            "threads": int(threads),
            "interim": bool(interim),
            "has_scale_head": has_scale_head,
        }
        LOGGER.info("onnx predictor %s from %s", self.info["backend"], self.model_path)

    # Same functions as the torch backend: one validation / tiling / tiled.py path.
    _tiling = TorchPredictor._tiling
    __call__ = TorchPredictor.__call__


register_predictor("onnx", OnnxPredictor)
