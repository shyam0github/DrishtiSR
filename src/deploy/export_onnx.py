"""Export a src/train.py checkpoint to an FP32 ONNX graph.

Graph I/O is the space ``tiled.sr_array`` hands to a torch model: SURFACE
REFLECTANCE, float32, ``(N, C, h, w)`` in, ``(N, C_out, 4h, 4w)`` out,
unclipped, no normalisation (the project has none, docs/mvp/inventory.md §3).

Input ``lr``, single output ``sr``. A model with the log-variance head is
exported PACKED to 8 channels -- 0-3 the SR mean, 4-7 the Laplace scale
``b = sqrt(exp(logvar) / 2)`` -- the same conversion ``TorchPredictor`` applies,
so both backends report identical quantities.

TILE GEOMETRY. Batch, height and width are all dynamic. tiled.py pads every
tile to the full ``tile`` it is given, but ``TorchPredictor`` (mirrored by the
ONNX predictor) gives it ``tile = max(h, w)`` for inputs no larger than one
tile, so a 64-px patch and a 130-px VAL tile each arrive at their own size; a
static 256x256 graph would reject them. Logged in docs/mvp/decisions.md.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import onnx
import torch

from ..infer.predictor import build_model_from_checkpoint, file_sha256
from ..utils.gitmeta import git_metadata
from ..utils.logging import get_logger
from ..utils.paths import repo_root

LOGGER = get_logger("drishtisr.deploy.export_onnx")

OPSET = 17
INPUT_NAME, OUTPUT_NAME = "lr", "sr"
# Trace shape: cfg.patches.lr_size, the model's training input edge. Only the
# trace uses it; every axis but channels is dynamic.
EXPORT_LR = 64
DYNAMIC_AXES = {INPUT_NAME: {0: "N", 2: "h", 3: "w"}, OUTPUT_NAME: {0: "N", 2: "H", 3: "W"}}
GEOMETRY = "dynamic batch, H and W (predictor feeds single tiles of max(h, w) for small inputs)"


class PackedScaleHead(torch.nn.Module):
    """``(sr, logvar)`` -> one 8-channel tensor ``[sr, sqrt(exp(logvar) / 2)]``."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sr, logvar = self.model(x)
        return torch.cat([sr, torch.sqrt(torch.exp(logvar) / 2.0)], dim=1)


def artifact_dir(checkpoint_id: str, root: Optional[Union[str, Path]] = None) -> Path:
    """``artifacts/onnx/<checkpoint_id>`` under the worktree (or ``root``)."""
    return Path(root) if root is not None else repo_root() / "artifacts" / "onnx" / checkpoint_id


def exportable_module(model: torch.nn.Module, has_scale_head: bool) -> torch.nn.Module:
    """The module whose forward the graph records (packed when there is a head)."""
    return PackedScaleHead(model).eval() if has_scale_head else model.eval()


def export_onnx(checkpoint_path: Union[str, Path], out_dir: Optional[Union[str, Path]] = None,
                opset: int = OPSET, export_lr: int = EXPORT_LR) -> Tuple[Path, Dict[str, Any]]:
    """Write ``model_fp32.onnx`` and ``export_meta.json``; return (onnx path, meta).

    Raises:
        onnx.checker.ValidationError: the exported graph is malformed.
    """
    model, ck = build_model_from_checkpoint(checkpoint_path)
    module = exportable_module(model, ck["has_scale_head"])
    out = Path(out_dir) if out_dir is not None else artifact_dir(ck["checkpoint_id"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model_fp32.onnx"

    in_ch = int(ck["arch"]["in_ch"])
    gen = torch.Generator().manual_seed(0)
    dummy = torch.rand(1, in_ch, export_lr, export_lr, generator=gen) * 0.3
    with torch.no_grad():
        torch.onnx.export(module, dummy, str(path), input_names=[INPUT_NAME],
                          output_names=[OUTPUT_NAME], opset_version=int(opset),
                          do_constant_folding=True, dynamic_axes=DYNAMIC_AXES)
    onnx.checker.check_model(str(path))

    simplified = "onnxsim not installed; skipped"
    if importlib.util.find_spec("onnxsim") is not None:
        import onnxsim  # noqa: WPS433 -- optional, used only when already present

        simple, ok = onnxsim.simplify(onnx.load(str(path)))
        if ok:
            onnx.save(simple, str(path))
            onnx.checker.check_model(str(path))
            simplified = "onnxsim applied"
        else:
            simplified = "onnxsim check failed; unsimplified graph kept"
    LOGGER.info("exported %s (%s)", path, simplified)

    graph = onnx.load(str(path))
    out_dims = [d.dim_param or d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim]
    in_dims = [d.dim_param or d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
    init_params = int(sum(int(torch.tensor(list(t.dims)).prod()) if t.dims else 1
                          for t in graph.graph.initializer))
    meta: Dict[str, Any] = {
        "checkpoint_id": ck["checkpoint_id"],
        "checkpoint_path": ck["path"],
        "checkpoint_sha256": ck["sha256"],
        "iteration": ck["iteration"],
        "arch": ck["arch"],
        "params": ck["params"],
        "onnx_initializer_elements": init_params,
        "has_scale_head": ck["has_scale_head"],
        "opset": int(opset),
        "input": {"name": INPUT_NAME, "shape": in_dims, "units": "surface reflectance, unclipped"},
        "output": {"name": OUTPUT_NAME, "shape": out_dims,
                   "channels": "0-3 SR mean" + (", 4-7 Laplace scale b" if ck["has_scale_head"] else "")},
        "dynamic_axes": DYNAMIC_AXES,
        "tile_geometry": GEOMETRY,
        "trace_shape": list(dummy.shape),
        "constant_folding": True,
        "simplifier": simplified,
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "git_sha": git_metadata(repo_root()).get("commit"),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path, meta
