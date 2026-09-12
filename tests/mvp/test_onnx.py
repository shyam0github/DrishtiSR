"""P4: ONNX export parity and the ONNX predictor through tiled.py.

Exports the A2 checkpoint to a temp directory (no dependency on artifacts/).
SKIPS, naming the path, when the main tree's checkpoint or data cache is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.deploy.export_onnx import EXPORT_LR, export_onnx, exportable_module
from src.deploy.quantize import PairSource, main_tree, train_tiles
from src.infer.onnx_predictor import OnnxModule, OnnxPredictor, make_session
from src.infer.predictor import TorchPredictor, build_model_from_checkpoint, make_predictor

MAIN = main_tree()
A2_CKPT = MAIN / "runs" / "day3" / "a2" / "last.pt"
CACHE = MAIN / "outputs" / "cache"


def _require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"main-tree artefacts absent: {missing}")


@pytest.fixture(scope="module")
def fp32(tmp_path_factory) -> Path:
    _require(A2_CKPT)
    path, meta = export_onnx(A2_CKPT, out_dir=tmp_path_factory.mktemp("onnx"))
    assert meta["opset"] == 17 and meta["params"] == 855_652
    return path


@pytest.fixture(scope="module")
def src() -> PairSource:
    _require(CACHE)
    return PairSource()


def test_raw_graph_parity_two_train_tiles(fp32, src):
    model, meta = build_model_from_checkpoint(A2_CKPT)
    module = exportable_module(model, meta["has_scale_head"])
    sess = make_session(fp32)
    tiles, _ = train_tiles(src, 2, EXPORT_LR)
    with torch.no_grad():
        for t in tiles:
            diff = np.abs(module(torch.from_numpy(t)).numpy() - sess.run(None, {"lr": t})[0]).max()
            assert diff <= 1e-4, diff


def test_onnx_module_preserves_dtype_and_device(fp32):
    x = torch.rand(1, 4, 16, 16, dtype=torch.float64)
    y = OnnxModule(make_session(fp32))(x)
    assert y.dtype == x.dtype and y.device == x.device and tuple(y.shape) == (1, 4, 64, 64)


def test_predictor_shapes_match_torch_through_tiled(fp32, src):
    sid = src.ids(str(src.cfg.loader.val_split))[0]
    lr = src.lr(sid)[None]
    # tile=64 forces several overlapping tiles and reflect-padded edge tiles.
    tp = TorchPredictor(A2_CKPT, tile=64, overlap=16)
    op = OnnxPredictor(model_path=fp32, tile=64, overlap=16)
    a, b = tp(lr), op(lr)
    assert a.shape == b.shape and b.dtype == np.float32
    assert op.info["backend"] == "onnx-fp32" and op.info["has_scale_head"] is False
    assert set(op.info) == set(tp.info)


def test_make_predictor_returns_onnx_predictor(fp32):
    pred = make_predictor("onnx", model_path=fp32)
    assert isinstance(pred, OnnxPredictor)
