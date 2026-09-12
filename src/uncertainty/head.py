"""Post-hoc heteroscedastic Laplace scale head on a FROZEN SR backbone (P6, Day 4 "Run C").

The head reads the backbone's LR-resolution body output -- the tensor entering
the upsampler ``tail.0`` -- captured with a forward pre-hook, so
``src/models/edsr.py`` is untouched and the backbone's ``forward`` is the one
that produced every Day 3 number. The mean path is therefore bit-identical to
the selected A2 checkpoint.

    feats (B, n_feats, h, w) -> conv3x3 -> 32 -> ReLU -> conv3x3 -> bands*scale^2
          -> PixelShuffle(scale) -> raw (B, bands, H, W)
    b = B_FLOOR + softplus(raw)

``b`` is the per-pixel, per-band Laplace scale in SURFACE REFLECTANCE -- the
same units as the training L1 loss (``F.l1_loss`` on reflectance). The last
conv is zero-initialised and its bias set so that ``b == b0`` everywhere at
init, ``b0`` = mean |SR - HR| over TRAIN patches (the Laplace MLE of a constant
scale).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.edsr import count_params, model_from_checkpoint

__all__ = ["B_FLOOR", "HEAD_HIDDEN", "ScaleHead", "EDSRWithScale", "ChannelSlice",
           "ScalePredictor", "softplus_inverse", "load_unc_checkpoint", "total_params"]

B_FLOOR = 1e-4
HEAD_HIDDEN = 32


def softplus_inverse(y: float) -> float:
    """x with softplus(x) == y, for y > 0 (numerically stable for small y)."""
    if y <= 0:
        raise ValueError(f"softplus_inverse needs y > 0, got {y}.")
    return float(y + math.log(-math.expm1(-y)))


class ScaleHead(nn.Module):
    """LR features -> Laplace scale ``b`` at HR resolution (see module docstring)."""

    def __init__(self, n_feats: int, bands: int = 4, scale: int = 4, hidden: int = HEAD_HIDDEN,
                 b_floor: float = B_FLOOR, b0: float = 0.01) -> None:
        super().__init__()
        self.bands, self.scale, self.b_floor = int(bands), int(scale), float(b_floor)
        self.conv1 = nn.Conv2d(n_feats, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, self.bands * self.scale ** 2, 3, padding=1)
        self.shuffle = nn.PixelShuffle(self.scale)
        self.set_b0(b0)

    def set_b0(self, b0: float) -> None:
        """Zero the last conv's weights and set its bias so ``b == b0`` everywhere."""
        if not b0 > self.b_floor:
            raise ValueError(f"b0 {b0} must exceed the floor {self.b_floor}.")
        with torch.no_grad():
            nn.init.zeros_(self.conv2.weight)
            self.conv2.bias.fill_(softplus_inverse(float(b0) - self.b_floor))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        raw = self.shuffle(self.conv2(F.relu(self.conv1(feats))))
        return self.b_floor + F.softplus(raw)


class EDSRWithScale(nn.Module):
    """Frozen EDSR backbone + trainable :class:`ScaleHead`.

    ``forward(lr) -> (mu, b)``; ``forward_packed(lr) -> cat([mu, b], 1)`` (8 ch).
    The backbone runs under ``torch.no_grad`` with ``requires_grad=False`` and
    stays in eval mode, so no gradient or state change can reach it.
    """

    def __init__(self, backbone: nn.Module, b0: float = 0.01, hidden: int = HEAD_HIDDEN,
                 b_floor: float = B_FLOOR) -> None:
        super().__init__()
        if getattr(backbone, "uncertainty", False):
            raise ValueError("backbone must be a plain (no var_head) EDSR.")
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        n_feats = int(backbone.head.out_channels)
        self.head = ScaleHead(n_feats, bands=int(backbone.out_ch), scale=int(backbone.scale),
                              hidden=hidden, b_floor=b_floor, b0=b0)
        self._feats: Optional[torch.Tensor] = None
        # Input of the upsampler = LR-resolution body output (head(x) + body(head(x))).
        self._hook = backbone.tail[0].register_forward_pre_hook(self._capture)

    def _capture(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        self._feats = inputs[0]

    def train(self, mode: bool = True) -> "EDSRWithScale":
        super().train(mode)
        self.backbone.eval()
        return self

    def backbone_forward(self, lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(mu, feats)`` from one frozen backbone pass."""
        with torch.no_grad():
            mu = self.backbone(lr)
        feats, self._feats = self._feats, None
        if feats is None:
            raise RuntimeError("upsampler pre-hook did not fire; backbone is not an EDSR.")
        return mu, feats

    def forward(self, lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, feats = self.backbone_forward(lr)
        return mu, self.head(feats)

    def forward_packed(self, lr: torch.Tensor) -> torch.Tensor:
        mu, b = self.forward(lr)
        return torch.cat([mu, b], dim=1)


def total_params(model: EDSRWithScale) -> int:
    """Backbone + head parameters (counted regardless of requires_grad)."""
    return int(sum(p.numel() for p in model.parameters()))


class ChannelSlice(nn.Module):
    """Single-output view of ``forward_packed`` for ``tiled.sr_array``.

    sr_array sizes its output from the INPUT band count, so the 8-channel packed
    output goes through it as two 4-channel halves (mean, scale), each blended
    with the same linear Hann weights -- b is blended linearly, tiled.py unchanged.
    """

    def __init__(self, model: EDSRWithScale, part: str) -> None:
        super().__init__()
        self.model, self.part = model, part

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = self.model.forward_packed(x)
        n = packed.shape[1] // 2
        return packed[:, :n] if self.part == "mean" else packed[:, n:]


def load_unc_checkpoint(path: Union[str, Path], backbone_path: Optional[Union[str, Path]] = None,
                        device: str = "cpu") -> Tuple[EDSRWithScale, Dict[str, Any]]:
    """Rebuild ``EDSRWithScale`` from a train_unc.py head checkpoint.

    The backbone is loaded from ``backbone_path`` (default: the path recorded in
    the checkpoint) and its sha256 prefix must match the recorded
    ``backbone_checkpoint_id``.
    """
    from src.infer.predictor import checkpoint_id

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    bb_path = Path(backbone_path or payload["backbone_path"])
    bb_id = checkpoint_id(bb_path)
    if bb_id != payload["backbone_checkpoint_id"]:
        raise RuntimeError(f"backbone {bb_path} is {bb_id}, head was trained on "
                           f"{payload['backbone_checkpoint_id']}.")
    backbone, _ = model_from_checkpoint(bb_path, device=device)
    model = EDSRWithScale(backbone, b0=float(payload["b0"]), hidden=int(payload["head_hidden"]),
                          b_floor=float(payload["b_floor"]))
    model.head.load_state_dict(payload["head"])
    model.eval()
    return model.to(device), payload


class ScalePredictor:
    """Predictor (src.infer.predictor contract) for a learned-scale model: 8-channel output."""

    def __init__(self, head_checkpoint: Union[str, Path], threads: int = 6, interim: bool = False,
                 tile: int = 256, overlap: int = 32) -> None:
        from src.infer.predictor import checkpoint_id

        torch.set_num_threads(int(threads))
        self.model, self.payload = load_unc_checkpoint(head_checkpoint)
        self.tile, self.overlap, self.scale = int(tile), int(overlap), int(self.model.backbone.scale)
        self._parts = [ChannelSlice(self.model, "mean"), ChannelSlice(self.model, "scale")]
        self.info: Dict[str, Any] = {
            "backend": "torch-fp32",
            "checkpoint_id": checkpoint_id(head_checkpoint),
            "params": total_params(self.model),
            "model_bytes": int(sum(t.numel() * t.element_size()
                                   for t in self.model.state_dict().values())),
            "threads": int(torch.get_num_threads()),
            "interim": bool(interim),
            "has_scale_head": True,
        }

    @torch.no_grad()
    def __call__(self, lr: np.ndarray) -> np.ndarray:
        from src.infer.tiled import sr_array

        if not isinstance(lr, np.ndarray) or lr.dtype != np.float32 or lr.ndim != 4:
            raise TypeError("lr must be a float32 (N, 4, h, w) np.ndarray.")
        outs = []
        for x in lr:
            h, w = x.shape[1:]
            tile, overlap = ((max(h, w), min(self.overlap, max(h, w) // 2))
                             if h <= self.tile and w <= self.tile else (self.tile, self.overlap))
            outs.append(np.concatenate([sr_array(np.ascontiguousarray(x), m, scale=self.scale,
                                                 tile=tile, overlap=overlap, device="cpu", amp=False)
                                        for m in self._parts], axis=0))
        return np.stack(outs).astype(np.float32, copy=False)


if __name__ == "__main__":  # pragma: no cover
    from src.models.edsr import build_model

    m = EDSRWithScale(build_model())
    print(f"backbone={count_params(m.backbone) + 0:,} (trainable 0 expected: {count_params(m.backbone)}) "
          f"head={count_params(m.head):,} total={total_params(m):,}")
