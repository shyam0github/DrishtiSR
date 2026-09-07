"""EDSR-baseline for DrishtiSR (Sentinel-2 x4 SR, 4-band RGBN).

Params @ defaults (16 blocks, 64 feats, 4->4 ch, x4): ~1.52 M.
Expects inputs already normalised to roughly [0, 1] (reflectance / 10000).
Place at: src/drishtisr/models/edsr.py
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv(in_ch: int, out_ch: int, k: int = 3) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=True)


class ResBlock(nn.Module):
    """EDSR residual block: conv-relu-conv, no BN, scaled residual."""

    def __init__(self, n_feats: int, res_scale: float = 1.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _conv(n_feats, n_feats),
            nn.ReLU(inplace=True),
            _conv(n_feats, n_feats),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class Upsampler(nn.Sequential):
    """PixelShuffle upsampler. scale must be 2, 3, or a power of 2."""

    def __init__(self, scale: int, n_feats: int) -> None:
        layers: list[nn.Module] = []
        if scale & (scale - 1) == 0:  # power of two
            for _ in range(int(torch.log2(torch.tensor(float(scale))).item())):
                layers += [_conv(n_feats, 4 * n_feats), nn.PixelShuffle(2)]
        elif scale == 3:
            layers += [_conv(n_feats, 9 * n_feats), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"unsupported scale {scale}")
        super().__init__(*layers)


class EDSR(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        n_resblocks: int = 16,
        n_feats: int = 64,
        in_ch: int = 4,
        out_ch: int = 4,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.head = _conv(in_ch, n_feats)
        body: list[nn.Module] = [ResBlock(n_feats, res_scale) for _ in range(n_resblocks)]
        body.append(_conv(n_feats, n_feats))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(Upsampler(scale, n_feats), _conv(n_feats, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = x + self.body(x)
        return self.tail(x)


def build_model(name: str = "edsr_baseline", **kw) -> nn.Module:
    """Single entry point so train/infer/eval never hardcode a constructor."""
    if name in ("edsr", "edsr_baseline"):
        return EDSR(**kw)
    raise ValueError(f"unknown model '{name}'")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model()
    x = torch.randn(2, 4, 64, 64)
    y = m(x)
    print(f"params={count_params(m):,}  in={tuple(x.shape)}  out={tuple(y.shape)}")
    assert y.shape == (2, 4, 256, 256)
