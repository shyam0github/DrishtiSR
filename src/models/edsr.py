"""EDSR-baseline for DrishtiSR (Sentinel-2 x4 SR, 4-band RGBN).

Params @ defaults (16 blocks, 48 feats, 4->4 ch, x4): 855,652 -- under
``cfg.runtime.max_parameters`` (1,000,000), which is a hard deliverable, not a
target. The width was 64 through Run A, which is 1,518,724 and over budget; it
also had more capacity than ~2400 training patches can support, and Run A duly
peaked at 8k iterations and declined to 40k. Both problems have the same fix.

WIDTH IS THE KNOB, NOT DEPTH. Params scale as ``n_resblocks * n_feats^2``, so
narrowing 64 -> 48 buys a 44% cut while keeping all 16 blocks and therefore the
receptive field. Dropping blocks instead would have cost context, which is what
an SR model uses to decide what the missing detail should be.

Expects surface reflectance, float32, nominally [0, 1] and UNCLIPPED -- bright
targets (cloud, snow, specular water, bright roofs) legitimately exceed 1.0 and
are NOT clamped anywhere in this file. No ImageNet or per-channel statistics are
applied: the input is a physical quantity and stays one.

THE UNCERTAINTY HEAD (technical contribution 2), behind ``uncertainty=True``.
A second output branch predicts per-pixel, per-band LOG-VARIANCE of the SR
reflectance, at the SR output's spatial size. It branches off the upsampled
HR-resolution features -- the same tensor the SR output conv reads -- through a
small conv-ReLU-conv head (``var_feats`` hidden channels; +7,508 params at the
defaults, 863,160 total). Three decisions, each load-bearing:

- LOG variance, never variance. A variance head needs a positivity transform
  whose gradient vanishes exactly where a confident model sits; log-variance is
  unconstrained and the NLL's ``exp(-logvar)`` is well conditioned.
- The output is CLAMPED to ``[logvar_min, logvar_max]`` inside ``forward``,
  not only in the loss, so the exported ONNX graph emits a bounded raster (an
  unbounded output is also what makes INT8 output quantisation fall apart).
  This clamps the UNCERTAINTY, never the reflectance.
- The final conv is zero-initialised with its bias at ``logvar_init``, so an
  untrained head says "log-variance = logvar_init everywhere" rather than
  noise. Adam moves a bias by roughly lr per step; from a default-init bias
  near 0, reaching the ~-9 a reflectance residual of ~0.01 implies would take
  tens of thousands of iterations, far longer than the NLL ramp.

With the flag off the module is bit-for-bit the Day 3 architecture: same
parameter names, same count (855,652), same forward. A Day 3 checkpoint loads
into an uncertainty-enabled model through :func:`load_backbone_state`, which is
``strict=False`` restricted to exactly the head's own keys.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

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
    """EDSR-baseline, optionally with the heteroscedastic log-variance head.

    Args:
        scale: Integer upscaling factor.
        n_resblocks: Residual blocks in the body.
        n_feats: Feature width.
        in_ch: Input bands (4 = RGBN, ``cfg.dataset.bands`` order).
        out_ch: Output bands, same order and units as the input.
        res_scale: Residual scaling inside each block.
        uncertainty: Add the log-variance head. Off by default, so every
            existing call site builds the Day 3 architecture unchanged.
        var_feats: Hidden width of the log-variance head.
        logvar_min: Lower clamp on the emitted log-variance (natural log of
            reflectance^2). -10 is sigma = exp(-5) = 0.0067 reflectance.
        logvar_max: Upper clamp on the emitted log-variance.
        logvar_init: Initial constant log-variance (bias of the zeroed final
            conv). Must lie inside the clamp.

    Raises:
        ValueError: The clamp is empty or ``logvar_init`` lies outside it.
    """

    def __init__(
        self,
        scale: int = 4,
        n_resblocks: int = 16,
        n_feats: int = 48,
        in_ch: int = 4,
        out_ch: int = 4,
        res_scale: float = 1.0,
        uncertainty: bool = False,
        var_feats: int = 16,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
        logvar_init: float = -9.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.uncertainty = bool(uncertainty)
        self.head = _conv(in_ch, n_feats)
        body: list[nn.Module] = [ResBlock(n_feats, res_scale) for _ in range(n_resblocks)]
        body.append(_conv(n_feats, n_feats))
        self.body = nn.Sequential(*body)
        # `tail` keeps its Day 3 structure and parameter names (tail.0.* is the
        # upsampler, tail.1.* the output conv) so Day 3 state dicts address the
        # same tensors. forward() calls the two halves separately to reach the
        # upsampled features in between.
        self.tail = nn.Sequential(Upsampler(scale, n_feats), _conv(n_feats, out_ch))

        if self.uncertainty:
            if not float(logvar_min) < float(logvar_max):
                raise ValueError(
                    f"logvar clamp [{logvar_min}, {logvar_max}] is empty."
                )
            if not float(logvar_min) <= float(logvar_init) <= float(logvar_max):
                raise ValueError(
                    f"logvar_init {logvar_init} lies outside the clamp "
                    f"[{logvar_min}, {logvar_max}]; the head would start pinned "
                    "at a bound with zero gradient."
                )
            self.logvar_min = float(logvar_min)
            self.logvar_max = float(logvar_max)
            self.var_head = nn.Sequential(
                _conv(n_feats, int(var_feats)),
                nn.ReLU(inplace=True),
                _conv(int(var_feats), out_ch),
            )
            nn.init.zeros_(self.var_head[2].weight)
            nn.init.constant_(self.var_head[2].bias, float(logvar_init))

    def forward(
        self, x: torch.Tensor, detach_var: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Super-resolve, and with the head enabled, predict log-variance.

        Args:
            x: ``float32``, ``(B, in_ch, h, w)``. Surface reflectance,
                nominally [0, 1], UNCLIPPED.
            detach_var: Feed the log-variance head DETACHED features, so no
                gradient from it reaches the shared trunk. Used during the NLL
                warmup (see :mod:`src.losses.nll`). Ignored without the head.

        Returns:
            Without the head: ``sr``, ``float32``, ``(B, out_ch, h*scale,
            w*scale)``, surface reflectance, unclipped -- exactly the Day 3
            return type.

            With the head: ``(sr, logvar)``. ``logvar`` is ``float32``, the
            same shape as ``sr``, the natural log of the predicted variance of
            the reflectance (units reflectance^2), clamped to
            ``[logvar_min, logvar_max]``. ``sr`` is unclipped as above.
        """
        x = self.head(x)
        x = x + self.body(x)
        feats = self.tail[0](x)
        sr = self.tail[1](feats)
        if not self.uncertainty:
            return sr
        logvar = self.var_head(feats.detach() if detach_var else feats)
        return sr, logvar.clamp(self.logvar_min, self.logvar_max)


def build_model(name: str = "edsr_baseline", **kw) -> nn.Module:
    """Single entry point so train/infer/eval never hardcode a constructor."""
    if name in ("edsr", "edsr_baseline"):
        return EDSR(**kw)
    raise ValueError(f"unknown model '{name}'")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# The only keys an uncertainty-enabled model may be MISSING when it is loaded
# from a checkpoint trained without the head.
VAR_HEAD_PREFIX = "var_head."


def load_backbone_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> List[str]:
    """Load a checkpoint trained WITHOUT the head into a model that may have one.

    This is the ``strict=False`` path, and it is narrower than ``strict=False``:
    the only tolerated discrepancy is that the log-variance head's own
    parameters are absent from ``state`` (they keep their initialisation). Any
    other missing key, ANY unexpected key, and any shape mismatch raises -- a
    silently partial load is a model that scores badly for a reason no metric
    can show.

    Args:
        model: An :class:`EDSR`, with or without the head.
        state: A state dict, e.g. ``torch.load(ckpt)["model"]`` from a Day 3
            (A2/B1/B2) or Run A checkpoint.

    Returns:
        The head keys that were left at their initialisation (empty when the
        model has no head, or when ``state`` already carried the head).

    Raises:
        RuntimeError: Anything other than head keys is missing, anything is
            unexpected, or a tensor shape does not match (the last from
            ``load_state_dict`` itself, which checks shapes even when not
            strict).
    """
    result = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in result.missing_keys if not k.startswith(VAR_HEAD_PREFIX)]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            "checkpoint does not fit the model beyond the uncertainty head: "
            f"missing {bad_missing}, unexpected {list(result.unexpected_keys)}. "
            "Only var_head.* may be absent. Check n_feats / n_resblocks against "
            "the checkpoint's args."
        )
    return list(result.missing_keys)


# Architecture keys every src/train.py checkpoint records in its "args".
ARCH_KEYS = ("scale", "n_resblocks", "n_feats", "in_ch")


def model_from_checkpoint(
    path: Union[str, Path], device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild the architecture a checkpoint was trained as and load it strictly.

    Works for Run A (64 features), the Day 3 runs (48, no head) and Day 4
    uncertainty runs (``args["uncertainty"]`` true). A checkpoint predating the
    ``--uncertainty`` flag has no such key, and is built without the head --
    not a guess: those runs could not have had one.

    Args:
        path: A ``.pt`` written by ``src/train.py``'s ``save()``.
        device: Torch device string; CPU is the working default.

    Returns:
        ``(model, payload)``: the model in ``eval()`` mode on ``device``, and
        the raw checkpoint dict (``it``, ``args``, provenance, ...).

    Raises:
        KeyError: The checkpoint has no ``model``/``args`` or its args omit an
            architecture key. Refused rather than defaulted: a wrong width
            fails with an unrelated shape error or loads a different model.
        RuntimeError: The state dict does not fit (strict load).
    """
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    for key in ("model", "args"):
        if key not in payload:
            raise KeyError(f"{path} has no {key!r} entry; not a src/train.py checkpoint.")
    args = dict(payload["args"])
    missing = [k for k in ARCH_KEYS if k not in args]
    if missing:
        raise KeyError(f"{path}'s args lack {missing}; refusing to guess the architecture.")

    kwargs: Dict[str, Any] = dict(
        scale=int(args["scale"]),
        n_resblocks=int(args["n_resblocks"]),
        n_feats=int(args["n_feats"]),
        in_ch=int(args["in_ch"]),
        out_ch=int(args["in_ch"]),
    )
    if bool(int(args.get("uncertainty", 0))):
        kwargs.update(
            uncertainty=True,
            var_feats=int(args["var_feats"]),
            logvar_min=float(args["logvar_min"]),
            logvar_max=float(args["logvar_max"]),
            logvar_init=float(args["logvar_init"]),
        )
    model = build_model("edsr_baseline", **kwargs)
    model.load_state_dict(payload["model"])
    model.eval()
    return model.to(device), payload


if __name__ == "__main__":
    m = build_model()
    x = torch.randn(2, 4, 64, 64)
    y = m(x)
    print(f"params={count_params(m):,}  in={tuple(x.shape)}  out={tuple(y.shape)}")
    assert y.shape == (2, 4, 256, 256)
    assert count_params(m) == 855_652, count_params(m)
    mu = build_model(uncertainty=True)
    sr, lv = mu(x)
    print(f"uncertainty params={count_params(mu):,}  sr={tuple(sr.shape)}  logvar={tuple(lv.shape)}")
    assert sr.shape == lv.shape == (2, 4, 256, 256)
