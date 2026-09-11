"""Predictor interface shared by every MVP backend (torch now, ONNX in P4).

A predictor maps a batch of Sentinel-2 LR reflectance to SR reflectance and
carries an ``info`` dict that the API reports verbatim as its ``MODEL`` block
(docs/mvp/api_contract.md). Backends are looked up by name through a small
registry, so the API never imports a backend module directly.

Units, stated once for the whole module: everything that goes in or comes out is
SURFACE REFLECTANCE, float32, nominally [0, 1] and UNCLIPPED -- cloud, snow and
bright roofs exceed 1.0 and pass through untouched. Nothing here normalises,
standardises or clips. Band order is ``cfg.dataset.bands`` (B04, B03, B02, B08).

HOW THIS RELATES TO EVALUATION. ``scripts/eval_all_ckpts.py`` and
``scripts/eval_runA.py`` call the model directly, ``model(lr)``, on
reflectance patches (see ``eval_runA.make_model_sr_fn``); they do not use
``src/infer/tiled.py``. :class:`TorchPredictor` goes through ``tiled.sr_array``
(the project's inference entry point) with the same reflectance in / out and no
conversion, and picks the tile so that an input no larger than one tile is a
single, unpadded forward pass -- bit-for-bit what eval computes. Larger inputs
are tiled with ``cfg.frontend.tile`` (which mirrors the tiled.py CLI defaults).
tests/mvp/test_predictor.py asserts the equivalence on a real VAL patch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import torch

try:
    from drishtisr.infer.tiled import sr_array
    from drishtisr.models.edsr import ARCH_KEYS, build_model, count_params, model_from_checkpoint
    from drishtisr.utils.config import load_config
    from drishtisr.utils.logging import get_logger
except ImportError:
    from src.infer.tiled import sr_array  # type: ignore
    from src.models.edsr import ARCH_KEYS, build_model, count_params, model_from_checkpoint  # type: ignore
    from src.utils.config import load_config  # type: ignore
    from src.utils.logging import get_logger  # type: ignore

LOGGER = get_logger("drishtisr.infer.predictor")

# Keys every predictor's ``info`` dict must carry (the contract's MODEL block).
INFO_KEYS = ("backend", "checkpoint_id", "params", "model_bytes", "threads",
             "interim", "has_scale_head")


@runtime_checkable
class Predictor(Protocol):
    """LR reflectance batch in, SR reflectance batch out.

    ``__call__(lr)``:
        lr: ``np.ndarray`` float32, ``(N, 4, h, w)`` (N, C, H, W axis order),
            surface reflectance, nominally [0, 1], unclipped.
        returns: ``np.ndarray`` float32, ``(N, C, 4h, 4w)``. ``C == 4`` is the
            SR mean in the input's band order. ``C == 8`` is packed: channels
            0-3 the SR mean, 4-7 the per-pixel Laplace scale ``b`` of each band.
            Both halves are in reflectance units; ``b`` is non-negative.

    ``info``: dict with exactly :data:`INFO_KEYS` -- backend, checkpoint_id,
    params, model_bytes, threads, interim (bool), has_scale_head (bool).
    """

    info: Dict[str, Any]

    def __call__(self, lr: np.ndarray) -> np.ndarray: ...


# ------------------------------------------------------------------ registry
_REGISTRY: Dict[str, Callable[..., Predictor]] = {}


def register_predictor(name: str, factory: Callable[..., Predictor]) -> None:
    """Register a backend factory under ``name``.

    Raises:
        ValueError: ``name`` is already registered to a different factory. A
            silent overwrite would let two modules fight over which backend the
            API serves, and the loser would never know.
    """
    key = str(name)
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not factory:
        raise ValueError(f"predictor {key!r} is already registered to {existing!r}.")
    _REGISTRY[key] = factory


def available_predictors() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def make_predictor(kind: str, **kw: Any) -> Predictor:
    """Build the predictor registered as ``kind`` with keyword arguments ``kw``.

    Raises:
        KeyError: ``kind`` is not registered (the message lists what is).
    """
    key = str(kind)
    if key not in _REGISTRY:
        raise KeyError(f"no predictor {key!r}; registered: {list(available_predictors())}.")
    return _REGISTRY[key](**kw)


# ------------------------------------------------------------ checkpoints
def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_id(path: Union[str, Path], sha256: Optional[str] = None) -> str:
    """``<run-dir-name>-<file stem>-<first 8 hex of the file's sha256>``.

    e.g. ``runs/day3/a2/last.pt`` -> ``a2-last-dce224ec``.
    """
    p = Path(path)
    return f"{p.parent.name}-{p.stem}-{(sha256 or file_sha256(p))[:8]}"


def infer_architecture(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Recover EDSR constructor kwargs from a state dict alone.

    Used only when a checkpoint does not record its architecture in ``args``.
    Every value is read off a tensor shape or key name; nothing is defaulted
    except the log-variance clamp of an uncertainty head, which has no weights
    and comes from ``cfg.uncertainty``.

    - ``n_resblocks``: distinct ``i`` with ``body.<i>.body.0.weight``
      (the body's trailing conv has no inner ``.body``).
    - ``n_feats`` / ``in_ch``: ``head.weight`` is ``(n_feats, in_ch, 3, 3)``.
    - ``out_ch``: ``tail.1.weight`` is ``(out_ch, n_feats, 3, 3)``.
    - ``scale``: 2 ** (convs in ``tail.0``) when each conv expands 4x
      (PixelShuffle(2)); 3 when the single conv expands 9x.

    Raises:
        KeyError: A required key is missing -- not an EDSR state dict.
        ValueError: The upsampler shape matches no supported scale.
    """
    for key in ("head.weight", "tail.1.weight"):
        if key not in state:
            raise KeyError(f"state dict lacks {key!r}; not an EDSR from src/models/edsr.py.")
    n_feats, in_ch = (int(v) for v in state["head.weight"].shape[:2])
    blocks = {k.split(".")[1] for k in state if k.startswith("body.") and ".body.0.weight" in k}
    up = sorted(k for k in state if k.startswith("tail.0.") and k.endswith(".weight"))
    expand = {int(state[k].shape[0]) // n_feats for k in up}
    if expand == {4}:
        scale = 2 ** len(up)
    elif expand == {9} and len(up) == 1:
        scale = 3
    else:
        raise ValueError(f"upsampler convs {up} expand by {expand}; no supported scale.")
    arch: Dict[str, Any] = dict(scale=scale, n_resblocks=len(blocks), n_feats=n_feats,
                                in_ch=in_ch, out_ch=int(state["tail.1.weight"].shape[0]))
    if "var_head.0.weight" in state:
        unc = load_config()["uncertainty"]
        arch.update(uncertainty=True, var_feats=int(state["var_head.0.weight"].shape[0]),
                    logvar_min=float(unc["logvar_min"]), logvar_max=float(unc["logvar_max"]),
                    logvar_init=float(unc["logvar_init"]))
    return arch


def build_model_from_checkpoint(path: Union[str, Path]) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Rebuild and strictly load any src/train.py checkpoint (Run A, A2/B1/B2, ...).

    Architecture comes from the checkpoint's recorded ``args`` when they carry
    every :data:`ARCH_KEYS` entry (delegating to
    :func:`src.models.edsr.model_from_checkpoint`), otherwise from
    :func:`infer_architecture` on the state dict. A bare state dict (no
    ``model`` key) is accepted through the second path.

    Returns:
        ``(model, meta)``: the model in ``eval()`` on CPU; ``meta`` holds
        ``checkpoint_id``, ``path``, ``sha256``, ``file_bytes``,
        ``weights_bytes`` (fp32 parameter bytes), ``params``, ``arch``,
        ``arch_source`` ("args" | "state_dict"), ``iteration`` (1-based, as
        eval_all_ckpts labels it; None if unrecorded), ``git_sha``,
        ``config_hash``, ``has_scale_head``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        KeyError / ValueError / RuntimeError: from the builders or the strict
            load -- never swallowed; a partial load is a silently wrong model.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    # weights_only=False: our own training output; its "args" dict is not a tensor.
    payload = torch.load(p, map_location="cpu", weights_only=False)
    is_train_ckpt = isinstance(payload, dict) and "model" in payload
    args = dict(payload.get("args", {})) if is_train_ckpt else {}

    if is_train_ckpt and all(k in args for k in ARCH_KEYS):
        model, _ = model_from_checkpoint(p, device="cpu")
        arch_source = "args"
        arch = {k: args[k] for k in ARCH_KEYS}
        if bool(int(args.get("uncertainty", 0))):
            arch["uncertainty"] = True
    else:
        state = payload["model"] if is_train_ckpt else payload
        arch = infer_architecture(state)
        model = build_model("edsr_baseline", **arch)
        model.load_state_dict(state)
        model.eval()
        arch_source = "state_dict"
    LOGGER.info("architecture of %s from %s: %s", p, arch_source, arch)

    sha = file_sha256(p)
    it = payload.get("it") if is_train_ckpt else None
    meta = {
        "checkpoint_id": checkpoint_id(p, sha),
        "path": str(p.resolve()),
        "sha256": sha,
        "file_bytes": p.stat().st_size,
        "weights_bytes": int(sum(t.numel() * t.element_size() for t in model.state_dict().values())),
        "params": int(count_params(model)),
        "arch": arch,
        "arch_source": arch_source,
        "iteration": int(it) + 1 if it is not None else None,
        "git_sha": payload.get("git_sha") if is_train_ckpt else None,
        "config_hash": payload.get("config_hash") if is_train_ckpt else None,
        "has_scale_head": bool(getattr(model, "uncertainty", False)),
    }
    return model, meta


# ------------------------------------------------------------------ torch
class _Select(torch.nn.Module):
    """Expose one half of an uncertainty-head model as a single-output module.

    ``tiled.sr_array`` allocates its output with the INPUT's channel count and
    expects a tensor back, so a ``(sr, logvar)`` model cannot go through it
    as-is. Rather than edit tiled.py, the model is run through it twice: once
    returning the SR mean, once returning the Laplace scale.

    The head predicts Gaussian LOG-VARIANCE (src/models/edsr.py). The contract
    reports a Laplace scale ``b``; a Laplace with variance ``v`` has
    ``b = sqrt(v / 2)``, so ``b = sqrt(exp(logvar) / 2)`` in reflectance units.
    A backend whose head predicts ``b`` directly (P6) registers its own
    predictor instead of relying on this conversion.
    """

    def __init__(self, model: torch.nn.Module, part: str) -> None:
        super().__init__()
        self.model, self.part = model, part

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sr, logvar = self.model(x)
        return sr if self.part == "mean" else torch.sqrt(torch.exp(logvar) / 2.0)


class TorchPredictor:
    """CPU fp32 PyTorch backend over ``tiled.sr_array``.

    Args:
        checkpoint_path: A src/train.py checkpoint (see
            :func:`build_model_from_checkpoint`).
        threads: ``torch.set_num_threads`` value; the deployment target is 6.
        interim: Reported in ``info``. False for the A2 winner (Day 3 addendum).
        tile / overlap: LR tile edge and overlap in pixels. ``None`` reads
            ``cfg.frontend.tile.lr_px`` / ``overlap_lr_px``, which mirror the
            tiled.py CLI defaults. An input no larger than ``tile`` in either
            axis is run as ONE tile of its own size -- no padding, no blending
            -- which is exactly eval's ``model(lr)``.
        config: Config path for ``load_config``; ``None`` = configs/base.yaml.
    """

    def __init__(self, checkpoint_path: Union[str, Path], threads: int = 6, interim: bool = False,
                 tile: Optional[int] = None, overlap: Optional[int] = None,
                 config: Optional[Union[str, Path]] = None) -> None:
        cfg = load_config(config) if config is not None else load_config()
        self.scale = int(cfg["sr"]["scale"])
        self.n_bands = len(cfg["dataset"]["bands"])
        self.tile = int(tile if tile is not None else cfg["frontend"]["tile"]["lr_px"])
        self.overlap = int(overlap if overlap is not None else cfg["frontend"]["tile"]["overlap_lr_px"])
        if not 0 <= self.overlap < self.tile:
            raise ValueError(f"overlap {self.overlap} must be in [0, tile={self.tile}).")

        torch.set_num_threads(int(threads))
        self.model, self.meta = build_model_from_checkpoint(checkpoint_path)
        self.model.eval()
        if int(self.meta["arch"]["scale"]) != self.scale:
            raise ValueError(f"checkpoint scale {self.meta['arch']['scale']} != cfg.sr.scale {self.scale}.")
        self._modules = ({"mean": _Select(self.model, "mean"), "scale": _Select(self.model, "scale")}
                         if self.meta["has_scale_head"] else {"mean": self.model})
        self.info: Dict[str, Any] = {
            "backend": "torch-fp32",
            "checkpoint_id": self.meta["checkpoint_id"],
            "params": self.meta["params"],
            # fp32 weight bytes, not the .pt size (which also holds optimiser
            # state), so the number is comparable with an ONNX file size.
            "model_bytes": self.meta["weights_bytes"],
            "threads": int(torch.get_num_threads()),
            "interim": bool(interim),
            "has_scale_head": self.meta["has_scale_head"],
        }

    def _tiling(self, h: int, w: int) -> Tuple[int, int]:
        if h <= self.tile and w <= self.tile:
            # One tile covering the whole input. With tile == max(h, w) the
            # grid in sr_array has exactly one origin; a non-square input is
            # reflect-padded on its short axis by sr_array itself.
            return max(h, w), min(self.overlap, max(h, w) // 2)
        return self.tile, self.overlap

    @torch.no_grad()
    def __call__(self, lr: np.ndarray) -> np.ndarray:
        """See :class:`Predictor`. ``lr`` float32 ``(N, 4, h, w)`` reflectance.

        Raises:
            TypeError: ``lr`` is not a float32 ndarray.
            ValueError: wrong rank or band count, or non-finite values.
        """
        if not isinstance(lr, np.ndarray) or lr.dtype != np.float32:
            raise TypeError(f"lr must be a float32 np.ndarray, got {type(lr).__name__} "
                            f"{getattr(lr, 'dtype', None)}.")
        if lr.ndim != 4 or lr.shape[1] != self.n_bands:
            raise ValueError(f"lr must be (N, {self.n_bands}, h, w), got {lr.shape}.")
        if not np.isfinite(lr).all():
            raise ValueError("lr contains non-finite values; refusing to super-resolve them.")
        n, _, h, w = lr.shape
        tile, overlap = self._tiling(h, w)
        outs = []
        for i in range(n):
            parts = [sr_array(np.ascontiguousarray(lr[i]), module, scale=self.scale, tile=tile,
                              overlap=overlap, device="cpu", amp=False)
                     for module in self._modules.values()]
            outs.append(np.concatenate(parts, axis=0))
        return np.stack(outs).astype(np.float32, copy=False)


register_predictor("torch", TorchPredictor)
