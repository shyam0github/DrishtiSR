"""Non-learned super-resolution baselines: the numbers every model must beat.

Two baselines, and the pair of them is the point:

- :func:`bicubic_upsample` -- **the bar**. Antialiased bicubic is what a GIS
  analyst gets for free, in one line, with no GPU hours. A learned model that
  does not clear it by a comfortable margin has not earned its parameter budget,
  and the judges will run this comparison whether or not we do.
- :func:`nearest_upsample` -- **the floor**. Pixel replication adds nothing
  whatsoever. It exists so a metric can be read as a scale rather than as a
  number: knowing bicubic scores 30 dB means little until you know replication
  scores 28 and the gap the metric can express is that narrow.

Neither of these clips. Bicubic overshoots at sharp edges and can push
reflectance slightly below 0 or above the local maximum; that overshoot is real
output and is carried into the metrics, because the spectral-consistency
contribution stands or falls on reflectance staying physically meaningful end to
end, including where a method makes it unphysical.

Which bicubic kernel, and why it matters
----------------------------------------
"Bicubic" names a family, and the members disagree by more than the effect we are
trying to measure. MEASURED on random 16x16 -> 64x64 upsamples with torch 2.5.1:
``F.interpolate(..., mode="bicubic", antialias=True)`` and the same call with
``antialias=False`` differ by up to **0.083 reflectance**, because the two use
different cubic coefficients (``a = -0.5`` and ``a = -0.75`` respectively). That
is far larger than the difference between two competing SR architectures.

This module uses ``antialias=True``. The reason is comparability, not the
antialiasing itself: the ``a = -0.5`` kernel is the one MATLAB's ``imresize`` and
PIL implement, and MATLAB ``imresize`` bicubic is the de facto baseline in the
super-resolution literature, so a bicubic PSNR reported here is comparable to a
published one. The ``antialias`` flag additionally widens the kernel when
*down*sampling, which is what makes it correct for the degradation direction
(``src/data/`` and the spectral-consistency loss) as well as the upsampling one.

``src/eval/alignment.py`` deliberately calls this function with
``antialias=False``: registration needs the kernel that audit was measured with,
and for phase correlation the choice is immaterial. Both call sites now go
through the one implementation here, so there is no second bicubic in the
repository to drift from this one.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "bicubic_upsample",
    "nearest_upsample",
    "available_baselines",
    "get_baseline",
]

Image = Union[np.ndarray, torch.Tensor]


def _to_batch(image: Image, name: str) -> Tuple[torch.Tensor, bool, bool]:
    """Normalise to a float32 ``(B, C, H, W)`` tensor for resampling.

    Args:
        image: ``(C, H, W)`` or ``(B, C, H, W)``, NumPy array or torch tensor,
            float32 surface reflectance, nominally ``[0, 1]`` and unclipped.
        name: Argument name, for error messages.

    Returns:
        ``(tensor, was_batched, was_numpy)``. The tensor shares memory with a
        torch input where possible; resampling allocates a new one regardless.

    Raises:
        TypeError: ``image`` is neither an array nor a tensor.
        ValueError: Rank is not 3 or 4, or an axis is empty.
    """
    if isinstance(image, torch.Tensor):
        tensor, was_numpy = image, False
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        was_numpy = True
    else:
        raise TypeError(
            f"{name} must be a numpy.ndarray or torch.Tensor of surface "
            f"reflectance, got {type(image).__name__}."
        )

    if tensor.ndim == 3:
        batched = False
        tensor = tensor[None]
    elif tensor.ndim == 4:
        batched = True
    else:
        raise ValueError(
            f"{name} must be (C, H, W) or (B, C, H, W); got shape "
            f"{tuple(tensor.shape)}. The channel axis comes before the spatial "
            "axes everywhere in this project."
        )

    if min(tensor.shape) == 0:
        raise ValueError(f"{name} has an empty axis: shape {tuple(tensor.shape)}.")

    return tensor.to(torch.float32), batched, was_numpy


def _restore(
    tensor: torch.Tensor, batched: bool, was_numpy: bool
) -> Image:
    """Return a resampled batch in the caller's container type and rank."""
    out = tensor if batched else tensor[0]
    if was_numpy:
        return out.numpy().astype(np.float32, copy=False)
    return out


def _check_scale(scale: Any) -> int:
    factor = int(scale)
    if factor < 1:
        raise ValueError(
            f"scale must be a positive integer (cfg.sr.scale); got {scale!r}."
        )
    return factor


def bicubic_upsample(lr: Image, scale: int, antialias: bool = True) -> Image:
    """Bicubically upsample low-resolution reflectance to the HR grid.

    The reference baseline for this project. See the module docstring for why
    ``antialias=True`` is the default and why the choice of kernel is not
    cosmetic.

    **The result is not clipped.** Bicubic interpolation overshoots at sharp
    edges -- a field boundary against a bright roof produces a ringing halo -- so
    the output can dip below 0 or rise above the input maximum. Those values are
    kept. Clipping them would flatter the baseline, would change the radiometry
    the spectral-consistency objective is defined against, and would hide the
    exact artefact a learned model is supposed to remove.

    Args:
        lr: Low-resolution image. ``(C, H, W)`` or ``(B, C, H, W)``, NumPy array
            or torch tensor, float32 **surface reflectance**, nominally
            ``[0, 1]`` but unclipped -- bright targets exceed 1.0.
        scale: Upsampling factor, from ``cfg.sr.scale``. 4 for 10 m -> 2.5 m.
        antialias: Select the cubic kernel. ``True`` (default) is the
            ``a = -0.5`` MATLAB/PIL kernel that the SR literature reports
            against, and it widens the kernel when downsampling. ``False`` is
            torch's plain ``a = -0.75`` kernel; ``src/eval/alignment.py`` passes
            it to preserve the behaviour its audit was measured with.

    Returns:
        ``(C, H*scale, W*scale)`` or ``(B, C, H*scale, W*scale)`` -- the same
        container type and rank as ``lr`` -- float32 surface reflectance in the
        same units and band order, unclipped.

    Raises:
        TypeError: ``lr`` is neither an array nor a tensor.
        ValueError: ``lr`` is not rank 3 or 4, has an empty axis, or ``scale`` is
            not a positive integer.
    """
    tensor, batched, was_numpy = _to_batch(lr, "lr")
    factor = _check_scale(scale)
    out = F.interpolate(
        tensor,
        scale_factor=factor,
        mode="bicubic",
        align_corners=False,
        antialias=bool(antialias),
    )
    return _restore(out, batched, was_numpy)


def nearest_upsample(lr: Image, scale: int) -> Image:
    """Replicate each pixel ``scale`` times in both axes -- the absolute floor.

    Nearest-neighbour upsampling adds no information at all: the output is the
    input with bigger pixels. Its scores are the value of the metric on a method
    that has done literally nothing, which is what makes a bicubic score legible
    and what stops a marginal model being reported as a success.

    Implemented as exact pixel replication (``repeat_interleave``), which is
    bit-identical to ``F.interpolate(mode="nearest")`` for integer factors and
    involves no kernel, no interpolation, and therefore no overshoot: every
    output value is exactly an input reflectance value.

    Args:
        lr: Low-resolution image. ``(C, H, W)`` or ``(B, C, H, W)``, NumPy array
            or torch tensor, float32 surface reflectance, nominally ``[0, 1]``,
            unclipped.
        scale: Replication factor, from ``cfg.sr.scale``.

    Returns:
        ``(C, H*scale, W*scale)`` or ``(B, C, H*scale, W*scale)`` -- same
        container type and rank as ``lr`` -- float32 surface reflectance,
        containing exactly the input values.

    Raises:
        TypeError: ``lr`` is neither an array nor a tensor.
        ValueError: ``lr`` is not rank 3 or 4, has an empty axis, or ``scale`` is
            not a positive integer.
    """
    tensor, batched, was_numpy = _to_batch(lr, "lr")
    factor = _check_scale(scale)
    out = tensor.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)
    return _restore(out.contiguous(), batched, was_numpy)


_BASELINES: Dict[str, Callable[..., Image]] = {
    "bicubic": bicubic_upsample,
    "nearest": nearest_upsample,
}


def available_baselines() -> List[str]:
    """Names accepted by :func:`get_baseline`, for ``--baseline`` help text."""
    return sorted(_BASELINES)


def get_baseline(name: str, scale: int) -> Callable[[Image], Image]:
    """Bind a baseline to a scale factor, giving an ``sr_fn`` for the Evaluator.

    :class:`src.metrics.aggregate.Evaluator` takes a callable of one argument, so
    that a bicubic baseline and a trained model are evaluated by exactly the same
    code path. This is the adapter that makes a baseline fit that shape.

    Args:
        name: ``"bicubic"`` or ``"nearest"``; see :func:`available_baselines`.
        scale: Upsampling factor, from ``cfg.sr.scale``.

    Returns:
        ``sr_fn(lr) -> sr``, taking and returning ``(C, H, W)`` or
        ``(B, C, H, W)`` float32 surface reflectance, unclipped.

    Raises:
        KeyError: ``name`` is not a registered baseline.
        ValueError: ``scale`` is not a positive integer.
    """
    key = str(name)
    if key not in _BASELINES:
        raise KeyError(
            f"Unknown baseline {key!r}. Available: {available_baselines()}."
        )
    factor = _check_scale(scale)
    function = _BASELINES[key]

    def sr_fn(lr: Image) -> Image:
        return function(lr, factor)

    sr_fn.__name__ = f"{key}_x{factor}"
    sr_fn.__doc__ = (
        f"{key} upsampling by x{factor}. Takes and returns float32 surface "
        "reflectance, (C, H, W) or (B, C, H, W), unclipped."
    )
    return sr_fn
