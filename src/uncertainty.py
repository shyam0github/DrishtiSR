"""TTA-disagreement uncertainty: the fallback that ships if the NLL head collapses.

Run the model on all 8 dihedral transforms of the input (the symmetries of the
square, :data:`src.data.augment.DIHEDRAL`), map each output back to the
original frame with the exact inverse, and report the per-pixel mean and
standard deviation across the 8. A convolutional SR model is not equivariant to
the dihedral group -- nothing in training forces it to be, augmentation only
encourages it -- so where the 8 predictions disagree, the model's answer
depends on which way up the scene was presented. That dependence is an
uncertainty signal, and it needs no retraining: it works on ANY checkpoint,
Run A, the Day 3 runs, or an uncertainty-enabled model (whose SR output is
used; its log-variance is ignored here).

BATCHED. The 8 transformed copies are stacked into the batch dimension and run
through the model in as few forward calls as the shapes allow: one call for
square inputs, two for non-square ones (odd quarter-turns swap H and W, so the
even and odd elements form two shape groups). ``chunk_size`` caps images per
call when memory is the constraint -- on CPU at x4 the HR feature maps of a
64-image stack are the bulk of it.

NO VALUE IS TOUCHED. Transforms and inverses are index permutations
(``torch.flip`` / ``torch.rot90``); reflectance is neither normalised nor
clipped, and the inversion is exact, which ``tests/test_tta.py`` asserts
bit-for-bit with an identity model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from src.data.augment import DIHEDRAL, dihedral, invert_dihedral

__all__ = ["TTAResult", "tta_predict", "sr_output"]

ModelOutput = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


@dataclass
class TTAResult:
    """What :func:`tta_predict` returns.

    Attributes:
        mean: ``float32``, ``(B, C, H*s, W*s)``. The mean of the 8 inverted
            predictions -- surface reflectance, unclipped. This is the SR
            output that ships alongside ``std``.
        std: ``float32``, same shape. Population standard deviation
            (``correction=0``) across the 8, in reflectance units. Population,
            not sample: these are all 8 members of a finite group, not a draw
            from a larger population.
        members: ``float32``, ``(8, B, C, H*s, W*s)`` in :data:`DIHEDRAL`
            order (index 0 is the identity, i.e. the plain forward pass), or
            None unless requested.
    """

    mean: torch.Tensor
    std: torch.Tensor
    members: Optional[torch.Tensor] = None


def sr_output(out: ModelOutput) -> torch.Tensor:
    """The SR tensor from a model output that may be ``(sr, logvar)``.

    Args:
        out: A tensor, or a tuple/list whose first element is the SR tensor
            (the uncertainty-enabled :class:`src.models.edsr.EDSR` returns
            ``(sr, logvar)``).

    Returns:
        The SR tensor.

    Raises:
        TypeError: ``out`` is neither.
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    raise TypeError(f"model returned {type(out).__name__}; expected a tensor or (sr, ...).")


@torch.no_grad()
def tta_predict(
    model: Callable[[torch.Tensor], ModelOutput],
    lr: torch.Tensor,
    chunk_size: Optional[int] = None,
    return_members: bool = False,
) -> TTAResult:
    """8-way dihedral test-time augmentation: mean prediction and disagreement.

    Args:
        model: Callable ``(N, C, h, w) -> (N, C, h*s, w*s)`` (or a tuple whose
            first element is that), fully convolutional so it accepts the
            rotated ``(w, h)`` shape too. Called in whatever mode it is in; put
            a module in ``eval()`` first.
        lr: ``float32``, ``(B, C, h, w)``. Surface reflectance, nominally
            [0, 1], UNCLIPPED, in ``cfg.dataset.bands`` order.
        chunk_size: Maximum images per forward call. None stacks each shape
            group whole (8B images for a square input).
        return_members: Also return the 8 inverted predictions.

    Returns:
        :class:`TTAResult`.

    Raises:
        ValueError: ``lr`` is not 4-D, ``chunk_size`` is < 1, or the inverted
            outputs disagree in shape (the model is not fully convolutional,
            and averaging misaligned outputs would be meaningless).
    """
    if lr.ndim != 4:
        raise ValueError(f"expected lr of shape (B, C, h, w), got {tuple(lr.shape)}")
    if chunk_size is not None and int(chunk_size) < 1:
        raise ValueError(f"chunk_size must be >= 1 or None, got {chunk_size!r}")
    batch = lr.shape[0]

    # Group the 8 elements by the SHAPE of the transformed input, so each group
    # stacks into one batch. Square input: one group of 8.
    groups: Dict[Tuple[int, int], List[int]] = {}
    for index, (hflip, vflip, k) in enumerate(DIHEDRAL):
        shape = tuple(lr.shape[-2:]) if k % 2 == 0 else tuple(lr.shape[-2:][::-1])
        groups.setdefault(shape, []).append(index)

    members: List[Optional[torch.Tensor]] = [None] * len(DIHEDRAL)
    for indices in groups.values():
        stacked = torch.cat([dihedral(lr, *DIHEDRAL[i]) for i in indices], dim=0)
        step = stacked.shape[0] if chunk_size is None else int(chunk_size)
        outs = [
            sr_output(model(stacked[start:start + step])).float()
            for start in range(0, stacked.shape[0], step)
        ]
        out = torch.cat(outs, dim=0)
        for position, index in enumerate(indices):
            piece = out[position * batch:(position + 1) * batch]
            members[index] = invert_dihedral(piece, *DIHEDRAL[index])

    shapes = {tuple(m.shape) for m in members}
    if len(shapes) != 1:
        raise ValueError(
            f"inverted TTA outputs disagree in shape: {sorted(shapes)}. The model "
            "is not fully convolutional over rotated inputs."
        )
    stack = torch.stack(members, dim=0)
    std, mean = torch.std_mean(stack, dim=0, correction=0)
    return TTAResult(mean=mean, std=std, members=stack if return_members else None)
