"""Geometric augmentation for LR/HR reflectance pairs.

WHY THIS FILE EXISTS. Run A peaked at 8k iterations and declined to 40k on
~2400 training patches: the model memorised the set. The cheapest correct fix
for an SR model is the dihedral group -- the 8 symmetries of the square -- which
is label-preserving for super-resolution in a way that photometric jitter is
not. Reflectance is a physical quantity; brightness or contrast jitter would
break the spectral-consistency contract this project is built on, so NOTHING
here touches pixel VALUES. Only the spatial axes move.

THE COUPLING IS THE WHOLE POINT. An LR/HR pair is only a training target while
the HR is the same ground, at 4x, as the LR. Applying a transform drawn twice --
once for LR, once for HR -- silently destroys that, and the loss becomes noise
the model cannot fit, which looks exactly like "the architecture is bad". Hence:

  * one draw per pair, applied to both tensors by the same code path;
  * torch ``flip``/``rot90`` on the spatial dims, NOT torchvision transforms,
    which are constructed per call and would need the RNG threaded through two
    separate invocations to stay in step;
  * ``tests/test_augment.py`` proves the coupling by area-downsampling the
    augmented HR and comparing it against the augmented LR.

Flips and rot90 commute with block-mean downsampling, so a pair that satisfied
``area_downsample(hr, scale) == lr`` before the transform still satisfies it
after. That identity is what the test checks, and it is why this augmentation
is safe to apply after patch extraction rather than before.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "DIHEDRAL",
    "dihedral",
    "invert_dihedral",
    "apply_dihedral_pair",
    "augment_pair",
    "area_downsample",
]

# The 8 elements of the dihedral group of the square, as ``(hflip, vflip, k)``
# triples in the order :func:`_transform` applies them. Exported so a test can
# enumerate the group rather than re-derive it, and so the set is stated once.
DIHEDRAL: Tuple[Tuple[bool, bool, int], ...] = (
    (False, False, 0),  # identity
    (False, False, 1),  # rot90
    (False, False, 2),  # rot180
    (False, False, 3),  # rot270
    (True, False, 0),   # horizontal mirror
    (True, False, 1),   # mirror then rot90
    (True, False, 2),   # mirror then rot180  (== vertical mirror)
    (True, False, 3),   # mirror then rot270
)


def _transform(x: torch.Tensor, hflip: bool, vflip: bool, k: int) -> torch.Tensor:
    """Apply one dihedral element to the spatial axes of a CHW tensor.

    Fixed order -- horizontal flip, then vertical flip, then ``k`` quarter
    turns -- because the group is non-abelian: swapping the order gives a
    different (still valid) element, and a caller comparing two tensors
    transformed by "the same" parameters would get a mismatch.

    Args:
        x: ``torch.float32``, shape ``(C, H, W)``. Surface reflectance,
            nominally [0, 1] and UNCLIPPED -- bright targets exceed 1.0. Values
            are not read, only moved.
        hflip: Mirror left-right (along ``W``, dim ``-1``).
        vflip: Mirror top-bottom (along ``H``, dim ``-2``).
        k: Quarter turns counter-clockwise in the ``(H, W)`` plane, 0..3.

    Returns:
        ``torch.float32``, shape ``(C, H, W)`` when ``k`` is even and
        ``(C, W, H)`` when odd -- identical for the square patches this project
        trains on. Contiguous, and a copy: the input is never mutated. Surface
        reflectance passes through bit for bit; no value is scaled, clipped, or
        reordered across channels.

    Raises:
        ValueError: ``x`` has fewer than two dimensions, or ``k`` is outside
            0..3.
    """
    if x.ndim < 2:
        raise ValueError(
            f"expected a tensor with at least 2 spatial dims (C, H, W), got "
            f"shape {tuple(x.shape)}"
        )
    if not 0 <= int(k) <= 3:
        raise ValueError(f"k must be one of 0, 1, 2, 3; got {k!r}")

    if hflip:
        x = torch.flip(x, dims=(-1,))
    if vflip:
        x = torch.flip(x, dims=(-2,))
    if k:
        x = torch.rot90(x, int(k), dims=(-2, -1))
    return x.contiguous()


def dihedral(x: torch.Tensor, hflip: bool, vflip: bool, k: int) -> torch.Tensor:
    """Public form of :func:`_transform`, for tensors with any leading dims.

    Used by test-time augmentation (:mod:`src.uncertainty`), which applies the
    same group to ``(B, C, H, W)`` batches. Only the last two axes move.

    Args:
        x: Any float dtype, shape ``(..., H, W)``. Surface reflectance,
            unclipped; values are not read, only moved.
        hflip: Mirror along ``W``.
        vflip: Mirror along ``H``.
        k: Counter-clockwise quarter turns, 0..3.

    Returns:
        Same dtype, shape ``(..., H, W)`` for even ``k`` and ``(..., W, H)``
        for odd ``k``; a contiguous copy.
    """
    return _transform(x, hflip, vflip, k)


def invert_dihedral(x: torch.Tensor, hflip: bool, vflip: bool, k: int) -> torch.Tensor:
    """Undo :func:`dihedral` with the same parameters, exactly.

    ``dihedral`` applies hflip, then vflip, then ``k`` turns; the inverse is
    ``-k`` turns, then vflip, then hflip (flips are their own inverses). Pure
    index permutation: ``invert_dihedral(dihedral(x, *g), *g)`` equals ``x``
    bit for bit, which ``tests/test_tta.py`` asserts for all eight elements.

    Args:
        x: Any float dtype, shape ``(..., H', W')`` -- a tensor in the
            transformed frame, e.g. a model output on a transformed input.
            Values are not read, only moved.
        hflip: As passed to :func:`dihedral`.
        vflip: As passed to :func:`dihedral`.
        k: As passed to :func:`dihedral`, 0..3.

    Returns:
        Same dtype, the tensor in the original frame; a contiguous copy.

    Raises:
        ValueError: ``x`` has fewer than two dims or ``k`` is outside 0..3.
    """
    if x.ndim < 2:
        raise ValueError(f"expected at least 2 spatial dims, got shape {tuple(x.shape)}")
    if not 0 <= int(k) <= 3:
        raise ValueError(f"k must be one of 0, 1, 2, 3; got {k!r}")
    if k:
        x = torch.rot90(x, -int(k), dims=(-2, -1))
    if vflip:
        x = torch.flip(x, dims=(-2,))
    if hflip:
        x = torch.flip(x, dims=(-1,))
    return x.contiguous()


def apply_dihedral_pair(
    lr: torch.Tensor,
    hr: torch.Tensor,
    hflip: bool,
    vflip: bool,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply ONE dihedral element to both halves of a pair.

    This is the only function in the project that transforms a pair, so LR and
    HR cannot drift apart: there is no code path where one receives a transform
    the other did not.

    Args:
        lr: ``torch.float32``, shape ``(C, h, w)``. Surface reflectance,
            nominally [0, 1], unclipped.
        hr: ``torch.float32``, shape ``(C, h*scale, w*scale)``. Surface
            reflectance in the SAME units and band order as ``lr``, nominally
            [0, 1], unclipped.
        hflip: See :func:`_transform`.
        vflip: See :func:`_transform`.
        k: See :func:`_transform`.

    Returns:
        ``(lr, hr)`` transformed identically: same dtypes, and for the square
        patches this project trains on, the same shapes as the inputs. Surface
        reflectance values are unchanged.

    Raises:
        ValueError: ``lr`` and ``hr`` disagree on channel count, or ``k`` is
            outside 0..3 (raised by :func:`_transform`).
    """
    if lr.shape[0] != hr.shape[0]:
        raise ValueError(
            f"lr and hr must have the same channel count; got {lr.shape[0]} "
            f"and {hr.shape[0]}. A pair with mismatched bands is not a pair."
        )
    return (
        _transform(lr, hflip, vflip, k),
        _transform(hr, hflip, vflip, k),
    )


def augment_pair(
    lr: torch.Tensor,
    hr: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw one dihedral element and apply it to both halves of a pair.

    Draws exactly three values -- ``hflip`` (p=0.5), ``vflip`` (p=0.5) and ``k``
    (uniform on 0..3) -- from a SINGLE stream, then applies them once. The 16
    ``(hflip, vflip, k)`` combinations cover the 8-element dihedral group twice
    each, so the distribution over distinct transforms is uniform.

    RNG. ``generator=None`` uses the process- or worker-global torch RNG, which
    is the intended path inside a DataLoader worker: ``torch.utils.data`` seeds
    each worker from the base seed (itself set by ``set_seed(args.seed)`` in
    ``src/train.py``), so the sequence is reproducible from seed plus worker
    count AND it advances with every item drawn. That second property is why
    this is not seeded from ``(seed, epoch, index)`` like the crop RNG in
    ``src/data/loader.py``: ``src/train.py`` iterates an infinite loader and
    never calls ``set_epoch``, so an epoch-seeded draw would hand every index
    the same transform on every pass -- 2400 fixed patches wearing 2400 fixed
    disguises, which is not augmentation. Pass an explicit ``generator`` when a
    test or an ablation needs an isolated, repeatable stream.

    Args:
        lr: ``torch.float32``, shape ``(C, h, w)``. Surface reflectance,
            nominally [0, 1], unclipped.
        hr: ``torch.float32``, shape ``(C, h*scale, w*scale)``. Surface
            reflectance, same units and band order as ``lr``, unclipped.
        generator: Optional ``torch.Generator`` for the three draws.

    Returns:
        ``(lr, hr)`` under the same randomly drawn dihedral element. Shapes,
        dtypes and reflectance values are unchanged; only the spatial layout
        moves.
    """
    bits = torch.randint(0, 2, (2,), generator=generator)
    k = int(torch.randint(0, 4, (1,), generator=generator).item())
    return apply_dihedral_pair(lr, hr, bool(bits[0].item()), bool(bits[1].item()), k)


def area_downsample(hr: torch.Tensor, scale: int) -> torch.Tensor:
    """Block-mean downsample -- the operation the augmentation must commute with.

    Used by the tests to reconstruct the LR structure from the HR and so prove
    the two were transformed together. It lives here, beside the transform,
    because a second implementation of "downsample" living in the test file is
    how such a check quietly starts testing something else.

    Args:
        hr: ``torch.float32``, shape ``(C, H, W)`` with ``H`` and ``W`` both
            divisible by ``scale``. Surface reflectance, nominally [0, 1],
            unclipped.
        scale: Integer downsampling factor, >= 1.

    Returns:
        ``torch.float32``, shape ``(C, H // scale, W // scale)``. Surface
        reflectance in the same units and band order, unclipped -- the mean of
        a block of reflectances is a reflectance.

    Raises:
        ValueError: ``scale`` is below 1, or the spatial dims are not divisible
            by it.
    """
    if int(scale) < 1:
        raise ValueError(f"scale must be >= 1; got {scale!r}")
    scale = int(scale)
    c, h, w = hr.shape[-3], hr.shape[-2], hr.shape[-1]
    if h % scale or w % scale:
        raise ValueError(
            f"cannot block-mean a {h}x{w} tensor by {scale}: both spatial dims "
            f"must be divisible by scale."
        )
    return hr.reshape(c, h // scale, scale, w // scale, scale).mean(dim=(2, 4))
