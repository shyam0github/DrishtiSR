"""Reproducibility: one call that seeds every RNG the project touches.

Two layers live here.

``seed_everything`` seeds the GLOBAL RNGs once per process, which covers weight
initialisation and anything incidental.

``derive_seed`` / ``torch_generator`` / ``numpy_generator`` cover the data
stream, and they exist because the global RNGs cannot. A DataLoader worker's
global RNG advances by however many items that worker happened to draw, so the
crop and augmentation a given sample receives depends on ``num_workers``, on
batch order, and on how far into the epoch the process got before it was
restarted. That is fatal here: Run A2 (the control) and Run B (the fix) are only
comparable if they see a BYTE-IDENTICAL data stream, and a headline claim built
on two runs that quietly differed in their crops is worthless.

So every random decision about a sample is a pure function of the triple
``(seed, epoch, index)`` plus a stream name. Same triple, same crop, same flip,
on 0 workers or 6, resumed or not, on Windows or on Kaggle.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch

__all__ = [
    "seed_everything",
    "derive_seed",
    "torch_generator",
    "numpy_generator",
]

# Width of the derived seed. 63 bits, not 64: torch.Generator.manual_seed takes
# a signed 64-bit integer, so the top bit must stay clear or a perfectly valid
# derived seed raises on some platforms.
_SEED_BITS = 63
_SEED_MASK = (1 << _SEED_BITS) - 1


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA) RNGs.

    Call this once at the top of every entry-point script, before constructing
    datasets, models, or dataloaders. Worker processes additionally need their
    own per-worker seeding via the DataLoader ``worker_init_fn``.

    Args:
        seed: The seed. Read from ``cfg.seed`` -- do not pass a literal.
        deterministic: When True, force cuDNN into deterministic mode and disable
            its autotuner. This costs throughput on the Kaggle P100 but makes
            ablation numbers comparable across runs, which the report depends on.
            Set False only for a deliberate speed run.

    Returns:
        The seed, so callers can log exactly what was applied.
    """
    seed = int(seed)

    # Affects hash randomisation in child processes (dataloader workers).
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # No-op when CUDA is unavailable.

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    return seed


def derive_seed(seed: int, epoch: int, index: int, stream: str) -> int:
    """A reproducible seed for one sample's one random decision.

    BLAKE2b over the decimal spelling of the inputs, NOT Python's ``hash()``:
    ``hash()`` on a str is salted per process by ``PYTHONHASHSEED`` and would
    give a different crop in every worker, which is the exact bug this function
    exists to make impossible.

    ``stream`` separates independent decisions about the SAME sample. Crops and
    augmentation draws must not share a stream, or advancing one silently
    shifts the other and a change to the augmentation code changes which pixels
    were cropped.

    Args:
        seed: The run seed, ``cfg.seed`` / ``--seed``. Not a literal.
        epoch: Epoch number, 0-based. Fixed at 0 for deterministic splits.
        index: Item index within the dataset.
        stream: Name of the decision, e.g. ``"crop"`` or ``"augment"``.

    Returns:
        A non-negative int below ``2**63``, safe for both
        ``torch.Generator.manual_seed`` and ``numpy.random.default_rng``.
        Stable across processes, platforms, and Python versions.
    """
    payload = f"{int(seed)}:{int(epoch)}:{int(index)}:{stream}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & _SEED_MASK


def torch_generator(seed: int, epoch: int, index: int, stream: str) -> torch.Generator:
    """A CPU ``torch.Generator`` seeded from :func:`derive_seed`.

    CPU-side on purpose: this seeds data-pipeline decisions, which happen in
    DataLoader worker processes where no CUDA context exists.

    Args:
        seed: See :func:`derive_seed`.
        epoch: See :func:`derive_seed`.
        index: See :func:`derive_seed`.
        stream: See :func:`derive_seed`.

    Returns:
        A freshly seeded generator. Cheap enough to build per item.
    """
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, epoch, index, stream))
    return generator


def numpy_generator(seed: int, epoch: int, index: int, stream: str) -> np.random.Generator:
    """A NumPy ``Generator`` seeded from :func:`derive_seed`.

    NumPy rather than torch for crop origins because
    ``src.data.patches.extract_patches`` draws through the NumPy API. The SEED
    is derived identically, so the determinism guarantee is the same one; only
    the bit generator downstream of it differs.

    Args:
        seed: See :func:`derive_seed`.
        epoch: See :func:`derive_seed`.
        index: See :func:`derive_seed`.
        stream: See :func:`derive_seed`.

    Returns:
        A freshly seeded generator.
    """
    return np.random.default_rng(derive_seed(seed, epoch, index, stream))
