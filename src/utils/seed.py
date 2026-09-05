"""Reproducibility: one call that seeds every RNG the project touches."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


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
