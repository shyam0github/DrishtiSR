"""Laplace negative log-likelihood for the post-hoc scale head (P6).

    L = mean( w * ( log(2b) + |y - stopgrad(mu)| / b ) ),   w = stopgrad(b) ** beta

``mu`` is always detached: the backbone is frozen, and stopping the gradient
here as well means the mean cannot move even if a caller forgets to freeze it.
beta = 0 is the plain NLL; beta > 0 is beta-NLL weighting (Seitzer et al.
2022), which matters only when the mean is trained jointly -- kept as an option.
Units: ``y``, ``mu`` and ``b`` in surface reflectance.
"""

from __future__ import annotations

import torch

__all__ = ["laplace_nll"]


def laplace_nll(mu: torch.Tensor, b: torch.Tensor, y: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    """Scalar Laplace NLL (see module docstring). ``b`` must be > 0."""
    resid = (y - mu.detach()).abs()
    nll = torch.log(2.0 * b) + resid / b
    if beta != 0.0:
        nll = nll * b.detach() ** float(beta)
    return nll.mean()
