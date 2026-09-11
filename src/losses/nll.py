"""Heteroscedastic Gaussian NLL, and the warmup that keeps it from eating the model.

THE OBJECTIVE. With ``s = logvar`` the head's predicted log-variance of the
reflectance at each pixel and band::

    NLL = mean( 0.5 * ( exp(-s) * (y - yhat)^2 + s ) )

(the constant ``0.5 * log(2*pi)`` is dropped). Minimising over ``s`` for a fixed
residual ``r`` gives ``s* = log(r^2)``: the head is trained to predict the
squared error it will make, which is exactly what turns ``exp(0.5 * s)`` into a
per-pixel hallucination map (technical contribution 2).

THE FAILURE IT MUST BE PROTECTED FROM. Trained from scratch, the cheapest way to
lower this loss is to raise ``s`` everywhere: ``exp(-s)`` then scales the
reconstruction gradient toward zero, the mean stops learning, and a large
uniform variance "explains" the resulting bad reconstruction. Hence the
schedule in :func:`nll_weight_at`:

1. ``it < warmup_iters``: weight 0. The objective is pure L1 and the trainer
   calls the model with ``detach_var=True``, so the variance branch has no path
   to the shared trunk at all -- the reconstruction learns exactly as Day 3's.
2. the next ``ramp_iters``: the NLL weight rises linearly from 0 to ``weight``.
3. afterwards: ``weight``, alongside the L1 term, which is kept as an anchor.

UNITS, and a number worth knowing before Day 4. In reflectance, Run A's L1 was
0.0088, i.e. a Gaussian sigma near 0.011 and ``s*`` near -9. The default clamp
floor of -10 is sigma = 0.0067. Pixels whose conditional error is smaller than
that -- flat, dark, well-predicted ground -- will sit ON the floor with zero
gradient, so a non-trivial ``logvar_clamp_lo_frac`` is expected from a healthy
head and is not by itself the collapse signal; a collapsing
``logvar_spatial_std`` is. ``scripts/calibrate_uncertainty.py`` reports how
much pixel mass has an implied log-variance below the floor.

NUMERICS. Computed in float32 outside autocast by the caller: ``exp(10)``
overflows nothing in fp32, but a residual^2 of 1e-6 is below fp16's normal
range, and a loss that moves with the AMP setting is not a loss.

GRADIENT CONTROL -- WHY THE NLL DOES NOT TRAIN THE SR OUTPUT BY DEFAULT. At the
SR output, dL1/dsr = sign(r)/N while dNLL/dsr = exp(-s) * r / N. With s ~ -9
and r ~ 0.01 reflectance that is exp(9) * 0.01 ~ 80x the L1 gradient per pixel
-- the "90x" measured before Day 4 -- so a joint path would hand the
reconstruction to the NLL, which down-weights exactly the high-error pixels
(large s) and spends PSNR, a headline-table number, to do it. Hence
:func:`nll_objective`'s ``detach_sr`` (``--nll-detach-sr``, default on): the
NLL sees a DETACHED SR, and the trainer also feeds the head detached features,
so the NLL's gradient reaches the variance branch and nothing else. The
reconstruction then trains exactly as a Day 3 run's. :func:`sr_grad_ratio`
measures the ratio every validation, in both modes, so the 90x is a logged
number rather than an assumption.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["gaussian_nll", "nll_objective", "nll_weight_at", "sr_grad_ratio"]


def gaussian_nll(
    sr: torch.Tensor,
    logvar: torch.Tensor,
    hr: torch.Tensor,
    logvar_min: float = -10.0,
    logvar_max: float = 10.0,
) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood, averaged.

    Args:
        sr: ``float32``, ``(B, C, H, W)``. Predicted surface reflectance,
            nominally [0, 1], unclipped.
        logvar: ``float32``, same shape. Natural log of the predicted variance
            of the reflectance (units reflectance^2). Clamped here to
            ``[logvar_min, logvar_max]`` even if the model already did, so a
            caller passing a raw head output cannot bypass the bound.
        hr: ``float32``, same shape. Ground-truth surface reflectance,
            unclipped.
        logvar_min: Lower clamp. Gradient w.r.t. ``logvar`` is zero below it.
        logvar_max: Upper clamp.

    Returns:
        Scalar ``float32`` tensor: ``mean(0.5 * (exp(-s) * (hr - sr)^2 + s))``
        over every element. Can be negative (``s`` < 0 is the normal regime in
        reflectance units).

    Raises:
        ValueError: The three tensors disagree in shape. Broadcasting a
            per-band variance over a per-pixel residual would be a silent
            change of model.
    """
    if not (sr.shape == logvar.shape == hr.shape):
        raise ValueError(
            f"sr {tuple(sr.shape)}, logvar {tuple(logvar.shape)} and hr "
            f"{tuple(hr.shape)} must have identical shapes."
        )
    s = logvar.clamp(float(logvar_min), float(logvar_max))
    return (0.5 * (torch.exp(-s) * (hr - sr).pow(2) + s)).mean()


def nll_objective(
    sr: torch.Tensor,
    logvar: torch.Tensor,
    hr: torch.Tensor,
    logvar_min: float,
    logvar_max: float,
    detach_sr: bool,
) -> torch.Tensor:
    """The NLL term as the trainer adds it, with the SR-output gradient controlled.

    Args:
        sr: ``float32``, ``(B, C, H, W)``. Predicted surface reflectance,
            unclipped, still attached to the graph.
        logvar: ``float32``, same shape. Predicted log-variance (reflectance^2).
        hr: ``float32``, same shape. Ground-truth surface reflectance.
        logvar_min: Lower clamp, as :func:`gaussian_nll`.
        logvar_max: Upper clamp.
        detach_sr: True stops the NLL gradient at the SR output, so the term
            trains only what produced ``logvar``. The caller must ALSO call the
            model with ``detach_var=True`` for "only the variance branch" to
            hold -- otherwise the gradient still reaches the trunk through the
            head's input features, and from there the SR output.

    Returns:
        Scalar ``float32`` tensor, the unweighted :func:`gaussian_nll`.
    """
    return gaussian_nll(sr.detach() if detach_sr else sr, logvar, hr, logvar_min, logvar_max)


def sr_grad_ratio(
    sr: torch.Tensor,
    logvar: torch.Tensor,
    hr: torch.Tensor,
    logvar_min: float,
    logvar_max: float,
) -> float:
    """``||dNLL/dsr|| / ||dL1/dsr||`` at the SR output, both terms UNWEIGHTED.

    Measured on detached float32 copies, so it is the same number whichever
    mode trained the model and never touches the training graph. Both
    gradients at the SR output are functions of the values alone
    (``sign(r)/N`` and ``exp(-s) * r / N``), so this is exactly what each term
    would push into the network before the chain rule, not an estimate. The
    trainer multiplies it by the schedule's weight for the ``applied`` column,
    and records 0 for that column when ``--nll-detach-sr`` blocks the path.

    Args:
        sr: ``(B, C, H, W)``, any float dtype. Predicted surface reflectance.
        logvar: Same shape. Predicted log-variance.
        hr: Same shape. Ground-truth surface reflectance.
        logvar_min: Lower clamp, as :func:`gaussian_nll`.
        logvar_max: Upper clamp.

    Returns:
        A non-negative Python float.

    Raises:
        ValueError: ``sr == hr`` everywhere, so the L1 gradient is zero and the
            ratio undefined. Raised rather than reported as inf or 0: either
            would read as a measurement.
    """
    with torch.enable_grad():
        s = sr.detach().float().requires_grad_(True)
        target = hr.detach().float()
        (g_l1,) = torch.autograd.grad(F.l1_loss(s, target), s)
        (g_nll,) = torch.autograd.grad(
            gaussian_nll(s, logvar.detach().float(), target, logvar_min, logvar_max), s
        )
    denom = float(g_l1.norm())
    if denom == 0.0:
        raise ValueError("sr equals hr at every element; the L1 gradient is zero and "
                         "the NLL/L1 gradient ratio is undefined.")
    return float(g_nll.norm()) / denom


def nll_weight_at(it: int, warmup_iters: int, ramp_iters: int, weight: float) -> float:
    """The NLL weight at 0-based iteration ``it``.

    Args:
        it: 0-based training iteration.
        warmup_iters: Iterations of pure L1 (weight exactly 0).
        ramp_iters: Iterations over which the weight rises linearly to
            ``weight``. 0 means a step at ``warmup_iters``.
        weight: The final NLL weight.

    Returns:
        ``0.0`` for ``it < warmup_iters``; ``weight * (it - warmup_iters + 1) /
        ramp_iters`` during the ramp, so the first ramp iteration already has a
        small non-zero weight and the last one reaches ``weight``; ``weight``
        afterwards.

    Raises:
        ValueError: A negative length or weight.
    """
    if warmup_iters < 0 or ramp_iters < 0 or weight < 0:
        raise ValueError(
            f"warmup_iters={warmup_iters}, ramp_iters={ramp_iters} and "
            f"weight={weight} must all be non-negative."
        )
    if it < warmup_iters:
        return 0.0
    if ramp_iters == 0:
        return float(weight)
    return float(weight) * min(1.0, (it - warmup_iters + 1) / float(ramp_iters))
