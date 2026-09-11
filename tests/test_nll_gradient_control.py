"""Day 4 gradient control: the NLL trains the variance branch, and the ratio is measured.

``--nll-detach-sr`` (default on) must mean what it says -- no NLL gradient
reaches any parameter outside ``var_head`` -- and the logged NLL/L1 gradient
ratio must be the exact ratio at the SR output, not an estimate.

CPU only, no data.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.losses import gaussian_nll, nll_objective, sr_grad_ratio
from src.models.edsr import VAR_HEAD_PREFIX, build_model

SMALL = dict(scale=4, n_resblocks=2, n_feats=16, in_ch=4, out_ch=4)


def _model_and_batch(seed=0):
    torch.manual_seed(seed)
    model = build_model(**SMALL, uncertainty=True)
    # Zero-initialised, the head's last conv would pass no gradient to the rest
    # of the head, and "var_head got a gradient" would be vacuous.
    torch.nn.init.normal_(model.var_head[2].weight, std=0.1)
    x = torch.rand(2, 4, 8, 8) * 0.4
    hr = torch.rand(2, 4, 32, 32) * 0.4
    return model, x, hr


def test_detached_nll_reaches_only_the_variance_branch():
    model, x, hr = _model_and_batch()
    sr, logvar = model(x, detach_var=True)
    nll_objective(sr, logvar, hr, -10.0, 10.0, detach_sr=True).backward()

    trunk = {n: p.grad for n, p in model.named_parameters() if not n.startswith(VAR_HEAD_PREFIX)}
    head = {n: p.grad for n, p in model.named_parameters() if n.startswith(VAR_HEAD_PREFIX)}
    leaked = [n for n, g in trunk.items() if g is not None and g.abs().sum() > 0]
    assert not leaked, f"NLL gradient reached the reconstruction path: {leaked}"
    assert all(g is not None and g.abs().sum() > 0 for g in head.values())


def test_the_joint_path_does_train_the_reconstruction():
    """The control for the test above: without the detach, the trunk moves."""
    model, x, hr = _model_and_batch()
    sr, logvar = model(x, detach_var=False)
    nll_objective(sr, logvar, hr, -10.0, 10.0, detach_sr=False).backward()
    assert model.head.weight.grad is not None
    assert model.head.weight.grad.abs().sum() > 0


def test_detaching_sr_alone_is_not_enough():
    """Why the trainer ALSO passes detach_var: the head's input is the trunk."""
    model, x, hr = _model_and_batch()
    sr, logvar = model(x, detach_var=False)
    nll_objective(sr, logvar, hr, -10.0, 10.0, detach_sr=True).backward()
    assert model.head.weight.grad is not None and model.head.weight.grad.abs().sum() > 0


def test_the_detached_value_is_the_same_nll():
    """Detaching changes where the gradient goes, never the number."""
    model, x, hr = _model_and_batch()
    sr, logvar = model(x)
    a = nll_objective(sr, logvar, hr, -10.0, 10.0, detach_sr=True)
    b = gaussian_nll(sr, logvar, hr, -10.0, 10.0)
    assert float(a) == float(b)


def test_the_ratio_is_the_analytic_one():
    """dL1/dsr = sign(r)/N, dNLL/dsr = exp(-s) r / N -> ratio ||exp(-s) r|| / sqrt(N)."""
    torch.manual_seed(1)
    hr = torch.rand(2, 4, 16, 16) * 0.3
    sr = hr + 0.01 * torch.randn_like(hr)
    logvar = -9.0 + 0.3 * torch.randn_like(hr)
    s = logvar.clamp(-10.0, 10.0)  # the loss clamps; the reference must too
    expected = float((torch.exp(-s) * (sr - hr)).norm()) / math.sqrt(sr.numel())
    assert sr_grad_ratio(sr, logvar, hr, -10.0, 10.0) == pytest.approx(expected, rel=1e-4)


def test_the_ratio_at_day3_scale_is_near_ninety():
    """The number behind the 0.1 default, reproduced: s = -9, |r| = 0.01."""
    hr = torch.zeros(1, 4, 16, 16)
    sr = hr + 0.01
    logvar = torch.full_like(hr, -9.0)
    assert sr_grad_ratio(sr, logvar, hr, -10.0, 10.0) == pytest.approx(math.exp(9) * 0.01, rel=1e-4)


def test_the_ratio_does_not_touch_the_training_graph():
    model, x, hr = _model_and_batch()
    sr, logvar = model(x)
    sr_grad_ratio(sr, logvar, hr, -10.0, 10.0)
    assert all(p.grad is None for p in model.parameters())
    F.l1_loss(sr, hr).backward()  # the graph is still intact and usable


def test_an_exact_reconstruction_has_no_ratio():
    t = torch.rand(1, 4, 8, 8)
    with pytest.raises(ValueError, match="undefined"):
        sr_grad_ratio(t, torch.zeros_like(t), t, -10.0, 10.0)
