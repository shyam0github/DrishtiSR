"""The log-variance head: off means Day 3, on means bounded, and Day 3 weights load.

The contract being protected:

- ``uncertainty=False`` (the default) builds the Day 3 architecture exactly --
  parameter names, count, and return type -- so A2/B1/B2 and every eval path
  that loads them are untouched.
- ``uncertainty=True`` adds a head that emits LOG-variance at the SR output's
  size, clamped, inside the parameter budget.
- A Day 3 checkpoint loads into the head-enabled model through
  ``load_backbone_state`` -- ``strict=False`` restricted to the head's own
  keys -- and the SR output is then bit-identical to the model it trained as.

CPU only, no data. The one test that reads real checkpoints skips, visibly,
when runs/day3 has not been fetched.
"""

import inspect

import pytest
import torch
from omegaconf import OmegaConf

from src.models.edsr import (
    EDSR,
    VAR_HEAD_PREFIX,
    build_model,
    count_params,
    load_backbone_state,
    model_from_checkpoint,
)
from src.train import build_parser
from src.utils.paths import repo_root

BASE = OmegaConf.load(repo_root() / "configs" / "base.yaml")
DAY3 = dict(scale=4, n_resblocks=16, n_feats=48, in_ch=4, out_ch=4)
SMALL = dict(scale=4, n_resblocks=2, n_feats=16, in_ch=4, out_ch=4)
HEAD_KEYS = {"var_head.0.weight", "var_head.0.bias", "var_head.2.weight", "var_head.2.bias"}


def _x(h=12, w=12, seed=0):
    torch.manual_seed(seed)
    return torch.rand(2, 4, h, w) * 0.4


# -- flag off: the Day 3 architecture, unchanged ------------------------------


def test_flag_off_is_the_day3_architecture():
    plain = build_model(**DAY3)
    assert count_params(plain) == 855_652
    assert not hasattr(plain, "var_head")
    out = plain(_x())
    assert torch.is_tensor(out) and out.shape == (2, 4, 48, 48)


def test_the_head_adds_only_var_head_keys():
    plain = set(build_model(**DAY3).state_dict())
    headed = set(build_model(**DAY3, uncertainty=True).state_dict())
    assert headed - plain == HEAD_KEYS
    assert plain <= headed


# -- flag on: shape, bounds, budget --------------------------------------------


def test_head_output_matches_sr_shape_and_fits_the_budget():
    model = build_model(**DAY3, uncertainty=True)
    assert count_params(model) == 863_160
    assert count_params(model) <= int(BASE.runtime.max_parameters)
    sr, logvar = model(_x(16, 20))
    assert sr.shape == logvar.shape == (2, 4, 64, 80)


def test_an_untrained_head_says_logvar_init_everywhere():
    model = build_model(**SMALL, uncertainty=True, logvar_init=-9.0)
    _, logvar = model(_x())
    assert torch.equal(logvar, torch.full_like(logvar, -9.0))


@pytest.mark.parametrize("bias, expected", [(50.0, 10.0), (-50.0, -10.0)])
def test_logvar_is_clamped_in_forward(bias, expected):
    """The bound must hold in the exported graph, not only inside the loss."""
    model = build_model(**SMALL, uncertainty=True)
    torch.nn.init.constant_(model.var_head[2].bias, bias)
    _, logvar = model(_x() * 1e3)
    assert torch.equal(logvar, torch.full_like(logvar, expected))


@pytest.mark.parametrize("kw", [dict(logvar_min=1.0, logvar_max=1.0),
                                dict(logvar_init=-11.0)])
def test_an_impossible_clamp_is_refused(kw):
    with pytest.raises(ValueError):
        build_model(**SMALL, uncertainty=True, **kw)


def test_detach_var_cuts_the_head_off_from_the_trunk():
    """The NLL warmup relies on this: no gradient path from the head to the trunk."""
    torch.manual_seed(3)
    model = build_model(**SMALL, uncertainty=True)
    # Zero-initialised, the final conv would pass no gradient either way.
    torch.nn.init.normal_(model.var_head[2].weight, std=0.1)
    x = _x()

    _, logvar = model(x, detach_var=True)
    logvar.sum().backward()
    assert model.head.weight.grad is None
    assert model.var_head[0].weight.grad is not None

    model.zero_grad(set_to_none=True)
    _, logvar = model(x, detach_var=False)
    logvar.sum().backward()
    assert model.head.weight.grad is not None
    assert model.head.weight.grad.abs().sum() > 0


# -- the strict=False path -------------------------------------------------------


def test_the_sr_output_is_untouched_by_the_head():
    torch.manual_seed(1)
    plain = build_model(**SMALL).eval()
    headed = build_model(**SMALL, uncertainty=True).eval()
    fresh = load_backbone_state(headed, plain.state_dict())
    assert set(fresh) == HEAD_KEYS
    x = _x()
    with torch.no_grad():
        assert torch.equal(headed(x)[0], plain(x))


def _day3_payload(model):
    """A checkpoint in exactly the shape src/train.py save() wrote on Day 3."""
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    return {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": {},
        "it": 11999,
        "best": 35.21,
        "args": {"scale": 4, "n_resblocks": 16, "n_feats": 48, "in_ch": 4,
                 "spectral_lambda1": 0.1, "spectral_lambda2": 0.02,
                 "spectral_downsample": "area", "seed": 1337},
        "config_hash": "e6082c69bba3086aaf71d0a413e64d621d860e5506da507ce2d76edb92f1fc0a",
        "git_sha": "fb8659d5e8b2d858b0d16c3f88555d9e5f1cb2ac",
        "git_dirty": False,
        "lambdas": {"spectral_lambda1": 0.1, "spectral_lambda2": 0.02},
        "spectral_downsample": "area",
        "val_metrics": None,
        "val_metrics_iter": None,
        "sharpness_reference": None,
    }


def test_a_day3_checkpoint_loads_into_the_uncertainty_model(tmp_path):
    """THE test the task asked for: Day 3 weights, head-enabled model, no error."""
    torch.manual_seed(2)
    trained = build_model(**DAY3).eval()
    path = tmp_path / "best.pt"
    torch.save(_day3_payload(trained), path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    headed = build_model(**DAY3, uncertainty=True).eval()
    fresh = load_backbone_state(headed, payload["model"])
    assert set(fresh) == HEAD_KEYS

    x = _x(8, 8)
    with torch.no_grad():
        sr, logvar = headed(x)
        assert torch.equal(sr, trained(x))
    assert torch.equal(logvar, torch.full_like(logvar, float(BASE.uncertainty.logvar_init)))

    # And the Day 3 file still rebuilds as what it was: no head, strict load.
    rebuilt, _ = model_from_checkpoint(path)
    assert not hasattr(rebuilt, "var_head")
    with torch.no_grad():
        assert torch.equal(rebuilt(x), trained(x))


@pytest.mark.parametrize("run", ["a2", "b1", "b2"])
def test_the_real_day3_checkpoints_load_into_the_uncertainty_model(run):
    path = repo_root() / "runs" / "day3" / run / "best.pt"
    if not path.is_file():
        pytest.skip(f"{path} not fetched; the fabricated-checkpoint test above still runs.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args = payload["args"]
    arch = dict(scale=int(args["scale"]), n_resblocks=int(args["n_resblocks"]),
                n_feats=int(args["n_feats"]), in_ch=int(args["in_ch"]), out_ch=int(args["in_ch"]))
    headed = build_model(**arch, uncertainty=True).eval()
    assert set(load_backbone_state(headed, payload["model"])) == HEAD_KEYS
    plain, _ = model_from_checkpoint(path)
    x = _x(8, 8)
    with torch.no_grad():
        assert torch.equal(headed(x)[0], plain(x))


def test_a_missing_backbone_key_is_refused():
    state = build_model(**SMALL).state_dict()
    del state["body.0.body.0.weight"]
    with pytest.raises(RuntimeError, match="missing"):
        load_backbone_state(build_model(**SMALL, uncertainty=True), state)


def test_an_unexpected_key_is_refused():
    state = build_model(**SMALL).state_dict()
    state["stray.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected"):
        load_backbone_state(build_model(**SMALL, uncertainty=True), state)


def test_a_width_mismatch_is_refused():
    """Run A (64 features) into a 48-feature model must fail, strict or not."""
    state = build_model(**dict(SMALL, n_feats=24)).state_dict()
    with pytest.raises(RuntimeError):
        load_backbone_state(build_model(**SMALL, uncertainty=True), state)


def test_an_uncertainty_checkpoint_rebuilds_with_its_head(tmp_path):
    model = build_model(**SMALL, uncertainty=True, var_feats=8, logvar_init=-7.0)
    args = dict(scale=4, n_resblocks=2, n_feats=16, in_ch=4, uncertainty=1, var_feats=8,
                logvar_min=-10.0, logvar_max=10.0, logvar_init=-7.0)
    torch.save({"model": model.state_dict(), "it": 0, "args": args}, tmp_path / "c.pt")
    rebuilt, _ = model_from_checkpoint(tmp_path / "c.pt")
    sr, logvar = rebuilt(_x())
    assert torch.equal(logvar, torch.full_like(logvar, -7.0))


# -- the defaults: off, and agreeing with base.yaml ------------------------------


def test_the_cli_defaults_leave_the_head_off():
    defaults = {a.dest: a.default for a in build_parser()._actions}
    assert defaults["uncertainty"] == 0
    assert defaults["init_from"] == "none"


def test_the_cli_and_model_defaults_mirror_base_yaml():
    u = BASE.uncertainty
    defaults = {a.dest: a.default for a in build_parser()._actions}
    assert defaults["uncertainty"] == int(bool(u.enabled))
    assert defaults["var_feats"] == int(u.var_feats)
    assert defaults["logvar_min"] == float(u.logvar_min)
    assert defaults["logvar_max"] == float(u.logvar_max)
    assert defaults["logvar_init"] == float(u.logvar_init)
    assert defaults["nll_weight"] == float(u.nll.weight)
    assert defaults["nll_warmup"] == int(u.nll.warmup_iters)
    assert defaults["nll_ramp"] == int(u.nll.ramp_iters)

    sig = inspect.signature(EDSR.__init__).parameters
    assert sig["uncertainty"].default is False
    assert sig["var_feats"].default == int(u.var_feats)
    assert sig["logvar_min"].default == float(u.logvar_min)
    assert sig["logvar_max"].default == float(u.logvar_max)
    assert sig["logvar_init"].default == float(u.logvar_init)


def test_var_head_prefix_names_the_head():
    assert all(k.startswith(VAR_HEAD_PREFIX) for k in HEAD_KEYS)
