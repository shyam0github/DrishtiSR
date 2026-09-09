"""The freeze, checked: the hash matches, and the config describes a real model.

A frozen config is a promise that a hash in a run's metadata identifies exactly
one set of hyperparameters. Two things can break that promise silently:

1. Someone edits ``configs/frozen_day3.yaml`` and does not re-freeze it. Then
   the hash cited by an old run no longer describes the file, and the hash cited
   by a new run is wrong from the start. ``load_frozen`` raises on this, and
   these tests confirm it raises rather than warns.
2. The config declares an architecture nobody instantiated. ``expected_parameters``
   is the number the whole 48-feature decision rests on -- AGENTS.md requires it
   be asserted in code, not counted mentally -- so it is built and counted here.

CPU only, no data, no network.
"""

import pytest
from omegaconf import OmegaConf

from src.config import (
    FROZEN_CONFIG,
    HASH_FIELD,
    FrozenConfigError,
    canonical_bytes,
    compute_config_hash,
    load_frozen,
)
from src.models.edsr import build_model, count_params


def test_the_frozen_config_matches_its_recorded_hash():
    """The freeze itself. A failure here means the file was edited unfrozen."""
    cfg = load_frozen(FROZEN_CONFIG)
    assert compute_config_hash(cfg) == str(cfg[HASH_FIELD])


def test_a_changed_value_invalidates_the_hash(tmp_path):
    """The hash must be sensitive to VALUES -- otherwise it certifies nothing."""
    cfg = load_frozen(FROZEN_CONFIG)
    tampered = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    tampered.optim.weight_decay = 1e-4

    assert compute_config_hash(tampered) != str(cfg[HASH_FIELD])

    path = tmp_path / "tampered.yaml"
    path.write_text(OmegaConf.to_yaml(tampered), encoding="utf-8")
    with pytest.raises(FrozenConfigError, match="does not match its recorded hash"):
        load_frozen(path)


def test_the_hash_ignores_comments_and_key_order(tmp_path):
    """...and insensitive to formatting, or every reflowed comment is a false alarm."""
    cfg = load_frozen(FROZEN_CONFIG)
    container = OmegaConf.to_container(cfg, resolve=True)

    reordered = OmegaConf.create({k: container[k] for k in sorted(container, reverse=True)})
    path = tmp_path / "reordered.yaml"
    path.write_text(
        "# a comment that did not exist before\n" + OmegaConf.to_yaml(reordered),
        encoding="utf-8",
    )
    # Same values, different bytes on disk -- and it still loads.
    assert compute_config_hash(reordered) == compute_config_hash(cfg)
    assert str(load_frozen(path)[HASH_FIELD]) == str(cfg[HASH_FIELD])


def test_the_hash_field_is_excluded_from_its_own_hash():
    cfg = load_frozen(FROZEN_CONFIG)
    assert HASH_FIELD.encode() not in canonical_bytes(cfg)


def test_an_unfrozen_config_is_an_error_not_a_default(tmp_path):
    """Fail loudly: a missing hash must not be treated as "not frozen, carry on"."""
    path = tmp_path / "unfrozen.yaml"
    path.write_text("seed: 1337\n", encoding="utf-8")
    with pytest.raises(FrozenConfigError, match="not frozen"):
        load_frozen(path)


def test_a_malformed_hash_is_an_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(f'seed: 1337\n{HASH_FIELD}: "not-a-digest"\n', encoding="utf-8")
    with pytest.raises(FrozenConfigError, match="not a 64-character"):
        load_frozen(path)


def test_a_missing_frozen_config_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frozen(tmp_path / "nope.yaml")


# -- the config describes a model that actually exists ---------------------


def test_the_frozen_downsampler_is_one_the_loss_will_accept():
    """A frozen operator the loss rejects would fail at the first validation.

    ``src/losses/spectral.py`` refuses decimating downsamplers by name, so a
    freeze naming one would take a queue wait and a session start to discover.
    """
    from src.losses import ANTIALIASED_MODES

    cfg = load_frozen(FROZEN_CONFIG)
    assert str(cfg.loss.spectral_downsample) in ANTIALIASED_MODES


def test_the_freeze_and_base_yaml_agree_on_the_downsampler():
    """Two files name D. If they disagree, the floor and the runs are not comparable.

    ``configs/base.yaml`` drives ``scripts/spectral_floor.py``; the freeze drives
    the trainer. The floor is the number a trained run's spectral term is
    judged against, and that judgement is meaningless if the two were measured
    through different operators.
    """
    from src.utils.paths import repo_root

    frozen = load_frozen(FROZEN_CONFIG)
    base = OmegaConf.load(repo_root() / "configs" / "base.yaml")
    assert str(frozen.loss.spectral_downsample) == str(base.loss.spectral.downsample)


def test_the_frozen_architecture_has_the_declared_parameter_count():
    """855,652 -- built and counted, never assumed. AGENTS.md section 1, model budget."""
    cfg = load_frozen(FROZEN_CONFIG)
    model = build_model(
        cfg.model.name,
        scale=int(cfg.model.scale),
        n_resblocks=int(cfg.model.n_resblocks),
        n_feats=int(cfg.model.n_feats),
        in_ch=int(cfg.model.in_ch),
        out_ch=int(cfg.model.out_ch),
        res_scale=float(cfg.model.res_scale),
    )
    actual = count_params(model)
    assert actual == int(cfg.model.expected_parameters), (
        f"the frozen architecture builds to {actual:,} parameters but the "
        f"config declares {int(cfg.model.expected_parameters):,}."
    )


def test_the_frozen_architecture_is_within_the_parameter_budget():
    cfg = load_frozen(FROZEN_CONFIG)
    limit = int(cfg.runtime.max_parameters)
    declared = int(cfg.model.expected_parameters)
    assert declared <= limit, f"{declared:,} parameters against a {limit:,} budget."
    # The budget in the frozen config must agree with the project-wide one, or
    # a run can pass its own check and still violate the deliverable.
    base = OmegaConf.load(__import__("src.utils.paths", fromlist=["repo_root"]).repo_root()
                          / "configs" / "base.yaml")
    assert limit == int(base.runtime.max_parameters)


def test_the_frozen_config_agrees_with_the_cli_defaults():
    """The frozen file and src/train.py's argparse defaults must not drift apart.

    A default that disagrees with the freeze is worse than no default: a run
    launched without the flag trains something the hash does not describe.
    """
    from src.train import build_parser

    cfg = load_frozen(FROZEN_CONFIG)
    # The real parser, not a transcription of it.
    defaults = {a.dest: a.default for a in build_parser()._actions}

    assert defaults["wd"] == float(cfg.optim.weight_decay) == 0.0
    assert defaults["n_feats"] == int(cfg.model.n_feats)
    assert defaults["n_resblocks"] == int(cfg.model.n_resblocks)
    assert defaults["seed"] == int(cfg.seed)
    assert defaults["lr"] == float(cfg.schedule.lr)
    assert defaults["min_lr"] == float(cfg.schedule.min_lr)
    assert defaults["warmup"] == int(cfg.schedule.warmup_iters)
    assert defaults["iters"] == int(cfg.schedule.iters)
    assert defaults["batch"] == int(cfg.data.batch_size)
    assert defaults["patch_lr"] == int(cfg.data.patch_lr)
    assert defaults["scale"] == int(cfg.data.scale)
    assert defaults["in_ch"] == int(cfg.model.in_ch)
    assert defaults["clip"] == float(cfg.optim.grad_clip)
    assert defaults["workers"] == int(cfg.data.num_workers)
    assert defaults["data_module"] is None and defaults["data_class"] is None, (
        "--data-module/--data-class are required flags with no default; the "
        "frozen config supplies them."
    )

    # BOTH ZERO. The run this file freezes is the CONTROL for the spectral run,
    # and the whole comparison rests on the control not having acquired a
    # spectral term from a default. Asserted against the freeze rather than
    # against the literal 0.0 alone, so the two cannot drift apart in either
    # direction.
    assert defaults["spectral_lambda1"] == float(cfg.loss.spectral_lambda1) == 0.0
    assert defaults["spectral_lambda2"] == float(cfg.loss.spectral_lambda2) == 0.0
    # Frozen even though the lambdas are 0: the control MEASURES both terms at
    # every validation, and a control that measured with a different
    # downsampler from the spectral run would not be logging the same quantity.
    assert defaults["spectral_downsample"] == str(cfg.loss.spectral_downsample)

    assert defaults["amp"] == int(bool(cfg.runtime.amp))
    assert defaults["max_hours"] == float(cfg.runtime.max_hours)
    assert defaults["val_every"] == int(cfg.runtime.val_every)
    assert defaults["val_batches"] == int(cfg.runtime.val_batches)
    assert defaults["ckpt_every"] == int(cfg.runtime.ckpt_every)
    assert defaults["log_every"] == int(cfg.runtime.log_every)
    assert defaults["resume"] == str(cfg.runtime.resume)
