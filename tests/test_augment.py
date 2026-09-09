"""Proof that geometric augmentation does not decouple the LR from the HR.

THE FAILURE THIS EXISTS TO CATCH. Augmenting an LR/HR pair is one line of code
and one silent catastrophe: draw the transform twice, once per tensor, and the
HR is no longer the 4x version of the LR. Nothing raises. The shapes still
match, the dtypes still match, the reflectance is still in range, the loss still
decreases -- towards the mean image, because half the batch is asking the model
to hallucinate a rotation. It looks exactly like "the architecture is too small",
which is a diagnosis that costs GPU-hours to disprove.

HOW IT IS CAUGHT. An SR pair satisfies ``area_downsample(hr, scale) == lr``
by construction. Flips and quarter turns commute with block-mean pooling, so
that identity must SURVIVE any correctly applied dihedral transform:

    area_downsample(T(hr), scale) == T(lr)   for every T in the dihedral group

Applying T twice independently breaks it for six of the eight elements. The
tests below assert it for all eight, on a pair drawn through the real patch
pipeline rather than a hand-built array.

Everything runs on the synthetic stub: no network, no cache, no GPU.
"""

import pytest
import torch

from src.data.augment import (
    DIHEDRAL,
    apply_dihedral_pair,
    area_downsample,
    augment_pair,
)
from src.data.loader import PatchDataset
from src.data.registry import get_dataset
from src.utils.config import load_config
from src.utils.logging import get_logger

# The stub derives LR as an exact block mean of HR in float32, so the round trip
# is limited by float32 summation order, not by any real resampling error. 1e-6
# is roughly three orders of magnitude above that noise floor and roughly four
# below the difference an actually-decoupled transform produces (which is on the
# order of the image contrast itself, ~1e-1 in reflectance).
TOL = 1e-6


def _cfg(tmp_path, *overrides):
    """A smoke config -- synthetic stub dataset -- writing to a temp directory."""
    base = [
        f"paths.manifest_dir={tmp_path.as_posix()}",
        f"paths.log_file={(tmp_path / 'run.log').as_posix()}",
        "loader.cached_only=false",
    ]
    return load_config("configs/base.yaml", smoke=True, overrides=base + list(overrides))


@pytest.fixture
def pair(tmp_path):
    """One real LR/HR patch pair, pulled through the Day 1 patch pipeline.

    Grid mode, so the pair is the deterministic one and a failure below is
    about the augmentation rather than about which crop was drawn.

    Returns:
        ``(lr, hr, scale)`` -- ``lr`` float32 ``(C, h, w)`` and ``hr`` float32
        ``(C, h*scale, w*scale)``, both surface reflectance, nominally [0, 1]
        and unclipped.
    """
    cfg = _cfg(tmp_path)
    dataset = get_dataset(cfg)
    patches = PatchDataset(
        cfg, dataset, [0], "grid", "val", logger=get_logger("test.augment",
                                                            log_file=tmp_path / "run.log")
    )
    item = patches[0]
    return item["lr"], item["hr"], int(cfg.sr.scale)


def test_the_unaugmented_pair_is_already_scale_consistent(pair):
    """The premise. If this fails, every test below is measuring the wrong thing."""
    lr, hr, scale = pair
    assert torch.allclose(area_downsample(hr, scale), lr, atol=TOL)


def test_every_dihedral_transform_keeps_lr_and_hr_coupled(pair):
    """The point of the file: all 8 symmetries, HR reduced back onto the LR."""
    lr, hr, scale = pair
    assert len(DIHEDRAL) == 8

    seen = []
    for hflip, vflip, k in DIHEDRAL:
        lr_a, hr_a = apply_dihedral_pair(lr, hr, hflip, vflip, k)

        assert lr_a.shape == lr.shape
        assert hr_a.shape == hr.shape
        assert lr_a.dtype == lr.dtype == torch.float32

        reduced = area_downsample(hr_a, scale)
        assert torch.allclose(reduced, lr_a, atol=TOL), (
            f"hflip={hflip} vflip={vflip} k={k}: the augmented HR does not "
            f"reduce to the augmented LR (max abs diff "
            f"{(reduced - lr_a).abs().max().item():.3e}) -- the two halves of "
            f"the pair were not transformed together."
        )
        seen.append(lr_a)

    # Eight DISTINCT elements, not one transform applied eight times. Without
    # this the test above would pass trivially if every triple mapped to the
    # identity. A patch with no symmetry of its own is what makes this valid,
    # which is why the fixture uses real textured data rather than a constant.
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert not torch.equal(seen[i], seen[j]), (
                f"DIHEDRAL[{i}] and DIHEDRAL[{j}] produced identical output; "
                f"the group is not being enumerated."
            )


def test_independent_draws_would_be_caught(pair):
    """Negative control: the bug this file guards against must actually fail here.

    Transforms LR and HR by DIFFERENT elements -- the exact shape of the
    decoupling bug -- and asserts the scale-consistency check rejects it. A test
    that cannot fail is not evidence, so this proves the assertion above has
    teeth.
    """
    lr, hr, scale = pair
    lr_a, _ = apply_dihedral_pair(lr, hr, False, False, 1)  # rot90
    _, hr_b = apply_dihedral_pair(lr, hr, False, False, 2)  # rot180
    assert not torch.allclose(area_downsample(hr_b, scale), lr_a, atol=TOL)


def test_augment_pair_draws_from_the_group_and_stays_coupled(pair):
    """The random entry point, over a seeded stream: every draw stays coupled."""
    lr, hr, scale = pair
    gen = torch.Generator().manual_seed(1337)

    drawn = set()
    for _ in range(200):
        lr_a, hr_a = augment_pair(lr, hr, generator=gen)
        assert torch.allclose(area_downsample(hr_a, scale), lr_a, atol=TOL)
        drawn.add(tuple(lr_a.reshape(-1)[:8].tolist()))

    # 200 draws over a uniform 8-element group: seeing fewer than 8 distinct
    # outcomes has probability well below 1e-9, so this is a fixed-seed
    # assertion, not a flaky one.
    assert len(drawn) == 8, f"expected all 8 symmetries, saw {len(drawn)}"


def test_augment_pair_leaves_reflectance_values_untouched(pair):
    """Geometric only. The multiset of pixel values must be identical.

    Reflectance is a physical quantity and the spectral-consistency
    contribution is defined over it; an augmentation that rescaled or clipped
    values -- including clipping a legitimate above-1.0 bright target -- would
    invalidate that. Sorting the flattened tensor detects any change in value
    while ignoring the change in position, which is the whole permitted effect.
    """
    lr, hr, _ = pair
    gen = torch.Generator().manual_seed(7)
    for _ in range(8):
        lr_a, hr_a = augment_pair(lr, hr, generator=gen)
        assert torch.equal(torch.sort(lr_a.reshape(-1)).values,
                           torch.sort(lr.reshape(-1)).values)
        assert torch.equal(torch.sort(hr_a.reshape(-1)).values,
                           torch.sort(hr.reshape(-1)).values)


def test_a_mismatched_pair_is_an_error_not_a_silent_broadcast(pair):
    """Fail loudly: a pair whose halves disagree on bands is not a pair."""
    lr, hr, _ = pair
    with pytest.raises(ValueError, match="same channel count"):
        apply_dihedral_pair(lr[:1], hr, False, False, 0)


def test_an_out_of_range_rotation_is_an_error(pair):
    lr, hr, _ = pair
    with pytest.raises(ValueError, match="k must be one of"):
        apply_dihedral_pair(lr, hr, False, False, 4)


def test_area_downsample_rejects_an_indivisible_shape(pair):
    _, hr, _ = pair
    with pytest.raises(ValueError, match="divisible by scale"):
        area_downsample(hr[:, :-1, :], 4)


# -- the dataset wiring ----------------------------------------------------


def test_the_val_split_cannot_be_augmented():
    """augment=True must be FORCED off for val, not merely defaulted off.

    Checked on the class's own resolution rule rather than by constructing the
    dataset, which needs a cache directory this test deliberately does not
    have. The rule is one expression and it is the one that matters: an
    augmented val set silently moves the reference the Day 1 bicubic baseline
    and Run A were both measured against.
    """
    from src.data.adapter import _AUGMENTED_SPLITS

    assert "train" in _AUGMENTED_SPLITS
    assert "val" not in _AUGMENTED_SPLITS
    assert "test" not in _AUGMENTED_SPLITS
