"""TTA disagreement: exact inversion, batching, and any-checkpoint compatibility.

The fallback uncertainty raster is only as good as the inversion: a transform
inverted with the wrong element misaligns the 8 predictions, and their "std"
becomes a measure of the misalignment -- large on edges, plausible-looking,
and meaningless. So inversion is checked bit for bit.

CPU only, no data.
"""

import pytest
import torch

from src.data.augment import DIHEDRAL, dihedral, invert_dihedral
from src.models.edsr import build_model
from src.uncertainty import sr_output, tta_predict


def _x(b=2, h=16, w=16, seed=0):
    torch.manual_seed(seed)
    return torch.rand(b, 4, h, w)


class Counting:
    """Wraps a callable and counts forward calls and images seen."""

    def __init__(self, fn):
        self.fn, self.calls, self.images = fn, 0, 0

    def __call__(self, x):
        self.calls += 1
        self.images += x.shape[0]
        return self.fn(x)


def _nearest_x4(x):
    """A dihedral-EQUIVARIANT x4 operator: every TTA member must agree exactly."""
    return x.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)


# -- the group ------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2, 4, 16, 16), (2, 4, 7, 5)])
@pytest.mark.parametrize("g", DIHEDRAL)
def test_inversion_is_exact_for_every_element(shape, g):
    torch.manual_seed(0)
    x = torch.rand(*shape)
    assert torch.equal(invert_dihedral(dihedral(x, *g), *g), x)


def test_the_eight_elements_are_distinct():
    x = _x(1)
    images = [dihedral(x, *g) for g in DIHEDRAL]
    for i in range(8):
        for j in range(i + 1, 8):
            assert not torch.equal(images[i], images[j]), (DIHEDRAL[i], DIHEDRAL[j])


# -- the acceptance test the task set -------------------------------------------


@pytest.mark.parametrize("h, w", [(16, 16), (12, 20)])
def test_identity_model_members_are_pixel_identical_to_the_input(h, w):
    """For an identity model the 8 inverted outputs must equal the input exactly."""
    x = _x(3, h, w)
    res = tta_predict(lambda t: t.clone(), x, return_members=True)
    assert res.members.shape == (8, 3, 4, h, w)
    for k in range(8):
        assert torch.equal(res.members[k], x), f"member {k} ({DIHEDRAL[k]}) differs"
    assert float(res.std.abs().max()) <= 1e-7
    assert torch.allclose(res.mean, x, rtol=1e-6, atol=0.0)


def test_an_equivariant_x4_operator_has_zero_disagreement():
    """Inversion must be right at x4, where output and input grids differ."""
    x = _x(2, 8, 8)
    res = tta_predict(_nearest_x4, x, return_members=True)
    for k in range(8):
        assert torch.equal(res.members[k], _nearest_x4(x))
    assert float(res.std.max()) <= 1e-7


# -- batching ---------------------------------------------------------------------


def test_a_square_input_is_one_forward_call():
    model = Counting(_nearest_x4)
    tta_predict(model, _x(2))
    assert (model.calls, model.images) == (1, 16)


def test_a_non_square_input_is_two_forward_calls():
    """Odd quarter-turns swap H and W, so the group splits into two shape groups."""
    model = Counting(_nearest_x4)
    res = tta_predict(model, _x(2, 8, 12))
    assert (model.calls, model.images) == (2, 16)
    assert res.mean.shape == (2, 4, 32, 48)


def test_chunking_changes_the_calls_not_the_answer():
    torch.manual_seed(5)
    net = build_model(scale=2, n_resblocks=1, n_feats=8, in_ch=4, out_ch=4).eval()
    x = _x(2, 8, 8)
    whole = tta_predict(net, x)
    counted = Counting(net)
    chunked = tta_predict(counted, x, chunk_size=3)
    assert counted.calls == 6  # ceil(16 / 3)
    assert torch.allclose(chunked.mean, whole.mean, atol=1e-6)
    assert torch.allclose(chunked.std, whole.std, atol=1e-6)


def test_batched_equals_a_per_transform_loop():
    """The batched path against the obvious reference, on a NON-equivariant model."""
    torch.manual_seed(7)
    net = build_model(scale=2, n_resblocks=1, n_feats=8, in_ch=4, out_ch=4).eval()
    x = _x(2, 8, 10)
    with torch.no_grad():
        ref = torch.stack([invert_dihedral(net(dihedral(x, *g)), *g) for g in DIHEDRAL])
    res = tta_predict(net, x, return_members=True)
    assert torch.allclose(res.members, ref, atol=1e-6)
    assert torch.allclose(res.mean, ref.mean(0), atol=1e-6)
    assert torch.allclose(res.std, ref.std(0, correction=0), atol=1e-6)
    assert float(res.std.mean()) > 0.0  # a real CNN is not dihedral-equivariant


# -- any checkpoint ---------------------------------------------------------------


def test_an_uncertainty_model_is_accepted_and_its_sr_used():
    torch.manual_seed(9)
    headed = build_model(scale=2, n_resblocks=1, n_feats=8, uncertainty=True).eval()
    x = _x(1, 8, 8)
    res = tta_predict(headed, x)
    ref = tta_predict(lambda t: headed(t)[0], x)
    assert torch.equal(res.mean, ref.mean) and torch.equal(res.std, ref.std)


def test_reflectance_is_not_clipped():
    x = _x(1, 8, 8) * 3.0  # bright targets well above 1.0
    res = tta_predict(lambda t: t.clone(), x)
    assert float(res.mean.max()) > 2.0


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        tta_predict(lambda t: t, torch.rand(4, 8, 8))
    with pytest.raises(ValueError):
        tta_predict(lambda t: t, _x(), chunk_size=0)
    with pytest.raises(TypeError):
        sr_output({"sr": torch.zeros(1)})
