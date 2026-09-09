"""The spectral-consistency loss, and the two ways it can be silently vacuous.

This file exists because the loss in ``src/losses/spectral.py`` is the project's
first technical contribution, and both of the ways it fails fail *quietly* --
they produce excellent-looking numbers rather than errors:

1. **A downsampler that decimates instead of averaging.** With stride-4 slicing
   the loss constrains one HR pixel in sixteen. The other fifteen can be
   anything at all and the reported spectral term is unaffected.
   :func:`test_stride4_slicing_fails_the_antialias_test` runs the *same*
   checkerboard through stride-4 slicing and asserts it survives intact, so the
   distinction between the two operators is documented here as a property, not
   as a comment.
2. **SAM computed after a shifting normalisation.** The spectral angle is
   invariant to positive scaling but not to translation, so a per-band mean
   subtraction turns it into a number with no physical meaning that is still
   called SAM in the logs. :func:`test_sam_is_invariant_to_scale_but_not_shift`
   pins both halves of that.

Plus the numerics that decide whether a run survives contact with real data:
gradients must flow to ``sr``, ``lr`` must be detached, and a dead pixel must
produce finite loss *and* finite gradients -- masking a NaN after the fact does
not undo it.

CPU only, no data, no network.
"""

import math

import pytest
import torch

from src.losses import (
    ANTIALIASED_MODES,
    DEFAULT_COS_CLAMP,
    down4,
    spectral_consistency,
    spectral_settings_from_cfg,
    spectral_terms,
)

SCALE = 4
BANDS = 4
# Checkerboard extremes, in reflectance units. SWING is the peak-to-peak
# structure the downsampler is asked to destroy, and every "did it flatten"
# threshold below is expressed as a fraction of it rather than as a bare number.
LOW = 0.05
HIGH = 0.45
SWING = HIGH - LOW


def checkerboard(size: int = 32, bands: int = BANDS, low: float = LOW,
                 high: float = HIGH) -> torch.Tensor:
    """A 1-pixel-period checkerboard: the highest frequency the grid can carry.

    Args:
        size: Spatial edge in pixels. Must be even.
        bands: Number of bands, all carrying the same pattern.
        low: Reflectance of the dark squares (float32, reflectance units).
        high: Reflectance of the bright squares.

    Returns:
        ``(1, bands, size, size)`` float32 surface reflectance alternating
        between ``low`` and ``high`` every pixel. Its 4x4 block mean is exactly
        ``(low + high) / 2`` everywhere, so a correct antialiased x4 downsample
        of it is uniform -- while any stride-4 pick returns one phase of the
        pattern at full contrast.
    """
    rows = torch.arange(size).reshape(-1, 1)
    cols = torch.arange(size).reshape(1, -1)
    pattern = torch.where(((rows + cols) % 2) == 0, high, low).to(torch.float32)
    return pattern.expand(bands, size, size).unsqueeze(0).clone()


# -- the downsampler must average, not decimate ----------------------------


@pytest.mark.parametrize("mode", ANTIALIASED_MODES)
def test_down4_flattens_a_checkerboard(mode):
    """Aliasing would preserve the structure; a real low-pass destroys it."""
    board = checkerboard()
    out = down4(board, scale=SCALE, mode=mode)

    assert out.shape == (1, BANDS, 8, 8)
    # The input swings 0.40 reflectance peak to peak. MEASURED residual swing
    # after down4: area 0.000000 (it is an exact block mean), bicubic_antialias
    # 0.002420, bilinear_antialias 0.002041 -- i.e. at most 0.6% of the input
    # pattern survives. The bound is set at 1%, two orders of magnitude below
    # the 100% that stride-4 slicing leaves in
    # test_stride4_slicing_fails_the_antialias_test.
    swing = float(out.max() - out.min())
    assert swing < 0.01 * SWING, (
        f"{mode} left {swing:.4f} of a {SWING:.2f} reflectance checkerboard "
        "standing; it is not antialiasing."
    )
    assert float(out.mean()) == pytest.approx(0.25, abs=1e-4)


def test_stride4_slicing_fails_the_antialias_test():
    """The counter-example, asserted rather than described.

    Stride-4 slicing is what a plain ``sr[..., ::4, ::4]`` gives, and it is the
    implementation the loss must never use. On a 1-pixel-period checkerboard
    every 4th pixel has the same parity, so the "downsampled" image is a
    constant at ONE phase of the pattern -- full contrast against the other
    phase, and identical to what a network that got only those pixels right
    would produce. Both facts are asserted: the slice does not flatten, and the
    two phases disagree by the full input swing.
    """
    board = checkerboard()

    # (0, 0) and (0, 1) are the two parities. Note that (1, 1) is NOT a second
    # phase: shifting both axes by one on a checkerboard returns the parity you
    # started from, and a stride of 4 preserves parity on every axis -- which is
    # itself a demonstration that the slice carries the full pattern through.
    bright_phase = board[..., ::SCALE, ::SCALE]
    dark_phase = board[..., ::SCALE, 1::SCALE]
    area = down4(board, scale=SCALE, mode="area")

    # It does not flatten: each phase comes through at its own extreme.
    assert float(bright_phase.mean()) == pytest.approx(HIGH, abs=1e-5)
    assert float(dark_phase.mean()) == pytest.approx(LOW, abs=1e-5)
    # The exact assertion the antialiased test above passes, inverted: stride-4
    # slicing leaves 100% of the checkerboard standing where down4 leaves <1%.
    surviving = float(bright_phase.mean() - dark_phase.mean())
    assert surviving > 0.99 * SWING, (
        "stride-4 slicing was expected to preserve the pattern in full"
    )

    # And the consequence for the loss: the antialiased operator gives the same
    # answer for both phases (it looked at every pixel), while the slice gives
    # two completely different answers depending on which quarter it happened to
    # sample. That is what "constrains one pixel in sixteen" means numerically.
    assert float(area.mean()) == pytest.approx(0.25, abs=1e-4)


@pytest.mark.parametrize("mode", ["nearest", "bicubic", "bilinear", "subsample"])
def test_down4_rejects_aliasing_modes_by_name(mode):
    with pytest.raises(ValueError, match="vacuous"):
        down4(checkerboard(), scale=SCALE, mode=mode)


def test_down4_refuses_a_size_it_cannot_divide():
    with pytest.raises(ValueError, match="not divisible"):
        down4(torch.rand(1, BANDS, 30, 30), scale=SCALE)


def test_down4_area_is_an_exact_block_mean():
    """'area' at integer scale must be the block mean the physical claim uses."""
    x = torch.rand(2, BANDS, 16, 16)
    out = down4(x, scale=SCALE, mode="area")
    manual = x.reshape(2, BANDS, 4, SCALE, 4, SCALE).mean(dim=(3, 5))
    assert torch.allclose(out, manual, atol=1e-6)


# -- the loss itself -------------------------------------------------------


def test_a_perfect_downsample_scores_at_the_numerical_floor():
    """If sr degrades exactly to lr, only the arccos clamp is left."""
    torch.manual_seed(0)
    sr = torch.rand(2, BANDS, 32, 32)
    lr = down4(sr, scale=SCALE, mode="area")

    parts = spectral_terms(sr, lr)
    assert float(parts["l1_spec"]) == pytest.approx(0.0, abs=1e-7)
    # NOT zero, and deliberately so: the cosine is clamped off 1.0 to keep
    # arccos differentiable. This is the floor documented in the module, and it
    # is the reason a trained run's SAM term can never read 0.
    #
    # The analytic value is acos(1 - 1e-7) = 4.472e-4 rad; float32 delivers
    # 4.883e-4, because 1 - 1e-7 is not representable (float32 eps is 1.19e-7)
    # and the clamp lands on the neighbouring float. MEASURED, and asserted
    # loosely on purpose -- the number that matters is that the floor is ~5e-4
    # rad, i.e. ~1% of the ~0.036 rad physical floor, not which of two adjacent
    # floats it rounds to.
    analytic = math.acos(1.0 - DEFAULT_COS_CLAMP)
    assert 0.0 < float(parts["sam"]) < 2.0 * analytic
    assert float(parts["sam"]) == pytest.approx(analytic, rel=0.15)
    assert float(parts["sam_valid_frac"]) == 1.0


def test_total_is_the_weighted_sum_of_the_reported_components():
    torch.manual_seed(1)
    sr = torch.rand(2, BANDS, 32, 32)
    lr = torch.rand(2, BANDS, 8, 8)

    total, parts = spectral_consistency(sr, lr, lam1=0.5, lam2=0.1)
    assert float(total) == pytest.approx(
        0.5 * parts["l1_spec"] + 0.1 * parts["sam"], rel=1e-6
    )
    assert set(parts) == {"l1_spec", "sam", "sam_valid_frac"}
    assert all(isinstance(v, float) for v in parts.values())


def test_the_reported_components_are_unweighted():
    """Two lambdas, one pair of tensors, identical logged components.

    This is what makes the control run's log comparable with the spectral run's:
    both log the term itself, not the term times a weight the other run did not
    use. With lambdas at 0.0 a weighted component would log 0.0 and prove
    nothing.
    """
    torch.manual_seed(2)
    sr = torch.rand(2, BANDS, 32, 32)
    lr = torch.rand(2, BANDS, 8, 8)

    _, off = spectral_consistency(sr, lr, lam1=0.0, lam2=0.0)
    _, on = spectral_consistency(sr, lr, lam1=0.5, lam2=0.1)
    assert off == on


def test_lambdas_at_zero_give_a_zero_total():
    sr = torch.rand(1, BANDS, 32, 32)
    lr = torch.rand(1, BANDS, 8, 8)
    total, _ = spectral_consistency(sr, lr, lam1=0.0, lam2=0.0)
    assert float(total) == 0.0


# -- SAM is a physical quantity, and only in reflectance space -------------


def test_sam_is_invariant_to_scale_but_not_shift():
    """The reason SAM must not be computed on mean-subtracted tensors.

    Scaling both spectra by a positive constant leaves the angle between them
    untouched -- that is the property that makes SAM blind to brightness error
    and therefore worth having. Adding a constant does NOT, which is exactly
    what a standardising normalisation does per band. A SAM computed after that
    shift is a different number wearing the same name.
    """
    torch.manual_seed(3)
    sr = torch.rand(1, BANDS, 32, 32) * 0.4 + 0.05
    lr = torch.rand(1, BANDS, 8, 8) * 0.4 + 0.05

    base = float(spectral_terms(sr, lr)["sam"])
    scaled = float(spectral_terms(sr * 3.0, lr * 3.0)["sam"])
    assert scaled == pytest.approx(base, rel=1e-4)

    shift = torch.tensor([0.1, -0.02, 0.05, 0.2]).reshape(1, BANDS, 1, 1)
    shifted = float(spectral_terms(sr - shift, lr - shift)["sam"])
    assert abs(shifted - base) > 0.05, (
        "a per-band shift left SAM unchanged; the test cannot demonstrate the "
        "invariance it exists to demonstrate."
    )


def test_denorm_recovers_the_reflectance_space_answer():
    """Given the normalisation's statistics, the loss undoes it and agrees."""
    torch.manual_seed(4)
    sr = torch.rand(1, BANDS, 32, 32) * 0.4 + 0.05
    lr = torch.rand(1, BANDS, 8, 8) * 0.4 + 0.05

    mean = [0.09, 0.08, 0.06, 0.25]
    std = [0.05, 0.04, 0.03, 0.10]
    mean_t = torch.tensor(mean).reshape(1, BANDS, 1, 1)
    std_t = torch.tensor(std).reshape(1, BANDS, 1, 1)

    reference = spectral_terms(sr, lr)
    recovered = spectral_terms(
        (sr - mean_t) / std_t, (lr - mean_t) / std_t, denorm=(mean, std)
    )

    assert float(recovered["sam"]) == pytest.approx(float(reference["sam"]), rel=1e-4)
    assert float(recovered["l1_spec"]) == pytest.approx(
        float(reference["l1_spec"]), rel=1e-4
    )


def test_denorm_rejects_statistics_that_do_not_match_the_band_count():
    sr = torch.rand(1, BANDS, 32, 32)
    lr = torch.rand(1, BANDS, 8, 8)
    with pytest.raises(ValueError, match="must match cfg.dataset.bands"):
        spectral_terms(sr, lr, denorm=([0.1, 0.1], [1.0, 1.0]))


# -- gradients -------------------------------------------------------------


def test_gradients_flow_to_sr_and_lr_is_detached():
    torch.manual_seed(5)
    sr = torch.rand(2, BANDS, 32, 32, requires_grad=True)
    lr = torch.rand(2, BANDS, 8, 8, requires_grad=True)

    total, _ = spectral_consistency(sr, lr, lam1=0.5, lam2=0.1)
    total.backward()

    assert sr.grad is not None
    assert torch.isfinite(sr.grad).all()
    assert float(sr.grad.abs().sum()) > 0.0
    # lr is a measurement. Nothing may propagate into it, even when the caller
    # hands over a tensor that asks for gradients.
    assert lr.grad is None


def test_both_terms_contribute_gradient_independently():
    """Neither term may be a no-op: lam1-only and lam2-only must both move sr."""
    torch.manual_seed(6)
    base = torch.rand(2, BANDS, 32, 32)
    lr = torch.rand(2, BANDS, 8, 8)

    grads = {}
    for label, (lam1, lam2) in {"l1": (1.0, 0.0), "sam": (0.0, 1.0)}.items():
        sr = base.clone().requires_grad_(True)
        total, _ = spectral_consistency(sr, lr, lam1=lam1, lam2=lam2)
        total.backward()
        grads[label] = sr.grad.clone()
        assert torch.isfinite(grads[label]).all()
        assert float(grads[label].abs().sum()) > 0.0

    # And they are not the same gradient wearing two names.
    assert not torch.allclose(grads["l1"], grads["sam"], atol=1e-6)


# -- dead pixels: finite loss AND finite gradients -------------------------


def test_a_dead_pixel_gives_finite_loss_and_finite_gradients():
    """The NaN this loss would otherwise emit, pinned.

    An all-zero spectrum has an undefined angle and, worse, a zero norm whose
    square-root derivative is infinite. Masking the pixel afterwards does not
    help -- ``torch.where`` propagates a NaN that was already created. The eps
    inside the square root is what makes this pass.
    """
    torch.manual_seed(7)
    sr = (torch.rand(1, BANDS, 32, 32) * 0.4 + 0.05).requires_grad_(True)
    lr = torch.rand(1, BANDS, 8, 8) * 0.4 + 0.05
    lr[0, :, 3, 5] = 0.0  # nodata, filled with cfg.dataset.nodata_fill

    total, parts = spectral_consistency(sr, lr, lam1=0.5, lam2=0.1)
    total.backward()

    assert math.isfinite(float(total))
    assert math.isfinite(parts["sam"])
    assert torch.isfinite(sr.grad).all(), "a dead pixel produced NaN gradients"
    # The pixel was excluded and counted, not quietly scored as 0 degrees.
    assert parts["sam_valid_frac"] == pytest.approx(63.0 / 64.0, rel=1e-6)


def test_an_entirely_dead_patch_reports_zero_valid_rather_than_perfection():
    sr = torch.zeros(1, BANDS, 32, 32, requires_grad=True)
    lr = torch.zeros(1, BANDS, 8, 8)

    total, parts = spectral_consistency(sr, lr, lam1=0.5, lam2=0.1)
    total.backward()

    assert parts["sam_valid_frac"] == 0.0
    assert math.isfinite(float(total))
    assert torch.isfinite(sr.grad).all()


def test_non_finite_input_raises_instead_of_being_masked():
    sr = torch.rand(1, BANDS, 32, 32)
    sr[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="sr contains non-finite"):
        spectral_terms(sr, torch.rand(1, BANDS, 8, 8))


# -- contract errors -------------------------------------------------------


def test_a_three_dimensional_input_is_rejected_not_guessed():
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        spectral_terms(torch.rand(BANDS, 32, 32), torch.rand(BANDS, 8, 8))


def test_a_scale_that_disagrees_with_the_tensors_raises():
    sr = torch.rand(1, BANDS, 32, 32)
    lr = torch.rand(1, BANDS, 8, 8)
    with pytest.raises(ValueError, match="config and the data disagree"):
        spectral_terms(sr, lr, scale=2)


def test_a_single_band_input_is_rejected():
    with pytest.raises(ValueError, match="at least 2 bands"):
        spectral_terms(torch.rand(1, 1, 32, 32), torch.rand(1, 1, 8, 8))


def test_a_negative_lambda_is_rejected():
    with pytest.raises(ValueError, match="rewards"):
        spectral_consistency(
            torch.rand(1, BANDS, 32, 32), torch.rand(1, BANDS, 8, 8), lam1=-0.1
        )


def test_per_sample_values_average_to_the_batch_value():
    """The floor script reports a std, so the per-patch path must be consistent."""
    torch.manual_seed(8)
    sr = torch.rand(4, BANDS, 32, 32)
    lr = torch.rand(4, BANDS, 8, 8)

    batch = spectral_terms(sr, lr, per_sample=False)
    each = spectral_terms(sr, lr, per_sample=True)

    assert each["l1_spec"].shape == (4,)
    assert float(each["l1_spec"].mean()) == pytest.approx(
        float(batch["l1_spec"]), rel=1e-5
    )
    assert float(each["sam"].mean()) == pytest.approx(float(batch["sam"]), rel=1e-5)


def test_per_sample_is_refused_by_the_weighted_wrapper():
    with pytest.raises(TypeError, match="per_sample"):
        spectral_consistency(
            torch.rand(1, BANDS, 32, 32),
            torch.rand(1, BANDS, 8, 8),
            per_sample=True,
        )


# -- the settings come from config, not from the code ----------------------


def test_settings_come_from_the_config_and_are_what_the_loss_takes():
    from src.utils.config import load_config

    cfg = load_config()
    settings = spectral_settings_from_cfg(cfg)
    assert settings["mode"] in ANTIALIASED_MODES
    # Splattable into the loss without renaming anything.
    sr = torch.rand(1, BANDS, 32, 32)
    lr = torch.rand(1, BANDS, 8, 8)
    spectral_terms(sr, lr, **settings)


def test_the_config_and_the_module_defaults_have_not_drifted():
    """The floor script reads base.yaml; the trainer uses the module defaults.

    ``src/train.py`` takes no ``--config`` (see configs/kaggle_jobs.yaml), so
    its numerics come from the module constants, while
    ``scripts/spectral_floor.py`` reads ``cfg.loss.spectral``. If the two ever
    disagree, the training curve and the floor it is judged against would be
    two different quantities plotted on one axis -- and nothing would say so.
    """
    from src.losses import (
        DEFAULT_COS_CLAMP,
        DEFAULT_DOWNSAMPLE,
        DEFAULT_EPS,
        DEFAULT_LAMBDA1,
        DEFAULT_LAMBDA2,
        DEFAULT_MIN_NORM,
    )
    from src.utils.config import load_config

    block = load_config().loss.spectral
    assert str(block.downsample) == DEFAULT_DOWNSAMPLE
    assert float(block.eps) == DEFAULT_EPS
    assert float(block.cos_clamp) == DEFAULT_COS_CLAMP
    assert float(block.min_norm) == DEFAULT_MIN_NORM
    # The candidate lambdas too: the reference values quoted in the module
    # docstring must be the ones a run would be launched with.
    assert float(block.lambda1) == DEFAULT_LAMBDA1
    assert float(block.lambda2) == DEFAULT_LAMBDA2


def test_a_config_without_the_block_raises_rather_than_defaulting():
    with pytest.raises(KeyError):
        spectral_settings_from_cfg({"loss": {}})
