"""Tests for the reflectance-domain metrics.

These are correctness tests, not smoke tests. A metric that runs and returns a
plausible float is worse than no metric at all: every model comparison, every
ablation, and the final submission number rest on these five functions, and a
sign error or a transposed axis in one of them is invisible in a training curve.

Four properties are asserted, and if any of them fails the metric is wrong:

1. **Identity.** SSIM of an image against itself is exactly 1.0; PSNR is ``+inf``
   (asserted, not worked around); SAM is 0 degrees; ERGAS is 0.
2. **Known values.** SAM is checked against hand-computed angles for constructed
   spectra -- identical -> 0, orthogonal -> 90, and a pair whose angle is 45 by
   construction. This is the metric the spectral-consistency contribution is
   judged on, so it is the one pinned to arithmetic rather than to self-consistency.
3. **Monotone degradation.** Adding increasing Gaussian noise must make every
   metric monotonically worse -- PSNR and SSIM down, SAM and ERGAS up. This is
   what catches an inverted sign or a metric that is accidentally measuring the
   input against itself.
4. **Loud failure.** Shape mismatches, non-finite inputs, and degenerate patches
   raise rather than returning a number.
"""

import numpy as np
import pytest
import torch

from src.eval.baselines import (
    available_baselines,
    bicubic_upsample,
    get_baseline,
    nearest_upsample,
)
from src.metrics.image_quality import ergas, psnr, sam, ssim

BANDS = 4
SIZE = 64
DATA_RANGE = 1.0

# Noise levels in reflectance units, ascending. 0.005 is about the reflectance
# resolution of a Sentinel-2 scene; 0.08 is gross corruption.
NOISE_LEVELS = (0.005, 0.01, 0.02, 0.04, 0.08)


def _scene(seed=0, bands=BANDS, size=SIZE):
    """A smooth, textured reflectance scene with plausible per-band levels.

    Returns:
        ``(bands, size, size)`` float32 surface reflectance, roughly
        ``[0.02, 0.45]`` -- the range measured on real SEN2NAIPv2 vegetation --
        with enough structure for SSIM's window to be meaningful.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    levels = (0.058, 0.057, 0.034, 0.249)
    image = np.empty((bands, size, size), dtype=np.float32)
    for band in range(bands):
        field = gaussian_filter(rng.random((size, size)), sigma=2.5)
        field = (field - field.min()) / (field.max() - field.min() + 1e-9)
        level = levels[band % len(levels)]
        image[band] = (level * (0.5 + field)).astype(np.float32)
    return image


def _noisy(image, sigma, seed=1234):
    """Add zero-mean Gaussian noise in reflectance units. Deliberately unclipped."""
    rng = np.random.default_rng(seed)
    return (image + rng.normal(0.0, sigma, image.shape)).astype(np.float32)


# -- 1. identity -----------------------------------------------------------


def test_psnr_of_an_image_against_itself_is_infinite():
    """MSE is exactly 0, so PSNR is +inf. Asserted, not clamped to a big number.

    Anything that quietly returns 100 dB here is hiding a division guard that
    would also distort every real measurement.
    """
    image = _scene()
    result = psnr(image, image, data_range=DATA_RANGE)
    assert np.all(np.isinf(result.per_band))
    assert np.all(result.per_band > 0)
    assert np.isinf(result.mean) and result.mean > 0


def test_ssim_of_an_image_against_itself_is_one():
    image = _scene()
    result = ssim(image, image, data_range=DATA_RANGE)
    assert result.per_band.shape == (BANDS,)
    np.testing.assert_allclose(result.per_band, 1.0, atol=1e-12)
    assert result.mean == pytest.approx(1.0, abs=1e-12)


def test_sam_of_an_image_against_itself_is_zero():
    """Every pixel's spectrum is parallel to itself, so every angle is 0."""
    image = _scene()
    result = sam(image, image)
    assert result.map_deg.shape == (SIZE, SIZE)
    assert result.num_undefined == 0
    # float64 accumulation: sqrt-then-square rounding leaves at most ~1e-6 deg.
    assert result.mean_deg == pytest.approx(0.0, abs=1e-5)
    assert float(np.nanmax(result.map_deg)) == pytest.approx(0.0, abs=1e-5)


def test_ergas_of_an_image_against_itself_is_zero():
    image = _scene()
    assert ergas(image, image, scale=4) == pytest.approx(0.0, abs=1e-12)


# -- 2. SAM against hand-computed values -----------------------------------


def _uniform(spectrum, size=8):
    """An image where every pixel carries the same given spectrum."""
    vector = np.asarray(spectrum, dtype=np.float32)
    return np.repeat(vector[:, None, None], size, axis=1).repeat(size, axis=2)


def test_sam_identical_spectra_is_exactly_zero_degrees():
    image = _uniform([0.05, 0.06, 0.03, 0.25])
    result = sam(image, image)
    np.testing.assert_allclose(result.map_deg, 0.0, atol=1e-6)


def test_sam_orthogonal_spectra_is_ninety_degrees():
    """Two spectra with disjoint support are orthogonal: the angle is exactly 90.

    Hand-computed: <(1,0,0,0), (0,1,0,0)> = 0, so arccos(0) = pi/2 = 90 degrees.
    """
    first = _uniform([1.0, 0.0, 0.0, 0.0])
    second = _uniform([0.0, 1.0, 0.0, 0.0])
    result = sam(first, second)
    np.testing.assert_allclose(result.map_deg, 90.0, atol=1e-6)
    assert result.mean_deg == pytest.approx(90.0, abs=1e-6)


def test_sam_forty_five_degrees_by_construction():
    """<(1,0), (1,1)>/(1 * sqrt(2)) = 1/sqrt(2), so the angle is exactly 45."""
    first = _uniform([1.0, 0.0])
    second = _uniform([1.0, 1.0])
    result = sam(first, second)
    np.testing.assert_allclose(result.map_deg, 45.0, atol=1e-6)


def test_sam_is_invariant_to_a_positive_scaling_of_the_spectrum():
    """The property that makes SAM the spectral-consistency metric.

    Doubling every band changes brightness, not spectrum shape, so SAM must not
    move -- while PSNR must. If this ever fails, SAM has picked up a magnitude
    dependence and no longer measures what the contribution claims.
    """
    image = _scene(seed=3)
    scaled = (image * 2.0).astype(np.float32)
    assert sam(scaled, image).mean_deg == pytest.approx(0.0, abs=1e-5)
    assert np.isfinite(psnr(scaled, image, data_range=DATA_RANGE).mean)


def test_sam_angles_stay_within_zero_and_ninety_for_reflectance():
    """Non-negative spectra cannot be more than a right angle apart."""
    first, second = _scene(seed=4), _scene(seed=5)
    angles = sam(first, second).map_deg
    assert float(np.nanmin(angles)) >= 0.0
    assert float(np.nanmax(angles)) <= 90.0 + 1e-9


def test_sam_counts_undefined_pixels_instead_of_calling_them_zero():
    """A zero spectrum has no direction. It must be counted, never scored 0."""
    first = _scene(seed=6)
    second = first.copy()
    second[:, 0, 0] = 0.0
    result = sam(first, second)
    assert result.num_undefined == 1
    assert np.isnan(result.map_deg[0, 0])
    assert np.isfinite(result.mean_deg)

    with pytest.raises(ValueError, match="undefined"):
        sam(first, second, zero_vector_policy="error")


# -- 3. monotone degradation ------------------------------------------------


def _series(metric_fn):
    """Evaluate ``metric_fn(sr, hr)`` over ascending noise levels."""
    reference = _scene(seed=7)
    return [metric_fn(_noisy(reference, sigma), reference) for sigma in NOISE_LEVELS]


def test_psnr_falls_monotonically_with_noise():
    values = _series(lambda sr, hr: psnr(sr, hr, data_range=DATA_RANGE).mean)
    assert all(np.isfinite(values))
    assert values == sorted(values, reverse=True), values


def test_ssim_falls_monotonically_with_noise():
    values = _series(lambda sr, hr: ssim(sr, hr, data_range=DATA_RANGE).mean)
    assert all(0.0 <= v <= 1.0 for v in values), values
    assert values == sorted(values, reverse=True), values


def test_sam_rises_monotonically_with_noise():
    values = _series(lambda sr, hr: sam(sr, hr).mean_deg)
    assert all(np.isfinite(values))
    assert values == sorted(values), values


def test_ergas_rises_monotonically_with_noise():
    values = _series(lambda sr, hr: ergas(sr, hr, scale=4))
    assert all(np.isfinite(values))
    assert values == sorted(values), values


def test_per_band_psnr_degrades_only_in_the_band_that_was_corrupted():
    """Catches a transposed channel axis, which no aggregate metric would show."""
    reference = _scene(seed=8)
    corrupted = reference.copy()
    rng = np.random.default_rng(0)
    corrupted[1] += rng.normal(0.0, 0.05, corrupted[1].shape).astype(np.float32)

    per_band = psnr(corrupted, reference, data_range=DATA_RANGE).per_band
    assert np.isfinite(per_band[1])
    assert np.isinf(per_band[0]) and np.isinf(per_band[2]) and np.isinf(per_band[3])


# -- batching and containers -----------------------------------------------


def test_batched_and_unbatched_results_agree():
    """A (B, C, H, W) call must give exactly what B separate calls give."""
    first, second = _scene(seed=9), _scene(seed=10)
    sr = np.stack([first, _noisy(second, 0.02)])
    hr = np.stack([_noisy(first, 0.01), second])

    batched_psnr = psnr(sr, hr, data_range=DATA_RANGE)
    batched_ssim = ssim(sr, hr, data_range=DATA_RANGE)
    batched_sam = sam(sr, hr)
    batched_ergas = ergas(sr, hr, scale=4)

    assert batched_psnr.per_band.shape == (2, BANDS)
    assert batched_sam.map_deg.shape == (2, SIZE, SIZE)

    for index in range(2):
        single_psnr = psnr(sr[index], hr[index], data_range=DATA_RANGE)
        single_ssim = ssim(sr[index], hr[index], data_range=DATA_RANGE)
        single_sam = sam(sr[index], hr[index])
        np.testing.assert_allclose(batched_psnr.per_band[index], single_psnr.per_band)
        np.testing.assert_allclose(batched_ssim.per_band[index], single_ssim.per_band)
        assert batched_sam.mean_deg[index] == pytest.approx(single_sam.mean_deg)
        assert batched_ergas[index] == pytest.approx(
            ergas(sr[index], hr[index], scale=4)
        )


def test_numpy_and_torch_inputs_agree():
    sr, hr = _noisy(_scene(seed=11), 0.02), _scene(seed=11)
    from_numpy = psnr(sr, hr, data_range=DATA_RANGE).mean
    from_torch = psnr(torch.from_numpy(sr), torch.from_numpy(hr), data_range=DATA_RANGE)
    assert from_torch.mean == pytest.approx(from_numpy)


def test_reflectance_above_one_is_measured_not_clipped():
    """Bright targets exceed 1.0 reflectance and must reach the metric intact.

    A metric that clipped internally would score two different bright tiles
    identically, which is precisely the radiometric information the
    spectral-consistency contribution depends on.
    """
    hr = _scene(seed=12)
    hr[:, 0, 0] = 1.6  # a cloud or a specular roof
    sr_close = hr.copy()
    sr_close[:, 0, 0] = 1.5
    sr_far = hr.copy()
    sr_far[:, 0, 0] = 1.0

    assert psnr(sr_close, hr, data_range=DATA_RANGE).mean > psnr(
        sr_far, hr, data_range=DATA_RANGE
    ).mean


# -- 4. loud failure --------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    [
        lambda a, b: psnr(a, b),
        lambda a, b: ssim(a, b),
        lambda a, b: sam(a, b),
        lambda a, b: ergas(a, b, scale=4),
    ],
)
def test_shape_mismatch_raises(metric):
    with pytest.raises(ValueError, match="same shape"):
        metric(_scene(size=64), _scene(size=32))


@pytest.mark.parametrize(
    "metric",
    [
        lambda a, b: psnr(a, b),
        lambda a, b: ssim(a, b),
        lambda a, b: sam(a, b),
        lambda a, b: ergas(a, b, scale=4),
    ],
)
def test_non_finite_input_raises(metric):
    reference = _scene(seed=13)
    corrupted = reference.copy()
    corrupted[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        metric(corrupted, reference)


def test_wrong_rank_raises():
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        psnr(np.zeros((SIZE, SIZE), dtype=np.float32), np.zeros((SIZE, SIZE), np.float32))


def test_non_positive_data_range_raises():
    image = _scene(seed=14)
    with pytest.raises(ValueError, match="positive"):
        psnr(image, image, data_range=0.0)


def test_ergas_raises_on_an_all_zero_reference_band_by_default():
    """An all-zero band means a nodata patch, which should never be scored."""
    reference = _scene(seed=15)
    reference[2] = 0.0
    with pytest.raises(ValueError, match="undefined"):
        ergas(reference, reference, scale=4)
    assert np.isnan(ergas(reference, reference, scale=4, zero_mean_policy="nan"))


def test_sam_needs_at_least_two_bands():
    single = _scene(seed=16, bands=1)
    with pytest.raises(ValueError, match="at least 2 bands"):
        sam(single, single)


# -- baselines --------------------------------------------------------------


def test_baselines_produce_the_target_geometry():
    lr = _scene(seed=17, size=16)
    for name in available_baselines():
        sr = get_baseline(name, 4)(lr)
        assert sr.shape == (BANDS, 64, 64), name
        assert sr.dtype == np.float32, name


def test_nearest_upsample_replicates_exact_reflectance_values():
    """The floor adds nothing: every output value is an input value."""
    lr = _scene(seed=18, size=8)
    sr = nearest_upsample(lr, 4)
    np.testing.assert_array_equal(sr[:, ::4, ::4], lr)
    assert set(np.unique(sr).tolist()) <= set(np.unique(lr).tolist())


def test_nearest_upsample_matches_torch_nearest_interpolation():
    lr = torch.from_numpy(_scene(seed=19, size=8))[None]
    ours = nearest_upsample(lr, 4)
    theirs = torch.nn.functional.interpolate(lr, scale_factor=4, mode="nearest")
    torch.testing.assert_close(ours, theirs)


def test_bicubic_beats_nearest_on_a_smooth_scene():
    """The ordering the two baselines exist to establish.

    Bicubic must beat pixel replication on every metric, on a scene built by
    block-mean degradation -- the same relationship the SR task inverts. If this
    fails, the resampling direction or the scale factor is wrong.
    """
    hr = _scene(seed=20, size=64)
    lr = hr.reshape(BANDS, 16, 4, 16, 4).mean(axis=(2, 4)).astype(np.float32)

    bicubic = bicubic_upsample(lr, 4)
    nearest = nearest_upsample(lr, 4)

    assert psnr(bicubic, hr, data_range=DATA_RANGE).mean > psnr(
        nearest, hr, data_range=DATA_RANGE
    ).mean
    assert ssim(bicubic, hr, data_range=DATA_RANGE).mean > ssim(
        nearest, hr, data_range=DATA_RANGE
    ).mean
    assert ergas(bicubic, hr, scale=4) < ergas(nearest, hr, scale=4)


def test_bicubic_does_not_clip_its_overshoot():
    """Ringing at a hard edge pushes reflectance out of range, and it is kept.

    Clipping here would flatter the baseline and would hide the artefact a
    learned model is supposed to remove.
    """
    lr = np.zeros((2, 8, 8), dtype=np.float32)
    lr[:, :, 4:] = 0.9
    sr = bicubic_upsample(lr, 4)
    assert float(sr.min()) < 0.0 or float(sr.max()) > 0.9


def test_bicubic_antialias_flag_selects_a_different_kernel():
    """The two torch bicubic kernels genuinely differ; the choice is recorded.

    ``src/eval/baselines.py`` documents a measured 0.083 reflectance difference
    and picks ``antialias=True`` for comparability with the SR literature;
    ``src/eval/alignment.py`` deliberately keeps ``antialias=False``. This pins
    that they are not interchangeable, so neither call site can be "tidied" into
    the other.
    """
    lr = _scene(seed=21, size=16)
    difference = np.abs(
        bicubic_upsample(lr, 4, antialias=True)
        - bicubic_upsample(lr, 4, antialias=False)
    ).max()
    assert difference > 1e-4


def test_baselines_preserve_the_container_type_and_rank():
    lr_np = _scene(seed=22, size=8)
    lr_torch = torch.from_numpy(lr_np)[None]

    assert isinstance(bicubic_upsample(lr_np, 2), np.ndarray)
    assert bicubic_upsample(lr_np, 2).shape == (BANDS, 16, 16)

    out = bicubic_upsample(lr_torch, 2)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, BANDS, 16, 16)


def test_unknown_baseline_raises():
    with pytest.raises(KeyError, match="Unknown baseline"):
        get_baseline("lanczos", 4)
