"""The blur diagnostic has to order blur correctly or it is worse than nothing.

These tests do not check that the numbers are "right" -- neither proxy has a
correct value. They check the three properties the Day 3 comparison actually
rests on:

1. **Ordering.** Blurring an image must LOWER both proxies, and bicubic
   upsampling of an x4-decimated image must score below the original. If that
   ordering fails, "B drifted toward bicubic" means nothing.
2. **Invariance to the things that must not move it.** Batch composition, and
   for the frequency ratio a constant brightness offset, since the mean is
   removed by construction.
3. **Loud failure.** A NaN in a validation batch must raise here rather than
   travel into ``log.csv``, where it would void the diagnostic for the rest of
   an eight-hour run without failing it.
"""

from __future__ import annotations

import pytest
import torch

from src.metrics.sharpness import (
    DEFAULT_HF_CUTOFF,
    gradient_magnitude,
    hf_energy_ratio,
    reference_sharpness,
    sharpness_terms,
)


def _textured(batch: int = 2, channels: int = 4, size: int = 64) -> torch.Tensor:
    """A deterministic reflectance-like batch with detail at several scales.

    Returns:
        ``(batch, channels, size, size)`` float32, positive, spanning roughly
        ``[0, 1]`` -- surface reflectance in shape and units, not in origin.
        Built from fixed sinusoids plus a seeded noise term so every assertion
        below is reproducible without a data mount.
    """
    generator = torch.Generator().manual_seed(1337)
    coords = torch.linspace(0, 1, size)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    base = (
        0.30
        + 0.15 * torch.sin(2 * torch.pi * 3 * grid_x)
        + 0.10 * torch.cos(2 * torch.pi * 11 * grid_y)
    )
    image = base.expand(batch, channels, size, size).clone()
    image += 0.05 * torch.rand(image.shape, generator=generator)
    return image.float()


def _blur(image: torch.Tensor, sigma_kernel: int = 5) -> torch.Tensor:
    """Box-blur every band identically, as a stand-in for a blurry SR output."""
    channels = image.shape[1]
    kernel = torch.ones(
        channels, 1, sigma_kernel, sigma_kernel, dtype=image.dtype
    ) / (sigma_kernel**2)
    padded = torch.nn.functional.pad(
        image, (sigma_kernel // 2,) * 4, mode="replicate"
    )
    return torch.nn.functional.conv2d(padded, kernel, groups=channels)


# -- ordering: the property the whole diagnostic depends on ------------------


def test_blur_lowers_both_proxies():
    sharp = _textured()
    blurred = _blur(sharp)

    assert gradient_magnitude(blurred).mean() < gradient_magnitude(sharp).mean()
    assert hf_energy_ratio(blurred).mean() < hf_energy_ratio(sharp).mean()


def test_bicubic_upsample_scores_below_its_source():
    """The bicubic reference line must sit below ground truth, or it is not a floor.

    This is the exact operation ``reference_sharpness`` performs, so a failure
    here means the two logged reference lines would be in the wrong order and a
    model between them could not be located.
    """
    hr = _textured(size=64)
    lr = torch.nn.functional.interpolate(hr, scale_factor=0.25, mode="area")

    refs = reference_sharpness(hr, lr, scale=4)
    assert refs["bicubic_sharpness"] < refs["hr_sharpness"]
    assert refs["bicubic_hf_energy"] < refs["hr_hf_energy"]


def test_hf_energy_ratio_is_bounded():
    values = hf_energy_ratio(_textured())
    assert torch.all(values >= 0.0)
    assert torch.all(values <= 1.0)


def test_flat_image_has_no_high_frequency_energy():
    """A constant band has no AC power at all; the answer is 0, not a NaN."""
    flat = torch.full((1, 4, 32, 32), 0.42)
    assert float(hf_energy_ratio(flat).max()) == 0.0
    assert float(gradient_magnitude(flat).max()) == pytest.approx(0.0, abs=1e-6)


# -- invariance --------------------------------------------------------------


def test_batch_mean_matches_per_sample():
    """A batch's entries are independent: the diagnostic cannot leak across samples."""
    image = _textured(batch=3)
    batched = gradient_magnitude(image)
    for index in range(image.shape[0]):
        alone = gradient_magnitude(image[index])
        assert float(alone[0]) == pytest.approx(float(batched[index]), rel=1e-5)


def test_hf_energy_ignores_a_brightness_offset():
    """The mean is removed by construction, so DC brightness must not move it."""
    image = _textured()
    assert float(hf_energy_ratio(image + 0.25).mean()) == pytest.approx(
        float(hf_energy_ratio(image).mean()), rel=1e-5
    )


def test_unclipped_bright_reflectance_is_accepted():
    """Cloud and snow exceed 1.0 legitimately; neither proxy may clip or refuse."""
    image = _textured()
    image[:, :, :8, :8] = 1.7
    terms = sharpness_terms(image)
    assert terms["sharpness"] > 0.0
    assert 0.0 <= terms["hf_energy"] <= 1.0


def test_single_sample_and_batch_shapes():
    assert gradient_magnitude(torch.rand(4, 16, 16)).shape == (1,)
    assert gradient_magnitude(torch.rand(3, 4, 16, 16)).shape == (3,)
    assert hf_energy_ratio(torch.rand(4, 16, 16)).shape == (1,)


# -- loud failure ------------------------------------------------------------


def test_nan_raises_rather_than_propagating():
    image = _textured()
    image[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        gradient_magnitude(image)


def test_rejects_non_tensor():
    with pytest.raises(TypeError, match="torch.Tensor"):
        gradient_magnitude([[0.0, 1.0], [1.0, 0.0]])


def test_rejects_bad_rank():
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        gradient_magnitude(torch.rand(16, 16))


def test_rejects_image_smaller_than_sobel_support():
    with pytest.raises(ValueError, match="Sobel support"):
        gradient_magnitude(torch.rand(4, 2, 2))


def test_rejects_cutoff_outside_range():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="fraction of Nyquist"):
            hf_energy_ratio(_textured(), cutoff=bad)


def test_reference_sharpness_rejects_a_scale_that_does_not_fit():
    hr = _textured(size=64)
    lr = torch.nn.functional.interpolate(hr, scale_factor=0.25, mode="area")
    with pytest.raises(ValueError, match="does not describe"):
        reference_sharpness(hr, lr, scale=2)


def test_sharpness_terms_keys_and_types():
    terms = sharpness_terms(_textured(), cutoff=DEFAULT_HF_CUTOFF)
    assert set(terms) == {"sharpness", "hf_energy"}
    assert all(isinstance(value, float) for value in terms.values())
