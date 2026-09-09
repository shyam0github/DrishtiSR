"""Sharpness proxies: is the model buying spectral consistency with blur?

WHY THIS MODULE EXISTS. The spectral-consistency term penalises
``D(SR) != LR``, and there is a trivial way to satisfy it: return the bicubic
upsample of the LR. Blur is spectrally consistent. So a spectral run can post a
better ``l1_spec`` and a better ``sam`` while producing a *worse* image, and
none of PSNR, SSIM, SAM or ERGAS reliably says so -- PSNR in particular
*rewards* blur, because the mean-squared-error minimiser of an ambiguous
high-frequency detail is its average.

The two quantities here are the ones that do say so, and they are logged at
every validation for EVERY run including the control. The comparison they exist
to make is:

    if B's sharpness migrates toward the BICUBIC reference line while A2's
    holds, the spectral term is buying consistency with blur.

That has to be visible at hour 5 of a 9-hour session, not in the results table
afterwards, which is why these are training-loop instrumentation rather than a
post-hoc analysis script.

WHAT THEY ARE NOT. Neither is a quality metric and neither has a "good" value in
isolation: a model that hallucinates noise scores *higher* on both than the
ground truth does. They are only meaningful against the two fixed reference
lines this module's callers compute once per run --
:func:`reference_sharpness` over the ground-truth HR and over bicubic. Sharper
than GT is as much a red flag as blurrier than bicubic; the target is to sit at
the GT line, not above it.

Conventions are those of :mod:`src.metrics.image_quality`: surface reflectance,
float32, ``(C, H, W)`` or ``(B, C, H, W)``, band order ``cfg.dataset.bands``,
nominally ``[0, 1]`` and UNCLIPPED. Nothing here clips, rescales or normalises.

Unlike that module these functions stay in torch and return torch-free floats
without a NumPy round trip, because they run inside ``validate()`` on whatever
device the model is on, 25 batches at a time, every 500 iterations.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

__all__ = [
    "DEFAULT_HF_CUTOFF",
    "SOBEL_X",
    "SOBEL_Y",
    "gradient_magnitude",
    "hf_energy_ratio",
    "sharpness_terms",
    "reference_sharpness",
]

# Fraction of the Nyquist frequency above which power counts as "high".
#
# 0.25 -- i.e. radial frequency > 0.125 cycles/pixel -- and the value is not
# arbitrary. This is an x4 task: the LR carries detail only up to 1/4 of the HR
# Nyquist rate, so everything ABOVE this cutoff is exactly the band the network
# has to invent and the band bicubic leaves empty. A lower cutoff would measure
# structure the LR already contains, which both methods reproduce equally well,
# and the two reference lines would collapse onto each other.
DEFAULT_HF_CUTOFF = 0.25

# Sobel kernels, (H, W), unnormalised. Applied per band via a grouped conv.
# Sobel rather than a bare finite difference because its 3x3 support smooths
# along the edge while differentiating across it, so single-pixel sensor noise
# in the NIR band does not read as detail.
SOBEL_X = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
SOBEL_Y = ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))


def _as_batch(name: str, image: Any) -> torch.Tensor:
    """Validate an image and return it as ``(B, C, H, W)`` float32.

    Args:
        name: Argument name, for error messages.
        image: ``(C, H, W)`` or ``(B, C, H, W)`` torch tensor of surface
            reflectance. A single sample is promoted to a batch of one.

    Returns:
        A detached float32 tensor of shape ``(B, C, H, W)``, on the input's
        device. Values are untouched -- no clipping, no rescaling.

    Raises:
        TypeError: ``image`` is not a torch tensor. These run inside the
            training loop on device tensors; accepting arrays would mean a
            silent host round trip per validation batch.
        ValueError: The rank is not 3 or 4, an axis is empty, the image is
            smaller than the 3x3 Sobel support, or a value is non-finite.
            Non-finite is an error rather than a propagated NaN: a NaN here
            would poison the sharpness column of every subsequent row of
            ``log.csv`` while the run continued to look healthy.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor of surface reflectance, got "
            f"{type(image).__name__}."
        )
    if image.ndim not in (3, 4):
        raise ValueError(
            f"{name} must be (C, H, W) or (B, C, H, W), got shape "
            f"{tuple(image.shape)}."
        )

    batched = image.detach().float()
    if batched.ndim == 3:
        batched = batched.unsqueeze(0)

    if any(size == 0 for size in batched.shape):
        raise ValueError(
            f"{name} has an empty axis: shape {tuple(batched.shape)}."
        )
    height, width = batched.shape[-2:]
    if height < 3 or width < 3:
        raise ValueError(
            f"{name} is {height}x{width}, smaller than the 3x3 Sobel support. "
            f"Sharpness is undefined on a patch this small."
        )
    if not torch.isfinite(batched).all():
        raise ValueError(
            f"{name} contains non-finite values (NaN or inf). Sharpness is a "
            f"measurement, not a loss -- a NaN reaching log.csv would silently "
            f"void the blur diagnostic for the rest of the run."
        )
    return batched


def gradient_magnitude(image: Any) -> torch.Tensor:
    """Mean Sobel gradient magnitude -- the sharpness proxy.

    Per pixel and band, ``sqrt(Gx^2 + Gy^2)`` from the 3x3 Sobel operators, then
    averaged over bands and pixels. Higher means more local contrast, which for
    a super-resolution output means more detail -- real or hallucinated; this
    quantity cannot tell those apart, which is why it is read against the
    ground-truth reference line rather than maximised.

    Edges are handled by ``replicate`` padding, not zeros. Zero padding would
    manufacture a reflectance step at every border -- a patch edge against
    "reflectance 0" is a huge gradient -- and on 256px val patches that border
    artefact is a measurable fraction of the mean.

    Units are reflectance per pixel: the same units as the input, per unit
    spatial displacement. Not normalised by image brightness, deliberately --
    two runs are compared on the identical val patches, so a shared brightness
    factor cancels, and dividing by a per-image mean would make a dark tile look
    sharp.

    Args:
        image: ``(C, H, W)`` or ``(B, C, H, W)`` torch tensor, float32 surface
            reflectance, nominally ``[0, 1]`` and unclipped. May live on any
            device; the result is computed there.

    Returns:
        A 1-D tensor of shape ``(B,)`` -- one scalar per sample, band-averaged
        -- on the input's device, detached. A ``(C, H, W)`` input gives
        ``(1,)``. Non-negative and finite.

    Raises:
        TypeError: See :func:`_as_batch`.
        ValueError: See :func:`_as_batch`.

    Good values (MEASURED intent, not yet a table): the ground-truth HR and the
    bicubic upsample of the same val patches bracket the useful range. Bicubic
    sits well below GT because interpolation cannot create an edge that is not
    in the LR. A trained model should sit between them and drift toward GT; a
    model drifting toward bicubic is blurring, which is the failure this exists
    to catch.
    """
    batched = _as_batch("image", image)
    channels = batched.shape[1]

    kernel = torch.tensor(
        [SOBEL_X, SOBEL_Y], dtype=batched.dtype, device=batched.device
    ).unsqueeze(1)
    # One (2, 1, 3, 3) kernel repeated per band, run as a grouped conv, so the
    # bands are filtered independently rather than summed: a gradient is a
    # per-band quantity and mixing RGBN before the magnitude would let a bright
    # NIR edge cancel a dark red one.
    kernel = kernel.repeat(channels, 1, 1, 1)

    padded = torch.nn.functional.pad(batched, (1, 1, 1, 1), mode="replicate")
    grads = torch.nn.functional.conv2d(padded, kernel, groups=channels)

    # (B, 2C, H, W) -> (B, C, 2, H, W): channel c contributes rows 2c and 2c+1.
    grads = grads.view(batched.shape[0], channels, 2, *batched.shape[-2:])
    magnitude = torch.sqrt(grads[:, :, 0] ** 2 + grads[:, :, 1] ** 2)
    return magnitude.mean(dim=(1, 2, 3))


def hf_energy_ratio(image: Any, cutoff: float = DEFAULT_HF_CUTOFF) -> torch.Tensor:
    """Fraction of spatial-frequency power above ``cutoff`` x Nyquist.

    The frequency-domain companion to :func:`gradient_magnitude`, and the more
    direct statement of the question: an x4 super-resolver's entire job is to
    put power into the band the LR does not contain. This measures how much is
    there.

    Power is the squared magnitude of the 2-D DFT, and a pixel counts as "high
    frequency" when its RADIAL frequency ``sqrt(fy^2 + fx^2)`` exceeds
    ``cutoff * 0.5`` cycles/pixel (0.5 being Nyquist). Radial rather than
    per-axis so a diagonal edge is not counted differently from a vertical one.

    THE MEAN IS REMOVED PER BAND FIRST, and that choice decides what the number
    means. The DC term of a reflectance image is scene brightness, and it
    carries orders of magnitude more power than all the detail combined; leaving
    it in the denominator would make this a brightness metric with a sharpness
    rounding error attached. So the ratio here is of AC power: of the variation
    in the image, what fraction is fine detail. Removing the mean does not touch
    reflectance physics -- the returned quantity is dimensionless and the input
    tensor is not modified.

    Args:
        image: ``(C, H, W)`` or ``(B, C, H, W)`` torch tensor, float32 surface
            reflectance, nominally ``[0, 1]`` and unclipped.
        cutoff: Fraction of Nyquist, in ``(0, 1)``. Default
            :data:`DEFAULT_HF_CUTOFF` (0.25), which is the x4 band edge -- see
            the constant's comment before changing it, and change it in both
            the run and its reference lines or the comparison is meaningless.

    Returns:
        A 1-D tensor of shape ``(B,)`` on the input's device, detached. Each
        entry is in ``[0, 1]``: 0 when the sample is spatially flat in every
        band (no AC power at all), rising with fine detail.

    Raises:
        TypeError: See :func:`_as_batch`.
        ValueError: See :func:`_as_batch`, or ``cutoff`` outside ``(0, 1)``.

    Good values: as :func:`gradient_magnitude`, read only against the bicubic
    and ground-truth reference lines from :func:`reference_sharpness`. Bicubic's
    value is near-zero by construction above the x4 band edge -- interpolation
    puts almost no power there -- so this is the more sensitive of the two
    proxies for detecting a model that has given up and started interpolating.
    """
    if not 0.0 < float(cutoff) < 1.0:
        raise ValueError(
            f"cutoff must be a fraction of Nyquist in (0, 1), got {cutoff!r}. "
            f"0.25 is the x4 band edge; see DEFAULT_HF_CUTOFF."
        )
    batched = _as_batch("image", image)
    height, width = batched.shape[-2:]

    centred = batched - batched.mean(dim=(-2, -1), keepdim=True)
    # float32 FFT on 256px patches: the magnitudes involved are ~1e-1 to ~1e2,
    # nowhere near float32's dynamic range, and float64 would double the
    # validation cost of a diagnostic.
    power = torch.fft.fft2(centred).abs() ** 2

    freq_y = torch.fft.fftfreq(height, device=batched.device).view(-1, 1)
    freq_x = torch.fft.fftfreq(width, device=batched.device).view(1, -1)
    radial = torch.sqrt(freq_y**2 + freq_x**2)
    high = radial > (float(cutoff) * 0.5)

    total = power.sum(dim=(-2, -1))
    above = (power * high).sum(dim=(-2, -1))
    # Per band, then averaged, so a band with almost no variance cannot dominate
    # the ratio through the denominator. A truly flat band contributes 0 rather
    # than a division by zero; that is a real answer -- flat has no detail.
    ratio = torch.where(total > 0, above / total, torch.zeros_like(total))
    return ratio.mean(dim=1)


def sharpness_terms(
    image: Any, cutoff: float = DEFAULT_HF_CUTOFF
) -> Dict[str, float]:
    """Both proxies as batch-mean Python floats, for the training loop.

    The convenience wrapper ``validate()`` calls. Returns floats rather than
    tensors because the caller's next move is a CSV row and a running total, and
    keeping a graph-free scalar per batch is cheaper than accumulating device
    tensors it will only ever ``float()``.

    Args:
        image: ``(C, H, W)`` or ``(B, C, H, W)`` torch tensor, float32 surface
            reflectance, nominally ``[0, 1]`` and unclipped.
        cutoff: As :func:`hf_energy_ratio`.

    Returns:
        ``{"sharpness": float, "hf_energy": float}`` -- the mean over the batch
        of :func:`gradient_magnitude` (reflectance per pixel) and of
        :func:`hf_energy_ratio` (dimensionless, ``[0, 1]``).

    Raises:
        TypeError: See :func:`_as_batch`.
        ValueError: See :func:`_as_batch` / :func:`hf_energy_ratio`.
    """
    return {
        "sharpness": float(gradient_magnitude(image).mean()),
        "hf_energy": float(hf_energy_ratio(image, cutoff=cutoff).mean()),
    }


def reference_sharpness(
    hr: Any, lr: Any, scale: int, cutoff: float = DEFAULT_HF_CUTOFF
) -> Dict[str, float]:
    """The two fixed lines every run's sharpness curve is read against.

    Computed ONCE per run, over the same validation batches the model is scored
    on, and written into the run directory. Both runs use the same val split at
    the same crops (grid mode, epoch pinned at 0), so these lines are identical
    across A2 and B1 by construction -- which is exactly what makes "B drifted
    toward bicubic" a statement about B and not about its data.

    Bicubic is produced with ``align_corners=False`` and antialiasing OFF, which
    is the standard SR-literature upsample and the one
    ``src/eval/baselines.py`` scored the Day 1 floor with. It is NOT clipped:
    bicubic overshoot at a bright edge produces reflectance above the source
    maximum, and clipping it would flatten exactly the high-frequency content
    being measured.

    Args:
        hr: Ground-truth HR batch, ``(B, C, H, W)`` float32 surface reflectance,
            nominally ``[0, 1]`` and unclipped.
        lr: The matching LR batch, ``(B, C, H/scale, W/scale)``, same dtype,
            units and band order.
        scale: SR factor, ``cfg.sr.scale`` / ``--scale``. Never a literal.
        cutoff: As :func:`hf_energy_ratio`.

    Returns:
        ``{"hr_sharpness", "hr_hf_energy", "bicubic_sharpness",
        "bicubic_hf_energy"}``, all floats, in the units of
        :func:`sharpness_terms`.

    Raises:
        TypeError: See :func:`_as_batch`.
        ValueError: See :func:`_as_batch`, or the upsampled LR does not match
            the HR's spatial shape -- which means ``scale`` disagrees with the
            data and every reference line computed from it would be wrong.
    """
    hr_batch = _as_batch("hr", hr)
    lr_batch = _as_batch("lr", lr)

    upsampled = torch.nn.functional.interpolate(
        lr_batch, scale_factor=int(scale), mode="bicubic", align_corners=False
    )
    if upsampled.shape[-2:] != hr_batch.shape[-2:]:
        raise ValueError(
            f"bicubic reference is {tuple(upsampled.shape[-2:])} but the HR is "
            f"{tuple(hr_batch.shape[-2:])}: scale={scale} does not describe "
            f"this LR/HR pair, so the reference lines would be measured on a "
            f"different image than the model's output."
        )

    hr_terms = sharpness_terms(hr_batch, cutoff=cutoff)
    bicubic_terms = sharpness_terms(upsampled, cutoff=cutoff)
    return {
        "hr_sharpness": hr_terms["sharpness"],
        "hr_hf_energy": hr_terms["hf_energy"],
        "bicubic_sharpness": bicubic_terms["sharpness"],
        "bicubic_hf_energy": bicubic_terms["hf_energy"],
    }
