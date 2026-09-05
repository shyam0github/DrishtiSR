"""Reflectance-domain image quality metrics.

Every number the project reports about image quality comes from this module, for
the bicubic baseline and for every model that follows. That is deliberate: a
metric implemented once and unit-tested is a metric whose change between two
experiments means something. A metric re-derived per experiment is a coin flip.

Conventions that hold for every function here
---------------------------------------------
- **Input is surface reflectance**, float32, ``(C, H, W)`` for a single sample or
  ``(B, C, H, W)`` for a batch, in the band order of ``cfg.dataset.bands``
  (``[B04, B03, B02, B08]`` -- red first, NIR last). Values are nominally
  ``[0, 1]`` but **unclipped**: cloud, snow, specular water, and bright roofs
  legitimately exceed 1.0.
- **Nothing here clips, rescales, or normalises reflectance.** The single
  exception is :func:`lpips`, whose network only accepts ``[-1, 1]``; that clip
  is explicit, configurable, counted, and reported. No ImageNet statistics are
  used anywhere -- which is part of why :func:`lpips` carries the caveat it does.
- **``data_range`` is fixed, not per-image.** PSNR and SSIM take the range from
  ``cfg.metrics.data_range`` (1.0 reflectance) rather than from each tile's own
  min/max. A per-image range would make a bright tile score better for being
  bright, and would make two runs incomparable.
- **Non-finite input is an error, not a NaN that propagates.** Every function
  validates its inputs and raises. A NaN reaching a summary statistic turns the
  whole evaluation into a silent no-op.
- Outputs are NumPy arrays / Python floats, always detached from any autograd
  graph. These are measurement functions, not losses; nothing here is meant to be
  backpropagated through. Losses live in ``src/losses/``.

Shape convention for returns
----------------------------
A metric given ``(C, H, W)`` returns scalars (and a ``(H, W)`` map where it has
one). Given ``(B, C, H, W)`` it returns arrays with a leading batch axis. The
per-band results of :func:`psnr` and :func:`ssim` are ``(C,)`` and ``(B, C)``
respectively.

Reference values
----------------
The "good values" note in each docstring gives the MEASURED baseline on this
dataset, so a number can be sanity-checked the moment it appears.

MEASURED, ``scripts/run_baseline.py --baseline all --set
loader.cached_only=false``, over the COMPLETE validation split: 1199 patches from
300 tiles of SEN2NAIPv2 crosssensor (all 3000 samples downloaded and indexed),
x4, ``data_range=1.0``, geographic split from ``scripts/make_splits.py``. Both
methods scored on the identical patch set in one invocation:

===============  ==========  ==========  ==========  ===========
metric           bicubic     nearest     bicubic p5  bicubic p95
===============  ==========  ==========  ==========  ===========
PSNR (dB)          38.47       37.86       30.17       46.35
SSIM                0.883       0.867       0.714       0.977
SAM (degrees)       2.09        2.24        0.75        4.39
ERGAS               3.05        3.28        1.07        6.07
LPIPS (alex)        0.403       0.329       0.153       0.621
===============  ==========  ==========  ==========  ===========

These are the numbers every model result is quoted against. Beating the bicubic
column is the minimum bar for the submission.

Two things in that table are worth internalising before reading any model result:

1. **The spread dwarfs the method difference.** Bicubic beats pixel replication
   by 0.6 dB, while bicubic's own p5-to-p95 range spans 16 dB. A model reported
   only as a mean has not been evaluated.
2. **LPIPS ranks nearest-neighbour above bicubic** -- it prefers the method that
   adds literally nothing, because replication's blocky edges read as "sharp" to
   a network trained on photographs. That is not a bug in the implementation; it
   is :data:`LPIPS_CAVEAT` demonstrated on our own data, and it is why LPIPS must
   never be the metric a decision rests on.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from skimage.metrics import structural_similarity as _skimage_ssim

__all__ = [
    "BandMetric",
    "SAMResult",
    "LPIPSResult",
    "LPIPS_CAVEAT",
    "psnr",
    "ssim",
    "lpips",
    "sam",
    "ergas",
    "rgb_band_indices",
    "clear_lpips_cache",
]


# Repeated verbatim in the results table and in the report. It is a sentence
# about what the number does NOT mean, which is the part that gets dropped when
# a metric is copied onto a slide.
LPIPS_CAVEAT = (
    "LPIPS is a perceptual proxy: its backbone was trained on natural RGB "
    "photographs, not on multispectral surface reflectance. On satellite data it "
    "is indicative of perceptual sharpness, not authoritative. It sees only the "
    "RGB bands, ignores NIR entirely, and requires reflectance to be clipped "
    "into [0, 1] before use. Rank models by PSNR/SSIM/SAM/ERGAS; read LPIPS as a "
    "tie-breaker on apparent detail."
)


class BandMetric(NamedTuple):
    """A per-band metric and its across-band mean.

    Attributes:
        per_band: ``(C,)`` for a single sample, ``(B, C)`` for a batch. Float64.
        mean: Arithmetic mean over the band axis -- a Python ``float`` for a
            single sample, ``(B,)`` float64 for a batch.
    """

    per_band: np.ndarray
    mean: Union[float, np.ndarray]


class SAMResult(NamedTuple):
    """Spectral angle mapper output.

    Attributes:
        mean_deg: Mean spectral angle in degrees over the valid pixels -- a
            ``float`` for a single sample, ``(B,)`` float64 for a batch.
        map_deg: The full per-pixel angle map in degrees, ``(H, W)`` or
            ``(B, H, W)`` float64. Undefined pixels (see ``num_undefined``) hold
            ``nan``.
        num_undefined: Count of pixels whose spectrum is the zero vector in
            either image, so the angle between them does not exist -- an ``int``
            for a single sample, ``(B,)`` int64 for a batch.
    """

    mean_deg: Union[float, np.ndarray]
    map_deg: np.ndarray
    num_undefined: Union[int, np.ndarray]


class LPIPSResult(NamedTuple):
    """LPIPS distance plus the bookkeeping that makes it interpretable.

    Attributes:
        distance: LPIPS distance, ``float`` or ``(B,)`` float64. Lower is better.
        clipped_fraction: Fraction of RGB reflectance values that had to be
            clipped into ``[0, 1]`` before the network could see them, ``float``
            or ``(B,)``. A large value means the tile is bright (cloud, snow,
            roofs) and its LPIPS number describes a clipped image, not the data.
        caveat: :data:`LPIPS_CAVEAT`, carried alongside the number so it reaches
            the results table.
    """

    distance: Union[float, np.ndarray]
    clipped_fraction: Union[float, np.ndarray]
    caveat: str = LPIPS_CAVEAT


# -- input handling --------------------------------------------------------


def _as_batch(
    image: Any, name: str, *, dtype: torch.dtype = torch.float64
) -> Tuple[torch.Tensor, bool]:
    """Normalise an input to ``(B, C, H, W)`` and say whether it was batched.

    Args:
        image: ``(C, H, W)`` or ``(B, C, H, W)``, NumPy array or torch tensor,
            surface reflectance, nominally ``[0, 1]`` and unclipped.
        name: Argument name, for error messages.
        dtype: Working dtype. Float64 by default: PSNR of a well-matched pair
            takes the log of a ratio of two small numbers, and float32
            accumulation over a 512x512 patch costs a visible fraction of a dB.

    Returns:
        ``(tensor, was_batched)`` where ``tensor`` is contiguous, detached, on
        the CPU, and of ``dtype``. Reflectance values are unchanged.

    Raises:
        TypeError: ``image`` is neither an array nor a tensor.
        ValueError: Rank is not 3 or 4, an axis is empty, or a value is
            non-finite.
    """
    if isinstance(image, torch.Tensor):
        tensor = image.detach().to(device="cpu", dtype=dtype)
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(image)).to(dtype=dtype)
    else:
        raise TypeError(
            f"{name} must be a numpy.ndarray or torch.Tensor of surface "
            f"reflectance, got {type(image).__name__}."
        )

    if tensor.ndim == 3:
        batched = False
        tensor = tensor[None]
    elif tensor.ndim == 4:
        batched = True
    else:
        raise ValueError(
            f"{name} must be (C, H, W) or (B, C, H, W); got shape "
            f"{tuple(tensor.shape)}. The channel axis comes before the spatial "
            "axes everywhere in this project."
        )

    if min(tensor.shape) == 0:
        raise ValueError(f"{name} has an empty axis: shape {tuple(tensor.shape)}.")

    if not torch.isfinite(tensor).all():
        num_bad = int((~torch.isfinite(tensor)).sum())
        raise ValueError(
            f"{name} contains {num_bad} non-finite value(s) (NaN or inf). "
            "Reflectance arrays must be finite before they reach a metric -- "
            "nodata is masked and filled by the dataset "
            "(cfg.dataset.nodata_fill). A NaN here would propagate silently into "
            "every summary statistic."
        )

    return tensor.contiguous(), batched


def _check_pair(sr: Any, hr: Any) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Validate and normalise an ``(sr, hr)`` pair to matching ``(B, C, H, W)``.

    Args:
        sr: Super-resolved reflectance, ``(C, H, W)`` or ``(B, C, H, W)``.
        hr: Reference reflectance, same shape, band order and units.

    Returns:
        ``(sr_tensor, hr_tensor, was_batched)``, both float64 ``(B, C, H, W)``.

    Raises:
        ValueError: The two differ in rank, batch size, channel count, or
            spatial size. A shape mismatch is always a bug in the caller -- an SR
            output that is not the size of its target means the scale factor or
            the crop geometry is wrong, and resizing one to match would hide it.
    """
    sr_t, sr_batched = _as_batch(sr, "sr")
    hr_t, hr_batched = _as_batch(hr, "hr")
    if sr_batched != hr_batched:
        raise ValueError(
            f"sr is {'batched' if sr_batched else 'unbatched'} but hr is "
            f"{'batched' if hr_batched else 'unbatched'}: shapes "
            f"{tuple(sr_t.shape)} and {tuple(hr_t.shape)}."
        )
    if sr_t.shape != hr_t.shape:
        raise ValueError(
            f"sr and hr must have the same shape; got {tuple(sr_t.shape)} and "
            f"{tuple(hr_t.shape)}. If the super-resolved output is not the size "
            "of the target, the scale factor or the patch geometry is wrong; "
            "resampling here would hide that."
        )
    return sr_t, hr_t, sr_batched


def _unbatch(values: np.ndarray, batched: bool) -> Union[float, np.ndarray]:
    """Return ``values`` unchanged when batched, else its single element."""
    return values if batched else float(values[0])


def _check_data_range(data_range: float) -> float:
    """Validate a reflectance ``data_range`` and return it as a float."""
    value = float(data_range)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"data_range must be a positive, finite number of reflectance units; "
            f"got {data_range!r}. Use cfg.metrics.data_range (1.0), the nominal "
            "reflectance span -- never a per-image max, which makes two runs "
            "incomparable."
        )
    return value


# -- PSNR ------------------------------------------------------------------


def psnr(sr: Any, hr: Any, data_range: float = 1.0) -> BandMetric:
    """Peak signal-to-noise ratio, per band and averaged across bands.

    Computed against a **fixed** ``data_range`` in reflectance units, not against
    each tile's own dynamic range, so numbers are comparable between tiles, runs,
    and models.

    The across-band ``mean`` is the arithmetic mean of the per-band dB values,
    the convention in the remote-sensing SR literature. It is **not** the PSNR of
    the pooled MSE; the two differ whenever bands have unequal error, and the
    per-band mean is the more pessimistic and more informative of the two.
    Because it averages dB, one perfectly reconstructed band makes the mean
    ``inf`` -- see below.

    Args:
        sr: Super-resolved image. ``(C, H, W)`` or ``(B, C, H, W)``, float32
            surface reflectance, nominally ``[0, 1]`` and unclipped.
        hr: Reference image, same dtype, shape, band order and units as ``sr``.
        data_range: Peak signal value in reflectance units. Pass
            ``cfg.metrics.data_range``. Reflectance above this value is legal and
            is **not** clipped; a bright tile simply reports a lower PSNR than its
            raw error suggests, which is the honest reading.

    Returns:
        :class:`BandMetric`. ``per_band`` is ``(C,)`` / ``(B, C)`` in dB; ``mean``
        is a float / ``(B,)`` in dB. Range is ``(0, inf]`` for any pair whose
        error is below the data range.

        A band reconstructed exactly (MSE = 0) yields ``+inf``, which is
        mathematically correct and is what the identity unit test asserts.
        Callers that aggregate must handle it explicitly --
        :class:`src.metrics.aggregate.Evaluator` counts non-finite entries per
        metric and reports the count rather than dropping them quietly.

    Raises:
        ValueError: Shapes disagree, an input is non-finite, or ``data_range`` is
            not positive and finite.

    Good values (MEASURED on SEN2NAIPv2, x4, ``data_range=1.0``; see the module
    docstring for the full baseline table):
        Bicubic scores 38.47 dB mean, with p5 30.17 and a worst tile at 24.18 --
        a 16 dB p5-to-p95 spread, which is why the mean alone says very little.
        Pixel replication scores 37.86, so the whole dynamic range this metric
        has to express the difference between "did nothing" and "interpolated
        properly" is 0.6 dB; judge a model against the percentiles, not just the
        mean. Below ~25 dB on a typical tile, suspect LR/HR misregistration
        rather than a weak model -- run ``scripts/qa_alignment.py`` first.
    """
    sr_t, hr_t, batched = _check_pair(sr, hr)
    peak = _check_data_range(data_range)

    mse = ((sr_t - hr_t) ** 2).mean(dim=(-2, -1)).numpy()  # (B, C)
    with np.errstate(divide="ignore"):
        per_band = 10.0 * np.log10((peak**2) / mse)
    mean = per_band.mean(axis=1)

    if batched:
        return BandMetric(per_band=per_band, mean=mean)
    return BandMetric(per_band=per_band[0], mean=float(mean[0]))


# -- SSIM ------------------------------------------------------------------


def ssim(
    sr: Any,
    hr: Any,
    data_range: float = 1.0,
    gaussian_weights: bool = True,
    sigma: float = 1.5,
    win_size: Optional[int] = None,
) -> BandMetric:
    """Structural similarity, per band and averaged across bands.

    Implementation: :func:`skimage.metrics.structural_similarity`, called once
    per band on the 2-D reflectance plane.

    Why scikit-image rather than a hand-rolled or torch SSIM: it is the reference
    against which most published SR numbers are reproducible, it is already a
    pinned dependency (0.25.2, preinstalled on Kaggle), and it carries its own
    upstream test suite -- so a disagreement between our number and a paper's is
    a difference in settings, not a difference in somebody's convolution. A torch
    SSIM would also invite use as a loss, and losses belong in ``src/losses/``.

    Defaults reproduce Wang et al. (2004): an 11x11 Gaussian window with
    ``sigma=1.5`` and the population (not sample) covariance. These are *not*
    scikit-image's own defaults -- it defaults to a 7x7 uniform window with the
    sample covariance -- so they are passed explicitly here and recorded in the
    evaluation summary.

    Args:
        sr: ``(C, H, W)`` or ``(B, C, H, W)`` float32 surface reflectance,
            nominally ``[0, 1]``, unclipped.
        hr: Reference, same shape, dtype, band order and units.
        data_range: Reflectance span used for SSIM's stabilising constants. Pass
            ``cfg.metrics.data_range`` (1.0). Fixed, never per-image.
        gaussian_weights: Use a Gaussian window (Wang et al.) rather than a
            uniform one.
        sigma: Gaussian window sigma, in pixels.
        win_size: Window edge in pixels. ``None`` lets scikit-image derive it
            (11 for ``sigma=1.5`` with ``gaussian_weights=True``). Must be odd
            and no larger than the shorter image side.

    Returns:
        :class:`BandMetric` with ``per_band`` ``(C,)`` / ``(B, C)`` and ``mean`` a
        float / ``(B,)``. SSIM lies in ``[-1, 1]``; an image against itself is
        exactly 1.0.

    Raises:
        ValueError: Shapes disagree, an input is non-finite, or the image is
            smaller than the SSIM window (scikit-image's message is re-raised
            with the patch size attached, because in this project that means
            ``cfg.patches.lr_size * cfg.sr.scale`` is too small).

    Good values (MEASURED on SEN2NAIPv2, x4):
        Bicubic scores 0.883 mean, p5 0.714, worst tile 0.520; pixel replication
        scores 0.867. The p5 is the number to beat -- it is where the built-up
        and field-boundary tiles sit. Below ~0.70 with a plausible PSNR usually
        means structural misalignment rather than blur.
    """
    sr_t, hr_t, batched = _check_pair(sr, hr)
    peak = _check_data_range(data_range)

    batch, channels, height, width = sr_t.shape
    sr_np = sr_t.numpy()
    hr_np = hr_t.numpy()

    kwargs = {
        "data_range": peak,
        "gaussian_weights": bool(gaussian_weights),
        "sigma": float(sigma),
        # Population covariance -- the Wang et al. definition. scikit-image's
        # default (sample covariance) shifts SSIM by a few thousandths, which is
        # exactly the size of the differences we will be reporting between
        # ablations.
        "use_sample_covariance": False,
    }
    if win_size is not None:
        kwargs["win_size"] = int(win_size)

    per_band = np.empty((batch, channels), dtype=np.float64)
    for b in range(batch):
        for c in range(channels):
            try:
                per_band[b, c] = _skimage_ssim(sr_np[b, c], hr_np[b, c], **kwargs)
            except ValueError as exc:
                raise ValueError(
                    f"SSIM failed on a {height}x{width} patch (band index {c}): "
                    f"{exc} In this project the HR patch size is "
                    "cfg.patches.lr_size * cfg.sr.scale; raise it, or pass an "
                    "explicit odd win_size no larger than the shorter side."
                ) from exc

    mean = per_band.mean(axis=1)
    if batched:
        return BandMetric(per_band=per_band, mean=mean)
    return BandMetric(per_band=per_band[0], mean=float(mean[0]))


# -- LPIPS -----------------------------------------------------------------


_LPIPS_CACHE: dict = {}


def clear_lpips_cache() -> None:
    """Drop cached LPIPS networks. For tests, and for freeing memory."""
    _LPIPS_CACHE.clear()


def rgb_band_indices(cfg: Any) -> Tuple[int, ...]:
    """Channel indices of the RGB bands, looked up by name -- never assumed.

    ``cfg.dataset.bands`` is ``[B04, B03, B02, B08]``: red first, NIR last. The
    naive ``(0, 1, 2)`` happens to be correct for that order, but it is wrong the
    moment the band list changes, and the failure is invisible -- LPIPS on a
    blue/green/red image still returns a plausible number. So the indices are
    resolved from the names in config every time.

    Args:
        cfg: The loaded config. Reads ``dataset.bands`` and
            ``metrics.lpips.rgb_bands``.

    Returns:
        A 3-tuple of channel indices, in (R, G, B) order.

    Raises:
        KeyError: A requested RGB band is not in ``cfg.dataset.bands``.
        ValueError: ``metrics.lpips.rgb_bands`` does not name exactly three bands.
    """
    bands = [str(b) for b in cfg["dataset"]["bands"]]
    wanted = [str(b) for b in cfg["metrics"]["lpips"]["rgb_bands"]]
    missing = [b for b in wanted if b not in bands]
    if missing:
        raise KeyError(
            f"metrics.lpips.rgb_bands requests {missing}, which are not in "
            f"cfg.dataset.bands={bands}. LPIPS needs three real RGB channels; "
            "either fix the band list or disable LPIPS "
            "(metrics.lpips.enabled=false)."
        )
    if len(wanted) != 3:
        raise ValueError(
            "metrics.lpips.rgb_bands must name exactly three bands in (R, G, B) "
            f"order; got {wanted}."
        )
    return tuple(bands.index(b) for b in wanted)


def _load_lpips(net: str):
    """Load and cache an LPIPS network, failing loudly when it is unavailable.

    Args:
        net: Backbone name, ``"alex"`` or ``"vgg"``.

    Returns:
        An ``lpips.LPIPS`` module in eval mode with gradients disabled.

    Raises:
        ImportError: The ``lpips`` package is not installed.
        RuntimeError: The pretrained weights could not be fetched or loaded.
    """
    key = str(net)
    if key in _LPIPS_CACHE:
        return _LPIPS_CACHE[key]
    try:
        import lpips as _lpips_pkg
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "metrics.lpips.enabled is true but the 'lpips' package is not "
            "installed. Install it with `pip install lpips==0.1.4`, or set "
            "metrics.lpips.enabled=false to report the other metrics without it. "
            "LPIPS is a perceptual proxy (see LPIPS_CAVEAT); the evaluation is "
            "still valid without it."
        ) from exc
    try:
        model = _lpips_pkg.LPIPS(net=key)
    except Exception as exc:  # pragma: no cover - network / weights failure
        raise RuntimeError(
            f"Could not construct the LPIPS {key!r} network. It downloads "
            "pretrained backbone weights on first use, so this fails on a "
            "machine with no internet -- on Kaggle, enable the notebook's "
            "internet toggle, or set metrics.lpips.enabled=false."
        ) from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _LPIPS_CACHE[key] = model
    return model


def lpips(
    sr: Any,
    hr: Any,
    rgb_indices: Sequence[int] = (0, 1, 2),
    net: str = "alex",
    data_range: float = 1.0,
    clip_reflectance: bool = True,
) -> LPIPSResult:
    """Learned perceptual image patch similarity, on the RGB bands only.

    .. warning::

       LPIPS is a **perceptual proxy, not an authoritative metric here**. Its
       backbone (AlexNet/VGG) was trained on natural RGB photographs. Sentinel-2
       surface reflectance is neither natural-photograph statistics nor RGB: the
       NIR band, which carries most of the vegetation signal, is discarded
       outright, and the remaining three channels must be squeezed into the
       ``[-1, 1]`` domain the network expects. Treat it as an indicator of
       apparent sharpness and rank models by PSNR / SSIM / SAM / ERGAS. This
       caveat is returned with every result (:data:`LPIPS_CAVEAT`) and printed as
       a footnote under the results table.

    The conversion is done here explicitly rather than hidden in a helper,
    because every step of it is a place where reflectance stops being physical:

    1. Select the R, G, B channels by index (from :func:`rgb_band_indices`, which
       looks them up by name in config). NIR is dropped.
    2. Divide by ``data_range`` so nominal reflectance maps to ``[0, 1]``.
    3. **Clip to ``[0, 1]``** -- required, because the network's domain is
       bounded and bright targets exceed 1.0 reflectance. This is the only clip
       in the metrics module. It is configurable (``clip_reflectance``), it
       applies to a temporary copy that never leaves this function, and the
       fraction of clipped values is returned, so a suspiciously good LPIPS on a
       cloudy tile can be recognised as an artefact of the clip.
    4. Map to ``[-1, 1]`` with ``x * 2 - 1``, the domain LPIPS was calibrated on.

    Note what step 2 is *not*: it is a division by a physical constant, not an
    ImageNet mean/std normalisation. The ImageNet-style scaling LPIPS applies
    internally is part of the pretrained network and is one more reason the
    number is a proxy.

    Args:
        sr: ``(C, H, W)`` or ``(B, C, H, W)`` float32 surface reflectance,
            nominally ``[0, 1]``, unclipped, in ``cfg.dataset.bands`` order.
        hr: Reference, same shape, dtype, band order and units.
        rgb_indices: Channel indices for (R, G, B). The default ``(0, 1, 2)``
            matches ``cfg.dataset.bands = [B04, B03, B02, B08]``; pass
            :func:`rgb_band_indices` output rather than relying on it.
        net: LPIPS backbone, ``"alex"`` (default, the version upstream
            recommends for evaluation) or ``"vgg"``.
        data_range: Reflectance value mapped to 1.0 before the ``[-1, 1]`` shift.
        clip_reflectance: Apply step 3. Setting this False feeds out-of-domain
            values to the network and is only meaningful for an experiment about
            the clip itself.

    Returns:
        :class:`LPIPSResult`: ``distance`` (float / ``(B,)``, lower is better,
        exactly 0.0 for an image against itself, typically below ~1 for real
        pairs), ``clipped_fraction`` (float / ``(B,)``, over both images' RGB
        values together), and ``caveat``.

    Raises:
        ImportError: The ``lpips`` package is not installed.
        RuntimeError: The pretrained weights could not be loaded (no internet).
        ValueError: Shapes disagree, an input is non-finite, or ``rgb_indices``
            is not three valid channel indices.

    Good values (MEASURED on SEN2NAIPv2, x4, ``net="alex"``):
        Bicubic scores 0.403. **Nearest-neighbour replication scores 0.329** --
        that is, LPIPS ranks the method that adds nothing at all ABOVE the one
        that interpolates properly, because replication's blocky edges read as
        sharpness to a network trained on photographs, while every other metric
        ranks them the other way round. That inversion, measured on our own
        validation split, is the caveat above made concrete: it is the same
        mechanism by which a hallucinating model will score *better* here for
        inventing texture. Read LPIPS next to the uncertainty map, never alone,
        and never let it decide a ranking.
    """
    sr_t, hr_t, batched = _check_pair(sr, hr)
    peak = _check_data_range(data_range)

    indices = [int(i) for i in rgb_indices]
    if len(indices) != 3:
        raise ValueError(
            "rgb_indices must give exactly three channels (R, G, B); got "
            f"{list(rgb_indices)}."
        )
    channels = sr_t.shape[1]
    out_of_range = [i for i in indices if not 0 <= i < channels]
    if out_of_range:
        raise ValueError(
            f"rgb_indices {out_of_range} are outside the {channels} channels of "
            "the input. Resolve them with rgb_band_indices(cfg) so they follow "
            "cfg.dataset.bands."
        )

    sr_rgb = sr_t[:, indices] / peak
    hr_rgb = hr_t[:, indices] / peak

    if clip_reflectance:
        # THE ONE CLIP IN THIS MODULE. Explicit, configured, counted, and applied
        # to a copy that never leaves this function. See step 3 above.
        outside = ((sr_rgb < 0.0) | (sr_rgb > 1.0)).sum(dim=(1, 2, 3)) + (
            (hr_rgb < 0.0) | (hr_rgb > 1.0)
        ).sum(dim=(1, 2, 3))
        total = 2 * int(sr_rgb[0].numel())
        clipped_fraction = (outside.to(torch.float64) / float(total)).numpy()
        sr_rgb = sr_rgb.clamp(0.0, 1.0)
        hr_rgb = hr_rgb.clamp(0.0, 1.0)
    else:
        clipped_fraction = np.zeros(sr_rgb.shape[0], dtype=np.float64)

    model = _load_lpips(net)
    with torch.no_grad():
        distance = model(
            (sr_rgb * 2.0 - 1.0).to(torch.float32),
            (hr_rgb * 2.0 - 1.0).to(torch.float32),
        )
    values = distance.reshape(-1).to(torch.float64).numpy()

    return LPIPSResult(
        distance=_unbatch(values, batched),
        clipped_fraction=_unbatch(clipped_fraction, batched),
        caveat=LPIPS_CAVEAT,
    )


# -- SAM -------------------------------------------------------------------


def sam(sr: Any, hr: Any, zero_vector_policy: str = "nan") -> SAMResult:
    r"""Spectral angle mapper, in degrees, per pixel.

    **This is the metric the spectral-consistency contribution is judged on.**
    Every other metric here is dominated by brightness error; SAM is the one that
    is blind to it. It measures the angle between the two spectra at a pixel,
    treating each as a vector in band space, so scaling a spectrum by any
    positive factor leaves SAM unchanged. A model that reconstructs the *shape*
    of the reflectance spectrum -- vegetation staying vegetation, water staying
    water -- scores well even where its absolute level drifts; a model that
    invents colour scores badly even where its PSNR looks fine.

    Definition, for a pixel with SR spectrum :math:`s` and reference :math:`r`
    over the ``C`` bands:

    .. math::

        \mathrm{SAM} = \arccos\left(
            \frac{\langle s, r \rangle}{\lVert s \rVert \lVert r \rVert}
        \right)

    reported in degrees, so it lies in ``[0, 180]`` -- and in ``[0, 90]`` for
    non-negative reflectance, since two vectors in the non-negative orthant
    cannot be more than a right angle apart.

    Three numerical decisions, all deliberate:

    - The cosine is clamped to ``[-1, 1]``. That is a guard against float
      rounding pushing a genuinely parallel pair to 1.0000000002 and making
      ``arccos`` return NaN. It is not a radiometric clip and cannot alter a real
      angle.
    - No epsilon is added to the denominator. An epsilon would silently turn an
      undefined angle into a small one; instead, zero-norm pixels are handled by
      an explicit policy and counted.
    - Accumulation is float64, so a spectrum against itself returns 0 to within
      about ``1e-6`` degrees rather than to within float32 noise.

    Args:
        sr: ``(C, H, W)`` or ``(B, C, H, W)`` float32 surface reflectance,
            nominally ``[0, 1]``, unclipped, in ``cfg.dataset.bands`` order. At
            least 2 bands.
        hr: Reference, same shape, dtype, band order and units.
        zero_vector_policy: What to do where either spectrum is exactly the zero
            vector, which makes the angle undefined (this happens on nodata that
            was filled with ``cfg.dataset.nodata_fill = 0.0``). ``"nan"``
            (default) writes ``nan`` into the map, excludes those pixels from
            ``mean_deg``, and counts them in ``num_undefined``. ``"error"``
            raises. There is deliberately no option that quietly calls them 0
            degrees, because that reports perfect spectral fidelity over nodata.

    Returns:
        :class:`SAMResult` -- ``mean_deg`` (float / ``(B,)``, degrees in
        ``[0, 90]`` for non-negative reflectance), ``map_deg`` (``(H, W)`` /
        ``(B, H, W)`` float64 degrees, ``nan`` where undefined), and
        ``num_undefined`` (int / ``(B,)``).

    Raises:
        ValueError: Shapes disagree, an input is non-finite, the input has fewer
            than 2 bands, ``zero_vector_policy`` is unknown, or the policy is
            ``"error"`` and a zero spectrum was found.

    Good values (MEASURED on SEN2NAIPv2, x4):
        Bicubic scores 2.09 degrees mean, with a p95 across tiles of 4.39 and a
        worst tile at 10.15; pixel replication scores 2.24. Within a single tile
        the tail is far heavier than the mean suggests -- bicubic's mean
        per-tile 95th-percentile angle is 5.93 degrees and reaches 65 on the
        worst tile, concentrated on field boundaries, roads, and built-up edges
        (visible in ``outputs/figures/baseline_qualitative.png``). Below 1 degree
        mean is the target for a spectrally consistent model; above 5 degrees the
        output is a different land cover from the target, whatever its PSNR says.
        Always report the mean alongside the per-pixel 95th percentile -- a
        1 degree mean with a 12 degree tail is a model that is wrong exactly
        where it matters.
    """
    sr_t, hr_t, batched = _check_pair(sr, hr)

    if sr_t.shape[1] < 2:
        raise ValueError(
            f"SAM needs at least 2 bands to define a spectrum; got "
            f"{sr_t.shape[1]}. With one band the angle is 0 everywhere by "
            "construction and the metric is meaningless."
        )

    policy = str(zero_vector_policy)
    if policy not in {"nan", "error"}:
        raise ValueError(
            f"zero_vector_policy must be 'nan' or 'error'; got {policy!r}. There "
            "is deliberately no policy that reports an undefined angle as 0 "
            "degrees."
        )

    dot = (sr_t * hr_t).sum(dim=1)  # (B, H, W)
    sr_norm = sr_t.pow(2).sum(dim=1).sqrt()
    hr_norm = hr_t.pow(2).sum(dim=1).sqrt()
    denominator = sr_norm * hr_norm

    undefined = denominator == 0.0
    num_undefined = undefined.sum(dim=(1, 2)).numpy().astype(np.int64)

    if policy == "error" and bool(undefined.any()):
        raise ValueError(
            f"{int(undefined.sum())} pixel(s) have an all-zero spectrum in sr or "
            "hr, so the spectral angle is undefined there. This is normally "
            "nodata filled with cfg.dataset.nodata_fill=0.0. Set "
            "metrics.sam.zero_vector_policy='nan' to exclude and count them."
        )

    safe = torch.where(undefined, torch.ones_like(denominator), denominator)
    cosine = (dot / safe).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(cosine))
    angles = torch.where(undefined, torch.full_like(angles, float("nan")), angles)

    maps = angles.numpy()
    with np.errstate(invalid="ignore"):
        means = np.nanmean(maps.reshape(maps.shape[0], -1), axis=1)

    if batched:
        return SAMResult(mean_deg=means, map_deg=maps, num_undefined=num_undefined)
    return SAMResult(
        mean_deg=float(means[0]),
        map_deg=maps[0],
        num_undefined=int(num_undefined[0]),
    )


# -- ERGAS -----------------------------------------------------------------


def ergas(
    sr: Any, hr: Any, scale: int, zero_mean_policy: str = "error"
) -> Union[float, np.ndarray]:
    r"""ERGAS -- *erreur relative globale adimensionnelle de synthese*.

    The standard remote-sensing composite: a band-averaged, mean-normalised RMSE
    scaled by the resolution ratio, giving one number comparable across scenes of
    different brightness and across different scale factors. Unlike PSNR it is
    relative to each band's own mean, so a dark band (blue over vegetation) and a
    bright one (NIR) contribute on equal terms instead of the bright one
    dominating.

    .. math::

        \mathrm{ERGAS} = 100 \cdot \frac{h}{l} \cdot \sqrt{
            \frac{1}{C} \sum_{b=1}^{C}
            \left( \frac{\mathrm{RMSE}_b}{\mu_b} \right)^2 }

    where :math:`h/l` is the ratio of high-resolution to low-resolution pixel
    size -- ``1 / scale`` for super-resolution, i.e. ``2.5 m / 10 m`` at
    ``cfg.sr.scale = 4`` -- and :math:`\mu_b` is the mean of the **reference**
    band. Lower is better; 0 is a perfect reconstruction.

    Args:
        sr: ``(C, H, W)`` or ``(B, C, H, W)`` float32 surface reflectance,
            nominally ``[0, 1]``, unclipped.
        hr: Reference, same shape, dtype, band order and units. The band means
            come from here, not from ``sr``: normalising by the prediction's own
            mean would let a model improve its score by darkening its output.
        scale: Super-resolution factor, from ``cfg.sr.scale``. Enters as
            ``h/l = 1/scale``, so passing the wrong value rescales every ERGAS
            number linearly.
        zero_mean_policy: What to do when a reference band's mean is 0, which
            makes the relative error undefined. ``"error"`` (default) raises -- an
            all-zero band in a validation patch means the patch is nodata and
            should have been filtered, not scored. ``"nan"`` returns ``nan`` for
            that sample, to be counted by the aggregator.

    Returns:
        A ``float`` for a single sample, ``(B,)`` float64 for a batch.
        Dimensionless and non-negative; 0 for an identical pair.

    Raises:
        ValueError: Shapes disagree, an input is non-finite, ``scale`` is not a
            positive integer, ``zero_mean_policy`` is unknown, or the policy is
            ``"error"`` and a reference band mean is 0.

    Good values (MEASURED on SEN2NAIPv2, x4):
        Bicubic scores 3.05 mean, p95 6.07, worst tile 11.77; pixel replication
        scores 3.28. The classical pansharpening reading is ERGAS < 3 for a good
        fusion at x4, so bicubic sits just the wrong side of that line and its
        p95 is double it. Because it is normalised per band, ERGAS is the metric most
        sensitive to a band-dependent radiometric bias -- if ERGAS is poor while
        PSNR looks acceptable, check for a per-band scaling error before blaming
        the architecture.
    """
    sr_t, hr_t, batched = _check_pair(sr, hr)

    factor = int(scale)
    if factor <= 0:
        raise ValueError(
            f"scale must be a positive integer (cfg.sr.scale); got {scale!r}."
        )

    policy = str(zero_mean_policy)
    if policy not in {"error", "nan"}:
        raise ValueError(f"zero_mean_policy must be 'error' or 'nan'; got {policy!r}.")

    mse = ((sr_t - hr_t) ** 2).mean(dim=(-2, -1))  # (B, C)
    means = hr_t.mean(dim=(-2, -1))  # (B, C)

    zero = means == 0.0
    if bool(zero.any()):
        if policy == "error":
            offenders = torch.nonzero(zero).tolist()
            raise ValueError(
                "ERGAS is undefined: the reference band mean is exactly 0 for "
                f"(sample, band) {offenders[:8]}"
                f"{' ...' if len(offenders) > 8 else ''}. An all-zero reference "
                "band means the patch is nodata and should have been rejected by "
                "cfg.patches.filters, not scored. Set "
                "metrics.ergas.zero_mean_policy='nan' to score the rest of the "
                "set and have the aggregator count these."
            )
        means = torch.where(zero, torch.full_like(means, float("nan")), means)

    relative = mse.sqrt() / means
    values = (100.0 / factor) * relative.pow(2).mean(dim=1).sqrt()
    array = values.numpy()

    return array if batched else float(array[0])
