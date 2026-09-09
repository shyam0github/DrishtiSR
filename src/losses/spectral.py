"""Spectral-consistency loss: degrading the SR output must reproduce the input.

THE CONTRIBUTION, STATED AS AN EQUATION. A super-resolved image is only
physically admissible if it still says what the sensor said. Sentinel-2 measured
surface reflectance on a 10 m grid; whatever the network invents at 2.5 m must
average back to that measurement. So, with ``D`` an antialiased x4 downsampler::

    L_spec = lambda1 * L1(x_LR, D(y_SR)) + lambda2 * SAM(x_LR, D(y_SR))

The L1 term holds the *level* -- total reflected energy per 10 m cell -- and the
SAM term holds the *shape* of the spectrum across bands, which L1 barely
constrains because it is dominated by whichever band is brightest. A model can
satisfy L1 by getting the mean right while turning vegetation into bare soil;
SAM is the term that forbids it. See :func:`src.metrics.image_quality.sam` for
the same angle used as a reported metric.

-- THE DOWNSAMPLER IS THE LOAD-BEARING PART -------------------------------

``D`` MUST be antialiased. With stride-4 slicing (``y[..., ::4, ::4]``) the loss
constrains one HR pixel in sixteen and says nothing whatever about the other
fifteen: a network could paint noise into them and score a perfect spectral
term. The entire consistency claim would be vacuous, and it would be vacuous
*invisibly*, because the reported number would look excellent.

``mode="area"`` -- an exact 4x4 block mean at integer scale -- is the default and
is the operator the claim is actually about ("the 2.5 m pixels inside a 10 m
cell average to what the satellite recorded"). ``bicubic_antialias`` and
``bilinear_antialias`` are accepted alternatives; ``nearest`` and
non-antialiased ``bicubic`` are REJECTED by name in :func:`down4` rather than
merely discouraged. ``tests/test_spectral_loss.py`` puts a checkerboard through
both and asserts the antialiased path flattens it while stride-4 slicing
preserves it in full.

-- REFLECTANCE SPACE, NOT NORMALISED SPACE --------------------------------

SAM is the angle between two vectors in band space. It is invariant to positive
scaling but NOT to translation, so subtracting a per-band mean -- what any
standardising normalisation does -- turns the angle into something with no
physical meaning: two spectra 2 degrees apart in reflectance can be 40 degrees
apart after a shift, and the number would still be called "SAM" in the logs.

This pipeline never normalises (``configs/frozen_day3.yaml``, ``normalization:
kind: none``; AGENTS.md section 1.3), so both arguments arrive as reflectance
already and ``denorm=None`` is the correct and only used setting. The ``denorm``
argument exists so that if a future change ever puts a normalisation in front of
the network, the fix is to pass its statistics here -- one argument -- rather
than to discover months later that the spectral term was measuring nothing.

Effective lambda scaling, since it decides whether a lambda transfers: L1 is
scale-equivariant, so an L1 computed on data divided by a per-band std ``s`` is
``1/s`` times the reflectance-domain L1, and a lambda tuned in one space is
wrong by that factor in the other. Both terms here are therefore computed in
reflectance, matching the main ``L1(sr, hr)`` reconstruction term, which is also
in reflectance -- so lambda1 is directly interpretable as "this many times the
main loss's units".

MEASURED magnitudes, 2026-09-09, and they are what the lambda decision rests on:

===========================================  ==================
main L1(sr, hr), Run A at convergence        0.008784
spectral L1 floor, 1199 val patches          0.005733 +- 0.003528
spectral SAM floor, same patches             0.022249 +- 0.012147 rad (1.275 deg)
===========================================  ==================

The floor is the value ``D(ground-truth HR)`` vs ``x_LR`` already costs, from
``scripts/spectral_floor.py``; it is a property of the dataset (Sentinel-2 LR
against NAIP-derived HR, two point-spread functions) and NO model can go below
it while remaining faithful to the HR. At lambda1=0.5, lambda2=0.1 the two
weighted terms come to 0.00287 + 0.00222 = 0.00509 -- 58% of the reconstruction
loss, i.e. more than a third of the total objective would be a constant nothing
can remove. That is why the floor is printed before a lambda is chosen, and why
0.5 is a candidate in ``cfg.loss.spectral`` rather than a default here.

The failure mode the ratio guards against: the spectral term ALONE is minimised
by an output that averages exactly to the LR, which a blurry bicubic-like
upsample already does -- scoring near 0, far below the 0.0057 an HR-faithful
output pays. The term therefore pulls towards doing no super-resolution at all,
and lambda1 sets how hard it pulls.

-- NUMERICS ---------------------------------------------------------------

Every one of these guards is about gradients, not about the forward value:

- Norms are ``sqrt(sum(x^2) + eps^2)``, not ``sqrt(sum(x^2)) + eps``. The
  derivative of ``sqrt`` at 0 is infinite, so a dead pixel would emit NaN into
  the backward pass even though the forward value looked fine and even though
  the pixel is later masked out -- masking with ``torch.where`` does not undo a
  NaN that was already created. Adding ``eps^2`` inside moves a unit-norm vector
  by about 5e-17 and makes the derivative finite everywhere.
- The cosine is clamped to ``[-1 + cos_clamp, 1 - cos_clamp]``. ``arccos`` has
  infinite derivative at +-1, which is exactly where a well-trained model sits.
  CONSEQUENCE, and it is not zero: the SAM term can never reach 0. Its floor is
  ``arccos(1 - 1e-7) = 4.47e-4`` rad analytically and 4.88e-4 rad in float32
  (1 - 1e-7 is not representable; float32 eps is 1.19e-7), which is 2.2% of the
  0.0222 rad physical floor. It applies identically to the training loss and to
  ``scripts/spectral_floor.py``, so the two remain comparable.
- Pixels whose LR or SR spectrum has norm below ``min_norm`` are excluded from
  the SAM mean and counted. Their angle is undefined (nodata filled with
  ``cfg.dataset.nodata_fill = 0.0`` is the usual source), and there is
  deliberately no option that reports an undefined angle as 0 degrees -- that
  would announce perfect spectral fidelity over nodata. The excluded fraction is
  returned as ``sam_valid_frac`` so it is logged rather than assumed to be 1.

``lr`` is detached inside :func:`spectral_terms`. It is a measurement, not a
prediction; a gradient flowing into it is meaningless, and if the same tensor
were ever also an input to the network it would be a silent second gradient
path.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "ANTIALIASED_MODES",
    "REJECTED_MODES",
    "DEFAULT_EPS",
    "DEFAULT_COS_CLAMP",
    "DEFAULT_MIN_NORM",
    "DEFAULT_DOWNSAMPLE",
    "DEFAULT_LAMBDA1",
    "DEFAULT_LAMBDA2",
    "down4",
    "spectral_terms",
    "spectral_consistency",
    "spectral_settings_from_cfg",
]

# Downsamplers that low-pass before decimating. Anything not in this set cannot
# be used as D: see the module docstring, and tests/test_spectral_loss.py.
ANTIALIASED_MODES: Tuple[str, ...] = (
    "area",
    "bicubic_antialias",
    "bilinear_antialias",
)

# Named explicitly so the error message can say WHY, rather than "unknown mode".
REJECTED_MODES: Tuple[str, ...] = (
    "nearest",
    "bicubic",
    "bilinear",
    "stride",
    "subsample",
)

DEFAULT_EPS = 1e-8
DEFAULT_COS_CLAMP = 1e-7
DEFAULT_MIN_NORM = 1e-6
DEFAULT_DOWNSAMPLE = "area"

# The reference weights, mirrored by cfg.loss.spectral.lambda1/lambda2 in
# configs/base.yaml. They are NOT src/train.py's defaults: the trainer defaults
# both to 0.0 so that the control run cannot acquire a spectral term by
# forgetting a flag. A run opts in explicitly.
DEFAULT_LAMBDA1 = 0.5
DEFAULT_LAMBDA2 = 0.1


def _check_4d(name: str, tensor: Any) -> torch.Tensor:
    """Require a 4-D ``(B, C, H, W)`` float tensor and return it.

    Args:
        name: Argument name, for the error message.
        tensor: The candidate.

    Returns:
        The tensor unchanged.

    Raises:
        TypeError: Not a ``torch.Tensor``.
        ValueError: Not 4-D. A 3-D ``(C, H, W)`` array is rejected rather than
            unsqueezed, because this module must never guess an axis order --
            ``(C, H, W)`` and a batch of single-band ``(B, H, W)`` planes are
            indistinguishable by shape alone.
    """
    if not torch.is_tensor(tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(tensor).__name__}. This "
            "is a differentiable loss; convert with torch.as_tensor at the "
            "caller so it is clear where the graph starts."
        )
    if tensor.ndim != 4:
        raise ValueError(
            f"{name} must be (B, C, H, W); got shape {tuple(tensor.shape)}. A "
            "3-D array is not unsqueezed here -- (C, H, W) and (B, H, W) are "
            "the same shape and guessing between them silently transposes the "
            "band axis. Add the batch axis at the caller."
        )
    return tensor


def down4(
    x: torch.Tensor,
    scale: int = 4,
    mode: str = DEFAULT_DOWNSAMPLE,
) -> torch.Tensor:
    """Antialiased downsample by an integer factor. The heart of the loss.

    Args:
        x: ``(B, C, H, W)`` float32 surface reflectance, nominally ``[0, 1]``
            and **unclipped** -- cloud, snow and bright roofs exceed 1.0 and are
            not touched here. ``H`` and ``W`` must be divisible by ``scale``.
            Differentiable: gradients flow to ``x``.
        scale: Integer downsampling factor, ``cfg.sr.scale`` (4 for this
            project). The name ``down4`` reflects the project's x4 task; the
            factor is still an argument, never a literal in the body.
        mode: One of :data:`ANTIALIASED_MODES`. ``"area"`` (default) is an exact
            block mean at integer scale and is the operator the spectral claim
            is about.

    Returns:
        ``(B, C, H // scale, W // scale)``, same dtype and device as ``x``,
        surface reflectance in the same units and band order. **Nothing is
        clipped**: an interpolating kernel may overshoot past the input range
        and that overshoot is information about the kernel, not an error.

    Raises:
        TypeError: ``x`` is not a tensor.
        ValueError: ``x`` is not 4-D; ``scale`` is not a positive integer; the
            spatial dimensions are not divisible by ``scale``; or ``mode`` is
            not antialiased. The last case names the failure explicitly --
            stride-4 slicing or ``nearest`` would let the network satisfy the
            loss on one pixel in sixteen and leave the other fifteen
            unconstrained, which makes the reported spectral number meaningless
            while it still looks excellent.
    """
    _check_4d("x", x)
    if int(scale) != scale or scale < 1:
        raise ValueError(f"scale must be a positive integer; got {scale!r}.")
    scale = int(scale)

    mode = str(mode)
    if mode not in ANTIALIASED_MODES:
        hint = (
            " That mode decimates without low-pass filtering, so the loss would "
            "constrain one pixel in scale**2 and say nothing about the rest -- "
            "the consistency claim would be vacuous. See "
            "tests/test_spectral_loss.py."
            if mode in REJECTED_MODES
            else ""
        )
        raise ValueError(
            f"down4 mode must be one of {list(ANTIALIASED_MODES)}; got "
            f"{mode!r}.{hint}"
        )

    height, width = int(x.shape[-2]), int(x.shape[-1])
    if height % scale or width % scale:
        raise ValueError(
            f"spatial dims {(height, width)} are not divisible by scale={scale}, "
            "so no kernel produces an exact LR grid. Crop at the caller; this "
            "function will not pad or round, because either would shift the "
            "grid the reflectance comparison is defined on."
        )
    target = (height // scale, width // scale)

    if mode == "area":
        return F.interpolate(x, size=target, mode="area")
    if mode == "bicubic_antialias":
        return F.interpolate(
            x, size=target, mode="bicubic", align_corners=False, antialias=True
        )
    return F.interpolate(
        x, size=target, mode="bilinear", align_corners=False, antialias=True
    )


def _to_reflectance(
    name: str,
    x: torch.Tensor,
    denorm: Optional[Any],
) -> torch.Tensor:
    """Undo an input normalisation so the tensor is reflectance again.

    Args:
        name: Argument name, for error messages.
        x: ``(B, C, H, W)`` tensor, normalised if ``denorm`` is given.
        denorm: ``None`` when the pipeline feeds reflectance directly -- the
            case for this project, see the module docstring -- or a
            ``(mean, std)`` pair of per-band sequences of length ``C``, or a
            mapping with ``"mean"`` and ``"std"`` keys. The inverse applied is
            ``x * std + mean``.

    Returns:
        ``(B, C, H, W)`` surface reflectance, differentiably.

    Raises:
        ValueError: ``denorm`` is malformed, its lengths do not match ``C``, or
            any std is non-positive (which would be a division-by-zero
            normalisation and cannot be inverted).
    """
    if denorm is None:
        return x

    if isinstance(denorm, Mapping):
        try:
            mean_raw, std_raw = denorm["mean"], denorm["std"]
        except KeyError as exc:
            raise ValueError(
                f"denorm mapping for {name} must have 'mean' and 'std' keys; "
                f"got keys {sorted(denorm)}."
            ) from exc
    else:
        pair = tuple(denorm)
        if len(pair) != 2:
            raise ValueError(
                f"denorm for {name} must be a (mean, std) pair or a mapping "
                f"with 'mean'/'std'; got {len(pair)} items."
            )
        mean_raw, std_raw = pair

    channels = int(x.shape[1])
    mean = torch.as_tensor(mean_raw, dtype=x.dtype, device=x.device).reshape(-1)
    std = torch.as_tensor(std_raw, dtype=x.dtype, device=x.device).reshape(-1)
    if mean.numel() != channels or std.numel() != channels:
        raise ValueError(
            f"denorm for {name} has {mean.numel()} means and {std.numel()} "
            f"stds but the tensor has {channels} bands. Per-band statistics "
            "must match cfg.dataset.bands exactly; a broadcast scalar would "
            "quietly apply one band's statistics to all of them."
        )
    if bool((std <= 0).any()):
        raise ValueError(
            f"denorm std for {name} contains a non-positive value "
            f"({std.tolist()}). A normalisation with a zero or negative std is "
            "not invertible."
        )
    shape = (1, channels, 1, 1)
    return x * std.reshape(shape) + mean.reshape(shape)


def spectral_terms(
    sr: torch.Tensor,
    lr: torch.Tensor,
    scale: Optional[int] = None,
    mode: str = DEFAULT_DOWNSAMPLE,
    eps: float = DEFAULT_EPS,
    cos_clamp: float = DEFAULT_COS_CLAMP,
    min_norm: float = DEFAULT_MIN_NORM,
    denorm: Optional[Any] = None,
    per_sample: bool = False,
) -> dict:
    """The two spectral terms, unweighted. Both in reflectance units.

    Downsamples ``sr`` to the LR grid with :func:`down4` and compares it against
    ``lr``: mean absolute reflectance error, and mean spectral angle in radians.
    Splitting this out of :func:`spectral_consistency` is what lets
    ``scripts/spectral_floor.py`` measure the same quantities on ground-truth HR
    with the identical numerics, so the floor and the training term are directly
    comparable rather than two similar-looking numbers.

    Args:
        sr: ``(B, C, H, W)`` float, the network output, surface reflectance
            (nominally ``[0, 1]``, unclipped) unless ``denorm`` is given.
            Gradients flow to this argument.
        lr: ``(B, C, H // scale, W // scale)`` float, the Sentinel-2 input in
            the same units and band order. **Detached inside**: it is a
            measurement, and a gradient into it would be meaningless.
        scale: Integer downsampling factor. ``None`` (default) derives it from
            ``sr.shape[-2] // lr.shape[-2]`` and verifies both axes agree,
            which keeps a config's scale and the tensors that arrived from
            disagreeing silently.
        mode: Downsampler, one of :data:`ANTIALIASED_MODES`.
        eps: Added *inside* the norm square root as ``eps**2``. See the module
            docstring: this is a gradient guard, not a value fudge.
        cos_clamp: The cosine is clamped to ``[-1 + cos_clamp, 1 - cos_clamp]``
            before ``arccos``. Puts a floor of ``arccos(1 - cos_clamp)`` rad on
            the SAM term -- 4.88e-4 rad in float32 at the default, against a
            0.0222 rad physical floor.
        min_norm: Pixels whose SR or LR spectrum has norm below this, in
            reflectance units, are excluded from the SAM mean and counted. The
            default 1e-6 is four orders of magnitude below any real
            land-surface spectrum and corresponds to a digital number of 0.01 at
            ``cfg.dataset.reflectance_scale = 10000``, so it catches exact
            nodata fill and nothing else.
        denorm: ``None`` when ``sr`` and ``lr`` are already reflectance -- the
            only case in this project -- else the ``(mean, std)`` of the input
            normalisation, which is undone on both arguments before either term
            is computed.
        per_sample: ``False`` (default) reduces over the whole batch and returns
            0-d tensors. ``True`` returns ``(B,)`` tensors, which is what a
            floor or an ablation needs to report a standard deviation across
            patches.

    Returns:
        ``{"l1_spec": tensor, "sam": tensor, "sam_valid_frac": tensor}``.
        ``l1_spec`` is mean absolute error in **reflectance units**; ``sam`` is
        the mean spectral angle in **radians** over valid pixels (multiply by
        ``180 / pi`` for the degrees :func:`src.metrics.image_quality.sam`
        reports); ``sam_valid_frac`` is the fraction of pixels that entered that
        mean. All three are 0-d tensors unless ``per_sample``, in which case all
        three are ``(B,)``. ``l1_spec`` and ``sam`` carry gradients to ``sr``;
        ``sam_valid_frac`` does not.

    Raises:
        TypeError: An argument is not a tensor.
        ValueError: Shapes are not 4-D, batch sizes or band counts disagree,
            ``lr``'s spatial size is not ``sr``'s divided by an integer factor,
            the derived factors differ between the two axes, ``sr`` has fewer
            than 2 bands (an angle needs a spectrum), ``mode`` is not
            antialiased, or an input is non-finite. A non-finite input raises
            rather than being nan-masked: it means something upstream already
            diverged, and averaging around it hides the divergence for another
            few thousand iterations.
    """
    _check_4d("sr", sr)
    _check_4d("lr", lr)

    if sr.shape[0] != lr.shape[0] or sr.shape[1] != lr.shape[1]:
        raise ValueError(
            f"sr {tuple(sr.shape)} and lr {tuple(lr.shape)} must share batch "
            "and band axes; they are the same scene at two resolutions."
        )
    if int(sr.shape[1]) < 2:
        raise ValueError(
            f"the spectral angle needs at least 2 bands; got {int(sr.shape[1])}. "
            "With one band the angle is 0 everywhere by construction and the "
            "SAM term would silently contribute nothing."
        )

    factors = []
    for axis, name in ((-2, "height"), (-1, "width")):
        hi, lo = int(sr.shape[axis]), int(lr.shape[axis])
        if lo <= 0 or hi % lo:
            raise ValueError(
                f"sr {name} {hi} is not an integer multiple of lr {name} {lo}; "
                "the two grids cannot be compared without resampling lr, which "
                "this loss will not do -- resampling the measurement is exactly "
                "the error the loss exists to detect."
            )
        factors.append(hi // lo)
    if factors[0] != factors[1]:
        raise ValueError(
            f"sr/lr size ratios differ between axes: {factors[0]} vertically, "
            f"{factors[1]} horizontally. One of the tensors is transposed or "
            "cropped."
        )
    derived = factors[0]
    if scale is not None and int(scale) != derived:
        raise ValueError(
            f"scale={int(scale)} was passed but sr {tuple(sr.shape[-2:])} over "
            f"lr {tuple(lr.shape[-2:])} is a factor of {derived}. The config "
            "and the data disagree; fix the caller rather than trusting either."
        )

    # Detached here, once, and before anything else touches it.
    lr = lr.detach()

    if not bool(torch.isfinite(sr).all()):
        raise ValueError(
            "sr contains non-finite values. The network has already diverged; "
            "this loss will not mask it, because a masked NaN buys a few "
            "thousand more iterations of a run that is already lost."
        )
    if not bool(torch.isfinite(lr).all()):
        raise ValueError(
            "lr contains non-finite values. Reflectance from the dataset must "
            "be finite; check cfg.dataset.nodata_fill and the cache."
        )

    sr_ref = _to_reflectance("sr", sr, denorm)
    lr_ref = _to_reflectance("lr", lr, denorm)

    down = down4(sr_ref, scale=derived, mode=mode)

    # -- L1, in reflectance units ------------------------------------------
    abs_err = (down - lr_ref).abs()
    l1_spec = abs_err.mean(dim=(1, 2, 3)) if per_sample else abs_err.mean()

    # -- SAM, in radians ---------------------------------------------------
    eps_sq = float(eps) ** 2
    dot = (down * lr_ref).sum(dim=1)                            # (B, H, W)
    down_norm = torch.sqrt(down.pow(2).sum(dim=1) + eps_sq)     # finite grad at 0
    lr_norm = torch.sqrt(lr_ref.pow(2).sum(dim=1) + eps_sq)

    valid = (down_norm > float(min_norm)) & (lr_norm > float(min_norm))
    limit = 1.0 - float(cos_clamp)
    cosine = (dot / (down_norm * lr_norm)).clamp(-limit, limit)
    angle = torch.arccos(cosine)                                # radians

    zero = torch.zeros((), dtype=angle.dtype, device=angle.device)
    masked = torch.where(valid, angle, zero)
    counts = valid.sum(dim=(1, 2)) if per_sample else valid.sum()
    per_item = int(valid.shape[1]) * int(valid.shape[2])
    total = per_item if per_sample else per_item * int(valid.shape[0])
    # clamp(min=1) so an all-nodata patch divides by 1 instead of 0. It
    # contributes 0 to the loss AND reports sam_valid_frac 0.0, which is what
    # the trainer logs -- a zero angle beside a zero valid fraction reads as
    # "nothing was measured", where a bare 0.0 would read as "perfect".
    denominator = counts.clamp(min=1).to(masked.dtype)
    summed = masked.sum(dim=(1, 2)) if per_sample else masked.sum()
    sam = summed / denominator
    valid_frac = counts.detach().to(masked.dtype) / float(total)

    return {"l1_spec": l1_spec, "sam": sam, "sam_valid_frac": valid_frac}


def spectral_consistency(
    sr: torch.Tensor,
    lr: torch.Tensor,
    lam1: float = DEFAULT_LAMBDA1,
    lam2: float = DEFAULT_LAMBDA2,
    **kwargs: Any,
) -> Tuple[torch.Tensor, dict]:
    """Weighted spectral-consistency loss and its unweighted components.

    ``total = lam1 * L1(lr, down4(sr)) + lam2 * SAM(lr, down4(sr))``

    Args:
        sr: ``(B, C, H, W)`` float surface reflectance, the network output.
            Gradients flow here.
        lr: ``(B, C, H // scale, W // scale)`` float surface reflectance, the
            Sentinel-2 input. Detached inside.
        lam1: Weight on the reflectance-L1 term. Reference value
            :data:`DEFAULT_LAMBDA1`; ``src/train.py`` defaults it to **0.0** so
            a control run cannot pick up a spectral term by omission.
        lam2: Weight on the spectral-angle term (radians). Reference value
            :data:`DEFAULT_LAMBDA2`.
        **kwargs: Passed to :func:`spectral_terms` -- ``scale``, ``mode``,
            ``eps``, ``cos_clamp``, ``min_norm``, ``denorm``. ``per_sample`` is
            rejected: a weighted loss must be a scalar to call ``.backward()``.

    Returns:
        ``(total, {"l1_spec": float, "sam": float, "sam_valid_frac": float})``.
        ``total`` is a 0-d tensor carrying gradients to ``sr``. The dict holds
        plain Python floats of the **unweighted** terms, detached, so a trainer
        can log them without holding the graph alive and so the logged numbers
        stay comparable across runs with different lambdas -- and comparable
        against the floor printed by ``scripts/spectral_floor.py``. Logging the
        weighted terms instead would make the control (lambdas 0.0) log two
        zeros and prove nothing.

    Raises:
        TypeError: ``per_sample`` was passed.
        ValueError: A lambda is negative, or as :func:`spectral_terms`.

    Example:
        >>> import torch
        >>> sr = torch.rand(2, 4, 32, 32, requires_grad=True)
        >>> lr = torch.rand(2, 4, 8, 8)
        >>> total, parts = spectral_consistency(sr, lr, lam1=0.5, lam2=0.1)
        >>> total.backward()
        >>> sorted(parts)
        ['l1_spec', 'sam', 'sam_valid_frac']
    """
    if "per_sample" in kwargs:
        raise TypeError(
            "spectral_consistency does not take per_sample: the weighted total "
            "must be a scalar to backward from. Call spectral_terms directly "
            "for per-patch values."
        )
    for name, value in (("lam1", lam1), ("lam2", lam2)):
        if float(value) < 0.0:
            raise ValueError(
                f"{name} must be >= 0; got {value!r}. A negative weight rewards "
                "spectral inconsistency."
            )

    parts = spectral_terms(sr, lr, per_sample=False, **kwargs)
    total = float(lam1) * parts["l1_spec"] + float(lam2) * parts["sam"]
    scalars = {key: float(value.detach()) for key, value in parts.items()}
    return total, scalars


def spectral_settings_from_cfg(cfg: Any) -> dict:
    """Read the numerics of the loss out of ``cfg.loss.spectral``.

    Keeps ``scripts/spectral_floor.py`` and the trainer on the same settings
    without either of them restating a constant. AGENTS.md section 2: no
    hyperparameter lives in code.

    Args:
        cfg: Loaded config with a ``loss.spectral`` block
            (``configs/base.yaml``).

    Returns:
        ``{"mode": str, "eps": float, "cos_clamp": float, "min_norm": float}``
        -- exactly the keyword arguments :func:`spectral_terms` takes for its
        numerics, so it can be splatted in. The lambdas are deliberately NOT
        included: they are a per-run decision that arrives on the command line.

    Raises:
        KeyError: The config has no ``loss.spectral`` block, or it is missing a
            key. Not defaulted -- a run that silently used a different
            downsampler from the floor script would produce two numbers that
            look comparable and are not.
    """
    block = cfg["loss"]["spectral"]
    return {
        "mode": str(block["downsample"]),
        "eps": float(block["eps"]),
        "cos_clamp": float(block["cos_clamp"]),
        "min_norm": float(block["min_norm"]),
    }
