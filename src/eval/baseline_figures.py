"""The qualitative figure for a baseline or model evaluation.

Four panels per sample: the LR input (nearest-upsampled, so LR pixels stay
visibly blocky and are never mistaken for detail), the method's output, the HR
reference, and the per-pixel SAM map in degrees.

The SAM panel is the one that earns the figure. A side-by-side of SR and HR at
2.5 m is easy to look at and hard to read -- both are green, both are plausible,
and the eye forgives a great deal. The SAM map is not forgiving: it marks every
pixel where the reconstructed *spectrum* points the wrong way, which is exactly
the failure the spectral-consistency contribution exists to prevent and exactly
the failure a PSNR number cannot show. Its colour bar is shared across rows so
two samples can be compared directly, and the scale used is stated in the
caption.

Everything here is display code. The percentile stretch used to render an RGB
composite is applied to a **copy**, for the figure only: reflectance that reaches
a metric is never stretched, scaled, or clipped. SR and HR share one stretch,
derived from HR, so the comparison between the two panels is honest -- stretching
each to its own range would hide a systematic brightness bias.

Matplotlib runs on the non-interactive Agg backend: these must render on a
headless Kaggle notebook and on a local CPU box with no display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend switch)
import numpy as np  # noqa: E402

from src.eval.alignment_figures import (  # noqa: E402
    nearest_upsample,
    stretch_for_display,
)

__all__ = ["plot_baseline_qualitative"]


def _display_rgb_indices(cfg: Any) -> Optional[List[int]]:
    """Channel indices for the RGB preview, or None when the bands are absent.

    Args:
        cfg: The loaded config. Reads ``dataset.bands`` and
            ``baseline.display_rgb_bands``.

    Returns:
        Three channel indices in (R, G, B) order, or ``None`` when the dataset
        does not carry all three -- the caller then falls back to greyscale
        rather than compositing whatever happens to be in channels 0-2.
    """
    bands = [str(b) for b in cfg["dataset"]["bands"]]
    wanted = [str(b) for b in cfg["baseline"]["display_rgb_bands"]]
    if not all(band in bands for band in wanted):
        return None
    return [bands.index(band) for band in wanted]


def _wrap_id(sample_id: str, width: int = 26) -> str:
    """Break a long sample id onto several lines for a row label.

    Args:
        sample_id: The id, e.g.
            ``NA5120_E1186N0721__m_3812251_ne_10_060_20200602``.
        width: Maximum characters per line.

    Returns:
        The id with newlines inserted, split on underscores where possible so
        the tile key and the NAIP scene key stay readable.
    """
    parts = str(sample_id).split("_")
    lines, current = [], ""
    for part in parts:
        candidate = f"{current}_{part}" if current else part
        if len(candidate) > width and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _render(
    image: np.ndarray,
    rgb: Optional[Sequence[int]],
    percentiles: Tuple[float, float],
    reference: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Render one ``(C, H, W)`` reflectance array as RGB or greyscale.

    Args:
        image: ``(C, H, W)`` float32 surface reflectance, unclipped.
        rgb: Channel indices for the composite, or ``None`` for band 0 greyscale.
        percentiles: ``(low, high)`` display stretch, from
            ``cfg.baseline.display_percentiles``.
        reference: Array whose percentiles set the limits instead, so two panels
            share one stretch.

    Returns:
        ``(rendered, (low, high))`` -- an ``(H, W, 3)`` or ``(H, W)`` array in
        ``[0, 1]`` for rendering, and the stretch limits in reflectance units.
    """
    if rgb is not None:
        selected = np.stack([image[i] for i in rgb], axis=-1)
        ref = None if reference is None else np.stack([reference[i] for i in rgb], axis=-1)
    else:
        selected = image[0]
        ref = None if reference is None else reference[0]
    return stretch_for_display(selected, percentiles, reference=ref)


def plot_baseline_qualitative(
    samples: Sequence[Mapping[str, Any]],
    cfg: Any,
    path: Any,
    method: str = "baseline",
) -> Path:
    """Write the LR / SR / HR / SAM-map grid for a handful of samples.

    Args:
        samples: The ``qualitative`` list from
            :class:`src.metrics.aggregate.EvaluationResult`. Each entry needs
            ``sample_id`` (str), ``lr`` ``(C, h, w)``, ``sr`` and ``hr``
            ``(C, h*scale, w*scale)`` -- all float32 surface reflectance,
            nominally ``[0, 1]`` and unclipped -- and ``sam_map`` ``(H, W)``
            float64 degrees, which may contain ``nan`` where the spectral angle
            is undefined. ``psnr_mean`` and ``sam_mean_deg`` are used in the row
            labels when present.
        cfg: The loaded config. Reads ``sr.scale``, ``dataset.bands``,
            ``dataset.hr_gsd_m``, and the ``baseline`` display settings.
        path: Destination ``.png``. Parent directories are created.
        method: Name of the method whose output is in the ``sr`` panel, used in
            the column heading and the title.

    Returns:
        The path written.

    Raises:
        ValueError: ``samples`` is empty, or an entry is missing a required key.
            An empty figure is worse than no figure: it gets pasted into a report
            as evidence of something.
    """
    if not samples:
        raise ValueError(
            "plot_baseline_qualitative got no samples. Pass "
            "collect_samples=cfg.baseline.figure_samples to Evaluator.run(); a "
            "blank figure would still be written and later quoted as evidence."
        )
    for index, sample in enumerate(samples):
        missing = [k for k in ("lr", "sr", "hr", "sam_map") if k not in sample]
        if missing:
            raise ValueError(
                f"Qualitative sample {index} is missing {missing}."
            )

    scale = int(cfg["sr"]["scale"])
    hr_gsd = float(cfg["dataset"]["hr_gsd_m"])
    lr_gsd = float(cfg["dataset"]["lr_gsd_m"])
    percentiles = tuple(float(p) for p in cfg["baseline"]["display_percentiles"])
    dpi = int(cfg["baseline"]["figure_dpi"])
    rgb = _display_rgb_indices(cfg)
    band_label = (
        "/".join(str(b) for b in cfg["baseline"]["display_rgb_bands"])
        if rgb is not None
        else f"{cfg['dataset']['bands'][0]} (greyscale)"
    )

    # One colour scale for every SAM panel, so two rows can be compared. A
    # per-panel scale would make the worst sample look like the best one.
    configured_vmax = cfg["baseline"]["sam_vmax_deg"]
    if configured_vmax is None:
        finite = np.concatenate(
            [np.asarray(s["sam_map"])[np.isfinite(s["sam_map"])].ravel() for s in samples]
        )
        sam_vmax = float(np.percentile(finite, 99.0)) if finite.size else 1.0
        sam_vmax = max(sam_vmax, 1e-3)
        vmax_source = "99th percentile of the maps shown"
    else:
        sam_vmax = float(configured_vmax)
        vmax_source = "cfg.baseline.sam_vmax_deg"

    rows = len(samples)
    columns = ["LR (nearest x%d)" % scale, f"{method}", "HR reference", "SAM (degrees)"]
    figure, axes = plt.subplots(
        rows,
        4,
        figsize=(4 * 2.9 + 1.6, rows * 2.9 + 1.0),
        squeeze=False,
        layout="constrained",
    )

    sam_image = None
    for row, sample in enumerate(samples):
        lr = np.asarray(sample["lr"], dtype=np.float32)
        sr = np.asarray(sample["sr"], dtype=np.float32)
        hr = np.asarray(sample["hr"], dtype=np.float32)
        sam_map = np.asarray(sample["sam_map"], dtype=np.float64)

        # HR sets the stretch for all three imagery panels: the point of the
        # figure is whether SR matches HR, which a per-panel stretch would erase.
        lr_rendered, _ = _render(nearest_upsample(lr, scale), rgb, percentiles, hr)
        sr_rendered, _ = _render(sr, rgb, percentiles, hr)
        hr_rendered, limits = _render(hr, rgb, percentiles, hr)

        for column, rendered in enumerate((lr_rendered, sr_rendered, hr_rendered)):
            axis = axes[row][column]
            axis.imshow(rendered, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])

        axis = axes[row][3]
        sam_image = axis.imshow(
            sam_map, cmap="inferno", vmin=0.0, vmax=sam_vmax, interpolation="nearest"
        )
        axis.set_xticks([])
        axis.set_yticks([])

        # Sample ids are long (a full NAIP quarter-quad key). Wrapped onto its
        # own lines so the label never collides with the metric lines beneath --
        # the id is what makes a row traceable back to a CSV row.
        label = str(sample.get("sample_id", f"sample {row}"))
        detail = [_wrap_id(label)]
        if np.isfinite(sample.get("psnr_mean", np.nan)):
            detail.append(f"PSNR {float(sample['psnr_mean']):.2f} dB")
        if np.isfinite(sample.get("sam_mean_deg", np.nan)):
            detail.append(f"SAM {float(sample['sam_mean_deg']):.2f}°")
        detail.append(f"stretch [{limits[0]:.3f}, {limits[1]:.3f}] refl.")
        axes[row][0].set_ylabel(
            "\n".join(detail),
            fontsize=6.5,
            rotation=0,
            ha="right",
            va="center",
            labelpad=10,
            linespacing=1.6,
        )

    for column, title in enumerate(columns):
        axes[0][column].set_title(title, fontsize=10)

    figure.suptitle(
        f"{method} baseline, x{scale} ({lr_gsd:g} m -> {hr_gsd:g} m) -- "
        f"{band_label} composite",
        fontsize=12,
    )

    colorbar = figure.colorbar(
        sam_image, ax=axes[:, 3].tolist(), fraction=0.05, pad=0.02
    )
    colorbar.set_label("spectral angle (degrees)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    figure.supxlabel(
        "Imagery is percentile-stretched for display only "
        f"({percentiles[0]:g}-{percentiles[1]:g} percentile of the HR panel, "
        "shared across LR/SR/HR within a row); metrics use unstretched, "
        "unclipped reflectance.\nSAM colour bar 0-"
        f"{sam_vmax:.2f} degrees, from the {vmax_source}. Blank pixels in the "
        "SAM panel are undefined angles (an all-zero spectrum).",
        fontsize=7,
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return out
