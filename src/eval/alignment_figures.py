"""Figures for the LR/HR alignment audit.

Two figures, both of which exist so a number in the report can be disbelieved
and checked by eye:

- :func:`plot_alignment_grid` -- per pair: the LR tile (nearest-neighbour
  upsampled, so LR pixels stay visibly blocky and are never mistaken for
  detail), the bicubic upsample, the HR target, and the absolute difference
  between the last two. A systematic misregistration shows up in the difference
  panel as bright edge outlines: every boundary in the scene traced twice.
- :func:`plot_reflectance_histograms` -- per band, the LR and HR reflectance
  distributions overlaid, which is how "are these two sources radiometrically
  comparable" is answered without taking the summary statistic on trust.

Everything here is display code. The percentile stretch used to render an RGB
composite is applied to a **copy**, for the figure only: reflectance that reaches
a loss, a metric, or the report's histograms is never stretched, scaled, or
clipped. The colour bars and axis labels are in reflectance units so a reader can
tell what the stretch was.

Matplotlib runs on the non-interactive Agg backend: these must render on a
headless Kaggle notebook and on a local CPU box with no display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend switch)
import numpy as np  # noqa: E402

from src.eval.alignment import bicubic_upsample, reference_band_index  # noqa: E402

__all__ = [
    "nearest_upsample",
    "stretch_for_display",
    "plot_alignment_grid",
    "plot_reflectance_histograms",
]


def nearest_upsample(image: np.ndarray, scale: int) -> np.ndarray:
    """Replicate each pixel ``scale`` times in both axes.

    Nearest neighbour, deliberately: the LR panel must look like LR data. A
    smooth upsample in that column would invite the reader to compare detail
    that is not there.

    Args:
        image: ``(C, H, W)`` float32 surface reflectance.
        scale: Replication factor.

    Returns:
        ``(C, H*scale, W*scale)`` float32, same reflectance values.
    """
    array = np.asarray(image, dtype=np.float32)
    return np.repeat(np.repeat(array, int(scale), axis=1), int(scale), axis=2)


def stretch_for_display(
    image: np.ndarray,
    percentiles: Tuple[float, float],
    reference: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Percentile-stretch reflectance to ``[0, 1]`` for rendering only.

    The clip here is on a **display copy**. It is the one place in the project
    where reflectance is clipped, it is explicit, it never propagates, and the
    limits it used are returned so a caption can state them.

    Args:
        image: ``(C, H, W)`` or ``(H, W)`` float32 surface reflectance.
        percentiles: ``(low, high)`` from ``cfg.alignment.display_percentiles``.
        reference: Optional array whose percentiles set the limits instead, so
            two panels can share one stretch and be compared honestly.

    Returns:
        ``(rendered, (low, high))`` where ``rendered`` has the same shape as
        ``image`` and lies in ``[0, 1]``, and the limits are in reflectance
        units.
    """
    array = np.asarray(image, dtype=np.float32)
    source = array if reference is None else np.asarray(reference, dtype=np.float32)
    low, high = np.percentile(source, list(percentiles))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(source)), float(np.nanmax(source))
        if high <= low:
            high = low + 1e-6
    return np.clip((array - low) / (high - low), 0.0, 1.0), (float(low), float(high))


def _rgb_indices(cfg: Any) -> Optional[List[int]]:
    """Channel indices for the RGB preview, or None when the bands are absent."""
    bands = [str(b) for b in cfg["dataset"]["bands"]]
    wanted = [str(b) for b in cfg["alignment"]["display_rgb_bands"]]
    if not all(band in bands for band in wanted):
        return None
    return [bands.index(band) for band in wanted]


def _to_display(
    image: np.ndarray,
    rgb: Optional[Sequence[int]],
    band_index: int,
    percentiles: Tuple[float, float],
    reference: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Render one ``(C, H, W)`` reflectance array as an RGB or greyscale image."""
    if rgb is not None:
        selected = np.stack([image[i] for i in rgb], axis=-1)
        ref = None if reference is None else np.stack([reference[i] for i in rgb], axis=-1)
    else:
        selected = image[band_index]
        ref = None if reference is None else reference[band_index]
    rendered, limits = stretch_for_display(selected, percentiles, reference=ref)
    return rendered, limits


def plot_alignment_grid(
    pairs: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    cfg: Any,
    path: Any,
    max_rows: Optional[int] = None,
) -> Path:
    """Render the four-panel-per-pair qualitative grid.

    Columns, left to right: nearest-neighbour upsampled LR, bicubic upsampled LR,
    HR, and ``|bicubic - HR|``. Each row is annotated with the measured
    ``(dy, dx)`` in HR pixels and both correlations, so the picture and the
    number are read together.

    Args:
        pairs: Loaded pairs from :func:`~src.eval.alignment.collect_pairs`, with
            ``lr`` and ``hr`` as ``(C, H, W)`` float32 surface reflectance.
        records: The per-pair records from the audit report, matched to ``pairs``
            by ``sample_id``.
        cfg: The loaded config; reads ``alignment`` and ``sr``.
        path: Destination PNG.
        max_rows: Rows to draw. Defaults to ``cfg.alignment.figure_pairs``.

    Returns:
        The path written.

    Raises:
        ValueError: ``pairs`` is empty, or a pair has no matching record -- which
            would mean the figure and the report describe different data.
    """
    if not pairs:
        raise ValueError("plot_alignment_grid received no pairs.")

    align_cfg = cfg["alignment"]
    scale = int(cfg["sr"]["scale"])
    hr_gsd = float(cfg["dataset"]["hr_gsd_m"])
    percentiles = tuple(float(p) for p in align_cfg["display_percentiles"])
    rows = int(max_rows if max_rows is not None else align_cfg["figure_pairs"])
    rows = min(rows, len(pairs))

    by_id = {str(r["sample_id"]): r for r in records}
    rgb = _rgb_indices(cfg)
    band_index = reference_band_index(cfg)
    band_name = str(align_cfg["reference_band"])

    figure, axes = plt.subplots(
        rows, 4, figsize=(11.0, 2.75 * rows), squeeze=False
    )

    for row in range(rows):
        pair = pairs[row]
        record = by_id.get(str(pair["sample_id"]))
        if record is None:
            raise ValueError(
                f"Pair {pair['sample_id']!r} has no record in the audit report. "
                "The figure would then describe different pairs from the "
                "numbers beside it."
            )

        lr = np.asarray(pair["lr"], dtype=np.float32)
        hr = np.asarray(pair["hr"], dtype=np.float32)
        upsampled = bicubic_upsample(lr, scale)
        nearest = nearest_upsample(lr, scale)

        # HR sets the stretch for all three image panels, so a brightness
        # difference between the sources is visible rather than normalised away.
        hr_display, limits = _to_display(hr, rgb, band_index, percentiles)
        nearest_display, _ = _to_display(
            nearest, rgb, band_index, percentiles, reference=hr
        )
        bicubic_display, _ = _to_display(
            upsampled, rgb, band_index, percentiles, reference=hr
        )

        difference = np.abs(upsampled - hr).mean(axis=0)
        # The difference panel gets its own stretch, in reflectance units, with
        # the top of the scale printed in the title -- an unlabelled heat map
        # would make any pair look equally bad.
        diff_max = float(np.percentile(difference, float(percentiles[1])))
        if diff_max <= 0:
            diff_max = float(difference.max()) or 1e-6

        panels = [
            (nearest_display, f"LR x{scale} nearest", None),
            (bicubic_display, f"LR x{scale} bicubic", None),
            (hr_display, "HR target", None),
            (difference, f"|bicubic - HR|  max {diff_max:.3f}", "inferno"),
        ]
        for column, (image, title, cmap) in enumerate(panels):
            axis = axes[row][column]
            if cmap is None:
                axis.imshow(image, interpolation="nearest", vmin=0.0, vmax=1.0)
            else:
                handle = axis.imshow(
                    image, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=diff_max
                )
                bar = figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.02)
                bar.ax.tick_params(labelsize=6)
                bar.set_label("reflectance", fontsize=6)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=9)

        magnitude = float(record["shift_magnitude"])
        label = (
            f"{pair['sample_id']}\n"
            f"dy {record['dy']:+.2f}  dx {record['dx']:+.2f} HR px\n"
            f"|shift| {magnitude:.2f} px = {magnitude * hr_gsd:.2f} m\n"
            f"r(LR grid) {record['correlation_lr_grid']:.3f}\n"
            f"r(HR grid) {record['correlation_hr_grid']:.3f}"
        )
        axes[row][0].set_ylabel(label, fontsize=6.5, rotation=0, ha="right", va="center")
        axes[row][0].yaxis.set_label_coords(-0.04, 0.5)

    injected = records[0].get("injected_shift") if records else None
    subtitle = (
        f"registration band {band_name}   |   display stretch "
        f"p{percentiles[0]:g}-p{percentiles[1]:g} "
        f"(reflectance {limits[0]:.3f}-{limits[1]:.3f}), figures only"
    )
    if injected:
        subtitle += (
            f"\nHR deliberately shifted by (dy {injected[0]:+.1f}, "
            f"dx {injected[1]:+.1f}) HR px -- estimator self-test"
        )
    figure.suptitle(
        f"LR/HR alignment -- {cfg['dataset']['name']}\n{subtitle}",
        fontsize=9,
    )
    # Reserve a fixed number of inches for the title, not a fixed fraction: at 20
    # rows the figure is over four feet tall and a percentage would leave a hand's
    # width of blank paper above the first row.
    title_inches = 0.85 if injected else 0.6
    figure.tight_layout(
        rect=(0.06, 0.0, 1.0, 1.0 - title_inches / figure.get_figheight())
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=int(align_cfg["figure_dpi"]))
    plt.close(figure)
    return out


def plot_reflectance_histograms(
    pairs: Sequence[Mapping[str, Any]],
    cfg: Any,
    path: Any,
    radiometry: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Overlay the LR and HR reflectance histograms, one subplot per band.

    Reflectance is plotted as it is: no stretch, no clip, no normalisation. The
    x-axis covers the full range of both sources, so a bright tail above 1.0 is
    visible rather than hidden -- that tail is real signal over cloud, snow, and
    specular water, and a model trained on clipped data cannot reproduce it.

    Args:
        pairs: Loaded pairs with ``lr`` and ``hr`` as ``(C, H, W)`` float32
            surface reflectance.
        cfg: The loaded config; reads ``dataset.bands`` and ``alignment``.
        path: Destination PNG.
        radiometry: The ``radiometry`` block of the audit report. When given, the
            per-band percentile limits and overlap score are drawn on the plot.

    Returns:
        The path written.

    Raises:
        ValueError: ``pairs`` is empty.
    """
    if not pairs:
        raise ValueError("plot_reflectance_histograms received no pairs.")

    band_names = [str(b) for b in cfg["dataset"]["bands"]]
    align_cfg = cfg["alignment"]
    bins = int(align_cfg["histogram_bins"])
    nominal_max = float(cfg["dataset"]["reflectance_nominal_max"])

    lr_stack = np.concatenate(
        [np.asarray(p["lr"], dtype=np.float32).reshape(len(band_names), -1) for p in pairs],
        axis=1,
    )
    hr_stack = np.concatenate(
        [np.asarray(p["hr"], dtype=np.float32).reshape(len(band_names), -1) for p in pairs],
        axis=1,
    )

    columns = min(len(band_names), 2)
    rows = int(np.ceil(len(band_names) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(6.0 * columns, 3.4 * rows), squeeze=False
    )

    for index, name in enumerate(band_names):
        axis = axes[index // columns][index % columns]
        lr_values = lr_stack[index]
        hr_values = hr_stack[index]
        low = float(min(lr_values.min(), hr_values.min()))
        high = float(max(lr_values.max(), hr_values.max()))
        edges = np.linspace(low, high, bins + 1)

        axis.hist(
            lr_values, bins=edges, density=True, alpha=0.55, label="LR (10 m)",
            color="#1f77b4",
        )
        axis.hist(
            hr_values, bins=edges, density=True, alpha=0.55, label="HR (2.5 m)",
            color="#d62728",
        )
        if high > nominal_max:
            axis.axvline(
                nominal_max,
                color="black",
                linestyle=":",
                linewidth=1.0,
                label=f"nominal max {nominal_max:g} (values above are real, not clipped)",
            )
        title = f"{name}"
        if radiometry is not None and name in radiometry["bands"]:
            band = radiometry["bands"][name]
            for value, colour in (
                (band["lr_low"], "#1f77b4"),
                (band["lr_high"], "#1f77b4"),
                (band["hr_low"], "#d62728"),
                (band["hr_high"], "#d62728"),
            ):
                axis.axvline(value, color=colour, linestyle="--", linewidth=0.8)
            title += f"   range overlap {band['overlap']:.3f}"
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("surface reflectance")
        axis.set_ylabel("density")
        axis.legend(fontsize=7)

    for spare in range(len(band_names), rows * columns):
        axes[spare // columns][spare % columns].axis("off")

    percentiles = (
        radiometry["percentiles"] if radiometry is not None else ["-", "-"]
    )
    figure.suptitle(
        f"LR vs HR reflectance distributions -- {cfg['dataset']['name']}, "
        f"{len(pairs)} pairs. Dashed lines: p{percentiles[0]}/p{percentiles[1]} "
        "range limits. Unstretched and unclipped.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=int(align_cfg["figure_dpi"]))
    plt.close(figure)
    return out
