"""Consistency error against training iteration, with the spectral floor.

One axis, one quantity: ``L1(x_LR, D(y_SR))`` in reflectance units, with ``D``
the same area operator the spectral loss trains through and the floor was
measured with. Two sources of points go on it, and they are deliberately drawn
differently because they are different measurements:

- **The line** is the training loop's own validation, read from ``log.csv``:
  every ``--val-every`` iterations, on the ``--val-batches`` subsample, under
  AMP. Dense in iteration, thin in patches.
- **The markers** are the full validation split, float32, at the checkpoints
  that exist on disk. Sparse in iteration, complete in patches. A ring marks
  the checkpoint the selection rule chose.

If the markers do not sit on the line, that is the subsample disagreeing with
the full split, not a plotting error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

__all__ = ["read_validation_curve", "plot_consistency_vs_iter"]


def read_validation_curve(log_csv: Path) -> pd.DataFrame:
    """The validation rows of a ``src/train.py`` ``log.csv``.

    Args:
        log_csv: Path to the run's ``log.csv``. Training rows (loss, lr) and
            validation rows (``val_*``) are interleaved; a validation row is one
            with a ``val_psnr`` value.

    Returns:
        One row per validation, sorted by ``iter``, with every ``val_*`` column
        the log has. A legacy Run A log has no ``val_l1_spec``; callers check.

    Raises:
        FileNotFoundError: The log is missing.
        KeyError: The log has no ``iter`` or ``val_psnr`` column.
    """
    path = Path(log_csv)
    if not path.is_file():
        raise FileNotFoundError(f"No training log at {path}.")
    frame = pd.read_csv(path)
    for column in ("iter", "val_psnr"):
        if column not in frame.columns:
            raise KeyError(f"{path} has no {column!r} column; columns: {list(frame.columns)}.")
    rows = frame[pd.to_numeric(frame["val_psnr"], errors="coerce").notna()]
    keep = ["iter"] + [c for c in rows.columns if c.startswith("val_")]
    return rows[keep].apply(pd.to_numeric, errors="coerce").sort_values("iter").reset_index(drop=True)


def plot_consistency_vs_iter(
    series: Sequence[Mapping[str, Any]],
    floor: float,
    out_path: Path,
    style: Mapping[str, Any],
    title: str,
    ylabel: str,
    floor_label: str,
) -> Path:
    """Draw consistency vs iteration for several runs on one axis.

    Args:
        series: One mapping per run: ``display`` (legend/label text),
            ``color`` (hex), ``curve`` (frame with ``iter`` and ``value``: the
            log.csv subsample), ``points`` (frame with ``iter``, ``value`` and
            boolean ``selected``: the full-split measurements).
        floor: The spectral floor in the same units, drawn as a dashed line.
        out_path: PNG destination; parent directories are created.
        style: ``surface``, ``ink``, ``ink_secondary``, ``muted`` (hex) and
            ``dpi``, from ``cfg.eval_all_ckpts.figure``.
        title: Figure title.
        ylabel: Y-axis label, including units.
        floor_label: Text placed on the floor line.

    Returns:
        ``out_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    surface = str(style["surface"])
    ink = str(style["ink"])
    ink2 = str(style["ink_secondary"])
    muted = str(style["muted"])

    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor=surface)
    ax.set_facecolor(surface)

    x_max = 0.0
    ends = []
    for entry in series:
        curve = entry["curve"]
        points = entry["points"]
        color = str(entry["color"])
        ax.plot(curve["iter"], curve["value"], color=color, lw=2.0, zorder=3)
        ax.plot(
            points["iter"], points["value"], ls="none", marker="o", ms=8,
            color=color, markeredgecolor=surface, markeredgewidth=2.0, zorder=5,
        )
        chosen = points[points["selected"]]
        ax.plot(
            chosen["iter"], chosen["value"], ls="none", marker="o", ms=17,
            markerfacecolor="none", markeredgecolor=color, markeredgewidth=2.0,
            zorder=4,
        )
        last = curve.iloc[-1]
        ends.append((float(last["iter"]), float(last["value"]), str(entry["display"])))
        x_max = max(x_max, float(curve["iter"].max()), float(points["iter"].max()))

    ax.set_ylim(0, None)
    # Direct labels at the line ends, in ink (the colour beside them carries
    # identity). Nudged apart to a minimum gap so close endpoints stay legible;
    # the label moves, never the data.
    y_low, y_high = ax.get_ylim()
    gap = 0.045 * (y_high - y_low)
    placed: list = []
    for x_end, y_end, text in sorted(ends, key=lambda e: e[1]):
        y_text = y_end if not placed else max(y_end, placed[-1] + gap)
        placed.append(y_text)
        ax.annotate(
            text, (x_end, y_end), xytext=(x_end + 0.015 * x_max, y_text), textcoords="data",
            va="center", ha="left", color=ink, fontsize=9,
            xycoords="data", annotation_clip=False,
        )

    ax.axhline(floor, color=ink2, lw=1.5, ls=(0, (6, 4)), zorder=2)
    ax.annotate(
        floor_label, (0.0, floor), xycoords=("axes fraction", "data"),
        xytext=(6, 5), textcoords="offset points", color=ink2, fontsize=9,
    )

    ax.set_xlim(0, x_max * 1.16)
    ax.set_ylim(0, None)
    ax.set_xlabel("training iteration", color=ink2)
    ax.set_ylabel(ylabel, color=ink2)
    ax.set_title(title, color=ink, fontsize=11, loc="left")
    ax.grid(axis="y", color=muted, alpha=0.25, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=ink2)

    handles = [
        Line2D([], [], color=str(e["color"]), lw=2.0, label=f"{e['display']} (log subsample)")
        for e in series
    ] + [
        Line2D([], [], ls="none", marker="o", ms=8, color=muted,
               markeredgecolor=surface, markeredgewidth=2.0, label="full val split"),
        Line2D([], [], ls="none", marker="o", ms=13, markerfacecolor="none",
               markeredgecolor=muted, markeredgewidth=2.0, label="selected checkpoint"),
    ]
    legend = ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(ink)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=int(style["dpi"]), facecolor=surface)
    plt.close(fig)
    return out
