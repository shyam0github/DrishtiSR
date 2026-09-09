"""Training-curve figures from the ``log.csv`` a training run writes.

``src/train.py`` appends two different kinds of row to one CSV, distinguished by
which columns are empty:

- a **training** row every ``--log-every`` iterations:
  ``iter, loss, lr, "", sec_per_100it``
- a **validation** row every ``--val-every`` iterations:
  ``iter, "", "", val_psnr, ""``

so the file is read here by splitting on those empty cells rather than by
assuming an interleaving pattern. A resumed run appends to the same file, which
means iterations can repeat; rows are therefore sorted by iteration and the last
value wins, and any repetition is reported to the caller instead of being
smoothed over.

WHAT THE FIGURE IS FOR. One question decides whether a run should have been
longer: **was validation still improving when the run stopped?** A loss curve
alone cannot answer it -- training loss falls long after validation has turned
-- so the validation panel marks the best point, the final point, and the gap
between them explicitly, and :func:`curve_verdict` turns that into a sentence a
report can quote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

# Non-interactive backend, selected before pyplot is imported: these figures are
# written from headless entry points (local CPU runs and Kaggle jobs alike) and
# a default backend that wants a display makes the import itself fail there.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.train import KNOWN_SCHEMAS, LEGACY_COLUMNS, LOG_COLUMNS  # noqa: E402

__all__ = ["read_training_log", "curve_verdict", "plot_training_curves"]


def read_training_log(path: Any) -> Dict[str, pd.DataFrame]:
    """Split a training ``log.csv`` into its training and validation rows.

    Args:
        path: Path to the CSV written by ``src/train.py``. Must carry one of
            the headers in :data:`src.train.KNOWN_SCHEMAS` -- the legacy Run A
            five, those plus the spectral scalars, or the current
            :data:`src.train.LOG_COLUMNS` which adds the blur diagnostic and
            the reference metric suite. Each generation only appends, so all
            three are read by the same code.

    Returns:
        ``{"train": DataFrame, "val": DataFrame}``. ``train`` has columns
        ``iter`` (int), ``loss`` (float), ``lr`` (float) and ``sec_per_100it``
        (float); ``val`` has ``iter`` (int), ``val_psnr`` (float) and, when
        the log carries them, ``val_l1_spec`` (float, reflectance),
        ``val_sam`` (float, radians -- the SPECTRAL-domain angle, LR against
        the downsampled SR, not the image-domain ``val_sam_hr`` in degrees),
        ``val_sam_valid_frac``, ``val_sharpness`` (reflectance per pixel),
        ``val_hf_energy`` (dimensionless), ``val_ssim``, ``val_lpips``,
        ``val_sam_hr`` (degrees) and ``val_ergas`` (all float). Both are
        sorted by ``iter`` with duplicates resolved to the last occurrence, so a
        resumed run yields one curve rather than a sawtooth.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: The header is not the expected one, or one of the two row
            kinds is absent -- an empty validation set would silently produce a
            figure with a blank panel, which reads as "no validation was run"
            and "validation was flat" identically.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Training log {csv_path} does not exist. It is written by "
            "src/train.py into the run's --out directory and fetched with "
            "scripts/kaggle_run.py fetch."
        )

    frame = pd.read_csv(csv_path)
    # Only the schemas the trainer has actually written are accepted, and the
    # list is IMPORTED from it rather than restated, so a new column cannot
    # appear in one file and not the other. Each generation appends only, which
    # is what lets one reader handle a Run A log and a Day 3 log unchanged.
    columns = list(frame.columns)
    if columns not in [list(schema) for schema in KNOWN_SCHEMAS]:
        raise ValueError(
            f"{csv_path} has columns {columns}, expected one of "
            f"{[list(schema) for schema in KNOWN_SCHEMAS]} -- oldest is Run A's "
            f"{list(LEGACY_COLUMNS)}, newest is {list(LOG_COLUMNS)}. This is not "
            "a log.csv written by src/train.py."
        )

    train = frame[frame["loss"].notna()].copy()
    val = frame[frame["val_psnr"].notna()].copy()

    if train.empty:
        raise ValueError(
            f"{csv_path} contains no training rows (every 'loss' cell is empty). "
            "The run logged nothing; there is no curve to draw."
        )
    if val.empty:
        raise ValueError(
            f"{csv_path} contains no validation rows (every 'val_psnr' cell is "
            "empty). The run never validated, so whether it was still improving "
            "cannot be answered from this file -- do not draw a figure that "
            "looks as though it could."
        )

    train = train[["iter", "loss", "lr", "sec_per_100it"]].astype(
        {"iter": int, "loss": float, "lr": float, "sec_per_100it": float}
    )
    val_columns = ["iter", "val_psnr"] + [
        name for name in LOG_COLUMNS[len(LEGACY_COLUMNS):] if name in columns
    ]
    types = {name: float for name in val_columns}
    types["iter"] = int
    val = val[val_columns].astype(types)

    # A resumed run re-logs iterations it already logged. Keep the last value for
    # each iteration -- that is the one the surviving checkpoint came from.
    train = train.drop_duplicates(subset="iter", keep="last").sort_values("iter")
    val = val.drop_duplicates(subset="iter", keep="last").sort_values("iter")

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
    }


def curve_verdict(val: pd.DataFrame, tail_fraction: float = 0.25) -> Dict[str, Any]:
    """Decide whether validation was still improving when the run stopped.

    The test is deliberately not "is the last point the best point": validation
    PSNR is noisy, and a run whose best point is one evaluation before the end
    is not undertrained. Instead the curve is split at ``tail_fraction`` and the
    question is where the maximum falls and which way the tail is sloping.

    Args:
        val: The ``val`` frame from :func:`read_training_log`: columns ``iter``
            and ``val_psnr``, sorted, one row per validation.
        tail_fraction: Fraction of the iteration range treated as "the tail",
            in ``(0, 1]``. The slope is fitted over the validation points that
            fall inside it.

    Returns:
        A dict with ``best_iter``, ``best_psnr``, ``final_iter``, ``final_psnr``,
        ``delta_final_minus_best`` (dB, never positive by definition of best),
        ``best_at_fraction`` (where the peak sits in the run, 0-1),
        ``tail_slope_db_per_1k`` (least-squares slope over the tail),
        ``still_improving`` (bool) and ``verdict`` (a sentence).

    Raises:
        ValueError: Fewer than two validation points -- a slope needs two -- or
            ``tail_fraction`` is outside ``(0, 1]``.
    """
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError(
            f"tail_fraction must be in (0, 1]; got {tail_fraction!r}."
        )
    if len(val) < 2:
        raise ValueError(
            f"Need at least 2 validation points to judge a trend; got {len(val)}. "
            "Re-run with a smaller --val-every, or say nothing about the trend."
        )

    iters = val["iter"].to_numpy(dtype=float)
    psnr = val["val_psnr"].to_numpy(dtype=float)

    best_row = int(np.argmax(psnr))
    best_iter, best_psnr = float(iters[best_row]), float(psnr[best_row])
    final_iter, final_psnr = float(iters[-1]), float(psnr[-1])
    span = float(iters[-1] - iters[0]) or 1.0
    best_at_fraction = (best_iter - iters[0]) / span

    tail_start = iters[-1] - float(tail_fraction) * span
    tail = iters >= tail_start
    # polyfit needs two distinct x values; widen to the last two points if the
    # tail collapsed onto one iteration.
    if int(tail.sum()) < 2:
        tail = np.zeros_like(iters, dtype=bool)
        tail[-2:] = True
    slope_per_iter = float(np.polyfit(iters[tail], psnr[tail], 1)[0])
    slope_db_per_1k = slope_per_iter * 1000.0

    # Both conditions must hold: the peak is in the last quarter AND the tail is
    # still sloping up. Either alone is satisfied by a curve that has plateaued.
    still_improving = bool(
        best_at_fraction >= 1.0 - float(tail_fraction) and slope_db_per_1k > 0.0
    )

    if still_improving:
        verdict = (
            f"Validation was STILL IMPROVING at {final_iter:,.0f} iterations: "
            f"the best point ({best_psnr:.3f} dB) is at {best_iter:,.0f}, in the "
            f"final {tail_fraction:.0%} of the run, and the tail is rising at "
            f"{slope_db_per_1k:+.4f} dB per 1k iterations. The run is "
            f"UNDERTRAINED -- more iterations should still buy PSNR."
        )
    else:
        verdict = (
            f"Validation was NOT still improving at {final_iter:,.0f} iterations: "
            f"it peaked at {best_psnr:.3f} dB at iteration {best_iter:,.0f} "
            f"({best_at_fraction:.0%} of the way through) and ended at "
            f"{final_psnr:.3f} dB, {final_psnr - best_psnr:+.3f} dB from the "
            f"peak, with the final {tail_fraction:.0%} sloping "
            f"{slope_db_per_1k:+.4f} dB per 1k iterations. The run is NOT "
            f"undertrained; the extra iterations after the peak bought nothing."
        )

    return {
        "best_iter": int(best_iter),
        "best_psnr": best_psnr,
        "final_iter": int(final_iter),
        "final_psnr": final_psnr,
        "delta_final_minus_best": final_psnr - best_psnr,
        "best_at_fraction": best_at_fraction,
        "tail_fraction": float(tail_fraction),
        "tail_slope_db_per_1k": slope_db_per_1k,
        "still_improving": still_improving,
        "num_val_points": int(len(val)),
        "verdict": verdict,
    }


def _smooth(values: np.ndarray, window: int) -> Optional[np.ndarray]:
    """Centred moving average, or None when the series is too short for one.

    Args:
        values: 1-D series.
        window: Window length in samples. Values below 3 disable smoothing.

    Returns:
        The smoothed series, same length (edges averaged over fewer points), or
        ``None`` if no smoothing was applied.
    """
    width = int(window)
    if width < 3 or len(values) < width:
        return None
    return (
        pd.Series(values)
        .rolling(width, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def plot_training_curves(
    log_path: Any,
    cfg: Any,
    out_path: Any,
    title: str = "",
    best_checkpoint_iter: Optional[int] = None,
) -> Dict[str, Any]:
    """Draw the loss and validation curves for one training run.

    Two stacked panels sharing the iteration axis:

    1. **Training L1 loss** on a log scale (raw trace at low alpha, moving
       average over it), with the learning-rate schedule on a twin axis so a
       flattening loss can be read against the cosine decay that caused it.
    2. **Validation PSNR**, with the peak marked, the final value marked, and a
       dashed line at the peak so the gap is visible rather than inferred.

    Args:
        log_path: The run's ``log.csv``.
        cfg: Loaded config. Reads ``eval_runA.curves_dpi`` and
            ``eval_runA.curves_loss_smooth``.
        out_path: Destination ``.png``. Parent directories are created.
        title: Figure title. The verdict is appended to it.
        best_checkpoint_iter: Iteration the saved ``best.pt`` came from, marked
            on the validation panel. ``None`` omits the marker. This is read
            from the checkpoint rather than inferred from the curve, so a
            disagreement between the two is visible instead of assumed away.

    Returns:
        :func:`curve_verdict` output, with ``figure_path`` added.

    Raises:
        FileNotFoundError: ``log_path`` does not exist.
        ValueError: The log is not a ``src/train.py`` log, or lacks training or
            validation rows.
    """
    frames = read_training_log(log_path)
    train, val = frames["train"], frames["val"]
    verdict = curve_verdict(val)

    dpi = int(cfg["eval_runA"]["curves_dpi"])
    smooth_window = int(cfg["eval_runA"]["curves_loss_smooth"])

    figure, (loss_ax, psnr_ax) = plt.subplots(
        2, 1, figsize=(9.0, 7.0), sharex=True, dpi=dpi
    )

    # -- panel 1: training loss, with the LR schedule behind it ---------------
    iters = train["iter"].to_numpy(dtype=float)
    loss = train["loss"].to_numpy(dtype=float)
    loss_ax.plot(iters, loss, color="#9ecae1", linewidth=0.8, alpha=0.65,
                 label="L1 loss (raw, per 100 it)")
    smoothed = _smooth(loss, smooth_window)
    if smoothed is not None:
        loss_ax.plot(iters, smoothed, color="#08519c", linewidth=1.8,
                     label=f"L1 loss (moving mean, {smooth_window} pts)")
    loss_ax.set_yscale("log")
    loss_ax.set_ylabel("training L1 loss (reflectance)")
    loss_ax.grid(True, which="both", alpha=0.25)

    lr_ax = loss_ax.twinx()
    lr_ax.plot(iters, train["lr"].to_numpy(dtype=float), color="#cb181d",
               linewidth=1.2, linestyle=":", label="learning rate")
    lr_ax.set_yscale("log")
    lr_ax.set_ylabel("learning rate", color="#cb181d")
    lr_ax.tick_params(axis="y", labelcolor="#cb181d")

    handles, labels = loss_ax.get_legend_handles_labels()
    lr_handles, lr_labels = lr_ax.get_legend_handles_labels()
    loss_ax.legend(handles + lr_handles, labels + lr_labels, loc="upper right",
                   fontsize=8, framealpha=0.9)

    # -- panel 2: validation PSNR --------------------------------------------
    val_iters = val["iter"].to_numpy(dtype=float)
    val_psnr = val["val_psnr"].to_numpy(dtype=float)
    psnr_ax.plot(val_iters, val_psnr, color="#238b45", marker="o", markersize=4,
                 linewidth=1.6, label="val PSNR (training-loop subsample)")
    psnr_ax.axhline(verdict["best_psnr"], color="#238b45", linestyle="--",
                    linewidth=0.9, alpha=0.7)
    psnr_ax.plot([verdict["best_iter"]], [verdict["best_psnr"]], marker="*",
                 markersize=15, color="#d94801", linestyle="none",
                 label=(f"peak {verdict['best_psnr']:.3f} dB "
                        f"@ {verdict['best_iter']:,}"))
    psnr_ax.plot([verdict["final_iter"]], [verdict["final_psnr"]], marker="s",
                 markersize=7, color="#54278f", linestyle="none",
                 label=(f"final {verdict['final_psnr']:.3f} dB "
                        f"({verdict['delta_final_minus_best']:+.3f} dB)"))
    if best_checkpoint_iter is not None:
        psnr_ax.axvline(int(best_checkpoint_iter), color="#d94801",
                        linestyle="-.", linewidth=1.0, alpha=0.8,
                        label=f"best.pt saved @ {int(best_checkpoint_iter):,}")
    psnr_ax.set_ylabel("validation PSNR (dB)")
    psnr_ax.set_xlabel("iteration")
    psnr_ax.grid(True, alpha=0.25)
    # Upper right, not lower right: a decaying curve puts its FINAL point in the
    # bottom-right corner, and the whole question this panel answers is where
    # that point sits relative to the peak. A legend on top of it hides the
    # answer.
    psnr_ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    heading = title or Path(str(log_path)).parent.name
    figure.suptitle(heading, fontsize=12, y=0.985)
    # The verdict travels with the figure. A curve that is read without it is
    # exactly how "we should have trained longer" survives as an assumption.
    figure.text(
        0.5, 0.005,
        "STILL IMPROVING at the end" if verdict["still_improving"]
        else (f"NOT still improving: peak at {verdict['best_iter']:,}, "
              f"{verdict['delta_final_minus_best']:+.3f} dB by "
              f"{verdict['final_iter']:,}"),
        ha="center", fontsize=9,
        color="#238b45" if verdict["still_improving"] else "#cb181d",
    )

    figure.tight_layout(rect=(0, 0.022, 1, 0.975))

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)

    result = dict(verdict)
    result["figure_path"] = str(destination)
    return result
