"""Is a per-pixel uncertainty raster monotone against error? Binned, exactly.

The question ``scripts/calibrate_uncertainty.py`` answers: rank every validation
pixel by a predictor ``u`` (TTA disagreement, a head's sigma, a control), bin
the ranking, and read the mean absolute error per bin. If error rises bin over
bin, ``u`` orders pixels by how wrong the model is, and the raster is usable as
a hallucination map whatever its absolute scale.

STREAMING AND EXACT. The full validation split is 1199 patches x 4 bands x
256^2 = 314 M pixel values -- 2.5 GB for ``u`` and ``err`` in float32, before
sorting. Nothing here stores them. Each batch is folded into a FINE histogram
over ``log10(u)`` (default 800 bins across 8 decades, i.e. 2.3% wide), carrying
per bin the pixel count and the sums of ``|err|``, ``err^2`` and ``u``, in
float64. Every reported number is then computed from those sums:

- :func:`equal_mass_bins` merges consecutive fine bins into ``n_bins`` groups
  of (as near as the fine grid allows) equal pixel count -- the conventional
  calibration binning, where every point on the curve rests on the same
  amount of data;
- :func:`sparsification_curve` removes the most-uncertain pixels first and
  tracks the MAE of what remains; the same function applied to a histogram of
  the error ITSELF is the oracle, and :func:`ause` is the area between them
  (Ilg et al., ECCV 2018). Within a fine bin, removal is taken as proportional,
  which is exact to the fine-bin width.

A CONTROL, BECAUSE MONOTONE IS A LOW BAR. Error in SR concentrates on edges and
texture, and so does almost any spatial statistic of the output. A monotone
TTA curve is only evidence of UNCERTAINTY if TTA beats a predictor that knows
nothing about the model's doubt. :func:`gradient_magnitude` of the SR output is
that control: free, model-agnostic, and precisely the "it's just an edge
detector" hypothesis. Both are scored on the same pixels by the same code.

THE CONTROL IS NOT OPTIONAL. :func:`score_against_control` is the only function
here that turns a predictor into an AUSE for reporting, and it requires the
control's histogram and returns both scores and their delta together;
:func:`controlled_ause_markdown` renders every predictor row with its control
row and the delta directly beneath. A head's AUSE therefore cannot be reported
without the gradient-magnitude AUSE beside it -- which is the bar Run C's head
has to clear to be worth presenting.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "PredictorHistogram",
    "equal_mass_bins",
    "monotonicity",
    "sparsification_curve",
    "ause",
    "score_against_control",
    "controlled_ause_markdown",
    "gradient_magnitude",
    "plot_calibration",
]


class PredictorHistogram:
    """Per-band fine histogram of a predictor, carrying error sums per bin.

    Bin 0 is the underflow (``u < 10**log10_min``, including ``u == 0``), bin
    ``n_fine + 1`` the overflow (``u >= 10**log10_max``); both are kept, never
    dropped, so the pixel count always equals what was fed in.

    Args:
        n_bands: Channel count of the tensors fed to :meth:`update`.
        log10_min: Lower edge of the fine grid, log10 of the predictor's units.
        log10_max: Upper edge.
        n_fine: Fine bins between the two edges.

    Raises:
        ValueError: An empty grid or a non-positive bin count.
    """

    def __init__(self, n_bands: int, log10_min: float, log10_max: float, n_fine: int) -> None:
        if not float(log10_min) < float(log10_max) or int(n_fine) < 1 or int(n_bands) < 1:
            raise ValueError(
                f"invalid grid: bands={n_bands}, log10 [{log10_min}, {log10_max}], n_fine={n_fine}"
            )
        self.n_bands = int(n_bands)
        self.n_fine = int(n_fine)
        self.edges = torch.linspace(float(log10_min), float(log10_max), self.n_fine + 1,
                                    dtype=torch.float64)
        size = self.n_fine + 2
        self.count = torch.zeros(self.n_bands, size, dtype=torch.float64)
        self.sum_abs = torch.zeros_like(self.count)
        self.sum_sq = torch.zeros_like(self.count)
        self.sum_u = torch.zeros_like(self.count)

    @torch.no_grad()
    def update(self, u: torch.Tensor, err: torch.Tensor) -> None:
        """Fold in one batch.

        Args:
            u: ``(B, C, H, W)``, any float dtype. The predictor, >= 0 (a
                standard deviation, a gradient magnitude, an absolute error).
            err: Same shape. Signed or absolute error of the SR output against
                the HR, in reflectance; only ``|err|`` and ``err^2`` are used.

        Raises:
            ValueError: Shape mismatch, wrong channel count, a negative ``u``
                or a non-finite value in either -- binning a NaN would drop
                the pixel from every bin while it still counted as "seen".
        """
        if u.shape != err.shape or u.ndim != 4 or u.shape[1] != self.n_bands:
            raise ValueError(
                f"u {tuple(u.shape)} and err {tuple(err.shape)} must both be "
                f"(B, {self.n_bands}, H, W)."
            )
        u64 = u.detach().double().cpu()
        e64 = err.detach().double().cpu().abs()
        if not (torch.isfinite(u64).all() and torch.isfinite(e64).all()):
            raise ValueError("non-finite predictor or error value.")
        if (u64 < 0).any():
            raise ValueError("predictor must be non-negative.")

        size = self.n_fine + 2
        log_u = torch.log10(u64)  # log10(0) = -inf -> underflow bin
        bucket = torch.bucketize(log_u, self.edges, right=True)
        band = torch.arange(self.n_bands).view(1, -1, 1, 1)
        flat = (band * size + bucket).flatten()
        n = self.n_bands * size
        self.count += torch.bincount(flat, minlength=n).double().view(self.n_bands, size)
        self.sum_abs += torch.bincount(flat, weights=e64.flatten(), minlength=n).view(self.n_bands, size)
        self.sum_sq += torch.bincount(flat, weights=e64.pow(2).flatten(), minlength=n).view(self.n_bands, size)
        self.sum_u += torch.bincount(flat, weights=u64.flatten(), minlength=n).view(self.n_bands, size)

    def totals(self, band: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Per-fine-bin sums, for one band or pooled over all bands.

        Returns:
            ``count``, ``sum_abs``, ``sum_sq``, ``sum_u`` (float64 arrays of
            length ``n_fine + 2``), ``lo``/``hi`` (bin edges in the predictor's
            linear units; ``lo`` of the underflow is 0, ``hi`` of the overflow
            is ``inf``).
        """
        sel = slice(None) if band is None else slice(int(band), int(band) + 1)
        edges = (10.0 ** self.edges).numpy()
        return {
            "count": self.count[sel].sum(0).numpy(),
            "sum_abs": self.sum_abs[sel].sum(0).numpy(),
            "sum_sq": self.sum_sq[sel].sum(0).numpy(),
            "sum_u": self.sum_u[sel].sum(0).numpy(),
            "lo": np.concatenate([[0.0], edges]),
            "hi": np.concatenate([edges, [np.inf]]),
        }


def equal_mass_bins(hist: PredictorHistogram, n_bins: int,
                    band: Optional[int] = None) -> List[Dict[str, float]]:
    """Merge fine bins into ``n_bins`` groups of near-equal pixel count.

    A fine bin is never split, so a single fine bin holding more than
    ``1/n_bins`` of the mass (a saturated underflow, say) yields fewer groups
    than asked for; the count is visible in the output rather than padded.

    Args:
        hist: The accumulated histogram.
        n_bins: Target number of groups.
        band: One band index, or None to pool all bands.

    Returns:
        One dict per non-empty group, ordered by increasing ``u``: ``count``,
        ``mass`` (fraction of pixels), ``quantile_mid`` (cumulative mass at the
        group's centre, 0..1), ``u_lo``, ``u_hi`` (edges; ``u_hi`` None for an
        open top), ``u_mean``, ``mae`` and ``rmse`` (reflectance).

    Raises:
        ValueError: ``n_bins`` < 1 or the histogram is empty.
    """
    if int(n_bins) < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    t = hist.totals(band)
    total = float(t["count"].sum())
    if total == 0:
        raise ValueError("histogram is empty.")
    cum_before = np.cumsum(t["count"]) - t["count"]
    group = np.minimum(
        int(n_bins) - 1,
        np.floor((cum_before + 0.5 * t["count"]) * int(n_bins) / total).astype(int),
    )
    out: List[Dict[str, float]] = []
    running = 0.0
    for g in range(int(n_bins)):
        sel = (group == g) & (t["count"] > 0)
        count = float(t["count"][sel].sum())
        if count == 0:
            continue
        idx = np.flatnonzero(sel)
        hi = float(t["hi"][idx[-1]])
        out.append({
            "count": count,
            "mass": count / total,
            "quantile_mid": (running + 0.5 * count) / total,
            "u_lo": float(t["lo"][idx[0]]),
            "u_hi": None if math.isinf(hi) else hi,
            "u_mean": float(t["sum_u"][sel].sum()) / count,
            "mae": float(t["sum_abs"][sel].sum()) / count,
            "rmse": math.sqrt(float(t["sum_sq"][sel].sum()) / count),
        })
        running += count
    return out


def monotonicity(bins: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    """How monotone MAE is across bins ordered by increasing predictor.

    Args:
        bins: :func:`equal_mass_bins` output.

    Returns:
        ``n_bins``; ``inversions`` (adjacent pairs where MAE falls as ``u``
        rises); ``monotone`` (no inversions); ``spearman`` (rank correlation
        of MAE with bin order, 1.0 for strictly increasing); ``mae_ratio_top_bottom``
        (MAE of the most-uncertain bin over the least).
    """
    mae = np.array([b["mae"] for b in bins], dtype=np.float64)
    n = int(mae.size)
    inversions = int((np.diff(mae) < 0).sum()) if n > 1 else 0
    if n > 1:
        ranks = np.argsort(np.argsort(mae)).astype(np.float64)
        order = np.arange(n, dtype=np.float64)
        spearman = float(np.corrcoef(order, ranks)[0, 1])
    else:
        spearman = float("nan")
    return {
        "n_bins": n,
        "inversions": inversions,
        "monotone": bool(n > 1 and inversions == 0),
        "spearman": spearman,
        "mae_ratio_top_bottom": float(mae[-1] / mae[0]) if n > 1 and mae[0] > 0 else float("nan"),
    }


def sparsification_curve(hist: PredictorHistogram, fractions: Sequence[float],
                         band: Optional[int] = None) -> np.ndarray:
    """MAE of the pixels kept after removing the most-uncertain fraction.

    Args:
        hist: Histogram of the predictor. A histogram whose predictor is the
            absolute error itself gives the oracle curve.
        fractions: Fractions removed, each in ``[0, 1)``.
        band: One band, or None to pool.

    Returns:
        float64 array, one MAE (reflectance) per fraction. Removal walks from
        the top bin down; a partially removed fine bin gives up its error in
        proportion to the pixels taken.

    Raises:
        ValueError: A fraction outside ``[0, 1)``.
    """
    fr = np.asarray(fractions, dtype=np.float64)
    if ((fr < 0) | (fr >= 1)).any():
        raise ValueError("fractions must lie in [0, 1).")
    t = hist.totals(band)
    count = t["count"][::-1]
    sums = t["sum_abs"][::-1]
    total_n, total_s = float(count.sum()), float(sums.sum())
    cc = np.concatenate([[0.0], np.cumsum(count)])
    cs = np.concatenate([[0.0], np.cumsum(sums)])
    out = np.empty_like(fr)
    for i, f in enumerate(fr):
        r = f * total_n
        j = int(np.searchsorted(cc, r, side="right")) - 1
        j = min(j, count.size - 1)
        removed_s = cs[j] + ((r - cc[j]) * sums[j] / count[j] if count[j] > 0 else 0.0)
        out[i] = (total_s - removed_s) / (total_n - r)
    return out


def ause(pred: np.ndarray, oracle: np.ndarray, fractions: Sequence[float]) -> Dict[str, float]:
    """Area under the sparsification error, normalised by the full-set MAE.

    Args:
        pred: :func:`sparsification_curve` of the predictor.
        oracle: The same for the error itself, on the same fractions.
        fractions: The shared fraction grid, starting at 0.

    Returns:
        ``ause`` (area between predictor and oracle, in units of the full MAE
        times fraction); ``ause_random`` (the same for random removal, whose
        curve is flat at the full MAE); ``ause_ratio`` = ``ause /
        ause_random`` -- 0 is the oracle, 1 is no better than random, above 1
        is worse than random.
    """
    fr = np.asarray(fractions, dtype=np.float64)
    mae0 = float(oracle[0])
    trap = getattr(np, "trapezoid", None) or np.trapz
    a = float(trap(pred - oracle, fr)) / mae0
    a_rand = float(trap(mae0 - oracle, fr)) / mae0
    return {"ause": a, "ause_random": a_rand, "ause_ratio": a / a_rand if a_rand > 0 else float("nan")}


def score_against_control(
    predictor: PredictorHistogram,
    control: PredictorHistogram,
    oracle: PredictorHistogram,
    fractions: Sequence[float],
    band: Optional[int] = None,
) -> Dict[str, Any]:
    """AUSE of an uncertainty predictor, only ever returned beside the control's.

    Args:
        predictor: Histogram of the uncertainty predictor (TTA std, a head's
            sigma) against the error of the SR output it describes.
        control: Histogram of :func:`gradient_magnitude` of THE SAME SR output,
            against THE SAME error.
        oracle: Histogram of that error against itself.
        fractions: Sparsification grid, starting at 0.
        band: One band, or None to pool.

    Returns:
        ``predictor`` and ``control`` (each :func:`ause`'s dict),
        ``delta_ause`` (predictor minus control; negative means the predictor
        ranks error better than the edge detector), ``delta_ratio`` (the same
        on the ratio-to-random scale) and ``beats_control``.

    Raises:
        ValueError: The three histograms did not count the same pixels in every
            band. A control scored on different pixels, or against a different
            error, is not a control, and the delta would be meaningless.
    """
    reference = predictor.count.sum(dim=1)
    for name, hist in (("control", control), ("oracle", oracle)):
        if hist.n_bands != predictor.n_bands or not torch.equal(hist.count.sum(dim=1), reference):
            raise ValueError(
                f"the {name} histogram counted different pixels from the predictor "
                f"({hist.count.sum(dim=1).tolist()} vs {reference.tolist()} per band); "
                "the control must be scored on the same pixels against the same error."
            )
    fr = np.asarray(fractions, dtype=np.float64)
    oracle_curve = sparsification_curve(oracle, fr, band=band)
    pred = ause(sparsification_curve(predictor, fr, band=band), oracle_curve, fr)
    ctl = ause(sparsification_curve(control, fr, band=band), oracle_curve, fr)
    return {
        "predictor": pred,
        "control": ctl,
        "delta_ause": pred["ause"] - ctl["ause"],
        "delta_ratio": pred["ause_ratio"] - ctl["ause_ratio"],
        "beats_control": bool(pred["ause"] < ctl["ause"]),
    }


def controlled_ause_markdown(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """The AUSE table: each predictor, its control directly beneath, and the delta.

    Args:
        rows: One mapping per predictor: ``display`` (name), ``control_display``
            (what the control was computed on) and ``score``
            (:func:`score_against_control` output).

    Returns:
        Markdown table lines.

    Raises:
        KeyError: A row lacks its control score -- there is no way to render a
            predictor's AUSE alone.
    """
    lines = [
        "| predictor | AUSE ↓ | AUSE ÷ random ↓ | Δ AUSE vs control | verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        score = row["score"]
        pred, ctl = score["predictor"], score["control"]
        verdict = ("**beats** the gradient control" if score["beats_control"]
                   else "**does not beat** the gradient control")
        lines.append(
            f"| {row['display']} | {pred['ause']:.4f} | {pred['ause_ratio']:.3f} | "
            f"{score['delta_ause']:+.4f} | {verdict} |"
        )
        lines.append(
            f"| ↳ control: {row['control_display']} | {ctl['ause']:.4f} | "
            f"{ctl['ause_ratio']:.3f} | — | — |"
        )
    return lines


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Per-band local gradient magnitude -- the texture CONTROL predictor.

    Args:
        x: ``float32``, ``(B, C, H, W)``. Surface reflectance (the SR output),
            unclipped.

    Returns:
        ``float32``, same shape, >= 0: ``sqrt(dx^2 + dy^2)`` from forward
        differences, the last row/column repeating its neighbour's difference.
        Reflectance per pixel.
    """
    dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0), mode="replicate")
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1), mode="replicate")
    return torch.sqrt(dx * dx + dy * dy)


def plot_calibration(
    curves: Sequence[Mapping[str, Any]],
    fractions: Sequence[float],
    oracle: np.ndarray,
    out_path: Path,
    style: Mapping[str, Any],
    title: str,
) -> Path:
    """Two panels: MAE per equal-mass bin, and the sparsification curves.

    Left: pixels ranked by each predictor, grouped into equal-mass bins; x is
    the bin's centre quantile (both predictors share that axis even though
    their units differ), y is MAE in reflectance on a log scale. Right: MAE of
    the retained pixels, relative to the full-set MAE, as the most-uncertain
    fraction is removed; the oracle (ranked by the error itself) and random
    removal bound it.

    Args:
        curves: One mapping per predictor: ``display``, ``color`` (hex),
            ``bins`` (:func:`equal_mass_bins` output), ``sparsification``
            (array on ``fractions``).
        fractions: Shared fraction grid of the right panel.
        oracle: Oracle sparsification curve on ``fractions``.
        out_path: PNG destination; parents are created.
        style: ``surface``, ``ink``, ``ink_secondary``, ``muted`` (hex),
            ``dpi``.
        title: Figure title.

    Returns:
        ``out_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink = str(style["surface"]), str(style["ink"])
    ink2, muted = str(style["ink_secondary"]), str(style["muted"])
    fr = np.asarray(fractions, dtype=np.float64)
    mae0 = float(oracle[0])

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=surface)
    for ax in (left, right):
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=muted, alpha=0.25, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=ink2)

    ends = []
    for entry in curves:
        q = [100.0 * b["quantile_mid"] for b in entry["bins"]]
        m = [b["mae"] for b in entry["bins"]]
        color = str(entry["color"])
        left.plot(q, m, color=color, lw=2.0, marker="o", ms=5,
                  markeredgecolor=surface, markeredgewidth=1.5, zorder=3,
                  label=str(entry["display"]))
        ends.append((q[-1], m[-1], str(entry["display"])))
        right.plot(100.0 * fr, np.asarray(entry["sparsification"]) / mae0, color=color,
                   lw=2.0, zorder=3, label=str(entry["display"]))

    right.plot(100.0 * fr, oracle / mae0, color=ink, lw=1.5, ls=(0, (6, 4)), zorder=2,
               label="oracle (ranked by true error)")
    right.axhline(1.0, color=muted, lw=1.5, ls=(0, (2, 3)), zorder=1,
                  label="random removal")

    left.set_yscale("log")
    # Labelled ticks at 1-2-5 per decade: a log axis with one labelled tick
    # cannot be read as numbers.
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    left.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    left.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    left.yaxis.set_minor_formatter(NullFormatter())
    # Direct labels at the line ends, in ink. Ends that land close together
    # (the two curves meet at the top bin) are nudged apart vertically by a
    # fixed number of points; the label moves, never the data.
    placed = []
    for x_end, y_end, text in sorted(ends, key=lambda e: e[1]):
        dy = 0.0
        if placed and y_end / placed[-1][0] < 1.25:
            dy = placed[-1][1] + 12.0
        placed.append((y_end, dy))
        left.annotate(text, (x_end, y_end), xytext=(6, dy), textcoords="offset points",
                      va="center", color=ink, fontsize=9, annotation_clip=False)
    left.set_xlim(0, 125)
    left.set_xlabel("predictor quantile (equal-mass bin centre, %)", color=ink2)
    left.set_ylabel("MAE vs HR (reflectance, log scale)", color=ink2)
    left.set_title("Error per bin of the predictor", color=ink, fontsize=10, loc="left")

    right.set_ylim(0, None)
    right.set_xlabel("most-uncertain pixels removed (%)", color=ink2)
    right.set_ylabel("MAE of retained pixels / full-set MAE", color=ink2)
    right.set_title("Sparsification", color=ink, fontsize=10, loc="left")
    legend = right.legend(loc="lower left", frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(ink)

    fig.suptitle(title, color=ink, fontsize=11, x=0.01, ha="left")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=int(style["dpi"]), facecolor=surface)
    plt.close(fig)
    return out
