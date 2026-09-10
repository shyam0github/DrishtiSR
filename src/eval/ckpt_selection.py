"""Post-hoc checkpoint selection and paired significance, for the Day 3 table.

Two questions, kept apart because they fail in different ways:

1. **Which checkpoint of a run goes in the table?** ``best.pt`` is not an
   answer: the training loop chose it on PSNR over a 25-batch AMP subsample, and
   PSNR is the metric the spectral loss is EXPECTED to cost. Selecting on it
   would hand the control an advantage the comparison exists to measure.
   :func:`select_checkpoint` applies one rule, stated in config before any
   result is read, identically to every run.
2. **Is the difference between two runs real?** :func:`compare_paired` takes
   per-pair deltas over the same validation patches and reports a bootstrap CI
   on the mean delta and a Wilcoxon signed-rank p-value -- at the pair level,
   and again with the patches grouped by tile.

Why the tile-level repeat exists
--------------------------------
The 1199 validation patches come from 300 tiles, about four per tile. Patches
from one tile share a scene, an acquisition and a NAIP flight, so their deltas
are correlated, and an i.i.d. bootstrap over patches then reports a CI that is
narrower than the evidence supports. The pair-level CI is what was asked for
and is reported first; the tile-clustered CI (resample tiles, keep each tile's
patches together) is the one to trust if the two disagree. Neither is chosen
after looking at which is more flattering: both are always reported.

Every function here is pure -- frames in, numbers out, no I/O -- so the tests
can pin the statistics without data or checkpoints.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "PAIR_KEY",
    "CLUSTER_KEY",
    "paired_frame",
    "bootstrap_mean_ci",
    "wilcoxon_p",
    "compare_paired",
    "select_checkpoint",
]

#: Identifies one validation patch across evaluations. ``sample_id`` names the
#: tile and ``(lr_row, lr_col)`` the patch inside it; the grid-mode val loader
#: emits the same set every time, so a join on these is a join on pixels.
PAIR_KEY: Tuple[str, ...] = ("sample_id", "lr_row", "lr_col")

#: The column patches are grouped by for the clustered statistics: the tile.
CLUSTER_KEY = "sample_id"

# Bootstrap replicates are drawn in blocks of this many, so 10,000 replicates
# over 1199 pairs never materialise a 12-million-entry index matrix at once.
_BOOT_BLOCK = 500


def paired_frame(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    metric: str,
    key: Sequence[str] = PAIR_KEY,
) -> pd.DataFrame:
    """Join two per-patch tables on the patch identity, for one metric.

    Args:
        treatment: One row per patch, holding ``key`` columns and ``metric``.
        control: The same, for the other method.
        metric: Column to pair, e.g. ``"reflectance"`` or ``"psnr_mean"``.
        key: Identity columns. Must be unique within each frame.

    Returns:
        A frame with the ``key`` columns, ``treatment``, ``control`` and
        ``delta = treatment - control``, one row per patch. Non-finite values
        are KEPT here; :func:`compare_paired` drops and counts them.

    Raises:
        KeyError: A frame lacks ``metric`` or a key column.
        ValueError: A key is duplicated in either frame, or the two frames do
            not describe the same set of patches. Both are refused: a paired
            test over a partial or mismatched join compares different pixels
            while reporting as though it compared the same ones.
    """
    for name, frame in (("treatment", treatment), ("control", control)):
        missing = [c for c in (*key, metric) if c not in frame.columns]
        if missing:
            raise KeyError(f"{name} frame lacks columns {missing}.")
        dupes = int(frame.duplicated(list(key)).sum())
        if dupes:
            raise ValueError(
                f"{name} frame has {dupes} duplicated patch key(s) on {list(key)}; "
                "a paired comparison needs exactly one row per patch."
            )

    left = treatment[list(key) + [metric]].rename(columns={metric: "treatment"})
    right = control[list(key) + [metric]].rename(columns={metric: "control"})
    merged = left.merge(right, on=list(key), how="outer", indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} patch(es) appear in only one of the two frames "
            f"(treatment {len(treatment)} rows, control {len(control)}). The "
            "evaluations did not cover the same validation patches, so no "
            "per-pair delta is defined for them."
        )
    merged = merged.drop(columns="_merge")
    merged["delta"] = merged["treatment"].astype(float) - merged["control"].astype(float)
    return merged


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
    clusters: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap CI on the mean of ``values``.

    Args:
        values: ``(n,)`` float, finite. Typically per-patch deltas.
        n_boot: Bootstrap replicates.
        ci: Coverage, e.g. ``0.95``.
        rng: Seeded generator; the caller owns the seed.
        clusters: Optional ``(n,)`` labels. When given, whole clusters are
            resampled with replacement and each replicate's statistic is the
            mean over all values in the drawn clusters -- the patch-weighted
            mean, so it estimates the same quantity as the unclustered case.

    Returns:
        ``(low, high)``.

    Raises:
        ValueError: ``values`` is empty or non-finite, ``ci`` is not in
            ``(0, 1)``, or ``clusters`` has the wrong length.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"values must be a non-empty 1-D array; got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contains non-finite entries; drop them before bootstrapping.")
    if not 0.0 < float(ci) < 1.0:
        raise ValueError(f"ci must be in (0, 1); got {ci!r}.")

    if clusters is None:
        units_sum = values
        units_count = np.ones_like(values)
    else:
        clusters = np.asarray(clusters)
        if clusters.shape != values.shape:
            raise ValueError(
                f"clusters has shape {clusters.shape}, values {values.shape}."
            )
        _, inverse = np.unique(clusters, return_inverse=True)
        units_sum = np.bincount(inverse, weights=values)
        units_count = np.bincount(inverse).astype(np.float64)

    k = units_sum.size
    means = np.empty(int(n_boot), dtype=np.float64)
    for start in range(0, int(n_boot), _BOOT_BLOCK):
        stop = min(start + _BOOT_BLOCK, int(n_boot))
        draw = rng.integers(0, k, size=(stop - start, k))
        means[start:stop] = units_sum[draw].sum(axis=1) / units_count[draw].sum(axis=1)

    tail = (1.0 - float(ci)) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high)


def wilcoxon_p(values: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value for ``values`` centred on zero.

    Args:
        values: ``(n,)`` finite paired differences.

    Returns:
        The p-value. ``1.0`` when every difference is exactly zero -- scipy
        raises there, and "no difference at all" is the correct reading, not
        an error. Zero differences are otherwise dropped (``zero_method
        ="wilcox"``), scipy's default.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.any(values != 0.0):
        return 1.0
    return float(stats.wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)


def compare_paired(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    metric: str,
    better: str,
    n_boot: int,
    ci: float,
    seed: int,
    key: Sequence[str] = PAIR_KEY,
    cluster_key: str = CLUSTER_KEY,
) -> Dict[str, Any]:
    """Per-pair ``treatment - control`` on one metric, with its uncertainty.

    Args:
        treatment: Per-patch frame of the treatment (Run B).
        control: Per-patch frame of the control (Run A2), same patches.
        metric: Column to compare.
        better: ``"lower"`` or ``"higher"``: which sign of the delta favours
            the treatment.
        n_boot: Bootstrap replicates.
        ci: CI coverage, e.g. 0.95.
        seed: Seed for the bootstrap generator. Each metric reseeds from it, so
            adding a metric never moves another metric's CI.
        key: Patch identity columns.
        cluster_key: Column grouping patches into tiles.

    Returns:
        ``{metric, better, n, n_dropped_nonfinite, n_clusters, mean_treatment,
        mean_control, mean_delta, median_delta, frac_treatment_better,
        ci_pair, ci_tile, p_wilcoxon_pair, p_wilcoxon_tile,
        straddles_zero_pair, straddles_zero_tile, verdict}``. ``verdict`` is
        ``"treatment better"``, ``"control better"`` or ``"not resolved"``,
        judged on the TILE CI -- the conservative one -- so a claim never rests
        on the interval that ignores within-tile correlation.

    Raises:
        ValueError: ``better`` is not ``"lower"``/``"higher"``, or no finite
            pair survives.
    """
    if better not in ("lower", "higher"):
        raise ValueError(f"better must be 'lower' or 'higher'; got {better!r}.")

    pairs = paired_frame(treatment, control, metric, key=key)
    finite = np.isfinite(pairs["treatment"].astype(float)) & np.isfinite(
        pairs["control"].astype(float)
    )
    dropped = int((~finite).sum())
    pairs = pairs[finite]
    if pairs.empty:
        raise ValueError(f"no pair has a finite {metric!r} in both frames.")

    deltas = pairs["delta"].to_numpy(dtype=np.float64)
    tiles = pairs[cluster_key].astype(str).to_numpy()
    tile_means = pd.Series(deltas).groupby(tiles).mean().to_numpy()

    ci_pair = bootstrap_mean_ci(deltas, n_boot, ci, np.random.default_rng(seed))
    ci_tile = bootstrap_mean_ci(
        deltas, n_boot, ci, np.random.default_rng(seed), clusters=tiles
    )
    sign = -1.0 if better == "lower" else 1.0

    def straddles(interval: Tuple[float, float]) -> bool:
        return interval[0] <= 0.0 <= interval[1]

    if straddles(ci_tile):
        verdict = "not resolved"
    else:
        # Both ends share a sign, so the sign of either end, oriented by the
        # metric's direction, says who wins.
        verdict = "treatment better" if sign * ci_tile[0] > 0.0 else "control better"

    return {
        "metric": metric,
        "better": better,
        "n": int(deltas.size),
        "n_dropped_nonfinite": dropped,
        "n_clusters": int(tile_means.size),
        "mean_treatment": float(pairs["treatment"].astype(float).mean()),
        "mean_control": float(pairs["control"].astype(float).mean()),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "frac_treatment_better": float(np.mean(sign * deltas > 0.0)),
        "ci": float(ci),
        "ci_pair": [ci_pair[0], ci_pair[1]],
        "ci_tile": [ci_tile[0], ci_tile[1]],
        "p_wilcoxon_pair": wilcoxon_p(deltas),
        "p_wilcoxon_tile": wilcoxon_p(tile_means),
        "straddles_zero_pair": straddles(ci_pair),
        "straddles_zero_tile": straddles(ci_tile),
        "verdict": verdict,
    }


def select_checkpoint(
    candidates: Mapping[str, pd.DataFrame],
    metric: str,
    tie_metric: str,
    tie_better: str,
    n_boot: int,
    ci: float,
    seed: int,
    key: Sequence[str] = PAIR_KEY,
) -> Dict[str, Any]:
    """Apply the Day 3 selection rule to one run's checkpoints.

    THE RULE, fixed before any full-split result was read and applied
    identically to every run: pick the checkpoint with the LOWEST mean
    ``metric`` (consistency error); every other checkpoint whose paired
    bootstrap CI on ``(candidate - minimum)`` contains zero is TIED with it;
    among the tied set, pick the one best on ``tie_metric``.

    A CI-based tie is used because an exact tie never happens on a continuous
    metric, so "tie-broken by LPIPS" would otherwise be a rule that never fires
    -- and a 0.1% consistency difference inside the noise would silently decide
    the table.

    Args:
        candidates: ``{checkpoint label: per-patch frame}``, all over the same
            patches. Labels are opaque here; the caller maps them back to files.
        metric: Consistency column, lower is better.
        tie_metric: Tie-break column.
        tie_better: ``"lower"`` or ``"higher"`` for ``tie_metric``.
        n_boot: Bootstrap replicates for the tie test.
        ci: CI coverage.
        seed: Bootstrap seed.
        key: Patch identity columns.

    Returns:
        ``{selected, minimum, rows, rule}``. ``rows`` has one entry per
        candidate, in the input order, with its mean ``metric``, mean
        ``tie_metric``, delta and CI against the minimum, and whether it was
        tied -- the whole curve, so the selection can be audited.

    Raises:
        ValueError: No candidates, ``tie_better`` invalid, or a candidate has a
            non-finite mean on either metric.
    """
    if not candidates:
        raise ValueError("no checkpoints to select from.")
    if tie_better not in ("lower", "higher"):
        raise ValueError(f"tie_better must be 'lower' or 'higher'; got {tie_better!r}.")

    means = {}
    tie_means = {}
    for label, frame in candidates.items():
        for column, store in ((metric, means), (tie_metric, tie_means)):
            if column not in frame.columns:
                raise KeyError(f"checkpoint {label!r} has no {column!r} column.")
            value = float(pd.to_numeric(frame[column], errors="coerce").mean())
            if not math.isfinite(value):
                raise ValueError(
                    f"checkpoint {label!r} has a non-finite mean {column!r}; the "
                    "rule cannot rank it and will not guess."
                )
            store[label] = value

    minimum = min(means, key=means.get)
    rows = []
    tied = []
    for label, frame in candidates.items():
        entry: Dict[str, Any] = {
            "label": label,
            f"mean_{metric}": means[label],
            f"mean_{tie_metric}": tie_means[label],
            "delta_vs_min": 0.0,
            "ci_vs_min": [0.0, 0.0],
            "tied_with_min": True,
            "is_min": label == minimum,
        }
        if label != minimum:
            pairs = paired_frame(frame, candidates[minimum], metric, key=key)
            deltas = pairs["delta"].to_numpy(dtype=np.float64)
            deltas = deltas[np.isfinite(deltas)]
            low, high = bootstrap_mean_ci(deltas, n_boot, ci, np.random.default_rng(seed))
            entry["delta_vs_min"] = float(np.mean(deltas))
            entry["ci_vs_min"] = [low, high]
            entry["tied_with_min"] = bool(low <= 0.0 <= high)
        if entry["tied_with_min"]:
            tied.append(label)
        rows.append(entry)

    pick = min if tie_better == "lower" else max
    selected = pick(tied, key=tie_means.get)
    for entry in rows:
        entry["selected"] = entry["label"] == selected

    return {
        "selected": selected,
        "minimum": minimum,
        "tied_set": tied,
        "rows": rows,
        "rule": {
            "metric": metric,
            "metric_better": "lower",
            "tie_metric": tie_metric,
            "tie_better": tie_better,
            "tie_definition": (
                f"paired bootstrap {ci:.0%} CI on (candidate - minimum) of "
                f"{metric} contains 0 (pair-level, {n_boot} replicates)"
            ),
        },
    }
