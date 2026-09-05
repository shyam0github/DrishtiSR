"""Evaluation harness: run a super-resolver over a split and summarise it.

One class, :class:`Evaluator`, which takes a dataloader and a callable
``sr_fn(lr) -> sr``. A bicubic baseline, a trained checkpoint, and an
INT8-quantised ONNX session are all just different callables, so all three are
measured by identical code and their numbers are directly comparable. Nothing
about this module knows what produced the pixels.

Why the percentiles are not optional
------------------------------------
A mean PSNR is the one number that cannot fail a model. Validation sets are
dominated by easy tiles -- uniform crops, farmland, forest -- and a method that
falls apart on the 5% of tiles containing a harbour, a city edge, or a cloud
boundary still posts a respectable mean. Those are exactly the tiles a judge
zooms into. So every metric is summarised with mean, standard deviation, and the
5th and 95th percentiles (``cfg.metrics.percentiles``), the per-sample table is
always written to CSV, and the worst sample per metric is named in the summary.
If the p5 PSNR is 6 dB below the mean, that is the result, not a footnote.

Non-finite values are counted, never dropped
--------------------------------------------
``psnr`` of a perfectly reconstructed band is ``+inf``; ``ergas`` and ``sam`` can
be ``nan`` under their configured policies for degenerate patches. Summary
statistics are computed over the finite entries and the count of non-finite ones
is reported per metric, with a warning in the log. A silently dropped NaN is how
an evaluation ends up describing half the validation set.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.metrics.image_quality import (
    LPIPS_CAVEAT,
    ergas,
    lpips,
    psnr,
    rgb_band_indices,
    sam,
    ssim,
)
from src.utils.logging import get_logger

__all__ = ["Evaluator", "EvaluationResult", "summarise", "IDENTITY_COLUMNS"]


# Per-sample columns that identify the patch rather than score it. Excluded from
# the summary statistics; carried into the CSV so any row can be traced back to
# a tile and re-rendered.
IDENTITY_COLUMNS = (
    "sample_id",
    "dataset_index",
    "split",
    "lr_row",
    "lr_col",
    "hr_row",
    "hr_col",
    "filter_reason",
)

# Metrics where a smaller number is better. Used to pick the worst sample and to
# annotate the markdown table, so a reader never has to remember which way each
# metric points.
LOWER_IS_BETTER = ("sam_mean_deg", "sam_p95_deg", "ergas", "lpips")


def summarise(
    frame: pd.DataFrame,
    percentiles: Sequence[float] = (5.0, 95.0),
    logger: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Summarise every numeric metric column of a per-sample table.

    Args:
        frame: Per-sample metrics, one row per evaluated patch. Columns named in
            :data:`IDENTITY_COLUMNS` are skipped, as are non-numeric columns.
        percentiles: Percentiles to report, from ``cfg.metrics.percentiles``.
            Reported under keys ``"p5"``, ``"p95"``, ... formatted from the
            values themselves.
        logger: Logger for the non-finite warning. Optional.

    Returns:
        ``{column: {"count", "num_finite", "num_nonfinite", "mean", "std",
        "min", "p5", "median", "p95", "max", "worst_sample_id",
        "worst_value"}}``. Statistics are computed over the finite entries;
        ``std`` uses ``ddof=1`` and is ``nan`` for a single sample. A column with
        no finite entries reports ``nan`` throughout rather than raising, and its
        ``num_nonfinite`` says why.

    Raises:
        ValueError: ``frame`` is empty, or a requested percentile is outside
            ``[0, 100]``.
    """
    if len(frame) == 0:
        raise ValueError(
            "Cannot summarise an empty per-sample table: the evaluation produced "
            "no rows. Check that the split is non-empty and that max_batches is "
            "not 0."
        )
    for value in percentiles:
        if not 0.0 <= float(value) <= 100.0:
            raise ValueError(
                f"cfg.metrics.percentiles must lie in [0, 100]; got {value!r}."
            )

    summary: Dict[str, Dict[str, Any]] = {}
    for column in frame.columns:
        if column in IDENTITY_COLUMNS:
            continue
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        values = series.to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        num_finite = int(finite.sum())
        num_nonfinite = int(values.size - num_finite)

        entry: Dict[str, Any] = {
            "count": int(values.size),
            "num_finite": num_finite,
            "num_nonfinite": num_nonfinite,
        }

        if num_finite == 0:
            entry.update(
                {
                    key: float("nan")
                    for key in ("mean", "std", "min", "median", "max", "worst_value")
                }
            )
            for value in percentiles:
                entry[f"p{float(value):g}"] = float("nan")
            entry["worst_sample_id"] = None
            summary[column] = entry
            if logger is not None:
                logger.warning(
                    "Metric %r has no finite values across %d samples; its "
                    "summary is all NaN.",
                    column,
                    values.size,
                )
            continue

        kept = values[finite]
        entry["mean"] = float(kept.mean())
        entry["std"] = float(kept.std(ddof=1)) if kept.size > 1 else float("nan")
        entry["min"] = float(kept.min())
        entry["median"] = float(np.median(kept))
        entry["max"] = float(kept.max())
        for value in percentiles:
            entry[f"p{float(value):g}"] = float(np.percentile(kept, float(value)))

        # "Worst" means worst for this metric's direction, so the summary points
        # at the sample a reader should open first.
        lower_better = column in LOWER_IS_BETTER
        positions = np.flatnonzero(finite)
        worst_position = int(
            positions[np.argmax(kept)] if lower_better else positions[np.argmin(kept)]
        )
        entry["worst_value"] = float(values[worst_position])
        entry["worst_sample_id"] = (
            str(frame["sample_id"].iloc[worst_position])
            if "sample_id" in frame.columns
            else None
        )

        if num_nonfinite and logger is not None:
            logger.warning(
                "Metric %r has %d non-finite value(s) of %d samples. They are "
                "EXCLUDED from mean/std/percentiles and counted in "
                "num_nonfinite. For PSNR this means a band was reconstructed "
                "exactly (+inf); for SAM/ERGAS it means a degenerate patch under "
                "the configured policy.",
                column,
                num_nonfinite,
                values.size,
            )

        summary[column] = entry

    return summary


@dataclass
class EvaluationResult:
    """Everything one evaluation run produced.

    Attributes:
        name: Method name, e.g. ``"bicubic"``. Used in filenames and headings.
        per_sample: One row per evaluated patch: the identity columns plus every
            metric. This is the artefact that survives -- percentiles, box plots,
            and failure-case hunting all come from it.
        summary: :func:`summarise` output, keyed by metric column.
        settings: What the numbers were computed with -- dataset, split, band
            names, data range, scale, SSIM window parameters, LPIPS backbone,
            policies, thread count. A metric without its settings is not
            reproducible.
        qualitative: Samples retained for figures. Each entry is a dict with
            ``sample_id``, ``lr``, ``sr``, ``hr`` (float32 ``(C, H, W)``
            reflectance) and ``sam_map`` (float64 ``(H, W)`` degrees). Empty
            unless ``collect_samples`` was passed to :meth:`Evaluator.run`.
    """

    name: str
    per_sample: pd.DataFrame
    summary: Dict[str, Dict[str, Any]]
    settings: Dict[str, Any]
    qualitative: List[Dict[str, Any]] = field(default_factory=list)

    def to_csv(self, path: Any) -> Path:
        """Write the per-sample table, creating parent directories.

        Args:
            path: Destination ``.csv``.

        Returns:
            The path written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.per_sample.to_csv(out, index=False)
        return out

    def to_json(self, path: Any) -> Path:
        """Write the summary, settings, and LPIPS caveat as JSON.

        The per-sample rows are deliberately not duplicated here; they are the
        CSV's job, and a JSON with 400 rows in it does not get read.

        Args:
            path: Destination ``.json``.

        Returns:
            The path written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.name,
            "written_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "num_samples": int(len(self.per_sample)),
            "settings": self.settings,
            "summary": self.summary,
            "lpips_caveat": LPIPS_CAVEAT,
        }
        with out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False, default=str)
        return out

    def to_markdown(self, columns: Optional[Sequence[str]] = None) -> str:
        """Render the summary as a markdown table ready to paste into the report.

        Args:
            columns: Metric columns to show, in order. ``None`` shows the
                headline metrics that are present, then everything else.

        Returns:
            A markdown string: a heading line, the table (metric, direction,
            mean, std, p5, p95, min, max, n), and -- whenever LPIPS is in the
            table -- the :data:`LPIPS_CAVEAT` footnote. The caveat travels with
            the number by construction, so it cannot be lost between this
            function and the report.
        """
        headline = [
            "psnr_mean",
            "ssim_mean",
            "sam_mean_deg",
            "sam_p95_deg",
            "ergas",
            "lpips",
        ]
        if columns is None:
            present = [c for c in headline if c in self.summary]
            rest = [c for c in self.summary if c not in present]
            columns = present + rest
        columns = [c for c in columns if c in self.summary]

        percentile_keys = [
            key
            for key in (self.summary[columns[0]] if columns else {})
            if key.startswith("p") and key not in {"p50"}
        ]
        percentile_keys.sort(key=lambda k: float(k[1:]))

        header = ["metric", "better", "mean", "std"]
        header += [key for key in percentile_keys]
        header += ["min", "max", "n"]

        lines = [
            f"**{self.name}** -- {self.settings.get('dataset', '?')} / "
            f"{self.settings.get('split', '?')} split, "
            f"{len(self.per_sample)} patches, x{self.settings.get('scale', '?')}",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|",
        ]
        for column in columns:
            entry = self.summary[column]
            better = "lower" if column in LOWER_IS_BETTER else "higher"
            cells = [column, better, _fmt(entry["mean"]), _fmt(entry["std"])]
            cells += [_fmt(entry.get(key)) for key in percentile_keys]
            cells += [
                _fmt(entry["min"]),
                _fmt(entry["max"]),
                str(entry["num_finite"])
                + (
                    f" (+{entry['num_nonfinite']} non-finite)"
                    if entry["num_nonfinite"]
                    else ""
                ),
            ]
            lines.append("| " + " | ".join(cells) + " |")

        if any(c == "lpips" for c in columns):
            lines += ["", f"> **LPIPS caveat.** {LPIPS_CAVEAT}"]
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    """Format a summary number for the markdown table."""
    if value is None:
        return "-"
    number = float(value)
    if not np.isfinite(number):
        return "inf" if number > 0 else ("-inf" if number < 0 else "nan")
    if abs(number) >= 1000 or (number != 0 and abs(number) < 1e-3):
        return f"{number:.3e}"
    return f"{number:.4f}"


class Evaluator:
    """Run a super-resolver over a dataloader and score every sample.

    The same object evaluates a baseline, a checkpoint, and a quantised ONNX
    session; only ``sr_fn`` changes. Metrics, settings, and the CSV schema are
    therefore identical across all of them, which is the only way an ablation
    table means anything.

    All configuration comes from ``cfg.metrics``: which metrics run, the fixed
    reflectance ``data_range``, the SSIM window, the LPIPS backbone and RGB band
    names, the SAM and ERGAS degenerate-value policies, and the reported
    percentiles.
    """

    def __init__(self, cfg: Any, logger: Any = None, device: Any = None) -> None:
        """
        Args:
            cfg: The loaded config. Reads ``metrics``, ``dataset.bands``,
                ``sr.scale``, and ``paths.log_file``.
            logger: Logger; created if None.
            device: Device for ``sr_fn``'s input. ``None`` selects CUDA when it
                is available and CPU otherwise -- CPU is the working default, not
                a degraded fallback, since the deployment target is a 6-thread
                CPU. Metrics always run on CPU in float64.

        Raises:
            KeyError: ``cfg`` has no ``metrics`` block, or names an unknown
                metric in ``metrics.enabled``.
        """
        self.cfg = cfg
        self.logger = logger or get_logger(
            "metrics.aggregate", log_file=cfg["paths"]["log_file"]
        )

        try:
            metrics_cfg = cfg["metrics"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                "Config is missing the 'metrics' section. Load configs/base.yaml "
                "or a config that merges over it."
            ) from exc

        self.bands = [str(b) for b in cfg["dataset"]["bands"]]
        self.scale = int(cfg["sr"]["scale"])
        self.data_range = float(metrics_cfg["data_range"])
        self.percentiles = [float(p) for p in metrics_cfg["percentiles"]]

        known = {"psnr", "ssim", "sam", "ergas", "lpips"}
        self.enabled = [str(m) for m in metrics_cfg["enabled"]]
        unknown = [m for m in self.enabled if m not in known]
        if unknown:
            raise KeyError(
                f"cfg.metrics.enabled names unknown metric(s) {unknown}. "
                f"Available: {sorted(known)}."
            )

        self.ssim_cfg = metrics_cfg["ssim"]
        self.sam_policy = str(metrics_cfg["sam"]["zero_vector_policy"])
        self.ergas_policy = str(metrics_cfg["ergas"]["zero_mean_policy"])
        self.sam_percentile = float(metrics_cfg["sam"]["report_percentile"])

        self.lpips_cfg = metrics_cfg["lpips"]
        self.lpips_enabled = "lpips" in self.enabled and bool(self.lpips_cfg["enabled"])
        self.lpips_indices = (
            rgb_band_indices(cfg) if self.lpips_enabled else None
        )
        if "lpips" in self.enabled and not bool(self.lpips_cfg["enabled"]):
            self.logger.info(
                "LPIPS is listed in cfg.metrics.enabled but "
                "cfg.metrics.lpips.enabled is false, so it is SKIPPED. It needs "
                "pretrained weights downloaded on first use, which a no-internet "
                "run (including --smoke) cannot do."
            )

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # -- per-batch scoring -------------------------------------------------

    def metrics_for_batch(
        self, sr: torch.Tensor, hr: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Score one batch, returning a ``(B,)`` array per metric column.

        Args:
            sr: ``(B, C, H, W)`` float32 surface reflectance produced by
                ``sr_fn``, nominally ``[0, 1]``, unclipped.
            hr: ``(B, C, H, W)`` reference reflectance, same shape, band order,
                and units.

        Returns:
            ``{column_name: (B,) float64 array}``. Column names are
            ``psnr_mean``, ``psnr_<band>`` per band, ``ssim_mean``,
            ``ssim_<band>``, ``sam_mean_deg``, ``sam_p<pct>_deg``,
            ``sam_undefined_px``, ``ergas``, ``lpips``, ``lpips_clipped_frac``,
            plus the reflectance sanity columns ``hr_reflectance_mean``,
            ``sr_reflectance_mean``, ``sr_reflectance_min``,
            ``sr_reflectance_max`` -- those last three are how an unannounced
            clip or a scaling error in a future model gets noticed.

        Raises:
            ValueError: ``sr`` and ``hr`` disagree in shape, or an input is
                non-finite (raised by the metric functions).
        """
        columns: Dict[str, np.ndarray] = {}

        if "psnr" in self.enabled:
            result = psnr(sr, hr, data_range=self.data_range)
            columns["psnr_mean"] = np.asarray(result.mean, dtype=np.float64)
            for index, band in enumerate(self.bands):
                columns[f"psnr_{band}"] = result.per_band[:, index]

        if "ssim" in self.enabled:
            result = ssim(
                sr,
                hr,
                data_range=self.data_range,
                gaussian_weights=bool(self.ssim_cfg["gaussian_weights"]),
                sigma=float(self.ssim_cfg["sigma"]),
                win_size=(
                    None
                    if self.ssim_cfg["win_size"] is None
                    else int(self.ssim_cfg["win_size"])
                ),
            )
            columns["ssim_mean"] = np.asarray(result.mean, dtype=np.float64)
            for index, band in enumerate(self.bands):
                columns[f"ssim_{band}"] = result.per_band[:, index]

        if "sam" in self.enabled:
            result = sam(sr, hr, zero_vector_policy=self.sam_policy)
            maps = result.map_deg.reshape(result.map_deg.shape[0], -1)
            columns["sam_mean_deg"] = np.asarray(result.mean_deg, dtype=np.float64)
            with np.errstate(invalid="ignore"):
                columns[f"sam_p{self.sam_percentile:g}_deg"] = np.nanpercentile(
                    maps, self.sam_percentile, axis=1
                )
            columns["sam_undefined_px"] = result.num_undefined.astype(np.float64)

        if "ergas" in self.enabled:
            columns["ergas"] = np.asarray(
                ergas(sr, hr, self.scale, zero_mean_policy=self.ergas_policy),
                dtype=np.float64,
            )

        if self.lpips_enabled:
            result = lpips(
                sr,
                hr,
                rgb_indices=self.lpips_indices,
                net=str(self.lpips_cfg["net"]),
                data_range=self.data_range,
                clip_reflectance=bool(self.lpips_cfg["clip_reflectance"]),
            )
            columns["lpips"] = np.asarray(result.distance, dtype=np.float64)
            columns["lpips_clipped_frac"] = np.asarray(
                result.clipped_fraction, dtype=np.float64
            )

        sr64 = sr.detach().to(torch.float64)
        hr64 = hr.detach().to(torch.float64)
        columns["hr_reflectance_mean"] = hr64.mean(dim=(1, 2, 3)).numpy()
        columns["sr_reflectance_mean"] = sr64.mean(dim=(1, 2, 3)).numpy()
        columns["sr_reflectance_min"] = sr64.amin(dim=(1, 2, 3)).numpy()
        columns["sr_reflectance_max"] = sr64.amax(dim=(1, 2, 3)).numpy()

        return columns

    # -- the run -----------------------------------------------------------

    def run(
        self,
        dataloader: Any,
        sr_fn: Callable[[torch.Tensor], Any],
        name: str,
        max_batches: Optional[int] = None,
        collect_samples: int = 0,
        split: str = "",
    ) -> EvaluationResult:
        """Evaluate ``sr_fn`` over ``dataloader`` and summarise the result.

        Args:
            dataloader: Yields the batch dicts produced by
                :class:`src.data.loader.PatchDataset` -- ``lr`` ``(B, C, h, w)``
                and ``hr`` ``(B, C, h*scale, w*scale)`` float32 surface
                reflectance, plus the identity fields. Must be the deterministic
                validation loader (``shuffle=False``) if two runs are to be
                compared row by row.
            sr_fn: ``sr_fn(lr) -> sr``. Receives the LR batch as a float32 tensor
                on ``self.device`` and must return ``(B, C, h*scale, w*scale)``
                reflectance in the same units and band order -- a tensor or a
                NumPy array. It is called inside ``torch.no_grad()``.
            name: Method name for the result, e.g. ``"bicubic"``.
            max_batches: Stop after this many batches. For smoke runs only; the
                result records it and the summary is over the batches actually
                seen.
            collect_samples: Retain this many samples, spread evenly across the
                run, for the qualitative figure. Retained arrays are copies on
                the CPU; 6 patches costs a few MB.
            split: Split name recorded in the settings, e.g. ``"val"``.

        Returns:
            An :class:`EvaluationResult`.

        Raises:
            ValueError: ``sr_fn`` returned the wrong shape or a non-tensor, the
                loader produced no batches, or a batch is missing ``lr``/``hr``.
        """
        started = time.perf_counter()
        rows: List[Dict[str, Any]] = []
        qualitative: List[Dict[str, Any]] = []
        num_batches = 0

        total_batches = _loader_length(dataloader)
        if max_batches is not None:
            total_batches = (
                min(total_batches, int(max_batches))
                if total_batches is not None
                else int(max_batches)
            )
        collect_every = _collect_stride(total_batches, collect_samples)

        self.logger.info(
            "Evaluating %r on %s: metrics=%s, data_range=%.4g, scale=x%d, "
            "device=%s, torch threads=%d.",
            name,
            split or "?",
            [m for m in self.enabled if m != "lpips" or self.lpips_enabled],
            self.data_range,
            self.scale,
            self.device,
            torch.get_num_threads(),
        )

        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= int(max_batches):
                self.logger.warning(
                    "Stopping after max_batches=%d. The summary describes those "
                    "batches only, NOT the whole split.",
                    int(max_batches),
                )
                break

            lr, hr = _require_pair(batch, batch_index)
            lr_device = lr.to(self.device, dtype=torch.float32)

            call_started = time.perf_counter()
            with torch.no_grad():
                sr_out = sr_fn(lr_device)
            elapsed_ms = (time.perf_counter() - call_started) * 1000.0

            sr = _as_cpu_tensor(sr_out, name)
            _check_output_shape(sr, lr, hr, self.scale, name)

            columns = self.metrics_for_batch(sr, hr)
            batch_size = int(hr.shape[0])
            per_sample_ms = elapsed_ms / batch_size

            for i in range(batch_size):
                row: Dict[str, Any] = {
                    "sample_id": _field(batch, "sample_id", i, ""),
                    "dataset_index": _field(batch, "dataset_index", i, -1),
                    "split": _field(batch, "split", i, split),
                    "lr_row": _field(batch, "lr_row", i, -1),
                    "lr_col": _field(batch, "lr_col", i, -1),
                    "hr_row": _field(batch, "hr_row", i, -1),
                    "hr_col": _field(batch, "hr_col", i, -1),
                    "filter_reason": _field(batch, "filter_reason", i, ""),
                }
                row.update({key: float(values[i]) for key, values in columns.items()})
                row["sr_time_ms"] = per_sample_ms
                rows.append(row)

            if (
                collect_every
                and batch_index % collect_every == 0
                and len(qualitative) < collect_samples
            ):
                # The first sample of this batch. Recomputed unbatched so the
                # retained map is (H, W) -- the shape the figure code expects.
                first_row = rows[-batch_size]
                sam_result = sam(sr[0], hr[0], zero_vector_policy=self.sam_policy)
                qualitative.append(
                    {
                        "sample_id": _field(batch, "sample_id", 0, ""),
                        "lr": lr[0].detach().cpu().numpy().astype(np.float32),
                        "sr": sr[0].numpy().astype(np.float32),
                        "hr": hr[0].detach().cpu().numpy().astype(np.float32),
                        "sam_map": sam_result.map_deg,
                        "psnr_mean": first_row.get("psnr_mean", float("nan")),
                        "sam_mean_deg": float(sam_result.mean_deg),
                    }
                )

            num_batches += 1

        if not rows:
            raise ValueError(
                f"Evaluation of {name!r} produced no samples: the dataloader was "
                "empty. Check the split selection and cfg.loader.cached_only."
            )

        frame = pd.DataFrame(rows)
        summary = summarise(frame, self.percentiles, logger=self.logger)
        duration = time.perf_counter() - started

        settings = {
            "method": name,
            "dataset": str(self.cfg["dataset"]["name"]),
            "split": split,
            "bands": list(self.bands),
            "scale": self.scale,
            "data_range_reflectance": self.data_range,
            "metrics_enabled": list(self.enabled),
            "lpips_computed": bool(self.lpips_enabled),
            "lpips_net": str(self.lpips_cfg["net"]) if self.lpips_enabled else None,
            "lpips_rgb_bands": (
                [self.bands[i] for i in self.lpips_indices]
                if self.lpips_indices
                else None
            ),
            "ssim": {
                "implementation": "skimage.metrics.structural_similarity",
                "gaussian_weights": bool(self.ssim_cfg["gaussian_weights"]),
                "sigma": float(self.ssim_cfg["sigma"]),
                "win_size": self.ssim_cfg["win_size"],
                "use_sample_covariance": False,
            },
            "sam_zero_vector_policy": self.sam_policy,
            "sam_report_percentile": self.sam_percentile,
            "ergas_zero_mean_policy": self.ergas_policy,
            "percentiles": list(self.percentiles),
            "num_batches": num_batches,
            "max_batches": max_batches,
            "device": str(self.device),
            "torch_threads": int(torch.get_num_threads()),
            "wall_time_s": round(duration, 3),
        }

        self.logger.info(
            "%s: %d samples over %d batches in %.1f s (%.1f ms/sample in sr_fn).",
            name,
            len(frame),
            num_batches,
            duration,
            float(frame["sr_time_ms"].mean()),
        )
        for column in ("psnr_mean", "ssim_mean", "sam_mean_deg", "ergas"):
            if column in summary:
                entry = summary[column]
                self.logger.info(
                    "  %-14s mean %.4f  std %.4f  p%g %.4f  p%g %.4f  worst %s "
                    "(%.4f)",
                    column,
                    entry["mean"],
                    entry["std"],
                    self.percentiles[0],
                    entry[f"p{self.percentiles[0]:g}"],
                    self.percentiles[-1],
                    entry[f"p{self.percentiles[-1]:g}"],
                    entry["worst_sample_id"],
                    entry["worst_value"],
                )

        return EvaluationResult(
            name=name,
            per_sample=frame,
            summary=summary,
            settings=settings,
            qualitative=qualitative,
        )


# -- helpers ---------------------------------------------------------------


def _loader_length(dataloader: Any) -> Optional[int]:
    """Number of batches, or None for a loader that cannot report one."""
    try:
        return int(len(dataloader))
    except TypeError:
        return None


def _collect_stride(total_batches: Optional[int], collect_samples: int) -> int:
    """Batch stride that spreads ``collect_samples`` evenly over the run.

    Returns 0 when nothing is to be collected. Spreading rather than taking the
    first N matters: the first N patches of a grid-ordered validation set all
    come from the same tile, so a figure built from them shows one scene four
    times.
    """
    wanted = int(collect_samples)
    if wanted <= 0:
        return 0
    if not total_batches:
        return 1
    return max(1, total_batches // wanted)


def _require_pair(batch: Mapping[str, Any], index: int):
    """Extract ``lr`` and ``hr`` from a batch, failing loudly if absent."""
    missing = [key for key in ("lr", "hr") if key not in batch]
    if missing:
        raise ValueError(
            f"Batch {index} is missing {missing}. The Evaluator expects the "
            "batch dicts produced by src.data.loader.PatchDataset."
        )
    return batch["lr"], batch["hr"]


def _as_cpu_tensor(value: Any, name: str) -> torch.Tensor:
    """Coerce an ``sr_fn`` return value to a CPU float32 tensor."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
    raise ValueError(
        f"sr_fn for {name!r} returned {type(value).__name__}; it must return a "
        "torch.Tensor or numpy.ndarray of (B, C, H, W) surface reflectance."
    )


def _check_output_shape(
    sr: torch.Tensor, lr: torch.Tensor, hr: torch.Tensor, scale: int, name: str
) -> None:
    """Assert the SR output is exactly the target's shape.

    Raises:
        ValueError: The output rank or size is wrong. Never resamples to fit --
            a size mismatch means the scale factor or the patch geometry is
            wrong, and papering over it produces a plausible metric for the wrong
            comparison.
    """
    if sr.shape != hr.shape:
        raise ValueError(
            f"sr_fn for {name!r} returned {tuple(sr.shape)} but the target is "
            f"{tuple(hr.shape)} (LR input was {tuple(lr.shape)}, "
            f"cfg.sr.scale={scale}). The Evaluator does not resample to fit: a "
            "size mismatch means the scale factor or the patch geometry is wrong."
        )


def _field(batch: Mapping[str, Any], key: str, index: int, default: Any) -> Any:
    """Read element ``index`` of a collated batch field, with a default.

    The dataloader collates strings into lists and ints into tensors; this
    normalises both to a plain Python value for the DataFrame.
    """
    if key not in batch:
        return default
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value[index].item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value
