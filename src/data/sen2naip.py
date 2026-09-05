"""SEN2NAIPv2 via tacoreader -- the primary dataset.

Every fact encoded here was measured against the live dataset (record
``NA5120_E1183N0757__m_3912321_nw_10_060_20220710`` of
``sen2naipv2-crosssensor``), not read off the dataset card. Where the card and
the data disagree, the measurement wins and the discrepancy is noted.

Measured facts
--------------
- **API**: the card's ``tacoreader.load(...)`` example is the *v1* API. Current
  tacoreader (2.x) exposes an incompatible v2 ``load`` returning a ``TacoDataset``
  with no ``.read()``, and the ``tacofoundation:`` protocol exists only in v1.
  The working call is ``tacoreader.v1.load("tacofoundation:<subset>")``.
- **dtype**: ``uint16``, both LR and HR.
- **Reflectance divisor**: ``10000``. The card's ``/3000`` is a display stretch.
  Measured band means on a forest record were R=578.7 G=565.5 B=343.5 NIR=2486.3;
  ``/10000`` yields R=0.058 G=0.057 B=0.034 NIR=0.249, textbook vegetation
  surface reflectance. ``/3000`` would put a forest at 0.83 NIR, which is
  impossible. 10000 is also the standard Sentinel-2 L2A quantification value.
- **nodata**: ``65535``, declared on both LR and HR. Masked before division --
  65535/10000 = 6.55 reflectance would fail validation on any tile touching it.
- **Band order**: ``(B04, B03, B02, B08)`` = R, G, B, NIR.
- **Shapes**: LR ``(4, 130, 130)`` at 10 m, HR ``(4, 520, 520)`` at 2.5 m,
  identical geotransform origin, so pairs are co-registered.
- **Cropping**: none, anywhere in this module. The cache stores the full
  ``(4, 130, 130)`` / ``(4, 520, 520)`` uint16 tile and ``load_sample`` returns
  it whole. The deterministic validation crop lives in
  :func:`src.data.patches.centre_crop_pair` and is applied by the loader.
- **Record layout**: ``ds.read(i)`` returns a nested frame of exactly two rows
  with ``tortilla:id`` of ``"lr"`` and ``"hr"``; ``sub.read(j)`` returns a GDAL
  ``/vsisubfile/...`` string that rasterio opens.

Known upstream hazard: ``tacoreader.v1.load_files`` catches per-file errors,
prints them, and then fails with ``ValueError: No objects to concatenate`` three
frames from the real cause -- including on HTTP 429 from HuggingFace. We detect
the empty result and re-raise with the actual reason.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from src.data.base import SRPairDataset
from src.data.registry import register_dataset
from src.data.splits import parse_wkt_point
from src.utils.logging import get_logger
from src.utils.paths import resolve_cache_dir

__all__ = ["SEN2NAIPv2Dataset", "NATIVE_BANDS", "format_index_summary"]

# Channel order as stored in the GeoTIFFs. VERIFIED, see module docstring.
NATIVE_BANDS = ("B04", "B03", "B02", "B08")

# Subsets that actually resolve in the tacoreader registry. "sen2naipv2-temporal"
# is advertised on the dataset card but is NOT present -- do not add it back
# without checking load_tacofoundation_datasets() first.
KNOWN_SUBSETS = (
    "sen2naipv2-crosssensor",
    "sen2naipv2-unet",
    "sen2naipv2-histmatch",
)


@register_dataset("sen2naipv2")
class SEN2NAIPv2Dataset(SRPairDataset):
    """LR/HR Sentinel-2 reflectance pairs streamed from SEN2NAIPv2.

    Samples are cached as raw ``uint16`` NPZ next to a JSON sidecar, one file per
    pair. Caching is resumable: an existing, readable cache entry is never
    re-fetched, so an interrupted download resumes where it stopped rather than
    re-spending the Kaggle session's network time.

    Raw digital numbers are cached, not reflectance. The conversion is cheap and
    keeping the cache lossless means a corrected divisor or nodata value does not
    invalidate gigabytes of downloads.
    """

    SOURCE_NAME = "sen2naipv2"

    def __init__(self, cfg: Any, validate: bool = True) -> None:
        self.BANDS = tuple(cfg["dataset"]["bands"])
        super().__init__(cfg, validate=validate)

        ds_cfg = cfg["dataset"]
        sub_cfg = cfg["sen2naipv2"]

        self.subset = str(sub_cfg["subset"])
        if self.subset not in KNOWN_SUBSETS:
            raise ValueError(
                f"Unknown SEN2NAIPv2 subset {self.subset!r}. Verified subsets: "
                f"{list(KNOWN_SUBSETS)}. Note that 'sen2naipv2-temporal' appears "
                "on the dataset card but does not resolve in the tacoreader "
                "registry."
            )

        self.num_samples = int(sub_cfg["num_samples"])
        self.min_correlation = sub_cfg.get("min_correlation")
        self.split = sub_cfg.get("split")
        self.max_retries = int(sub_cfg.get("max_retries", 6) or 1)
        self.retry_backoff_s = float(sub_cfg.get("retry_backoff_s", 15) or 0)
        self.request_delay_s = float(sub_cfg.get("request_delay_s", 0.0) or 0.0)

        self.nodata_value = ds_cfg["nodata_value"]
        self.nodata_fill = float(ds_cfg["nodata_fill"])
        self.max_nodata_fraction = float(ds_cfg["max_nodata_fraction"])
        # Reporting-only, never a clip. See the config comments.
        self.reflectance_report_threshold = float(
            ds_cfg["reflectance_report_threshold"]
        )
        self.divisor_suspect_fraction = float(ds_cfg["divisor_suspect_fraction"])

        self.lr_size = int(cfg["sr"]["lr_patch_size"])
        self.hr_size = int(cfg["sr"]["hr_patch_size"])

        self.band_indices = self._resolve_band_indices(self.BANDS)

        self.cache_dir = resolve_cache_dir(cfg) / self.subset
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("data.sen2naipv2", log_file=cfg["paths"]["log_file"])

        self._catalog: Optional[List[Dict[str, Any]]] = None

    # -- catalog -----------------------------------------------------------

    @staticmethod
    def _resolve_band_indices(bands) -> List[int]:
        """Map requested band names to zero-based indices in the stored GeoTIFF.

        Raises:
            ValueError: A requested band is not one of :data:`NATIVE_BANDS`.
        """
        indices = []
        for band in bands:
            name = str(band)
            if name not in NATIVE_BANDS:
                raise ValueError(
                    f"Band {name!r} is not available in SEN2NAIPv2, which stores "
                    f"{list(NATIVE_BANDS)} (red, green, blue, NIR). Fix "
                    "cfg.dataset.bands."
                )
            indices.append(NATIVE_BANDS.index(name))
        return indices

    def _load_catalog(self) -> List[Dict[str, Any]]:
        """Fetch and filter the taco catalog. Network access happens here.

        Returns:
            One dict per selected pair, carrying the metadata needed to fetch and
            describe it, truncated to ``num_samples``.

        Raises:
            ImportError: tacoreader is not installed.
            RuntimeError: The catalog came back empty (see the upstream hazard in
                the module docstring), or every row was filtered out.
        """
        try:
            from tacoreader import v1 as taco_v1
        except ImportError as exc:
            raise ImportError(
                "tacoreader is required for the sen2naipv2 dataset. "
                "pip install tacoreader (and requests+aiohttp, which fsspec needs "
                "for HTTP range reads). Note we use tacoreader.v1 deliberately: "
                "the v2 API has no .read() and does not accept the "
                "'tacofoundation:' protocol."
            ) from exc

        self.logger.info("Loading taco catalog for %s", self.subset)
        frame = taco_v1.load(f"tacofoundation:{self.subset}")

        if frame is None or len(frame) == 0:
            raise RuntimeError(
                f"tacoreader returned an empty catalog for {self.subset!r}. "
                "tacoreader swallows per-file errors and reports this as an "
                "empty result, so the real cause is printed above this "
                "traceback -- most often HTTP 429 rate limiting from "
                "HuggingFace, or missing 'requests'/'aiohttp' for fsspec HTTP."
            )

        total = len(frame)
        if self.split is not None and "tortilla:data_split" in frame.columns:
            frame = frame[frame["tortilla:data_split"] == self.split]
            self.logger.info(
                "Split filter %r: %d -> %d rows", self.split, total, len(frame)
            )
        if self.min_correlation is not None and "correlation" in frame.columns:
            before = len(frame)
            frame = frame[frame["correlation"] >= float(self.min_correlation)]
            self.logger.info(
                "Correlation filter >=%s: %d -> %d rows",
                self.min_correlation,
                before,
                len(frame),
            )

        if len(frame) == 0:
            raise RuntimeError(
                f"Every row of {self.subset!r} was filtered out "
                f"(split={self.split!r}, min_correlation={self.min_correlation}). "
                "Relax the filters in cfg.sen2naipv2."
            )

        frame = frame.head(self.num_samples)
        if len(frame) < self.num_samples:
            self.logger.warning(
                "Requested %d samples but only %d remain after filtering.",
                self.num_samples,
                len(frame),
            )

        catalog = []
        for position in range(len(frame)):
            row = frame.iloc[position]
            catalog.append(
                {
                    "row_position": position,
                    "sample_id": str(row["tortilla:id"]),
                    "crs": str(row.get("stac:crs", "")) or None,
                    "geotransform": _as_tuple(row.get("stac:geotransform")),
                    "data_split": row.get("tortilla:data_split"),
                    "correlation": _as_float(row.get("correlation")),
                    # (lon, lat) in degrees, parsed from the WKT stac:centroid.
                    # Drives the geographic split -- see src/data/splits.py.
                    "centroid_lonlat": parse_wkt_point(row.get("stac:centroid")),
                }
            )
        self._frame = frame
        return catalog

    @property
    def catalog(self) -> List[Dict[str, Any]]:
        """The filtered catalog, loaded on first access."""
        if self._catalog is None:
            self._catalog = self._load_catalog()
        return self._catalog

    def __len__(self) -> int:
        return len(self.catalog)

    # -- caching -----------------------------------------------------------

    def _cache_paths(self, sample_id: str):
        stem = sample_id.replace("/", "_")
        return (
            self.cache_dir / f"{stem}.npz",
            self.cache_dir / f"{stem}.json",
        )

    def is_cached(self, idx: int) -> bool:
        """Whether sample ``idx`` is already on disk and readable."""
        npz_path, json_path = self._cache_paths(self.catalog[idx]["sample_id"])
        if not (npz_path.is_file() and json_path.is_file()):
            return False
        try:
            with np.load(npz_path) as handle:
                return "lr" in handle and "hr" in handle
        except (OSError, ValueError, EOFError):
            # A truncated file from an interrupted download. Treat as absent so
            # the next prepare() rewrites it; do not raise, and do not use it.
            self.logger.warning("Discarding unreadable cache entry %s", npz_path)
            return False

    def ensure_cached(self, idx: int) -> Path:
        """Fetch sample ``idx`` into the cache if not already there.

        Returns:
            Path to the cached ``.npz``.

        Raises:
            ImportError: rasterio is not installed.
            RuntimeError: The nested record did not contain both an ``lr`` and an
                ``hr`` row.
        """
        entry = self.catalog[idx]
        npz_path, json_path = self._cache_paths(entry["sample_id"])
        if self.is_cached(idx):
            return npz_path

        try:
            import rasterio as rio
        except ImportError as exc:
            raise ImportError(
                "rasterio is required to read SEN2NAIPv2 GeoTIFFs. "
                "pip install rasterio."
            ) from exc

        sub = self._frame.read(entry["row_position"])
        hrefs = {}
        for position in range(len(sub)):
            row = sub.iloc[position]
            hrefs[str(row["tortilla:id"]).lower()] = sub.read(position)

        missing = {"lr", "hr"} - set(hrefs)
        if missing:
            raise RuntimeError(
                f"Record {entry['sample_id']!r} is missing {sorted(missing)}; "
                f"nested ids were {sorted(hrefs)}. The record layout is not what "
                "this loader was written against."
            )

        arrays = {}
        profile = {}
        for key in ("lr", "hr"):
            with rio.open(hrefs[key]) as src:
                arrays[key] = src.read()  # (C, H, W), uint16
                profile[key] = {
                    "crs": str(src.crs) if src.crs else None,
                    "transform": tuple(src.transform)[:6],
                    "nodata": src.nodata,
                    "dtype": str(src.dtypes[0]),
                    "shape": [int(d) for d in arrays[key].shape],
                }

        # Write to a temporary name then rename, so an interrupted write can
        # never leave a half-file that looks cached.
        # Note: np.savez_compressed appends ".npz" when given a *path* that does
        # not already end in it, which would silently write somewhere other than
        # `tmp`. Passing an open handle sidesteps that entirely.
        tmp = npz_path.with_name(npz_path.name + ".partial")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, lr=arrays["lr"], hr=arrays["hr"])
        tmp.replace(npz_path)

        sidecar = dict(entry)
        sidecar["profile"] = profile
        json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        return npz_path

    def _fetch_with_retry(self, idx: int) -> bool:
        """Fetch one sample, retrying transient network failures with backoff.

        Retries are bounded and every attempt is logged at WARNING, so this is
        not silent error handling -- it is a bounded response to a specific,
        measured upstream behaviour. fsspec maps every non-OK HTTP status to
        ``FileNotFoundError``, so a 429 from HuggingFace is indistinguishable
        from a genuinely absent file at this layer; that is why the exception
        net is wide rather than narrow.

        Args:
            idx: Catalog index to fetch.

        Returns:
            True on success, False if every attempt failed. The caller records
            the failure; it is never silently dropped.
        """
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.ensure_cached(idx)
                return True
            except Exception as exc:  # noqa: BLE001 - logged, bounded, reported
                last_exc = exc
                if attempt == self.max_retries:
                    break
                delay = self.retry_backoff_s * (2 ** (attempt - 1))
                self.logger.warning(
                    "Fetch failed for %s (attempt %d/%d): %s: %s -- retrying in %.0fs",
                    self.catalog[idx]["sample_id"],
                    attempt,
                    self.max_retries,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                time.sleep(delay)

        self.logger.error(
            "Giving up on %s after %d attempts. Last error: %s: %s",
            self.catalog[idx]["sample_id"],
            self.max_retries,
            type(last_exc).__name__,
            last_exc,
        )
        return False

    def prepare(self, limit: Optional[int] = None) -> Dict[str, int]:
        """Populate the cache for every catalog entry. Resumable.

        Samples that fail every retry are counted and logged as errors, and the
        run continues rather than aborting. That is deliberate: at ~10 s/sample a
        full fetch is many hours, and discarding all of it because one record was
        briefly unreachable is worse than finishing and reporting the gap. The
        failures are never hidden -- they are logged at ERROR, returned in the
        counts, surfaced in the manifest's ``validation_error`` column, and make
        ``scripts/prepare_data.py`` exit non-zero.

        Args:
            limit: Stop after this many entries. None means all of them.

        Returns:
            Counts: ``{"cached", "fetched", "failed"}``.
        """
        total = len(self) if limit is None else min(limit, len(self))
        counts = {"cached": 0, "fetched": 0, "failed": 0}
        for idx in range(total):
            if self.is_cached(idx):
                counts["cached"] += 1
                continue
            if self._fetch_with_retry(idx):
                counts["fetched"] += 1
            else:
                counts["failed"] += 1
            done = counts["fetched"] + counts["cached"] + counts["failed"]
            if counts["fetched"] and counts["fetched"] % 50 == 0:
                self.logger.info(
                    "Fetched %d/%d samples (%d failed)", done, total, counts["failed"]
                )
            if self.request_delay_s:
                time.sleep(self.request_delay_s)

        log = self.logger.error if counts["failed"] else self.logger.info
        log(
            "Cache for %s: %d already present, %d fetched, %d FAILED, dir=%s",
            self.subset,
            counts["cached"],
            counts["fetched"],
            counts["failed"],
            self.cache_dir,
        )
        return counts

    # -- sample loading ----------------------------------------------------

    def load_sample(self, idx: int) -> Dict[str, Any]:
        """Load one cached pair as float32 reflectance.

        Fetches the sample first if it is not cached.

        Args:
            idx: Index in ``[0, len(self))``.

        Returns:
            Sample dict per :meth:`SRPairDataset.__getitem__`. ``lr`` is float32
            ``(C, 130, 130)`` and ``hr`` float32 ``(C, 520, 520)`` -- the FULL
            stored tile. Nothing is cropped here: a centre crop in the read path
            silently decides which pixels the project ever sees, and the tile
            periphery is exactly where nodata and bright targets live. The
            deterministic validation crop is
            :func:`~src.data.patches.centre_crop_pair`, applied by the loader;
            training draws random crops from the full tile. Both are surface
            reflectance
            (raw ``uint16`` digital numbers divided by 10000), nominally
            ``[0, 1]``, UNCLIPPED above 1.0. Channel order is
            ``cfg.dataset.bands``. Nodata (raw 65535) is masked to
            ``cfg.dataset.nodata_fill`` before division. ``meta`` carries
            ``sample_id``, ``source_dataset``, ``crs``, ``geotransform`` (GDAL
            6-tuple at HR scale), ``correlation``, and ``nodata_fraction`` (the
            fraction of FULL-tile pixels where any selected band was nodata).

        Raises:
            RuntimeError: The pair's nodata fraction exceeds
                ``cfg.dataset.max_nodata_fraction``, or the cached shapes do not
                satisfy the scale ratio.
        """
        entry = self.catalog[idx]
        npz_path = self.ensure_cached(idx)

        with np.load(npz_path) as handle:
            lr_raw = handle["lr"]
            hr_raw = handle["hr"]

        # Select configured bands out of the native RGBN order.
        lr_raw = lr_raw[self.band_indices]
        hr_raw = hr_raw[self.band_indices]

        lr_ref, lr_mask = self.to_reflectance(
            lr_raw, nodata_value=self.nodata_value, nodata_fill=self.nodata_fill
        )
        hr_ref, hr_mask = self.to_reflectance(
            hr_raw, nodata_value=self.nodata_value, nodata_fill=self.nodata_fill
        )

        # Fraction of PIXELS where any selected band is nodata, over the FULL
        # tile. Not the mean of the (C, H, W) mask: one dead band makes the
        # whole pixel unusable, and averaging over channels would report a
        # quarter of the true loss on a 4-band tile with one dead band.
        lr_nodata = np.any(lr_mask, axis=0)
        hr_nodata = np.any(hr_mask, axis=0)
        nodata_fraction = max(float(lr_nodata.mean()), float(hr_nodata.mean()))
        if nodata_fraction > self.max_nodata_fraction:
            raise RuntimeError(
                f"Sample {entry['sample_id']!r} is {nodata_fraction:.1%} nodata, "
                f"above cfg.dataset.max_nodata_fraction="
                f"{self.max_nodata_fraction:.1%}. Filtering it silently would "
                "bias the training set, so it is raised instead. Run "
                "scripts/prepare_data.py to see the whole rejection tally, then "
                "widen the threshold or tighten the catalog filters."
            )

        return {
            "lr": torch.from_numpy(np.ascontiguousarray(lr_ref)),
            "hr": torch.from_numpy(np.ascontiguousarray(hr_ref)),
            "meta": {
                "sample_id": entry["sample_id"],
                "source_dataset": self.SOURCE_NAME,
                "crs": entry["crs"],
                "geotransform": entry["geotransform"],
                "correlation": entry["correlation"],
                "data_split": entry["data_split"],
                "centroid_lonlat": entry["centroid_lonlat"],
                "nodata_fraction": nodata_fraction,
                "subset": self.subset,
            },
        }

    # -- index -------------------------------------------------------------
    #
    # The index reads the CACHE DIRECTLY rather than going through
    # load_sample(). That is deliberate on three counts:
    #
    #   1. It sees the FULL stored tile (130x130 LR, 520x520 HR). Any statistic
    #      taken after a centre crop describes the middle of the tile, and the
    #      things an index exists to catch -- nodata, tile-edge artefacts, the
    #      bright targets that expose a wrong reflectance divisor -- sit at the
    #      periphery the crop throws away.
    #   2. Nothing is zero-filled. Nodata pixels are EXCLUDED from every
    #      statistic rather than replaced by nodata_fill and then averaged in as
    #      if they were real observations of a perfectly black surface.
    #   3. It can describe a sample the contract would reject, instead of
    #      raising and leaving a blank row.

    # Number of distinct digital numbers a uint16 band can hold. The exact
    # percentile machinery below depends on the source being integer DN.
    DN_LEVELS = 65536

    def _nodata_pixel_mask(self, raw: np.ndarray) -> np.ndarray:
        """Pixels where ANY selected band is nodata.

        Args:
            raw: ``(C, H, W)`` uint16 digital numbers, bands already subset to
                ``cfg.dataset.bands``.

        Returns:
            ``(H, W)`` bool array, True where at least one band equals
            ``cfg.dataset.nodata_value``. A pixel with one dead band is not a
            usable observation, so the rule is ANY rather than ALL.
        """
        if self.nodata_value is None:
            return np.zeros(raw.shape[1:], dtype=bool)
        return np.any(raw == np.asarray(self.nodata_value, dtype=raw.dtype), axis=0)

    def _band_stats(
        self,
        raw: np.ndarray,
        valid: np.ndarray,
        prefix: str,
        histograms: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Per-band reflectance statistics over the full tile, nodata excluded.

        Args:
            raw: ``(C, H, W)`` uint16 digital numbers for the FULL tile, bands
                already subset to ``cfg.dataset.bands``.
            valid: ``(H, W)`` bool, True where the pixel is a real observation.
            prefix: ``"lr"`` or ``"hr"``, used to name the columns.
            histograms: Accumulator, keyed ``"{prefix}_{band}"``, of DN counts
                pooled across samples. Mutated in place; exact percentiles are
                read off it later.

        Returns:
            ``{prefix}_b{i}_{band}_min/_max/_mean/_std`` in surface reflectance
            (DN / ``cfg.dataset.reflectance_scale``), nominally ``[0, 1]`` and
            deliberately UNCLIPPED -- bright targets exceed 1.0. Empty strings
            when every pixel of the tile is nodata, which is reported rather
            than substituted with a zero.
        """
        out: Dict[str, Any] = {}
        scale = np.float32(self.reflectance_scale)

        for band_pos, band_index in enumerate(self.band_indices):
            band_name = NATIVE_BANDS[band_index]
            key = f"{prefix}_b{band_pos}_{band_name}"
            values = raw[band_pos][valid]

            if values.size == 0:
                for suffix in ("min", "max", "mean", "std"):
                    out[f"{key}_{suffix}"] = ""
                continue

            counts = np.bincount(values.astype(np.int64), minlength=self.DN_LEVELS)
            histograms.setdefault(
                f"{prefix}_{band_name}", np.zeros(self.DN_LEVELS, dtype=np.int64)
            )
            histograms[f"{prefix}_{band_name}"] += counts

            reflectance = values.astype(np.float32) / scale
            out[f"{key}_min"] = round(float(reflectance.min()), 6)
            out[f"{key}_max"] = round(float(reflectance.max()), 6)
            out[f"{key}_mean"] = round(float(reflectance.mean()), 6)
            out[f"{key}_std"] = round(float(reflectance.std()), 6)

        return out

    def index_row(self, idx: int, histograms: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Describe one cached sample, from the raw cache, without cropping.

        Never raises on a bad sample: the whole point is to SEE the bad ones.
        A sample that cannot be described is recorded with the reason in
        ``validation_error``; a sample that is describable but fails policy is
        recorded with ``rejected=True`` and ``rejection_reason``.

        Args:
            idx: Catalog index.
            histograms: DN-count accumulator, mutated in place.

        Returns:
            One manifest row. Reflectance columns are surface reflectance
            (uint16 DN / ``cfg.dataset.reflectance_scale``), nominally ``[0, 1]``
            and UNCLIPPED. ``nodata_fraction`` columns are fractions in
            ``[0, 1]`` of FULL-tile pixels where any selected band is nodata.
        """
        entry = self.catalog[idx]
        row: Dict[str, Any] = {
            "index": int(idx),
            "sample_id": str(entry["sample_id"]),
            "source_dataset": self.SOURCE_NAME,
            "subset": self.subset,
            "crs": entry["crs"] or "",
            "data_split": entry["data_split"] or "",
            "correlation": entry["correlation"] if entry["correlation"] is not None else "",
            "lr_shape": "",
            "hr_shape": "",
            "scale_ok": False,
            "nodata_fraction": "",
            "nodata_fraction_lr": "",
            "nodata_fraction_hr": "",
            "max_reflectance": "",
            "exceeds_report_threshold": False,
            "rejected": False,
            "rejection_reason": "",
            "validation_error": "",
        }

        npz_path, _ = self._cache_paths(entry["sample_id"])
        if not self.is_cached(idx):
            row["rejected"] = True
            row["rejection_reason"] = "not_cached"
            row["validation_error"] = (
                f"Sample is not in the cache at {npz_path}. It has not been "
                "downloaded yet, or its cache entry was unreadable and discarded."
            )
            return row

        try:
            with np.load(npz_path) as handle:
                lr_raw = handle["lr"][self.band_indices]
                hr_raw = handle["hr"][self.band_indices]
        except (OSError, ValueError, KeyError) as exc:
            # Recorded, not swallowed: a corrupt cache entry must be visible in
            # the manifest, and re-raising here would abandon the other 2000.
            row["rejected"] = True
            row["rejection_reason"] = "unreadable_cache"
            row["validation_error"] = f"{type(exc).__name__}: {exc}"
            self.logger.error("Cache entry %s is unreadable: %s", npz_path, exc)
            return row

        row["lr_shape"] = "x".join(str(int(d)) for d in lr_raw.shape)
        row["hr_shape"] = "x".join(str(int(d)) for d in hr_raw.shape)
        row["scale_ok"] = bool(
            hr_raw.shape[-2] == lr_raw.shape[-2] * self.scale
            and hr_raw.shape[-1] == lr_raw.shape[-1] * self.scale
        )

        lr_nodata = self._nodata_pixel_mask(lr_raw)
        hr_nodata = self._nodata_pixel_mask(hr_raw)
        lr_fraction = float(lr_nodata.mean())
        hr_fraction = float(hr_nodata.mean())
        # The pair is only as good as its worse half: a clean LR paired with a
        # 30%-dead HR is not a usable training target.
        fraction = max(lr_fraction, hr_fraction)

        row["nodata_fraction_lr"] = round(lr_fraction, 6)
        row["nodata_fraction_hr"] = round(hr_fraction, 6)
        row["nodata_fraction"] = round(fraction, 6)

        row.update(self._band_stats(lr_raw, ~lr_nodata, "lr", histograms))
        row.update(self._band_stats(hr_raw, ~hr_nodata, "hr", histograms))

        maxima = [
            row[key]
            for key in row
            if key.endswith("_max") and isinstance(row[key], float)
        ]
        if maxima:
            row["max_reflectance"] = round(max(maxima), 6)
            row["exceeds_report_threshold"] = bool(
                max(maxima) > self.reflectance_report_threshold
            )

        reasons = []
        if not row["scale_ok"]:
            reasons.append(
                f"shape_ratio (LR {row['lr_shape']}, HR {row['hr_shape']}, "
                f"scale={self.scale})"
            )
        if fraction > self.max_nodata_fraction:
            reasons.append(
                f"nodata_fraction {fraction:.4f} > "
                f"cfg.dataset.max_nodata_fraction={self.max_nodata_fraction:.4f}"
            )
        if isinstance(row["max_reflectance"], float) and (
            row["max_reflectance"] > self.reflectance_valid_max
        ):
            reasons.append(
                f"reflectance {row['max_reflectance']:.4f} > "
                f"cfg.dataset.reflectance_valid_max={self.reflectance_valid_max}"
            )
        if reasons:
            row["rejected"] = True
            row["rejection_reason"] = "; ".join(reasons)

        return row

    def build_index(self, path: Any, force: bool = False) -> Dict[str, Any]:
        """Index the cached samples into a manifest CSV. Resumable and incremental.

        Reads whatever is cached NOW, so it can be run against a partial
        download and again when the download finishes. On re-run, rows already
        in the manifest are kept and only samples missing from it are described,
        unless ``force`` rewrites everything.

        The incremental path is keyed on ``sample_id``. A kept row is NOT
        re-measured, so its columns are whatever schema wrote them -- after
        changing what the index measures, pass ``force`` (the summary block
        prints a warning when it reuses rows).

        Args:
            path: Destination CSV. Parent directories are created.
            force: Re-measure every cached sample, discarding existing rows.

        Returns:
            The summary from :meth:`summarise_index`, with ``manifest_path``
            added. Render it with :func:`format_index_summary`.

        Raises:
            RuntimeError: The catalog is empty, or nothing is cached yet.
        """
        import csv
        from pathlib import Path as _Path

        out = _Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if len(self) == 0:
            raise RuntimeError(
                f"{type(self).__name__} has an empty catalog, so no manifest was "
                "written. Check cfg.sen2naipv2 filters."
            )

        existing: Dict[str, Dict[str, Any]] = {}
        if out.is_file() and not force:
            with out.open("r", newline="", encoding="utf-8") as handle:
                for old in csv.DictReader(handle):
                    existing[str(old.get("sample_id", ""))] = dict(old)

        histograms: Dict[str, np.ndarray] = {}
        rows: List[Dict[str, Any]] = []
        reused = 0
        measured = 0
        uncached = 0

        for idx in range(len(self)):
            sample_id = str(self.catalog[idx]["sample_id"])

            if not self.is_cached(idx):
                uncached += 1
                continue

            prior = existing.get(sample_id)
            if prior is not None and str(prior.get("rejection_reason", "")) != "not_cached":
                prior["index"] = int(idx)
                rows.append(prior)
                reused += 1
                continue

            rows.append(self.index_row(idx, histograms))
            measured += 1

            if measured % 250 == 0:
                self.logger.info(
                    "Indexed %d cached samples (%d reused, %d not yet cached)",
                    measured,
                    reused,
                    uncached,
                )

        if not rows:
            raise RuntimeError(
                f"Nothing is cached yet under {self.cache_dir}, so there is "
                "nothing to index. Run scripts/prepare_data.py to download "
                "first -- caching is resumable."
            )

        fieldnames: List[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)

        self.logger.info(
            "Manifest written to %s: %d rows (%d measured, %d reused, %d of %d "
            "catalog samples not yet cached)",
            out,
            len(rows),
            measured,
            reused,
            uncached,
            len(self),
        )

        summary = self.summarise_index(
            rows,
            histograms,
            measured=measured,
            reused=reused,
            uncached=uncached,
        )
        summary["manifest_path"] = str(out)
        return summary

    @staticmethod
    def _percentile_from_histogram(counts: np.ndarray, fraction: float) -> int:
        """Exact percentile of a DN distribution held as a count-per-level array.

        Exact rather than approximate because the source is integer digital
        numbers: every distinct value the data can take has its own bin, so
        there is no binning error to trade off. Uses the inverse-CDF
        (nearest-rank) definition -- the smallest DN whose cumulative count
        reaches ``fraction`` of the total.

        Args:
            counts: Length-``DN_LEVELS`` non-negative counts, indexed by DN.
            fraction: In ``[0, 1]``.

        Returns:
            The digital number at that rank.

        Raises:
            ValueError: ``counts`` is empty.
        """
        total = int(counts.sum())
        if total == 0:
            raise ValueError("Cannot take a percentile of an empty histogram.")
        rank = max(1, int(np.ceil(fraction * total)))
        return int(np.searchsorted(np.cumsum(counts), rank, side="left"))

    def summarise_index(
        self,
        rows: Sequence[Mapping[str, Any]],
        histograms: Mapping[str, np.ndarray],
        measured: int = 0,
        reused: int = 0,
        uncached: int = 0,
    ) -> Dict[str, Any]:
        """Reduce the manifest rows to the numbers worth reading before training.

        Args:
            rows: The manifest rows just written.
            histograms: Pooled DN counts per ``"{prefix}_{band}"``, from
                :meth:`index_row`. Empty when every row was reused, in which
                case the percentile block is omitted rather than faked.
            measured: Rows measured this run.
            reused: Rows carried over from an existing manifest.
            uncached: Catalog samples not yet downloaded.

        Returns:
            ``total_indexed``, ``total_catalog``, ``measured``, ``reused``,
            ``uncached``, ``accepted``, ``rejected``, ``rejected_by_reason``,
            ``percentiles`` (per band, reflectance at p1/p50/p99),
            ``exceeding``, ``exceed_fraction``, ``report_threshold``, and
            ``divisor_suspect`` -- True when the exceed fraction is above
            ``cfg.dataset.divisor_suspect_fraction``.
        """
        from collections import Counter

        rejected = [r for r in rows if str(r.get("rejected", "")).lower() in ("true", "1")]
        reasons = Counter()
        for row in rejected:
            reason = str(row.get("rejection_reason", "")) or "unspecified"
            # Collapse the varying numbers so the tally is by CAUSE, not by
            # every distinct nodata fraction in the archive.
            reasons[reason.split(" ")[0].rstrip(";")] += 1

        exceeding = [
            str(r["sample_id"])
            for r in rows
            if str(r.get("exceeds_report_threshold", "")).lower() in ("true", "1")
        ]
        exceed_fraction = len(exceeding) / len(rows) if rows else 0.0

        percentiles: Dict[str, Dict[str, float]] = {}
        for key, counts in histograms.items():
            counts = np.asarray(counts)
            if counts.sum() == 0:
                continue
            percentiles[key] = {
                f"p{int(q * 100) if q != 0.01 else 1}": round(
                    self._percentile_from_histogram(counts, q)
                    / self.reflectance_scale,
                    6,
                )
                for q in (0.01, 0.50, 0.99)
            }

        return {
            "total_indexed": len(rows),
            "total_catalog": len(self),
            "measured": int(measured),
            "reused": int(reused),
            "uncached": int(uncached),
            "accepted": len(rows) - len(rejected),
            "rejected": len(rejected),
            "rejected_by_reason": dict(reasons.most_common()),
            "percentiles": percentiles,
            "report_threshold": float(self.reflectance_report_threshold),
            "exceeding": exceeding,
            "exceed_fraction": exceed_fraction,
            "divisor_suspect": exceed_fraction > self.divisor_suspect_fraction,
            "divisor_suspect_fraction": float(self.divisor_suspect_fraction),
            "reflectance_scale": float(self.reflectance_scale),
            "max_nodata_fraction": float(self.max_nodata_fraction),
        }


def format_index_summary(summary: Mapping[str, Any], max_listed: int = 25) -> str:
    """Render :meth:`SEN2NAIPv2Dataset.summarise_index` as a printable block.

    The divisor verdict is stated, not left as a number for the reader to
    interpret. ``cfg.dataset.reflectance_scale`` being wrong is not a subtle
    failure -- it rescales every reflectance in the project, and the spectral
    consistency contribution is built on those values being physical -- so the
    block says so in words when the evidence is there.

    Args:
        summary: From :meth:`SEN2NAIPv2Dataset.summarise_index`.
        max_listed: Sample ids to name before truncating the list.

    Returns:
        A multi-line block. Percentiles are surface reflectance, unclipped.
    """
    threshold = summary["report_threshold"]
    exceeding = summary["exceeding"]
    fraction = summary["exceed_fraction"]

    lines = [
        "=" * 74,
        "  INDEX SUMMARY",
        "=" * 74,
        f"  manifest            {summary.get('manifest_path', '(not written)')}",
        f"  indexed             {summary['total_indexed']} of "
        f"{summary['total_catalog']} catalog samples",
        f"    measured now      {summary['measured']}",
        f"    reused from disk  {summary['reused']}",
        f"    not yet cached    {summary['uncached']}",
        "",
        f"  accepted            {summary['accepted']}",
        f"  rejected            {summary['rejected']}",
    ]

    if summary["rejected_by_reason"]:
        for reason, count in summary["rejected_by_reason"].items():
            share = count / summary["total_indexed"] if summary["total_indexed"] else 0.0
            lines.append(f"    {reason:<18} {count:>6}  ({share:.1%})")
    else:
        lines.append("    (nothing rejected)")

    if summary["reused"]:
        lines += [
            "",
            f"  NOTE: {summary['reused']} row(s) were carried over from the "
            "existing manifest and were",
            "  NOT re-measured. Their columns are whatever schema wrote them. "
            "Re-run with --force",
            "  after changing what the index measures, or the two halves of "
            "this manifest are not",
            "  comparable.",
        ]

    lines += ["", "  per-band reflectance percentiles, pooled over every indexed pixel"]
    if summary["percentiles"]:
        lines.append(
            f"  (nodata excluded, unclipped, DN / {summary['reflectance_scale']:g})"
        )
        lines.append(f"    {'band':<10} {'p1':>10} {'p50':>10} {'p99':>10}")
        for key in sorted(summary["percentiles"]):
            stats = summary["percentiles"][key]
            lines.append(
                f"    {key:<10} {stats['p1']:>10.4f} {stats['p50']:>10.4f} "
                f"{stats['p99']:>10.4f}"
            )
    else:
        lines.append(
            "    (no pixels measured this run -- every row was reused; "
            "re-run with --force)"
        )

    lines += [
        "",
        f"  samples exceeding {threshold:g} reflectance: {len(exceeding)} of "
        f"{summary['total_indexed']} ({fraction:.2%})",
    ]
    for sample_id in exceeding[:max_listed]:
        lines.append(f"    {sample_id}")
    if len(exceeding) > max_listed:
        lines.append(f"    ... and {len(exceeding) - max_listed} more (see the manifest)")

    lines.append("")
    if summary["divisor_suspect"]:
        lines += [
            "  *** THE REFLECTANCE DIVISOR IS SUSPECT ***",
            f"  {fraction:.2%} of samples exceed {threshold:g} reflectance, above the "
            f"{summary['divisor_suspect_fraction']:.0%} threshold at which a",
            "  bright tail stops being credible. Cloud, snow and specular roofs are a "
            "few percent of",
            "  a scene archive, not a majority of it. A divisor that is too small "
            "inflates EVERY tile",
            f"  at once, which is what this looks like. Check "
            f"cfg.dataset.reflectance_scale (currently",
            f"  {summary['reflectance_scale']:g}) and cfg.dataset.bands against the "
            "source metadata BEFORE",
            "  training: the spectral-consistency objective is built on these values "
            "being physical.",
            "  DO NOT respond by clipping.",
        ]
    else:
        lines += [
            f"  Divisor looks sound: {fraction:.2%} of samples exceed {threshold:g}, "
            f"below the {summary['divisor_suspect_fraction']:.0%} threshold.",
            "  A small bright tail is expected and physical (cloud, snow, specular "
            "water, bright roofs)",
            "  and is deliberately NOT clipped.",
        ]

    lines.append("=" * 74)
    return "\n".join(lines)


def _as_tuple(value):
    if value is None:
        return None
    try:
        return tuple(float(v) for v in np.asarray(value).ravel())
    except (TypeError, ValueError):
        return None


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
