"""The SR pair dataset interface.

Everything downstream -- training, evaluation, export -- talks to
:class:`SRPairDataset` and never to a concrete dataset class. Swapping the
primary dataset for a fallback is therefore a change to ``cfg.dataset.name``,
not a rewrite.

Design note: ``__getitem__`` is implemented here as a template method that calls
the abstract :meth:`SRPairDataset.load_sample` and then validates the result.
Subclasses implement ``load_sample``, not ``__getitem__``. This is deliberate --
if validation were left to each subclass, the contract would hold only as well
as the least careful implementation, and a dataset swapped in under time
pressure at hour three is exactly the one that would skip it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

__all__ = ["SRPairDataset", "SampleValidationError"]

# Keys every sample dict must carry.
REQUIRED_KEYS: Tuple[str, ...] = ("lr", "hr", "meta")
# Keys every sample's meta dict must carry. ``geotransform`` is intentionally
# absent: not every fallback dataset georeferences its patches, so it is
# optional-but-preferred and must be present as a key with value None when
# genuinely unavailable.
REQUIRED_META_KEYS: Tuple[str, ...] = ("sample_id", "source_dataset", "crs")


class SampleValidationError(ValueError):
    """A sample violated the dataset contract.

    Raised, never logged-and-skipped. A sample that fails validation means the
    loader is wrong about the data, and training on the remainder would produce
    a model whose reflectance calibration is quietly incorrect.
    """


class SRPairDataset(ABC):
    """Abstract base for a low-res / high-res reflectance pair dataset.

    Subclasses must set :attr:`BANDS` and :attr:`SOURCE_NAME`, and implement
    :meth:`__len__` and :meth:`load_sample`.

    Attributes:
        BANDS: Band names in channel order, e.g. ``("B04", "B03", "B02", "B08")``.
            This is the channel axis contract for every tensor the dataset emits.
        SOURCE_NAME: Registry key, mirrored into ``meta["source_dataset"]`` so a
            manifest or a checkpoint can be traced back to its data source.
    """

    BANDS: Sequence[str] = ()
    SOURCE_NAME: str = ""

    def __init__(self, cfg: Any, validate: bool = True) -> None:
        """
        Args:
            cfg: The loaded config. Retained as ``self.cfg``; subclasses read
                their own block from it.
            validate: Whether :meth:`__getitem__` validates each sample. Leave
                True unless profiling shows the check is a training bottleneck.
        """
        self.cfg = cfg
        self.validate = validate

        self.scale = int(cfg["sr"]["scale"])
        dataset_cfg = cfg["dataset"]
        self.reflectance_scale = float(dataset_cfg["reflectance_scale"])
        self.reflectance_valid_max = float(dataset_cfg["reflectance_valid_max"])
        self.reflectance_valid_min = float(dataset_cfg["reflectance_nominal_min"])

        if not self.BANDS:
            raise NotImplementedError(
                f"{type(self).__name__} must define a non-empty BANDS attribute "
                "giving band names in channel order."
            )
        if not self.SOURCE_NAME:
            raise NotImplementedError(
                f"{type(self).__name__} must define SOURCE_NAME."
            )

    # -- interface ---------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int:
        """Number of available LR/HR pairs."""

    @abstractmethod
    def load_sample(self, idx: int) -> Dict[str, Any]:
        """Load one pair. Implemented by subclasses; called by :meth:`__getitem__`.

        Returns:
            The sample dict described in :meth:`__getitem__`. Implementations do
            not need to validate -- ``__getitem__`` does it for them.
        """

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one LR/HR reflectance pair.

        Args:
            idx: Index in ``[0, len(self))``.

        Returns:
            A dict with:

            - ``lr``: ``torch.float32`` tensor, shape ``(C, H, W)``, channel order
              :attr:`BANDS`. Surface reflectance, nominally ``[0, 1]``; values
              may mildly exceed 1.0 over specular targets and are NOT clipped.
              Validation caps them at ``cfg.dataset.reflectance_valid_max``.
            - ``hr``: ``torch.float32`` tensor, shape ``(C, H*scale, W*scale)``,
              same channel order, same reflectance units and range as ``lr``.
            - ``meta``: dict with ``sample_id`` (str), ``source_dataset`` (str),
              ``crs`` (str or None), ``geotransform`` (6-tuple of float, GDAL
              order ``(x0, xres, xskew, y0, yskew, yres)``, or None when the
              source does not georeference), plus any dataset-specific extras.

        Raises:
            SampleValidationError: The loaded sample violates the contract and
                ``self.validate`` is True.
        """
        sample = self.load_sample(idx)
        if self.validate:
            self.validate_sample(sample, idx=idx)
        return sample

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    # -- contract enforcement ---------------------------------------------

    def validate_sample(self, sample: Mapping[str, Any], idx: Any = None) -> None:
        """Assert that a sample satisfies the dataset contract.

        Checks, in order:

        1. Required keys present, ``meta`` carries its required keys.
        2. ``lr`` and ``hr`` are float32 tensors of rank 3, ``(C, H, W)``.
        3. Channel count equals ``len(BANDS)`` on both.
        4. ``hr`` spatial dims are exactly ``scale`` times ``lr``'s.
        5. No NaN and no inf in either tensor.
        6. Reflectance within ``[reflectance_nominal_min,
           reflectance_valid_max]`` (default ``[0, 1.2]``).

        Args:
            sample: The sample dict to check.
            idx: Optional index, included in error messages.

        Raises:
            SampleValidationError: On any violation. Never returns False and
                never warns -- a bad sample is a bug in the loader.
        """
        where = f"{type(self).__name__}[{idx}]" if idx is not None else type(self).__name__

        for key in REQUIRED_KEYS:
            if key not in sample:
                raise SampleValidationError(
                    f"{where}: sample is missing required key {key!r}. "
                    f"Got keys: {sorted(sample)}."
                )

        meta = sample["meta"]
        if not isinstance(meta, Mapping):
            raise SampleValidationError(
                f"{where}: meta must be a mapping, got {type(meta).__name__}."
            )
        for key in REQUIRED_META_KEYS:
            if key not in meta:
                raise SampleValidationError(
                    f"{where}: meta is missing required key {key!r}. "
                    f"Got keys: {sorted(meta)}."
                )

        lr, hr = sample["lr"], sample["hr"]
        for name, tensor in (("lr", lr), ("hr", hr)):
            self._check_tensor(where, name, tensor)

        n_bands = len(self.BANDS)
        if lr.shape[0] != n_bands or hr.shape[0] != n_bands:
            raise SampleValidationError(
                f"{where}: channel count must equal len(BANDS)={n_bands} "
                f"(BANDS={tuple(self.BANDS)}), got lr C={lr.shape[0]} "
                f"hr C={hr.shape[0]}."
            )

        _, lr_h, lr_w = lr.shape
        _, hr_h, hr_w = hr.shape
        if hr_h != lr_h * self.scale or hr_w != lr_w * self.scale:
            raise SampleValidationError(
                f"{where}: shape ratio must equal scale={self.scale}. "
                f"lr is {lr_h}x{lr_w}, hr is {hr_h}x{hr_w}, expected hr to be "
                f"{lr_h * self.scale}x{lr_w * self.scale}."
            )

    def _check_tensor(self, where: str, name: str, tensor: Any) -> None:
        if not torch.is_tensor(tensor):
            raise SampleValidationError(
                f"{where}: {name!r} must be a torch.Tensor, got "
                f"{type(tensor).__name__}."
            )
        if tensor.dtype != torch.float32:
            raise SampleValidationError(
                f"{where}: {name!r} must be float32, got {tensor.dtype}. "
                "Reflectance is converted to float32 at load time; an integer "
                "dtype here means the reflectance divisor was never applied."
            )
        if tensor.ndim != 3:
            raise SampleValidationError(
                f"{where}: {name!r} must be rank 3 (C, H, W), got shape "
                f"{tuple(tensor.shape)}. Note the channel axis is FIRST."
            )

        if not torch.isfinite(tensor).all():
            n_nan = int(torch.isnan(tensor).sum())
            n_inf = int(torch.isinf(tensor).sum())
            raise SampleValidationError(
                f"{where}: {name!r} contains {n_nan} NaN and {n_inf} inf values. "
                "This usually means nodata was not masked before scaling."
            )

        lo = float(tensor.min())
        hi = float(tensor.max())
        if lo < self.reflectance_valid_min or hi > self.reflectance_valid_max:
            raise SampleValidationError(
                f"{where}: {name!r} reflectance out of range "
                f"[{self.reflectance_valid_min}, {self.reflectance_valid_max}]: "
                f"min={lo:.4f} max={hi:.4f}. "
                "Do NOT fix this by clipping -- clipping destroys the radiometry "
                "the spectral-consistency loss depends on. Check the reflectance "
                "divisor (cfg.dataset.reflectance_scale) and the nodata mask "
                "(cfg.dataset.nodata_value) instead."
            )

    # -- manifest ----------------------------------------------------------

    def manifest_rows(self) -> "list[Dict[str, Any]]":
        """Walk every sample and describe it, recording failures rather than raising.

        This is the one place in the data layer that catches
        :class:`SampleValidationError`, and it does so to *report* -- the failure
        is written into the ``validation_error`` column, never discarded. Use it
        to inspect a dataset before committing GPU hours to it.

        Returns:
            One row per sample, with ``sample_id``, ``source_dataset``, ``crs``,
            ``lr_shape``/``hr_shape`` as ``"CxHxW"`` strings, ``scale_ok``,
            per-band ``lr_b{i}_{band}_min``/``_max`` and the ``hr_`` equivalents
            (reflectance units), ``n_nan``, and ``validation_error`` (empty
            string when the sample passed).
        """
        rows: list[Dict[str, Any]] = []
        for idx in range(len(self)):
            row: Dict[str, Any] = {
                "index": idx,
                "sample_id": "",
                "source_dataset": self.SOURCE_NAME,
                "crs": "",
                "lr_shape": "",
                "hr_shape": "",
                "scale_ok": False,
                "n_nan": "",
                "validation_error": "",
            }
            try:
                sample = self.load_sample(idx)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                row["validation_error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                continue

            meta = sample.get("meta", {})
            row["sample_id"] = meta.get("sample_id", "")
            row["crs"] = meta.get("crs") or ""
            row["source_dataset"] = meta.get("source_dataset", self.SOURCE_NAME)

            for prefix in ("lr", "hr"):
                tensor = sample.get(prefix)
                if not torch.is_tensor(tensor):
                    continue
                row[f"{prefix}_shape"] = "x".join(str(d) for d in tensor.shape)
                for band_idx in range(tensor.shape[0]):
                    band_name = (
                        self.BANDS[band_idx]
                        if band_idx < len(self.BANDS)
                        else f"c{band_idx}"
                    )
                    band = tensor[band_idx]
                    finite = band[torch.isfinite(band)]
                    key = f"{prefix}_b{band_idx}_{band_name}"
                    row[f"{key}_min"] = round(float(finite.min()), 6) if finite.numel() else ""
                    row[f"{key}_max"] = round(float(finite.max()), 6) if finite.numel() else ""

            nan_total = 0
            for prefix in ("lr", "hr"):
                tensor = sample.get(prefix)
                if torch.is_tensor(tensor):
                    nan_total += int(torch.isnan(tensor).sum())
            row["n_nan"] = nan_total

            lr, hr = sample.get("lr"), sample.get("hr")
            if torch.is_tensor(lr) and torch.is_tensor(hr):
                row["scale_ok"] = bool(
                    hr.shape[-2] == lr.shape[-2] * self.scale
                    and hr.shape[-1] == lr.shape[-1] * self.scale
                )

            try:
                self.validate_sample(sample, idx=idx)
            except SampleValidationError as exc:
                row["validation_error"] = str(exc)

            rows.append(row)
        return rows

    def build_index(self, path: Any) -> Any:
        """Write the manifest from :meth:`manifest_rows` to a CSV.

        Args:
            path: Destination CSV path. Parent directories are created.

        Returns:
            The :class:`~pathlib.Path` written.

        Raises:
            RuntimeError: The dataset is empty, so there is nothing to inspect.
        """
        import csv
        from pathlib import Path as _Path

        rows = self.manifest_rows()
        if not rows:
            raise RuntimeError(
                f"{type(self).__name__} produced no samples, so no manifest was "
                "written. Check the dataset's num_samples and any split or "
                "quality filters in its config block."
            )

        # Union of keys, preserving first-seen order, so a row that failed early
        # (and therefore has no per-band columns) does not truncate the header.
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        out = _Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        return out

    # -- shared helpers ----------------------------------------------------

    def to_reflectance(
        self,
        array: np.ndarray,
        nodata_value: Any = None,
        nodata_fill: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert raw integer digital numbers to float32 surface reflectance.

        Args:
            array: Raw band data as read from the source, shape ``(C, H, W)``,
                integer dtype (typically ``uint16``). Digital numbers, i.e.
                reflectance multiplied by ``cfg.dataset.reflectance_scale``.
            nodata_value: Sentinel digital number marking "no observation", in
                RAW units (e.g. 65535). Masked BEFORE division -- 65535/10000
                is 6.55 reflectance and would fail validation. None disables
                masking.
            nodata_fill: Reflectance value written at masked pixels.

        Returns:
            ``(reflectance, nodata_mask)`` where ``reflectance`` is float32
            ``(C, H, W)`` surface reflectance, nominally ``[0, 1]`` and
            deliberately UNCLIPPED above 1.0, and ``nodata_mask`` is a bool array
            of the same shape, True where the pixel was nodata.
        """
        if array.ndim != 3:
            raise ValueError(
                f"Expected (C, H, W) raw array, got shape {array.shape}. "
                "Channel axis must be first."
            )

        if nodata_value is None:
            mask = np.zeros(array.shape, dtype=bool)
        else:
            mask = array == np.asarray(nodata_value, dtype=array.dtype)

        reflectance = array.astype(np.float32) / np.float32(self.reflectance_scale)
        if mask.any():
            reflectance[mask] = np.float32(nodata_fill)

        return reflectance, mask
