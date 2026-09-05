"""WorldStrat fallback dataset, read from a Kaggle-attached copy.

.. warning::

   **THE DIRECTORY LAYOUT BELOW IS UNVERIFIED.** The Kaggle mount was not
   available when this module was written, so the globs are a documented
   assumption driven entirely by ``cfg.worldstrat``, not a measured fact. Nothing
   here guesses silently: discovery failures raise with a listing of what was
   actually found under the data root so the globs can be corrected in config.

   To verify, run::

       python scripts/prepare_data.py --config configs/base.yaml --dataset worldstrat

   and read the error. Then fix ``cfg.worldstrat.hr_glob`` / ``lr_glob`` /
   ``lr_dir_name``. No code change should be needed.

Known limitations, all deliberate and all consequential
-------------------------------------------------------
1. **Multi-frame LR collapsed to one revisit.** WorldStrat pairs each HR scene
   with multiple Sentinel-2 revisits. This loader takes exactly one
   (``cfg.worldstrat.revisit_strategy``, default ``"first"``) and discards the
   rest. That throws away the multi-temporal signal WorldStrat exists to
   provide, and makes results here not directly comparable to published
   multi-frame WorldStrat numbers. It is the right trade for a fallback that
   must drop into a single-frame x4 pipeline unchanged.
2. **Native scale is not 4.** WorldStrat HR is SPOT 6/7 at ~1.5 m against
   Sentinel-2 at 10 m -- a ratio near 6.7, not 4. To satisfy the x4 contract the
   HR patch is area-resampled to exactly ``scale x`` the LR patch. That is a real
   resampling step, not a crop, and it slightly softens the HR target.
3. **Reflectance is NOT cross-sensor harmonised.** Unlike SEN2NAIPv2, whose NAIP
   HR was harmonised into Sentinel-2 reflectance (measured: LR and HR band means
   agree to within 0.1%), WorldStrat SPOT HR and Sentinel-2 LR are different
   sensors with different calibration. **The spectral-consistency objective is
   therefore weaker on this dataset**, and any spectral-fidelity number measured
   here must be reported with that caveat. This is the single biggest reason to
   keep SEN2NAIPv2 primary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.data.base import SRPairDataset
from src.data.registry import register_dataset
from src.utils.logging import get_logger
from src.utils.paths import resolve_data_root

__all__ = ["WorldStratDataset"]

# UNVERIFIED. WorldStrat HR is SPOT 6/7 pan-sharpened RGB(N); the band order
# below is the assumption. If the mounted copy differs, correct this constant
# and say so -- do not silently reorder cfg.dataset.bands to compensate.
ASSUMED_NATIVE_BANDS = ("B04", "B03", "B02", "B08")


@register_dataset("worldstrat")
class WorldStratDataset(SRPairDataset):
    """Single-frame LR/HR pairs from a Kaggle-mounted WorldStrat copy."""

    SOURCE_NAME = "worldstrat"

    def __init__(self, cfg: Any, validate: bool = True) -> None:
        self.BANDS = tuple(cfg["dataset"]["bands"])
        super().__init__(cfg, validate=validate)

        ds_cfg = cfg["dataset"]
        ws_cfg = cfg["worldstrat"]

        self.root_subdir = str(ws_cfg["root_subdir"])
        self.hr_glob = str(ws_cfg["hr_glob"])
        self.lr_dir_name = str(ws_cfg["lr_dir_name"])
        self.lr_glob = str(ws_cfg["lr_glob"])
        self.num_samples = int(ws_cfg["num_samples"])
        self.revisit_strategy = str(ws_cfg["revisit_strategy"])

        if self.revisit_strategy != "first":
            raise NotImplementedError(
                f"revisit_strategy={self.revisit_strategy!r} is not implemented. "
                "Only 'first' is supported; see the multi-frame limitation in "
                "this module's docstring."
            )

        self.nodata_value = ds_cfg["nodata_value"]
        self.nodata_fill = float(ds_cfg["nodata_fill"])
        self.lr_size = int(cfg["sr"]["lr_patch_size"])
        self.hr_size = int(cfg["sr"]["hr_patch_size"])

        self.band_indices = [
            ASSUMED_NATIVE_BANDS.index(str(b))
            for b in self.BANDS
            if str(b) in ASSUMED_NATIVE_BANDS
        ]
        if len(self.band_indices) != len(self.BANDS):
            unknown = [b for b in self.BANDS if str(b) not in ASSUMED_NATIVE_BANDS]
            raise ValueError(
                f"Bands {unknown} are not in the ASSUMED WorldStrat band order "
                f"{list(ASSUMED_NATIVE_BANDS)}. This ordering is unverified -- "
                "confirm it against the mounted data before trusting it."
            )

        self.data_root = resolve_data_root(cfg)
        self.root = self.data_root / self.root_subdir
        self.logger = get_logger("data.worldstrat", log_file=cfg["paths"]["log_file"])

        self._pairs: Optional[List[Dict[str, Any]]] = None

    # -- discovery ---------------------------------------------------------

    def _discover(self) -> List[Dict[str, Any]]:
        """Find HR scenes and their first LR revisit.

        Returns:
            One dict per pair with ``sample_id``, ``hr_path``, ``lr_path``.

        Raises:
            FileNotFoundError: The root, the HR glob, or the LR revisits matched
                nothing. The message lists what *was* found, so the config globs
                can be fixed without another round trip.
        """
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"WorldStrat root {self.root} does not exist.\n"
                f"data_root resolved to {self.data_root}, and "
                f"cfg.worldstrat.root_subdir={self.root_subdir!r}.\n"
                f"Present under data_root: {_listdir(self.data_root)}\n"
                "THIS LAYOUT IS UNVERIFIED -- fix cfg.worldstrat.root_subdir."
            )

        hr_paths = sorted(self.root.glob(self.hr_glob))
        if not hr_paths:
            raise FileNotFoundError(
                f"cfg.worldstrat.hr_glob={self.hr_glob!r} matched no files under "
                f"{self.root}.\n"
                f"Present at that level: {_listdir(self.root)}\n"
                "THIS LAYOUT IS UNVERIFIED -- correct the glob in config."
            )

        pairs: List[Dict[str, Any]] = []
        skipped: List[str] = []
        for hr_path in hr_paths:
            lr_dir = hr_path.parent / self.lr_dir_name
            revisits = sorted(lr_dir.glob(self.lr_glob)) if lr_dir.is_dir() else []
            if not revisits:
                skipped.append(hr_path.parent.name)
                continue
            # Limitation 1: one revisit only.
            pairs.append(
                {
                    "sample_id": hr_path.stem,
                    "hr_path": hr_path,
                    "lr_path": revisits[0],
                    "n_revisits_available": len(revisits),
                }
            )
            if len(pairs) >= self.num_samples:
                break

        if not pairs:
            raise FileNotFoundError(
                f"Found {len(hr_paths)} HR scenes under {self.root} but none had "
                f"an LR directory {self.lr_dir_name!r} containing "
                f"{self.lr_glob!r}.\n"
                f"Example HR scene: {hr_paths[0]}\n"
                f"Its siblings: {_listdir(hr_paths[0].parent)}\n"
                "THIS LAYOUT IS UNVERIFIED -- correct cfg.worldstrat.lr_dir_name "
                "and lr_glob."
            )

        if skipped:
            self.logger.warning(
                "%d HR scenes had no matching LR revisits (e.g. %s); they are "
                "excluded from the index.",
                len(skipped),
                ", ".join(skipped[:5]),
            )
        self.logger.info(
            "WorldStrat: %d pairs discovered under %s (UNVERIFIED layout)",
            len(pairs),
            self.root,
        )
        return pairs

    @property
    def pairs(self) -> List[Dict[str, Any]]:
        if self._pairs is None:
            self._pairs = self._discover()
        return self._pairs

    def __len__(self) -> int:
        return len(self.pairs)

    # -- loading -----------------------------------------------------------

    def load_sample(self, idx: int) -> Dict[str, Any]:
        """Load one WorldStrat pair as float32 reflectance.

        Args:
            idx: Index in ``[0, len(self))``.

        Returns:
            Sample dict per :meth:`SRPairDataset.__getitem__`. ``lr`` is float32
            ``(C, lr_patch_size, lr_patch_size)`` centre-cropped from the
            Sentinel-2 revisit; ``hr`` is float32
            ``(C, hr_patch_size, hr_patch_size)`` area-resampled from the SPOT
            scene (see limitation 2). Both are surface reflectance divided by
            ``cfg.dataset.reflectance_scale``, nominally ``[0, 1]`` and
            UNCLIPPED. ``meta`` carries ``sample_id``, ``source_dataset``,
            ``crs``, ``geotransform``, ``n_revisits_available`` and
            ``revisit_used``.

        Raises:
            ImportError: rasterio is not installed.
            RuntimeError: The LR scene is smaller than ``lr_patch_size``.
        """
        try:
            import rasterio as rio
        except ImportError as exc:
            raise ImportError(
                "rasterio is required to read WorldStrat GeoTIFFs."
            ) from exc

        entry = self.pairs[idx]

        with rio.open(entry["lr_path"]) as src:
            lr_raw = src.read()
            lr_crs = str(src.crs) if src.crs else None
            lr_transform = tuple(src.transform)[:6]
        with rio.open(entry["hr_path"]) as src:
            hr_raw = src.read()
            hr_crs = str(src.crs) if src.crs else None
            hr_transform = tuple(src.transform)[:6]

        lr_raw = _select_bands(lr_raw, self.band_indices, entry["lr_path"])
        hr_raw = _select_bands(hr_raw, self.band_indices, entry["hr_path"])

        lr_ref, _ = self.to_reflectance(
            lr_raw, nodata_value=self.nodata_value, nodata_fill=self.nodata_fill
        )
        hr_ref, _ = self.to_reflectance(
            hr_raw, nodata_value=self.nodata_value, nodata_fill=self.nodata_fill
        )

        _, lr_h, lr_w = lr_ref.shape
        if lr_h < self.lr_size or lr_w < self.lr_size:
            raise RuntimeError(
                f"{entry['lr_path']} is {lr_h}x{lr_w}, smaller than "
                f"cfg.sr.lr_patch_size={self.lr_size}."
            )

        top = (lr_h - self.lr_size) // 2
        left = (lr_w - self.lr_size) // 2
        lr_crop = lr_ref[:, top : top + self.lr_size, left : left + self.lr_size]

        # Limitation 2: HR is resampled, not cropped, because the native ratio
        # is ~6.7 rather than 4. The same centre region is taken by fraction.
        frac_top = top / lr_h
        frac_left = left / lr_w
        frac_size_h = self.lr_size / lr_h
        frac_size_w = self.lr_size / lr_w
        _, hr_h, hr_w = hr_ref.shape
        h0 = int(round(frac_top * hr_h))
        w0 = int(round(frac_left * hr_w))
        h1 = max(h0 + 1, int(round((frac_top + frac_size_h) * hr_h)))
        w1 = max(w0 + 1, int(round((frac_left + frac_size_w) * hr_w)))
        hr_region = hr_ref[:, h0:h1, w0:w1]

        hr_crop = (
            F.interpolate(
                torch.from_numpy(np.ascontiguousarray(hr_region))[None],
                size=(self.hr_size, self.hr_size),
                mode="area",
            )[0]
            .contiguous()
            .numpy()
        )

        return {
            "lr": torch.from_numpy(np.ascontiguousarray(lr_crop)),
            "hr": torch.from_numpy(np.ascontiguousarray(hr_crop)),
            "meta": {
                "sample_id": entry["sample_id"],
                "source_dataset": self.SOURCE_NAME,
                "crs": hr_crs or lr_crs,
                "geotransform": hr_transform or lr_transform,
                "lr_crs": lr_crs,
                "n_revisits_available": entry["n_revisits_available"],
                "revisit_used": entry["lr_path"].name,
                "hr_resampled": True,
                "spectrally_harmonised": False,
            },
        }


def _select_bands(array: np.ndarray, indices: List[int], path: Path) -> np.ndarray:
    if array.ndim != 3:
        raise RuntimeError(f"{path}: expected (C, H, W), got shape {array.shape}.")
    if max(indices) >= array.shape[0]:
        raise RuntimeError(
            f"{path} has {array.shape[0]} bands but band index {max(indices)} was "
            "requested. The ASSUMED WorldStrat band order in worldstrat.py is "
            "wrong for this file -- correct it rather than reordering "
            "cfg.dataset.bands."
        )
    return array[indices]


def _listdir(path: Path, limit: int = 25) -> str:
    try:
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
    except OSError as exc:
        return f"<could not list: {exc}>"
    if len(names) > limit:
        return ", ".join(names[:limit]) + f", ... ({len(names)} total)"
    return ", ".join(names) or "<empty>"
