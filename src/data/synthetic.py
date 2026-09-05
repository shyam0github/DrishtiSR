"""A synthetic in-memory SR pair dataset. Never touches the network or disk.

Two jobs:

1. It backs ``--smoke``, so every entry-point script can be exercised end to end
   on a CPU with no data downloaded and no optional dependencies installed.
2. It is the reference implementation the contract tests in
   ``tests/test_dataset_contract.py`` run against, alongside the real datasets.

Because it is deterministic and cheap, a smoke run that fails here is a bug in
our code, never a bad tile or a flaky network.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from src.data.base import SRPairDataset
from src.data.registry import register_dataset

__all__ = ["SyntheticStubDataset"]


@register_dataset("synthetic_stub")
class SyntheticStubDataset(SRPairDataset):
    """Fabricated LR/HR reflectance pairs with plausible radiometry.

    HR is built first as smooth low-frequency structure plus texture; LR is a
    strict block-mean downsample of it by ``cfg.sr.scale``. Deriving LR from HR
    rather than generating the two independently means the pair is genuinely
    consistent, so a spectral-consistency loss computed on stub data has a real
    optimum near zero and a smoke run can catch a sign error in that loss.

    Per-band levels approximate the vegetation reflectance measured in the real
    SEN2NAIPv2 records (R 0.058, G 0.057, B 0.034, NIR 0.249), so smoke output
    is not merely in-range but physically plausible.
    """

    SOURCE_NAME = "synthetic_stub"

    # Reflectance levels keyed by band name, from the measured SEN2NAIPv2 record
    # NA5120_E1183N0757. Bands not listed fall back to 0.1.
    _BAND_LEVELS = {"B04": 0.058, "B03": 0.057, "B02": 0.034, "B08": 0.249}

    def __init__(self, cfg: Any, validate: bool = True) -> None:
        self.BANDS = tuple(cfg["dataset"]["bands"])
        super().__init__(cfg, validate=validate)

        stub_cfg = cfg.get("synthetic_stub", {}) if hasattr(cfg, "get") else {}
        self.num_samples = int(stub_cfg.get("num_samples", 4) or 4)
        self.base_seed = int(stub_cfg.get("seed", 0) or 0)

        self.lr_size = int(cfg["sr"]["lr_patch_size"])
        self.hr_size = int(cfg["sr"]["hr_patch_size"])
        if self.hr_size != self.lr_size * self.scale:
            raise ValueError(
                f"cfg.sr is inconsistent: hr_patch_size={self.hr_size} but "
                f"lr_patch_size={self.lr_size} * scale={self.scale} = "
                f"{self.lr_size * self.scale}."
            )

    def __len__(self) -> int:
        return self.num_samples

    def load_sample(self, idx: int) -> Dict[str, Any]:
        """Fabricate one LR/HR pair.

        Args:
            idx: Index in ``[0, len(self))``.

        Returns:
            Sample dict per :meth:`SRPairDataset.__getitem__`. ``lr`` is float32
            ``(C, lr_patch_size, lr_patch_size)`` and ``hr`` is float32
            ``(C, hr_patch_size, hr_patch_size)``, both surface reflectance in
            roughly ``[0.01, 0.4]`` -- comfortably inside the valid range, with
            no nodata and no NaN. ``meta["crs"]`` is a plausible UTM string and
            ``meta["geotransform"]`` a GDAL 6-tuple at 2.5 m; both are fake.

        Raises:
            IndexError: ``idx`` is out of range.
        """
        if not 0 <= idx < self.num_samples:
            raise IndexError(
                f"{type(self).__name__} index {idx} out of range for "
                f"{self.num_samples} samples."
            )

        rng = np.random.default_rng(self.base_seed + idx)
        n_bands = len(self.BANDS)

        # Smooth structure shared across bands, so channels stay correlated the
        # way real imagery is, plus per-band texture.
        coarse = rng.random((n_bands, self.lr_size // 2, self.lr_size // 2))
        structure = np.repeat(
            np.repeat(coarse, self.scale * 2, axis=1), self.scale * 2, axis=2
        )[:, : self.hr_size, : self.hr_size].astype(np.float32)

        hr = np.empty((n_bands, self.hr_size, self.hr_size), dtype=np.float32)
        for band_idx, band_name in enumerate(self.BANDS):
            level = self._BAND_LEVELS.get(str(band_name), 0.1)
            texture = rng.normal(0.0, level * 0.08, (self.hr_size, self.hr_size))
            hr[band_idx] = level * (0.6 + 0.8 * structure[band_idx]) + texture

        # Keep strictly positive without clipping the top: negative reflectance
        # is unphysical, but we must not cap bright values (see the working agreement).
        np.maximum(hr, np.float32(0.001), out=hr)

        # LR is the exact block mean of HR -- a clean scale-4 degradation.
        lr = (
            hr.reshape(
                n_bands, self.lr_size, self.scale, self.lr_size, self.scale
            )
            .mean(axis=(2, 4))
            .astype(np.float32)
        )

        return {
            "lr": torch.from_numpy(lr),
            "hr": torch.from_numpy(hr),
            "meta": {
                "sample_id": f"stub_{idx:04d}",
                "source_dataset": self.SOURCE_NAME,
                "crs": "EPSG:32643",
                # GDAL order: (x0, xres, xskew, y0, yskew, yres) at HR scale.
                "geotransform": (
                    500000.0 + idx * 1000.0,
                    self.cfg["dataset"]["hr_gsd_m"],
                    0.0,
                    2000000.0,
                    0.0,
                    -self.cfg["dataset"]["hr_gsd_m"],
                ),
                # (lon, lat) degrees. Spread across a ~2 degree box near
                # Bengaluru so the geographic split has something real to chew
                # on in smoke mode: samples land in a handful of clusters rather
                # than one blob or N singletons.
                "centroid_lonlat": (
                    77.5 + (idx % 4) * 0.5 + (idx // 16) * 0.01,
                    12.9 + (idx // 4 % 4) * 0.5 + (idx // 16) * 0.01,
                ),
                "synthetic": True,
            },
        }
