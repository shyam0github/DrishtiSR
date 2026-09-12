"""GeoTIFF I/O for the MVP API: uploads in, float32 SR / uncertainty GeoTIFFs out.

Units: everything returned by :func:`read_input` is SURFACE REFLECTANCE, float32,
unclipped, band order ``cfg.dataset.bands`` (B04, B03, B02, B08). DN conversion
follows the project's convention (``reflectance = (dn - offset) / 10000``,
``src/infer/tiled.py`` ``run_file``); nothing is normalised or clipped.

A "profile" here is ``{"crs": rasterio CRS, "transform": Affine}`` of the LR
grid, or ``None`` when the input carries no CRS.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from src.utils.config import load_config

_CFG = load_config()
BAND_ORDER: Tuple[str, ...] = tuple(str(b) for b in _CFG["dataset"]["bands"])
REFLECTANCE_SCALE = float(_CFG["dataset"]["reflectance_scale"])
L2A_DN_OFFSET = float(_CFG["delhi"]["dn_offset"])  # BOA_ADD_OFFSET, baseline >= 04.00
SCALE = int(_CFG["sr"]["scale"])
MIN_SIDE, MAX_SIDE = 32, 512
DN_MODES = ("auto", "reflectance", "dn10000", "dn10000_offset1000")
AUTO_REFLECTANCE_P99_MAX = 1.5

Profile = Optional[Dict[str, Any]]


class InputError(ValueError):
    """The upload is unusable; the API answers 400 with this message."""

    status = 400


def band_order_warning() -> str:
    return (f"Assumed band order {', '.join(BAND_ORDER)} (R, G, B, NIR); the upload's "
            "band metadata is not used to reorder bands.")


def _convert(raw: np.ndarray, mode: str) -> np.ndarray:
    x = raw.astype(np.float32)
    if mode == "reflectance":
        return x
    if mode == "dn10000":
        return x / np.float32(REFLECTANCE_SCALE)
    if mode == "dn10000_offset1000":
        return (x - np.float32(L2A_DN_OFFSET)) / np.float32(REFLECTANCE_SCALE)
    raise ValueError(f"dn_mode must be one of {DN_MODES}; got {mode!r}.")


def read_input(data: bytes, dn_mode: str = "auto") -> Tuple[np.ndarray, Profile, str, List[str]]:
    """Decode an uploaded 4-band GeoTIFF.

    Returns:
        ``(lr, profile, dn_mode_applied, warnings)``: ``lr`` float32 ``(4, h, w)``
        reflectance; ``profile`` as in the module docstring or ``None``.

    Raises:
        InputError: unreadable file, band count != 4, a side outside
            [32, 512], or non-finite pixels.
        ValueError: unknown ``dn_mode`` (the server rejects it earlier, 422).
    """
    if dn_mode not in DN_MODES:
        raise ValueError(f"dn_mode must be one of {DN_MODES}; got {dn_mode!r}.")
    notes = [band_order_warning()]
    try:
        with MemoryFile(data) as mem, warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with mem.open() as src:
                count, h, w = src.count, src.height, src.width
                if count != len(BAND_ORDER):
                    raise InputError(f"expected {len(BAND_ORDER)} bands ({', '.join(BAND_ORDER)}), "
                                     f"got {count}.")
                if not (MIN_SIDE <= h <= MAX_SIDE and MIN_SIDE <= w <= MAX_SIDE):
                    raise InputError(f"LR size {h}x{w} is outside the supported range: height and width "
                                     f"must each be within [{MIN_SIDE}, {MAX_SIDE}] pixels. Crop the "
                                     "image first.")
                raw = src.read()
                dtype = np.dtype(src.dtypes[0])
                nodata = src.nodata
                crs, transform = src.crs, src.transform
                descriptions = [d for d in src.descriptions if d]
    except RasterioIOError as exc:
        raise InputError(f"could not read the upload as a GeoTIFF: {exc}") from exc

    if len(descriptions) == count and [d.upper() for d in descriptions] != list(BAND_ORDER):
        notes.append(f"Band descriptions in the file are {descriptions}, which differ from the "
                     f"assumed order {list(BAND_ORDER)}; bands were NOT reordered.")

    mask = np.zeros(raw.shape[1:], dtype=bool)
    if nodata is not None and np.isfinite(nodata):
        mask = (raw == np.asarray(nodata, dtype=raw.dtype)).any(axis=0)
    valid = raw[:, ~mask]
    if valid.size and not np.isfinite(valid).all():
        raise InputError("the image contains NaN or infinite pixels outside its nodata mask.")

    applied = dn_mode
    if dn_mode == "auto":
        p99 = float(np.percentile(valid, 99)) if valid.size else 0.0
        if np.issubdtype(dtype, np.floating) and p99 <= AUTO_REFLECTANCE_P99_MAX:
            applied = "reflectance"
        else:
            applied = "dn10000"
            notes.append(f"dn_mode auto: values read as DN / {REFLECTANCE_SCALE:g} (dtype {dtype}, "
                         f"p99 {p99:g}). For Sentinel-2 L2A with processing baseline >= 04.00 "
                         "(DN not offset-corrected) use dn_mode=dn10000_offset1000.")
    lr = _convert(raw, applied)
    if mask.any():
        lr[:, mask] = 0.0
        notes.append(f"{int(mask.sum())} nodata pixels (value {nodata:g}) set to reflectance 0.")
    p99_ref = float(np.percentile(lr, 99))
    if p99_ref > 2.0 or p99_ref < 0.0:
        notes.append(f"Reflectance p99 is {p99_ref:.3g} after dn_mode={applied}; outside the plausible "
                     "range. Check dn_mode.")

    profile: Profile = None
    if crs is not None:
        profile = {"crs": crs, "transform": transform}
    else:
        notes.append("The upload has no CRS; output GeoTIFFs are not georeferenced.")
    return np.ascontiguousarray(lr, dtype=np.float32), profile, applied, notes


# ------------------------------------------------------------------ writing
def write_tif(path: Union[str, Path], data: np.ndarray, profile: Profile,
              pixel_div: float = 1.0) -> Path:
    """float32 GeoTIFF, deflate. ``transform = src.transform x Affine.scale(1/pixel_div)``."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                dtype="float32", compress="deflate", predictor=3)
    if profile is not None:
        meta.update(crs=profile["crs"],
                    transform=profile["transform"] @ Affine.scale(1.0 / pixel_div))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(p, "w", **meta) as dst:
            dst.write(arr)
            if arr.shape[0] == len(BAND_ORDER):
                dst.descriptions = BAND_ORDER
    return p


def write_sr_tif(path: Union[str, Path], sr: np.ndarray, profile: Profile) -> Path:
    """4-band float32 SR reflectance at 1/4 of the LR pixel size, same CRS and origin."""
    return write_tif(path, sr, profile, pixel_div=SCALE)


def write_unc_tif(path: Union[str, Path], unc: np.ndarray, profile: Profile) -> Path:
    """1-band float32 uncertainty (reflectance) on the SR grid."""
    return write_tif(path, unc, profile, pixel_div=SCALE)


# ------------------------------------------------------------------ JSON form
def profile_to_json(profile: Profile) -> Optional[Dict[str, Any]]:
    if profile is None:
        return None
    t = profile["transform"]
    return {"crs": profile["crs"].to_string() if profile["crs"] is not None else None,
            "transform": [t.a, t.b, t.c, t.d, t.e, t.f]}


def profile_from_json(d: Optional[Dict[str, Any]]) -> Profile:
    if not d or not d.get("crs"):
        return None
    return {"crs": CRS.from_user_input(d["crs"]), "transform": Affine(*d["transform"][:6])}
