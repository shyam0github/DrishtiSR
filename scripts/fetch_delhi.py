"""Fetch real Sentinel-2 L2A scenes over Delhi NCR from Planetary Computer.

WHY THIS EXISTS. The model trains on SEN2NAIPv2, which is North American. The
deployment geography is India, and the two do not look alike: NCR is dense
low-albedo urban fabric, small irregular agricultural parcels, and a winter
aerosol load that no NAIP-derived training pair contains. A super-resolution
result that has only ever been measured on the training distribution is a
result about California. This script pulls the actual target imagery so the
model can be measured on the geography it is for.

These are INFERENCE targets, not training data. Nothing here produces LR/HR
pairs and nothing here enters a training split.

BAND ORDER. The stack order is read from ``cfg.dataset.bands`` and is NOT
restated in this file or in the ``delhi`` config block. That list is
``[B04, B03, B02, B08]`` -- red, green, blue, NIR -- verified in
``configs/base.yaml`` against a SEN2NAIPv2 record, and it is the channel axis
every tensor in this project uses. Reading it from the same key the dataset
reads means a Delhi scene cannot silently disagree with the network's input
layout; a hardcoded list here is exactly how red and blue end up transposed at
inference time with no error message.

RADIOMETRY, AND ONE THING THE CALLER MUST KNOW. The written GeoTIFF holds RAW
uint16 L2A digital numbers, untouched -- not clipped, not rescaled, not
offset-corrected. Converting to surface reflectance is::

    reflectance = (dn + boa_add_offset) / cfg.dataset.reflectance_scale

For Sentinel-2 processing baseline 04.00 and later (anything after 2022-01-25,
which is every item this script's date window can return) ESA applies a
``BOA_ADD_OFFSET`` of -1000. Ignoring it inflates reflectance by 0.1
everywhere, which over dark targets -- water, asphalt, shadow, i.e. most of an
urban scene -- is a large fractional error and would break the
spectral-consistency objective outright. The offset is therefore READ from the
item's own metadata, recorded per band in the manifest, and reported at the
end. It is deliberately not applied to the pixels: the file stays the raw
product, and the single place that scales digital numbers to reflectance stays
the dataset code.

NO REPROJECTION. Bands are read in the item's native UTM CRS with a windowed
read, and written with the window's own transform. The AOI bbox is given in
WGS84 and transformed into that CRS to find the window. Resampling a scene
before super-resolving it would destroy the high-frequency content the whole
project is about.

--smoke. Offline, per the contract stated in the ``smoke`` block of
``configs/base.yaml``: --smoke must never touch the network. A download script
has no meaningful offline half, so smoke fabricates a raster in memory and runs
it through the entire write path -- band ordering, the WGS84 -> UTM transform,
the uint16 stack, the GeoTIFF profile, the statistics, the manifest schema --
skipping only the STAC search and the HTTP read, and writes to ``*_smoke``
names so it can never overwrite a real scene. It proves the plumbing. It cannot
prove the imagery.

Examples:
    # Offline pre-flight:
    .venv/Scripts/python.exe scripts/fetch_delhi.py --smoke

    # The real fetch:
    .venv/Scripts/python.exe scripts/fetch_delhi.py

    # Loosen the cloud threshold if the window returns nothing:
    .venv/Scripts/python.exe scripts/fetch_delhi.py --set delhi.max_cloud_cover=10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

__all__ = [
    "search_best_item",
    "read_bbox_stack",
    "write_stack",
    "summarise_stack",
    "fetch_aoi",
]

# STAC property holding the scene-level cloud percentage. Named once so the
# search filter and the sort key cannot drift apart.
_CLOUD_KEY = "eo:cloud_cover"


def _require(cfg: Any, *keys: str) -> Any:
    """Fetch a nested config value, failing with the full key path if absent.

    Args:
        cfg: The composed config.
        *keys: Successive keys, e.g. ``("delhi", "collection")``.

    Returns:
        The value at that path.

    Raises:
        KeyError: Any key in the path is missing.
    """
    node = cfg
    for i, key in enumerate(keys):
        if key not in node:
            raise KeyError(
                f"configs/base.yaml is missing {'.'.join(keys[: i + 1])}. "
                f"scripts/fetch_delhi.py reads every setting from config and "
                f"hardcodes none of them."
            )
        node = node[key]
    return node


def _bbox_contains(outer: Sequence[float], inner: Sequence[float]) -> bool:
    """Whether ``outer`` fully contains ``inner``. Both WGS84 [w, s, e, n]."""
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def search_best_item(cfg: Any, aoi: Any, logger) -> Any:
    """Return the least-cloudy STAC item that fully covers one AOI.

    Queries ``cfg.delhi.stac_endpoint`` for ``cfg.delhi.collection`` over the
    configured date window with ``eo:cloud_cover < cfg.delhi.max_cloud_cover``,
    asks the server to sort ascending by cloud cover, and re-sorts client-side
    so the choice is deterministic regardless of whether the endpoint honoured
    the sort.

    Candidates whose own bbox does not CONTAIN the AOI bbox are discarded. STAC
    bbox search is intersects, not contains, so without this filter a granule
    clipping one corner of the AOI ranks on its scene-wide cloud cover and then
    yields a raster that is mostly nodata -- a partial scene that looks like a
    successful fetch.

    Args:
        cfg: Composed config; reads the ``delhi`` block.
        aoi: One entry of ``cfg.delhi.aois`` (``name``, ``bbox``).
        logger: Logger for the search trace.

    Returns:
        A signed :class:`pystac.Item`, assets rewritten with SAS tokens by
        ``planetary_computer.sign``.

    Raises:
        RuntimeError: No item satisfies the window, the cloud threshold and the
            containment requirement. The message states which of the three
            filters emptied the set, because the fix differs for each.
    """
    import planetary_computer
    from pystac_client import Client

    d = cfg["delhi"]
    bbox = [float(v) for v in aoi["bbox"]]
    max_cloud = float(d["max_cloud_cover"])
    window = f"{d['datetime_start']}/{d['datetime_end']}"

    logger.info(
        "AOI %s: searching %s for %s over %s, %s < %.1f%%",
        aoi["name"],
        d["stac_endpoint"],
        d["collection"],
        window,
        _CLOUD_KEY,
        max_cloud,
    )

    catalog = Client.open(str(d["stac_endpoint"]))
    search = catalog.search(
        collections=[str(d["collection"])],
        bbox=bbox,
        datetime=window,
        query={_CLOUD_KEY: {"lt": max_cloud}},
        sortby=[{"field": f"properties.{_CLOUD_KEY}", "direction": "asc"}],
        max_items=int(d["max_items"]),
    )
    candidates = list(search.items())
    logger.info(
        "AOI %s: %d item(s) matched the window and cloud filter.",
        aoi["name"],
        len(candidates),
    )
    if not candidates:
        raise RuntimeError(
            f"AOI {aoi['name']}: no {d['collection']} item over bbox {bbox} in "
            f"{window} with {_CLOUD_KEY} < {max_cloud}. Widen the date window "
            f"(delhi.datetime_start / delhi.datetime_end) or raise "
            f"delhi.max_cloud_cover -- both are config keys, do not edit this "
            f"script."
        )

    covering = [it for it in candidates if _bbox_contains(it.bbox, bbox)]
    logger.info(
        "AOI %s: %d of those fully contain the AOI bbox.",
        aoi["name"],
        len(covering),
    )
    if not covering:
        raise RuntimeError(
            f"AOI {aoi['name']}: {len(candidates)} item(s) passed the date and "
            f"cloud filters but none CONTAINS bbox {bbox} -- they only "
            f"intersect it. The AOI likely straddles an MGRS tile boundary. "
            f"Shrink the AOI in delhi.aois, or mosaic, which this script "
            f"deliberately does not do."
        )

    # Server-side sortby is advisory; sort again so the selection is a property
    # of the data rather than of the endpoint's behaviour on the day.
    covering.sort(key=lambda it: (float(it.properties[_CLOUD_KEY]), it.id))
    best = covering[0]
    runner_up = (
        f"{covering[1].id} at {float(covering[1].properties[_CLOUD_KEY]):.3f}%"
        if len(covering) > 1
        else "none"
    )
    logger.info(
        "AOI %s: selected %s (%s, cloud %.3f%%); next best %s.",
        aoi["name"],
        best.id,
        best.properties["datetime"],
        float(best.properties[_CLOUD_KEY]),
        runner_up,
    )
    return planetary_computer.sign(best)


def _band_offset(
    item: Any, band: str, cfg: Any
) -> Tuple[Optional[float], str]:
    """Return the BOA additive offset for one band, in digital numbers.

    Sentinel-2 processing baseline 04.00+ carries a ``BOA_ADD_OFFSET`` (-1000)
    that must be added to the digital number before scaling to reflectance.
    The item's own metadata is preferred, in each of the shapes a STAC item can
    declare it.

    THE FALLBACK IS NOT COSMETIC. MEASURED 2026-09-07: Planetary Computer's
    ``sentinel-2-l2a`` items publish the offset in none of those shapes -- no
    ``s2:boa_add_offset`` property, and assets carrying no ``raster:bands``
    block at all. Reporting "no offset" on a baseline-05.11 scene would put
    reflectance 0.1 too high across the image, which over water, asphalt and
    shadow is a large fractional error that looks like a plausible number
    rather than a missing one. So when the item is silent the offset is derived
    from ``s2:processing_baseline`` against
    ``cfg.delhi.boa_offset_baseline_threshold``, and the caller is told which
    of the two happened.

    Args:
        item: The STAC item.
        band: Asset key, e.g. ``"B04"``.
        cfg: Composed config; reads ``delhi.boa_offset_baseline_threshold`` and
            ``delhi.boa_offset_dn``.

    Returns:
        ``(offset_dn, source)``. ``source`` is ``"item-metadata"`` when the item
        declared it, ``"derived-from-processing-baseline"`` when it was inferred,
        ``"none-pre-baseline-04.00"`` when the product genuinely predates the
        offset, or ``"unknown"`` when the item declares neither an offset nor a
        baseline -- in which case ``offset_dn`` is ``None`` and reflectance from
        this scene must not be trusted until it is resolved by hand.
    """
    props = item.properties
    flat = props.get("s2:boa_add_offset")
    if flat is not None:
        return float(flat), "item-metadata"

    raster_bands = item.assets[band].extra_fields.get("raster:bands")
    if raster_bands and raster_bands[0].get("offset") is not None:
        return float(raster_bands[0]["offset"]), "item-metadata"

    per_band = props.get("boa_add_offset") or props.get("s2:boa_add_offset_per_band")
    if isinstance(per_band, dict) and band in per_band:
        return float(per_band[band]), "item-metadata"

    baseline = props.get("s2:processing_baseline")
    if baseline is None:
        return None, "unknown"

    d = cfg["delhi"]
    threshold = str(d["boa_offset_baseline_threshold"])
    as_tuple = lambda v: tuple(int(p) for p in str(v).split("."))  # noqa: E731
    if as_tuple(baseline) >= as_tuple(threshold):
        return float(d["boa_offset_dn"]), "derived-from-processing-baseline"
    return 0.0, f"none-pre-baseline-{threshold}"


def read_bbox_stack(
    item: Any,
    bbox: Sequence[float],
    bands: Sequence[str],
    cfg: Any,
    logger,
) -> Tuple[np.ndarray, Affine, Any, Dict[str, Dict[str, Any]]]:
    """Windowed-read one AOI from a signed item, stacked in ``bands`` order.

    Every band is read in the item's NATIVE CRS with no warping. The WGS84 bbox
    is transformed into that CRS and turned into a pixel window; all the bands
    named here are 10 m, so all their windows must agree, and that agreement is
    asserted rather than assumed.

    Args:
        item: Signed STAC item.
        bbox: AOI as WGS84 ``[west, south, east, north]``.
        bands: Asset keys in the desired channel order -- pass
            ``cfg.dataset.bands`` so the stack matches the training layout.
        cfg: Composed config; reads ``delhi.gdal_env`` and the BOA-offset keys.
        logger: Logger.

    Returns:
        ``(stack, transform, crs, offsets)``, where ``stack`` is
        ``(C, H, W) uint16`` RAW L2A digital numbers in ``bands`` order --
        nominally 0-10000 for reflectance 0-1, UNCLIPPED and NOT
        offset-corrected (see the module docstring); ``transform`` is the
        window's affine in the native CRS; ``crs`` is that native UTM CRS; and
        ``offsets`` maps each band to
        ``{"offset_dn": float | None, "source": str}`` as returned by
        :func:`_band_offset`.

    Raises:
        KeyError: The item has no asset for a requested band.
        RuntimeError: Bands disagree on CRS, transform or shape; a band is not
            uint16; or the AOI is not fully inside the raster.
    """
    stack: List[np.ndarray] = []
    offsets: Dict[str, Dict[str, Any]] = {}
    ref_transform: Optional[Affine] = None
    ref_crs = None
    ref_shape: Optional[Tuple[int, int]] = None

    gdal_env = {str(k): str(v) for k, v in dict(cfg["delhi"]["gdal_env"]).items()}
    with rasterio.Env(**gdal_env):
        for band in bands:
            if band not in item.assets:
                raise KeyError(
                    f"Item {item.id} has no asset {band!r}. Available: "
                    f"{sorted(item.assets)}. cfg.dataset.bands must name assets "
                    f"that exist in cfg.delhi.collection."
                )
            href = item.assets[band].href
            with rasterio.open(href) as src:
                if src.dtypes[0] != "uint16":
                    raise RuntimeError(
                        f"Item {item.id} band {band} is {src.dtypes[0]}, not "
                        f"uint16. L2A digital numbers are uint16; a different "
                        f"dtype means the asset is not what this script assumes."
                    )
                native = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
                requested = window_from_bounds(*native, transform=src.transform)
                requested = requested.round_offsets().round_lengths()
                full = Window(0, 0, src.width, src.height)
                window = requested.intersection(full)
                if window.width < requested.width or window.height < requested.height:
                    raise RuntimeError(
                        f"Item {item.id} band {band}: the AOI window "
                        f"{requested.width}x{requested.height} px is not fully "
                        f"inside the {src.width}x{src.height} px raster -- only "
                        f"{window.width}x{window.height} px overlap. The "
                        f"containment filter in search_best_item should have "
                        f"prevented this, so the item's declared bbox and its "
                        f"actual footprint disagree."
                    )
                data = src.read(1, window=window)
                transform = src.window_transform(window)
                crs = src.crs

            if ref_transform is None:
                ref_transform, ref_crs, ref_shape = transform, crs, data.shape
                logger.info(
                    "Item %s: window %dx%d px in %s at 10 m, no reprojection.",
                    item.id,
                    data.shape[1],
                    data.shape[0],
                    crs,
                )
            elif (transform, crs, data.shape) != (ref_transform, ref_crs, ref_shape):
                raise RuntimeError(
                    f"Item {item.id} band {band} does not align with {bands[0]}: "
                    f"crs {crs} vs {ref_crs}, shape {data.shape} vs {ref_shape}, "
                    f"transform {transform} vs {ref_transform}. Every band "
                    f"requested here is 10 m and must share one grid; stacking "
                    f"them anyway would put different ground positions on the "
                    f"same pixel."
                )

            offset_dn, offset_source = _band_offset(item, band, cfg)
            offsets[str(band)] = {"offset_dn": offset_dn, "source": offset_source}
            stack.append(data)

    sources = {v["source"] for v in offsets.values()}
    if "unknown" in sources:
        logger.warning(
            "Item %s declares neither a BOA offset nor a processing baseline. "
            "Reflectance from this scene is UNRESOLVED -- do not scale it until "
            "the offset is established by hand.",
            item.id,
        )
    else:
        logger.info(
            "BOA additive offset %s DN (%s).",
            sorted({v["offset_dn"] for v in offsets.values()}),
            "/".join(sorted(sources)),
        )
    return np.stack(stack, axis=0), ref_transform, ref_crs, offsets


def write_stack(
    path: Path,
    stack: np.ndarray,
    transform: Affine,
    crs: Any,
    cfg: Any,
    bands: Sequence[str],
    logger,
) -> None:
    """Write a ``(C, H, W)`` uint16 stack as a tiled, band-named GeoTIFF.

    Args:
        path: Destination ``.tif``. Parent directories are created.
        stack: ``(C, H, W) uint16`` raw L2A digital numbers, nominally 0-10000
            for reflectance 0-1 but unclipped.
        transform: Affine in ``crs``.
        crs: The item's native CRS. Written as-is; nothing is reprojected.
        cfg: Composed config; reads ``delhi.tiled``/``blocksize``/``compress``.
        bands: Channel names, written as per-band descriptions so the file
            carries its own band order rather than relying on the manifest.
        logger: Logger.

    Raises:
        ValueError: ``stack`` is not 3-D uint16, or its channel count does not
            match ``bands``.
    """
    if stack.ndim != 3 or stack.dtype != np.uint16:
        raise ValueError(
            f"Expected a (C, H, W) uint16 stack, got shape {stack.shape} dtype "
            f"{stack.dtype}."
        )
    if stack.shape[0] != len(bands):
        raise ValueError(
            f"Stack has {stack.shape[0]} channels but {len(bands)} band names "
            f"{list(bands)} were given."
        )

    d = cfg["delhi"]
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(stack.shape[1]),
        "width": int(stack.shape[2]),
        "count": int(stack.shape[0]),
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "tiled": bool(d["tiled"]),
        "blockxsize": int(d["blocksize"]),
        "blockysize": int(d["blocksize"]),
        "compress": str(d["compress"]),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)
        for i, band in enumerate(bands, start=1):
            dst.set_band_description(i, str(band))
    logger.info("Wrote %s (%.1f MB).", path, path.stat().st_size / 1e6)


def summarise_stack(stack: np.ndarray, bands: Sequence[str], cfg: Any, logger) -> Dict:
    """Per-band statistics on raw digital numbers, with a range sanity check.

    The check is REPORTED, never enforced. Bright roofs, specular water and
    haze legitimately exceed the nominal ceiling, and clipping them would
    destroy the radiometry the spectral-consistency objective depends on -- so
    a breach is a logged warning and a manifest flag, not an exception and
    certainly not a clamp.

    Args:
        stack: ``(C, H, W) uint16`` raw L2A digital numbers.
        bands: Channel names, in ``stack``'s channel order.
        cfg: Composed config; reads ``delhi.expect_dn_min``/``expect_dn_max``
            and ``dataset.reflectance_scale``.
        logger: Logger.

    Returns:
        ``{"per_band": {band: {...}}, "overall": {...},
        "in_expected_range": bool}``. ``zero_fraction`` is the share of pixels
        equal to 0, which in an L2A COG is the granule's nodata fill -- a high
        value means the AOI sits in a black margin. The reflectance figures are
        marked ``_uncorrected`` because the BOA additive offset has not been
        applied; see the module docstring.
    """
    scale = float(cfg["dataset"]["reflectance_scale"])
    lo = float(cfg["delhi"]["expect_dn_min"])
    hi = float(cfg["delhi"]["expect_dn_max"])

    per_band: Dict[str, Dict[str, float]] = {}
    for i, band in enumerate(bands):
        plane = stack[i]
        per_band[str(band)] = {
            "min_dn": int(plane.min()),
            "max_dn": int(plane.max()),
            "mean_dn": float(plane.mean()),
            "min_reflectance_uncorrected": float(plane.min()) / scale,
            "max_reflectance_uncorrected": float(plane.max()) / scale,
            "mean_reflectance_uncorrected": float(plane.mean()) / scale,
            "zero_fraction": float((plane == 0).mean()),
        }

    overall = {
        "min_dn": int(stack.min()),
        "max_dn": int(stack.max()),
        "mean_dn": float(stack.mean()),
        "zero_fraction": float((stack == 0).mean()),
    }
    in_range = lo <= overall["min_dn"] and overall["max_dn"] <= hi
    if not in_range:
        logger.warning(
            "Digital numbers span [%d, %d], outside the expected [%g, %g]. NOT "
            "clipped -- bright targets legitimately exceed the ceiling. Inspect "
            "the scene before drawing conclusions from it.",
            overall["min_dn"],
            overall["max_dn"],
            lo,
            hi,
        )
    if overall["zero_fraction"] > 0.01:
        logger.warning(
            "%.2f%% of pixels are exactly 0, i.e. granule nodata. The AOI may "
            "sit partly in the black margin of the tile.",
            100.0 * overall["zero_fraction"],
        )
    return {"per_band": per_band, "overall": overall, "in_expected_range": in_range}


def _entry(
    aoi: Any,
    item_id: str,
    item_datetime: str,
    cloud_cover: Optional[float],
    crs: Any,
    transform: Affine,
    stack: np.ndarray,
    bands: Sequence[str],
    offsets: Dict[str, Dict[str, Any]],
    path: Path,
    stats: Dict,
    cfg: Any,
    extra: Dict,
) -> Dict:
    """Assemble one AOI's manifest record. Pure; touches no filesystem."""
    bounds = rasterio.transform.array_bounds(stack.shape[1], stack.shape[2], transform)
    return {
        "aoi": str(aoi["name"]),
        "description": str(aoi.get("description", "")),
        "file": path.relative_to(repo_root()).as_posix(),
        "item_id": item_id,
        "datetime": item_datetime,
        "cloud_cover": cloud_cover,
        "crs": str(crs),
        "epsg": rasterio.crs.CRS.from_user_input(crs).to_epsg(),
        "requested_bbox_wgs84": [float(v) for v in aoi["bbox"]],
        "bounds_native": [float(v) for v in bounds],
        "transform_gdal": [float(v) for v in transform.to_gdal()],
        "pixel_size_m": [abs(float(transform.a)), abs(float(transform.e))],
        "shape_chw": [int(v) for v in stack.shape],
        "band_order": [str(b) for b in bands],
        "band_order_source": "configs/base.yaml :: dataset.bands",
        "dtype": str(stack.dtype),
        "reflectance_scale": float(cfg["dataset"]["reflectance_scale"]),
        "boa_add_offset_dn": {b: v["offset_dn"] for b, v in offsets.items()},
        "boa_add_offset_source": {b: v["source"] for b, v in offsets.items()},
        "reflectance_formula": (
            "(dn + boa_add_offset_dn[band]) / reflectance_scale; pixels on disk "
            "are RAW digital numbers, unclipped and not offset-corrected"
        ),
        "statistics": stats,
        **extra,
    }


def fetch_aoi(cfg: Any, aoi: Any, bands: Sequence[str], logger) -> Dict:
    """Search, read, write and describe one AOI. Requires network.

    Args:
        cfg: Composed config.
        aoi: One entry of ``cfg.delhi.aois``.
        bands: Channel order, from ``cfg.dataset.bands``.
        logger: Logger.

    Returns:
        The manifest record for this AOI.
    """
    d = cfg["delhi"]
    item = search_best_item(cfg, aoi, logger)
    stack, transform, crs, offsets = read_bbox_stack(
        item, [float(v) for v in aoi["bbox"]], bands, cfg, logger
    )

    stamp = datetime.fromisoformat(
        str(item.properties["datetime"]).replace("Z", "+00:00")
    )
    filename = str(d["filename_template"]).format(
        name=str(aoi["name"]),
        date=stamp.strftime(str(d["filename_date_format"])),
    )
    path = repo_root() / str(d["output_dir"]) / filename

    write_stack(path, stack, transform, crs, cfg, bands, logger)
    stats = summarise_stack(stack, bands, cfg, logger)
    return _entry(
        aoi,
        item.id,
        str(item.properties["datetime"]),
        float(item.properties[_CLOUD_KEY]),
        crs,
        transform,
        stack,
        bands,
        offsets,
        path,
        stats,
        cfg,
        {
            "platform": item.properties.get("platform"),
            "mgrs_tile": item.properties.get("s2:mgrs_tile"),
            "processing_baseline": item.properties.get("s2:processing_baseline"),
            "source": "microsoft-planetary-computer",
            "synthetic": False,
        },
    )


def fetch_aoi_smoke(cfg: Any, aoi: Any, bands: Sequence[str], logger) -> Dict:
    """Offline stand-in for :func:`fetch_aoi`. No network, fabricated pixels.

    Runs the identical transform / stack / write / summarise / manifest path
    against a raster built in memory, so ``--smoke`` exercises everything
    except the STAC query and the HTTP read.

    Args:
        cfg: Composed config, with the ``smoke`` block merged.
        aoi: One entry of ``cfg.delhi.aois``.
        bands: Channel order, from ``cfg.dataset.bands``.
        logger: Logger.

    Returns:
        The manifest record, flagged ``"synthetic": True``.
    """
    d = cfg["delhi"]
    size = int(d["smoke_size_px"])
    crs = rasterio.crs.CRS.from_user_input(str(d["smoke_crs"]))
    bbox = [float(v) for v in aoi["bbox"]]

    # A real bbox through a real projection, so the transform path is genuinely
    # exercised even though the pixels are not.
    west, south, east, north = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    transform = rasterio.transform.from_bounds(west, south, east, north, size, size)

    rng = np.random.default_rng(int(cfg["seed"]))
    stack = rng.integers(0, 10000, size=(len(bands), size, size), dtype=np.uint16)

    filename = str(d["filename_template"]).format(
        name=str(aoi["name"]),
        date=datetime.now(timezone.utc).strftime(str(d["filename_date_format"])),
    )
    path = repo_root() / str(d["output_dir"]) / filename

    write_stack(path, stack, transform, crs, cfg, bands, logger)
    stats = summarise_stack(stack, bands, cfg, logger)
    return _entry(
        aoi,
        "SMOKE-SYNTHETIC-NO-ITEM",
        datetime.now(timezone.utc).isoformat(),
        None,
        crs,
        transform,
        stack,
        bands,
        {str(b): {"offset_dn": None, "source": "smoke-synthetic"} for b in bands},
        path,
        stats,
        cfg,
        {
            "source": "fabricated in memory by --smoke; NOT imagery",
            "synthetic": True,
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the least-cloudy Sentinel-2 L2A scene per Delhi NCR AOI from "
            "Planetary Computer, clipped to the AOI in native UTM."
        )
    )
    add_standard_args(parser)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)
    seed_everything(cfg["seed"])
    logger = get_logger("fetch_delhi", log_file=cfg["paths"]["log_file"])

    bands = [str(b) for b in _require(cfg, "dataset", "bands")]
    d = _require(cfg, "delhi")
    logger.info(
        "Band order %s, read from cfg.dataset.bands -- the same key the training "
        "dataset reads, so the stack matches the network's channel axis.",
        bands,
    )
    if args.smoke:
        logger.warning(
            "--smoke: fabricating pixels in memory, NO network. This proves the "
            "write path only. It does not fetch imagery."
        )

    records = []
    for aoi in d["aois"]:
        if args.smoke:
            records.append(fetch_aoi_smoke(cfg, aoi, bands, logger))
        else:
            records.append(fetch_aoi(cfg, aoi, bands, logger))

    manifest_path = repo_root() / str(d["output_dir"]) / str(d["manifest_name"])
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/fetch_delhi.py",
        "smoke": bool(args.smoke),
        "stac_endpoint": str(d["stac_endpoint"]),
        "collection": str(d["collection"]),
        "datetime_window": [str(d["datetime_start"]), str(d["datetime_end"])],
        "max_cloud_cover": float(d["max_cloud_cover"]),
        "band_order": bands,
        "band_order_source": "configs/base.yaml :: dataset.bands",
        "reflectance_scale": float(cfg["dataset"]["reflectance_scale"]),
        "items": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %s", manifest_path)

    print()
    for rec in records:
        s = rec["statistics"]["overall"]
        cloud = "n/a" if rec["cloud_cover"] is None else f"{rec['cloud_cover']:.3f}%"
        print(
            f"AOI {rec['aoi']}: {Path(rec['file']).name}  {rec['datetime'][:10]}  "
            f"cloud {cloud}  shape {tuple(rec['shape_chw'])}  {rec['crs']}  "
            f"bands {'/'.join(rec['band_order'])}  "
            f"DN [{s['min_dn']}, {s['max_dn']}] mean {s['mean_dn']:.0f}"
        )
    offsets = sorted({v for r in records for v in r["boa_add_offset_dn"].values()})
    sources = sorted({v for r in records for v in r["boa_add_offset_source"].values()})
    print(
        f"Reflectance = (dn + {offsets}) / "
        f"{manifest['reflectance_scale']:g}   [offset source: {'/'.join(sources)}]"
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
