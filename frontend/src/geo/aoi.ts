/**
 * AOI geometry and the tile budget. Pure functions, no map, no DOM.
 *
 * Distances use a spherical Earth. At Delhi's latitude that is good to about
 * 0.5 %, well inside the ceil() of the pixel counts for any AOI the budget
 * admits. The real input raster lives on the Sentinel-2 UTM grid, so every
 * pixel count here is an estimate and is labelled "≈" in the UI.
 */
import type { FrontendConfig } from "../config";

/** [west, south, east, north], degrees, EPSG:4326 (WGS84 lon/lat). */
export type Bbox = [number, number, number, number];
/** [lon, lat], degrees, EPSG:4326. */
export type LngLat = [number, number];

/** IUGG mean Earth radius, metres. */
const EARTH_RADIUS_M = 6_371_008.8;
const DEG = Math.PI / 180;
/** Fraction of a pixel below which a remainder is float noise, not a partial pixel. */
const PIXEL_EPS = 1e-6;

/** The bbox spanned by two opposite corners, in either order. */
export function bboxFromCorners(a: LngLat, b: LngLat): Bbox {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
}

/** East-west extent (measured at the mid-latitude) and north-south extent, metres. */
export function bboxSizeM([west, south, east, north]: Bbox): { widthM: number; heightM: number } {
  const midLat = ((south + north) / 2) * DEG;
  return {
    widthM: EARTH_RADIUS_M * Math.cos(midLat) * (east - west) * DEG,
    heightM: EARTH_RADIUS_M * (north - south) * DEG,
  };
}

/** The bbox of widthM x heightM metres centred on `center`. Inverse of bboxSizeM. */
export function bboxAround(center: LngLat, widthM: number, heightM: number): Bbox {
  const dLat = heightM / EARTH_RADIUS_M / DEG;
  const dLon = widthM / (EARTH_RADIUS_M * Math.cos(center[1] * DEG)) / DEG;
  return [center[0] - dLon / 2, center[1] - dLat / 2, center[0] + dLon / 2, center[1] + dLat / 2];
}

/**
 * Tile origins along one axis, enumerated exactly as src/infer/tiled.py::sr_array
 * does: len(range(0, max(1, n - overlap), tile - overlap)).
 */
export function tilesAlongAxis(nLrPx: number, tile: number, overlap: number): number {
  if (!Number.isInteger(nLrPx) || nLrPx < 1) throw new RangeError(`tilesAlongAxis: nLrPx must be an integer >= 1, got ${nLrPx}`);
  if (!(overlap >= 0 && overlap < tile)) throw new RangeError(`tilesAlongAxis: need 0 <= overlap < tile, got overlap=${overlap} tile=${tile}`);
  return Math.ceil(Math.max(1, nLrPx - overlap) / (tile - overlap));
}

export interface AoiEstimate {
  widthM: number;
  heightM: number;
  /** Input raster size at cfg.lrGsdM, pixels (≈). */
  lrPx: { width: number; height: number };
  /** Output raster size, exactly lrPx * scale. */
  srPx: { width: number; height: number };
  srGsdM: number;
  tiles: { x: number; y: number; total: number; max: number };
  /** Size of the SR raster alone: width * height * bands * bytes per band. */
  outputBytes: number;
  overBudget: boolean;
}

export function estimateAoi(
  bbox: Bbox,
  cfg: Pick<FrontendConfig, "lrGsdM" | "scale" | "tile" | "bands" | "outputBytesPerBand">,
): AoiEstimate {
  const { widthM, heightM } = bboxSizeM(bbox);
  // Partial pixels round up, but not float noise: a degrees<->metres round trip
  // turns an exact 300 px into 300.0000000001, which a bare ceil makes 301.
  const pixels = (m: number) => Math.max(1, Math.ceil(m / cfg.lrGsdM - PIXEL_EPS));
  const lrPx = { width: pixels(widthM), height: pixels(heightM) };
  const srPx = { width: lrPx.width * cfg.scale, height: lrPx.height * cfg.scale };
  const x = tilesAlongAxis(lrPx.width, cfg.tile.lrPx, cfg.tile.overlapLrPx);
  const y = tilesAlongAxis(lrPx.height, cfg.tile.lrPx, cfg.tile.overlapLrPx);
  return {
    widthM,
    heightM,
    lrPx,
    srPx,
    srGsdM: cfg.lrGsdM / cfg.scale,
    tiles: { x, y, total: x * y, max: cfg.tile.maxTiles },
    outputBytes: srPx.width * srPx.height * cfg.bands.length * cfg.outputBytesPerBand,
    overBudget: x * y > cfg.tile.maxTiles,
  };
}
