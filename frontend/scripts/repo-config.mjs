/**
 * Read the frontend's settings out of configs/base.yaml, the repository's
 * single source of defaults (AGENTS.md section 2).
 *
 * Used by vite.config.ts, which bakes the result into the bundle as
 * __DRISHTI_CONFIG__, and by make_placeholders.mjs. The returned object's
 * shape is `FrontendConfig` in src/config.ts; keep the two in step.
 *
 * Fails loudly on any missing or malformed key. A demo that quietly falls back
 * to a default tile budget is the one that hangs in front of the judges.
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";

export const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
export const BASE_YAML = path.join(REPO_ROOT, "configs", "base.yaml");

function fail(dotted, problem) {
  throw new Error(`configs/base.yaml: '${dotted}' ${problem}`);
}

function get(cfg, dotted) {
  let node = cfg;
  for (const key of dotted.split(".")) {
    if (node === null || typeof node !== "object" || !(key in node)) fail(dotted, "is missing");
    node = node[key];
  }
  return node;
}

function number(cfg, dotted, { integer = false, min = -Infinity, above = -Infinity } = {}) {
  const v = get(cfg, dotted);
  if (typeof v !== "number" || !Number.isFinite(v)) fail(dotted, `must be a finite number, got ${JSON.stringify(v)}`);
  if (integer && !Number.isInteger(v)) fail(dotted, `must be an integer, got ${v}`);
  if (v < min) fail(dotted, `must be >= ${min}, got ${v}`);
  if (v <= above) fail(dotted, `must be > ${above}, got ${v}`);
  return v;
}

function string(cfg, dotted) {
  const v = get(cfg, dotted);
  if (typeof v !== "string" || v.length === 0) fail(dotted, `must be a non-empty string, got ${JSON.stringify(v)}`);
  return v;
}

function stringList(cfg, dotted) {
  const v = get(cfg, dotted);
  if (!Array.isArray(v) || v.length === 0 || !v.every((s) => typeof s === "string" && s.length > 0)) {
    fail(dotted, `must be a non-empty list of strings, got ${JSON.stringify(v)}`);
  }
  return v;
}

/** Parse configs/base.yaml (or `file`) into the FrontendConfig shape. */
export function loadFrontendConfig(file = BASE_YAML) {
  const cfg = parse(readFileSync(file, "utf8"));

  const center = get(cfg, "frontend.map.center_lonlat");
  if (!Array.isArray(center) || center.length !== 2 || !center.every(Number.isFinite)) {
    fail("frontend.map.center_lonlat", `must be [lon, lat], got ${JSON.stringify(center)}`);
  }

  const scale = number(cfg, "sr.scale", { integer: true, min: 1 });
  const lrPx = number(cfg, "frontend.tile.lr_px", { integer: true, min: 1 });
  const overlapLrPx = number(cfg, "frontend.tile.overlap_lr_px", { integer: true, min: 0 });
  if (overlapLrPx >= lrPx) fail("frontend.tile.overlap_lr_px", `must be < frontend.tile.lr_px (${lrPx}), got ${overlapLrPx}`);
  const hrPx = number(cfg, "frontend.placeholder.hr_px", { integer: true, min: 1 });
  if (hrPx % scale !== 0) fail("frontend.placeholder.hr_px", `must be a multiple of sr.scale (${scale}), got ${hrPx}`);

  return {
    map: {
      center,
      zoom: number(cfg, "frontend.map.zoom", { min: 0 }),
      basemap: {
        tiles: stringList(cfg, "frontend.map.basemap.tiles"),
        tileSize: number(cfg, "frontend.map.basemap.tile_size", { integer: true, min: 1 }),
        maxZoom: number(cfg, "frontend.map.basemap.max_zoom", { min: 0 }),
        attribution: string(cfg, "frontend.map.basemap.attribution"),
      },
    },
    lrGsdM: number(cfg, "frontend.lr_gsd_m", { above: 0 }),
    scale,
    bands: stringList(cfg, "dataset.bands"),
    outputBytesPerBand: number(cfg, "frontend.output_bytes_per_band", { integer: true, min: 1 }),
    tile: {
      lrPx,
      overlapLrPx,
      maxTiles: number(cfg, "frontend.tile.max_tiles", { integer: true, min: 1 }),
    },
    maxParameters: number(cfg, "runtime.max_parameters", { integer: true, min: 1 }),
    placeholder: {
      hrPx,
      seed: number(cfg, "seed", { integer: true }),
    },
  };
}
