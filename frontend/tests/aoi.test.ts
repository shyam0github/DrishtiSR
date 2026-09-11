import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";
import { bboxAround, bboxFromCorners, bboxSizeM, estimateAoi, tilesAlongAxis, type Bbox } from "../src/geo/aoi";
import { BASE_YAML, loadFrontendConfig } from "../scripts/repo-config.mjs";

const cfg = loadFrontendConfig();

/** The loop src/infer/tiled.py::sr_array runs, spelled out: for y in range(0, max(1, n - overlap), step). */
function pythonRangeCount(n: number, tile: number, overlap: number): number {
  let count = 0;
  for (let y = 0; y < Math.max(1, n - overlap); y += tile - overlap) count++;
  return count;
}

describe("tilesAlongAxis", () => {
  it("enumerates exactly as tiled.py for every size up to 3000 px", () => {
    for (let n = 1; n <= 3000; n++) {
      expect(tilesAlongAxis(n, cfg.tile.lrPx, cfg.tile.overlapLrPx)).toBe(pythonRangeCount(n, cfg.tile.lrPx, cfg.tile.overlapLrPx));
    }
  });

  it("pins the 256/32 boundaries", () => {
    expect(tilesAlongAxis(256, 256, 32)).toBe(1);
    expect(tilesAlongAxis(257, 256, 32)).toBe(2);
    expect(tilesAlongAxis(480, 256, 32)).toBe(2);
    expect(tilesAlongAxis(481, 256, 32)).toBe(3);
  });

  it("rejects nonsense instead of returning a count", () => {
    expect(() => tilesAlongAxis(0, 256, 32)).toThrow(RangeError);
    expect(() => tilesAlongAxis(100, 256, 256)).toThrow(RangeError);
  });
});

describe("bbox geometry", () => {
  it("bboxAround and bboxSizeM are inverses", () => {
    const b = bboxAround([77.21, 28.61], 2560, 1800);
    const { widthM, heightM } = bboxSizeM(b);
    expect(widthM).toBeCloseTo(2560, 6);
    expect(heightM).toBeCloseTo(1800, 6);
  });

  it("orders corners regardless of drag direction", () => {
    expect(bboxFromCorners([77.3, 28.5], [77.2, 28.6])).toEqual([77.2, 28.5, 77.3, 28.6]);
  });

  it("output pixels are exactly input pixels times sr.scale", () => {
    const est = estimateAoi(bboxAround([77.21, 28.61], 5000, 3000), cfg);
    expect(est.lrPx).toEqual({ width: 500, height: 300 });
    expect(est.srPx).toEqual({ width: 500 * cfg.scale, height: 300 * cfg.scale });
    expect(est.outputBytes).toBe(500 * 300 * cfg.scale ** 2 * cfg.bands.length * cfg.outputBytesPerBand);
  });
});

describe("the tile budget in configs/base.yaml", () => {
  const raw = parse(readFileSync(BASE_YAML, "utf8")) as { delhi: { aois: { name: string; bbox: Bbox }[] } };

  it("admits delhi AOI A, the largest AOI the demo is meant to show (the stated basis for max_tiles)", () => {
    const aoiA = raw.delhi.aois.find((a) => a.name === "A");
    if (!aoiA) throw new Error("configs/base.yaml: delhi.aois has no AOI named A");
    const est = estimateAoi(aoiA.bbox, cfg);
    expect(est.tiles.total).toBe(36);
    expect(est.overBudget).toBe(false);
  });

  it("refuses all of India", () => {
    expect(estimateAoi([68.1, 6.5, 97.4, 35.7], cfg).overBudget).toBe(true);
  });

  it("admits the placeholder demo footprint", () => {
    const side = cfg.placeholder.hrPx * (cfg.lrGsdM / cfg.scale);
    expect(estimateAoi(bboxAround(cfg.map.center, side, side), cfg).overBudget).toBe(false);
  });
});
