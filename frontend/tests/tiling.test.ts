import { describe, expect, it } from "vitest";
import { tileCount } from "../src/data/tiling";

// Expected values worked through src/infer/tiled.py by hand: step = 256 - 32 = 224,
// starts = range(0, max(1, len - 32), 224).
describe("tileCount mirrors the served predictor's tiling", () => {
  it("an image within one tile is one tile", () => {
    expect(tileCount(128, 128, 256, 32)).toBe(1);
    expect(tileCount(256, 256, 256, 32)).toBe(1);
  });
  it("larger images tile with overlap", () => {
    expect(tileCount(257, 256, 256, 32)).toBe(2); // range(0, 225, 224) -> [0, 224]; range(0, 224, 224) -> [0]
    expect(tileCount(480, 480, 256, 32)).toBe(4); // range(0, 448, 224) -> [0, 224]
    expect(tileCount(1243, 1294, 256, 32)).toBe(6 * 6); // the Delhi scene
    expect(tileCount(100, 600, 256, 32)).toBe(3); // range(0, 68) -> 1; range(0, 568, 224) -> 3
  });
  it("refuses a bad overlap or an empty image", () => {
    expect(() => tileCount(10, 10, 256, 256)).toThrow(RangeError);
    expect(() => tileCount(0, 10, 256, 32)).toThrow(RangeError);
  });
});
