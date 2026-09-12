import { describe, expect, it } from "vitest";
import { countUpValue, easeOutExpo, formatNumber } from "../src/design/motion";
import { clampPan, IDENTITY, MAX_SCALE, zoomAt } from "../src/design/zoom";
import { NAV_ROUTES } from "../src/routes";

describe("count-up", () => {
  it("starts at 0 and lands exactly on the target", () => {
    expect(countUpValue(33.91, 0, 1400)).toBe(0);
    expect(countUpValue(33.91, 1400, 1400)).toBe(33.91);
    expect(countUpValue(33.91, 5000, 1400)).toBe(33.91);
  });
  it("is monotonic", () => {
    let prev = -1;
    for (let t = 0; t <= 1; t += 0.01) {
      const v = easeOutExpo(t);
      expect(v).toBeGreaterThanOrEqual(prev);
      prev = v;
    }
  });
  it("formats with fixed decimals and separators", () => {
    expect(formatNumber(987654.321, 2)).toBe("987,654.32");
    expect(formatNumber(0.5, 0)).toBe("1");
  });
});

describe("lightbox zoom/pan", () => {
  const vw = 800;
  const vh = 600;
  it("keeps the point under the cursor fixed", () => {
    const v = zoomAt(IDENTITY, 2, 200, 150, vw, vh);
    // content coordinate under (200,150) before: (200,150); after: (200 - x)/scale
    expect((200 - v.x) / v.scale).toBeCloseTo(200);
    expect((150 - v.y) / v.scale).toBeCloseTo(150);
  });
  it("clamps scale to [1, MAX_SCALE] and cannot pan past the edges", () => {
    expect(zoomAt(IDENTITY, 0.1, 0, 0, vw, vh)).toEqual(IDENTITY);
    expect(zoomAt(IDENTITY, 1e6, 0, 0, vw, vh).scale).toBe(MAX_SCALE);
    const v = clampPan({ scale: 2, x: 500, y: -5000 }, vw, vh);
    expect(v).toEqual({ scale: 2, x: 0, y: vh - vh * 2 });
  });
});

describe("site map", () => {
  it("has the six nav routes, unique", () => {
    expect(NAV_ROUTES.map((r) => r.path)).toEqual(["/", "/novelty/spectral", "/novelty/uncertainty", "/novelty/efficiency", "/compare", "/about"]);
  });
});
