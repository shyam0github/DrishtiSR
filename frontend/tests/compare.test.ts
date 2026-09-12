import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { COMPARE_SOURCES, isComplete, MEASURED_SPLIT, METRICS, MODELS } from "../src/data/compare";
import { logDomain, MIN_RADIUS, radarRadius } from "../src/design/radar";
import { REPO_ROOT } from "../scripts/repo-config.mjs";

const read = (rel: string) => readFileSync(path.join(REPO_ROOT, rel), "utf8");

describe("/compare numbers match their sources", () => {
  it("measured rows equal the opensr-test snapshot", () => {
    const snap = JSON.parse(read(COMPARE_SOURCES.measured));
    expect(snap.smoke).toBe(false);
    expect(snap.split).toBe(MEASURED_SPLIT.split);
    expect([...snap.metrics].sort()).toEqual(METRICS.map((d) => d.key).sort());
    const measured = MODELS.filter((m) => m.source === "measured");
    expect(measured).toHaveLength(snap.rows.length);
    for (const m of measured) {
      const row = snap.rows.find((r: { label: string }) => r.label === m.snapshot!.label);
      expect(row, m.id).toBeDefined();
      expect(row.checkpoint).toBe(m.snapshot!.checkpoint);
      expect(row.n_pairs).toBe(MEASURED_SPLIT.pairs);
      for (const d of METRICS) expect(m.values[d.key], `${m.id}.${d.key}`).toBeCloseTo(row.means[d.key], 12);
    }
  });

  it("our row is the checkpoint the headline serves", () => {
    const h = JSON.parse(read("reports/mvp/headline.json"));
    const ours = MODELS.find((m) => m.id === "drishtisr")!;
    expect(ours.values.reflectance).toBeCloseTo(h.spectral_consistency_opensr.winner, 8);
  });

  it("reported rows are quoted verbatim from the README table", () => {
    const md = read(COMPARE_SOURCES.reported);
    for (const m of MODELS.filter((r) => r.readmeModel)) {
      const line = md.split("\n").find((l) => l.startsWith(`| ${m.readmeModel} `));
      expect(line, m.id).toBeDefined();
      const cells = line!.split("|").slice(2, 9).map((c) => c.trim().split(" ")[0]);
      expect(cells).toEqual(METRICS.map((d) => m.values[d.key]!.toFixed(4)));
    }
  });

  it("literature rows without a source show N/A with a reason, never a number", () => {
    for (const m of MODELS.filter((r) => r.source === "reported" && !r.readmeModel)) {
      expect(METRICS.every((d) => m.values[d.key] === null)).toBe(true);
      expect(m.naReason).toBeTruthy();
      expect(isComplete(m)).toBe(false);
    }
  });
});

describe("radar scale", () => {
  it("puts the best value outermost in both directions", () => {
    const dom = logDomain([0.001, 0.01, 0.1, null]);
    expect(radarRadius(0.001, dom, "lower")).toBe(1);
    expect(radarRadius(0.1, dom, "lower")).toBe(MIN_RADIUS);
    expect(radarRadius(0.1, dom, "higher")).toBe(1);
    expect(radarRadius(0.01, dom, "higher")).toBeCloseTo(MIN_RADIUS + (1 - MIN_RADIUS) / 2, 12);
  });

  it("refuses values it cannot log-scale", () => {
    expect(() => logDomain([null])).toThrow();
    expect(() => logDomain([0, 1])).toThrow();
  });
});
