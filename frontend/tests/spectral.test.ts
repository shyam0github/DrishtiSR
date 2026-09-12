import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import BENCH from "../src/data/spectralBenchmark.json";
import { REPO_ROOT } from "../scripts/repo-config.mjs";

/** The spectral page's numbers must be the report's, byte for byte. Re-run `npm run fixtures:spectral` after the report changes. */
const report = JSON.parse(readFileSync(path.join(REPO_ROOT, BENCH.source), "utf8"));

describe("spectralBenchmark.json matches reports/day3_results.json", () => {
  it("is the same report run", () => {
    expect(report.smoke).toBe(false);
    expect(BENCH.fingerprint).toBe(report.fingerprint);
    expect(BENCH.written_utc).toBe(report.written_utc);
    expect(BENCH.n_pairs).toBe(report.n_pairs);
    expect(BENCH.headline).toBe(report.reading[0]);
    expect(BENCH.cost).toBe(report.reading[1]);
  });

  it("means and floor", () => {
    const rows = [...report.rows, ...report.supplementary_rows];
    for (const m of BENCH.methods) {
      const row = rows.find((r: { label: string }) => r.label === m.label);
      expect(row.checkpoint).toBe(m.checkpoint);
      for (const [k, v] of Object.entries(m.means)) expect(v).toBe(row.means[k]);
      // Quantiles come from the per-pair tables; they must bracket the report's mean sensibly.
      for (const d of Object.values(m.distribution)) expect(d.min <= d.q1 && d.q1 <= d.median && d.median <= d.q3 && d.q3 <= d.max).toBe(true);
    }
    for (const [k, v] of Object.entries(BENCH.floor)) expect(v).toBe(report.floor[k]);
  });

  it("significance: CIs and p-values copied, not recomputed", () => {
    for (const s of BENCH.significance) {
      const r = report.significance.find((x: { metric: string }) => x.metric === s.metric);
      for (const [k, v] of Object.entries(s)) expect(v).toEqual(r[k]);
    }
  });

  it("headline -33.3% is the reflectance change of B1 vs A2", () => {
    const s = BENCH.significance.find((x) => x.metric === "reflectance")!;
    expect(((s.mean_delta / s.mean_control) * 100).toFixed(1)).toBe("-33.3");
    expect(BENCH.headline).toContain("-33.3%");
  });
});
