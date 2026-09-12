import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { COLLAPSE_WATCH, TRUST_SCALES, UNC_EVAL, UNC_METHODS } from "../src/data/uncertaintyFacts";
import { REPO_ROOT } from "../scripts/repo-config.mjs";

const read = (rel: string) => readFileSync(path.join(REPO_ROOT, rel), "utf8");
const json = (rel: string) => JSON.parse(read(rel));

const EVAL_FILES = {
  tta4: "reports/mvp/unc_eval_tta4_a2-last-dce224ec.json",
  tta8: "reports/mvp/unc_eval_tta8_a2-last-dce224ec.json",
  learned: "reports/mvp/unc_eval_learned_unc_c1-head_last-19c2bd0a.json",
} as const;

const closeAll = (got: readonly number[], want: number[], digits: number) => {
  expect(got.length).toBe(want.length);
  got.forEach((v, i) => expect(v).toBeCloseTo(want[i]!, digits));
};

describe("uncertainty page facts match their reports", () => {
  it("per-method calibration (unc_eval_*.json)", () => {
    for (const m of UNC_METHODS) {
      const r = json(EVAL_FILES[m.id]);
      expect([r.split, r.n_samples]).toEqual([UNC_EVAL.split, UNC_EVAL.n]);
      expect([r.ause, r.spearman_rho, r.mean_unc, r.forward_pass_cost.measured_multiplier]).toEqual([m.ause, m.spearman, m.meanUnc, m.costMultiplier]);
      expect(r.mean_abs_err).toBe(UNC_EVAL.meanAbsErr);
      closeAll(UNC_EVAL.fractions, r.fractions, 9);
      closeAll(m.curve, r.curve_unc, 5);
      closeAll(UNC_EVAL.oracle, r.curve_oracle, 5);
      if (m.coverage) expect([r.coverage["50"].empirical, r.coverage["90"].empirical]).toEqual([m.coverage.p50, m.coverage.p90]);
      else expect(r.coverage).toBe("N/A");
    }
  });

  it("collapse watch (unc_watch_c1.jsonl, unc_train_c1.json)", () => {
    const rows = read("reports/mvp/unc_watch_c1.jsonl").trim().split("\n").map((l) => JSON.parse(l));
    expect(rows.map((r) => r.iter)).toEqual([...COLLAPSE_WATCH.iters]);
    closeAll(COLLAPSE_WATCH.rho, rows.map((r) => r.rho), 4);
    closeAll(COLLAPSE_WATCH.spatialCv, rows.map((r) => r.spatial_cv), 4);
    expect(rows.map((r) => r.frac_floor)).toEqual([...COLLAPSE_WATCH.fracFloor]);
    const t = rows[0].thresholds;
    const w = COLLAPSE_WATCH.thresholds;
    expect([t.rho_min, t.frac_floor_max, t.spatial_cv_min, t.floor]).toEqual([w.rhoMin, w.fracFloorMax, w.spatialCvMin, w.floor]);
    expect(rows.at(-1).verdict).toBe(COLLAPSE_WATCH.verdict);
    expect(rows[0].n_pixels).toBe(COLLAPSE_WATCH.nPixels);
    const train = json("reports/mvp/unc_train_c1.json");
    expect([train.iters_completed, train.elapsed_min, train.watch_verdict]).toEqual([COLLAPSE_WATCH.itersCompleted, COLLAPSE_WATCH.elapsedMin, COLLAPSE_WATCH.verdict]);
  });

  it("display scale and edge baseline (trust_scales.json)", () => {
    const t = json("reports/mvp/trust_scales.json");
    expect([t.unc_display_max, t.spearman_tta8_unc_vs_abs_err, t.spearman_sobel_bicubic_vs_abs_err]).toEqual([
      TRUST_SCALES.uncDisplayMax,
      TRUST_SCALES.spearmanTta8,
      TRUST_SCALES.spearmanSobelBicubic,
    ]);
    expect([t.n, t.spearman_n_pixels]).toEqual([TRUST_SCALES.nPatches, TRUST_SCALES.nPixels]);
  });
});
