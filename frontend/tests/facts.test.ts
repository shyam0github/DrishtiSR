import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { toImageResult, type UpscaleResponse } from "../src/api/client";
import { FACTS } from "../src/data/facts";
import { REPO_ROOT } from "../scripts/repo-config.mjs";

const json = (rel: string) => JSON.parse(readFileSync(path.join(REPO_ROOT, rel), "utf8"));

describe("Home page facts match their reports", () => {
  it("parameter counts", () => {
    const table = readFileSync(path.join(REPO_ROOT, "reports/mvp/deploy_table.md"), "utf8");
    expect(table).toMatch(new RegExp(`\\| onnx-fp32 \\| ${FACTS.params.toLocaleString("en-US")} \\|`));
    const fixture = json("frontend/src/api/fixtures/metrics.json");
    const runA = [...fixture.rows, ...fixture.supplementary_rows].find((r: { label: string }) => r.label === "runA");
    expect(Number(runA.params.replace(/,/g, ""))).toBe(FACTS.runAParams);
  });

  it("CPU benchmark and INT8 gate", () => {
    const bench = json("reports/mvp/bench_a2-last-dce224ec.json");
    const fp32 = bench.backends.find((b: { backend: string }) => b.backend === "onnx-fp32");
    expect(fp32.model_bytes).toBe(FACTS.onnxFp32Bytes);
    expect(fp32.median_ms).toBe(FACTS.onnxFp32MedianMs);
    expect(bench.threads).toBe(FACTS.benchThreads);
    expect(bench.runs).toBe(FACTS.benchRuns);
    expect(bench.cpu_model).toBe(FACTS.benchCpu);
    expect(bench.provisional).toBe(false);
    expect(bench.input).toContain(`${FACTS.benchLrPx}x${FACTS.benchLrPx}`);
    const gate = json("reports/mvp/quant_gate.json")["a2-last-dce224ec"];
    expect(gate.passed).toBe(FACTS.int8GatePassed);
    expect(bench.backends.find((b: { backend: string }) => b.backend === "onnx-int8").model_bytes).toBe(FACTS.int8Bytes);
  });

  it("headline quality and consistency", () => {
    const h = json("reports/mvp/headline.json");
    expect(h.lpips.winner).toBe(FACTS.lpipsServed);
    expect(h.lpips.bicubic).toBe(FACTS.lpipsBicubic);
    expect(h.lpips.relative_change_vs_bicubic).toBe(FACTS.lpipsRelChangeVsBicubic);
    expect(h.lpips.provenance.n_samples).toBe(FACTS.valPairs);
    const c = h.spectral_consistency_opensr;
    expect(c.winner).toBe(FACTS.consistencyServed);
    expect(c.hr_reference).toBe(FACTS.consistencyHrReference);
    expect(c.winner_vs_hr_reference_ratio).toBe(FACTS.consistencyVsHrReference);
  });

  it("served uncertainty (TTA-4)", () => {
    const u = json("reports/mvp/unc_eval_tta4_a2-last-dce224ec.json");
    expect(u.spearman_rho).toBe(FACTS.uncSpearman);
    expect(u.ause).toBe(FACTS.uncAuse);
    expect(u.n_samples).toBe(FACTS.uncN);
    expect(u.split).toBe("val");
  });
});

describe("toImageResult", () => {
  const base: UpscaleResponse = {
    job_id: "j1",
    input: { source: "upload", sample_id: null, lr_size: [64, 64], sr_size: [256, 256], georeferenced: true, dn_mode_applied: "dn10000" },
    model: { backend: "onnx-fp32", checkpoint_id: "c", params: 855652, model_bytes: 1, threads: 6, interim: false, has_scale_head: false },
    uncertainty_method: "tta4",
    images: {
      lr_rgb: "/files/j1/lr_rgb.png", lr_fcc: "", bicubic_rgb: "/files/j1/bic.png", bicubic_fcc: "", sr_rgb: "/files/j1/sr_rgb.png", sr_fcc: "",
      hr_rgb: null, hr_fcc: null, uncertainty: "/files/j1/unc.png", consistency: "/files/j1/cons.png",
    },
    downloads: { sr_tif: "/files/j1/sr.tif", uncertainty_tif: null },
    metrics: {
      reference_free: {
        spec_l1: 0.003, spec_sam_deg: 1, spec_l1_bicubic: 0.002, spec_sam_bicubic_deg: 1, hf_ratio_vs_bicubic: 1.3,
        unc_mean: null, unc_p95: null, runtime_ms: { sr: 300, uncertainty: 900, total: 1300 },
      },
      with_gt: null,
    },
    refs: { spec_l1_gt_floor: 0.005733, spec_sam_gt_floor_deg: 1.2748, unc_display_max: 1, cons_display_max: 1, sharpness_warn_below: 1.05 },
    warnings: [],
  };

  it("leaves reference metrics null (pending) for an upload without ground truth", () => {
    const r = toImageResult(base, "a.tif");
    expect(r.hasGroundTruth).toBe(false);
    expect([r.metrics.psnr, r.metrics.ssim, r.metrics.lpips, r.metrics.bicubic]).toEqual([null, null, null, null]);
    expect(r.metrics.specL1).toBe(0.003);
    expect(r.images.sr).toBe("/files/j1/sr_rgb.png");
    expect(r.images.hr).toBeNull();
  });

  it("lifts SR and bicubic metrics when ground truth exists", () => {
    const q = (v: number) => ({ lpips: v, ssim: v, psnr: v, sam_deg: v, ergas: v });
    const withGt = { ...base, metrics: { ...base.metrics, with_gt: { sr: q(0.3), bicubic: q(0.4), spec_l1_hr: 0, spec_sam_hr_deg: 0, hf_ratio_hr_vs_bicubic: 1 } } };
    const r = toImageResult(withGt, "sample");
    expect(r.metrics.lpips).toBe(0.3);
    expect(r.metrics.bicubic?.lpips).toBe(0.4);
  });
});
