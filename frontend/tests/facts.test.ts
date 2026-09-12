import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { toImageResult, type UpscaleResponse } from "../src/api/client";
import { parse } from "yaml";
import { ABOUT_FACTS, FACTS } from "../src/data/facts";
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

  it("efficiency page: INT8 timing, 1-thread benchmark, parity", () => {
    const bench = json("reports/mvp/bench_a2-last-dce224ec.json");
    expect(bench.backends.find((b: { backend: string }) => b.backend === "onnx-int8").median_ms).toBe(FACTS.int8MedianMs);
    const t1 = json("reports/mvp/bench_a2-last-dce224ec_t1.json");
    expect(t1.threads).toBe(1);
    expect(t1.runs).toBe(FACTS.benchRuns);
    expect(t1.cpu_model).toBe(FACTS.benchCpu);
    expect(t1.input).toBe(bench.input);
    expect(t1.provisional).toBe(FACTS.t1Provisional);
    const med = (b: string) => t1.backends.find((r: { backend: string }) => r.backend === b).median_ms;
    expect(med("onnx-fp32")).toBe(FACTS.t1OnnxFp32MedianMs);
    expect(med("onnx-int8")).toBe(FACTS.t1Int8MedianMs);
    const table = readFileSync(path.join(REPO_ROOT, "reports/mvp/deploy_table.md"), "utf8");
    expect(table).toContain(`parity max abs diff ${FACTS.onnxParityMaxAbsDiff}`);
  });

  it("efficiency page: every INT8 quantisation attempt", () => {
    const gate = json("reports/mvp/quant_gate.json")["a2-last-dce224ec"];
    expect(gate.gate).toEqual({ d_psnr_db: FACTS.quantGate.dPsnrDb, d_sam_deg: FACTS.quantGate.dSamDeg, d_lpips: FACTS.quantGate.dLpips });
    expect(gate.n_samples).toBe(FACTS.quantGateN);
    expect(gate.calibration.n).toBe(FACTS.quantCalibN);
    expect(gate.attempts.map((a: { attempt: string }) => a.attempt)).toEqual(FACTS.quantAttempts.map((a) => a.attempt));
    for (const [i, a] of gate.attempts.entries()) {
      const f = FACTS.quantAttempts[i];
      expect([a.bytes, a.d_psnr_db, a.d_sam_deg, a.d_lpips, a.passed]).toEqual([f.bytes, f.dPsnrDb, f.dSamDeg, f.dLpips, false]);
      expect(a.calibration).toBe(f.calibration.split(" ")[0]);
      expect(a.exclusion === null).toBe(f.keptFp32 === null);
    }
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

describe("About page facts match their sources", () => {
  const text = (rel: string) => readFileSync(path.join(REPO_ROOT, rel), "utf8");
  const yaml = (rel: string) => parse(text(rel));
  const A = ABOUT_FACTS;

  it("training environment", () => {
    const m = json("runs/day3/a2/run_metadata.json");
    expect([m.cuda, m.python, m.torch, m.args.iters]).toEqual([A.trainGpu, A.trainPython, A.trainTorch, A.a2Iters]);
    expect([m.args.n_resblocks, m.args.n_feats, m.args.batch, m.args.patch_lr]).toEqual([A.nResblocks, A.nFeats, A.batch, A.patchLr]);
    const f = yaml("configs/frozen_day3.yaml");
    expect(f.schedule.iters).toBe(A.scheduleIters);
    expect(f.model.expected_parameters).toBe(FACTS.params);
    expect(f.augment.photometric).toBe(false);
    const req = text("requirements.txt");
    expect(req).toContain(`omegaconf==${A.omegaconf}`);
    expect(req).toContain(`tacoreader==${A.tacoreader}`);
    expect(req).toContain(`opensr-test==${A.opensrTest}`);
    expect(req).toContain(`onnxruntime==${A.onnxruntime}`);
    expect(text("AGENTS.md")).toMatch(new RegExp(`${A.gpuHoursPerWeek} GPU-hours per\\s+week`));
  });

  it("data and splits", () => {
    const dv = json("reports/day3_data_validity.json");
    expect([dv.subset, dv.n_cached, dv.n_pairs]).toEqual([A.subset, A.cachedPairs, A.dvPairs]);
    for (const [band, n] of Object.entries(A.dvMeanEqual4dp)) expect(dv.per_band[band].pairs_with_mean_equal_4dp).toBe(n);
    expect(dv.histmatch_signature.minmax_equal_all_bands).toBe(A.dvMinMaxEqualAllBands);
    expect(dv.r_allbands_range).toEqual([A.dvRAllBandsMin, A.dvRAllBandsMax]);
    expect(text("PROJECT_STATE.md")).toContain(`Usable cached train = ${A.usableTrainPairs.toLocaleString("en-US")}`);
    const gate = text("reports/day1_gate.md");
    for (const [split, n] of Object.entries(A.splitTiles)) expect(gate).toMatch(new RegExp(`\\| ${split} \\| ${n} \\|`));
    expect(gate).toContain(`**${FACTS.valPairs} patches**`);
  });

  it("inference environment", () => {
    const b = json("reports/mvp/bench_a2-last-dce224ec.json");
    expect([b.python, b.torch, b.onnxruntime, b.logical_cpus]).toEqual([A.inferPython, A.inferTorch, A.onnxruntime, A.logicalCpus]);
    expect(b.backends.every((x: { peak_rss_bytes: unknown }) => x.peak_rss_bytes === null)).toBe(!A.peakRssMeasured);
    expect(b.backends.find((x: { backend: string }) => x.backend === "onnx-int8").median_ms).toBe(FACTS.int8MedianMs);
    const g = json("reports/mvp/quant_gate.json")["a2-last-dce224ec"].gate;
    expect([g.d_psnr_db, g.d_sam_deg, g.d_lpips]).toEqual([FACTS.quantGate.dPsnrDb, FACTS.quantGate.dSamDeg, FACTS.quantGate.dLpips]);
    expect(json("reports/mvp/unc_eval_learned_unc_c1-head_last-19c2bd0a.json").ause).toBe(FACTS.uncLearnedAuse);
    expect(json("reports/mvp/unc_eval_tta8_a2-last-dce224ec.json").ause).toBe(FACTS.uncTta8Ause);
  });

  it("evaluation method and limitations", () => {
    const sig = yaml("configs/base.yaml").eval_all_ckpts.significance;
    expect([sig.n_boot, sig.ci]).toEqual([A.nBoot, A.ci]);
    const c = json("reports/day2_runA.json").curve;
    expect([c.best_iter, c.best_psnr, c.final_iter, c.final_psnr]).toEqual([A.runABestIter, A.runABestPsnr, A.runAFinalIter, A.runAFinalPsnr]);
    const floor = text("reports/day3_spectral_floor.md");
    expect(floor).toContain(`**${A.spectralFloorL1}**`);
    expect(floor).toContain(`scores ${A.bicubicVsFloor}× the`);
    const r = text("reports/day3_results.md");
    expect(r).toContain(`from ${A.a2ConsistencyL1.toFixed(5)} to ${A.b1ConsistencyL1}`);
    expect(r).toContain(`B costs ${A.b1PsnrCostDb} dB PSNR`);
    expect(r).toContain(`LPIPS by +${A.b1LpipsCost.toFixed(4)}`);
    expect(r).toContain("Wilcoxon p");
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
