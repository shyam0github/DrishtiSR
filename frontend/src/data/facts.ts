/**
 * Every repository-wide number the Home and About pages quote, transcribed with its
 * source. tests/facts.test.ts re-reads each source file and fails on any
 * drift, so a number here cannot quietly disagree with the report it came
 * from. Numbers the repo does not have yet are not listed: the page shows a
 * Pending badge instead.
 */
export const FACTS = {
  /** Served model (A2, 16 blocks x 48 features). */
  params: 855_652,
  /** Run A, EDSR 16 x 64: the in-repo reference for "a typical EDSR". */
  runAParams: 1_518_724,

  /** onnx-fp32, the served backend: 256 x 256 LR -> 1024 x 1024 SR, 6 threads. */
  onnxFp32Bytes: 3_441_761,
  onnxFp32MedianMs: 1220.0843999999051,
  benchThreads: 6,
  benchLrPx: 256,
  benchRuns: 20,
  benchCpu: "Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz",
  int8Bytes: 981_131,
  int8GatePassed: false,
  /** Attempt C, the INT8 graph closest to the gate: timed for information, not served. */
  int8MedianMs: 781.9019500002469,
  /** onnx-fp32 vs torch-fp32, max abs difference in reflectance (deploy_table.md notes). */
  onnxParityMaxAbsDiff: "4.5e-07",

  /** Same benchmark at 1 intra-op thread (bench_*_t1.json, 2026-09-12, load 16% so not provisional). */
  t1OnnxFp32MedianMs: 2369.0246999994997,
  t1Int8MedianMs: 1603.7512499988225,
  t1Provisional: false,

  /** INT8 accuracy gate: degradations INT8 vs FP32 (positive = INT8 worse), must all be <= the limit. */
  quantGate: { dPsnrDb: 0.1, dSamDeg: 0.05, dLpips: 0.005 },
  quantGateN: 64,
  quantCalibN: 128,
  quantAttempts: [
    { attempt: "A", calibration: "MinMax", keptFp32: null, bytes: 973_708, dPsnrDb: 0.5922442764232443, dSamDeg: 0.32278018122080265, dLpips: -0.026800789521075785 },
    { attempt: "B", calibration: "Percentile 99.99", keptFp32: null, bytes: 973_732, dPsnrDb: 0.4804290597400893, dSamDeg: 0.15911356462980164, dLpips: -0.007758345629554242 },
    { attempt: "C", calibration: "Percentile 99.99", keptFp32: "first and output conv", bytes: 981_131, dPsnrDb: 0.32906808175675906, dSamDeg: 0.06616762393422126, dLpips: -0.005414303625002503 },
    { attempt: "D", calibration: "Percentile 99.99", keptFp32: "upsampler", bytes: 1_463_325, dPsnrDb: 0.43885113724944347, dSamDeg: 0.1505136693505289, dLpips: -0.006967683613765985 },
  ],

  /** Validation split, 1,199 pairs, served checkpoint vs bicubic. */
  valPairs: 1199,
  lpipsServed: 0.332209,
  lpipsBicubic: 0.403318,
  lpipsRelChangeVsBicubic: -0.17631000847966127,
  /** opensr-test reflectance L1 (LR vs SR degraded to 10 m); the HR reference is the true 2.5 m image scored the same way. */
  consistencyServed: 0.00319858,
  consistencyHrReference: 0.004782443621845941,
  consistencyVsHrReference: 0.6688170845107428,

  /** Served uncertainty: TTA-4 on VAL. */
  uncSpearman: 0.5247549715284988,
  uncAuse: 0.001912491078729058,
  uncN: 128,
  /** Learned heteroscedastic head (Run C) and TTA-8, same VAL subset: why the head is opt-in. */
  uncLearnedAuse: 0.002709924841899815,
  uncTta8Ause: 0.001817246639501023,
} as const;

/**
 * The About page's spec sheet, one entry per line of it, each with a source
 * checked by tests/facts.test.ts. Nothing here is estimated: a value the repo
 * never measured (peak RAM) is recorded as not measured.
 */
export const ABOUT_FACTS = {
  /** Training: runs/day3/a2/run_metadata.json (the served model's run). */
  trainGpu: "Tesla T4",
  trainPython: "3.12.13",
  trainTorch: "2.10.0+cu128",
  gpuHoursPerWeek: 30,
  a2Iters: 12_000,
  scheduleIters: 40_000,
  nResblocks: 16,
  nFeats: 48,
  batch: 16,
  patchLr: 64,
  omegaconf: "2.3.0",
  tacoreader: "2.1.0",
  subset: "sen2naipv2-crosssensor",
  cachedPairs: 3261,
  usableTrainPairs: 2659,
  splitTiles: { train: 2400, val: 300, test: 300 },

  /** Inference: reports/mvp/bench_a2-last-dce224ec.json. */
  inferPython: "3.11.9",
  inferTorch: "2.5.1+cpu",
  onnxruntime: "1.20.1",
  logicalCpus: 12,
  peakRssMeasured: false,

  /** Evaluation. */
  opensrTest: "1.3.3",
  nBoot: 10_000,
  ci: 0.95,

  /** Run A's iteration curve: reports/day2_runA.json. */
  runABestIter: 8000,
  runABestPsnr: 35.2842,
  runAFinalIter: 40_000,
  runAFinalPsnr: 35.0328,

  /** Spectral term: reports/day3_spectral_floor.md, reports/day3_results.md. */
  spectralFloorL1: 0.005733,
  bicubicVsFloor: 0.21,
  a2ConsistencyL1: 0.0032,
  b1ConsistencyL1: 0.00213,
  b1PsnrCostDb: 0.105,
  b1LpipsCost: 0.013,

  /** Data validity: reports/day3_data_validity.json. */
  dvPairs: 50,
  dvMeanEqual4dp: { B04: 23, B03: 18, B02: 21, B08: 25 },
  dvMinMaxEqualAllBands: 32,
  dvRAllBandsMin: 0.9806474820702712,
  dvRAllBandsMax: 0.9990410610463446,
} as const;

/** Repo-relative source of each fact, checked by tests/facts.test.ts. */
export const FACT_SOURCES = {
  params: "reports/mvp/deploy_table.md",
  runAParams: "src/api/fixtures/metrics.json (snapshot of reports/day3_results.json)",
  bench: "reports/mvp/bench_a2-last-dce224ec.json",
  bench1t: "reports/mvp/bench_a2-last-dce224ec_t1.json",
  int8: "reports/mvp/quant_gate.json",
  headline: "reports/mvp/headline.json",
  uncertainty: "reports/mvp/unc_eval_tta4_a2-last-dce224ec.json",
  training: "runs/day3/a2/run_metadata.json, configs/frozen_day3.yaml, requirements.txt, AGENTS.md",
  data: "reports/day1_gate.md, reports/day3_data_validity.json, PROJECT_STATE.md §6 C2",
  evaluation: "configs/base.yaml eval_all_ckpts.significance, reports/day3_results.md",
  runACurve: "reports/day2_runA.json",
  spectralFloor: "reports/day3_spectral_floor.md",
} as const;
