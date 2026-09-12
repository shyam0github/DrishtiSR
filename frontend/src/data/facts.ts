/**
 * Every repository-wide number the Home page quotes, transcribed with its
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
} as const;

/** Repo-relative source of each fact, checked by tests/facts.test.ts. */
export const FACT_SOURCES = {
  params: "reports/mvp/deploy_table.md",
  runAParams: "src/api/fixtures/metrics.json (snapshot of reports/day3_results.json)",
  bench: "reports/mvp/bench_a2-last-dce224ec.json",
  int8: "reports/mvp/quant_gate.json",
  headline: "reports/mvp/headline.json",
  uncertainty: "reports/mvp/unc_eval_tta4_a2-last-dce224ec.json",
} as const;
