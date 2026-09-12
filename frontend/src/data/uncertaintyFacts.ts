/**
 * Every number the uncertainty page quotes, transcribed from reports/mvp with
 * its source. tests/uncertaintyFacts.test.ts re-reads each file and fails on
 * drift. Anything the repo never computed (a binned reliability diagram, a
 * sigma^2 histogram) is deliberately absent: the page shows "not yet
 * benchmarked" instead.
 *
 * Units: uncertainty and error are surface reflectance (band-mean), unclipped.
 */
export interface UncMethodFacts {
  id: "tta4" | "tta8" | "learned";
  label: string;
  /** Area between this method's sparsification curve and the oracle's. Lower is better. */
  ause: number;
  spearman: number;
  /** Mean predicted uncertainty over the VAL subset (TTA: per-band std; learned: Laplace scale b). */
  meanUnc: number;
  /** Measured wall-clock cost relative to one plain forward pass. */
  costMultiplier: number;
  /** Empirical coverage of the nominal 50% / 90% intervals; null where the method has no distribution. */
  coverage: { p50: number; p90: number } | null;
  /** Mean |SR - HR| after removing the most-uncertain fraction `SPARSIFICATION.fractions[i]` of pixels. */
  curve: readonly number[];
  served: boolean;
}

/** Shared VAL subset of the three unc_eval reports. */
export const UNC_EVAL = {
  split: "val",
  n: 128,
  meanAbsErr: 0.009345882572233677,
  /** Removing pixels in true-error order: the best any uncertainty ranking can do. */
  oracle: [
    0.009346, 0.007508, 0.006588, 0.005895, 0.00533, 0.004849, 0.00443, 0.004057, 0.00372, 0.003411, 0.003124, 0.002856, 0.002601, 0.002358,
    0.002122, 0.001891, 0.00166, 0.001423, 0.00117, 0.000873,
  ],
  /** 0, 0.05, ..., 0.95. */
  fractions: Array.from({ length: 20 }, (_, i) => Math.round(i * 5) / 100),
} as const;

export const UNC_METHODS: readonly UncMethodFacts[] = [
  {
    id: "tta4",
    label: "TTA-4",
    ause: 0.001912491078729058,
    spearman: 0.5247549715284988,
    meanUnc: 0.0006104240892454982,
    costMultiplier: 3.1826722348324923,
    coverage: null,
    curve: [
      0.009346, 0.008387, 0.007884, 0.007481, 0.007131, 0.006817, 0.006525, 0.006246, 0.005974, 0.005705, 0.005437, 0.005173, 0.004916,
      0.004665, 0.00442, 0.004183, 0.00395, 0.003715, 0.003468, 0.003207,
    ],
    served: true,
  },
  {
    id: "tta8",
    label: "TTA-8",
    ause: 0.001817246639501023,
    spearman: 0.5423443922481764,
    meanUnc: 0.0006997198215685785,
    costMultiplier: 6.985766931286304,
    coverage: null,
    curve: [
      0.009346, 0.008353, 0.007836, 0.007421, 0.007064, 0.006743, 0.006445, 0.006162, 0.005883, 0.005605, 0.005328, 0.005053, 0.004784,
      0.004527, 0.004281, 0.004049, 0.003819, 0.003578, 0.003318, 0.003052,
    ],
    served: false,
  },
  {
    id: "learned",
    label: "Learned NLL head",
    ause: 0.002709924841899815,
    spearman: 0.37860800094092956,
    meanUnc: 0.01042428333312273,
    costMultiplier: 0.922361504080862,
    coverage: { p50: 0.5326952338218689, p90: 0.9039052724838257 },
    curve: [
      0.009346, 0.008426, 0.00801, 0.00768, 0.007395, 0.00718, 0.00698, 0.0068, 0.006624, 0.006468, 0.006319, 0.00617, 0.006027, 0.005883,
      0.005742, 0.005597, 0.005442, 0.005277, 0.005076, 0.005066,
    ],
    served: false,
  },
];

/** Run C collapse watch: the learned head re-checked every 250 iterations on a held-out VAL subset. */
export const COLLAPSE_WATCH = {
  run: "c1",
  itersCompleted: 2000,
  elapsedMin: 74.50435771500003,
  verdict: "passed",
  nPixels: 200000,
  thresholds: { rhoMin: 0.1, fracFloorMax: 0.2, spatialCvMin: 0.05, floor: 0.0002 },
  iters: [250, 500, 750, 1000, 1250, 1500, 1750, 2000],
  rho: [0.2128, 0.2625, 0.2914, 0.3179, 0.3392, 0.3579, 0.3628, 0.3671],
  spatialCv: [0.1816, 0.2831, 0.3569, 0.4056, 0.4377, 0.4376, 0.4578, 0.4594],
  fracFloor: [0, 0, 0, 0, 0, 0, 0, 0],
} as const;

/** The display scale of the served overlay, and an edge-detector baseline on the same pixels. */
export const TRUST_SCALES = {
  uncDisplayMax: 0.003713641131762385,
  spearmanTta8: 0.5509238248550024,
  spearmanSobelBicubic: 0.5240676687703899,
  nPatches: 64,
  nPixels: 200000,
} as const;

export const UNC_SOURCES = {
  eval: "reports/mvp/unc_eval_{tta4,tta8}_a2-last-dce224ec.json, reports/mvp/unc_eval_learned_unc_c1-head_last-19c2bd0a.json",
  watch: "reports/mvp/unc_watch_c1.jsonl, reports/mvp/unc_train_c1.json",
  trust: "reports/mvp/trust_scales.json",
  render: "src/infer/render.py overlay_rgba (magma LUT; alpha 0 below 5% of vmax, 0.85 at vmax)",
} as const;

/** src/infer/render.py LUTS["magma"] sampled at 0, 32, ..., 255: the served overlay's colours. */
export const MAGMA_STOPS = ["#000004", "#1d1147", "#51127c", "#832681", "#b73779", "#e75263", "#fc8961", "#fec488", "#fcfdbf"] as const;
