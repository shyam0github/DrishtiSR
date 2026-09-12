/**
 * The /compare page's data: the seven opensr-test metrics, and one row per
 * model with its provenance.
 *
 * Measured rows are transcribed from reports/compare/opensr_seven.json
 * (scripts/snapshot_opensr_seven.py: SEN2NAIPv2 val, 1,199 pairs, each run's
 * pre-registered checkpoint, opensr-test 1.3.3 on CPU). Reported rows are
 * quoted from the opensr-test README ND table, kept verbatim in
 * reports/compare/opensr_test_readme_nd.md. tests/compare.test.ts checks every
 * number against those two files.
 *
 * Parameter counts are not repeated here: the page reads them from
 * SR_MODELS (src/data/srModels.ts), the site's one list, by model name.
 *
 * A cell is null when no comparable number exists. The page shows it as
 * "N/A — not directly comparable" and never estimates it.
 */

export type MetricKey = "reflectance" | "spectral" | "spatial" | "synthesis" | "ha_metric" | "om_metric" | "im_metric";
export type Better = "higher" | "lower";

export interface MetricDef {
  key: MetricKey;
  name: string;
  /** Radar axis label. */
  short: string;
  group: "Consistency" | "Synthesis" | "Correctness";
  better: Better;
  unit: string;
  decimals: number;
  measures: string;
  whySatellite: string;
}

export const METRICS: readonly MetricDef[] = [
  {
    key: "reflectance",
    name: "Reflectance consistency",
    short: "Reflectance",
    group: "Consistency",
    better: "lower",
    unit: "reflectance L1",
    decimals: 5,
    measures: "Degrade the output back to 10 m and compare it with the input. This is the average brightness difference, in surface-reflectance units.",
    whySatellite:
      "Reflectance is a physical measurement that NDVI, water and soil indices are computed from. A small brightness drift biases every one of those indices, even if the picture still looks right.",
  },
  {
    key: "spectral",
    name: "Spectral angle",
    short: "Spectral",
    group: "Consistency",
    better: "lower",
    unit: "degrees",
    decimals: 3,
    measures: "How much each pixel's spectral signature (the ratios between its R, G, B and NIR values) rotates compared with the input.",
    whySatellite:
      "Land-cover classification reads band ratios, not colours on a screen. Photo SR benchmarks never check this, so a model can look sharp while shifting crops towards bare soil.",
  },
  {
    key: "spatial",
    name: "Spatial alignment",
    short: "Spatial",
    group: "Consistency",
    better: "lower",
    unit: "LR pixels",
    decimals: 4,
    measures: "Geometric shift between the degraded output and the input, estimated by phase correlation.",
    whySatellite:
      "The output is overlaid on maps, parcel boundaries and earlier acquisitions. Even a sub-pixel shift misregisters every change-detection or overlay analysis.",
  },
  {
    key: "synthesis",
    name: "Detail synthesis",
    short: "Synthesis",
    group: "Synthesis",
    better: "higher",
    unit: "reflectance L1",
    decimals: 5,
    measures: "How much high-frequency detail the model adds beyond a smooth upsample of the input.",
    whySatellite:
      "That detail is the reason to super-resolve at all: field edges, roads and building outlines at 2.5 m. The metric counts any added detail, right or wrong, so always read it with hallucination.",
  },
  {
    key: "ha_metric",
    name: "Hallucination",
    short: "Hallucination",
    group: "Correctness",
    better: "lower",
    unit: "0–1",
    decimals: 4,
    measures: "Roughly, the share of pixels whose added detail is in neither the 10 m input nor the true 2.5 m image: invented content.",
    whySatellite:
      "A plausible fake is harmless in a holiday photo. On a map, an invented road or roof can mislead a land record or a disaster response, which is why DrishtiSR ships an uncertainty map with every output.",
  },
  {
    key: "om_metric",
    name: "Omission",
    short: "Omission",
    group: "Correctness",
    better: "lower",
    unit: "0–1",
    decimals: 4,
    measures: "Roughly, the share of pixels where real 2.5 m detail was not recovered and the output stays as blurry as the input.",
    whySatellite:
      "Small features such as narrow roads, footpaths and small structures are the ones analysts want from 2.5 m. Conservative models score poorly here.",
  },
  {
    key: "im_metric",
    name: "Improvement",
    short: "Improvement",
    group: "Correctness",
    better: "higher",
    unit: "0–1",
    decimals: 4,
    measures: "Roughly, the share of pixels where the output moves correctly towards the true 2.5 m image.",
    whySatellite:
      "This is correct detail gained. Read next to hallucination, it separates genuine recovery from invention, which PSNR cannot do.",
  },
];

export type MetricValues = Record<MetricKey, number | null>;
export type Source = "measured" | "reported";

export interface ModelRow {
  id: string;
  name: string;
  /** Role label shown as a badge, e.g. "Baseline Architecture". */
  role?: string;
  roleVariant?: "ours" | "baseline";
  category: string;
  source: Source;
  /** Provenance, shown in the Sources tab and as the row tooltip. */
  detail: string;
  reference?: { text: string; href: string };
  /** measured rows: label + checkpoint in reports/compare/opensr_seven.json. */
  snapshot?: { label: string; checkpoint: string };
  /** reported rows: the model's name in the README ND table. */
  readmeModel?: string;
  /** Why every cell is null, when it is. */
  naReason?: string;
  /** A CSS color token; categorical order is fixed per model, never by rank. */
  color: string;
  defaultOn: boolean;
  values: MetricValues;
}

const NONE: MetricValues = { reflectance: null, spectral: null, spatial: null, synthesis: null, ha_metric: null, om_metric: null, im_metric: null };

const OPENSR_README = { text: "opensr-test README, Benchmark → ND table (Aybar et al., IEEE GRSL 2024)", href: "https://github.com/ESAOpenSR/opensr-test#benchmark" };

const REPORTED_CAVEAT =
  "Reported on the opensr-test benchmark datasets (NAIP/SPOT/SPAIN; the README does not say which subset), patch aggregation, GPU. Not our 1,199 SEN2NAIPv2 val pairs, so only rough positioning.";

export const MODELS: readonly ModelRow[] = [
  {
    id: "drishtisr",
    name: "DrishtiSR (A2)",
    role: "Our model",
    roleVariant: "ours",
    category: "CNN, EDSR-style 16×48",
    source: "measured",
    detail: "Current best run: A2 at 12,000 iterations (served as a2-last-dce224ec), chosen by the pre-registered rule. SEN2NAIPv2 val, 1,199 pairs, opensr-test 1.3.3, CPU.",
    snapshot: { label: "a2", checkpoint: "it12000" },
    color: "var(--color-series-1)",
    defaultOn: true,
    values: {
      reflectance: 0.003198582930068988,
      spectral: 0.7641632400446877,
      spatial: 0.005709370108844839,
      synthesis: 0.004054270413633532,
      ha_metric: 0.12832151947106493,
      om_metric: 0.7525003133364973,
      im_metric: 0.11917816910945545,
    },
  },
  {
    id: "edsr",
    name: "EDSR-baseline (Run A)",
    role: "Baseline Architecture",
    roleVariant: "baseline",
    category: "CNN, EDSR 16×64",
    source: "measured",
    detail: "The architecture DrishtiSR was built from, trained by us on the same data. Checkpoint 8,000 of 40,000, chosen by the same rule; same 1,199 pairs and settings.",
    snapshot: { label: "runA", checkpoint: "it08000" },
    color: "var(--color-series-2)",
    defaultOn: true,
    values: {
      reflectance: 0.003428249327163736,
      spectral: 0.9003644765069626,
      spatial: 0.023240478049103074,
      synthesis: 0.0044105124615755765,
      ha_metric: 0.13732506256796004,
      om_metric: 0.7327794038424201,
      im_metric: 0.1298955351120487,
    },
  },
  {
    id: "bicubic",
    name: "Bicubic",
    role: "Naive Baseline",
    roleVariant: "baseline",
    category: "Interpolation · 0 params",
    source: "measured",
    detail: "Bicubic upsampling of the same 10 m inputs, scored on the same 1,199 pairs and settings.",
    snapshot: { label: "bicubic", checkpoint: "bicubic" },
    color: "var(--color-series-3)",
    defaultOn: true,
    values: {
      reflectance: 0.0021704580961115693,
      spectral: 0.5151552355120439,
      spatial: 0.006195968478371244,
      synthesis: 0.0026408804489689583,
      ha_metric: 0.08084506550811547,
      om_metric: 0.8619920872866461,
      im_metric: 0.057162848507070464,
    },
  },
  {
    id: "swinir",
    name: "SwinIR",
    category: "Transformer · photo SR",
    source: "reported",
    detail: "Liang et al., ICCVW 2021. Published results are PSNR/SSIM on synthetic-bicubic photo benchmarks (Set5, Urban100, …).",
    reference: { text: "Liang et al., SwinIR, arXiv:2108.10257", href: "https://arxiv.org/abs/2108.10257" },
    naReason:
      "No opensr-test numbers are published for SwinIR (or HAT, or Real-ESRGAN), and we have not run them. Their photo-benchmark PSNR is a different metric on different data.",
    color: "var(--color-fg-subtle)",
    defaultOn: false,
    values: NONE,
  },
  {
    id: "satlas",
    name: "Satlas SR",
    category: "GAN, ESRGAN-based · NAIP + Sentinel-2",
    source: "reported",
    detail: REPORTED_CAVEAT,
    reference: OPENSR_README,
    readmeModel: "satlas",
    color: "var(--color-series-4)",
    defaultOn: false,
    values: { reflectance: 0.0489, spectral: 12.1231, spatial: 0.2742, synthesis: 0.0227, ha_metric: 0.8004, om_metric: 0.1073, im_metric: 0.0923 },
  },
  {
    id: "opensrmodel",
    name: "opensr-model",
    category: "Latent diffusion · Sentinel-2 (ESA OpenSR)",
    source: "reported",
    detail: REPORTED_CAVEAT,
    reference: OPENSR_README,
    readmeModel: "opensrmodel",
    color: "var(--color-series-5)",
    defaultOn: false,
    values: { reflectance: 0.0031, spectral: 1.2632, spatial: 0.0114, synthesis: 0.0068, ha_metric: 0.3431, om_metric: 0.4593, im_metric: 0.1976 },
  },
];

/** Measured-row settings, for the page header; checked against the snapshot. */
export const MEASURED_SPLIT = { dataset: "SEN2NAIPv2", split: "val", pairs: 1199, opensrVersion: "1.3.3" } as const;

export const COMPARE_SOURCES = {
  measured: "reports/compare/opensr_seven.json",
  reported: "reports/compare/opensr_test_readme_nd.md",
} as const;

export const isComplete = (m: ModelRow) => METRICS.every((d) => m.values[d.key] !== null);
