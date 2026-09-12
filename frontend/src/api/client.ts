/**
 * The ONLY module that talks to the backend. Every request and response the UI
 * uses is typed here; components never call fetch().
 *
 * MOCK_MODE (on unless VITE_MOCK_MODE=false) answers from fixtures and the
 * procedural placeholder imagery in public/placeholders/. Day 4 implements the
 * same two endpoints in FastAPI and sets VITE_MOCK_MODE=false; nothing else
 * changes.
 *
 *   GET  {API_BASE}/health               -> HealthResponse (always live, even in MOCK_MODE)
 *   POST {API_BASE}/sr       SrRequest   -> SrResponse
 *   GET  {API_BASE}/metrics              -> MetricsResponse
 *
 * The wire format is snake_case, which a Python backend emits natively, so the
 * Day 4 handlers can return pydantic models without an alias layer.
 */
import { APP_CONFIG } from "../config";
import { bboxAround, estimateAoi, type Bbox } from "../geo/aoi";
import metricsFixture from "./fixtures/metrics.json";

export const MOCK_MODE: boolean = import.meta.env.VITE_MOCK_MODE !== "false";
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
/** Simulated round trip in MOCK_MODE, so the loading states run before a backend exists. */
const MOCK_LATENCY_MS = 300;
/** A health probe slower than this counts as offline. */
const HEALTH_TIMEOUT_MS = 3000;

// --------------------------------------------------------- GET /health ----

export type HealthResult = { online: true } | { online: false; reason: string };

/**
 * One probe of the backend. Never throws: "offline" is an expected state before
 * Day 4, so it is returned with its reason rather than logged. The only console
 * line an unreachable backend produces is the browser's own failed-request
 * notice, once per page load. A 200 that is not JSON (an SPA fallback page, or
 * some other server on the port) counts as offline.
 */
export async function checkHealth(): Promise<HealthResult> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS) });
    if (!res.ok) return { online: false, reason: `GET /health -> HTTP ${res.status}` };
    await res.json();
    return { online: true };
  } catch (exc) {
    return { online: false, reason: exc instanceof Error ? exc.message : String(exc) };
  }
}

// ------------------------------------------------------------ POST /sr ----

export interface SrRequest {
  /** EPSG:4326 [west, south, east, north]. */
  bbox: Bbox;
}

/**
 * What a layer's pixels ARE. The UI keys provenance labelling off this, never
 * off a URL or a display name.
 *   measured          Sentinel-2 L2A as delivered.
 *   ai_reconstructed  network output. Always shown under the AI badge.
 *   ai_uncertainty    the heteroscedastic head's per-pixel sigma for ai_reconstructed.
 */
export type LayerKind = "measured" | "ai_reconstructed" | "ai_uncertainty";

export interface RasterLayer {
  /** 8-bit display rendering (PNG): a picture OF reflectance, not reflectance. */
  url: string;
  /** Geographic extent of the image, EPSG:4326 [west, south, east, north]. */
  footprint: Bbox;
  gsd_m: number;
  width_px: number;
  height_px: number;
  kind: LayerKind;
  /** One line shown beside the layer. */
  label: string;
  /** True when the pixels are a synthetic stand-in rather than data or a model output. */
  placeholder: boolean;
}

export interface SrResponse {
  request_id: string;
  /** The requested AOI, echoed. */
  bbox: Bbox;
  tiles: number;
  layers: { lr: RasterLayer; sr: RasterLayer; uncertainty: RasterLayer };
  provenance: { mode: "mock" | "live"; model: string | null; checkpoint: string | null; note: string };
}

// -------------------------------------------------------- GET /metrics ----

/** Per-method means over the validation split. Keys as in reports/day3_results.json. */
export interface MetricMeans {
  psnr_mean: number;
  ssim_mean: number;
  lpips: number;
  sam_mean_deg: number;
  ergas: number;
  /** opensr-test consistency error: L1 between the LR and the SR downsampled back to 10 m. */
  reflectance: number;
  spectral: number;
  spatial: number;
  /** The same comparison through the area operator the spectral loss trains on. */
  l1_spec: number;
  sam_spec_deg: number;
}

export interface MetricsRow {
  label: string;
  display: string;
  checkpoint: string;
  means: MetricMeans;
  lambdas: string;
  iters: string;
  /** Formatted with thousands separators, as in the report ("855,652"). */
  params: string;
  n_pairs: number;
}

/** A subset of reports/day3_results.json with keys unchanged, so Day 4 can serve that file's fields verbatim. */
export interface MetricsResponse {
  /** Repo-relative path of the report this came from. */
  source: string;
  written_utc: string;
  fingerprint: string;
  n_pairs: number;
  reading: string[];
  rows: MetricsRow[];
  supplementary_rows: MetricsRow[];
  /** The ground-truth HR scored as if it were the SR: the spectral floor. */
  floor: Pick<MetricMeans, "l1_spec" | "sam_spec_deg" | "reflectance" | "spectral" | "spatial">;
  below_floor: string[];
}

// ----------------------------------------------------------- transport ----

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? "GET";
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) throw new ApiError(res.status, `${method} ${path} -> HTTP ${res.status}: ${await res.text()}`);
  return (await res.json()) as T;
}

// ---------------------------------------------------------------- mock ----

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
const placeholderUrl = (name: string) => `${import.meta.env.BASE_URL}placeholders/${name}`;
const fmtM = (m: number) => `${Number.isInteger(m) ? m : m.toFixed(1)} m`;

const SR_GSD_M = APP_CONFIG.lrGsdM / APP_CONFIG.scale;
const DEMO_SIDE_M = APP_CONFIG.placeholder.hrPx * SR_GSD_M;

/**
 * The AOI the app opens on: the placeholder scene's footprint, centred on the
 * map's start position. One tile, so it is also a cheap first request on Day 4.
 */
export const DEMO_AOI: Bbox = bboxAround(APP_CONFIG.map.center, DEMO_SIDE_M, DEMO_SIDE_M);

const METRICS_FIXTURE: MetricsResponse = metricsFixture as MetricsResponse;

async function mockSr(req: SrRequest): Promise<SrResponse> {
  await sleep(MOCK_LATENCY_MS);
  const est = estimateAoi(req.bbox, APP_CONFIG);
  // What the Day 4 backend must do too: refuse, loudly, rather than start a job that hangs the demo.
  if (est.overBudget) {
    throw new ApiError(413, `AOI needs ${est.tiles.total} tiles (${est.tiles.x} x ${est.tiles.y}); the budget is ${est.tiles.max}.`);
  }
  const hr = APP_CONFIG.placeholder.hrPx;
  const lr = hr / APP_CONFIG.scale;
  const s = APP_CONFIG.scale;
  return {
    request_id: `mock-${Date.now().toString(36)}`,
    bbox: req.bbox,
    tiles: est.tiles.total,
    layers: {
      lr: {
        url: placeholderUrl("lr.png"), footprint: DEMO_AOI, gsd_m: APP_CONFIG.lrGsdM, width_px: lr, height_px: lr,
        kind: "measured", label: `${fmtM(APP_CONFIG.lrGsdM)} input · PLACEHOLDER`, placeholder: true,
      },
      sr: {
        url: placeholderUrl("hr.png"), footprint: DEMO_AOI, gsd_m: SR_GSD_M, width_px: hr, height_px: hr,
        kind: "ai_reconstructed", label: `${fmtM(SR_GSD_M)} ×${s} · PLACEHOLDER`, placeholder: true,
      },
      uncertainty: {
        url: placeholderUrl("uncertainty.png"), footprint: DEMO_AOI, gsd_m: SR_GSD_M, width_px: hr, height_px: hr,
        kind: "ai_uncertainty", label: "PLACEHOLDER: edge magnitude of the placeholder image, not a model σ", placeholder: true,
      },
    },
    provenance: {
      mode: "mock",
      model: null,
      checkpoint: null,
      note:
        `MOCK_MODE: no model ran. The layers are a procedural placeholder pair at a fixed footprint; ` +
        `the ${fmtM(APP_CONFIG.lrGsdM)} image is the exact ${s}×${s} block mean of the ${fmtM(SR_GSD_M)} one. ` +
        `Your AOI was checked against the tile budget and echoed, not processed.`,
    },
  };
}

// ----------------------------------------------------------------- API ----

export async function runSr(req: SrRequest): Promise<SrResponse> {
  if (MOCK_MODE) return mockSr(req);
  return request<SrResponse>("/sr", { method: "POST", body: JSON.stringify(req) });
}

export async function getMetrics(): Promise<MetricsResponse> {
  if (MOCK_MODE) {
    await sleep(MOCK_LATENCY_MS);
    return METRICS_FIXTURE;
  }
  return request<MetricsResponse>("/metrics");
}

// ================================ MVP API v1 (app/server.py, same origin) ====
//
// docs/mvp/api_contract.md. Unlike the Day 3 endpoints above these are
// same-origin: the server refuses cross-origin POSTs, so `npm run dev` proxies
// /api and /files to it (frontend.api.proxy_target in configs/base.yaml).
// MOCK_MODE answers with the placeholder pair and every per-image metric null:
// no model ran, so there is nothing to report.

/** Prefix for /api and /files. Empty = same origin (the dev proxy, or a build served by app/server.py). */
export const MVP_API_BASE: string = import.meta.env.VITE_MVP_API_BASE ?? "";
const UPLOAD_EXTENSIONS = [".tif", ".tiff"];

export interface ModelInfo {
  backend: "onnx-int8" | "onnx-fp32" | "torch-fp32";
  checkpoint_id: string;
  params: number;
  model_bytes: number;
  threads: number;
  interim: boolean;
  has_scale_head: boolean;
}

export interface QualityMetrics {
  lpips: number;
  ssim: number;
  psnr: number;
  sam_deg: number;
  ergas: number;
}

export type UncertaintyMethod = "none" | "tta4" | "tta8" | "learned_laplace";

/** POST /api/upscale. Image fields are URLs of 8-bit display PNGs; metrics are in surface-reflectance units. */
export interface UpscaleResponse {
  job_id: string;
  input: {
    source: "upload" | "sample";
    sample_id: string | null;
    lr_size: [number, number];
    sr_size: [number, number];
    georeferenced: boolean;
    dn_mode_applied: string;
  };
  model: ModelInfo;
  uncertainty_method: UncertaintyMethod;
  images: {
    lr_rgb: string;
    lr_fcc: string;
    bicubic_rgb: string;
    bicubic_fcc: string;
    sr_rgb: string;
    sr_fcc: string;
    hr_rgb: string | null;
    hr_fcc: string | null;
    uncertainty: string | null;
    consistency: string;
    /** D(SR): the SR degraded back to 10 m, rendered like lr_rgb. Absent from servers older than the spectral page. */
    sr_degraded_rgb?: string;
    /** Per-pixel spectral angle D(SR) vs LR, fixed scale 0..refs.spec_sam_gt_floor_deg. Absent from older servers. */
    consistency_sam?: string;
  };
  downloads: { sr_tif: string; uncertainty_tif: string | null };
  metrics: {
    reference_free: {
      spec_l1: number;
      spec_sam_deg: number;
      spec_l1_bicubic: number;
      spec_sam_bicubic_deg: number;
      hf_ratio_vs_bicubic: number;
      unc_mean: number | null;
      unc_p95: number | null;
      runtime_ms: { sr: number; uncertainty: number | null; total: number };
    };
    with_gt: null | {
      sr: QualityMetrics;
      bicubic: QualityMetrics;
      spec_l1_hr: number;
      spec_sam_hr_deg: number;
      hf_ratio_hr_vs_bicubic: number;
    };
  };
  refs: {
    spec_l1_gt_floor: number;
    spec_sam_gt_floor_deg: number;
    unc_display_max: number;
    cons_display_max: number;
    sharpness_warn_below: number;
  };
  warnings: string[];
}

export interface SampleInfo {
  id: string;
  label: string;
  thumb_url: string;
  has_gt: boolean;
  lr_size: [number, number];
}

export type TtaChoice = "0" | "4" | "8";
export type UpscaleInput = { file: File } | { sampleId: string };

/**
 * Per-image metrics as the pages show them. `null` means NOT MEASURED for this
 * image (no 2.5 m reference, or MOCK_MODE), never zero.
 */
export interface ImageMetrics {
  psnr: number | null;
  ssim: number | null;
  lpips: number | null;
  /** Bicubic on the same image, for deltas. Null without a reference. */
  bicubic: Pick<QualityMetrics, "psnr" | "ssim" | "lpips"> | null;
  /** Spectral consistency: L1 between the LR and the SR degraded back to 10 m, surface reflectance. */
  specL1: number | null;
  specL1Bicubic: number | null;
  /** HF energy of SR over bicubic (bicubic = 1). Below `sharpnessWarnBelow`, a low specL1 may just be blur. */
  hfRatioVsBicubic: number | null;
  sharpnessWarnBelow: number | null;
  runtimeMsTotal: number | null;
}

/**
 * The reconstruction-consistency check for one image: D(SR) against the LR,
 * in surface reflectance (L1) and degrees (spectral angle). `null` = not
 * measured (MOCK_MODE), never zero.
 */
export interface SpectralCheck {
  samDeg: number | null;
  samDegBicubic: number | null;
  /** The true 2.5 m image scored the same way on full VAL: a reference point, not a lower bound. */
  l1GtFloor: number | null;
  samGtFloorDeg: number | null;
  /** This image's own 2.5 m reference, degraded and compared with its LR. Null without a reference. */
  l1Hr: number | null;
  samHrDeg: number | null;
  /** Fixed overlay scales the server rendered with (never per-image min-max). */
  l1DisplayMax: number | null;
  samDisplayMax: number | null;
}

/** One image's result, shared app-wide through ResultContext. */
export interface ImageResult {
  mode: "mock" | "live";
  jobId: string;
  /** File name or sample label. */
  sourceLabel: string;
  lrSize: [number, number] | null;
  srSize: [number, number] | null;
  images: {
    lr: string;
    sr: string;
    bicubic: string | null;
    hr: string | null;
    uncertainty: string | null;
    /** Per-pixel L1 |D(SR) - LR|, fixed scale 0..spectral.l1DisplayMax. */
    consistency: string | null;
    /** D(SR) at 10 m, same render as `lr`. Null in MOCK_MODE or from an older server. */
    srDegraded: string | null;
    /** Per-pixel spectral angle D(SR) vs LR, fixed scale 0..spectral.samDisplayMax. */
    consistencySam: string | null;
  };
  srTifUrl: string | null;
  hasGroundTruth: boolean;
  metrics: ImageMetrics;
  spectral: SpectralCheck;
  model: ModelInfo | null;
  uncertaintyMethod: UncertaintyMethod | null;
  warnings: string[];
  /** The full response, for pages that need fields not lifted here. Null in MOCK_MODE. */
  raw: UpscaleResponse | null;
}

const mvpUrl = (u: string) => (u.startsWith("/") ? `${MVP_API_BASE}${u}` : u);
const mvpUrlOrNull = (u: string | null) => (u === null ? null : mvpUrl(u));

/** Lift an /api/upscale response into the shape the pages read. Pure; unit-tested. */
export function toImageResult(resp: UpscaleResponse, sourceLabel: string): ImageResult {
  const rf = resp.metrics.reference_free;
  const gt = resp.metrics.with_gt;
  return {
    mode: "live",
    jobId: resp.job_id,
    sourceLabel,
    lrSize: resp.input.lr_size,
    srSize: resp.input.sr_size,
    images: {
      lr: mvpUrl(resp.images.lr_rgb),
      sr: mvpUrl(resp.images.sr_rgb),
      bicubic: mvpUrl(resp.images.bicubic_rgb),
      hr: mvpUrlOrNull(resp.images.hr_rgb),
      uncertainty: mvpUrlOrNull(resp.images.uncertainty),
      consistency: mvpUrl(resp.images.consistency),
      srDegraded: mvpUrlOrNull(resp.images.sr_degraded_rgb ?? null),
      consistencySam: mvpUrlOrNull(resp.images.consistency_sam ?? null),
    },
    srTifUrl: mvpUrl(resp.downloads.sr_tif),
    hasGroundTruth: gt !== null,
    spectral: {
      samDeg: rf.spec_sam_deg,
      samDegBicubic: rf.spec_sam_bicubic_deg,
      l1GtFloor: resp.refs.spec_l1_gt_floor,
      samGtFloorDeg: resp.refs.spec_sam_gt_floor_deg,
      l1Hr: gt?.spec_l1_hr ?? null,
      samHrDeg: gt?.spec_sam_hr_deg ?? null,
      l1DisplayMax: resp.refs.cons_display_max,
      // app/pipeline.py renders consistency_sam on 0..spec_sam_gt_floor_deg.
      samDisplayMax: resp.images.consistency_sam ? resp.refs.spec_sam_gt_floor_deg : null,
    },
    metrics: {
      psnr: gt?.sr.psnr ?? null,
      ssim: gt?.sr.ssim ?? null,
      lpips: gt?.sr.lpips ?? null,
      bicubic: gt ? { psnr: gt.bicubic.psnr, ssim: gt.bicubic.ssim, lpips: gt.bicubic.lpips } : null,
      specL1: rf.spec_l1,
      specL1Bicubic: rf.spec_l1_bicubic,
      hfRatioVsBicubic: rf.hf_ratio_vs_bicubic,
      sharpnessWarnBelow: resp.refs.sharpness_warn_below,
      runtimeMsTotal: rf.runtime_ms.total,
    },
    model: resp.model,
    uncertaintyMethod: resp.uncertainty_method,
    warnings: resp.warnings,
    raw: resp,
  };
}

async function mvpRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? "GET";
  const res = await fetch(`${MVP_API_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    let detail = body;
    try {
      detail = (JSON.parse(body) as { error?: string }).error ?? body;
    } catch {
      // Not the server's {"error"} envelope (a proxy error page, say): report the raw body.
    }
    throw new ApiError(res.status, `${method} ${path} -> HTTP ${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

function mockUpscale(sourceLabel: string): ImageResult {
  const hr = APP_CONFIG.placeholder.hrPx;
  const lr = hr / APP_CONFIG.scale;
  return {
    mode: "mock",
    jobId: `mock-${Date.now().toString(36)}`,
    sourceLabel: `${sourceLabel} · PLACEHOLDER`,
    lrSize: [lr, lr],
    srSize: [hr, hr],
    images: { lr: placeholderUrl("lr.png"), sr: placeholderUrl("hr.png"), bicubic: null, hr: null, uncertainty: placeholderUrl("uncertainty.png"), consistency: null, srDegraded: null, consistencySam: null },
    srTifUrl: null,
    hasGroundTruth: false,
    spectral: { samDeg: null, samDegBicubic: null, l1GtFloor: null, samGtFloorDeg: null, l1Hr: null, samHrDeg: null, l1DisplayMax: null, samDisplayMax: null },
    metrics: { psnr: null, ssim: null, lpips: null, bicubic: null, specL1: null, specL1Bicubic: null, hfRatioVsBicubic: null, sharpnessWarnBelow: null, runtimeMsTotal: null },
    model: null,
    uncertaintyMethod: null,
    warnings: ["MOCK_MODE: no model ran. The images are the procedural placeholder pair, not your upload."],
    raw: null,
  };
}

/**
 * Super-resolve one uploaded 4-band GeoTIFF or one server-side sample.
 * Throws ApiError with the server's message on any refusal; never returns a partial result.
 */
export async function upscale(input: UpscaleInput, sourceLabel: string, tta: TtaChoice = "4"): Promise<ImageResult> {
  if ("file" in input) {
    const name = input.file.name.toLowerCase();
    if (!UPLOAD_EXTENSIONS.some((ext) => name.endsWith(ext))) {
      throw new ApiError(400, `"${input.file.name}" is not a GeoTIFF; upload a 4-band ${UPLOAD_EXTENSIONS.join(" / ")} (${APP_CONFIG.bands.join(", ")}).`);
    }
  }
  if (MOCK_MODE) {
    await sleep(MOCK_LATENCY_MS);
    return mockUpscale(sourceLabel);
  }
  const form = new FormData();
  if ("file" in input) form.append("file", input.file);
  else form.append("sample_id", input.sampleId);
  form.append("tta", tta);
  const resp = await mvpRequest<UpscaleResponse>("/api/upscale", { method: "POST", body: form });
  return toImageResult(resp, sourceLabel);
}

/** Server-side samples (with 2.5 m ground truth where `has_gt`). Empty in MOCK_MODE. */
export async function getSamples(): Promise<SampleInfo[]> {
  if (MOCK_MODE) return [];
  const { samples } = await mvpRequest<{ samples: SampleInfo[] }>("/api/samples");
  return samples.map((s) => ({ ...s, thumb_url: mvpUrl(s.thumb_url) }));
}
