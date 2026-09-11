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
