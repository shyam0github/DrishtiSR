/**
 * Settings baked in at build time from configs/base.yaml (the `frontend` block,
 * plus sr.scale, dataset.bands, runtime.max_parameters and seed). Produced by
 * scripts/repo-config.mjs, injected by vite.config.ts. Keep this interface in
 * step with that loader.
 */
export interface FrontendConfig {
  map: {
    /** WGS84 [lon, lat]. */
    center: [number, number];
    zoom: number;
    basemap: { tiles: string[]; tileSize: number; maxZoom: number; attribution: string };
  };
  /** Ground sample distance of the input, metres. */
  lrGsdM: number;
  /** Super-resolution factor (sr.scale). Output GSD = lrGsdM / scale. */
  scale: number;
  /** Band names of the SR raster, in channel order (dataset.bands). */
  bands: string[];
  outputBytesPerBand: number;
  /** Mirrors src/infer/tiled.py's --tile / --overlap, in input (LR) pixels. */
  tile: { lrPx: number; overlapLrPx: number; maxTiles: number };
  /** runtime.max_parameters: the model budget, shown against each row's parameter count. */
  maxParameters: number;
  /** frontend.api.proxy_target: where the dev server proxies /api and /files (app/server.py). */
  apiProxyTarget: string;
  teamName: string;
  repoUrl: string;
  placeholder: { hrPx: number; seed: number };
}

declare const __DRISHTI_CONFIG__: FrontendConfig;

export const APP_CONFIG: FrontendConfig = __DRISHTI_CONFIG__;
