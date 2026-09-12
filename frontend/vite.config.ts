import path from "node:path";
import { defineConfig, type Plugin, type ProxyOptions } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { BASE_YAML, loadFrontendConfig } from "./scripts/repo-config.mjs";

const samePath = (a: string, b: string) =>
  process.platform === "win32"
    ? path.resolve(a).toLowerCase() === path.resolve(b).toLowerCase()
    : path.resolve(a) === path.resolve(b);

/**
 * Restart the dev server when configs/base.yaml changes. The config is baked
 * in at startup, so without this an edit would be silently ignored until the
 * next manual restart.
 */
function restartOnBaseYaml(): Plugin {
  return {
    name: "drishtisr:base-yaml",
    configureServer(server) {
      server.watcher.add(BASE_YAML);
      server.watcher.on("change", (file) => {
        if (samePath(file, BASE_YAML)) void server.restart();
      });
    },
  };
}

/**
 * Forward the MVP API (app/server.py) so the page stays same-origin. The server
 * rejects any POST whose Origin differs from its Host; the proxy is the origin
 * boundary in dev, so it drops the browser's Origin header on the way through.
 */
function proxyTo(target: string): ProxyOptions {
  return {
    target,
    changeOrigin: true,
    configure: (proxy) => {
      proxy.on("proxyReq", (proxyReq) => proxyReq.removeHeader("origin"));
    },
  };
}

const FRONTEND_CONFIG = loadFrontendConfig();
// serve.py takes the first free port from 8000 up; DRISHTI_API_TARGET points the proxy at the one it printed.
const API_TARGET = process.env.DRISHTI_API_TARGET ?? FRONTEND_CONFIG.apiProxyTarget;
const API_PROXY = { "/api": proxyTo(API_TARGET), "/files": proxyTo(API_TARGET) };

export default defineConfig({
  plugins: [react(), tailwindcss(), restartOnBaseYaml()],
  // maplibre-gl 6 loads its web worker as a sibling file (maplibre-gl-worker.mjs).
  // Vite's dependency pre-bundler moves the main module into .vite/deps without
  // that sibling, so the worker 404s and the map renders blank. Serve it as-is.
  optimizeDeps: { exclude: ["maplibre-gl"] },
  server: { proxy: API_PROXY },
  preview: { proxy: API_PROXY },
  define: {
    __DRISHTI_CONFIG__: JSON.stringify(FRONTEND_CONFIG),
  },
});
