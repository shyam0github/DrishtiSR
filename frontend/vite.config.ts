import path from "node:path";
import { defineConfig, type Plugin } from "vite";
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

export default defineConfig({
  plugins: [react(), tailwindcss(), restartOnBaseYaml()],
  // maplibre-gl 6 loads its web worker as a sibling file (maplibre-gl-worker.mjs).
  // Vite's dependency pre-bundler moves the main module into .vite/deps without
  // that sibling, so the worker 404s and the map renders blank. Serve it as-is.
  optimizeDeps: { exclude: ["maplibre-gl"] },
  define: {
    __DRISHTI_CONFIG__: JSON.stringify(loadFrontendConfig()),
  },
});
