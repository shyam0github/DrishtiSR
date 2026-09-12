/**
 * End-to-end check of /novelty/efficiency in the system Chrome (playwright-core):
 * section B shows the Home CTA with an example (live) or an explicit "no
 * example" state (mock); the benchmark renders every chart from src/data/facts.ts
 * and srModels.ts; an upload made on Home is what section B then times.
 *
 *     node scripts/e2e_efficiency.mjs              # MOCK_MODE
 *     E2E_LIVE=1 node scripts/e2e_efficiency.mjs   # against scripts/serve.py via the dev proxy
 *
 * Fails loudly on the first failed assertion, page error or console error.
 * Screenshots go to outputs/figures/frontend_efficiency_*.png (gitignored).
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";
import { createServer } from "vite";
import { REPO_ROOT } from "./repo-config.mjs";

const LIVE = process.env.E2E_LIVE === "1";
if (LIVE) process.env.VITE_MOCK_MODE = "false";
const TAG = LIVE ? "live" : "mock";
const FRONTEND_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SHOT_DIR = path.join(REPO_ROOT, "outputs", "figures");
const MODEL_TIMEOUT_MS = 120_000;

const server = await createServer({ root: FRONTEND_ROOT, server: { port: 0, strictPort: false }, logLevel: "error" });
await server.listen();
const url = server.resolvedUrls.local[0].replace(/\/$/, "");
const browser = await chromium.launch({ channel: "chrome" });
mkdirSync(SHOT_DIR, { recursive: true });

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => m.type() === "error" && errors.push(`console.error: ${m.text()}`));

  await page.goto(`${url}/novelty/efficiency`);
  await page.getByTestId("page-efficiency").waitFor();
  assert.equal(await page.title(), "CPU-only efficiency · DrishtiSR");

  // B: no upload -> CTA + example (live) or an explicit no-example state (mock).
  await page.getByTestId("no-upload").waitFor();
  if (LIVE) await page.locator('[data-testid="eff-image"][data-source="example"]').waitFor({ timeout: MODEL_TIMEOUT_MS });
  else await page.getByTestId("example-unavailable").waitFor();

  // C: every benchmark block, with the measured numbers.
  for (const id of ["eff-param-chart", "eff-model-size", "eff-latency", "eff-quant"]) await page.getByTestId(id).waitFor();
  const chart = page.getByTestId("eff-param-chart");
  assert.equal(await chart.locator('[aria-label*="parameters ("]').count(), 11, "11 models on the parameter chart");
  assert.match(await chart.locator('[aria-label^="DrishtiSR"]').getAttribute("aria-label"), /855,652 parameters/);
  await page.getByTestId("eff-latency").scrollIntoViewIfNeeded();
  await page.waitForTimeout(1600); // bars and count-ups settle
  const latency = await page.getByTestId("eff-latency").innerText();
  for (const ms of ["2,369", "1,220", "1,604", "782"]) assert.ok(latency.includes(ms), `latency shows ${ms} ms`);
  const rows = page.getByTestId("eff-quant").locator("tbody tr");
  assert.equal(await rows.count(), 4, "four INT8 attempts");
  for (const t of await rows.allInnerTexts()) assert.match(t, /failed/i, `attempt row not marked failed: ${t}`);
  assert.match(await page.getByTestId("eff-quant").innerText(), /−0\.329 dB/);

  await page.screenshot({ path: path.join(SHOT_DIR, `frontend_efficiency_${TAG}.png`), fullPage: true });

  // Upload on Home, then back here through the nav: section B times "yours".
  await page.getByTestId("cta-upload").click();
  await page.getByTestId("live-demo").waitFor();
  if (LIVE) {
    await page.getByRole("button", { name: /Has 2.5 m truth/ }).first().click();
  } else {
    const tif = path.join(os.tmpdir(), "drishti_e2e_upload.tif");
    writeFileSync(tif, Buffer.from("not a real tiff: MOCK_MODE never reads it"));
    await page.locator("#upload-input").setInputFiles(tif);
  }
  await page.getByTestId("live-demo").getByRole("slider", { name: "Comparison divider" }).waitFor({ timeout: MODEL_TIMEOUT_MS });
  await page.getByRole("link", { name: /Efficiency/ }).first().click();
  await page.locator('[data-testid="eff-image"][data-source="yours"]').waitFor();
  const pending = await page.getByTestId("eff-image-timing").getByTestId("pending-stat").count();
  assert.equal(pending, LIVE ? 0 : 4, LIVE ? "live: every timing measured" : "mock: every timing pending");
  await page.getByTestId("section-this-image").screenshot({ path: path.join(SHOT_DIR, `frontend_efficiency_yours_${TAG}.png`) });

  // Phone width: no horizontal page scroll.
  await page.setViewportSize({ width: 390, height: 844 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  assert.ok(overflow <= 1, `page scrolls horizontally at 390 px by ${overflow}px`);
  await page.screenshot({ path: path.join(SHOT_DIR, `frontend_efficiency_phone_${TAG}.png`), fullPage: true });

  assert.deepEqual(errors, [], errors.join("\n"));
  console.log(`e2e_efficiency (${TAG}): OK`);
} finally {
  await browser.close();
  await server.close();
}
