/**
 * End-to-end check of /novelty/spectral in the system Chrome (playwright-core):
 * with no upload the page shows the Home CTA and an example (or, in MOCK_MODE,
 * an explicit "no example" state); the benchmark charts render from the
 * report snapshot; "How it works" starts collapsed; an upload made on Home is
 * what section B then shows.
 *
 *     npm run test:e2e:spectral              # MOCK_MODE
 *     E2E_LIVE=1 npm run test:e2e:spectral   # against scripts/serve.py via the dev proxy
 *
 * Fails loudly on the first failed assertion, page error or console error.
 * Screenshots go to outputs/figures/frontend_spectral_*.png (gitignored).
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

  await page.goto(`${url}/novelty/spectral`);
  await page.getByTestId("page-spectral").waitFor();
  assert.equal(await page.title(), "Spectral consistency · DrishtiSR");

  // B: no upload -> CTA + example (live) or an explicit no-example state (mock). Never empty.
  await page.getByTestId("no-upload").waitFor();
  if (LIVE) {
    const img = page.locator('[data-testid="spectral-image"][data-source="example"]');
    await img.waitFor({ timeout: MODEL_TIMEOUT_MS });
    await page.getByTestId("degraded-img").waitFor();
    await page.getByTestId("overlay-sam").waitFor();
    await page.getByTestId("overlay-legend").waitFor();
    const natural = await page.getByTestId("degraded-img").evaluate((el) => el.complete && el.naturalWidth);
    assert.ok(natural > 0, "D(SR) image must load");
  } else {
    await page.getByTestId("example-unavailable").waitFor();
  }

  // C: charts from the report snapshot, headline says -33.3%.
  for (const id of ["bench-headline", "bench-bars", "bench-ci", "box-l1_spec", "box-sam_spec_deg", "b2-not-benchmarked"]) {
    await page.getByTestId(id).waitFor();
  }
  assert.match(await page.getByTestId("bench-headline").innerText(), /-33\.3%/);
  assert.match(await page.getByTestId("section-benchmark").innerText(), /n = 1,199 val patches · 300 tiles/);

  // D: collapsed by default, opens on click.
  const details = page.getByTestId("section-how").locator("details");
  assert.equal(await details.evaluate((d) => d.open), false, "How it works starts collapsed");
  await page.getByTestId("section-how").locator("summary").click();
  assert.equal(await details.evaluate((d) => d.open), true);
  assert.match(await details.innerText(), /degenerate minimum/);

  await page.screenshot({ path: path.join(SHOT_DIR, `frontend_spectral_${TAG}.png`), fullPage: true });

  // CTA -> Home's upload section.
  await page.getByTestId("cta-upload").click();
  await page.getByTestId("live-demo").waitFor();
  await page.waitForFunction(() => Math.abs(document.getElementById("try-it").getBoundingClientRect().top) < 120, null, { timeout: 5000 });

  // Upload on Home, then back to the spectral page through the nav: section B is "yours".
  if (LIVE) {
    await page.getByRole("button", { name: /Has 2.5 m truth/ }).first().click();
  } else {
    const tif = path.join(os.tmpdir(), "drishti_e2e_upload.tif");
    writeFileSync(tif, Buffer.from("not a real tiff: MOCK_MODE never reads it"));
    await page.locator("#upload-input").setInputFiles(tif);
  }
  await page.getByTestId("live-demo").getByRole("slider", { name: "Comparison divider" }).waitFor({ timeout: MODEL_TIMEOUT_MS });
  await page.getByRole("link", { name: /Spectral/ }).first().click();
  await page.locator('[data-testid="spectral-image"][data-source="yours"]').waitFor();
  assert.equal(await page.getByTestId("no-upload").count(), 0, "with an upload the CTA is gone");
  if (!LIVE) {
    await page.getByTestId("degraded-pending").waitFor();
    assert.equal(await page.getByTestId("spectral-image-metrics").getByTestId("pending-stat").count(), 3, "mock: every per-image number pending");
  }
  await page.screenshot({ path: path.join(SHOT_DIR, `frontend_spectral_yours_${TAG}.png`) });

  // Phone width: no horizontal page scroll.
  await page.setViewportSize({ width: 390, height: 844 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  assert.ok(overflow <= 1, `page scrolls horizontally at 390 px by ${overflow}px`);

  assert.deepEqual(errors, [], errors.join("\n"));
  console.log(`e2e_spectral (${TAG}): OK`);
} finally {
  await browser.close();
  await server.close();
}
