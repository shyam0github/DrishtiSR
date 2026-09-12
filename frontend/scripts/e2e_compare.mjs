/**
 * End-to-end check of /compare in the system Chrome (playwright-core, channel
 * "chrome"): glossary has all seven metrics, the disclaimer is visible, the
 * radar starts with the three measured rows and toggles literature rows, a
 * row with no opensr-test numbers cannot be plotted and shows N/A, the table
 * sorts, and the page does not scroll sideways at phone width.
 *
 *     npm run test:e2e:compare
 *
 * Fails loudly on the first failed assertion, page error or console error.
 * Screenshots go to outputs/figures/frontend_compare_*.png (gitignored).
 */
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";
import { createServer } from "vite";
import { REPO_ROOT } from "./repo-config.mjs";

const FRONTEND_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SHOT_DIR = path.join(REPO_ROOT, "outputs", "figures");

function watchErrors(page, label) {
  const errors = [];
  page.on("pageerror", (e) => errors.push(`${label} pageerror: ${e.message}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(`${label} console.error: ${msg.text()} @ ${msg.location()?.url ?? ""}`);
  });
  return errors;
}

const server = await createServer({ root: FRONTEND_ROOT, server: { port: 0, strictPort: false }, logLevel: "error" });
await server.listen();
const url = new URL("/compare", server.resolvedUrls.local[0]).href;
const browser = await chromium.launch({ channel: "chrome" });
mkdirSync(SHOT_DIR, { recursive: true });

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = watchErrors(page, "desktop");
  await page.goto(url);
  await page.getByTestId("compare-intro").waitFor();

  assert.equal(await page.getByTestId("glossary").locator("tbody tr").count(), 7, "glossary: 7 metrics");
  assert.ok(await page.getByTestId("disclaimer").isVisible(), "disclaimer visible");

  const radar = page.getByTestId("radar");
  await radar.scrollIntoViewIfNeeded();
  const plotted = () => radar.locator("g[data-series]").count();
  assert.equal(await plotted(), 3, "radar starts with the 3 measured rows");
  await page.locator('input[data-model="satlas"]').check();
  assert.equal(await plotted(), 4, "toggling Satlas adds a polygon");
  await page.locator('input[data-model="bicubic"]').uncheck();
  assert.equal(await plotted(), 3, "unticking bicubic removes it");
  assert.ok(await page.locator('input[data-model="swinir"]').isDisabled(), "SwinIR has no numbers: cannot plot");

  // Hover a vertex: tooltip with an exact value.
  const box = await radar.locator("svg").boundingBox();
  const hits = radar.locator("circle[fill=transparent]");
  await hits.first().hover();
  await radar.getByRole("status").waitFor();
  await page.waitForTimeout(500);
  await radar.screenshot({ path: path.join(SHOT_DIR, "frontend_compare_radar.png") });
  assert.ok(box && box.width > 300, `radar too small: ${box?.width}`);

  const table = page.getByTestId("metric-table");
  assert.equal(await table.locator("td[data-na]").count(), 7, "SwinIR row: 7 N/A cells");
  await table.locator('button[data-sort="spectral"]').click();
  assert.equal(await table.locator("tbody tr").first().getAttribute("data-row"), "bicubic", "spectral best-first: bicubic");
  await table.locator('button[data-sort="om_metric"]').click();
  assert.equal(await table.locator("tbody tr").first().getAttribute("data-row"), "satlas", "omission best-first: satlas");
  assert.equal(await table.locator("tbody tr").last().getAttribute("data-row"), "swinir", "N/A sorts last");

  await page.screenshot({ path: path.join(SHOT_DIR, "frontend_compare_desktop.png"), fullPage: true });
  assert.deepEqual(errors, [], errors.join("\n"));

  const phone = await browser.newPage({ viewport: { width: 400, height: 860 } });
  const phoneErrors = watchErrors(phone, "phone");
  await phone.goto(url);
  await phone.getByTestId("comparison").waitFor();
  await phone.waitForTimeout(800);
  const overflow = await phone.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert.ok(overflow <= 0, `phone: page scrolls sideways by ${overflow}px`);
  await phone.screenshot({ path: path.join(SHOT_DIR, "frontend_compare_phone.png"), fullPage: true });
  assert.deepEqual(phoneErrors, [], phoneErrors.join("\n"));

  console.log("e2e_compare: OK");
} finally {
  await browser.close();
  await server.close();
}
