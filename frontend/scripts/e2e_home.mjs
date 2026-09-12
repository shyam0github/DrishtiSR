/**
 * End-to-end check of the Home page in the system Chrome (playwright-core,
 * channel "chrome"): hero CTA scrolls to the demo, an upload produces a large
 * swipe viewer, the viewer opens the zoom lightbox, the quick-metrics strip
 * and deep links render, and the footer carries the team/repo links.
 *
 *     npm run test:e2e:home              # MOCK_MODE: metrics must be Pending
 *     E2E_LIVE=1 npm run test:e2e:home   # against scripts/serve.py via the dev proxy:
 *                                        # a sample with ground truth must yield numbers
 *
 * Fails loudly on the first failed assertion, page error or console error.
 * Screenshots go to outputs/figures/frontend_home_*.png (gitignored).
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

const FRONTEND_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SHOT_DIR = path.join(REPO_ROOT, "outputs", "figures");
const TAG = LIVE ? "live" : "mock";

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
const url = server.resolvedUrls.local[0];
const browser = await chromium.launch({ channel: "chrome" });
mkdirSync(SHOT_DIR, { recursive: true });

try {
  // ---------------------------------------------------------- desktop ----
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = watchErrors(page, "desktop");
  await page.goto(url);
  await page.getByTestId("home-hero").waitFor();
  await page.waitForTimeout(1200); // staggered entrance settles
  await page.screenshot({ path: path.join(SHOT_DIR, `frontend_home_hero_${TAG}.png`) });

  await page.getByTestId("hero-cta").click();
  await page.waitForFunction(() => {
    const r = document.getElementById("try-it").getBoundingClientRect();
    return Math.abs(r.top) < 80;
  }, null, { timeout: 5000 });

  // Before any image: every per-image metric is Pending, never a number.
  assert.equal(await page.getByTestId("quick-metrics").getByTestId("pending-stat").count(), 4, "idle: 4 pending metrics");

  if (LIVE) {
    const sample = page.getByRole("button", { name: /Has 2.5 m truth/ }).first();
    await sample.waitFor({ timeout: 15_000 });
    await sample.click();
  } else {
    const tif = path.join(os.tmpdir(), "drishti_e2e_upload.tif");
    writeFileSync(tif, Buffer.from("not a real tiff: MOCK_MODE never reads it"));
    await page.locator("#upload-input").setInputFiles(tif);
  }

  const slider = page.getByTestId("live-demo").getByRole("slider", { name: "Comparison divider" });
  await slider.waitFor({ timeout: 120_000 });
  const vb = await page.getByTestId("live-demo").locator("figure > div").first().boundingBox();
  assert.ok(vb && vb.width >= 600, `viewer must be large, got width ${vb?.width}`);
  await page.getByTestId("ai-badge").first().waitFor();

  const pending = await page.getByTestId("quick-metrics").getByTestId("pending-stat").count();
  if (LIVE) {
    assert.equal(pending, 0, "live sample with ground truth: all 4 metrics measured");
    // The count-up must land exactly on the measured value (aria-label carries the exact figure).
    // StatCard starts counting only once seen, so bring the strip into view first.
    await page.getByTestId("quick-metrics").scrollIntoViewIfNeeded();
    await page.waitForTimeout(1800);
    const shown = await page.getByTestId("quick-metrics").locator("p[aria-label]").evaluateAll((ps) =>
      ps.map((p) => ({ label: p.getAttribute("aria-label"), text: p.querySelector(".text-h3")?.textContent ?? "" })),
    );
    assert.equal(shown.length, 4, "4 numeric metrics");
    for (const s of shown) assert.ok(s.label.startsWith(s.text) && Number(s.text) !== 0, `metric settled wrong: ${JSON.stringify(s)}`);
  } else assert.equal(pending, 4, "mock: all 4 metrics stay pending");

  for (const name of ["View spectral analysis", "View uncertainty map", "View efficiency stats"]) {
    await page.getByRole("link", { name }).waitFor();
  }
  await page.getByTestId("live-demo").scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(SHOT_DIR, `frontend_home_demo_${TAG}.png`) });

  // Click-to-expand: the lightbox opens, zooms, closes on Esc.
  await page.getByRole("button", { name: "Open full-size comparison with zoom" }).click();
  const dialog = page.getByRole("dialog");
  await dialog.waitFor();
  await page.getByRole("button", { name: "Zoom in" }).click();
  await page.getByRole("button", { name: "Zoom in" }).click();
  assert.match(await dialog.innerText(), /225%/, "two zoom steps -> 225%");
  await page.screenshot({ path: path.join(SHOT_DIR, `frontend_home_lightbox_${TAG}.png`) });
  await page.keyboard.press("Escape");
  await dialog.waitFor({ state: "detached" });

  // Shared state survives navigation: back on Home, the result is still there.
  await page.getByRole("link", { name: "View spectral analysis" }).click();
  await page.waitForURL(/\/novelty\/spectral$/);
  await page.getByTestId("nav-home").click();
  await slider.waitFor({ timeout: 5000 });

  const footer = page.getByTestId("site-footer");
  assert.match(await footer.innerText(), /Turing Testers[\s\S]*SIH26142/);
  assert.ok(await footer.getByRole("link", { name: "About" }).isVisible());
  assert.match(await footer.getByRole("link", { name: /GitHub/ }).getAttribute("href"), /github\.com\//);
  await page.getByTestId("impact").scrollIntoViewIfNeeded();
  await page.waitForTimeout(1600); // count-up finishes
  await page.screenshot({ path: path.join(SHOT_DIR, `frontend_home_impact_${TAG}.png`) });

  // ------------------------------------------------------------ phone ----
  const phone = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const phoneErrors = watchErrors(phone, "phone");
  await phone.goto(url);
  await phone.getByTestId("home-hero").waitFor();
  const overflow = await phone.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  assert.ok(overflow <= 0, `phone: page scrolls horizontally by ${overflow}px`);
  await phone.screenshot({ path: path.join(SHOT_DIR, `frontend_home_phone_${TAG}.png`), fullPage: true });

  const all = [...errors, ...phoneErrors];
  assert.deepEqual(all, [], `browser errors:\n${all.join("\n")}`);
  console.log(`e2e_home (${TAG}): OK. Screenshots in ${path.relative(process.cwd(), SHOT_DIR)}`);
} finally {
  await browser.close();
  await server.close();
}
