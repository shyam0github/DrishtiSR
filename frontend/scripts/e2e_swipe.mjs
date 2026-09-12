/**
 * End-to-end check of the swipe slider, the AI badge, the uncertainty layer
 * and the AOI picker in a real Chrome. It drives a mouse on a desktop viewport
 * and a finger (CDP touch events) on a phone viewport.
 *
 * Starts the Vite dev server in-process and drives the SYSTEM Chrome through
 * playwright-core (channel "chrome"), so no browser is downloaded. Fails loudly:
 * the first failed assertion, page error or console error exits non-zero.
 * Screenshots go to outputs/figures/frontend_e2e_{desktop,phone}.png
 * (gitignored).
 *
 *     npm run test:e2e
 */
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";
import { createServer } from "vite";
import { loadFrontendConfig, REPO_ROOT } from "./repo-config.mjs";

const FRONTEND_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SHOT_DIR = path.join(REPO_ROOT, "outputs", "figures");
const BADGE = "AI-RECONSTRUCTED — NOT MEASURED DATA";
/** Divider position tolerance after a drag, as a fraction of the width. */
const DRAG_TOL = 0.02;

const cfg = loadFrontendConfig();
const tileHosts = cfg.map.basemap.tiles.map((t) => new URL(t).host);

function watchErrors(page, label) {
  const errors = [];
  page.on("pageerror", (e) => errors.push(`${label} pageerror: ${e.message}`));
  page.on("console", (msg) => {
    if (msg.type() !== "error") return;
    const where = msg.location()?.url ?? "";
    const text = msg.text();
    // A basemap tile that fails to load is the network, not this code.
    if (tileHosts.some((h) => where.includes(h) || text.includes(h))) return;
    // The one health probe to an absent backend: the browser logs it, the badge shows it.
    if (where.endsWith("/health")) return;
    errors.push(`${label} console.error: ${text} @ ${where}`);
  });
  return errors;
}

const fraction = async (page) => Number(await page.getByTestId("swipe").getAttribute("data-fraction"));
const center = (page) => page.evaluate(() => window.__drishti.baseMap.getCenter().toArray());
const box = async (locator) => {
  const b = await locator.boundingBox();
  assert.ok(b, "element has no bounding box");
  return b;
};
const middle = (b) => ({ x: b.x + b.width / 2, y: b.y + b.height / 2 });

async function sceneReady(page) {
  await page.locator('[data-testid="swipe"][data-sr-loaded="true"]').waitFor({ timeout: 30_000 });
  await page.waitForFunction(() => window.__drishti?.baseMap && !window.__drishti.baseMap.isMoving(), null, { timeout: 30_000 });
}

async function checkBadge(page) {
  const badge = page.getByTestId("ai-badge");
  assert.equal((await badge.textContent())?.trim(), BADGE);
  assert.ok(await badge.isVisible(), "AI badge is not visible");
  assert.equal(await badge.locator("button, a, [role=button]").count(), 0, "the AI badge must have no controls");
}

/** Luminance std (0-255) below which a region counts as blank. */
const MIN_STD = 8;
/** Required ratio of fine detail (mean |d lum / dx|), 2.5 m side over 10 m side. */
const DETAIL_RATIO = 1.5;

/**
 * Measure what the map actually PAINTED. A DOM attribute can report "loaded"
 * over a blank canvas, and before the maplibre-gl worker fix in vite.config.ts
 * it did. This screenshots the swipe region and, inside the page, measures
 * luminance texture in a band left and a band right of the divider.
 */
async function paintedDetail(page) {
  const swipe = await box(page.getByTestId("swipe"));
  const png = await page.screenshot({ clip: swipe });
  return page.evaluate(
    async ({ b64, f }) => {
      const img = new Image();
      img.src = `data:image/png;base64,${b64}`;
      await img.decode();
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const lum = (x, y) => {
        const i = (y * width + x) * 4;
        return 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
      };
      // Middle 40 % of the height: clear of the badge (top) and the side labels (bottom).
      const region = (x0, x1) => {
        let sum = 0, sum2 = 0, grad = 0, n = 0;
        for (let y = Math.round(0.3 * height); y < Math.round(0.7 * height); y++) {
          for (let x = Math.round(x0 * width); x < Math.round(x1 * width) - 1; x++) {
            const v = lum(x, y);
            sum += v;
            sum2 += v * v;
            grad += Math.abs(lum(x + 1, y) - v);
            n++;
          }
        }
        const mean = sum / n;
        return { std: Math.sqrt(sum2 / n - mean * mean), grad: grad / n };
      };
      return { left: region(f - 0.3, f - 0.05), right: region(f + 0.05, f + 0.3) };
    },
    { b64: png.toString("base64"), f: await fraction(page) },
  );
}

async function desktop(browser, url) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = watchErrors(page, "desktop");
  const images = ["hr.png", "lr.png", "uncertainty.png"].map((f) =>
    page.waitForResponse((r) => r.url().endsWith(`/placeholders/${f}`) && r.status() === 200, { timeout: 30_000 }),
  );
  await page.goto(url);
  await Promise.all(images);
  await sceneReady(page);
  await checkBadge(page);
  assert.equal(await fraction(page), 0.5, "the divider should start at 50%");
  const detail = await paintedDetail(page);
  const fmt = (r) => `std ${r.std.toFixed(1)}, detail ${r.grad.toFixed(2)}`;
  console.log(`painted: left (10 m) ${fmt(detail.left)} | right (2.5 m) ${fmt(detail.right)} | ratio ${(detail.right.grad / detail.left.grad).toFixed(2)}`);
  assert.ok(detail.left.std > MIN_STD && detail.right.std > MIN_STD, `the map painted a blank region: ${JSON.stringify(detail)}`);
  assert.ok(detail.right.grad > DETAIL_RATIO * detail.left.grad, `the 2.5 m side is not visibly sharper than the 10 m side: ${JSON.stringify(detail)}`);
  mkdirSync(SHOT_DIR, { recursive: true });
  await page.screenshot({ path: path.join(SHOT_DIR, "frontend_e2e_desktop.png") });

  // Mouse: drag the divider to 25 %. The map must not pan underneath.
  const before = await center(page);
  const swipe = await box(page.getByTestId("swipe"));
  const knob = middle(await box(page.getByTestId("swipe-knob")));
  await page.mouse.move(knob.x, knob.y);
  await page.mouse.down();
  await page.mouse.move(swipe.x + swipe.width * 0.25, knob.y, { steps: 10 });
  await page.mouse.up();
  const f = await fraction(page);
  assert.ok(Math.abs(f - 0.25) < DRAG_TOL, `mouse drag: divider at ${f}, expected 0.25`);
  assert.deepEqual(await center(page), before, "dragging the divider panned the map");
  const clip = await page.getByTestId("swipe-overlay").evaluate((el) => el.style.clipPath);
  const clipPct = Number(clip.match(/([\d.]+)%\)$/)?.[1]);
  assert.ok(Math.abs(clipPct / 100 - f) < 1e-3, `overlay clip-path '${clip}' does not follow the divider (${f})`);

  // Keyboard.
  await page.getByTestId("swipe-handle").focus();
  await page.keyboard.press("ArrowRight");
  assert.ok(Math.abs((await fraction(page)) - (f + 0.02)) < 1e-3, "ArrowRight should move the divider by 2%");

  // Uncertainty layer: off by default, toggled on, faded.
  const opacity = () => page.evaluate(() => window.__drishti.overlayMap.getPaintProperty("uncertainty", "raster-opacity"));
  assert.equal(await opacity(), 0);
  await page.getByTestId("unc-toggle").check();
  assert.ok((await opacity()) > 0, "toggling uncertainty on left it invisible");
  await page.getByTestId("unc-opacity").fill("0.3");
  assert.equal(await opacity(), 0.3);

  // AOI by mouse: a small rectangle is in budget and runs.
  const mapCenter = await center(page);
  await page.getByTestId("aoi-draw").click();
  await page.mouse.move(swipe.x + swipe.width * 0.55, swipe.y + swipe.height * 0.4);
  await page.mouse.down();
  await page.mouse.move(swipe.x + swipe.width * 0.7, swipe.y + swipe.height * 0.55, { steps: 6 });
  await page.mouse.up();
  assert.deepEqual(await center(page), mapCenter, "drawing the AOI panned the map");
  assert.match(await page.getByTestId("aoi-bbox").innerText(), /EPSG:4326/);
  assert.equal(await page.getByTestId("aoi-over-budget").count(), 0, "a small AOI was flagged over budget");
  assert.ok(await page.getByTestId("aoi-run").isEnabled(), `Run is disabled for an in-budget AOI. Panel:\n${await page.getByTestId("aoi").innerText()}`);
  // Wait for a NEW result (the opening run already shows provenance), then for
  // the camera to finish fitting to it, or the next step races the fit.
  const previousId = await page.getByTestId("provenance").getAttribute("data-request-id");
  await page.getByTestId("aoi-run").click();
  await page.waitForFunction(
    (prev) => document.querySelector('[data-testid="provenance"]')?.getAttribute("data-request-id") !== prev,
    previousId,
    { timeout: 30_000 },
  );
  assert.match(await page.getByTestId("provenance").innerText(), /MOCK_MODE/);
  await sceneReady(page);

  // AOI over budget: zoom out to a whole-state view and drag across it.
  await page.evaluate(() => window.__drishti.baseMap.jumpTo({ zoom: 6 }));
  await page.getByTestId("aoi-draw").click();
  await page.mouse.move(swipe.x + swipe.width * 0.35, swipe.y + swipe.height * 0.15);
  await page.mouse.down();
  await page.mouse.move(swipe.x + swipe.width * 0.9, swipe.y + swipe.height * 0.85, { steps: 6 });
  await page.mouse.up();
  await page.getByTestId("aoi-over-budget").waitFor();
  assert.ok(await page.getByTestId("aoi-run").isDisabled(), "Run must be disabled for an over-budget AOI");

  // Metrics fixture rendered.
  const metrics = page.getByTestId("metrics");
  await metrics.getByRole("rowheader", { name: "Run B (B1)" }).waitFor();
  assert.match(await metrics.innerText(), /38\.765/);

  assert.deepEqual(errors, []);
  await page.close();
}

async function phone(browser, url) {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const page = await context.newPage();
  const errors = watchErrors(page, "phone");
  await page.goto(url);
  await sceneReady(page);
  await checkBadge(page);

  const cdp = await context.newCDPSession(page);
  const touch = (type, points) => cdp.send("Input.dispatchTouchEvent", { type, touchPoints: points });
  const touchDrag = async (from, to, steps = 10) => {
    await touch("touchStart", [from]);
    for (let i = 1; i <= steps; i++) {
      await touch("touchMove", [{ x: from.x + ((to.x - from.x) * i) / steps, y: from.y + ((to.y - from.y) * i) / steps }]);
    }
    await touch("touchEnd", []);
  };

  // Finger: drag the divider to 75 %. The map must not pan.
  const before = await center(page);
  const swipe = await box(page.getByTestId("swipe"));
  const knob = middle(await box(page.getByTestId("swipe-knob")));
  await touchDrag(knob, { x: swipe.x + swipe.width * 0.75, y: knob.y });
  const f = await fraction(page);
  assert.ok(Math.abs(f - 0.75) < DRAG_TOL, `touch drag: divider at ${f}, expected 0.75`);
  assert.deepEqual(await center(page), before, "dragging the divider by touch panned the map");
  await page.screenshot({ path: path.join(SHOT_DIR, "frontend_e2e_phone.png") });

  // Finger: draw an AOI left of the divider.
  await page.getByTestId("aoi-draw").tap();
  await touchDrag(
    { x: swipe.x + swipe.width * 0.15, y: swipe.y + swipe.height * 0.3 },
    { x: swipe.x + swipe.width * 0.5, y: swipe.y + swipe.height * 0.55 },
  );
  assert.deepEqual(await center(page), before, "drawing the AOI by touch panned the map");
  assert.equal(await page.getByTestId("aoi-draw").innerText(), "Draw AOI", "drawing mode should end after a touch drag");
  assert.match(await page.getByTestId("aoi-tiles").innerText(), /of \d+ allowed/);

  assert.deepEqual(errors, []);
  await context.close();
}

const server = await createServer({ root: FRONTEND_ROOT, logLevel: "warn", server: { strictPort: false } });
await server.listen();
const root = server.resolvedUrls?.local[0];
assert.ok(root, "vite did not report a local URL");
// The map demo lives at /demo since the router landed; / is the marketing shell.
const url = new URL("demo", root).href;
const browser = await chromium.launch({ channel: "chrome", headless: true });
try {
  await desktop(browser, url);
  console.log("e2e desktop (mouse): OK");
  await phone(browser, url);
  console.log("e2e phone (touch): OK");
} finally {
  await browser.close();
  await server.close();
}
console.log(`screenshots: ${path.relative(process.cwd(), SHOT_DIR)}/frontend_e2e_{desktop,phone}.png`);
