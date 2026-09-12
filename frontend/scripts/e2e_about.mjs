/**
 * End-to-end check of the About page in the system Chrome (playwright-core,
 * channel "chrome"): all five sections render, the spec sheet quotes the same
 * served-model latency as the Home page (FACTS), the repo link points at
 * configs/base.yaml's repo_url, and nothing scrolls sideways at phone width.
 *
 *     npm run test:e2e:about
 *
 * Fails loudly on the first failed assertion, page error or console error.
 * Screenshots go to outputs/figures/frontend_about_*.png (gitignored).
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
const SECTIONS = ["about-overview", "about-specs", "about-pipeline", "about-limitations", "about-links"];

const server = await createServer({ root: FRONTEND_ROOT, server: { port: 0, strictPort: false }, logLevel: "error" });
await server.listen();
const url = new URL("/about", server.resolvedUrls.local[0]).href;
const { FACTS } = await server.ssrLoadModule("/src/data/facts.ts");
const cfg = loadFrontendConfig();
const browser = await chromium.launch({ channel: "chrome" });
mkdirSync(SHOT_DIR, { recursive: true });

try {
  for (const [tag, viewport] of [["desktop", { width: 1440, height: 900 }], ["mobile", { width: 390, height: 844 }]]) {
    const page = await browser.newPage({ viewport });
    const errors = [];
    page.on("pageerror", (e) => errors.push(`${tag} pageerror: ${e.message}`));
    page.on("console", (m) => m.type() === "error" && errors.push(`${tag} console.error: ${m.text()}`));
    await page.goto(url);
    for (const id of SECTIONS) await page.getByTestId(id).waitFor();
    assert.equal(await page.title(), "About · DrishtiSR");

    const latency = `${(FACTS.onnxFp32MedianMs / 1000).toFixed(2)} s`;
    assert.ok((await page.getByTestId("spec-deploy").innerText()).includes(latency), `deploy card must quote ${latency}`);
    assert.equal(await page.getByTestId("about-repo-link").getAttribute("href"), cfg.repoUrl);
    // textContent, not innerText: the eyebrow is uppercased by CSS.
    assert.ok((await page.getByTestId("about-overview").textContent()).includes(cfg.teamName));

    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.ok(overflow <= 0, `${tag}: page scrolls sideways by ${overflow}px`);

    await page.evaluate(() => document.querySelectorAll(".reveal-item").forEach((el) => el.setAttribute("data-revealed", "true")));
    await page.waitForTimeout(700);
    await page.screenshot({ path: path.join(SHOT_DIR, `frontend_about_${tag}.png`), fullPage: true });
    assert.deepEqual(errors, []);
    await page.close();
  }
  console.log("e2e_about: all checks passed");
} finally {
  await browser.close();
  await server.close();
}
