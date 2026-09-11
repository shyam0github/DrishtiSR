/**
 * Procedural placeholder imagery for the swipe slider (MOCK_MODE only).
 *
 * Writes three PNGs to frontend/public/placeholders/ (gitignored, regenerated
 * by `npm run dev` / `npm run build`):
 *
 *   hr.png           uint8 RGB, (hr_px, hr_px, 3)             the "2.5 m" layer
 *   lr.png           uint8 RGB, (hr_px/scale, hr_px/scale, 3) the "10 m" layer
 *   uncertainty.png  uint8 RGBA, (hr_px, hr_px, 4)           edge magnitude of hr.png
 *
 * These are 8-bit DISPLAY images of a synthetic scene. They are not
 * reflectance and not a model output. lr.png is the exact scale x scale block
 * mean of hr.png, the same 'area' operator the spectral loss trains on, so the
 * pair shows what a 4x change in GSD looks like and nothing about what the
 * network achieves. uncertainty.png is a Sobel magnitude and stands in for the
 * heteroscedastic sigma; it is NOT sigma. Each image carries a burned-in
 * "PLACEHOLDER" mark, so a screenshot cannot be mistaken for a result.
 *
 * Deterministic: seeded from the top-level `seed` in configs/base.yaml.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";
import { loadFrontendConfig } from "./repo-config.mjs";

const OUT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "public", "placeholders");

// Scene geometry in METRES, converted to pixels at the output GSD, so the scene
// keeps real-world proportions if hr_px or the GSD changes. Artwork constants
// for a synthetic picture, not model hyperparameters.
const SCENE_M = {
  arterialWidth: 15, // a four-lane road
  blockMin: 240,
  blockMax: 380,
  laneWidth: 5, // an old-city gali
  laneSpacingMin: 45,
  laneSpacingMax: 80,
  buildingMin: 10,
  buildingMax: 35,
  treeRadiusMin: 4,
  treeRadiusMax: 9,
  riverWidth: 90,
  riverAmplitude: 150,
};

// ------------------------------------------------------------------ PNG ----

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const typed = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(typed));
  return Buffer.concat([len, typed, crc]);
}

/** pixels: Uint8Array, (H, W, C) row-major, C = 3 (RGB) or 4 (RGBA). */
function encodePng(pixels, w, h, channels) {
  if (pixels.length !== w * h * channels) throw new Error(`encodePng: ${pixels.length} bytes for ${w}x${h}x${channels}`);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = channels === 4 ? 6 : 2; // RGBA : RGB
  const stride = w * channels;
  const raw = Buffer.alloc((stride + 1) * h); // filter byte 0 (None) per row
  for (let y = 0; y < h; y++) raw.set(pixels.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", deflateSync(raw, { level: 9 })),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

// --------------------------------------------------------------- scene -----

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ROOFS = [
  [0.78, 0.76, 0.72], // concrete
  [0.9, 0.89, 0.86], // whitewashed
  [0.62, 0.38, 0.3], // brick / terracotta
  [0.55, 0.55, 0.56], // grey
  [0.3, 0.45, 0.68], // blue tarpaulin
  [0.7, 0.62, 0.48], // sandstone
];

/** Returns Float32Array (n, n, 3), display RGB in [0, 1]. */
function makeCity(n, gsd, rand) {
  const px = (m) => Math.max(1, Math.round(m / gsd));
  const between = (lo, hi) => lo + rand() * (hi - lo);
  const img = new Float32Array(n * n * 3);
  const set = (x, y, c) => {
    if (x < 0 || y < 0 || x >= n || y >= n) return;
    const i = (y * n + x) * 3;
    img[i] = c[0];
    img[i + 1] = c[1];
    img[i + 2] = c[2];
  };
  const rect = (x0, y0, w, h, c) => {
    for (let y = y0; y < y0 + h; y++) for (let x = x0; x < x0 + w; x++) set(x, y, c);
  };
  const disc = (cx, cy, r, c) => {
    for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) if (x * x + y * y <= r * r) set(cx + x, cy + y, c);
  };

  // Dusty ground with per-pixel grain.
  for (let i = 0; i < n * n; i++) {
    const v = 0.56 + 0.1 * (rand() - 0.5);
    img[i * 3] = v * 0.96;
    img[i * 3 + 1] = v * 0.88;
    img[i * 3 + 2] = v * 0.72;
  }

  // Arterial grid positions.
  const grid = () => {
    const out = [];
    for (let p = Math.round(between(0.2, 0.6) * px(SCENE_M.blockMin)); p < n; p += Math.round(between(px(SCENE_M.blockMin), px(SCENE_M.blockMax)))) out.push(p);
    return out;
  };
  const xs = [0, ...grid(), n];
  const ys = [0, ...grid(), n];
  const road = px(SCENE_M.arterialWidth);

  // Blocks: dense, sparse, or park.
  for (let bi = 0; bi + 1 < xs.length; bi++) {
    for (let bj = 0; bj + 1 < ys.length; bj++) {
      const x0 = xs[bi] + road, x1 = xs[bi + 1];
      const y0 = ys[bj] + road, y1 = ys[bj + 1];
      const kind = rand();
      if (kind < 0.15) {
        rect(x0, y0, x1 - x0, y1 - y0, [0.36, 0.5, 0.3]);
        const trees = Math.round(((x1 - x0) * (y1 - y0)) / (px(SCENE_M.treeRadiusMax) * px(SCENE_M.treeRadiusMax) * 6));
        for (let t = 0; t < trees; t++) {
          disc(Math.round(between(x0, x1)), Math.round(between(y0, y1)), Math.round(between(px(SCENE_M.treeRadiusMin), px(SCENE_M.treeRadiusMax))), [0.16, 0.32, 0.15]);
        }
        continue;
      }
      const dense = kind < 0.75;
      const lane = px(SCENE_M.laneWidth);
      let y = y0 + 1;
      while (y < y1 - 2) {
        const rowH = Math.min(y1 - 1 - y, Math.round(between(px(SCENE_M.laneSpacingMin), px(SCENE_M.laneSpacingMax))));
        let x = x0 + 1;
        while (x < x1 - 2) {
          const w = Math.min(x1 - 1 - x, Math.round(between(px(SCENE_M.buildingMin), px(SCENE_M.buildingMax))));
          const h = dense ? rowH - lane : Math.round(rowH * between(0.4, 0.8));
          const roof = ROOFS[Math.floor(rand() * ROOFS.length)];
          const shade = 0.9 + 0.2 * rand();
          rect(x, y, w, h, roof.map((c) => c * shade));
          rect(x + w, y + 1, 1, h, [0.2, 0.2, 0.22]); // shadow, sun in the south-east
          rect(x + 1, y + h, w, 1, [0.2, 0.2, 0.22]);
          x += w + (dense ? 1 + Math.round(rand() * 2) : Math.round(between(3, 10)));
          if (!dense && rand() < 0.3) disc(x, y + Math.round(h / 2), Math.round(between(px(SCENE_M.treeRadiusMin), px(SCENE_M.treeRadiusMax))), [0.18, 0.34, 0.16]);
        }
        rect(x0, y + rowH - lane, x1 - x0, lane, [0.42, 0.41, 0.4]); // gali
        y += rowH;
      }
    }
  }

  // River, drawn before the roads so the arterials cross it as bridges.
  const rw = px(SCENE_M.riverWidth), amp = px(SCENE_M.riverAmplitude);
  for (let x = 0; x < n; x++) {
    const yc = Math.round(n * 0.78 + amp * Math.sin((x / n) * Math.PI * 2 * 1.3));
    for (let dy = -rw / 2 - 3; dy <= rw / 2 + 3; dy++) {
      const bank = Math.abs(dy) > rw / 2;
      set(x, yc + Math.round(dy), bank ? [0.72, 0.68, 0.56] : [0.16, 0.25 + 0.02 * rand(), 0.3]);
    }
  }

  // Arterials with a dashed centre line.
  for (const x of xs.slice(1, -1)) {
    rect(x, 0, road, n, [0.3, 0.3, 0.32]);
    for (let y = 0; y < n; y += 8) rect(x + Math.floor(road / 2), y, 1, 4, [0.92, 0.92, 0.9]);
  }
  for (const y of ys.slice(1, -1)) {
    rect(0, y, n, road, [0.3, 0.3, 0.32]);
    for (let x = 0; x < n; x += 8) rect(x, y + Math.floor(road / 2), 4, 1, [0.92, 0.92, 0.9]);
  }
  return img;
}

/** Exact s x s block mean. Float32Array (n, n, 3) -> (n/s, n/s, 3). */
function blockMean(img, n, s) {
  const m = n / s;
  const out = new Float32Array(m * m * 3);
  for (let Y = 0; Y < m; Y++) {
    for (let X = 0; X < m; X++) {
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let dy = 0; dy < s; dy++) for (let dx = 0; dx < s; dx++) acc += img[((Y * s + dy) * n + X * s + dx) * 3 + c];
        out[(Y * m + X) * 3 + c] = acc / (s * s);
      }
    }
  }
  return out;
}

/**
 * Quantise display RGB in [0, 1] to uint8. The clamp is 8-bit display
 * quantisation of a synthetic picture; no reflectance passes through here.
 */
function toUint8(img) {
  const out = new Uint8Array(img.length);
  for (let i = 0; i < img.length; i++) out[i] = Math.round(Math.min(1, Math.max(0, img[i])) * 255);
  return out;
}

/** Sobel magnitude of luminance, 3x3 box-smoothed, as a yellow-to-red RGBA overlay. */
function edgeOverlay(img, n) {
  const lum = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) lum[i] = 0.2126 * img[i * 3] + 0.7152 * img[i * 3 + 1] + 0.0722 * img[i * 3 + 2];
  const at = (x, y) => lum[Math.min(n - 1, Math.max(0, y)) * n + Math.min(n - 1, Math.max(0, x))];
  const mag = new Float32Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const gx = at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1);
      const gy = at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1);
      mag[y * n + x] = Math.hypot(gx, gy);
    }
  }
  const smooth = new Float32Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let acc = 0, k = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < n && xx >= 0 && xx < n) { acc += mag[yy * n + xx]; k++; }
      }
      smooth[y * n + x] = acc / k;
    }
  }
  const sorted = Float32Array.from(smooth).sort();
  const p99 = sorted[Math.floor(0.99 * (sorted.length - 1))];
  if (!(p99 > 0)) throw new Error("edgeOverlay: the placeholder scene has no edges; the generator is broken");
  const out = new Uint8Array(n * n * 4);
  for (let i = 0; i < n * n; i++) {
    const v = Math.min(1, smooth[i] / p99);
    out[i * 4] = 255;
    out[i * 4 + 1] = Math.round(255 * (1 - v));
    out[i * 4 + 2] = 0;
    out[i * 4 + 3] = Math.round(235 * Math.pow(v, 0.7));
  }
  return out;
}

// ------------------------------------------------------------ watermark ----

const FONT = {
  P: ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
  L: ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
  A: [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
  C: [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
  E: ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
  H: ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
  O: [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
  D: ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
  R: ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
  M: ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
  0: [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
  1: ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
  2: [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
  5: ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
  ".": [".....", ".....", ".....", ".....", ".....", ".##..", ".##.."],
  " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
};

/** Burn `text` into a uint8 (n, n, 3) image: white 5x7 glyphs of `cell` px on a black box. */
function burnText(px, n, text, x0, y0, cell) {
  const w = (text.length * 6 + 1) * cell, h = 9 * cell;
  const put = (x, y, v) => {
    if (x < 0 || y < 0 || x >= n || y >= n) return;
    px.fill(v, (y * n + x) * 3, (y * n + x) * 3 + 3);
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) put(x0 + x, y0 + y, 0);
  [...text].forEach((ch, k) => {
    const glyph = FONT[ch];
    if (!glyph) throw new Error(`burnText: no glyph for '${ch}'`);
    glyph.forEach((row, gy) => [...row].forEach((bit, gx) => {
      if (bit !== "#") return;
      for (let dy = 0; dy < cell; dy++) for (let dx = 0; dx < cell; dx++) put(x0 + (1 + k * 6 + gx) * cell + dx, y0 + (1 + gy) * cell + dy, 255);
    }));
  });
}

function fmtGsd(m) {
  return `${Number.isInteger(m) ? m : m.toFixed(1)} M`;
}

// ----------------------------------------------------------------- main ----

const cfg = loadFrontendConfig();
const n = cfg.placeholder.hrPx;
const s = cfg.scale;
const hrGsd = cfg.lrGsdM / s;
const started = performance.now();

const hr = makeCity(n, hrGsd, mulberry32(cfg.placeholder.seed));
const lr = blockMean(hr, n, s);
const hrPx = toUint8(hr);
const lrPx = toUint8(lr);
// The same on-the-ground size in both images: `s` HR px per glyph cell, 1 LR px per cell.
burnText(hrPx, n, `PLACEHOLDER ${fmtGsd(hrGsd)}`, 2 * s, 2 * s, s);
burnText(lrPx, n / s, `PLACEHOLDER ${fmtGsd(cfg.lrGsdM)}`, 2, 2, 1);

mkdirSync(OUT_DIR, { recursive: true });
writeFileSync(path.join(OUT_DIR, "hr.png"), encodePng(hrPx, n, n, 3));
writeFileSync(path.join(OUT_DIR, "lr.png"), encodePng(lrPx, n / s, n / s, 3));
writeFileSync(path.join(OUT_DIR, "uncertainty.png"), encodePng(edgeOverlay(hr, n), n, n, 4));

console.log(
  `placeholders: ${n}px @ ${hrGsd} m and ${n / s}px @ ${cfg.lrGsdM} m (seed ${cfg.placeholder.seed}) -> ` +
    `${path.relative(process.cwd(), OUT_DIR)} in ${Math.round(performance.now() - started)} ms`,
);
