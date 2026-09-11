// DrishtiSR demo UI. Talks only to API contract v1 (docs/mvp/api_contract.md).
// ?mock=1 serves app/fixtures instead of the API. Demo deep links: &sample=<id> &tta=0|4|8
// &overlay=uncertainty|consistency &autorun=1 (runs the selected sample on load).

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const MOCK = params.get("mock") === "1";

const state = {
  samples: [],
  sample: null,      // selected sample object
  file: null,        // selected File
  result: null,      // last /api/upscale response
  band: "rgb",       // rgb | fcc
  left: "lr",        // lr | bicubic | hr
  overlay: "none",   // none | uncertainty | consistency
  opacity: 0.65,
  split: 50,         // % from the left
  view: { s: 1, x: 0, y: 0 },
  running: false,
};

const LEFT_LABELS = { lr: "Sentinel-2 10 m input (nearest)", bicubic: "Bicubic ×4 baseline", hr: "NAIP 2.5 m reference (HR)" };
const BAND_LABELS = { rgb: "RGB", fcc: "False colour (NIR·R·G)" };
const UNC_TITLES = { tta4: "Uncertainty (TTA disagreement)", tta8: "Uncertainty (TTA disagreement)", learned_laplace: "Uncertainty (learned σ)" };
const MINUS = "−";

// ───────────────────────── helpers ─────────────────────────
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const signed = (v, d) => (v > 0 ? "+" : v < 0 ? MINUS : "±") + Math.abs(v).toFixed(d);
const fmt = (v, d) => (v == null || !isFinite(v) ? "–" : Number(v).toFixed(d));
const fmtSig = (v) => (v === 0 ? "0" : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2));
const fmtMs = (v) => (v == null ? "–" : v >= 1000 ? (v / 1000).toFixed(2) + " s" : Math.round(v) + " ms");
const fmtParams = (n) => (n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : Math.round(n / 1000) + "k");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function fetchJSON(url, init) {
  let res;
  try {
    res = await fetch(url, init);
  } catch (e) {
    throw new Error(`Cannot reach the API (${url}).`);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(body && body.error ? body.error : `HTTP ${res.status} from ${url}`);
  return body;
}

// ───────────────────────── API layer ─────────────────────────
const api = MOCK
  ? {
      health: () => fetchJSON("/fixtures/health.json"),
      samples: () => fetchJSON("/fixtures/samples.json"),
      upscale: mockUpscale,
    }
  : {
      health: () => fetchJSON("/api/health"),
      samples: () => fetchJSON("/api/samples"),
      upscale: (form) => fetchJSON("/api/upscale", { method: "POST", body: form }),
    };

// Client-side variations of the one fixture response, so both metric layouts and the
// TTA-off state can be exercised offline. Shapes stay exactly as in the contract.
async function mockUpscale(form) {
  await sleep(900);
  const res = structuredClone(await fetchJSON("/fixtures/sample_response.json"));
  const tta = form.get("tta");
  const dn = form.get("dn_mode");
  const file = form.get("file");
  const sample = state.samples.find((s) => s.id === form.get("sample_id"));
  const noGt = file instanceof File || (sample && !sample.has_gt);
  if (file instanceof File) Object.assign(res.input, { source: "upload", sample_id: null });
  else if (sample) {
    res.input.sample_id = sample.id;
    res.input.lr_size = sample.lr_size;
    res.input.sr_size = sample.lr_size.map((v) => v * 4);
  }
  if (noGt) {
    res.metrics.with_gt = null;
    res.images.hr_rgb = null;
    res.images.hr_fcc = null;
  }
  if (tta === "0") {
    res.uncertainty_method = "none";
    res.images.uncertainty = null;
    res.downloads.uncertainty_tif = null;
    Object.assign(res.metrics.reference_free, { unc_mean: null, unc_p95: null });
    res.metrics.reference_free.runtime_ms = { sr: 180.0, uncertainty: null, total: 240.0 };
  } else if (tta === "8") res.uncertainty_method = "tta8";
  res.input.dn_mode_applied = dn === "auto" ? res.input.dn_mode_applied : dn;
  res.warnings.push("MOCK MODE: response derived in the browser from app/fixtures/sample_response.json; no model was run.");
  return res;
}

// ───────────────────────── header ─────────────────────────
function setStatus(kind, text) {
  const el = $("backend-status");
  el.className = `status status-${kind}`;
  el.textContent = text;
}

function renderModel(m) {
  if (!m) return;
  $("model-backend").textContent = m.backend;
  $("model-params").textContent = `${fmtParams(m.params)} params`;
  $("model-size").textContent = `${Math.round(m.model_bytes / 1000).toLocaleString("en-US")} KB`;
  $("model-threads").textContent = `${m.threads} threads`;
  $("model-badge").title = `checkpoint ${m.checkpoint_id} · ${m.params.toLocaleString("en-US")} parameters · ${m.model_bytes.toLocaleString("en-US")} bytes${m.has_scale_head ? " · learned σ head" : ""}`;
  $("interim-chip").hidden = !m.interim;
}

// ───────────────────────── input panel ─────────────────────────
function renderSamples() {
  const list = $("sample-list");
  if (!state.samples.length) {
    list.innerHTML = '<p class="muted small">No samples available.</p>';
    return;
  }
  list.innerHTML = state.samples
    .map(
      (s) => `<button type="button" class="sample" data-id="${esc(s.id)}" title="${esc(s.id)}">
        <img src="${esc(s.thumb_url)}" alt="" width="48" height="48">
        <span class="s-label">${esc(s.label)}</span>
        <span class="s-meta">${s.lr_size[0]}×${s.lr_size[1]} px · ${s.has_gt ? "with 2.5 m reference" : "no reference"}</span>
      </button>`
    )
    .join("");
  syncSelection();
}

function selectSample(id) {
  state.sample = state.samples.find((s) => s.id === id) || null;
  state.file = null;
  $("file-input").value = "";
  syncSelection();
}

function selectFile(file) {
  if (!file) return;
  if (!/\.tiff?$/i.test(file.name)) {
    showError(`“${file.name}” is not a .tif/.tiff GeoTIFF.`);
    return;
  }
  hideError();
  state.file = file;
  state.sample = null;
  syncSelection();
}

function syncSelection() {
  for (const b of $("sample-list").querySelectorAll(".sample")) {
    b.classList.toggle("selected", !!state.sample && b.dataset.id === state.sample.id);
  }
  $("drop-zone").classList.toggle("selected", !!state.file);
  $("file-name").textContent = state.file ? `${state.file.name} · ${(state.file.size / 1e6).toFixed(2)} MB` : "";
}

function showError(msg) {
  const box = $("error-box");
  box.textContent = msg;
  box.hidden = false;
}
function hideError() {
  $("error-box").hidden = true;
}

let timer = null;
function setRunning(on) {
  state.running = on;
  $("run-btn").disabled = on;
  const st = $("run-status");
  clearInterval(timer);
  if (on) {
    const t0 = performance.now();
    st.classList.add("running");
    const tick = () => (st.textContent = `Running… ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    tick();
    timer = setInterval(tick, 100);
    return t0;
  }
  st.classList.remove("running");
  return null;
}

async function run() {
  if (state.running) return;
  if (!state.sample && !state.file) {
    showError("Pick a sample or drop a GeoTIFF first.");
    return;
  }
  hideError();
  const form = new FormData();
  if (state.file) form.append("file", state.file);
  else form.append("sample_id", state.sample.id);
  form.append("tta", $("tta-select").value);
  form.append("dn_mode", $("dn-select").value);
  const t0 = setRunning(true);
  try {
    const res = await api.upscale(form);
    setRunning(false);
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    $("run-status").textContent = `Done in ${secs} s · server total ${fmtMs(res.metrics.reference_free.runtime_ms.total)}`;
    state.result = res;
    renderResult();
  } catch (e) {
    setRunning(false);
    $("run-status").textContent = "";
    showError(e.message || String(e));
  }
}

// ───────────────────────── viewer ─────────────────────────
function renderViewer() {
  const r = state.result;
  $("empty-state").hidden = !!r;
  if (!r) return;
  const im = r.images;
  const hasHr = !!(im.hr_rgb && im.hr_fcc);
  const hrOpt = $("left-layer").querySelector('option[value="hr"]');
  hrOpt.disabled = !hasHr;
  hrOpt.textContent = hasHr ? "HR 2.5 m reference" : "HR 2.5 m reference (n/a)";
  if (state.left === "hr" && !hasHr) state.left = "bicubic";
  $("left-layer").value = state.left;

  const leftImg = $("left-img");
  leftImg.src = im[`${state.left}_${state.band}`];
  leftImg.classList.toggle("lr", state.left === "lr");
  $("right-img").src = im[`sr_${state.band}`];
  $("left-label").textContent = `${LEFT_LABELS[state.left]} · ${BAND_LABELS[state.band]}`;
  $("mode-rgb").classList.toggle("active", state.band === "rgb");
  $("mode-fcc").classList.toggle("active", state.band === "fcc");
  renderOverlay();
}

function overlayAvailable(kind) {
  const r = state.result;
  if (!r) return kind === "none";
  if (kind === "uncertainty") return !!r.images.uncertainty;
  if (kind === "consistency") return !!r.images.consistency;
  return true;
}

function renderOverlay() {
  const r = state.result;
  const sel = $("overlay-select");
  for (const opt of sel.options) opt.disabled = !overlayAvailable(opt.value);
  const uncOpt = sel.querySelector('option[value="uncertainty"]');
  uncOpt.textContent = r && !r.images.uncertainty ? "Uncertainty (off — TTA 0)" : "Uncertainty";
  if (!overlayAvailable(state.overlay)) state.overlay = "none";
  sel.value = state.overlay;

  const img = $("overlay-img");
  const legend = $("legend");
  if (!r || state.overlay === "none") {
    img.hidden = true;
    legend.hidden = true;
    return;
  }
  img.src = r.images[state.overlay];
  img.style.opacity = state.opacity;
  img.hidden = false;
  const max = state.overlay === "uncertainty" ? r.refs.unc_display_max : r.refs.cons_display_max;
  $("legend-title").textContent =
    state.overlay === "uncertainty"
      ? UNC_TITLES[r.uncertainty_method] || "Uncertainty"
      : "Spectral consistency error (training operator)";
  $("legend-bar").className = `legend-bar ramp-${state.overlay}`;
  $("legend-ticks").innerHTML = [0, 0.5, 1]
    .map((f) => `<span style="left:${f * 100}%">${f === 1 ? "≥ " : ""}${fmtSig(max * f)}</span>`)
    .join("");
  $("legend-ticks").title = "reflectance units, fixed scale (not per-image min–max)";
  legend.hidden = false;
}

function cycleOverlay() {
  const order = ["none", "uncertainty", "consistency"];
  let i = order.indexOf(state.overlay);
  for (let k = 0; k < order.length; k++) {
    i = (i + 1) % order.length;
    if (overlayAvailable(order[i])) break;
  }
  state.overlay = order[i];
  renderOverlay();
}

function setSplit(pct) {
  state.split = clamp(pct, 0, 100);
  $("right-pane").style.clipPath = `inset(0 0 0 ${state.split}%)`;
  const h = $("slider-handle");
  h.style.left = `${state.split}%`;
  h.setAttribute("aria-valuenow", String(Math.round(state.split)));
}

function applyView() {
  const { s, x, y } = state.view;
  const t = `translate(${x}px, ${y}px) scale(${s})`;
  $("left-inner").style.transform = t;
  $("right-inner").style.transform = t;
  $("stage").classList.toggle("zoomed", s > 1.001);
  $("zoom-readout").textContent = `${s.toFixed(1)}×`;
}

function clampView() {
  const { width: w, height: h } = $("stage").getBoundingClientRect();
  const v = state.view;
  v.x = clamp(v.x, w * (1 - v.s), 0);
  v.y = clamp(v.y, h * (1 - v.s), 0);
}

function resetView() {
  state.view = { s: 1, x: 0, y: 0 };
  applyView();
}

function initViewer() {
  const stage = $("stage");
  stage.addEventListener(
    "wheel",
    (e) => {
      if (!state.result) return;
      e.preventDefault();
      const rect = stage.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const v = state.view;
      const ns = clamp(v.s * Math.exp(-e.deltaY * 0.0015), 1, 32);
      v.x = px - (px - v.x) * (ns / v.s);
      v.y = py - (py - v.y) * (ns / v.s);
      v.s = ns;
      clampView();
      applyView();
    },
    { passive: false }
  );

  let drag = null; // {kind:'pan'|'split', x0, y0, vx, vy}
  stage.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || !state.result) return;
    const onHandle = e.target.closest("#slider-handle");
    drag = { kind: onHandle ? "split" : "pan", x0: e.clientX, y0: e.clientY, vx: state.view.x, vy: state.view.y };
    stage.setPointerCapture(e.pointerId);
    if (drag.kind === "pan") stage.classList.add("panning");
  });
  stage.addEventListener("pointermove", (e) => {
    if (!drag) return;
    if (drag.kind === "split") {
      const rect = stage.getBoundingClientRect();
      setSplit(((e.clientX - rect.left) / rect.width) * 100);
    } else {
      state.view.x = drag.vx + (e.clientX - drag.x0);
      state.view.y = drag.vy + (e.clientY - drag.y0);
      clampView();
      applyView();
    }
  });
  const end = () => {
    drag = null;
    stage.classList.remove("panning");
  };
  stage.addEventListener("pointerup", end);
  stage.addEventListener("pointercancel", end);
  window.addEventListener("resize", () => {
    clampView();
    applyView();
  });

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = e.target.tagName;
    if (tag === "SELECT" || tag === "TEXTAREA" || (tag === "INPUT" && e.target.type !== "range")) return;
    if (tag === "INPUT" && (e.key === "ArrowLeft" || e.key === "ArrowRight")) return; // opacity slider
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      const step = e.shiftKey ? 10 : 2;
      setSplit(state.split + (e.key === "ArrowLeft" ? -step : step));
    } else if (e.key === "o" || e.key === "O") {
      cycleOverlay();
    }
  });
}

// ───────────────────────── metrics ─────────────────────────
// Horizontal scale with markers. markers: [{v, label, kind: 'sr'|'bic'|'hr'|'floor', pos: 'above'|'below'}]
function scaleHTML({ min, max, log = false, markers, zone }) {
  const f = (v) => {
    const t = log ? (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min)) : (v - min) / (max - min);
    return clamp(t, 0, 1) * 100;
  };
  const zoneHTML = zone ? `<div class="zone" style="left:${f(zone[0])}%;width:${f(zone[1]) - f(zone[0])}%" title="below the sharpness guard"></div>` : "";
  const mk = markers
    .map((m) => {
      const p = f(m.v);
      const align = p < 20 ? " al" : p > 70 ? " ar" : "";
      return `<div class="mk ${m.kind} ${m.pos || "below"}${align}" style="left:${p}%"><span>${esc(m.label)}</span></div>`;
    })
    .join("");
  return `<div class="scale"><div class="track"></div>${zoneHTML}${mk}</div>`;
}

function lpipsCard(g) {
  const sr = g.sr.lpips;
  const bic = g.bicubic.lpips;
  const rel = ((sr - bic) / bic) * 100;
  const max = Math.max(sr, bic) * 1.1;
  const bar = (label, v, cls) =>
    `<div class="bar-row"><span>${label}</span><div><div class="bar ${cls}" style="width:${(v / max) * 100}%"></div></div><span class="kv">${fmt(v, 3)}</span></div>`;
  return `<div class="card" id-card="lpips">
    <div class="card-head"><span class="card-title">LPIPS · perceptual distance to HR</span><span class="card-sub">lower is better</span></div>
    <div class="row"><span class="big">${fmt(sr, 3)}</span>
      <span class="kv delta ${rel <= 0 ? "good" : "bad"}">${signed(rel, 1)} % vs bicubic</span></div>
    <div class="bars">${bar("SR", sr, "sr")}${bar("Bicubic", bic, "")}</div>
  </div>`;
}

function consistencyCard(rf, g, refs) {
  const hrL1 = g ? g.spec_l1_hr : refs.spec_l1_gt_floor;
  const hrSam = g ? g.spec_sam_hr_deg : refs.spec_sam_gt_floor_deg;
  const hrName = g ? "HR" : "floor";
  const l1Max = Math.max(rf.spec_l1, hrL1, rf.spec_l1_bicubic) * 1.15;
  const samMax = Math.max(rf.spec_sam_deg, hrSam, rf.spec_sam_bicubic_deg) * 1.15;
  const l1 = scaleHTML({
    min: 0, max: l1Max,
    markers: [
      { v: rf.spec_l1_bicubic, label: `bicubic ${fmt(rf.spec_l1_bicubic, 4)}`, kind: "bic" },
      { v: hrL1, label: `${hrName} ${fmt(hrL1, 4)}`, kind: g ? "hr" : "floor" },
      { v: rf.spec_l1, label: `SR ${fmt(rf.spec_l1, 4)}`, kind: "sr", pos: "above" },
    ],
  });
  const sam = scaleHTML({
    min: 0, max: samMax,
    markers: [
      { v: rf.spec_sam_bicubic_deg, label: `bicubic ${fmt(rf.spec_sam_bicubic_deg, 2)}°`, kind: "bic" },
      { v: hrSam, label: `${hrName} ${fmt(hrSam, 2)}°`, kind: g ? "hr" : "floor" },
      { v: rf.spec_sam_deg, label: `SR ${fmt(rf.spec_sam_deg, 2)}°`, kind: "sr", pos: "above" },
    ],
  });
  return `<div class="card" id-card="consistency">
    <div class="card-head"><span class="card-title">Spectral consistency</span><span class="card-sub">degrade(SR) vs 10 m input · training operator</span></div>
    <div class="scale-title"><span>L1 (reflectance)</span><span>bicubic ≈ 0</span></div>${l1}
    <div class="scale-title"><span>SAM (degrees)</span></div>${sam}
    <p class="caption">HR itself sits at the floor — near zero can mean blur.${g ? "" : " “floor” = HR’s value on the full VAL split (no reference for this image)."}</p>
  </div>`;
}

function sharpnessCard(rf, g, refs) {
  const sr = rf.hf_ratio_vs_bicubic;
  const hr = g ? g.hf_ratio_hr_vs_bicubic : null;
  const max = Math.max(sr, hr || 0, refs.sharpness_warn_below) * (hr ? 1.3 : 1.6);
  const min = Math.min(0.8, sr * 0.9);
  const markers = [
    { v: 1.0, label: "bicubic 1.00", kind: "bic" },
    { v: sr, label: `SR ${fmt(sr, 2)}×`, kind: "sr", pos: "above" },
  ];
  if (hr) markers.push({ v: hr, label: `HR ${fmt(hr, 2)}×`, kind: "hr" });
  return `<div class="card" id-card="sharpness">
    <div class="card-head"><span class="card-title">Sharpness · HF energy vs bicubic</span><span class="card-sub">higher = more detail · log scale</span></div>
    ${scaleHTML({ min, max, log: true, markers, zone: [min, refs.sharpness_warn_below] })}
    <p class="caption">Shaded: below the blur guard (${fmt(refs.sharpness_warn_below, 2)}×).</p>
  </div>`;
}

function gtTable(g) {
  const rows = [
    ["SSIM", "ssim", 1, 4],
    ["SAM °", "sam_deg", -1, 3],
    ["ERGAS", "ergas", -1, 3],
    ["PSNR dB", "psnr", 1, 2],
  ];
  const body = rows
    .map(([name, k, dir, d]) => {
      const delta = g.sr[k] - g.bicubic[k];
      const cls = delta * dir > 0 ? "good" : delta * dir < 0 ? "bad" : "";
      return `<tr><td>${name}<span class="dir">${dir > 0 ? "↑" : "↓"}</span></td><td>${fmt(g.sr[k], d)}</td><td>${fmt(g.bicubic[k], d)}</td><td class="delta ${cls}">${signed(delta, d)}</td></tr>`;
    })
    .join("");
  return `<div class="card"><table class="metrics">
    <thead><tr><th>vs HR</th><th>SR</th><th>Bicubic</th><th>Δ</th></tr></thead><tbody>${body}</tbody></table></div>`;
}

function uncertaintyCard(r) {
  const rf = r.metrics.reference_free;
  if (rf.unc_mean == null) {
    return `<div class="card"><div class="card-head"><span class="card-title">Uncertainty</span><span class="card-sub">method: ${esc(r.uncertainty_method)}</span></div>
      <p class="caption">Not computed — set TTA to 4 or 8.</p></div>`;
  }
  const max = r.refs.unc_display_max;
  const bar = (label, v) =>
    `<div class="bar-row"><span>${label}</span><div><div class="bar sr" style="width:${clamp(v / max, 0, 1) * 100}%"></div></div><span class="kv">${fmt(v, 4)}</span></div>`;
  return `<div class="card" id-card="uncertainty">
    <div class="card-head"><span class="card-title">${esc(UNC_TITLES[r.uncertainty_method] || "Uncertainty")}</span><span class="card-sub">reflectance · scale 0–${fmtSig(max)}</span></div>
    <div class="bars">${bar("mean", rf.unc_mean)}${bar("p95", rf.unc_p95)}</div>
  </div>`;
}

function runtimeCard(r) {
  const t = r.metrics.reference_free.runtime_ms;
  return `<div class="card" id-card="runtime">
    <div class="card-head"><span class="card-title">Runtime</span><span class="card-sub">${esc(r.model.backend)} · ${r.model.threads} CPU threads</span></div>
    <div class="row"><span class="big">${fmtMs(t.total)}</span><span class="muted">total</span></div>
    <p class="foot-line">SR pass <b>${fmtMs(t.sr)}</b> · uncertainty <b>${fmtMs(t.uncertainty)}</b></p>
  </div>`;
}

function footLines(r) {
  const rf = r.metrics.reference_free;
  const t = rf.runtime_ms;
  const unc = rf.unc_mean == null ? "off" : `mean <b>${fmt(rf.unc_mean, 4)}</b> · p95 <b>${fmt(rf.unc_p95, 4)}</b>`;
  return `<p class="foot-line" id-card="runtime">Runtime: SR <b>${fmtMs(t.sr)}</b> · uncertainty <b>${fmtMs(t.uncertainty)}</b> · total <b>${fmtMs(t.total)}</b></p>
    <p class="foot-line">Uncertainty (${esc(r.uncertainty_method)}): ${unc}</p>`;
}

function renderMetrics() {
  const r = state.result;
  if (!r) return;
  const rf = r.metrics.reference_free;
  const g = r.metrics.with_gt;
  const html = g
    ? [lpipsCard(g), consistencyCard(rf, g, r.refs), sharpnessCard(rf, g, r.refs), gtTable(g), footLines(r)]
    : [consistencyCard(rf, null, r.refs), sharpnessCard(rf, null, r.refs), uncertaintyCard(r), runtimeCard(r)];
  $("metrics-body").innerHTML = html.join("");

  const blur = rf.hf_ratio_vs_bicubic < r.refs.sharpness_warn_below;
  $("blur-banner").hidden = !blur;
  if (blur) {
    $("blur-banner").textContent =
      `Blur guard: SR high-frequency energy is only ${fmt(rf.hf_ratio_vs_bicubic, 2)}× bicubic (threshold ${fmt(r.refs.sharpness_warn_below, 2)}×). ` +
      "Low spectral-consistency error on this image may come from blur, not fidelity.";
  }

  const w = r.warnings || [];
  $("warnings-block").hidden = !w.length;
  $("warnings-list").innerHTML = w.map((s) => `<li>${esc(s)}</li>`).join("");

  const inp = r.input;
  $("result-meta").textContent =
    `job ${r.job_id} · ${inp.source}${inp.sample_id ? " " + inp.sample_id.slice(0, 18) + (inp.sample_id.length > 18 ? "…" : "") : ""} · ` +
    `${inp.lr_size[0]}×${inp.lr_size[1]} → ${inp.sr_size[0]}×${inp.sr_size[1]} px · DN ${inp.dn_mode_applied}` +
    `${inp.georeferenced ? " · georeferenced" : " · not georeferenced"}`;
}

function renderDownloads() {
  const r = state.result;
  $("downloads").hidden = !r;
  if (!r) return;
  const set = (el, url) => {
    el.hidden = !url;
    if (!url) return;
    el.href = url;
    el.classList.toggle("disabled", MOCK);
    el.setAttribute("aria-disabled", MOCK ? "true" : "false");
  };
  set($("dl-sr"), r.downloads.sr_tif);
  set($("dl-unc"), r.downloads.uncertainty_tif);
  $("dl-note").hidden = !MOCK;
  $("dl-note").textContent = MOCK ? "Mock mode: GeoTIFF downloads need the running API." : "";
}

function renderResult() {
  renderModel(state.result.model);
  renderViewer();
  renderMetrics();
  renderDownloads();
}

// ───────────────────────── "What's novel" evidence links ─────────────────────────
function flash(el) {
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1200);
}

function gotoEvidence(kind) {
  if (kind === "model") {
    flash($("model-badge"));
    flash($("metrics-body").querySelector('[id-card="runtime"]'));
    return;
  }
  if (state.result && overlayAvailable(kind)) {
    state.overlay = kind;
    renderOverlay();
  }
  flash($("metrics-body").querySelector(`[id-card="${kind}"]`));
}

// ───────────────────────── wiring ─────────────────────────
function wire() {
  $("sample-list").addEventListener("click", (e) => {
    const b = e.target.closest(".sample");
    if (b) selectSample(b.dataset.id);
  });
  $("file-input").addEventListener("change", (e) => selectFile(e.target.files[0]));
  const dz = $("drop-zone");
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("over");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("over");
    selectFile(e.dataTransfer.files[0]);
  });
  // Dropping a file anywhere else must not navigate away from the page.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  $("run-btn").addEventListener("click", run);
  $("left-layer").addEventListener("change", (e) => {
    state.left = e.target.value;
    renderViewer();
  });
  $("mode-rgb").addEventListener("click", () => {
    state.band = "rgb";
    renderViewer();
  });
  $("mode-fcc").addEventListener("click", () => {
    state.band = "fcc";
    renderViewer();
  });
  $("overlay-select").addEventListener("change", (e) => {
    state.overlay = e.target.value;
    renderOverlay();
  });
  $("overlay-opacity").addEventListener("input", (e) => {
    state.opacity = e.target.value / 100;
    $("opacity-value").textContent = `${e.target.value} %`;
    $("overlay-img").style.opacity = state.opacity;
  });
  $("reset-view").addEventListener("click", resetView);
  for (const a of [$("dl-sr"), $("dl-unc")]) {
    a.addEventListener("click", (e) => {
      if (a.classList.contains("disabled")) e.preventDefault();
    });
  }
  $("novel").addEventListener("click", (e) => {
    const b = e.target.closest(".goto");
    if (b) gotoEvidence(b.dataset.goto);
  });
  initViewer();
}

async function init() {
  wire();
  setSplit(50);
  applyView();
  renderOverlay();
  $("mock-chip").hidden = !MOCK;

  api
    .health()
    .then((h) => {
      setStatus(MOCK ? "mock" : "online", MOCK ? "mock API" : "API online");
      renderModel(h.model);
    })
    .catch(() => {
      setStatus("offline", "API offline");
      $("model-backend").textContent = "model unavailable";
    });

  try {
    const s = await api.samples();
    state.samples = s.samples || [];
    renderSamples();
    const wanted = state.samples.find((x) => x.id === params.get("sample")) || state.samples[0];
    if (wanted) selectSample(wanted.id);
    if (params.get("tta")) $("tta-select").value = params.get("tta");
    if (params.get("overlay")) state.overlay = params.get("overlay");
    if (params.get("autorun") === "1") run();
  } catch (e) {
    $("sample-list").innerHTML = '<p class="muted small">Samples unavailable.</p>';
    showError(e.message || String(e));
  }
}

init();
