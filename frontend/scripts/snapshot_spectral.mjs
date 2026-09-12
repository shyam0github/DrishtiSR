/**
 * Snapshot the spectral-consistency benchmark into src/data/spectralBenchmark.json
 * for /novelty/spectral.
 *
 *   - means, spectral floor, significance (Wilcoxon + bootstrap CIs) and the
 *     headline sentence: copied verbatim from reports/day3_results.json, the
 *     machine twin of reports/day3_results.md. Nothing is recomputed.
 *   - distributions (box-plot quantiles of l1_spec and sam_spec_deg): read from
 *     the eval harness's own per-pair tables, `<eval_all_ckpts.cache_dir>/<label>/<checkpoint>/per_pair.csv`
 *     (gitignored), for the checkpoint each report row selected. Each method's
 *     per-pair mean must reproduce the report's mean, or this script refuses.
 *
 *     npm run fixtures:spectral
 */
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";
import { BASE_YAML, REPO_ROOT } from "./repo-config.mjs";

const REPORT = "reports/day3_results.json";
const METRICS = ["l1_spec", "sam_spec_deg"];
const SIG_METRICS = ["reflectance", "spectral", "l1_spec", "sam_spec_deg"];
const MEAN_REL_TOL = 1e-9;
const OUT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "src", "data", "spectralBenchmark.json");

const report = JSON.parse(readFileSync(path.join(REPO_ROOT, REPORT), "utf8"));
if (report.smoke !== false) throw new Error(`${REPORT} is a smoke run; refusing to snapshot it`);
const cacheDir = parse(readFileSync(BASE_YAML, "utf8"))?.eval_all_ckpts?.cache_dir;
if (typeof cacheDir !== "string") throw new Error("configs/base.yaml: 'eval_all_ckpts.cache_dir' is missing");

/** Linear-interpolated quantile (numpy's default), on a sorted array. */
function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function readPerPair(label, checkpoint) {
  const rel = path.join(cacheDir, label, checkpoint, "per_pair.csv");
  const lines = readFileSync(path.join(REPO_ROOT, rel), "utf8").trim().split(/\r?\n/);
  const header = lines[0].split(",");
  const cols = Object.fromEntries(METRICS.map((m) => [m, header.indexOf(m)]));
  for (const [m, i] of Object.entries(cols)) if (i < 0) throw new Error(`${rel}: no '${m}' column`);
  const values = Object.fromEntries(METRICS.map((m) => [m, []]));
  for (const line of lines.slice(1)) {
    const cells = line.split(",");
    if (cells.length !== header.length) throw new Error(`${rel}: row has ${cells.length} cells, header has ${header.length}`);
    for (const m of METRICS) {
      const v = Number(cells[cols[m]]);
      if (!Number.isFinite(v)) throw new Error(`${rel}: non-finite ${m} '${cells[cols[m]]}'`);
      values[m].push(v);
    }
  }
  return { rel: rel.replaceAll("\\", "/"), values };
}

const ROLE = { bicubic: "baseline", a2: "control", b1: "treatment", b2: "supplementary" };
const rows = [...report.rows, ...report.supplementary_rows].filter((r) => r.label in ROLE);

const methods = rows.map((row) => {
  const { rel, values } = readPerPair(row.label, row.checkpoint);
  if (values[METRICS[0]].length !== report.n_pairs) throw new Error(`${rel}: ${values[METRICS[0]].length} pairs, report has ${report.n_pairs}`);
  const dist = {};
  for (const m of METRICS) {
    const v = values[m];
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    const want = row.means[m];
    if (Math.abs(mean - want) > MEAN_REL_TOL * Math.abs(want)) {
      throw new Error(`${rel}: per-pair mean ${m} ${mean} != report ${want}; the cache is not the reported checkpoint`);
    }
    const s = [...v].sort((a, b) => a - b);
    dist[m] = {
      min: s[0], p5: quantile(s, 0.05), q1: quantile(s, 0.25), median: quantile(s, 0.5),
      q3: quantile(s, 0.75), p95: quantile(s, 0.95), max: s[s.length - 1],
    };
  }
  return {
    label: row.label, display: row.display, checkpoint: row.checkpoint, role: ROLE[row.label], lambdas: row.lambdas,
    means: { reflectance: row.means.reflectance, spectral: row.means.spectral, l1_spec: row.means.l1_spec, sam_spec_deg: row.means.sam_spec_deg },
    distribution: dist, per_pair_source: rel,
  };
});
for (const label of Object.keys(ROLE)) if (!methods.some((m) => m.label === label)) throw new Error(`${REPORT}: no row '${label}'`);

const significance = SIG_METRICS.map((metric) => {
  const s = report.significance.find((x) => x.metric === metric);
  if (!s) throw new Error(`${REPORT}: no significance record for '${metric}'`);
  const keep = ["metric", "better", "n", "n_clusters", "mean_control", "mean_treatment", "mean_delta", "median_delta",
    "frac_treatment_better", "ci", "ci_pair", "ci_tile", "p_wilcoxon_pair", "p_wilcoxon_tile", "verdict"];
  return Object.fromEntries(keep.map((k) => [k, s[k]]));
});

const out = {
  source: REPORT,
  report_md: "reports/day3_results.md",
  written_utc: report.written_utc,
  fingerprint: report.fingerprint,
  n_pairs: report.n_pairs,
  n_tiles: significance[0].n_clusters,
  headline: report.reading[0],
  cost: report.reading[1],
  comparison: "Run B (B1) minus Run A2 (control), per validation pair; B2 has no significance test in the report",
  floor: { reflectance: report.floor.reflectance, spectral: report.floor.spectral, l1_spec: report.floor.l1_spec, sam_spec_deg: report.floor.sam_spec_deg },
  methods,
  significance,
};
writeFileSync(OUT, JSON.stringify(out, null, 2) + "\n");
console.log(`spectral benchmark: ${REPORT} + ${methods.length} per-pair tables -> ${path.relative(process.cwd(), OUT)}`);
