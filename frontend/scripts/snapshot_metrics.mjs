/**
 * Snapshot reports/day3_results.json into src/api/fixtures/metrics.json, the
 * MOCK_MODE answer to GET /metrics.
 *
 * Copies a subset of top-level keys with their names and values unchanged, so
 * the fixture is exactly what the Day 4 endpoint will serve from that same file.
 * Re-run after scripts/eval_all_ckpts.py rewrites the report:
 *
 *     npm run fixtures:metrics
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { REPO_ROOT } from "./repo-config.mjs";

const SOURCE = "reports/day3_results.json";
const KEYS = ["written_utc", "fingerprint", "n_pairs", "reading", "rows", "supplementary_rows", "floor", "below_floor"];
const OUT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "src", "api", "fixtures", "metrics.json");

const report = JSON.parse(readFileSync(path.join(REPO_ROOT, SOURCE), "utf8"));
if (report.smoke !== false) throw new Error(`${SOURCE} is a smoke run (smoke=${report.smoke}); refusing to snapshot it as the demo's metrics`);

const fixture = { source: SOURCE };
for (const key of KEYS) {
  if (!(key in report)) throw new Error(`${SOURCE}: missing key '${key}'`);
  fixture[key] = report[key];
}
mkdirSync(path.dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(fixture, null, 2) + "\n");
console.log(`metrics fixture: ${SOURCE} (written ${report.written_utc}) -> ${path.relative(process.cwd(), OUT)}`);
