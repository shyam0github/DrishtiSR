import { useEffect, useState } from "react";
import { getMetrics, MOCK_MODE, type MetricMeans, type MetricsResponse, type MetricsRow } from "../api/client";
import { APP_CONFIG } from "../config";

interface Column {
  key: keyof MetricMeans;
  label: string;
  arrow: "↑" | "↓";
  digits: number;
}

/** The headline columns of reports/day3_results.md section 2, with its labels and precision. */
const COLUMNS: Column[] = [
  { key: "psnr_mean", label: "PSNR dB", arrow: "↑", digits: 3 },
  { key: "ssim_mean", label: "SSIM", arrow: "↑", digits: 4 },
  { key: "lpips", label: "LPIPS", arrow: "↓", digits: 4 },
  { key: "sam_mean_deg", label: "SAM°", arrow: "↓", digits: 3 },
  { key: "ergas", label: "ERGAS", arrow: "↓", digits: 3 },
  { key: "reflectance", label: "opensr refl. L1", arrow: "↓", digits: 5 },
  { key: "l1_spec", label: "L1_spec", arrow: "↓", digits: 6 },
];

const parseCount = (s: string): number | null => {
  const v = Number(s.replace(/,/g, ""));
  return s.trim() !== "" && Number.isFinite(v) ? v : null;
};

const cell = "px-1.5 py-0.5 text-right tabular-nums";

/** The Day 3 headline table, read through the API client (a fixture in MOCK_MODE). */
export function MetricsPanel() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    getMetrics().then(
      (d) => live && setData(d),
      (exc: unknown) => {
        console.error("getMetrics failed:", exc);
        if (live) setError(exc instanceof Error ? exc.message : String(exc));
      },
    );
    return () => {
      live = false;
    };
  }, []);

  if (error) {
    return (
      <section role="alert" data-testid="metrics" className="rounded border border-red-600 bg-red-50 p-2 text-xs text-red-800">
        Metrics failed to load: {error}
      </section>
    );
  }
  if (!data) return <section data-testid="metrics" className="text-xs text-slate-500">Loading metrics…</section>;

  const row = (r: MetricsRow, supplementary: boolean) => {
    const params = parseCount(r.params);
    const overBudget = params !== null && params > APP_CONFIG.maxParameters;
    return (
      <tr key={r.label} className={`border-t border-slate-200 ${supplementary ? "text-slate-500" : ""}`}>
        <th scope="row" className="whitespace-nowrap px-1.5 py-0.5 text-left font-medium">
          {r.display}
          {supplementary && <span className="font-normal"> (suppl.)</span>}
        </th>
        <td className="whitespace-nowrap px-1.5 py-0.5">{r.lambdas}</td>
        {COLUMNS.map((c) => (
          <td key={c.key} className={cell}>
            {r.means[c.key].toFixed(c.digits)}
          </td>
        ))}
        <td className={cell}>{(r.means.l1_spec / data.floor.l1_spec).toFixed(2)}×</td>
        <td className={`${cell} ${overBudget ? "font-semibold text-red-700" : ""}`} title={overBudget ? `over the ${APP_CONFIG.maxParameters.toLocaleString("en-US")}-parameter budget` : undefined}>
          {r.params}
          {overBudget && " ⚠"}
        </td>
        <td className="whitespace-nowrap px-1.5 py-0.5">{r.iters}</td>
      </tr>
    );
  };

  return (
    <section className="space-y-2" data-testid="metrics">
      <h2 className="font-semibold">Day 3 results</h2>
      <p className="text-xs text-slate-600">
        {data.n_pairs.toLocaleString("en-US")} validation pairs · written {data.written_utc} · <code>{data.source}</code>
        {MOCK_MODE && <span className="ml-1 rounded bg-slate-200 px-1">fixture snapshot</span>}
      </p>
      {data.reading.map((p) => (
        <p key={p} className="text-xs">
          {p}
        </p>
      ))}
      <div className="overflow-x-auto">
        <table className="min-w-full text-[11px]">
          <thead>
            <tr className="text-slate-600">
              <th className="px-1.5 text-left font-medium">method</th>
              <th className="px-1.5 text-left font-medium">λ1 / λ2</th>
              {COLUMNS.map((c) => (
                <th key={c.key} className="whitespace-nowrap px-1.5 text-right font-medium">
                  {c.label} {c.arrow}
                </th>
              ))}
              <th className="whitespace-nowrap px-1.5 text-right font-medium">L1_spec ÷ floor</th>
              <th className="px-1.5 text-right font-medium">params</th>
              <th className="whitespace-nowrap px-1.5 text-left font-medium">iters (sel. / trained)</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((r) => row(r, false))}
            <tr className="border-t border-slate-200 italic text-slate-600">
              <th scope="row" className="whitespace-nowrap px-1.5 py-0.5 text-left font-normal">
                GT HR as the SR (spectral floor)
              </th>
              <td className="px-1.5">—</td>
              {COLUMNS.map((c) => (
                <td key={c.key} className={cell}>
                  {c.key in data.floor ? data.floor[c.key as keyof MetricsResponse["floor"]].toFixed(c.digits) : "—"}
                </td>
              ))}
              <td className={cell}>1.00×</td>
              <td className={cell}>—</td>
              <td className="px-1.5">—</td>
            </tr>
            {data.supplementary_rows.map((r) => row(r, true))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-slate-500">
        ¹ Run A is not a controlled comparison: a different architecture over the parameter budget, and a different data pipeline. Below the
        spectral floor: {data.below_floor.join(", ")}. LPIPS sees RGB only and is a perceptual proxy; rank on PSNR/SSIM/SAM/ERGAS. Full
        context in reports/day3_results.md.
      </p>
    </section>
  );
}
