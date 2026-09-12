import { useMemo, useState } from "react";
import { RadarChart, type RadarAxis, type RadarSeries } from "../components/RadarChart";
import { Badge, Card, Reveal, SectionHeader, StatCard, TabbedPanel } from "../components/ui";
import { COMPARE_SOURCES, isComplete, MEASURED_SPLIT, METRICS, MODELS, type MetricDef, type MetricKey, type ModelRow } from "../data/compare";
import { FACTS } from "../data/facts";
import { SR_MODELS } from "../data/srModels";
import { formatNumber } from "../design/motion";
import { logDomain } from "../design/radar";

const NA_TEXT = "N/A — not directly comparable";
/** Parameter count from the site-wide SR_MODELS list, or null when the list has no entry. */
const paramsOf = (name: string) => SR_MODELS.find((r) => r.name === name)?.params ?? null;
const byId = (id: string) => {
  const m = MODELS.find((r) => r.id === id);
  if (!m) throw new Error(`compare data: no model '${id}'`);
  return m;
};
const fmt = (d: MetricDef, v: number, source: ModelRow["source"]) => formatNumber(v, source === "reported" ? 4 : d.decimals);
const val = (m: ModelRow, k: MetricKey) => {
  const v = m.values[k];
  if (v === null) throw new Error(`compare data: ${m.id}.${k} is null but was read as a number`);
  return v;
};

export default function ComparePage() {
  return (
    <>
      <Intro />
      <Glossary />
      <Comparison />
    </>
  );
}

// ------------------------------------------------------------------ intro ----

function Intro() {
  const measured = MODELS.filter((m) => m.source === "measured").length;
  return (
    <section className="relative overflow-hidden" data-testid="compare-intro">
      <div aria-hidden className="bg-graticule absolute inset-0" />
      <div className="page-container relative flex flex-col gap-10 pt-[calc(var(--spacing-section)*0.75)] pb-section">
        <SectionHeader
          level="h1"
          eyebrow="Cross-model rigor"
          title="Judged by a benchmark we did not write"
          description={
            <>
              PSNR and SSIM were built for photographs. They reward a smooth average and cannot tell real detail from invented detail.{" "}
              <a className="text-accent hover:text-accent-strong" href="https://github.com/ESAOpenSR/opensr-test">
                opensr-test
              </a>{" "}
              is ESA OpenSR's independent benchmark for Sentinel-2 super-resolution. It asks three questions. Is the output still faithful to the 10 m input it came from (consistency)? Did the model add detail at all (synthesis)? Is that detail real, invented, or missing (correctness)? Seven numbers answer them.
            </>
          }
        />
        <Reveal className="grid gap-4 sm:grid-cols-3">
          <StatCard value={METRICS.length} label="opensr-test metrics" hint={`library ${MEASURED_SPLIT.opensrVersion}, upstream defaults`} />
          <StatCard accent value={MEASURED_SPLIT.pairs} label="Validation pairs per measured row" hint={`${MEASURED_SPLIT.dataset} ${MEASURED_SPLIT.split}, identical for every row`} />
          <StatCard value={measured} label="Rows measured by us" hint={`${MODELS.length - measured} more are quoted from the literature`} />
        </Reveal>
      </div>
    </section>
  );
}

// --------------------------------------------------------------- glossary ----

function Glossary() {
  return (
    <section className="border-y border-line bg-ink-900/50" data-testid="glossary">
      <div className="page-container flex flex-col gap-8 py-section">
        <SectionHeader eyebrow="The seven metrics" title="What each number means" description="The arrow shows which direction is better. The last column explains why the metric matters more for satellite imagery than for photos." />
        <div className="overflow-x-auto rounded-lg border border-line">
          <table className="w-full min-w-[52rem] border-collapse text-left text-caption">
            <thead className="bg-ink-850 text-fg-muted">
              <tr>
                <th scope="col" className="px-4 py-3 font-medium">Metric</th>
                <th scope="col" className="px-4 py-3 font-medium">What it measures</th>
                <th scope="col" className="px-4 py-3 font-medium">Better</th>
                <th scope="col" className="px-4 py-3 font-medium">Why it matters for satellite imagery</th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map((d) => (
                <tr key={d.key} className="border-t border-line align-top">
                  <th scope="row" className="px-4 py-3 font-medium text-fg">
                    <span className="block">{d.name}</span>
                    <span className="num text-micro text-fg-subtle">
                      {d.group} · {d.key}
                    </span>
                  </th>
                  <td className="px-4 py-3 text-fg-muted">
                    {d.measures} <span className="text-fg-subtle">({d.unit})</span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <Badge variant="neutral">{d.better === "lower" ? "↓ lower" : "↑ higher"}</Badge>
                  </td>
                  <td className="px-4 py-3 text-fg-muted">{d.whySatellite}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

// ------------------------------------------------------------- comparison ----

function Comparison() {
  const plottable = MODELS.filter(isComplete);
  const [shown, setShown] = useState<Set<string>>(() => new Set(MODELS.filter((m) => m.defaultOn && isComplete(m)).map((m) => m.id)));

  // Domains over every plottable row, not just the shown ones, so a toggle never rescales the chart.
  const axes: RadarAxis[] = useMemo(
    () =>
      METRICS.map((d) => ({
        key: d.key,
        label: d.short,
        better: d.better,
        domain: logDomain(plottable.map((m) => m.values[d.key])),
        format: (v: number, reported: boolean) => `${fmt(d, v, reported ? "reported" : "measured")} ${d.unit === "0–1" ? "" : d.unit}`.trim(),
      })),
    [plottable],
  );
  const series: RadarSeries[] = plottable
    .filter((m) => shown.has(m.id))
    .reverse() // our row last, so it draws on top
    .map((m) => ({
      id: m.id,
      label: m.name,
      color: m.color,
      dashed: m.source === "reported",
      note: m.source === "measured" ? "Measured (ours)" : "Reported (literature), not comparable",
      values: Object.fromEntries(METRICS.map((d) => [d.key, val(m, d.key)])),
    }));

  const toggle = (id: string) =>
    setShown((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <section className="page-container flex flex-col gap-10 py-section" data-testid="comparison">
      <SectionHeader
        eyebrow="Model comparison"
        title="Where DrishtiSR sits"
        description="Three rows are measured by us on identical data and settings. The literature rows are for rough positioning only."
      />

      <div role="note" data-testid="disclaimer" className="flex flex-col gap-2 rounded-lg border border-warn/40 bg-warn/10 p-5 text-caption text-fg sm:flex-row sm:items-start sm:gap-4">
        <Badge variant="warn">Not a controlled benchmark</Badge>
        <p className="text-fg-muted">
          <span className="font-medium text-fg">Literature numbers were not measured under identical conditions.</span> They are quoted from the opensr-test
          README. That table uses different test images (opensr-test's own datasets, not our {formatNumber(MEASURED_SPLIT.pairs)} {MEASURED_SPLIT.dataset}{" "}
          pairs), patch-level aggregation and a GPU. We show them for rough positioning only and make no claim of beating those models. Where no
          opensr-test number exists, the cell says N/A rather than guessing.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_19rem]">
        <Card interactive={false} className="flex flex-col gap-4 p-4 sm:p-6">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-h5">Seven metrics, one shape</h3>
            <p className="text-caption text-fg-subtle">Outer = better on every axis · log-scaled per axis · hover a point for its value</p>
          </div>
          <div className="mx-auto w-full max-w-[36rem]">
            <RadarChart axes={axes} series={series} ariaLabel={`Radar chart of ${series.map((s) => s.label).join(", ") || "no models"} across the seven opensr-test metrics. Exact values are in the table below.`} />
          </div>
        </Card>

        <fieldset className="flex flex-col gap-3" data-testid="model-toggles">
          <legend className="eyebrow mb-3">Show models</legend>
          {MODELS.map((m) => {
            const ok = isComplete(m);
            return (
              <label
                key={m.id}
                className={`card flex items-start gap-3 p-3 ${ok ? "cursor-pointer" : "card-static cursor-not-allowed opacity-60"}`}
                title={ok ? m.detail : m.naReason}
              >
                <input type="checkbox" className="mt-1 accent-[var(--color-accent)]" checked={shown.has(m.id)} disabled={!ok} onChange={() => toggle(m.id)} data-model={m.id} />
                <span className="flex min-w-0 flex-col gap-1.5">
                  <span className="flex items-center gap-2 text-caption font-medium text-fg">
                    <svg aria-hidden width="22" height="8" className="shrink-0">
                      <line x1="1" y1="4" x2="21" y2="4" stroke={ok ? m.color : "var(--color-fg-subtle)"} strokeWidth="2.5" strokeDasharray={m.source === "reported" ? "5 3" : undefined} />
                    </svg>
                    {m.name}
                  </span>
                  <span className="flex flex-wrap gap-1.5">
                    <SourceBadge m={m} />
                  </span>
                  {!ok && <span className="text-micro normal-case tracking-normal text-fg-subtle">No opensr-test numbers: not plotted</span>}
                </span>
              </label>
            );
          })}
        </fieldset>
      </div>

      <Narrative />

      <TabbedPanel
        tabs={[
          { id: "numbers", label: "Exact numbers", content: <MetricTable /> },
          { id: "sources", label: "Sources", content: <Sources /> },
        ]}
      />
    </section>
  );
}

function SourceBadge({ m }: { m: ModelRow }) {
  return m.source === "measured" ? <Badge variant="good">Measured (ours)</Badge> : <Badge variant="literature">Reported (literature)</Badge>;
}

/** Wins and trade-offs, computed from the rows so the prose cannot drift from the table. */
function Narrative() {
  const us = byId("drishtisr");
  const edsr = byId("edsr");
  const bic = byId("bicubic");
  const gan = byId("satlas");
  const f = (m: ModelRow, k: MetricKey, dp: number) => formatNumber(val(m, k), dp);
  const fewer = formatNumber((1 - FACTS.params / FACTS.runAParams) * 100, 0);
  return (
    <Card interactive={false} accent className="flex flex-col gap-3 p-6" data-testid="narrative">
      <p className="eyebrow">Honest positioning</p>
      <p className="text-body text-fg-muted">
        <span className="font-medium text-fg">Where it wins:</span> DrishtiSR beats the EDSR-baseline it was built from on all three consistency metrics
        (spectral angle {f(us, "spectral", 2)}° vs {f(edsr, "spectral", 2)}°, alignment {f(us, "spatial", 4)} vs {f(edsr, "spatial", 4)} px). It also hallucinates
        less ({f(us, "ha_metric", 3)} vs {f(edsr, "ha_metric", 3)}) with {fewer}% fewer parameters.{" "}
        <span className="font-medium text-fg">Where it trades off:</span> EDSR-baseline recovers slightly more real detail (omission {f(edsr, "om_metric", 3)} vs{" "}
        {f(us, "om_metric", 3)}). Bicubic stays more faithful in reflectance and spectrum ({f(bic, "reflectance", 5)} vs {f(us, "reflectance", 5)}), because it adds
        nothing.{" "}
        <span className="font-medium text-fg">Against the literature,</span> as rough positioning only: the GAN-based Satlas recovers far more detail (omission{" "}
        {f(gan, "om_metric", 3)}) but reports {f(gan, "ha_metric", 2)} hallucination and {f(gan, "spectral", 1)}° spectral drift. DrishtiSR sits at the
        conservative, spectrally faithful end. It is not the best at everything.
      </p>
    </Card>
  );
}

type Sort = { key: MetricKey; bestFirst: boolean } | null;

function MetricTable() {
  const [sort, setSort] = useState<Sort>(null);
  const measured = MODELS.filter((m) => m.source === "measured");
  const bestMeasured = Object.fromEntries(
    METRICS.map((d) => {
      const vs = measured.map((m) => val(m, d.key));
      return [d.key, d.better === "lower" ? Math.min(...vs) : Math.max(...vs)];
    }),
  ) as Record<MetricKey, number>;

  const rows = useMemo(() => {
    if (!sort) return [...MODELS];
    const d = METRICS.find((x) => x.key === sort.key)!;
    const sign = (d.better === "lower") === sort.bestFirst ? 1 : -1;
    return [...MODELS].sort((a, b) => {
      const va = a.values[sort.key];
      const vb = b.values[sort.key];
      if (va === null && vb === null) return 0;
      if (va === null) return 1; // N/A always last
      if (vb === null) return -1;
      return sign * (va - vb);
    });
  }, [sort]);

  const clickSort = (key: MetricKey) => setSort((s) => (s?.key === key ? { key, bestFirst: !s.bestFirst } : { key, bestFirst: true }));
  const ariaSort = (d: MetricDef): "ascending" | "descending" | "none" => {
    if (sort?.key !== d.key) return "none";
    return (d.better === "lower") === sort.bestFirst ? "ascending" : "descending";
  };

  return (
    <div className="flex flex-col gap-3">
      {/* relative: the sr-only spans in cells are absolutely positioned; without a containing
          block here they escape the scroll box and widen the page on phones. */}
      <div className="relative overflow-x-auto rounded-lg border border-line">
        <table className="w-full min-w-[60rem] border-collapse text-left text-caption" data-testid="metric-table">
          <thead className="bg-ink-850 text-fg-muted">
            <tr>
              <th scope="col" className="px-4 py-3 font-medium">
                <button type="button" onClick={() => setSort(null)} className="hover:text-fg">
                  Model {sort && <span className="text-fg-subtle">(reset order)</span>}
                </button>
              </th>
              {METRICS.map((d) => (
                <th key={d.key} scope="col" aria-sort={ariaSort(d)} className="px-3 py-3 text-right font-medium">
                  <button type="button" onClick={() => clickSort(d.key)} data-sort={d.key} className={`inline-flex items-center gap-1 hover:text-fg ${sort?.key === d.key ? "text-fg" : ""}`}>
                    {d.short} {d.better === "lower" ? "↓" : "↑"}
                    <span aria-hidden className="text-fg-subtle">{sort?.key === d.key ? (sort.bestFirst ? "▲" : "▼") : "⇅"}</span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr key={m.id} data-row={m.id} className={`border-t border-line align-top ${m.roleVariant === "ours" ? "bg-accent-dim/40" : ""}`}>
                <th scope="row" className="px-4 py-3 font-normal" title={m.detail}>
                  <span className="flex flex-col gap-1.5">
                    <span className="font-medium text-fg">{m.name}</span>
                    <span className="text-micro normal-case tracking-normal text-fg-subtle">
                      {m.category}
                      {paramsOf(m.name) !== null && ` · ${formatNumber(paramsOf(m.name)!)} params`}
                    </span>
                    <span className="flex flex-wrap gap-1.5">
                      {m.role && <Badge variant={m.roleVariant ?? "neutral"}>{m.role}</Badge>}
                      <SourceBadge m={m} />
                    </span>
                  </span>
                </th>
                {METRICS.map((d) => {
                  const v = m.values[d.key];
                  if (v === null) {
                    return (
                      <td key={d.key} className="px-3 py-3 text-right text-micro normal-case tracking-normal text-fg-subtle" title={m.naReason} data-na>
                        {NA_TEXT}
                      </td>
                    );
                  }
                  const best = m.source === "measured" && v === bestMeasured[d.key];
                  return (
                    <td key={d.key} className={`num px-3 py-3 text-right ${best ? "font-semibold text-accent" : "text-fg"}`} title={m.source === "reported" ? `Reported: ${m.detail}` : undefined}>
                      {fmt(d, v, m.source)}
                      {m.source === "reported" && <sup className="ml-0.5 text-lit">†</sup>}
                      {best && <span className="sr-only"> (best measured)</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-caption text-fg-subtle">
        <span className="text-accent">Highlighted</span>: best among the three measured rows only; literature rows are never ranked against ours.{" "}
        <sup className="text-lit">†</sup> Reported value, quoted to the README's 4 decimals. Different test images and settings, so not directly comparable. Click
        a column to sort best-first; click again for worst-first. N/A rows always sort last.
      </p>
    </div>
  );
}

function Sources() {
  return (
    <ul className="flex flex-col gap-3" data-testid="sources">
      {MODELS.map((m) => (
        <li key={m.id} className="flex flex-col gap-1.5 rounded-lg border border-line p-4 text-caption">
          <span className="flex flex-wrap items-center gap-2">
            <span className="font-medium text-fg">{m.name}</span>
            <SourceBadge m={m} />
          </span>
          <span className="text-fg-muted">{m.detail}</span>
          {m.naReason && <span className="text-fg-subtle">{m.naReason}</span>}
          {m.reference && (
            <a href={m.reference.href} className="text-accent hover:text-accent-strong">
              {m.reference.text}
            </a>
          )}
          {m.source === "measured" && <span className="num text-fg-subtle">{COMPARE_SOURCES.measured} → {m.snapshot?.label} / {m.snapshot?.checkpoint}</span>}
          {m.readmeModel && <span className="num text-fg-subtle">{COMPARE_SOURCES.reported} → row “{m.readmeModel}”</span>}
        </li>
      ))}
    </ul>
  );
}
