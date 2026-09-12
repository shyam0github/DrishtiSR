import { useState, type ReactNode } from "react";
import type { ImageResult } from "../api/client";
import { Badge, Button, Card, PendingStatCard, StatCard } from "../components/ui";
import { APP_CONFIG } from "../config";
import BENCH from "../data/spectralBenchmark.json";
import { formatNumber } from "../design/motion";
import { NoveltyPageLayout, type ImageSource } from "../layout/NoveltyPageLayout";

const LR_GSD = `${APP_CONFIG.lrGsdM} m`;
const SR_GSD = `${APP_CONFIG.lrGsdM / APP_CONFIG.scale} m`;

type Method = (typeof BENCH.methods)[number];
type DistMetric = "l1_spec" | "sam_spec_deg";

/**
 * Series colours by method, never by rank. The three model hues passed the
 * dataviz palette validator on the card surface (#0a101c, dark): lightness
 * band, chroma, CVD and normal-vision separation, contrast. Bicubic is the
 * baseline and wears the neutral text-subtle grey instead of a hue.
 */
const COLOR: Record<string, string> = { bicubic: "var(--color-fg-subtle)", a2: "#0891b2", b1: "#16a34a", b2: "#6366f1" };
const ROLE_NOTE: Record<string, string> = {
  baseline: "baseline",
  control: "served model, no spectral loss",
  treatment: "spectral loss, pre-declared primary",
  supplementary: "stronger spectral loss",
};
const SERVED_LABEL = "a2";

/** Viridis stops matching app's overlay LUT (src/infer/render.py), for the legend only. */
const VIRIDIS = "linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)";

const sig = (metric: string) => {
  const s = BENCH.significance.find((x) => x.metric === metric);
  if (!s) throw new Error(`spectralBenchmark.json: no significance for ${metric}`);
  return s;
};
const pct = (v: number, base: number) => (v / base) * 100;
/** A p-value, with a real superscript for tiny ones (6.1×10⁻⁵¹, not 6.1e-51). */
function P({ p }: { p: number }) {
  if (p >= 1e-4) return <>{formatNumber(p, 4)}</>;
  const [mant, exp] = p.toExponential(1).split("e");
  return (
    <>
      {mant}×10<sup>{Number(exp)}</sup>
    </>
  );
}

export function SpectralPage() {
  return (
    <NoveltyPageLayout
      testId="page-spectral"
      index="01"
      title="Spectral consistency"
      summary={
        <>
          Super-resolution invents detail, but it must not invent colour. Take the {SR_GSD} output, blur and shrink it back to {LR_GSD} the way
          Sentinel-2 sees the ground, and it should give back the reflectance the satellite actually measured. The gap between the two is the
          consistency error: small means the new detail is physically compatible with the original measurement.
        </>
      }
      renderImage={(result, source) => <OnImage result={result} source={source} />}
      benchmarkLabel={`n = ${formatNumber(BENCH.n_pairs)} val patches · ${BENCH.n_tiles} tiles`}
      benchmarkSource={BENCH.report_md}
      benchmark={<Benchmark />}
      howItWorks={<HowItWorks />}
    />
  );
}

// ------------------------------------------------------------ this image ----

function OnImage({ result, source }: { result: ImageResult; source: ImageSource }) {
  const [map, setMap] = useState<"sam" | "l1">("sam");
  const [opacity, setOpacity] = useState(0.85);
  const s = result.spectral;
  const m = result.metrics;
  const overlay = map === "sam" ? result.images.consistencySam : result.images.consistency;
  const vmax = map === "sam" ? s.samDisplayMax : s.l1DisplayMax;
  const blurry = m.hfRatioVsBicubic != null && m.sharpnessWarnBelow != null && m.hfRatioVsBicubic < m.sharpnessWarnBelow;
  const mockReason = result.mode === "mock" ? "Mock mode: no model ran." : "Not reported by this server version.";

  return (
    <div className="flex flex-col gap-8" data-testid="spectral-image" data-source={source}>
      <div className="flex flex-wrap items-center gap-3 text-caption text-fg-muted">
        {source === "example" ? <Badge variant="neutral">Example</Badge> : <Badge variant="ours">Your image</Badge>}
        <span className="truncate">{result.sourceLabel}</span>
        <Badge variant="warn">AI-reconstructed</Badge>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Figure caption={`${LR_GSD} input, as measured by Sentinel-2`} tag="Measured">
          <img src={result.images.lr} alt={`${LR_GSD} input`} className="size-full object-cover [image-rendering:pixelated]" />
        </Figure>
        <Figure caption={`Our ${SR_GSD} output, degraded back to ${LR_GSD}`} tag="D(SR)">
          {result.images.srDegraded ? (
            <img src={result.images.srDegraded} alt="SR output degraded to 10 m" className="size-full object-cover [image-rendering:pixelated]" data-testid="degraded-img" />
          ) : (
            <Unavailable testId="degraded-pending" reason={mockReason} />
          )}
        </Figure>
      </div>
      <p className="-mt-4 max-w-prose text-caption text-fg-subtle">
        Same stretch, same pixel grid. If the model is spectrally consistent the two panels are hard to tell apart; any difference is colour or
        brightness the model added that the satellite did not see.
      </p>

      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex gap-2" role="group" aria-label="Error map">
              <Button variant={map === "sam" ? "primary" : "secondary"} onClick={() => setMap("sam")} aria-pressed={map === "sam"}>
                Spectral angle (SAM)
              </Button>
              <Button variant={map === "l1" ? "primary" : "secondary"} onClick={() => setMap("l1")} aria-pressed={map === "l1"}>
                Reflectance L1
              </Button>
            </div>
            <label className="flex items-center gap-2 text-caption text-fg-muted">
              Overlay
              <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} className="accent-[var(--color-accent)]" />
            </label>
          </div>
          <Figure caption={`Per-pixel ${map === "sam" ? "spectral-angle" : "L1"} error of D(SR) vs the input, drawn over the ${SR_GSD} output`} tag={map === "sam" ? "SAM map" : "L1 map"}>
            <img src={result.images.sr} alt={`${SR_GSD} output`} className="absolute inset-0 size-full object-cover" />
            {overlay ? (
              <img src={overlay} alt="" className="absolute inset-0 size-full object-cover [image-rendering:pixelated]" style={{ opacity }} data-testid={`overlay-${map}`} />
            ) : (
              <div className="absolute inset-0 bg-ink-950/70">
                <Unavailable testId={`overlay-${map}-pending`} reason={mockReason} />
              </div>
            )}
          </Figure>
          {vmax != null && overlay && (
            <div className="flex flex-col gap-1" data-testid="overlay-legend">
              <div className="h-2 rounded-full" style={{ background: VIRIDIS }} />
              <div className="num flex justify-between text-micro text-fg-subtle">
                <span>0</span>
                <span>
                  {map === "sam" ? `${formatNumber(vmax, 2)}° = the true ${SR_GSD} image's own mean angle (fixed scale)` : `${formatNumber(vmax, 4)} reflectance (fixed scale)`}
                </span>
              </div>
              <p className="text-caption text-fg-subtle">Each coloured block is one {LR_GSD} pixel. Faint areas are near zero; bright yellow saturates the fixed scale.</p>
            </div>
          )}
        </div>

        <aside className="flex flex-col gap-3" data-testid="spectral-image-metrics">
          <p className="eyebrow">This image</p>
          {m.specL1 != null ? (
            <StatCard
              size="sm"
              value={m.specL1}
              decimals={5}
              label="Consistency error, L1 ↓"
              hint={s.l1GtFloor != null ? `true ${SR_GSD} image, VAL mean: ${formatNumber(s.l1GtFloor, 5)}` : "reflectance"}
              delta={m.specL1Bicubic != null ? { value: m.specL1 - m.specL1Bicubic, decimals: 5, better: "neutral", label: "vs bicubic" } : undefined}
            />
          ) : (
            <PendingStatCard size="sm" label="Consistency error, L1 ↓" reason={mockReason} />
          )}
          {s.samDeg != null ? (
            <StatCard
              size="sm"
              value={s.samDeg}
              decimals={3}
              unit="°"
              label="Spectral angle ↓"
              hint={s.samGtFloorDeg != null ? `true ${SR_GSD} image, VAL mean: ${formatNumber(s.samGtFloorDeg, 3)}°` : undefined}
              delta={s.samDegBicubic != null ? { value: s.samDeg - s.samDegBicubic, decimals: 3, unit: "°", better: "neutral", label: "vs bicubic" } : undefined}
            />
          ) : (
            <PendingStatCard size="sm" label="Spectral angle ↓" reason={mockReason} />
          )}
          {m.hfRatioVsBicubic != null ? (
            <StatCard
              size="sm"
              value={m.hfRatioVsBicubic}
              decimals={2}
              unit="× bicubic"
              label="Sharpness (HF energy)"
              badge={blurry ? <Badge variant="warn">Blur?</Badge> : undefined}
              hint="guards against the blur shortcut (see How it works)"
            />
          ) : (
            <PendingStatCard size="sm" label="Sharpness (HF energy)" reason={mockReason} />
          )}
          {s.l1Hr != null && (
            <p className="text-caption text-fg-subtle">
              This image's own {SR_GSD} truth scores L1 {formatNumber(s.l1Hr, 5)} on the same check.
            </p>
          )}
          <p className="text-caption text-fg-subtle">
            Bicubic is near-perfect here by construction: it is a smooth copy of the input. A lower number is only good if sharpness stays up.
          </p>
        </aside>
      </div>
    </div>
  );
}

function Figure({ caption, tag, children }: { caption: string; tag: string; children: ReactNode }) {
  return (
    <figure className="flex flex-col gap-2">
      <div className="relative aspect-square w-full overflow-hidden rounded-lg border border-line bg-ink-900">
        {children}
        <span className="absolute top-2 left-2 rounded-sm bg-ink-950/80 px-2 py-1 font-mono text-micro uppercase text-fg">{tag}</span>
      </div>
      <figcaption className="text-caption text-fg-muted">{caption}</figcaption>
    </figure>
  );
}

function Unavailable({ reason, testId }: { reason: string; testId: string }) {
  return (
    <div className="flex size-full flex-col items-center justify-center gap-2 p-6 text-center" data-testid={testId}>
      <Badge variant="pending">Not available</Badge>
      <p className="text-caption text-fg-subtle">{reason}</p>
    </div>
  );
}

// ------------------------------------------------------------- benchmark ----

function Benchmark() {
  const head = sig("reflectance");
  const rel = pct(head.mean_delta, head.mean_control);
  const [lo, hi] = head.ci_tile as [number, number];
  return (
    <div className="flex flex-col gap-10">
      <div className="grid gap-6 lg:grid-cols-[20rem_minmax(0,1fr)]">
        <Card accent interactive={false} className="flex flex-col gap-4 p-6" data-testid="bench-headline">
          <p className="text-caption font-medium text-fg-muted">Consistency error, spectral loss (B1) vs no spectral loss (A2)</p>
          <p className="num text-stat font-medium text-fg">{formatNumber(rel, 1)}%</p>
          <p className="text-caption text-fg-subtle">
            opensr L1 {formatNumber(head.mean_control, 5)} → {formatNumber(head.mean_treatment, 5)}. Tile-clustered 95% CI on the mean change [
            {formatNumber(lo, 5)}, {formatNumber(hi, 5)}] ({formatNumber(pct(lo, head.mean_control), 1)}% to {formatNumber(pct(hi, head.mean_control), 1)}%
            of A2's mean). Better on {formatNumber(head.frac_treatment_better * 100, 0)}% of pairs; Wilcoxon p <P p={head.p_wilcoxon_tile} /> on tile means.
          </p>
          <p className="text-caption text-warn">The price: {BENCH.cost.replace(/`/g, "")} The served model is A2.</p>
        </Card>
        <ConsistencyBars />
      </div>
      <CiChart />
      <div className="grid gap-6 lg:grid-cols-2">
        <BoxPlot metric="l1_spec" title="L1_spec distribution" unit="reflectance" decimals={4} />
        <BoxPlot metric="sam_spec_deg" title="SAM_spec distribution" unit="°" decimals={2} />
      </div>
      <p className="max-w-prose text-caption text-fg-subtle">
        Every learned model sits below the true {SR_GSD} image's own score (the dashed "GT" line). That line is a reference point, not a target:
        a network trained on L1 averages its guesses, and averages are smoother, hence more "consistent", than real imagery. Going further
        below it is a warning sign for over-smoothing, not an achievement.
      </p>
      <BenchTable />
    </div>
  );
}

function ChartCard({ title, caption, children, testId }: { title: string; caption: ReactNode; children: ReactNode; testId: string }) {
  return (
    <Card interactive={false} className="flex flex-col gap-5 p-6" data-testid={testId}>
      <div className="flex flex-col gap-1">
        <h3 className="text-h5">{title}</h3>
        <p className="text-caption text-fg-subtle">{caption}</p>
      </div>
      {children}
    </Card>
  );
}

function Swatch({ label }: { label: string }) {
  return <span aria-hidden className="inline-block size-2.5 shrink-0 rounded-full" style={{ background: COLOR[label] }} />;
}

/** `compact` drops the Served badge for narrow chart gutters; the full name always shows. */
function MethodName({ m, compact = false }: { m: Method; compact?: boolean }) {
  return (
    <span className="flex min-w-0 items-center gap-2">
      <Swatch label={m.label} />
      <span className="text-fg">{m.display}</span>
      {m.label === SERVED_LABEL && !compact && <Badge variant="ours">Served</Badge>}
    </span>
  );
}

function ConsistencyBars() {
  const floor = BENCH.floor.reflectance;
  const max = Math.max(floor, ...BENCH.methods.map((m) => m.means.reflectance)) * 1.1;
  const [hover, setHover] = useState<string | null>(null);
  return (
    <ChartCard testId="bench-bars" title="Consistency error by method" caption={`opensr-test reflectance L1, mean over ${formatNumber(BENCH.n_pairs)} pairs. Lower is more consistent.`}>
      <ul className="flex flex-col gap-4">
        {BENCH.methods.map((m) => (
          <li key={m.label} className="grid grid-cols-[13rem_minmax(0,1fr)] items-center gap-3 text-caption" onMouseEnter={() => setHover(m.label)} onMouseLeave={() => setHover(null)}>
            <MethodName m={m} />
            <div className="relative h-6">
              <div
                className="absolute inset-y-1 left-0 rounded-r-[4px] transition-opacity"
                style={{ width: `${(m.means.reflectance / max) * 100}%`, background: COLOR[m.label], opacity: hover && hover !== m.label ? 0.35 : 1 }}
              />
              <span className="num absolute top-1/2 -translate-y-1/2 pl-2 text-fg" style={{ left: `${(m.means.reflectance / max) * 100}%` }}>
                {formatNumber(m.means.reflectance, 5)}
              </span>
            </div>
          </li>
        ))}
      </ul>
      <FloorAxis max={max} floor={floor} decimals={4} offset="13rem" />
      <p className="min-h-5 text-caption text-fg-muted" aria-live="polite">
        {hover ? hoverLine(hover) : "Hover a bar for its λ weights and role."}
      </p>
    </ChartCard>
  );
}

function hoverLine(label: string): string {
  const m = BENCH.methods.find((x) => x.label === label);
  if (!m) return "";
  return `${m.display}: ${ROLE_NOTE[m.role]}; λ1 / λ2 ${m.lambdas}; checkpoint ${m.checkpoint}; floor ratio ${formatNumber(m.means.reflectance / BENCH.floor.reflectance, 2)}×.`;
}

/** Shared x-axis with the GT floor marker, aligned under a chart column that starts `offset` from the left. */
function FloorAxis({ max, floor, decimals, offset }: { max: number; floor: number; decimals: number; offset: string }) {
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * max);
  return (
    <div className="grid gap-3 text-micro" style={{ gridTemplateColumns: `${offset} minmax(0,1fr)` }}>
      <span />
      <div className="relative h-8 border-t border-line">
        {ticks.map((t) => (
          <span key={t} className="num absolute top-1 -translate-x-1/2 text-fg-subtle" style={{ left: `${(t / max) * 100}%` }}>
            {formatNumber(t, decimals)}
          </span>
        ))}
        <span className="num absolute top-4 -translate-x-1/2 whitespace-nowrap text-fg-muted" style={{ left: `${(floor / max) * 100}%` }}>
          GT {formatNumber(floor, decimals)}
        </span>
        <span aria-hidden className="absolute -top-2 h-2 border-l border-dashed border-fg-muted" style={{ left: `${(floor / max) * 100}%` }} />
      </div>
    </div>
  );
}

const SIG_LABEL: Record<string, string> = {
  reflectance: "opensr reflectance L1 (headline)",
  spectral: "opensr spectral angle",
  l1_spec: "L1_spec (training operator)",
  sam_spec_deg: "SAM_spec (training operator)",
};

function CiChart() {
  const rows = BENCH.significance.map((s) => {
    const [tl, th] = s.ci_tile as [number, number];
    const [pl, ph] = s.ci_pair as [number, number];
    const b = s.mean_control;
    return { s, mean: pct(s.mean_delta, b), tile: [pct(tl, b), pct(th, b)], pair: [pct(pl, b), pct(ph, b)] };
  });
  const lo = Math.floor(Math.min(...rows.map((r) => Math.min(r.tile[0]!, r.pair[0]!))) / 10) * 10 - 10;
  const x = (v: number) => `${((v - lo) / (0 - lo)) * 100}%`;
  const ticks = Array.from({ length: -lo / 10 + 1 }, (_, i) => lo + i * 10);
  return (
    <ChartCard
      testId="bench-ci"
      title="Change from adding the spectral loss (B1 − A2), with 95% confidence intervals"
      caption={`Per-pair mean change as a % of A2's mean. Thick bar: tile-clustered bootstrap CI (${BENCH.n_tiles} tiles, the honest one); thin bar: per-pair CI. All values and p-values as computed in ${BENCH.report_md}; the % is that report's CI divided by A2's mean.`}
    >
      <ul className="flex flex-col gap-5">
        {rows.map(({ s, mean, tile, pair }) => (
          <li key={s.metric} className="grid gap-2 text-caption md:grid-cols-[18rem_minmax(0,1fr)] md:items-center md:gap-3">
            <span className="flex flex-col">
              <span className="text-fg">{SIG_LABEL[s.metric]}</span>
              <span className="num text-micro text-fg-subtle">
                p <P p={s.p_wilcoxon_tile} /> (tile) · {formatNumber(s.frac_treatment_better * 100, 0)}% of pairs better
              </span>
            </span>
            <div className="relative h-7" title={`${formatNumber(mean, 1)}%, tile CI [${formatNumber(tile[0]!, 1)}, ${formatNumber(tile[1]!, 1)}]%`}>
              <span aria-hidden className="absolute inset-y-0 border-l border-line-strong" style={{ left: x(0) }} />
              <span className="absolute top-1/2 h-px -translate-y-1/2 bg-fg-subtle" style={{ left: x(pair[0]!), width: `calc(${x(pair[1]!)} - ${x(pair[0]!)})` }} />
              <span className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full" style={{ left: x(tile[0]!), width: `calc(${x(tile[1]!)} - ${x(tile[0]!)})`, background: COLOR.b1 }} />
              <span className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-ink-850 bg-fg" style={{ left: x(mean) }} />
              <span className="num absolute -top-1 -translate-x-1/2 text-micro text-fg" style={{ left: x(mean) }}>
                {formatNumber(mean, 1)}%
              </span>
            </div>
          </li>
        ))}
      </ul>
      <div className="grid gap-3 text-micro md:grid-cols-[18rem_minmax(0,1fr)]">
        <span className="hidden md:block" />
        <div className="relative h-5 border-t border-line">
          {ticks.map((t) => (
            <span key={t} className="num absolute top-1 -translate-x-1/2 text-fg-subtle" style={{ left: x(t) }}>
              {t}%
            </span>
          ))}
        </div>
      </div>
      <p className="flex flex-wrap items-center gap-3 text-caption text-fg-subtle" data-testid="b2-not-benchmarked">
        <Badge variant="pending">Not yet benchmarked</Badge>
        Run B2 (λ 0.3 / 0.06) has no significance test against A2 in the report; its means appear in the charts above and below only.
      </p>
    </ChartCard>
  );
}

function BoxPlot({ metric, title, unit, decimals }: { metric: DistMetric; title: string; unit: string; decimals: number }) {
  const floor = BENCH.floor[metric];
  const max = Math.max(floor, ...BENCH.methods.map((m) => m.distribution[metric].p95)) * 1.08;
  const x = (v: number) => `${(Math.min(v, max) / max) * 100}%`;
  const [hover, setHover] = useState<string | null>(null);
  const hm = BENCH.methods.find((m) => m.label === hover);
  return (
    <ChartCard testId={`box-${metric}`} title={title} caption={`Per-pair ${unit}, ${formatNumber(BENCH.n_pairs)} pairs per method. Box: quartiles · line: median · dot: mean · whiskers: 5th–95th percentile.`}>
      <ul className="flex flex-col gap-3">
        {BENCH.methods.map((m) => {
          const d = m.distribution[metric];
          const c = COLOR[m.label];
          return (
            <li key={m.label} className="grid grid-cols-[9.5rem_minmax(0,1fr)] items-center gap-3 text-caption" onMouseEnter={() => setHover(m.label)} onMouseLeave={() => setHover(null)}>
              <MethodName m={m} compact />
              <div className="relative h-7" style={{ opacity: hover && hover !== m.label ? 0.4 : 1 }}>
                <span className="absolute top-1/2 h-px -translate-y-1/2" style={{ left: x(d.p5), width: `calc(${x(d.p95)} - ${x(d.p5)})`, background: c }} />
                <span
                  className="absolute inset-y-1 rounded-[4px] border"
                  style={{ left: x(d.q1), width: `calc(${x(d.q3)} - ${x(d.q1)})`, borderColor: c, background: `color-mix(in srgb, ${c} 30%, transparent)` }}
                />
                <span className="absolute inset-y-0.5 w-0.5 -translate-x-1/2 bg-fg" style={{ left: x(d.median) }} />
                <span className="absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-ink-850" style={{ left: x(m.means[metric]), background: c }} />
              </div>
            </li>
          );
        })}
      </ul>
      <FloorAxis max={max} floor={floor} decimals={decimals} offset="9.5rem" />
      <p className="num min-h-5 text-caption text-fg-muted" aria-live="polite">
        {hm
          ? `${hm.display}: median ${formatNumber(hm.distribution[metric].median, decimals + 1)}, IQR ${formatNumber(hm.distribution[metric].q1, decimals + 1)}–${formatNumber(hm.distribution[metric].q3, decimals + 1)}, mean ${formatNumber(hm.means[metric], decimals + 1)} ${unit}`
          : "Hover a row for its quartiles."}
      </p>
    </ChartCard>
  );
}

function BenchTable() {
  const cols: { key: string; label: string; get: (m: Method) => number; d: number }[] = [
    { key: "refl", label: "opensr L1", get: (m) => m.means.reflectance, d: 5 },
    { key: "spec", label: "opensr angle °", get: (m) => m.means.spectral, d: 3 },
    { key: "l1", label: "L1_spec mean", get: (m) => m.means.l1_spec, d: 5 },
    { key: "l1med", label: "L1_spec median", get: (m) => m.distribution.l1_spec.median, d: 5 },
    { key: "sam", label: "SAM_spec mean °", get: (m) => m.means.sam_spec_deg, d: 3 },
    { key: "sammed", label: "SAM_spec median °", get: (m) => m.distribution.sam_spec_deg.median, d: 3 },
  ];
  return (
    <details className="card card-static p-0" data-testid="bench-table">
      <summary className="cursor-pointer p-4 text-caption font-medium text-fg-muted">Show the numbers as a table</summary>
      <div className="overflow-x-auto border-t border-line">
        <table className="num w-full text-left text-caption">
          <thead className="text-fg-subtle">
            <tr>
              <th className="p-3 font-medium">Method</th>
              <th className="p-3 font-medium">λ1 / λ2</th>
              {cols.map((c) => (
                <th key={c.key} className="p-3 text-right font-medium">
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {BENCH.methods.map((m) => (
              <tr key={m.label} className="border-t border-line">
                <td className="p-3">
                  <MethodName m={m} />
                </td>
                <td className="p-3 text-fg-muted">{m.lambdas}</td>
                {cols.map((c) => (
                  <td key={c.key} className="p-3 text-right text-fg">
                    {formatNumber(c.get(m), c.d)}
                  </td>
                ))}
              </tr>
            ))}
            <tr className="border-t border-line text-fg-muted italic">
              <td className="p-3">True {SR_GSD} image (GT floor)</td>
              <td className="p-3">—</td>
              <td className="p-3 text-right">{formatNumber(BENCH.floor.reflectance, 5)}</td>
              <td className="p-3 text-right">{formatNumber(BENCH.floor.spectral, 3)}</td>
              <td className="p-3 text-right">{formatNumber(BENCH.floor.l1_spec, 5)}</td>
              <td className="p-3 text-right">—</td>
              <td className="p-3 text-right">{formatNumber(BENCH.floor.sam_spec_deg, 3)}</td>
              <td className="p-3 text-right">—</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="border-t border-line p-4 text-caption text-fg-subtle">
        Means from {BENCH.source} (written {BENCH.written_utc}); medians from the harness's per-pair tables for the same checkpoints.
      </p>
    </details>
  );
}

// ---------------------------------------------------------- how it works ----

function HowItWorks() {
  return (
    <>
      <p>
        <strong className="text-fg">The check.</strong> Sentinel-2 records each {LR_GSD} pixel as roughly the average of the ground inside it. So we
        take our {SR_GSD} output, average every {APP_CONFIG.scale}×{APP_CONFIG.scale} block back into one {LR_GSD} pixel (the operator D), and compare
        with the real input. Two numbers come out: the L1 difference in surface reflectance (how much brightness moved) and the spectral angle
        between the two {APP_CONFIG.bands.length}-band spectra (how much the colour, including near-infrared, changed). Reflectance is kept in physical
        units throughout: no ImageNet normalisation, no silent clipping.
      </p>
      <p>
        <strong className="text-fg">The loss.</strong> During training the same comparison becomes a penalty added to the usual pixel loss: one
        weight (λ1) on the L1 term and one (λ2) on the angle term. Run A2 uses no spectral penalty; Run B1 uses λ 0.1 / 0.02; Run B2, 0.3 / 0.06.
      </p>
      <p>
        <strong className="text-fg">The limitation, stated plainly: the blur shortcut.</strong> This loss has a degenerate minimum. Any image that is
        a smooth blow-up of the input, such as plain bicubic, already averages back to almost exactly the input, so it scores near zero while adding
        no real detail. That is why bicubic beats every model on this measure, and why the stronger B2 lands far below the true {SR_GSD}
        image's own score while losing PSNR and LPIPS. Pushed hard, the penalty rewards blur. We therefore never read the consistency number alone:
        it is always shown next to a sharpness check (high-frequency energy relative to bicubic), and the true image's score is marked as a
        reference, not a goal.
      </p>
      <p>
        <strong className="text-fg">What we serve.</strong> On the validation set the spectral loss did make outputs measurably more consistent (the
        −33% above), but it cost sharpness and PSNR, so the deployed model is A2, trained without it. An inference-time consistency projection
        was also tried and failed its pre-registered gate, so it is not served. The consistency check itself runs on every image you upload.
      </p>
    </>
  );
}
