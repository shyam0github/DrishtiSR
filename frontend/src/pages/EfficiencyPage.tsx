import type { ReactNode } from "react";
import type { ImageResult, UncertaintyMethod, UpscaleResponse } from "../api/client";
import { Badge, MetricBar, PendingStatCard, StatCard } from "../components/ui";
import { APP_CONFIG } from "../config";
import { FACTS, FACT_SOURCES } from "../data/facts";
import { SR_MODELS, type SrModelRef } from "../data/srModels";
import { tileCount } from "../data/tiling";
import { formatNumber } from "../design/motion";
import { NoveltyPageLayout, type ImageSource } from "../layout/NoveltyPageLayout";

const TILE = APP_CONFIG.tile;
const BENCH_IN = `${FACTS.benchLrPx}² px`;
const BENCH_OUT = `${FACTS.benchLrPx * APP_CONFIG.scale}² px`;

const mb = (bytes: number) => bytes / 1e6;
const fmtMs = (ms: number) => `${formatNumber(ms, 0)} ms`;
const fmtS = (ms: number) => `${formatNumber(ms / 1000, 2)} s`;

const UNC_LABEL: Record<UncertaintyMethod, string> = { none: "none", tta4: "TTA-4", tta8: "TTA-8", learned_laplace: "learned head" };

export function EfficiencyPage() {
  return (
    <NoveltyPageLayout
      testId="page-efficiency"
      index="03"
      title="CPU-only efficiency"
      summary={
        <>
          A super-resolution model is only useful where it can run. DrishtiSR has {formatNumber(FACTS.params)} parameters, ships as a{" "}
          {formatNumber(mb(FACTS.onnxFp32Bytes), 2)} MB ONNX file, and super-resolves a {BENCH_IN} tile in {fmtS(FACTS.onnxFp32MedianMs)} on {FACTS.benchThreads} laptop CPU
          threads, with no GPU. Every latency and size here was measured. INT8 quantisation failed its accuracy gate, and the page reports that failure.
        </>
      }
      renderImage={(result, source) => <ImageTiming result={result} source={source} />}
      benchmarkLabel={`${BENCH_IN} → ${BENCH_OUT} · median of ${FACTS.benchRuns} runs · laptop CPU`}
      benchmarkSource={`${FACT_SOURCES.bench}, ${FACT_SOURCES.bench1t}, ${FACT_SOURCES.int8}`}
      benchmark={<Benchmark />}
      howItWorks={<HowItWorks />}
    />
  );
}

// ------------------------------------------------------------ this image ----

function ImageTiming({ result, source }: { result: ImageResult; source: ImageSource }) {
  return (
    <div className="flex flex-col gap-8" data-testid="eff-image" data-source={source}>
      <div className="flex flex-wrap items-center gap-3 text-caption text-fg-muted">
        {source === "example" ? <Badge variant="neutral">Example</Badge> : <Badge variant="ours">Your image</Badge>}
        <span className="truncate">{result.sourceLabel}</span>
      </div>
      {result.mode === "mock" || !result.raw ? <MockTiming /> : <MeasuredTiming result={result} raw={result.raw} />}
    </div>
  );
}

function MockTiming() {
  const reason = "Mock mode: no model ran, so there is no timing to report.";
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4" data-testid="eff-image-timing">
      <PendingStatCard size="sm" label="Model inference" reason={reason} />
      <PendingStatCard size="sm" label="Per tile" reason={reason} />
      <PendingStatCard size="sm" label="Whole job" reason={reason} />
      <PendingStatCard size="sm" label="CPU threads" reason={reason} />
    </div>
  );
}

/** The server's own stage timings for this job (runtime_ms in the /api/upscale response). */
function MeasuredTiming({ result, raw }: { result: ImageResult; raw: UpscaleResponse }) {
  const rt = raw.metrics.reference_free.runtime_ms;
  const [h, w] = raw.input.lr_size;
  const [sh, sw] = raw.input.sr_size;
  const tiles = tileCount(h, w, TILE.lrPx, TILE.overlapLrPx);
  const unc = rt.uncertainty ?? 0;
  const other = rt.total - rt.sr - unc;
  const outMpxPerS = (sh * sw) / 1e6 / (rt.sr / 1000);
  const method = result.uncertaintyMethod ?? raw.uncertainty_method;

  const segments = [
    { key: "sr", label: "Model inference", ms: rt.sr, cls: "bg-accent" },
    { key: "unc", label: `Uncertainty (${UNC_LABEL[method]})`, ms: unc, cls: "bg-lit" },
    { key: "other", label: "Metrics, renders, GeoTIFFs", ms: other, cls: "bg-fg-subtle" },
  ].filter((s) => s.ms > 0);

  return (
    <div className="flex flex-col gap-8" data-testid="eff-image-timing">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard size="sm" accent value={rt.sr} unit="ms" label="Model inference" hint={`${h}×${w} → ${sh}×${sw} px · ${formatNumber(outMpxPerS, 2)} output Mpx/s`} />
        <StatCard
          size="sm"
          value={rt.sr / tiles}
          unit="ms"
          label="Per tile"
          hint={tiles === 1 ? `1 tile: the image fits in one ${TILE.lrPx}² px tile` : `${tiles} tiles of ${TILE.lrPx}² px, ${TILE.overlapLrPx} px overlap`}
        />
        <StatCard size="sm" value={rt.total} unit="ms" label="Whole job" hint={rt.uncertainty === null ? "uncertainty off for this run" : `includes ${UNC_LABEL[method]} uncertainty`} />
        <StatCard size="sm" value={raw.model.threads} unit="threads" label="CPU threads" hint={`${raw.model.backend} · no GPU`} />
      </div>

      <div className="flex flex-col gap-3">
        <p className="eyebrow">Where the time went</p>
        <div className="flex h-3 gap-0.5 overflow-hidden rounded-full bg-ink-800" role="img" aria-label={segments.map((s) => `${s.label} ${fmtMs(s.ms)}`).join(", ")}>
          {segments.map((s) => (
            <div key={s.key} className={`h-full ${s.cls}`} style={{ width: `${(s.ms / rt.total) * 100}%` }} title={`${s.label}: ${fmtMs(s.ms)}`} />
          ))}
        </div>
        <ul className="flex flex-wrap gap-x-6 gap-y-2 text-caption text-fg-muted">
          {segments.map((s) => (
            <li key={s.key} className="flex items-center gap-2">
              <span aria-hidden className={`size-2.5 rounded-sm ${s.cls}`} />
              {s.label} <span className="num text-fg">{fmtMs(s.ms)}</span>
              <span className="num text-fg-subtle">({formatNumber((s.ms / rt.total) * 100, 0)}%)</span>
            </li>
          ))}
        </ul>
        {other < 0 && <p className="text-caption text-warn">The server reported stage times that exceed its total ({fmtMs(rt.total)}); the breakdown omits the remainder.</p>}
        <p className="text-caption text-fg-subtle">
          Timed by the server around each stage of this job. Upload and download are not included. For scale: the benchmark {BENCH_IN} tile takes{" "}
          {fmtS(FACTS.onnxFp32MedianMs)} (median, model only, {FACTS.benchThreads} threads).
        </p>
      </div>
    </div>
  );
}

// ------------------------------------------------------------- benchmark ----

function Benchmark() {
  return (
    <div className="flex flex-col gap-16">
      <ParamChart />
      <div className="grid gap-12 lg:grid-cols-2">
        <ModelSize />
        <Latency />
      </div>
      <QuantImpact />
    </div>
  );
}

function SubHead({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-h4">{title}</h3>
      {children && <p className="max-w-prose text-caption text-fg-muted">{children}</p>}
    </div>
  );
}

const SOURCE_BADGE: Record<SrModelRef["source"], ReactNode> = {
  ours: <Badge variant="ours">Ours</Badge>,
  "in-repo": <Badge variant="baseline">Measured here</Badge>,
  literature: <Badge variant="literature">Literature</Badge>,
};

/** Log axis: 10^LOG_LO .. 10^LOG_HI parameters. */
const LOG_LO = 4;
const LOG_HI = 8;
const TICKS = [
  { v: 1e4, label: "10K" },
  { v: 1e5, label: "100K" },
  { v: 1e6, label: "1M" },
  { v: 1e7, label: "10M" },
  { v: 1e8, label: "100M" },
];
const logX = (v: number) => ((Math.log10(v) - LOG_LO) / (LOG_HI - LOG_LO)) * 100;
const compact = (n: number) => (n >= 1e6 ? `${formatNumber(n / 1e6, n >= 1e7 ? 1 : 2)}M` : `${formatNumber(n / 1e3, 0)}K`);
const ROW_GRID = "grid grid-cols-[minmax(0,8rem)_minmax(0,1fr)_4rem] gap-3 sm:grid-cols-[minmax(0,19rem)_minmax(0,1fr)_5rem]";

/**
 * Parameter counts on a log axis. Dots, not bars: a bar's length would have to
 * start at zero, which a log scale does not have. Ours is the only accent mark;
 * the source of every other number is a text badge, never colour alone.
 */
function ParamChart() {
  const rows = [...SR_MODELS].sort((a, b) => a.params - b.params);
  const budgetX = logX(APP_CONFIG.maxParameters);
  return (
    <div className="flex flex-col gap-6" data-testid="eff-param-chart">
      <SubHead title="Parameter count vs well-known SR networks">
        Log scale. The dashed line is the {formatNumber(APP_CONFIG.maxParameters)}-parameter budget. Literature counts are the ×4 RGB models as their papers report them;
        they were not measured here.
      </SubHead>
      <div className="flex flex-col">
        {rows.map((m) => {
          const ours = m.source === "ours";
          return (
            <div
              key={m.name}
              tabIndex={0}
              aria-label={`${m.name}: ${formatNumber(m.params)} parameters (${m.cite})`}
              className={`group ${ROW_GRID} items-center rounded-sm px-1 py-1.5 outline-none hover:bg-ink-800/60 focus-visible:bg-ink-800/60`}
            >
              <span className="flex min-w-0 flex-col gap-1 sm:flex-row sm:items-center sm:gap-2">
                <span className={`truncate text-caption ${ours ? "font-semibold text-fg" : "text-fg-muted"}`}>{m.name}</span>
                <span className="hidden sm:inline">{SOURCE_BADGE[m.source]}</span>
              </span>
              <span className="relative h-5">
                {TICKS.map((t) => (
                  <span key={t.v} aria-hidden className="absolute inset-y-0 w-px bg-line" style={{ left: `${logX(t.v)}%` }} />
                ))}
                <span aria-hidden className="absolute inset-y-0 border-l border-dashed border-warn/60" style={{ left: `${budgetX}%` }} />
                <span
                  aria-hidden
                  className={`absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-ink-950 transition-transform group-hover:scale-125 ${
                    ours ? "size-3.5 bg-accent shadow-[0_0_10px_var(--color-accent)]" : "size-2.5 bg-fg-subtle"
                  }`}
                  style={{ left: `${logX(m.params)}%` }}
                />
                <span
                  role="tooltip"
                  className="pointer-events-none absolute bottom-full z-10 mb-2 hidden -translate-x-1/2 whitespace-nowrap rounded-sm border border-line-strong bg-ink-800 px-2 py-1 text-caption text-fg shadow-lg group-hover:block group-focus-visible:block"
                  style={{ left: `${Math.min(Math.max(logX(m.params), 20), 80)}%` }}
                >
                  <span className="num">{formatNumber(m.params)}</span> params · <span className="text-fg-muted">{m.cite}</span>
                </span>
              </span>
              <span className={`num text-right text-caption ${ours ? "text-fg" : "text-fg-muted"}`}>{compact(m.params)}</span>
            </div>
          );
        })}
        <div aria-hidden className={`${ROW_GRID} px-1 pt-1`}>
          <span />
          <span className="relative h-4">
            {TICKS.map((t) => (
              <span key={t.v} className="num absolute -translate-x-1/2 text-micro text-fg-subtle" style={{ left: `${logX(t.v)}%` }}>
                {t.label}
              </span>
            ))}
          </span>
          <span />
        </div>
      </div>
      <details className="text-caption text-fg-muted">
        <summary className="cursor-pointer select-none hover:text-fg">Show as a table, with sources</summary>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[32rem] text-left">
            <thead className="text-fg-subtle">
              <tr>
                <th className="py-1.5 pr-4 font-medium">Model</th>
                <th className="py-1.5 pr-4 text-right font-medium">Parameters</th>
                <th className="py-1.5 pr-4 font-medium">Source</th>
                <th className="py-1.5 font-medium">Reference</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((m) => (
                <tr key={m.name} className="border-t border-line">
                  <td className="py-1.5 pr-4 text-fg">{m.name}</td>
                  <td className="num py-1.5 pr-4 text-right">{formatNumber(m.params)}</td>
                  <td className="py-1.5 pr-4">{SOURCE_BADGE[m.source]}</td>
                  <td className="py-1.5">{m.cite}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

function ModelSize() {
  return (
    <div className="flex flex-col gap-6" data-testid="eff-model-size">
      <SubHead title="Model size on disk">The served FP32 graph, and the INT8 graph that was built and benchmarked but failed its accuracy gate.</SubHead>
      <div className="grid gap-4 sm:grid-cols-2">
        <StatCard size="sm" accent value={mb(FACTS.onnxFp32Bytes)} decimals={2} unit="MB" label="ONNX FP32" badge={<Badge variant="good">Served</Badge>} hint={`${formatNumber(FACTS.onnxFp32Bytes)} bytes`} />
        <StatCard
          size="sm"
          value={mb(FACTS.int8Bytes)}
          decimals={2}
          unit="MB"
          label="ONNX INT8 (attempt C)"
          badge={<Badge variant="bad">Gate failed</Badge>}
          hint={`${formatNumber(FACTS.onnxFp32Bytes / FACTS.int8Bytes, 1)}× smaller · not served`}
        />
      </div>
    </div>
  );
}

function Latency() {
  const rows = [
    { name: "ONNX FP32 (served)", t1: FACTS.t1OnnxFp32MedianMs, t6: FACTS.onnxFp32MedianMs, served: true },
    { name: "ONNX INT8, attempt C (not served)", t1: FACTS.t1Int8MedianMs, t6: FACTS.int8MedianMs, served: false },
  ];
  const max = Math.max(...rows.flatMap((r) => [r.t1, r.t6]));
  return (
    <div className="flex flex-col gap-6" data-testid="eff-latency">
      <SubHead title={`CPU latency: 1 thread vs ${FACTS.benchThreads}`}>
        Median per {BENCH_IN} tile. Six threads buy about 2×, not 6×: a network this small does too little work per layer to keep six cores busy.
      </SubHead>
      <div className="flex flex-col gap-8">
        {rows.map((r) => (
          <div key={r.name} className="flex flex-col gap-3">
            <p className="flex flex-wrap items-center gap-2 text-caption text-fg">
              {r.name}
              <Badge variant="neutral">{formatNumber(r.t1 / r.t6, 2)}× faster at {FACTS.benchThreads} threads</Badge>
            </p>
            <MetricBar label="1 thread" value={r.t1} max={max} decimals={0} unit="ms" tone="muted" />
            <MetricBar label={`${FACTS.benchThreads} threads`} value={r.t6} max={max} decimals={0} unit="ms" tone={r.served ? "accent" : "muted"} />
          </div>
        ))}
      </div>
      <p className="text-caption text-fg-subtle">
        {FACTS.benchThreads} threads measured 2026-09-11, 1 thread 2026-09-12, same machine and input. Both runs started with CPU load under 20%, so neither is marked
        provisional.
      </p>
    </div>
  );
}

function QuantImpact() {
  const g = FACTS.quantGate;
  const c = FACTS.quantAttempts.find((a) => a.attempt === "C");
  if (!c) throw new Error("facts.quantAttempts has no attempt C (the benchmarked INT8 graph)");
  // Gate: positive = INT8 worse; each must be <= its limit. Shown here as INT8 minus FP32.
  const cells = (a: (typeof FACTS.quantAttempts)[number]) => [
    { v: -a.dPsnrDb, d: 3, unit: "dB", ok: a.dPsnrDb <= g.dPsnrDb },
    { v: a.dSamDeg, d: 3, unit: "°", ok: a.dSamDeg <= g.dSamDeg },
    { v: a.dLpips, d: 4, unit: "", ok: a.dLpips <= g.dLpips },
  ];
  const signed = (v: number, d: number) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${formatNumber(Math.abs(v), d)}`;
  return (
    <div className="flex flex-col gap-6" data-testid="eff-quant">
      <SubHead title="INT8 quantisation: what it costs in quality">
        INT8 minus FP32 on the same {FACTS.quantGateN} validation tiles, for each of the four quantisation attempts. Every attempt failed at least one limit, so the FP32 graph
        is what runs. A cell marked ✕ exceeds its limit.
      </SubHead>
      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard size="sm" value={-c.dPsnrDb} decimals={2} unit="dB" label="PSNR change, best attempt (C)" badge={<Badge variant="bad">✕ limit −{g.dPsnrDb}</Badge>} />
        <StatCard size="sm" value={c.dSamDeg} decimals={3} unit="°" prefix="+" label="Spectral angle change (C)" badge={<Badge variant="bad">✕ limit +{g.dSamDeg}</Badge>} />
        <StatCard size="sm" value={c.dLpips} decimals={4} label="LPIPS change (C)" badge={<Badge variant="good">✓ within limit</Badge>} hint="lower is better; INT8 scored slightly better" />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[44rem] text-left text-caption">
          <thead className="text-fg-subtle">
            <tr>
              <th className="py-2 pr-4 font-medium">Attempt</th>
              <th className="py-2 pr-4 font-medium">Calibration</th>
              <th className="py-2 pr-4 font-medium">Kept in FP32</th>
              <th className="py-2 pr-4 text-right font-medium">Size</th>
              <th className="py-2 pr-4 text-right font-medium">ΔPSNR (≥ −{g.dPsnrDb} dB)</th>
              <th className="py-2 pr-4 text-right font-medium">ΔSAM (≤ +{g.dSamDeg}°)</th>
              <th className="py-2 pr-4 text-right font-medium">ΔLPIPS (≤ +{g.dLpips})</th>
              <th className="py-2 font-medium">Gate</th>
            </tr>
          </thead>
          <tbody>
            {FACTS.quantAttempts.map((a) => (
              <tr key={a.attempt} className="border-t border-line text-fg-muted">
                <td className="py-2 pr-4 text-fg">{a.attempt}</td>
                <td className="py-2 pr-4">{a.calibration}</td>
                <td className="py-2 pr-4">{a.keptFp32 ?? "—"}</td>
                <td className="num py-2 pr-4 text-right">{formatNumber(mb(a.bytes), 2)} MB</td>
                {cells(a).map((cell, i) => (
                  <td key={i} className={`num py-2 pr-4 text-right ${cell.ok ? "text-fg" : "text-bad"}`}>
                    {signed(cell.v, cell.d)}
                    {cell.unit && ` ${cell.unit}`} <span aria-label={cell.ok ? "within limit" : "exceeds limit"}>{cell.ok ? "✓" : "✕"}</span>
                  </td>
                ))}
                <td className="py-2">
                  <Badge variant="bad">Failed</Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-caption text-fg-subtle">
        Static quantisation calibrated on {FACTS.quantCalibN} training tiles; gate on {FACTS.quantGateN} validation tiles. The limits are the ones fixed before the attempts
        ran; none was relaxed afterwards.
      </p>
    </div>
  );
}

// ---------------------------------------------------------- how it works ----

function HowItWorks() {
  const c = FACTS.quantAttempts[2];
  return (
    <>
      <p className="text-fg">From checkpoint to CPU, in five steps:</p>
      <ol className="flex list-decimal flex-col gap-3 pl-5 marker:text-accent">
        <li>
          <span className="text-fg">Train small, in reflectance.</span> An EDSR-style network of 16 residual blocks × 48 features, {formatNumber(FACTS.params)} parameters. The
          code asserts it stays under the {formatNumber(APP_CONFIG.maxParameters)} budget. Inputs and outputs are surface reflectance, so there is no normalisation layer to
          carry into deployment.
        </li>
        <li>
          <span className="text-fg">Export to ONNX.</span> The PyTorch graph is exported as ONNX FP32 and checked against PyTorch on the same input. The largest difference is{" "}
          {FACTS.onnxParityMaxAbsDiff} in reflectance, far below any metric's resolution.
        </li>
        <li>
          <span className="text-fg">Quantise to INT8.</span> ONNX Runtime static quantisation, calibrated on {FACTS.quantCalibN} training tiles. Four attempts ran, from plain
          MinMax up to percentile calibration with the most sensitive layers kept in FP32: the first and output convolutions in one attempt, the upsampler in another.
        </li>
        <li>
          <span className="text-fg">Gate on accuracy.</span> Each INT8 graph is scored against FP32 on {FACTS.quantGateN} validation tiles, with fixed limits on PSNR,
          spectral angle and LPIPS. None passed. The best attempt is {formatNumber(FACTS.onnxFp32Bytes / FACTS.int8Bytes, 1)}× smaller and{" "}
          {formatNumber(FACTS.onnxFp32MedianMs / FACTS.int8MedianMs, 2)}× faster, but costs {formatNumber(c.dPsnrDb, 2)} dB, so it is not served.
        </li>
        <li>
          <span className="text-fg">Serve on the CPU.</span> ONNX Runtime runs with {FACTS.benchThreads} intra-op threads and full graph optimisation. Large scenes run as{" "}
          {TILE.lrPx}² px tiles with {TILE.overlapLrPx} px overlap, blended with a raised-cosine window. The TTA-4 uncertainty map reruns the same graph on flipped and
          rotated copies of the input. It needs no extra operators, so it survives export unchanged.
        </li>
      </ol>
      <p className="text-fg">Why CPU-only matters:</p>
      <ul className="flex list-disc flex-col gap-3 pl-5">
        <li>
          <span className="text-fg">Field and edge hardware.</span> District offices, disaster-response teams and field survey units work on laptops and desktops, and few of
          them have a CUDA GPU. The benchmark machine is a 2018 laptop CPU.
        </li>
        <li>
          <span className="text-fg">No GPU cloud in the loop.</span> Inference runs on the machine that holds the imagery. Nothing goes to a rented GPU, and the tool keeps
          working where connectivity is poor.
        </li>
        <li>
          <span className="text-fg">Cheap to install and scale.</span> A {formatNumber(mb(FACTS.onnxFp32Bytes), 2)} MB model fits in any installer or container. More
          throughput needs ordinary CPU cores, not scarce accelerators.
        </li>
      </ul>
      <p>
        Limits: every latency is for one CPU model (i7-8750H). Peak memory was not recorded. Threading scales poorly for a model this small: about 2× from 1 to{" "}
        {FACTS.benchThreads} threads.
      </p>
    </>
  );
}
