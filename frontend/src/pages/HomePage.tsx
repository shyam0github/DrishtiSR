import { useEffect, useState, type DragEvent, type MouseEvent, type ReactNode } from "react";
import { getSamples, MOCK_MODE, type ImageResult, type SampleInfo } from "../api/client";
import {
  Badge,
  BeforeAfterSlider,
  Button,
  ButtonAnchor,
  ButtonLink,
  Card,
  Icon,
  ImageSkeleton,
  PendingStatCard,
  Reveal,
  SectionHeader,
  StatCard,
  StatCardSkeleton,
  type IconName,
} from "../components/ui";
import { APP_CONFIG } from "../config";
import { FACTS } from "../data/facts";
import { formatNumber, prefersReducedMotion } from "../design/motion";
import { useResult, type RunStatus } from "../state/ResultContext";

const UPLOAD_SECTION_ID = "try-it";
const LR_GSD = `${APP_CONFIG.lrGsdM} m`;
const SR_GSD = `${APP_CONFIG.lrGsdM / APP_CONFIG.scale} m`;
/** Keep a square viewer inside one screen: viewport minus the sticky nav and a margin. */
const VIEWER_MAX_WIDTH = "min(100%, calc(100dvh - var(--spacing-nav) - 7rem))";

const fmtMs = (ms: number) => `${formatNumber(ms / 1000, 2)} s`;
const fmtMb = (bytes: number) => formatNumber(bytes / 1e6, 2);

const CLAIMS: { icon: IconName; title: string; body: string }[] = [
  { icon: "spectral", title: "Spectrally consistent", body: `Degrade the output back to ${LR_GSD} and it should return the input reflectance. Measured on every image.` },
  { icon: "uncertainty", title: "Uncertainty-aware", body: "Every output ships with a per-pixel map of where the model is least sure." },
  { icon: "cpu", title: "CPU-only", body: `${formatNumber(FACTS.params)} parameters, ONNX, ${FACTS.benchThreads} CPU threads. No GPU at inference.` },
];

const NOVELTIES: { index: string; icon: IconName; title: string; pitch: string; to: string; metric: string; metricLabel: string }[] = [
  {
    index: "01",
    icon: "spectral",
    title: "Spectral consistency",
    pitch: `The ${SR_GSD} output is checked against physics: degraded to ${LR_GSD}, it must reproduce the Sentinel-2 reflectance it came from. No ImageNet normalisation, no silent clipping.`,
    to: "/novelty/spectral",
    metric: `${formatNumber(FACTS.consistencyVsHrReference, 2)}×`,
    metricLabel: `consistency error relative to the true ${SR_GSD} image (opensr L1 ${formatNumber(FACTS.consistencyServed, 5)} vs ${formatNumber(FACTS.consistencyHrReference, 5)}, val n = ${formatNumber(FACTS.valPairs)})`,
  },
  {
    index: "02",
    icon: "uncertainty",
    title: "Per-pixel uncertainty",
    pitch: "A hallucination map alongside every result, so an analyst can see which new detail to trust. Served from test-time augmentation disagreement.",
    to: "/novelty/uncertainty",
    metric: `ρ ${formatNumber(FACTS.uncSpearman, 2)}`,
    metricLabel: `rank correlation of predicted uncertainty with actual error (TTA-4, AUSE ${formatNumber(FACTS.uncAuse, 4)}, val n = ${FACTS.uncN})`,
  },
  {
    index: "03",
    icon: "cpu",
    title: "CPU-only deployment",
    pitch: `Under a million parameters, exported to ONNX and timed on a laptop CPU. Latency and model size are measured, not asserted.`,
    to: "/novelty/efficiency",
    metric: fmtMs(FACTS.onnxFp32MedianMs),
    metricLabel: `median per ${FACTS.benchLrPx}² px tile → ${FACTS.benchLrPx * APP_CONFIG.scale}² px, ONNX FP32, ${FACTS.benchThreads} threads (${FACTS.benchRuns} runs)`,
  },
];

const DEEP_LINKS = [
  { to: "/novelty/spectral", label: "View spectral analysis" },
  { to: "/novelty/uncertainty", label: "View uncertainty map" },
  { to: "/novelty/efficiency", label: "View efficiency stats" },
];

export function HomePage() {
  // Novelty pages link to /#try-it; the router does not scroll to hashes by itself.
  useEffect(() => {
    if (window.location.hash === `#${UPLOAD_SECTION_ID}`) document.getElementById(UPLOAD_SECTION_ID)?.scrollIntoView();
  }, []);
  return (
    <>
      <Hero />
      <LiveDemo />
      <Novelties />
      <ImpactStrip />
    </>
  );
}

// ------------------------------------------------------------------ hero ----

function scrollToUpload(e: MouseEvent) {
  e.preventDefault();
  document.getElementById(UPLOAD_SECTION_ID)?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth" });
}

function Hero() {
  return (
    <section className="relative overflow-hidden" data-testid="home-hero">
      <div aria-hidden className="bg-graticule absolute inset-0" />
      <div aria-hidden className="absolute -top-40 left-1/2 h-96 w-[48rem] max-w-full -translate-x-1/2 rounded-full bg-accent/10 blur-3xl" />
      <div className="page-container relative flex flex-col gap-10 pt-[calc(var(--spacing-section)*0.75)] pb-section">
        <Reveal className="flex flex-col gap-6">
          <p className="eyebrow">SIH26142 · Sentinel-2 L2A · {APP_CONFIG.bands.join(" ")}</p>
          <h1 className="max-w-4xl text-display">
            Sentinel-2 at <span className="text-accent">{SR_GSD}</span>, on a laptop CPU.
          </h1>
          <p className="max-w-prose text-lead text-fg-muted">
            ×{APP_CONFIG.scale} resolution recovery for {LR_GSD} Sentinel-2 imagery — CPU-deployable, and honest about where it is guessing.
          </p>
        </Reveal>
        <Reveal as="ul" className="grid gap-3 sm:grid-cols-3" stagger={90}>
          {CLAIMS.map((c) => (
            <li key={c.title} className="flex gap-3 rounded-lg border border-line bg-ink-900/60 p-4 backdrop-blur">
              <IconTile name={c.icon} />
              <div className="flex flex-col gap-1">
                <h2 className="text-h6">{c.title}</h2>
                <p className="text-caption text-fg-muted">{c.body}</p>
              </div>
            </li>
          ))}
        </Reveal>
        <div className="flex flex-wrap gap-3">
          <ButtonAnchor href={`#${UPLOAD_SECTION_ID}`} onClick={scrollToUpload} size="lg" data-testid="hero-cta">
            Try it on your image
            <Icon name="arrow-down" />
          </ButtonAnchor>
          <ButtonLink to="/compare" variant="secondary" size="lg">
            See the comparison
          </ButtonLink>
        </div>
      </div>
    </section>
  );
}

function IconTile({ name }: { name: IconName }) {
  return (
    <span aria-hidden className="grid size-10 shrink-0 place-items-center rounded-md border border-accent/35 bg-accent-dim text-accent">
      <Icon name={name} size={18} />
    </span>
  );
}

// ------------------------------------------------------------- live demo ----

function LiveDemo() {
  const { status, result, reset } = useResult();
  return (
    <section id={UPLOAD_SECTION_ID} className="scroll-mt-nav border-y border-line bg-ink-900/50" data-testid="live-demo">
      <div className="page-container flex flex-col gap-10 py-section">
        <SectionHeader
          eyebrow="Live demo"
          title="Upload a tile. Inspect every pixel."
          description={`A 4-band GeoTIFF (${APP_CONFIG.bands.join(", ")}) at ${LR_GSD}. Drag the divider, or click the image to open it full-size with zoom and pan.`}
          actions={
            result && (
              <Button variant="secondary" onClick={reset} disabled={status === "running"}>
                <Icon name="upload" />
                New image
              </Button>
            )
          }
        />
        {MOCK_MODE && (
          <p className="flex flex-wrap items-center gap-3 text-caption text-fg-muted" data-testid="mock-banner">
            <Badge variant="warn">Mock mode</Badge>
            No backend: an upload returns the placeholder pair and every metric stays pending. Run scripts/serve.py and start the UI with
            VITE_MOCK_MODE=false.
          </p>
        )}
        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <div className="mx-auto w-full" style={{ maxWidth: VIEWER_MAX_WIDTH }}>
            <Viewer />
          </div>
          <aside className="flex flex-col gap-6">
            <QuickMetrics status={status} result={result} />
            <nav aria-label="This image in depth" className="flex flex-col gap-2">
              {DEEP_LINKS.map((l) => (
                <ButtonLink key={l.to} to={l.to} variant="secondary" className="w-full justify-between">
                  {l.label}
                  <Icon name="arrow-right" />
                </ButtonLink>
              ))}
            </nav>
          </aside>
        </div>
      </div>
    </section>
  );
}

function Viewer() {
  const { status, result, input, error, run } = useResult();

  if (status === "running") return <ImageSkeleton aspect="1 / 1" label={`Super-resolving ${input?.label ?? "image"}…`} />;

  if (status === "done" && result) {
    return (
      <div className="flex flex-col gap-3">
        <BeforeAfterSlider
          key={result.jobId}
          beforeSrc={result.images.lr}
          afterSrc={result.images.sr}
          beforeLabel={`${LR_GSD} input`}
          afterLabel={`${SR_GSD} ×${APP_CONFIG.scale}`}
          alt={`${result.sourceLabel}${result.lrSize && result.srSize ? ` · ${result.lrSize.join("×")} → ${result.srSize.join("×")} px` : ""}`}
          aspect="1 / 1"
        />
        <ResultMeta result={result} />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {status === "error" && error && (
        <div role="alert" data-testid="upscale-error" className="rounded-md border border-bad/40 bg-bad/10 p-4 text-caption text-bad">
          Super-resolution failed: {error}
        </div>
      )}
      <Dropzone onFile={(file) => void run({ file }, file.name)} />
      <Samples onPick={(s) => void run({ sampleId: s.id }, s.label)} />
    </div>
  );
}

function ResultMeta({ result }: { result: ImageResult }) {
  return (
    <div className="flex flex-col gap-2 text-caption text-fg-subtle">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        {result.model && (
          <span className="num">
            {result.model.backend} · {formatNumber(result.model.params)} params · {result.model.threads} threads
          </span>
        )}
        {result.metrics.runtimeMsTotal !== null && <span className="num">{fmtMs(result.metrics.runtimeMsTotal)} total</span>}
        {result.srTifUrl && (
          <a href={result.srTifUrl} className="font-medium text-accent hover:text-accent-strong">
            Download SR GeoTIFF
          </a>
        )}
      </div>
      {result.warnings.map((w) => (
        <p key={w} className="text-warn">
          {w}
        </p>
      ))}
    </div>
  );
}

function Dropzone({ onFile }: { onFile: (f: File) => void }) {
  const [over, setOver] = useState(false);
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  };
  return (
    <label
      htmlFor="upload-input"
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      data-testid="dropzone"
      className={`card card-static flex aspect-square w-full cursor-pointer flex-col items-center justify-center gap-5 border-dashed p-8 text-center ${
        over ? "border-accent bg-accent-dim" : "hover:border-line-strong"
      }`}
    >
      <span aria-hidden className="grid size-14 place-items-center rounded-full border border-accent/35 bg-accent-dim text-accent">
        <Icon name="upload" size={22} />
      </span>
      <span className="flex flex-col gap-1">
        <span className="font-display text-h4 text-fg">Drop a GeoTIFF here</span>
        <span className="text-caption text-fg-muted">or click to choose a file · .tif / .tiff, 4 bands</span>
      </span>
      <input
        id="upload-input"
        type="file"
        accept=".tif,.tiff,image/tiff"
        className="sr-only"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
    </label>
  );
}

function Samples({ onPick }: { onPick: (s: SampleInfo) => void }) {
  const [samples, setSamples] = useState<SampleInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    getSamples().then(
      (s) => live && setSamples(s),
      (exc: unknown) => {
        console.error("getSamples failed:", exc);
        if (live) setError(exc instanceof Error ? exc.message : String(exc));
      },
    );
    return () => {
      live = false;
    };
  }, []);

  if (error) return <p role="alert" className="text-caption text-bad">Samples failed to load: {error}</p>;
  if (!samples || samples.length === 0) return null;
  return (
    <div className="flex flex-col gap-3">
      <p className="eyebrow">Or try a sample</p>
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {samples.map((s) => (
          <li key={s.id}>
            <button type="button" onClick={() => onPick(s)} className="card flex w-full flex-col gap-2 overflow-hidden p-2 text-left">
              <img src={s.thumb_url} alt="" className="aspect-square w-full rounded-sm object-cover [image-rendering:pixelated]" />
              <span className="truncate text-caption text-fg">{s.label}</span>
              {s.has_gt && <Badge variant="good">Has 2.5 m truth</Badge>}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ----------------------------------------------------------- quick metrics ----

function pendingReason(result: ImageResult | null, needsReference: boolean): string {
  if (!result) return "Awaiting an image.";
  if (result.mode === "mock") return "Mock mode: no model ran.";
  if (needsReference) return `Needs a ${SR_GSD} reference; uploads have none. Try a sample.`;
  return "Not reported for this image.";
}

function QuickMetrics({ status, result }: { status: RunStatus; result: ImageResult | null }) {
  const grid = "grid grid-cols-2 gap-3 md:grid-cols-4 lg:grid-cols-2";
  if (status === "running") {
    return (
      <div className={grid} aria-busy>
        {[0, 1, 2, 3].map((i) => (
          <StatCardSkeleton key={i} />
        ))}
      </div>
    );
  }
  const m = result?.metrics ?? null;
  const b = m?.bicubic ?? null;
  const blurry = m?.hfRatioVsBicubic != null && m.sharpnessWarnBelow != null && m.hfRatioVsBicubic < m.sharpnessWarnBelow;

  // Order per docs/mvp/api_contract.md: LPIPS, spectral consistency, then SSIM, PSNR last.
  const cards: ReactNode[] = [
    m?.lpips != null ? (
      <StatCard size="sm" value={m.lpips} decimals={4} label="LPIPS ↓" delta={b ? { value: m.lpips - b.lpips, decimals: 4, better: "down", label: "vs bicubic" } : undefined} />
    ) : (
      <PendingStatCard size="sm" label="LPIPS ↓" reason={pendingReason(result, true)} />
    ),
    m?.specL1 != null ? (
      <StatCard
        size="sm"
        value={m.specL1}
        decimals={5}
        label="Spectral consistency ↓"
        hint="L1, reflectance"
        badge={blurry ? <Badge variant="warn">Blur?</Badge> : undefined}
        delta={m.specL1Bicubic != null ? { value: m.specL1 - m.specL1Bicubic, decimals: 5, better: "neutral", label: "vs bicubic" } : undefined}
      />
    ) : (
      <PendingStatCard size="sm" label="Spectral consistency ↓" reason={pendingReason(result, false)} />
    ),
    m?.ssim != null ? (
      <StatCard size="sm" value={m.ssim} decimals={4} label="SSIM ↑" delta={b ? { value: m.ssim - b.ssim, decimals: 4, better: "up", label: "vs bicubic" } : undefined} />
    ) : (
      <PendingStatCard size="sm" label="SSIM ↑" reason={pendingReason(result, true)} />
    ),
    m?.psnr != null ? (
      <StatCard size="sm" value={m.psnr} decimals={2} unit="dB" label="PSNR ↑" delta={b ? { value: m.psnr - b.psnr, decimals: 2, unit: "dB", better: "up", label: "vs bicubic" } : undefined} />
    ) : (
      <PendingStatCard size="sm" label="PSNR ↑" reason={pendingReason(result, true)} />
    ),
  ];
  return (
    <div className="flex flex-col gap-3" data-testid="quick-metrics">
      <p className="eyebrow">This image</p>
      <div className={grid}>
        {cards.map((c, i) => (
          <div key={i} className="contents">
            {c}
          </div>
        ))}
      </div>
      {blurry && <p className="text-caption text-warn">Sharpness is near bicubic: a low spectral error here may come from blur.</p>}
    </div>
  );
}

// ------------------------------------------------------------- novelties ----

function Novelties() {
  return (
    <section className="page-container flex flex-col gap-12 py-section" data-testid="novelties">
      <SectionHeader eyebrow="Three contributions" title="More than a sharper picture" description="Each claim is measured on the validation split and carried through to the deployed model." />
      <Reveal className="grid gap-6 md:grid-cols-3" itemClassName="flex">
        {NOVELTIES.map((n) => (
          <Card key={n.to} className="flex w-full flex-col gap-6 p-6">
            <div className="flex items-center justify-between">
              <IconTile name={n.icon} />
              <span className="num text-caption text-fg-subtle">{n.index}</span>
            </div>
            <div className="flex flex-col gap-2">
              <h3 className="text-h4">{n.title}</h3>
              <p className="text-caption text-fg-muted">{n.pitch}</p>
            </div>
            <div className="mt-auto flex flex-col gap-1.5 border-t border-line pt-5">
              <p className="num text-h2 font-medium text-fg">{n.metric}</p>
              <p className="text-caption text-fg-subtle">{n.metricLabel}</p>
            </div>
            <ButtonLink to={n.to} variant="ghost" className="-mx-2 self-start px-2">
              Learn more
              <Icon name="arrow-right" />
            </ButtonLink>
          </Card>
        ))}
      </Reveal>
    </section>
  );
}

// ---------------------------------------------------------------- impact ----

function ImpactStrip() {
  const budgetPct = (FACTS.params / APP_CONFIG.maxParameters) * 100;
  const fewerPct = (1 - FACTS.params / FACTS.runAParams) * 100;
  return (
    <section className="relative overflow-hidden border-t border-line" data-testid="impact">
      <div aria-hidden className="absolute inset-0 bg-gradient-to-b from-accent/[0.06] to-transparent" />
      <div className="page-container relative flex flex-col gap-12 py-section">
        <SectionHeader eyebrow="By the numbers" title="Small enough to ship. Fast enough to use." />
        <Reveal className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard accent value={FACTS.params} label="Parameters" hint={`${formatNumber(budgetPct, 1)}% of the ${formatNumber(APP_CONFIG.maxParameters)} budget`} />
          <StatCard
            value={Number(fmtMb(FACTS.onnxFp32Bytes))}
            decimals={2}
            unit="MB"
            label="ONNX FP32 model"
            badge={<Badge variant="pending">INT8 pending</Badge>}
            hint={`The ${fmtMb(FACTS.int8Bytes)} MB INT8 build failed its accuracy gate`}
          />
          <StatCard
            value={FACTS.onnxFp32MedianMs / 1000}
            decimals={2}
            unit="s"
            label={`CPU inference, ${FACTS.benchLrPx}² px tile`}
            hint={`median of ${FACTS.benchRuns} · ${FACTS.benchThreads} threads · no GPU`}
          />
          <StatCard
            value={fewerPct}
            decimals={0}
            unit="% fewer"
            label="Parameters vs EDSR 16×64"
            badge={<Badge variant="pending">Lit. pending</Badge>}
            hint={`EDSR 16×64 (our Run A) has ${formatNumber(FACTS.runAParams)}`}
          />
        </Reveal>
      </div>
    </section>
  );
}
