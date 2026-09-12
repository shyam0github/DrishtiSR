import type { ReactNode } from "react";
import { Badge, Card, Icon, Reveal, SectionHeader, type IconName } from "../components/ui";
import { APP_CONFIG } from "../config";
import { ABOUT_FACTS as A, FACTS } from "../data/facts";
import { formatNumber } from "../design/motion";

const LR_GSD = `${APP_CONFIG.lrGsdM} m`;
const SR_GSD = `${APP_CONFIG.lrGsdM / APP_CONFIG.scale} m`;
const fmtMb = (bytes: number) => `${formatNumber(bytes / 1e6, 2)} MB`;
const fmtS = (ms: number) => `${formatNumber(ms / 1000, 2)} s`;
const n = (v: number, d?: number) => formatNumber(v, d);

export function AboutPage() {
  return (
    <>
      <Overview />
      <SpecSheet />
      <Pipeline />
      <Limitations />
      <Links />
    </>
  );
}

// -------------------------------------------------------------- overview ----

function Overview() {
  return (
    <section className="relative overflow-hidden" data-testid="about-overview">
      <div aria-hidden className="bg-graticule absolute inset-0" />
      <div className="page-container relative flex flex-col gap-10 pt-[calc(var(--spacing-section)*0.75)] pb-section">
        <SectionHeader
          level="h1"
          eyebrow={`${APP_CONFIG.teamName} · Smart India Hackathon 2026 · SIH26142`}
          title="About DrishtiSR"
          description={`Deep-learning ×${APP_CONFIG.scale} super-resolution of Sentinel-2 imagery, ${LR_GSD} to ${SR_GSD}, built to run on a CPU and to say where it is guessing.`}
        />
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <Card interactive={false} className="flex flex-col gap-3 p-6">
            <p className="eyebrow">Problem statement SIH26142</p>
            <p className="text-fg-muted">
              Recover {SR_GSD} detail from {LR_GSD} Sentinel-2 L2A surface reflectance ({APP_CONFIG.bands.join(", ")}), trained on SEN2NAIPv2 and
              demonstrated on Delhi. The model must fit under {n(APP_CONFIG.maxParameters)} parameters and run on a {FACTS.benchThreads}-thread CPU.
            </p>
          </Card>
          <Card interactive={false} className="flex flex-col gap-3 p-6">
            <p className="eyebrow">Technical summary</p>
            <p className="text-fg-muted">
              A narrowed EDSR backbone ({A.nResblocks} residual blocks × {A.nFeats} features, {n(FACTS.params)} parameters) is trained with L1 on
              unnormalised, unclipped reflectance. A spectral-consistency loss (area ×4 degradation, L1 + SAM) was built and ablated. Consistency
              is measured on every output and shipped as a map. Per-pixel uncertainty comes from a heteroscedastic NLL head, with test-time-augmentation
              disagreement (TTA-4) as the served estimate. Inference is tiled with raised-cosine blending and exported to ONNX for CPU.
            </p>
          </Card>
        </div>
      </div>
    </section>
  );
}

// ------------------------------------------------------------ spec sheet ----

type Row = { k: string; v: ReactNode; note?: ReactNode };

function SpecTable({ rows }: { rows: Row[] }) {
  return (
    <dl className="divide-y divide-line">
      {rows.map((r) => (
        <div key={r.k} className="grid gap-1 py-3 sm:grid-cols-[11rem_minmax(0,1fr)] sm:gap-4">
          <dt className="text-caption text-fg-subtle">{r.k}</dt>
          <dd className="flex flex-col gap-1">
            <span className="text-fg">{r.v}</span>
            {r.note && <span className="text-caption text-fg-subtle">{r.note}</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function SpecCard({ icon, title, badge, accent, rows, testId }: { icon: IconName; title: string; badge?: ReactNode; accent?: boolean; rows: Row[]; testId: string }) {
  return (
    <Card accent={accent} interactive={false} className="flex flex-col gap-4 p-6" data-testid={testId}>
      <header className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="flex items-center gap-3 text-h4">
          <span aria-hidden className="grid size-9 place-items-center rounded-md border border-accent/35 bg-accent-dim text-accent">
            <Icon name={icon} size={16} />
          </span>
          {title}
        </h3>
        {badge}
      </header>
      <SpecTable rows={rows} />
    </Card>
  );
}

const Num = ({ children }: { children: ReactNode }) => <span className="num">{children}</span>;

function SpecSheet() {
  const training: Row[] = [
    {
      k: "GPU",
      v: <>Kaggle {A.trainGpu}, {A.gpuHoursPerWeek} GPU-hours / week budget</>,
      note: "P100 is refused: the Kaggle CUDA 12.8 build dropped Pascal.",
    },
    { k: "Software", v: <>Python <Num>{A.trainPython}</Num>, PyTorch <Num>{A.trainTorch}</Num>, OmegaConf <Num>{A.omegaconf}</Num></>, note: "Frozen, hash-verified training config." },
    {
      k: "Dataset",
      v: <>SEN2NAIPv2 <Num>{A.subset.replace("sen2naipv2-", "")}</Num> via tacoreader <Num>{A.tacoreader}</Num> (TACO)</>,
      note: <>{n(A.cachedPairs)} pairs cached, {n(A.usableTrainPairs)} usable for training</>,
    },
    { k: "Architecture", v: <>EDSR, {A.nResblocks} blocks × {A.nFeats} features, <Num>{n(FACTS.params)}</Num> params</>, note: `${n((FACTS.params / APP_CONFIG.maxParameters) * 100, 1)}% of the ${n(APP_CONFIG.maxParameters)} budget` },
    { k: "Iterations", v: <><Num>{n(A.a2Iters)}</Num> (served run)</>, note: <>Frozen schedule allows {n(A.scheduleIters)}. Capped after Run A overfit (see Limitations). Batch {A.batch}, {A.patchLr} px LR patches.</> },
    { k: "Augmentation", v: "8 square symmetries (flips + 90° rotations), train only", note: "No photometric augmentation: it would alter reflectance." },
  ];
  const deploy: Row[] = [
    { k: "Hardware", v: "CPU only. No GPU, discrete or integrated.", note: `Measured on ${FACTS.benchCpu}, ${FACTS.benchThreads} of ${A.logicalCpus} logical CPUs.` },
    {
      k: "Latency",
      v: <><Num>{fmtS(FACTS.onnxFp32MedianMs)}</Num> median per {FACTS.benchLrPx}² px tile</>,
      note: `${FACTS.benchLrPx}² → ${FACTS.benchLrPx * APP_CONFIG.scale}² px, ONNX FP32, ${FACTS.benchThreads} threads, ${FACTS.benchRuns} runs`,
    },
    {
      k: "Served model",
      v: <>ONNX FP32, <Num>{fmtMb(FACTS.onnxFp32Bytes)}</Num></>,
      note: <>INT8 build: {fmtMb(FACTS.int8Bytes)}, {fmtS(FACTS.int8MedianMs)}, failed its accuracy gate, not served.</>,
    },
    { k: "Runtime", v: <>ONNX Runtime <Num>{A.onnxruntime}</Num>, Python <Num>{A.inferPython}</Num></>, note: <>PyTorch <Num>{A.inferTorch}</Num> path kept for parity.</> },
    {
      k: "Minimum system",
      v: <>Any x86-64 CPU with {FACTS.benchThreads} threads, no GPU</>,
      note: (
        <span className="inline-flex flex-wrap items-center gap-2">
          RAM: <Badge variant="pending">Not measured</Badge> peak memory was not recorded in the benchmark.
        </span>
      ),
    },
    { k: "Tiling", v: <>{APP_CONFIG.tile.lrPx} px LR tiles, {APP_CONFIG.tile.overlapLrPx} px overlap</>, note: "Raster size is bounded by disk, not by memory." },
  ];
  const evaluation: Row[] = [
    { k: "External benchmark", v: <>opensr-test <Num>{A.opensrTest}</Num> (ESA OpenSR)</> },
    { k: "Own metrics", v: "PSNR, SSIM, LPIPS, SAM, ERGAS", note: "Reflectance domain. LPIPS on RGB only." },
    {
      k: "Splits",
      v: <>{n(A.splitTiles.train)} / {A.splitTiles.val} / {A.splitTiles.test} tiles (train / val / test)</>,
      note: `Scene-disjoint. Val = ${n(FACTS.valPairs)} patches. Test held out.`,
    },
    {
      k: "Statistics",
      v: "Wilcoxon signed-rank + bootstrap CIs",
      note: `${n(A.nBoot)} resamples, ${A.ci * 100}% CI, pair-level and tile-clustered. Verdicts use the tile CI.`,
    },
    { k: "Checkpoint choice", v: "Pre-registered post-hoc rule", note: "Lowest consistency error, CI-tied checkpoints broken by LPIPS. Not best.pt." },
  ];

  return (
    <section className="border-y border-line bg-ink-900/50" data-testid="about-specs">
      <div className="page-container flex flex-col gap-10 py-section">
        <SectionHeader eyebrow="Spec sheet" title="Technical specifications" description="Every value comes from a run log, report or pinned requirement in the repository, and a test fails if one drifts." />
        <div className="grid gap-6 lg:grid-cols-2">
          <SpecCard testId="spec-training" icon="spectral" title="Training environment" rows={training} />
          <SpecCard testId="spec-deploy" icon="cpu" title="Deployment & inference" accent badge={<Badge variant="ours">No GPU required</Badge>} rows={deploy} />
        </div>
        <SpecCard testId="spec-eval" icon="uncertainty" title="Evaluation" rows={evaluation} />
      </div>
    </section>
  );
}

// -------------------------------------------------------------- pipeline ----

type Node = { title: string; body: string; tag?: ReactNode };

function PipeNode({ node, accent }: { node: Node; accent?: boolean }) {
  return (
    <div className={`card card-static flex flex-1 flex-col gap-1.5 p-4 ${accent ? "card-accent" : ""}`}>
      <p className="text-h6">{node.title}</p>
      <p className="text-caption text-fg-muted">{node.body}</p>
      {node.tag && <div className="pt-1">{node.tag}</div>}
    </div>
  );
}

function Arrow() {
  return (
    <span aria-hidden className="grid shrink-0 place-items-center text-accent/70 lg:w-6">
      <span className="lg:hidden">
        <Icon name="arrow-down" size={16} />
      </span>
      <span className="hidden lg:inline">
        <Icon name="arrow-right" size={16} />
      </span>
    </span>
  );
}

function Pipeline() {
  const main: Node[] = [
    { title: "Sentinel-2 L2A tile", body: `${APP_CONFIG.bands.join(" ")} at ${LR_GSD}, DN → reflectance, unclipped` },
    { title: "Tiled inference", body: `${APP_CONFIG.tile.lrPx} px tiles, ${APP_CONFIG.tile.overlapLrPx} px overlap, raised-cosine (Hann) blend` },
    { title: "EDSR backbone", body: `${A.nResblocks} × ${A.nFeats}, ${n(FACTS.params)} params, ×${APP_CONFIG.scale}` },
  ];
  const heads: Node[] = [
    { title: "Spectral consistency", body: `SR degraded to ${LR_GSD}, compared with the input`, tag: <Badge variant="neutral">Measured + map</Badge> },
    { title: "Uncertainty", body: "Heteroscedastic NLL head, or TTA disagreement", tag: <Badge variant="ours">TTA-4 served</Badge> },
  ];
  const outputs: Node[] = [
    { title: `SR GeoTIFF, ${SR_GSD}`, body: "Georeferenced, reflectance" },
    { title: "Uncertainty raster", body: "Per pixel, alongside every result" },
  ];
  return (
    <section className="page-container flex flex-col gap-10 py-section" data-testid="about-pipeline">
      <SectionHeader eyebrow="Architecture" title="From 10 m tile to trusted 2.5 m output" />
      <figure className="flex flex-col gap-4">
        <div className="flex flex-col items-stretch gap-2 lg:flex-row lg:items-center">
          {main.map((nd) => (
            <div key={nd.title} className="contents">
              <PipeNode node={nd} accent={nd.title === "EDSR backbone"} />
              <Arrow />
            </div>
          ))}
          <div className="flex flex-1 flex-col gap-2">
            {heads.map((h) => (
              <PipeNode key={h.title} node={h} />
            ))}
          </div>
          <Arrow />
          <div className="flex flex-1 flex-col gap-2">
            {outputs.map((o) => (
              <PipeNode key={o.title} node={o} />
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-3 rounded-lg border border-dashed border-line-strong p-4 sm:flex-row sm:items-center">
          <p className="eyebrow shrink-0">CPU runtime</p>
          <div className="flex flex-wrap items-center gap-2 text-caption text-fg-muted">
            <Badge variant="baseline">PyTorch FP32</Badge>
            <Icon name="arrow-right" size={14} />
            <Badge variant="ours">ONNX FP32 · served</Badge>
            <Icon name="arrow-right" size={14} />
            <Badge variant="pending">ONNX INT8 · gate failed</Badge>
          </div>
        </div>
        <figcaption className="text-caption text-fg-subtle">
          The served uncertainty is TTA-4. The learned head is opt-in: on validation it lost to TTA-8 on AUSE ({n(FACTS.uncLearnedAuse, 5)} vs{" "}
          {n(FACTS.uncTta8Ause, 5)}, n = {FACTS.uncN}).
        </figcaption>
      </figure>
    </section>
  );
}

// ----------------------------------------------------------- limitations ----

function Limitations() {
  const items: { title: string; status: ReactNode; body: ReactNode }[] = [
    {
      title: "Longer training made the model worse",
      status: <Badge variant="warn">Mitigated</Badge>,
      body: (
        <>
          Run A's training-loop validation PSNR peaked at <Num>{n(A.runABestPsnr, 2)} dB</Num> at iteration {n(A.runABestIter)}, then declined to{" "}
          <Num>{n(A.runAFinalPsnr, 2)} dB</Num> by {n(A.runAFinalIter)}. The served run adds augmentation and stops at {n(A.a2Iters)}. A{" "}
          {n(A.a2Iters)}-iteration run cannot rule out overfitting that only appears later.
        </>
      ),
    },
    {
      title: "The spectral loss has a degenerate minimum",
      status: <Badge variant="warn">Known</Badge>,
      body: (
        <>
          On its own, the term is minimised by any output that averages to the input, including a blurry one. Bicubic scores {A.bicubicVsFloor}× the
          ground truth's own consistency error ({A.spectralFloorL1}). With the loss on, consistency error fell from {A.a2ConsistencyL1.toFixed(5)} to{" "}
          {A.b1ConsistencyL1}. It cost {A.b1PsnrCostDb} dB PSNR and +{A.b1LpipsCost.toFixed(3)} LPIPS, so the served model trains without it and reports consistency as a metric.
        </>
      ),
    },
    {
      title: "Domain shift: trained in the US, shown on Delhi",
      status: <Badge variant="pending">Open</Badge>,
      body: (
        <>
          Every training target is NAIP aerial imagery over the United States. The Delhi demo is qualitative only, because no {SR_GSD} ground truth exists
          for it. The US → Delhi domain-shift measurement has not been run.
        </>
      ),
    },
    {
      title: "Input and target statistics match too closely",
      status: <Badge variant="pending">Explained, not closed</Badge>,
      body: (
        <>
          In {A.dvPairs} sampled pairs, per-band mean reflectance of the 10 m input and the 2.5 m target agree to four decimal places in{" "}
          {A.dvMeanEqual4dp.B03}–{A.dvMeanEqual4dp.B08} pairs per band. In {A.dvMinMaxEqualAllBands} of {A.dvPairs}, min and max are identical in all four bands. Our reading: the
          dataset authors radiometrically harmonised NAIP to Sentinel-2, so the targets inherit the input's statistics. They are not synthetically degraded:
          correlation is {n(A.dvRAllBandsMin, 3)}–{n(A.dvRAllBandsMax, 3)}, not 1, and the pairs are spatially offset. The reading rests on a {A.dvPairs}-pair sample and on
          the dataset's own description. Neither the operator the authors used nor the definition of their per-pair correlation field is recoverable
          from the data.
        </>
      ),
    },
  ];
  return (
    <section className="border-t border-line bg-ink-900/50" data-testid="about-limitations">
      <div className="page-container flex flex-col gap-10 py-section">
        <SectionHeader eyebrow="Known limitations" title="What we have not solved" description="Stated here so they do not have to be found." />
        <Reveal className="grid gap-6 md:grid-cols-2" itemClassName="flex">
          {items.map((it) => (
            <Card key={it.title} interactive={false} className="flex w-full flex-col gap-3 p-6">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <h3 className="text-h5">{it.title}</h3>
                {it.status}
              </div>
              <p className="text-caption text-fg-muted">{it.body}</p>
            </Card>
          ))}
        </Reveal>
      </div>
    </section>
  );
}

// ----------------------------------------------------------------- links ----

function Links() {
  const rows: Row[] = [
    {
      k: "Repository",
      v: (
        <a href={APP_CONFIG.repoUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 font-medium text-accent hover:text-accent-strong" data-testid="about-repo-link">
          <Icon name="github" />
          {APP_CONFIG.repoUrl.replace(/^https:\/\//, "")}
        </a>
      ),
    },
    { k: "Team", v: APP_CONFIG.teamName },
    { k: "Hackathon", v: "Smart India Hackathon 2026, problem statement SIH26142" },
    { k: "Track", v: <Badge variant="pending">Not recorded in the repo</Badge> },
  ];
  return (
    <section className="page-container flex flex-col gap-8 py-section" data-testid="about-links">
      <SectionHeader eyebrow="Links" title="Repository and team" />
      <Card interactive={false} className="max-w-3xl p-6">
        <SpecTable rows={rows} />
      </Card>
    </section>
  );
}
