import { useState, type ReactNode } from "react";
import {
  Badge,
  BeforeAfterSlider,
  Card,
  Gauge,
  ImageSkeleton,
  MetricBar,
  Reveal,
  SectionHeader,
  Skeleton,
  StatCard,
  StatCardSkeleton,
  TabbedPanel,
} from "../components/ui";

/**
 * Component reference for the design system (route /design, not in the nav).
 * Every value below is ILLUSTRATIVE sample data, labelled as such, not a result.
 */
export default function DesignSystemPage() {
  const [loading, setLoading] = useState(false);
  return (
    <div className="page-container flex flex-col gap-section py-section">
      <SectionHeader
        level="h1"
        eyebrow="Design system"
        title="Components & tokens"
        description="Reference for every page. Numbers on this page are sample values for layout only — not results."
      />

      <Block title="Type scale">
        <Card interactive={false} className="flex flex-col gap-5 p-8">
          <p className="font-display text-display">Display</p>
          <h1>Heading 1</h1>
          <h2>Heading 2</h2>
          <h3>Heading 3</h3>
          <h4>Heading 4</h4>
          <h5>Heading 5</h5>
          <h6>Heading 6</h6>
          <p className="text-lead text-fg-muted">Lead — reflectance stays physical end to end.</p>
          <p>Body — Inter at 15px, tuned for dense metric tables and captions.</p>
          <p className="text-caption text-fg-muted">Caption — secondary context.</p>
          <p className="eyebrow">Eyebrow / micro</p>
          <p className="num text-h4">0123456789 · 33.91 dB · 0.0412</p>
        </Card>
      </Block>

      <Block title="Colour">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
          {["ink-950", "ink-850", "ink-800", "ink-700", "accent", "spectral", "lit", "good", "warn", "bad", "fg", "fg-muted", "fg-subtle"].map((c) => (
            <div key={c} className="flex flex-col gap-2">
              <div className="h-14 rounded-md border border-line" style={{ background: `var(--color-${c})` }} />
              <span className="num text-caption text-fg-muted">{c}</span>
            </div>
          ))}
        </div>
      </Block>

      <Block title="Badges">
        <div className="flex flex-wrap gap-3">
          <Badge variant="ours">Ours</Badge>
          <Badge variant="baseline">Baseline</Badge>
          <Badge variant="literature">Literature</Badge>
          <Badge variant="pending">Pending</Badge>
          <Badge variant="good">Pass</Badge>
          <Badge variant="warn">Gate</Badge>
          <Badge variant="bad">Fail</Badge>
        </div>
      </Block>

      <Block title="StatCard (count-up on scroll)">
        <Reveal className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard accent value={28.47} decimals={2} unit="dB" label="PSNR (sample)" badge={<Badge variant="ours">Ours</Badge>} delta={{ value: 1.32, decimals: 2, unit: "dB", better: "up", label: "vs bicubic" }} />
          <StatCard value={0.187} decimals={3} label="LPIPS (sample)" delta={{ value: -0.041, decimals: 3, better: "down", label: "vs bicubic" }} />
          <StatCard value={987654} label="Parameters (sample)" hint="budget 1,000,000" />
          <StatCardSkeleton />
        </Reveal>
      </Block>

      <Block title="MetricBar & Gauge">
        <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
          <Card className="flex flex-col gap-6 p-6">
            <MetricBar label="Ours" badge={<Badge variant="ours">Ours</Badge>} value={0.92} max={1} marker={{ value: 0.8, label: "target" }} />
            <MetricBar label="Bicubic" tone="muted" value={0.61} max={1} />
            <MetricBar label="Literature" tone="warn" value={0.74} max={1} />
          </Card>
          <Card className="grid place-items-center p-6">
            <Gauge value={98.8} max={100} decimals={1} unit="%" label="Parameter budget used (sample)" />
          </Card>
        </div>
      </Block>

      <Block title="BeforeAfterSlider + lightbox">
        <div className="flex flex-col gap-4">
          <div>
            <button type="button" onClick={() => setLoading((l) => !l)} className="rounded-md border border-line px-3 py-1.5 text-caption text-fg-muted transition hover:border-line-strong hover:text-fg">
              Toggle loading state
            </button>
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            <BeforeAfterSlider
              loading={loading}
              beforeSrc="/placeholders/lr.png"
              afterSrc="/placeholders/hr.png"
              beforeLabel="10 m input"
              afterLabel="2.5 m"
              alt="Placeholder scene — procedural, not model output"
              aspect="1 / 1"
            />
            <ImageSkeleton aspect="1 / 1" />
          </div>
        </div>
      </Block>

      <Block title="TabbedPanel">
        <Card interactive={false} className="p-6">
          <TabbedPanel
            tabs={[
              { id: "rgb", label: "RGB", content: <p className="text-fg-muted">True-colour composite (B04, B03, B02).</p> },
              { id: "nir", label: "NIR", content: <p className="text-fg-muted">Near-infrared band B08.</p> },
              { id: "sigma", label: "Uncertainty σ", content: <Skeleton className="h-24 w-full" /> },
            ]}
          />
        </Card>
      </Block>
    </div>
  );
}

function Block({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-6">
      <h2 className="eyebrow">{title}</h2>
      {children}
    </section>
  );
}
