import type { ReactNode } from "react";
import type { ImageResult } from "../api/client";
import { Badge, ButtonLink, Icon, ImageSkeleton, SectionHeader } from "../components/ui";
import { useExampleResult } from "../state/exampleResult";
import { useResult } from "../state/ResultContext";

/** Home's upload section; HomePage scrolls to it when the URL carries this hash. */
export const HOME_UPLOAD_HREF = "/#try-it";

export type ImageSource = "yours" | "example";

/**
 * Shared frame for the three novelty pages:
 *   A. header: name + one plain-English paragraph
 *   B. "This image": the upload from ResultContext, else a server example (never "yours")
 *   C. "Model benchmark": aggregate eval-harness numbers, labelled with their n
 *   D. "How it works": collapsed by default
 * B and C carry distinct eyebrows and borders so a per-image number is never read as a benchmark.
 */
export function NoveltyPageLayout({
  index,
  title,
  summary,
  renderImage,
  benchmarkLabel,
  benchmarkSource,
  benchmark,
  howItWorks,
  testId,
}: {
  index: string;
  title: string;
  summary: ReactNode;
  /** Novelty-specific view of one result. `source` says whether it is the user's upload or the example. */
  renderImage: (result: ImageResult, source: ImageSource) => ReactNode;
  /** e.g. "n = 1,199 val patches · 300 tiles". */
  benchmarkLabel: string;
  /** Repo-relative report the numbers come from. */
  benchmarkSource: string;
  benchmark: ReactNode;
  howItWorks: ReactNode;
  testId: string;
}) {
  return (
    <div data-testid={testId}>
      <section className="relative overflow-hidden">
        <div aria-hidden className="bg-graticule absolute inset-0" />
        <div className="page-container relative flex flex-col gap-6 pt-[calc(var(--spacing-section)*0.6)] pb-[calc(var(--spacing-section)*0.5)]">
          <SectionHeader level="h1" eyebrow={`Novelty ${index}`} title={title} description={summary} />
        </div>
      </section>

      <section className="border-y border-line bg-ink-900/50" data-testid="section-this-image">
        <div className="page-container flex flex-col gap-8 py-section">
          <SectionHeader eyebrow="This image" title="On your image" description="One image, measured as it runs. These numbers describe this image only." />
          <ThisImage renderImage={renderImage} />
        </div>
      </section>

      <section data-testid="section-benchmark">
        <div className="page-container flex flex-col gap-8 py-section">
          <SectionHeader
            eyebrow={`Model benchmark · ${benchmarkLabel}`}
            title="Across the validation set"
            description={
              <>
                Aggregates from the evaluation harness, not from your image. Source: <span className="num text-caption">{benchmarkSource}</span>
              </>
            }
          />
          {benchmark}
        </div>
      </section>

      <section className="border-t border-line" data-testid="section-how">
        <div className="page-container py-section">
          <details className="group card card-static p-0">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-4 p-6 [&::-webkit-details-marker]:hidden">
              <span className="flex flex-col gap-1">
                <span className="eyebrow">How it works</span>
                <span className="text-h4">The method, and its limits</span>
              </span>
              <span aria-hidden className="text-fg-muted transition-transform group-open:rotate-90">
                <Icon name="arrow-right" />
              </span>
            </summary>
            <div className="flex max-w-prose flex-col gap-4 border-t border-line p-6 text-body text-fg-muted">{howItWorks}</div>
          </details>
        </div>
      </section>
    </div>
  );
}

function ThisImage({ renderImage }: { renderImage: (r: ImageResult, s: ImageSource) => ReactNode }) {
  const { result, status, input } = useResult();
  const hasUpload = status === "done" && result !== null;
  const example = useExampleResult(!hasUpload && status !== "running");

  if (status === "running") return <ImageSkeleton aspect="2 / 1" label={`Super-resolving ${input?.label ?? "your image"}…`} />;
  if (hasUpload) return <>{renderImage(result, "yours")}</>;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-4 rounded-lg border border-dashed border-line-strong p-5 sm:flex-row sm:items-center sm:justify-between" data-testid="no-upload">
        <p className="text-caption text-fg-muted">
          No image uploaded yet. Upload a 4-band GeoTIFF on Home and this section shows it. Until then, a validation sample stands in, labelled as an example.
        </p>
        <ButtonLink to={HOME_UPLOAD_HREF} data-testid="cta-upload">
          <Icon name="upload" />
          Upload on Home
        </ButtonLink>
      </div>
      {example.status === "loading" && <ImageSkeleton aspect="2 / 1" label="Loading the example…" />}
      {example.status === "done" && renderImage(example.result, "example")}
      {example.status === "unavailable" && (
        <p className="flex flex-wrap items-center gap-3 text-caption text-fg-muted" data-testid="example-unavailable">
          <Badge variant="pending">No example</Badge>
          {example.reason}
        </p>
      )}
      {example.status === "error" && (
        <p role="alert" className="rounded-md border border-bad/40 bg-bad/10 p-4 text-caption text-bad">
          The example failed to load: {example.error}
        </p>
      )}
    </div>
  );
}
