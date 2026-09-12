import type { ReactNode } from "react";
import { formatNumber, useCountUp, useInView } from "../../design/motion";
import { Card } from "./Card";

export interface StatDelta {
  /** Signed change, in the same unit as the stat. */
  value: number;
  decimals?: number;
  unit?: string;
  /** Which direction is an improvement. Sets green/red; "neutral" stays grey. */
  better: "up" | "down" | "neutral";
  /** Context, e.g. "vs bicubic". */
  label?: string;
}

/**
 * Big number + label, counting up from 0 the first time it scrolls into view.
 * The final rendered value is always exactly `value` at `decimals`.
 */
export function StatCard({
  value,
  decimals = 0,
  prefix,
  unit,
  label,
  hint,
  delta,
  badge,
  accent = false,
}: {
  value: number;
  decimals?: number;
  prefix?: string;
  unit?: string;
  label: ReactNode;
  hint?: ReactNode;
  delta?: StatDelta;
  badge?: ReactNode;
  accent?: boolean;
}) {
  const [ref, inView] = useInView<HTMLDivElement>();
  const shown = useCountUp(value, inView);
  return (
    <Card accent={accent} className="flex flex-col gap-5 p-6">
      <div ref={ref} className="flex items-start justify-between gap-3">
        <p className="text-caption font-medium text-fg-muted">{label}</p>
        {badge}
      </div>
      <p className="flex items-baseline gap-1.5 text-fg" aria-label={`${prefix ?? ""}${formatNumber(value, decimals)}${unit ? ` ${unit}` : ""}`}>
        {prefix && <span className="num text-h4 text-fg-muted">{prefix}</span>}
        <span className="num text-stat font-medium">{formatNumber(shown, decimals)}</span>
        {unit && <span className="num text-caption text-fg-muted">{unit}</span>}
      </p>
      {(delta || hint) && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-caption">
          {delta && <DeltaChip delta={delta} />}
          {hint && <span className="text-fg-subtle">{hint}</span>}
        </div>
      )}
    </Card>
  );
}

function DeltaChip({ delta }: { delta: StatDelta }) {
  const up = delta.value > 0;
  const flat = delta.value === 0;
  const good = delta.better === "neutral" || flat ? null : (delta.better === "up") === up;
  const tone = good === null ? "text-fg-muted bg-ink-800" : good ? "text-good bg-good/10" : "text-bad bg-bad/10";
  const arrow = flat ? "→" : up ? "↑" : "↓";
  return (
    <span className="inline-flex items-center gap-2">
      <span className={`num inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 ${tone}`}>
        <span aria-hidden>{arrow}</span>
        {up ? "+" : ""}
        {formatNumber(delta.value, delta.decimals ?? 0)}
        {delta.unit && ` ${delta.unit}`}
      </span>
      {delta.label && <span className="text-fg-subtle">{delta.label}</span>}
    </span>
  );
}
