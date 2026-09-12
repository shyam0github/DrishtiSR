import type { ReactNode } from "react";
import { formatNumber, useCountUp, useInView } from "../../design/motion";

type Tone = "accent" | "muted" | "good" | "warn" | "bad";

const FILL: Record<Tone, string> = {
  accent: "bg-gradient-to-r from-accent/70 to-accent shadow-[0_0_12px_-2px_var(--color-accent)]",
  muted: "bg-fg-subtle",
  good: "bg-good",
  warn: "bg-warn",
  bad: "bg-bad",
};

const STROKE: Record<Tone, string> = {
  accent: "var(--color-accent)",
  muted: "var(--color-fg-subtle)",
  good: "var(--color-good)",
  warn: "var(--color-warn)",
  bad: "var(--color-bad)",
};

function fraction(value: number, max: number): number {
  if (!(max > 0)) throw new RangeError(`MetricBar/Gauge: max must be > 0, got ${max}`);
  return Math.min(Math.max(value / max, 0), 1);
}

/**
 * Horizontal bar: `value` of `max`, with an optional reference marker (e.g. a
 * budget or a baseline). The bar fills when it scrolls into view. Values above
 * max are drawn full, but the printed number is never clamped.
 */
export function MetricBar({
  label,
  value,
  max,
  decimals = 2,
  unit,
  tone = "accent",
  marker,
  badge,
}: {
  label: ReactNode;
  value: number;
  max: number;
  decimals?: number;
  unit?: string;
  tone?: Tone;
  marker?: { value: number; label: string };
  badge?: ReactNode;
}) {
  const [ref, inView] = useInView<HTMLDivElement>();
  const f = fraction(value, max);
  return (
    <div ref={ref} className="group flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-3 text-caption">
        <span className="flex items-center gap-2 text-fg-muted transition-colors group-hover:text-fg">
          {label}
          {badge}
        </span>
        <span className="num text-fg">
          {formatNumber(value, decimals)}
          {unit && <span className="text-fg-subtle"> {unit}</span>}
        </span>
      </div>
      <div className="relative h-2 rounded-full bg-ink-800">
        <div
          className={`h-full rounded-full transition-[width] duration-[1200ms] ease-out-expo ${FILL[tone]}`}
          style={{ width: inView ? `${f * 100}%` : "0%" }}
        />
        {marker && (
          <div className="absolute -top-1 -bottom-1 w-px bg-fg/70" style={{ left: `${fraction(marker.value, max) * 100}%` }} title={marker.label}>
            <span className="absolute top-full left-1/2 mt-1 -translate-x-1/2 whitespace-nowrap font-mono text-[10px] text-fg-subtle">{marker.label}</span>
          </div>
        )}
      </div>
      {marker && <div className="h-3" aria-hidden />}
    </div>
  );
}

/** Semicircular gauge for a single headline ratio (e.g. parameter budget used). */
export function Gauge({
  value,
  max,
  label,
  decimals = 0,
  unit,
  tone = "accent",
  size = 180,
}: {
  value: number;
  max: number;
  label: ReactNode;
  decimals?: number;
  unit?: string;
  tone?: Tone;
  size?: number;
}) {
  const [ref, inView] = useInView<HTMLDivElement>();
  const shown = useCountUp(value, inView);
  const f = fraction(value, max);
  const stroke = 12;
  const r = (size - stroke) / 2;
  const arc = Math.PI * r;
  const cx = size / 2;
  const cy = size / 2;
  const d = `M ${stroke / 2} ${cy} A ${r} ${r} 0 0 1 ${size - stroke / 2} ${cy}`;
  return (
    <div ref={ref} className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size / 2 + stroke }}>
        <svg width={size} height={size / 2 + stroke} viewBox={`0 0 ${size} ${size / 2 + stroke}`} aria-hidden>
          <path d={d} fill="none" stroke="var(--color-ink-800)" strokeWidth={stroke} strokeLinecap="round" />
          <path
            d={d}
            fill="none"
            stroke={STROKE[tone]}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={arc}
            strokeDashoffset={inView ? arc * (1 - f) : arc}
            style={{ transition: "stroke-dashoffset 1400ms var(--ease-out-expo)", filter: tone === "accent" ? "drop-shadow(0 0 6px rgb(34 211 238 / 0.5))" : undefined }}
          />
          <circle cx={cx} cy={cy} r={2} fill="var(--color-fg-subtle)" />
        </svg>
        <div className="absolute inset-x-0 bottom-0 flex items-baseline justify-center gap-1">
          <span className="num text-h3 font-medium text-fg">{formatNumber(shown, decimals)}</span>
          {unit && <span className="num text-caption text-fg-muted">{unit}</span>}
        </div>
      </div>
      <p className="text-caption text-fg-muted">{label}</p>
    </div>
  );
}
