import { useState } from "react";
import { MIN_RADIUS, polar, radarRadius } from "../design/radar";

export interface RadarAxis {
  key: string;
  label: string;
  better: "higher" | "lower";
  /** Log domain shared by every series, so toggling a series never rescales the others. */
  domain: [number, number];
  /** `reported`: quote at the source's precision rather than the axis's. */
  format: (v: number, reported: boolean) => string;
}

export interface RadarSeries {
  id: string;
  label: string;
  color: string;
  /** Dashed outline: secondary encoding for reported (literature) rows. */
  dashed: boolean;
  note: string;
  values: Record<string, number>;
}

const SIZE = 560;
const C = SIZE / 2;
/** Leaves room for the longest side labels ("Improvement ↑") inside the viewBox. */
const R = 158;
const LABEL_GAP = 26;
const RINGS = [0.25, 0.5, 0.75, 1];

/**
 * Multi-series radar. Every axis runs outward towards "better". Hovering a
 * vertex shows the exact value; the table next to the chart is the accessible
 * view, so the SVG is one labelled image.
 */
export function RadarChart({ axes, series, ariaLabel }: { axes: RadarAxis[]; series: RadarSeries[]; ariaLabel: string }) {
  const [hover, setHover] = useState<{ s: string; k: string; x: number; y: number } | null>(null);
  const n = axes.length;
  const ring = (f: number) => axes.map((_, i) => polar(i, n, f, C, C, R).join(",")).join(" ");
  const hs = hover ? series.find((s) => s.id === hover.s) : undefined;
  const ha = hover ? axes.find((a) => a.key === hover.k) : undefined;

  return (
    <div className="relative" data-testid="radar">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label={ariaLabel} className="block h-auto w-full overflow-visible">
        {RINGS.map((f) => (
          <polygon key={f} points={ring(f)} fill="none" stroke="var(--color-line)" strokeWidth={1} />
        ))}
        {axes.map((a, i) => {
          const [x, y] = polar(i, n, 1, C, C, R);
          const [lx, ly] = polar(i, n, 1, C, C, R + LABEL_GAP);
          const cos = Math.cos(-Math.PI / 2 + (i * 2 * Math.PI) / n);
          const anchor = cos > 0.3 ? "start" : cos < -0.3 ? "end" : "middle";
          return (
            <g key={a.key}>
              <line x1={C} y1={C} x2={x} y2={y} stroke="var(--color-line)" strokeWidth={1} />
              <text x={lx} y={ly} textAnchor={anchor} dominantBaseline="middle" fill="var(--color-fg-muted)" fontSize={14} fontFamily="var(--font-sans)">
                {a.label}
                <tspan fill="var(--color-fg-subtle)" fontSize={12}>
                  {a.better === "lower" ? " ↓" : " ↑"}
                </tspan>
              </text>
            </g>
          );
        })}
        <circle cx={C} cy={C} r={R * MIN_RADIUS} fill="none" stroke="var(--color-line)" strokeDasharray="2 3" />

        {series.map((s) => {
          const dim = hover !== null && hover.s !== s.id;
          const pts = axes.map((a, i) => polar(i, n, radarRadius(s.values[a.key]!, a.domain, a.better), C, C, R));
          return (
            <g key={s.id} data-series={s.id} opacity={dim ? 0.25 : 1} style={{ transition: "opacity 200ms ease" }}>
              <polygon
                points={pts.map((p) => p.join(",")).join(" ")}
                fill={s.color}
                fillOpacity={0.08}
                stroke={s.color}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeDasharray={s.dashed ? "7 5" : undefined}
              />
              {pts.map(([x, y], i) => (
                <circle key={axes[i]!.key} cx={x} cy={y} r={4.5} fill={s.color} stroke="var(--color-ink-850)" strokeWidth={2} />
              ))}
            </g>
          );
        })}

        {/* Hit targets above every mark, larger than the dots. */}
        {series.map((s) =>
          axes.map((a, i) => {
            const [x, y] = polar(i, n, radarRadius(s.values[a.key]!, a.domain, a.better), C, C, R);
            return (
              <circle
                key={`${s.id}-${a.key}`}
                cx={x}
                cy={y}
                r={13}
                fill="transparent"
                onMouseEnter={() => setHover({ s: s.id, k: a.key, x, y })}
                onMouseLeave={() => setHover(null)}
              />
            );
          }),
        )}
      </svg>

      {hover && hs && ha && (
        <div
          role="status"
          className="pointer-events-none absolute z-10 w-max max-w-60 -translate-x-1/2 -translate-y-[calc(100%+14px)] rounded-md border border-line-strong bg-ink-800 px-3 py-2 text-caption shadow-[var(--shadow-card-hover)]"
          style={{ left: `${(hover.x / SIZE) * 100}%`, top: `${(hover.y / SIZE) * 100}%` }}
        >
          <p className="flex items-center gap-2 font-medium text-fg">
            <span aria-hidden className="size-2 rounded-full" style={{ background: hs.color }} />
            {hs.label}
          </p>
          <p className="text-fg-muted">
            {ha.label}: <span className="num text-fg">{ha.format(hs.values[ha.key]!, hs.dashed)}</span>
          </p>
          <p className="text-fg-subtle">{hs.note}</p>
        </div>
      )}
    </div>
  );
}
