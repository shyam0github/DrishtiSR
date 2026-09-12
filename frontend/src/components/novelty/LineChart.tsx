import { useLayoutEffect, useRef, useState, type PointerEvent } from "react";

export interface LineSeries {
  id: string;
  label: string;
  /** A validated categorical slot, never a status colour. */
  color: string;
  values: readonly number[];
  /** Dashed stroke: the reference curve (e.g. an oracle), not a method. */
  dashed?: boolean;
}

/** Horizontal threshold, drawn in neutral ink with its label. */
export interface RefLine {
  value: number;
  label: string;
}

const LABEL_GAP_PX = 13;

function niceStep(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(raw));
  const f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}

function ticks(lo: number, hi: number, n: number): number[] {
  const step = niceStep((hi - lo) / n);
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

/**
 * One-axis line chart in inline SVG, sized to its container. Legend above for
 * two or more series, direct end labels (nudged apart), a crosshair tooltip on
 * hover, and a table view, so identity never rests on colour alone.
 */
export function LineChart({
  x,
  series,
  xLabel,
  yLabel,
  xFormat,
  yFormat,
  yDomain,
  refLine,
  height = 280,
  title,
  testId,
}: {
  x: readonly number[];
  series: readonly LineSeries[];
  xLabel: string;
  yLabel: string;
  xFormat: (v: number) => string;
  yFormat: (v: number) => string;
  yDomain?: [number, number];
  refLine?: RefLine;
  height?: number;
  /** Accessible name of the plot. */
  title: string;
  testId?: string;
}) {
  if (x.length < 2) throw new Error("LineChart: need at least two x values");
  for (const s of series) if (s.values.length !== x.length) throw new Error(`LineChart: series ${s.id} has ${s.values.length} values for ${x.length} x`);

  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState<number | null>(null);
  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(([e]) => e && setWidth(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const multi = series.length > 1;
  const m = { top: 14, right: multi ? 112 : 28, bottom: 44, left: 60 };
  const iw = Math.max(width - m.left - m.right, 40);
  const ih = height - m.top - m.bottom;
  const x0 = x[0]!;
  const x1 = x[x.length - 1]!;
  const peak = Math.max(...series.flatMap((s) => s.values), refLine?.value ?? 0);
  const [y0, y1] = yDomain ?? [0, peak * 1.08];
  const sx = (v: number) => m.left + ((v - x0) / (x1 - x0)) * iw;
  const sy = (v: number) => m.top + ih - ((v - y0) / (y1 - y0)) * ih;
  const path = (vals: readonly number[]) => vals.map((v, i) => `${i ? "L" : "M"}${sx(x[i]!).toFixed(1)},${sy(v).toFixed(1)}`).join("");

  // End labels, pushed apart so near-equal curves stay legible.
  const ends = series.map((s) => ({ s, y: sy(s.values[s.values.length - 1]!) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i]!.y = Math.max(ends[i]!.y, ends[i - 1]!.y + LABEL_GAP_PX);

  const onMove = (e: PointerEvent<SVGRectElement>) => {
    const px = e.clientX - e.currentTarget.getBoundingClientRect().left + m.left;
    let best = 0;
    for (let i = 1; i < x.length; i++) if (Math.abs(sx(x[i]!) - px) < Math.abs(sx(x[best]!) - px)) best = i;
    setHover(best);
  };

  const hx = hover === null ? 0 : sx(x[hover]!);
  return (
    <figure className="flex flex-col gap-3" data-testid={testId}>
      {multi && (
        <ul className="flex flex-wrap gap-x-5 gap-y-2 text-caption text-fg-muted" aria-label="Legend">
          {series.map((s) => (
            <li key={s.id} className="inline-flex items-center gap-2">
              <svg aria-hidden width="20" height="6">
                <line x1="1" y1="3" x2="19" y2="3" stroke={s.color} strokeWidth="2" strokeLinecap="round" strokeDasharray={s.dashed ? "4 3" : undefined} />
              </svg>
              {s.label}
            </li>
          ))}
        </ul>
      )}
      <div ref={wrap} className="relative w-full">
        <svg role="img" aria-label={title} width={width} height={height} className="block overflow-visible">
          {ticks(y0, y1, 4).map((t) => (
            <g key={`y${t}`}>
              <line x1={m.left} x2={m.left + iw} y1={sy(t)} y2={sy(t)} stroke="var(--color-line)" />
              <text x={m.left - 8} y={sy(t)} dy="0.32em" textAnchor="end" className="num fill-fg-subtle text-[11px]">
                {yFormat(t)}
              </text>
            </g>
          ))}
          {ticks(x0, x1, 5).map((t) => (
            <text key={`x${t}`} x={sx(t)} y={m.top + ih + 18} textAnchor="middle" className="num fill-fg-subtle text-[11px]">
              {xFormat(t)}
            </text>
          ))}
          <line x1={m.left} x2={m.left + iw} y1={m.top + ih} y2={m.top + ih} stroke="var(--color-line-strong)" />
          <text x={m.left + iw / 2} y={height - 4} textAnchor="middle" className="fill-fg-muted text-[11px]">
            {xLabel}
          </text>
          <text x={m.left} y={m.top - 4} className="fill-fg-muted text-[11px]">
            {yLabel}
          </text>

          {refLine && (
            <g>
              <line x1={m.left} x2={m.left + iw} y1={sy(refLine.value)} y2={sy(refLine.value)} stroke="var(--color-fg-subtle)" strokeDasharray="2 4" />
              <text x={m.left + iw - 4} y={sy(refLine.value) - 5} textAnchor="end" className="fill-fg-muted text-[11px]">
                {refLine.label}
              </text>
            </g>
          )}

          {series.map((s) => (
            <path key={s.id} d={path(s.values)} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" strokeDasharray={s.dashed ? "5 4" : undefined} />
          ))}
          {!multi &&
            series[0]!.values.map((v, i) => <circle key={i} cx={sx(x[i]!)} cy={sy(v)} r="4" fill={series[0]!.color} stroke="var(--color-ink-850)" strokeWidth="2" />)}
          {multi &&
            ends.map(({ s, y }) => (
              <text key={s.id} x={m.left + iw + 8} y={y} dy="0.32em" className="fill-fg-muted text-[11px]">
                {s.label}
              </text>
            ))}

          {hover !== null && (
            <g pointerEvents="none">
              <line x1={hx} x2={hx} y1={m.top} y2={m.top + ih} stroke="var(--color-line-strong)" />
              {series.map((s) => (
                <circle key={s.id} cx={hx} cy={sy(s.values[hover]!)} r="4.5" fill={s.color} stroke="var(--color-ink-850)" strokeWidth="2" />
              ))}
            </g>
          )}
          <rect x={m.left} y={m.top} width={iw} height={ih} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} data-testid={testId && `${testId}-hit`} />
        </svg>
        {hover !== null && (
          <div
            role="tooltip"
            className="pointer-events-none absolute top-2 z-10 flex min-w-40 flex-col gap-1 rounded-md border border-line-strong bg-ink-800 p-3 text-caption shadow-lg"
            style={hx > width / 2 ? { right: width - hx + 12 } : { left: hx + 12 }}
          >
            <span className="text-fg-muted">
              {xLabel}: <span className="num text-fg">{xFormat(x[hover]!)}</span>
            </span>
            {series.map((s) => (
              <span key={s.id} className="flex items-center justify-between gap-4">
                <span className="inline-flex items-center gap-2 text-fg-muted">
                  <span aria-hidden className="h-0.5 w-3 rounded-full" style={{ background: s.color }} />
                  {s.label}
                </span>
                <span className="num text-fg">{yFormat(s.values[hover]!)}</span>
              </span>
            ))}
          </div>
        )}
      </div>
      <details className="text-caption text-fg-muted">
        <summary className="cursor-pointer">Table view</summary>
        <div className="mt-2 overflow-x-auto">
          <table className="num w-full text-left">
            <thead>
              <tr className="border-b border-line">
                <th className="py-1 pr-4 font-medium">{xLabel}</th>
                {series.map((s) => (
                  <th key={s.id} className="py-1 pr-4 font-medium">
                    {s.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {x.map((xv, i) => (
                <tr key={i} className="border-b border-line/50">
                  <td className="py-1 pr-4">{xFormat(xv)}</td>
                  {series.map((s) => (
                    <td key={s.id} className="py-1 pr-4 text-fg">
                      {yFormat(s.values[i]!)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
