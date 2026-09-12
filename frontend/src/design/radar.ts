/**
 * Radar geometry, kept free of React so it is unit-tested directly.
 *
 * The seven opensr-test metrics span four orders of magnitude and point in both
 * directions, so each axis is log-scaled over the values present on that axis
 * and flipped where lower is better. Outer = better on every axis. Radii never
 * reach 0: the innermost value sits at MIN_RADIUS so every polygon stays visible.
 */
export const MIN_RADIUS = 0.12;

/** [min, max] of the positive values in `values`; throws when there are none. */
export function logDomain(values: readonly (number | null)[]): [number, number] {
  const v = values.filter((x): x is number => x !== null && Number.isFinite(x));
  if (v.length === 0) throw new Error("logDomain: no finite values on this axis");
  if (v.some((x) => x <= 0)) throw new Error(`logDomain: non-positive value ${Math.min(...v)} cannot be log-scaled`);
  return [Math.min(...v), Math.max(...v)];
}

/** Radius in [MIN_RADIUS, 1] for `value` on an axis with `domain`; 1 is the best value on the axis. */
export function radarRadius(value: number, domain: [number, number], better: "higher" | "lower"): number {
  const [lo, hi] = domain;
  if (hi === lo) return 1;
  const t = (Math.log(value) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
  const good = better === "higher" ? t : 1 - t;
  return MIN_RADIUS + (1 - MIN_RADIUS) * Math.min(1, Math.max(0, good));
}

/** Point for axis `i` of `n` (axis 0 points straight up, clockwise) at fraction `r` of `radius`. */
export function polar(i: number, n: number, r: number, cx: number, cy: number, radius: number): [number, number] {
  const a = -Math.PI / 2 + (i * 2 * Math.PI) / n;
  return [cx + Math.cos(a) * r * radius, cy + Math.sin(a) * r * radius];
}
