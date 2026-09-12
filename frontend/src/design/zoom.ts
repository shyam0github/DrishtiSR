/**
 * Pure zoom/pan math for the lightbox. The content layer is exactly viewport
 * sized and transformed with `translate(x, y) scale(scale)`, origin top-left,
 * so at scale 1 with x = y = 0 the image fits the viewport.
 */
export interface View {
  scale: number;
  x: number;
  y: number;
}

export const IDENTITY: View = { scale: 1, x: 0, y: 0 };
export const MIN_SCALE = 1;
export const MAX_SCALE = 16;

export function clampScale(s: number): number {
  return Math.min(Math.max(s, MIN_SCALE), MAX_SCALE);
}

/** Keep the scaled layer covering the viewport: no empty margin can be dragged into view. */
export function clampPan(v: View, vw: number, vh: number): View {
  const minX = vw - vw * v.scale;
  const minY = vh - vh * v.scale;
  return { scale: v.scale, x: Math.min(0, Math.max(minX, v.x)), y: Math.min(0, Math.max(minY, v.y)) };
}

/** Multiply the scale by `factor`, keeping viewport point (px, py) fixed under the cursor. */
export function zoomAt(v: View, factor: number, px: number, py: number, vw: number, vh: number): View {
  const scale = clampScale(v.scale * factor);
  const k = scale / v.scale;
  return clampPan({ scale, x: px - (px - v.x) * k, y: py - (py - v.y) * k }, vw, vh);
}

export function panBy(v: View, dx: number, dy: number, vw: number, vh: number): View {
  return clampPan({ scale: v.scale, x: v.x + dx, y: v.y + dy }, vw, vh);
}
