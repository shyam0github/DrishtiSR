/**
 * Number of tiles the served predictor runs for an LR image of `h` x `w` px.
 * Mirrors src/infer/predictor.py TorchPredictor._tiling (which OnnxPredictor
 * reuses) and src/infer/tiled.py sr_array: an image no larger than one tile in
 * both axes is ONE tile of its own size; otherwise tiles start every
 * `tile - overlap` px over `range(0, max(1, len - overlap), step)` per axis.
 */
export function tileCount(h: number, w: number, tile: number, overlap: number): number {
  if (!(overlap >= 0 && overlap < tile)) throw new RangeError(`tileCount: overlap ${overlap} must be in [0, tile=${tile})`);
  if (!(h > 0 && w > 0)) throw new RangeError(`tileCount: image must be non-empty, got ${h} x ${w}`);
  if (h <= tile && w <= tile) return 1;
  const step = tile - overlap;
  const along = (len: number) => Math.ceil(Math.max(1, len - overlap) / step);
  return along(h) * along(w);
}
