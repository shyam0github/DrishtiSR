import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Map as MlMap } from "maplibre-gl";
import { APP_CONFIG } from "../config";
import { bboxFromCorners, estimateAoi, type Bbox, type LngLat } from "../geo/aoi";
import { setAoiLayer } from "../map/layers";

/** A press that moves less than this many screen px is a tap, not a rectangle. */
const MIN_DRAG_PX = 6;

const count = new Intl.NumberFormat("en-US");
const fmtM = (m: number) => `${Number.isInteger(m) ? m : m.toFixed(1)} m`;

interface Props {
  /** The map that receives the drawing gesture. */
  map: MlMap;
  /** Every map the rectangle is drawn on (both halves of the swipe view). */
  renderOn: MlMap[];
  bbox: Bbox | null;
  onChange: (bbox: Bbox | null) => void;
  onRun: (bbox: Bbox) => void;
  running: boolean;
  /** Shown under the run button: the last request's error or provenance. */
  runStatus?: ReactNode;
}

/**
 * Draw a rectangle on the map and report its EPSG:4326 bbox, the implied pixel
 * dimensions at the input and output GSD, and its cost in inference tiles. An
 * AOI over the tile budget is flagged and cannot be submitted. Renders two
 * panel sections, "Area of interest" and "Run super-resolution", because the
 * run button's enabled state depends on the drawing state and the estimate.
 *
 * The gesture uses Pointer Events on the map's canvas container, with the
 * map's own pan/zoom handlers disabled while drawing, so a mouse drag and a
 * one-finger drag draw the same rectangle.
 */
export function AoiPicker({ map, renderOn, bbox, onChange, onRun, running, runStatus }: Props) {
  const [drawing, setDrawing] = useState(false);
  const estimate = useMemo(() => (bbox ? estimateAoi(bbox, APP_CONFIG) : null), [bbox]);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const bboxRef = useRef(bbox);
  bboxRef.current = bbox;

  useEffect(() => {
    for (const m of renderOn) setAoiLayer(m, bbox, estimate?.overBudget ?? false);
  }, [renderOn, bbox, estimate]);

  useEffect(() => {
    if (!drawing) return;
    const el = map.getCanvasContainer();
    const handlers = [map.dragPan, map.dragRotate, map.touchZoomRotate, map.touchPitch, map.boxZoom, map.doubleClickZoom];
    for (const h of handlers) h.disable();
    const previous = { cursor: el.style.cursor, touchAction: el.style.touchAction };
    el.style.cursor = "crosshair";
    // With the touch handlers off, MapLibre drops its own touch-action rule and
    // the browser would claim the finger for scrolling (pointercancel).
    el.style.touchAction = "none";

    let active: { pointerId: number; x: number; y: number; start: LngLat; before: Bbox | null } | null = null;
    const lngLatAt = (e: PointerEvent): LngLat => {
      const r = el.getBoundingClientRect();
      const ll = map.unproject([e.clientX - r.left, e.clientY - r.top]);
      return [ll.lng, ll.lat];
    };
    const moved = (e: PointerEvent) => active !== null && Math.hypot(e.clientX - active.x, e.clientY - active.y) >= MIN_DRAG_PX;

    const down = (e: PointerEvent) => {
      if (!e.isPrimary || active) return;
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      active = { pointerId: e.pointerId, x: e.clientX, y: e.clientY, start: lngLatAt(e), before: bboxRef.current };
    };
    const move = (e: PointerEvent) => {
      if (active?.pointerId !== e.pointerId || !moved(e)) return;
      onChangeRef.current(bboxFromCorners(active.start, lngLatAt(e)));
    };
    const up = (e: PointerEvent) => {
      if (active?.pointerId !== e.pointerId) return;
      const wasDrag = moved(e);
      if (wasDrag) onChangeRef.current(bboxFromCorners(active.start, lngLatAt(e)));
      active = null;
      if (wasDrag) setDrawing(false); // a tap leaves drawing mode on
    };
    const cancel = (e: PointerEvent) => {
      if (active?.pointerId !== e.pointerId) return;
      onChangeRef.current(active.before);
      active = null;
    };
    const key = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (active) onChangeRef.current(active.before);
      active = null;
      setDrawing(false);
    };

    el.addEventListener("pointerdown", down);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", cancel);
    window.addEventListener("keydown", key);
    return () => {
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", cancel);
      window.removeEventListener("keydown", key);
      for (const h of handlers) h.enable();
      el.style.cursor = previous.cursor;
      el.style.touchAction = previous.touchAction;
    };
  }, [drawing, map]);

  const button = "rounded border border-slate-400 px-3 py-1.5 font-medium disabled:cursor-not-allowed disabled:opacity-40";
  return (
    <>
    <section className="space-y-2" data-testid="aoi">
      <h2 className="font-semibold">Area of interest</h2>
      <div className="flex flex-wrap gap-2">
        <button type="button" data-testid="aoi-draw" className={`${button} ${drawing ? "bg-blue-600 text-white" : "bg-white"}`} onClick={() => setDrawing((d) => !d)}>
          {drawing ? "Cancel drawing" : "Draw AOI"}
        </button>
      </div>
      {drawing && <p className="text-xs text-blue-700">Drag a rectangle on the map, with a mouse or a finger. Esc cancels.</p>}

      {bbox && estimate ? (
        <dl data-testid="aoi-bbox" className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
          <dt className="text-slate-500">bbox, EPSG:4326</dt>
          <dd className="font-mono break-all">{bbox.map((v) => v.toFixed(5)).join(", ")}</dd>
          <dt />
          <dd className="text-slate-500">west, south, east, north</dd>
          <dt className="text-slate-500">extent</dt>
          <dd>
            ≈ {(estimate.widthM / 1000).toFixed(2)} × {(estimate.heightM / 1000).toFixed(2)} km
          </dd>
          <dt className="text-slate-500">{fmtM(APP_CONFIG.lrGsdM)} input</dt>
          <dd>
            ≈ {count.format(estimate.lrPx.width)} × {count.format(estimate.lrPx.height)} px
          </dd>
          <dt className="text-slate-500">{fmtM(estimate.srGsdM)} output</dt>
          <dd>
            ≈ {count.format(estimate.srPx.width)} × {count.format(estimate.srPx.height)} px · ≈ {count.format(Math.ceil(estimate.outputBytes / 1e6))} MB
            <span className="text-slate-500">
              {" "}({APP_CONFIG.bands.length} bands × {APP_CONFIG.outputBytesPerBand * 8}-bit)
            </span>
          </dd>
          <dt className="text-slate-500">tiles</dt>
          <dd data-testid="aoi-tiles" className={estimate.overBudget ? "font-semibold text-red-700" : undefined}>
            {estimate.tiles.x} × {estimate.tiles.y} = {count.format(estimate.tiles.total)} of {estimate.tiles.max} allowed
          </dd>
        </dl>
      ) : (
        <p className="text-xs text-slate-500">No AOI yet.</p>
      )}

      {estimate?.overBudget && (
        <div role="alert" data-testid="aoi-over-budget" className="rounded border border-red-600 bg-red-50 p-2 text-xs text-red-800">
          <b>Too large for the demo.</b> This AOI needs {count.format(estimate.tiles.total)} tiles of {APP_CONFIG.tile.lrPx} × {APP_CONFIG.tile.lrPx} px;
          the budget is {estimate.tiles.max}. Draw a smaller area; this one will not be submitted.
        </div>
      )}
    </section>

    <section className="space-y-2" data-testid="run">
      <h2 className="font-semibold">Run super-resolution</h2>
      <button
        type="button"
        data-testid="aoi-run"
        className={`${button} bg-slate-900 text-white`}
        disabled={!bbox || !estimate || estimate.overBudget || running || drawing}
        onClick={() => bbox && onRun(bbox)}
      >
        {running ? "Running…" : "Super-resolve AOI"}
      </button>
      {runStatus}
    </section>
    </>
  );
}
