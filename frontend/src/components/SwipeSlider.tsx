import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { Map as MlMap } from "maplibre-gl";
import type { RasterLayer } from "../api/client";
import { basemapStyle, removeLayer, upsertImageLayer } from "../map/layers";
import { AiBadge } from "./AiBadge";
import { MAP_CONTAINER_STYLE } from "./MapView";

const SR_LAYER_ID = "sr";
const UNCERTAINTY_LAYER_ID = "uncertainty";
/** Divider movement per arrow key, as a fraction of the width (Shift: large step). */
const KEY_STEP = 0.02;
const KEY_STEP_LARGE = 0.1;

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

interface Props {
  /** The interactive map underneath, which carries the input layer. This view follows its camera. */
  baseMap: MlMap;
  sr: RasterLayer | null;
  uncertainty: RasterLayer | null;
  uncertaintyVisible: boolean;
  uncertaintyOpacity: number;
  /** Label for the base map's layer, shown left of the divider. */
  leftLabel: string | null;
  onOverlayReady: (map: MlMap | null) => void;
}

/**
 * The swipe comparison. A second, non-interactive MapLibre map sits exactly on
 * top of the base map, camera-locked to it, and carries the SR and uncertainty
 * layers. A CSS clip-path shows it only right of a draggable divider, so the
 * input is on the left and the reconstruction on the right.
 *
 * The divider is driven by Pointer Events (one code path for mouse, touch and
 * pen) with pointer capture and `touch-action: none`, so dragging it on a
 * phone neither scrolls the page nor pans the map. It is also a keyboard
 * slider (arrows, Shift+arrows, Home, End).
 *
 * This is an SR view, so it always renders the AI badge.
 */
export function SwipeSlider({ baseMap, sr, uncertainty, uncertaintyVisible, uncertaintyOpacity, leftLabel, onOverlayReady }: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const onOverlayReadyRef = useRef(onOverlayReady);
  onOverlayReadyRef.current = onOverlayReady;
  const drag = useRef<{ pointerId: number; offsetPx: number } | null>(null);
  const [overlay, setOverlay] = useState<MlMap | null>(null);
  const [fraction, setFraction] = useState(0.5);
  const [srLoaded, setSrLoaded] = useState(false);

  // The overlay map, locked to the base map's camera.
  useEffect(() => {
    const container = overlayRef.current;
    if (!container) throw new Error("SwipeSlider: overlay container not mounted");
    const camera = () => ({ center: baseMap.getCenter(), zoom: baseMap.getZoom(), bearing: baseMap.getBearing(), pitch: baseMap.getPitch() });
    const map = new MlMap({ container, style: basemapStyle(), interactive: false, attributionControl: false, ...camera() });
    map.on("error", (e) => console.error("MapLibre (SR overlay):", e.error));
    const sync = () => map.jumpTo(camera());
    baseMap.on("move", sync);
    const resize = new ResizeObserver(() => {
      map.resize();
      sync();
    });
    resize.observe(container);
    let disposed = false;
    map.on("load", () => {
      if (disposed) return;
      setOverlay(map);
      onOverlayReadyRef.current(map);
    });
    return () => {
      disposed = true;
      baseMap.off("move", sync);
      resize.disconnect();
      setOverlay(null);
      onOverlayReadyRef.current(null);
      map.remove();
    };
  }, [baseMap]);

  useEffect(() => {
    if (!overlay) return;
    if (sr) upsertImageLayer(overlay, SR_LAYER_ID, sr.url, sr.footprint);
    else removeLayer(overlay, SR_LAYER_ID);
    if (uncertainty) upsertImageLayer(overlay, UNCERTAINTY_LAYER_ID, uncertainty.url, uncertainty.footprint, 0);
    else removeLayer(overlay, UNCERTAINTY_LAYER_ID);
  }, [overlay, sr, uncertainty]);

  useEffect(() => {
    if (!overlay || !overlay.getLayer(UNCERTAINTY_LAYER_ID)) return;
    overlay.setPaintProperty(UNCERTAINTY_LAYER_ID, "raster-opacity", uncertaintyVisible ? uncertaintyOpacity : 0);
  }, [overlay, uncertainty, uncertaintyVisible, uncertaintyOpacity]);

  // Exposed as data-sr-loaded, so a test can wait for real pixels rather than a timer.
  useEffect(() => {
    if (!overlay) return;
    const check = () => setSrLoaded(overlay.getSource(SR_LAYER_ID) !== undefined && overlay.isSourceLoaded(SR_LAYER_ID));
    check();
    overlay.on("sourcedata", check);
    overlay.on("idle", check);
    return () => {
      overlay.off("sourcedata", check);
      overlay.off("idle", check);
    };
  }, [overlay, sr]);

  const fractionAt = (clientX: number) => {
    const rect = rootRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return null;
    return clamp01((clientX - rect.left) / rect.width);
  };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (!e.isPrimary || e.button !== 0 || drag.current) return;
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    // Keep the grab offset, so the divider does not jump to the finger.
    const handle = e.currentTarget.getBoundingClientRect();
    drag.current = { pointerId: e.pointerId, offsetPx: e.clientX - (handle.left + handle.width / 2) };
  };

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (drag.current?.pointerId !== e.pointerId) return;
    const f = fractionAt(e.clientX - drag.current.offsetPx);
    if (f !== null) setFraction(f);
  };

  const endDrag = (e: PointerEvent<HTMLDivElement>) => {
    if (drag.current?.pointerId !== e.pointerId) return;
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? KEY_STEP_LARGE : KEY_STEP;
    const next: Record<string, (f: number) => number> = {
      ArrowLeft: (f) => f - step,
      ArrowRight: (f) => f + step,
      Home: () => 0,
      End: () => 1,
    };
    const move = next[e.key];
    if (!move) return;
    e.preventDefault();
    setFraction((f) => clamp01(move(f)));
  };

  const pct = fraction * 100;
  return (
    <div
      ref={rootRef}
      data-testid="swipe"
      data-fraction={fraction.toFixed(4)}
      data-sr-loaded={srLoaded ? "true" : "false"}
      className="pointer-events-none absolute inset-0"
    >
      <div ref={overlayRef} data-testid="swipe-overlay" style={{ ...MAP_CONTAINER_STYLE, clipPath: `inset(0 0 0 ${pct}%)` }} />

      {leftLabel && <SideLabel side="left" pct={pct} text={leftLabel} />}
      {sr && <SideLabel side="right" pct={pct} text={sr.label} />}

      <div
        data-testid="swipe-handle"
        role="slider"
        tabIndex={0}
        aria-label="Swipe between the 10 m input (left) and the AI reconstruction (right)"
        aria-orientation="horizontal"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pct)}
        className="group pointer-events-auto absolute inset-y-0 z-10 w-11 -translate-x-1/2 cursor-ew-resize touch-none select-none outline-none"
        style={{ left: `${pct}%` }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onLostPointerCapture={endDrag}
        onKeyDown={onKeyDown}
      >
        <div className="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.6)]" />
        <span
          data-testid="swipe-knob"
          className="absolute left-1/2 top-1/2 flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white text-xs font-bold text-black shadow-md ring-1 ring-black/50 group-focus-visible:ring-4 group-focus-visible:ring-blue-500"
        >
          ◀▶
        </span>
      </div>

      <AiBadge />
    </div>
  );
}

function SideLabel({ side, pct, text }: { side: "left" | "right"; pct: number; text: string }) {
  // Follows the divider; capped to its own side and ellipsised, never pushed off-screen.
  const room = side === "left" ? pct : 100 - pct;
  const style = {
    ...(side === "left" ? { right: `calc(${100 - pct}% + 1.75rem)` } : { left: `calc(${pct}% + 1.75rem)` }),
    maxWidth: `max(0px, calc(${room}% - 2.25rem))`,
  };
  return (
    <div
      className="pointer-events-none absolute bottom-10 z-10 overflow-hidden text-ellipsis whitespace-nowrap rounded bg-black/75 px-1.5 py-0.5 text-[11px] text-white"
      style={style}
      title={text}
    >
      {text}
    </div>
  );
}
