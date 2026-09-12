import { useCallback, useEffect, useRef, useState, type PointerEvent as RPointerEvent, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { IDENTITY, MAX_SCALE, panBy, zoomAt, type View } from "../../design/zoom";
import { AiBadge } from "../AiBadge";
import { afterClip, SideLabel, type ComparePair } from "./BeforeAfterSlider";

const WHEEL_SENSITIVITY = 0.0015;
const BUTTON_STEP = 1.5;
const DOUBLE_CLICK_SCALE = 3;
const KEY_SPLIT_STEP = 2;

/**
 * Full-screen comparison with zoom and pan. Both layers share one transform,
 * so the same ground point sits under the divider on each side; the divider is
 * in screen space and independent of zoom.
 *
 * Wheel / pinch / double-click / +,-,0 keys zoom; drag pans; ←/→ move the
 * divider; Esc closes.
 */
export function Lightbox({
  beforeSrc,
  afterSrc,
  beforeLabel,
  afterLabel,
  alt,
  pixelated = true,
  initialSplit = 50,
  onClose,
}: ComparePair & { initialSplit?: number; onClose: () => void }) {
  const [view, setView] = useState<View>(IDENTITY);
  const [split, setSplit] = useState(initialSplit);
  const stageRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef<{ dist: number } | null>(null);
  const dividerDrag = useRef<number | null>(null);

  const size = () => {
    const r = stageRef.current?.getBoundingClientRect();
    return { vw: r?.width ?? 1, vh: r?.height ?? 1, left: r?.left ?? 0, top: r?.top ?? 0 };
  };
  const zoomBy = useCallback((factor: number, clientX?: number, clientY?: number) => {
    const { vw, vh, left, top } = size();
    const px = clientX === undefined ? vw / 2 : clientX - left;
    const py = clientY === undefined ? vh / 2 : clientY - top;
    setView((v) => zoomAt(v, factor, px, py, vw, vh));
  }, []);

  // Scroll lock, focus in, focus back out.
  useEffect(() => {
    const prevFocus = document.activeElement as HTMLElement | null;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    return () => {
      document.body.style.overflow = prevOverflow;
      prevFocus?.focus();
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "+" || e.key === "=") zoomBy(BUTTON_STEP);
      else if (e.key === "-") zoomBy(1 / BUTTON_STEP);
      else if (e.key === "0") setView(IDENTITY);
      else if (e.key === "ArrowLeft") setSplit((s) => Math.max(0, s - KEY_SPLIT_STEP));
      else if (e.key === "ArrowRight") setSplit((s) => Math.min(100, s + KEY_SPLIT_STEP));
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, zoomBy]);

  // Wheel must be non-passive to stop page zoom/scroll, so attach it natively.
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      zoomBy(Math.exp(-e.deltaY * WHEEL_SENSITIVITY), e.clientX, e.clientY);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomBy]);

  const onDown = (e: RPointerEvent) => {
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()] as [{ x: number; y: number }, { x: number; y: number }];
      pinch.current = { dist: Math.hypot(a.x - b.x, a.y - b.y) };
    }
  };
  const onMove = (e: RPointerEvent) => {
    if (dividerDrag.current === e.pointerId) {
      const { vw, left } = size();
      setSplit(Math.min(100, Math.max(0, ((e.clientX - left) / vw) * 100)));
      return;
    }
    const prev = pointers.current.get(e.pointerId);
    if (!prev) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const { vw, vh } = size();
    if (pointers.current.size === 2 && pinch.current) {
      const [a, b] = [...pointers.current.values()] as [{ x: number; y: number }, { x: number; y: number }];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch.current.dist > 0) zoomBy(dist / pinch.current.dist, (a.x + b.x) / 2, (a.y + b.y) / 2);
      pinch.current.dist = dist;
    } else if (pointers.current.size === 1) {
      setView((v) => panBy(v, e.clientX - prev.x, e.clientY - prev.y, vw, vh));
    }
  };
  const onUp = (e: RPointerEvent) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size < 2) pinch.current = null;
    if (dividerDrag.current === e.pointerId) dividerDrag.current = null;
  };

  const transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
  const layer = `absolute inset-0 origin-top-left will-change-transform`;
  const img = `h-full w-full select-none object-contain ${pixelated ? "[image-rendering:pixelated]" : ""}`;
  const dragging = pointers.current.size > 0;

  return createPortal(
    <div role="dialog" aria-modal="true" aria-label={`${alt} — full-size comparison`} className="fixed inset-0 z-[100] flex animate-route-in flex-col bg-ink-950/95 backdrop-blur-md">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3 sm:px-6">
        <p className="min-w-0 truncate text-caption text-fg-muted">{alt}</p>
        <div className="flex items-center gap-1.5">
          <ToolButton label="Zoom out" onClick={() => zoomBy(1 / BUTTON_STEP)} disabled={view.scale <= 1}>
            <path d="M3 8h10" />
          </ToolButton>
          <span className="num w-14 text-center text-caption text-fg" aria-live="polite">
            {Math.round(view.scale * 100)}%
          </span>
          <ToolButton label="Zoom in" onClick={() => zoomBy(BUTTON_STEP)} disabled={view.scale >= MAX_SCALE}>
            <path d="M3 8h10M8 3v10" />
          </ToolButton>
          <ToolButton label="Reset zoom" onClick={() => setView(IDENTITY)} disabled={view.scale === 1}>
            <path d="M3 8a5 5 0 1 0 1.5-3.5M3 3v2.5h2.5" />
          </ToolButton>
          <span className="mx-1 h-5 w-px bg-line" />
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="grid size-9 place-items-center rounded-md border border-line text-fg-muted transition hover:border-line-strong hover:text-fg"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
              <path d="m4 4 8 8M12 4l-8 8" />
            </svg>
          </button>
        </div>
      </div>

      <div
        ref={stageRef}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        onDoubleClick={(e) => (view.scale > 1 ? setView(IDENTITY) : zoomBy(DOUBLE_CLICK_SCALE, e.clientX, e.clientY))}
        className={`relative m-3 flex-1 touch-none overflow-hidden rounded-lg border border-line bg-ink-900 sm:m-6 ${view.scale > 1 ? (dragging ? "cursor-grabbing" : "cursor-grab") : "cursor-zoom-in"}`}
      >
        <div className={layer} style={{ transform }}>
          <img src={beforeSrc} alt={`${alt}: ${beforeLabel}`} draggable={false} className={img} />
        </div>
        <div className="absolute inset-0" style={{ clipPath: afterClip(split) }}>
          <div className={layer} style={{ transform }}>
            <img src={afterSrc} alt={`${alt}: ${afterLabel}`} draggable={false} className={img} />
          </div>
        </div>

        <div
          role="slider"
          tabIndex={0}
          aria-label="Comparison divider"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(split)}
          onPointerDown={(e) => {
            e.stopPropagation();
            (e.currentTarget as Element).setPointerCapture(e.pointerId);
            dividerDrag.current = e.pointerId;
          }}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onDoubleClick={(e) => e.stopPropagation()}
          className="absolute top-0 bottom-0 z-10 flex w-10 -translate-x-1/2 cursor-ew-resize touch-none justify-center"
          style={{ left: `${split}%` }}
        >
          <span className="h-full w-0.5 bg-fg shadow-[0_0_12px_rgb(34_211_238/0.8)]" />
          <span className="absolute top-1/2 grid size-10 -translate-y-1/2 place-items-center rounded-full border border-line-strong bg-ink-950/80 text-fg backdrop-blur">
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M6 4 2 8l4 4M10 4l4 4-4 4" />
            </svg>
          </span>
        </div>
        <SideLabel side="left">{beforeLabel}</SideLabel>
        <SideLabel side="right">{afterLabel}</SideLabel>
        <AiBadge />
      </div>
      <p className="hidden pb-4 text-center font-mono text-micro uppercase tracking-wider text-fg-subtle sm:block">
        Scroll or pinch to zoom · drag to pan · double-click to toggle · ←/→ divider · Esc to close
      </p>
    </div>,
    document.body,
  );
}

function ToolButton({ label, onClick, disabled, children }: { label: string; onClick: () => void; disabled?: boolean; children: ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      disabled={disabled}
      className="grid size-9 place-items-center rounded-md border border-line text-fg-muted transition hover:border-line-strong hover:text-accent disabled:opacity-35 disabled:hover:text-fg-muted"
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
        {children}
      </svg>
    </button>
  );
}
