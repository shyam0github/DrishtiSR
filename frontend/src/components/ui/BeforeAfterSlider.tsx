import { useCallback, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { AiBadge } from "../AiBadge";
import { Lightbox } from "./Lightbox";
import { ImageSkeleton } from "./Skeleton";

export interface ComparePair {
  /** Left side, e.g. the 10 m Sentinel-2 input. */
  beforeSrc: string;
  /** Right side, e.g. the 2.5 m super-resolved output. */
  afterSrc: string;
  beforeLabel: string;
  afterLabel: string;
  alt: string;
  /**
   * Nearest-neighbour upscaling (default true): a 10 m pixel stays a visible
   * square instead of being smoothed to look finer than it is.
   */
  pixelated?: boolean;
}

const KEY_STEP = 2;
const CLICK_SLOP_PX = 4;

/** Clip for the "after" layer: it shows to the right of the divider at `pct`. */
export const afterClip = (pct: number) => `inset(0 0 0 ${pct}%)`;

/**
 * Swipe comparison. Drag the divider (mouse, touch or arrow keys); click the
 * image, or the expand button, to open the zoomable full-size lightbox.
 * Always carries the AI-reconstructed badge: the right side is model output.
 */
export function BeforeAfterSlider({
  loading = false,
  loadingLabel,
  aspect = "4 / 3",
  initial = 50,
  ...pair
}: ComparePair & { loading?: boolean; loadingLabel?: string; aspect?: string; initial?: number }) {
  const { beforeSrc, afterSrc, beforeLabel, afterLabel, alt, pixelated = true } = pair;
  const [pos, setPos] = useState(initial);
  const [loaded, setLoaded] = useState<Record<string, boolean>>({});
  const [failed, setFailed] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ id: number; x0: number; moved: boolean } | null>(null);

  const setFromClientX = useCallback((clientX: number) => {
    const r = boxRef.current?.getBoundingClientRect();
    if (!r || r.width === 0) return;
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
  }, []);

  const onHandleDown = (e: PointerEvent) => {
    e.stopPropagation();
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    drag.current = { id: e.pointerId, x0: e.clientX, moved: true };
  };
  const onMove = (e: PointerEvent) => {
    if (drag.current?.id === e.pointerId) setFromClientX(e.clientX);
  };
  const onUp = () => {
    drag.current = null;
  };

  // A press on the image that does not travel is a click: expand.
  const press = useRef<{ x: number; y: number } | null>(null);
  const onImageDown = (e: PointerEvent) => {
    press.current = { x: e.clientX, y: e.clientY };
  };
  const onImageUp = (e: PointerEvent) => {
    const p = press.current;
    press.current = null;
    if (p && Math.hypot(e.clientX - p.x, e.clientY - p.y) < CLICK_SLOP_PX && !busy) setOpen(true);
  };

  const onKey = (e: KeyboardEvent) => {
    const next = { ArrowLeft: pos - KEY_STEP, ArrowRight: pos + KEY_STEP, Home: 0, End: 100 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    setPos(Math.min(100, Math.max(0, next)));
  };

  const busy = loading || !loaded[beforeSrc] || !loaded[afterSrc];
  const imgCls = `absolute inset-0 h-full w-full select-none object-cover ${pixelated ? "[image-rendering:pixelated]" : ""}`;
  const markLoaded = (src: string) => () => setLoaded((l) => ({ ...l, [src]: true }));
  const markFailed = (src: string) => () => setFailed(src);

  return (
    <figure className="group flex flex-col gap-3">
      <div
        ref={boxRef}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        className="card card-static relative cursor-zoom-in touch-pan-y overflow-hidden rounded-lg"
        style={{ aspectRatio: aspect }}
      >
        <div className="absolute inset-0" onPointerDown={onImageDown} onPointerUp={onImageUp}>
          {!loading && (
            <>
              <img src={beforeSrc} alt={`${alt}: ${beforeLabel}`} draggable={false} onLoad={markLoaded(beforeSrc)} onError={markFailed(beforeSrc)} className={imgCls} />
              <img
                src={afterSrc}
                alt={`${alt}: ${afterLabel}`}
                draggable={false}
                onLoad={markLoaded(afterSrc)}
                onError={markFailed(afterSrc)}
                className={imgCls}
                style={{ clipPath: afterClip(pos) }}
              />
            </>
          )}
        </div>

        {failed ? (
          <div role="alert" className="absolute inset-0 grid place-items-center bg-ink-900 p-6 text-center text-caption text-bad">
            Image failed to load: <span className="num break-all">{failed}</span>
          </div>
        ) : (
          busy && <ImageSkeleton label={loadingLabel} aspect={aspect} className="absolute inset-0 rounded-none border-0" />
        )}

        {!busy && !failed && (
          <>
            <div
              role="slider"
              tabIndex={0}
              aria-label="Comparison divider"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(pos)}
              aria-valuetext={`${Math.round(pos)}% ${beforeLabel}`}
              onKeyDown={onKey}
              onPointerDown={onHandleDown}
              className="absolute top-0 bottom-0 z-10 flex w-10 -translate-x-1/2 cursor-ew-resize touch-none justify-center focus-visible:outline-none [&:focus-visible>span]:ring-2 [&:focus-visible>span]:ring-accent"
              style={{ left: `${pos}%` }}
            >
              <span className="h-full w-0.5 bg-fg shadow-[0_0_12px_rgb(34_211_238/0.8)]" />
              <span className="absolute top-1/2 grid size-9 -translate-y-1/2 place-items-center rounded-full border border-line-strong bg-ink-950/80 text-fg backdrop-blur transition-transform group-hover:scale-110">
                <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden fill="none" stroke="currentColor" strokeWidth="1.6">
                  <path d="M6 4 2 8l4 4M10 4l4 4-4 4" />
                </svg>
              </span>
            </div>
            <SideLabel side="left">{beforeLabel}</SideLabel>
            <SideLabel side="right">{afterLabel}</SideLabel>
            <button
              type="button"
              onClick={() => setOpen(true)}
              aria-label="Open full-size comparison with zoom"
              className="absolute top-2 left-2 z-20 grid size-9 place-items-center rounded-md border border-line-strong bg-ink-950/70 text-fg-muted opacity-80 backdrop-blur transition hover:text-accent hover:opacity-100"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden fill="none" stroke="currentColor" strokeWidth="1.6">
                <path d="M9.5 2.5h4v4M6.5 13.5h-4v-4M13.5 2.5 9 7M2.5 13.5 7 9" />
              </svg>
            </button>
          </>
        )}
        <AiBadge />
      </div>
      <figcaption className="text-caption text-fg-subtle">{alt}</figcaption>
      {open && <Lightbox {...pair} pixelated={pixelated} initialSplit={pos} onClose={() => setOpen(false)} />}
    </figure>
  );
}

export function SideLabel({ side, children }: { side: "left" | "right"; children: string }) {
  return (
    <span
      className={`pointer-events-none absolute bottom-3 z-10 rounded-full border border-line-strong bg-ink-950/75 px-2.5 py-1 font-mono text-micro uppercase tracking-wider text-fg backdrop-blur ${side === "left" ? "left-3" : "right-3"}`}
    >
      {children}
    </span>
  );
}
