import { useEffect, useRef, useState, type RefObject } from "react";

/** Motion constants shared by the JS-driven animations. CSS durations live in index.css @theme. */
export const MOTION = {
  countUpMs: 1400,
  staggerMs: 70,
  /** Fraction of an element that must be visible before it counts as "in view". */
  inViewThreshold: 0.25,
} as const;

/** easeOutExpo on t in [0, 1]; returns [0, 1]. Exactly 1 at t = 1. */
export function easeOutExpo(t: number): number {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  return 1 - 2 ** (-10 * t);
}

/** Value of a count-up from 0 to `target` after `elapsedMs` of a `durationMs` animation. */
export function countUpValue(target: number, elapsedMs: number, durationMs: number): number {
  if (durationMs <= 0) return target;
  return target * easeOutExpo(elapsedMs / durationMs);
}

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * True once the element has scrolled into view; stays true (one-shot), so
 * entrances and count-ups never replay on scroll back.
 */
export function useInView<T extends Element>(threshold: number = MOTION.inViewThreshold): [RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || inView) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setInView(true);
          io.disconnect();
        }
      },
      { threshold, rootMargin: "0px 0px -8% 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [threshold, inView]);
  return [ref, inView];
}

/** Animates 0 -> target once `start` becomes true. Jumps straight to target under reduced motion. */
export function useCountUp(target: number, start: boolean, durationMs: number = MOTION.countUpMs): number {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (!start) return;
    if (prefersReducedMotion() || !Number.isFinite(target)) {
      setValue(target);
      return;
    }
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const elapsed = now - t0;
      setValue(countUpValue(target, elapsed, durationMs));
      if (elapsed < durationMs) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, start, durationMs]);
  return value;
}

/** Fixed-decimal formatting with thousands separators, for mono stat numbers. */
export function formatNumber(value: number, decimals = 0): string {
  return value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}
