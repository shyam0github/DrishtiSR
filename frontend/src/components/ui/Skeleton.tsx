/** Shimmer block for pending content. Size it with className (e.g. "h-4 w-32"). */
export function Skeleton({ className = "", rounded = "rounded-sm" }: { className?: string; rounded?: string }) {
  return <div aria-hidden className={`skeleton ${rounded} ${className}`} />;
}

/**
 * Placeholder for an image still being super-resolved: shimmer, a scanline
 * sweep and a status line. `label` should say what is happening, not "Loading".
 */
export function ImageSkeleton({ label = "Super-resolving tile…", aspect = "4 / 3", className = "" }: { label?: string; aspect?: string; className?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={`relative overflow-hidden rounded-lg border border-line bg-ink-900 ${className}`}
      style={{ aspectRatio: aspect }}
    >
      <div className="skeleton absolute inset-0 opacity-60" />
      <div className="bg-graticule absolute inset-0 [mask-image:none]" />
      <div className="absolute inset-x-0 h-16 animate-scan bg-gradient-to-b from-transparent via-accent/15 to-transparent" />
      <div className="absolute inset-x-0 bottom-0 flex items-center gap-2 p-4 font-mono text-caption text-fg-muted">
        <span className="size-1.5 animate-pulse rounded-full bg-accent" />
        {label}
      </div>
    </div>
  );
}

/** Skeleton shaped like a StatCard, for metrics still loading. */
export function StatCardSkeleton() {
  return (
    <div className="card card-static flex flex-col gap-5 p-6" aria-hidden>
      <Skeleton className="h-3.5 w-28" />
      <Skeleton className="h-10 w-36" />
      <Skeleton className="h-3.5 w-20" />
    </div>
  );
}
