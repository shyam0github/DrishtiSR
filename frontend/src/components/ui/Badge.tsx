import type { ReactNode } from "react";

export type BadgeVariant = "ours" | "baseline" | "literature" | "pending" | "neutral" | "good" | "warn" | "bad";

const VARIANT: Record<BadgeVariant, string> = {
  ours: "border-accent/40 bg-accent-dim text-accent-strong",
  baseline: "border-line-strong bg-ink-800 text-fg-muted",
  literature: "border-lit/30 bg-lit/10 text-lit",
  pending: "border-dashed border-warn/50 bg-transparent text-warn",
  neutral: "border-line bg-ink-800 text-fg-muted",
  good: "border-good/35 bg-good/10 text-good",
  warn: "border-warn/35 bg-warn/10 text-warn",
  bad: "border-bad/35 bg-bad/10 text-bad",
};

const DOT: Partial<Record<BadgeVariant, string>> = {
  ours: "bg-accent shadow-[0_0_8px_var(--color-accent)]",
  pending: "bg-warn animate-pulse",
};

/** Pill label for row provenance: "Ours" / "Baseline" / "Literature" / "Pending", plus status tones. */
export function Badge({ variant = "neutral", children, dot }: { variant?: BadgeVariant; children: ReactNode; dot?: boolean }) {
  const dotClass = DOT[variant] ?? "bg-current";
  return (
    <span
      className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-1 font-mono text-micro uppercase ${VARIANT[variant]}`}
    >
      {(dot ?? variant in DOT) && <span aria-hidden className={`size-1.5 rounded-full ${dotClass}`} />}
      {children}
    </span>
  );
}
