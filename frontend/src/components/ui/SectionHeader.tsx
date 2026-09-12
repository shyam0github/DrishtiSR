import type { ReactNode } from "react";

type Level = "h1" | "h2" | "h3";

/** Eyebrow + headline + optional lede. `level` sets the semantic tag; the visual size follows it. */
export function SectionHeader({
  eyebrow,
  title,
  description,
  level = "h2",
  align = "left",
  actions,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  level?: Level;
  align?: "left" | "center";
  actions?: ReactNode;
}) {
  const Tag = level;
  const centered = align === "center";
  return (
    <header className={`flex flex-col gap-6 md:flex-row md:items-end md:justify-between ${centered ? "items-center text-center md:flex-col md:items-center" : ""}`}>
      <div className={`flex max-w-prose flex-col gap-4 ${centered ? "items-center" : ""}`}>
        {eyebrow && (
          <p className="eyebrow inline-flex items-center gap-2">
            <span aria-hidden className="h-px w-6 bg-accent/60" />
            {eyebrow}
          </p>
        )}
        <Tag>{title}</Tag>
        {description && <p className="text-lead text-fg-muted">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 gap-3">{actions}</div>}
    </header>
  );
}
