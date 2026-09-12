import type { HTMLAttributes } from "react";

/** Base surface. Hover lift is on by default; `accent` adds the cyan glow, `interactive={false}` disables hover. */
export function Card({
  accent = false,
  interactive = true,
  className = "",
  ...rest
}: HTMLAttributes<HTMLDivElement> & { accent?: boolean; interactive?: boolean }) {
  const cls = ["card", accent && "card-accent", !interactive && "card-static", className].filter(Boolean).join(" ");
  return <div className={cls} {...rest} />;
}
