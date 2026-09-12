import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from "react";
import { Link } from "react-router";

export type ButtonVariant = "primary" | "secondary" | "ghost";

const BASE =
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md font-medium transition-[background-color,border-color,color,box-shadow,transform] duration-200 active:translate-y-px disabled:pointer-events-none disabled:opacity-40";

const VARIANT: Record<ButtonVariant, string> = {
  primary: "bg-accent text-ink-950 shadow-glow-soft hover:bg-accent-strong hover:shadow-glow",
  secondary: "border border-line-strong bg-ink-850 text-fg hover:border-accent/50 hover:text-accent-strong",
  ghost: "text-fg-muted hover:bg-ink-800 hover:text-fg",
};

const SIZE = { md: "h-10 px-4 text-caption", lg: "h-12 px-6 text-body" } as const;

export function buttonClass(variant: ButtonVariant = "primary", size: keyof typeof SIZE = "md", className = ""): string {
  return `${BASE} ${VARIANT[variant]} ${SIZE[size]} ${className}`;
}

type Common = { variant?: ButtonVariant; size?: keyof typeof SIZE; className?: string; children: ReactNode };

export function Button({ variant, size, className, ...rest }: Common & ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button type="button" className={buttonClass(variant, size, className)} {...rest} />;
}

/** In-app route link styled as a button. */
export function ButtonLink({ to, variant, size, className, children, ...rest }: Common & { to: string; "data-testid"?: string }) {
  return (
    <Link to={to} className={buttonClass(variant, size, className)} {...rest}>
      {children}
    </Link>
  );
}

/** External or same-page anchor styled as a button. */
export function ButtonAnchor({ variant, size, className, ...rest }: Common & AnchorHTMLAttributes<HTMLAnchorElement>) {
  return <a className={buttonClass(variant, size, className)} {...rest} />;
}
