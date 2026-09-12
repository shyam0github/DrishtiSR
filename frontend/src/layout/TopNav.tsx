import { useEffect, useState } from "react";
import { Link, NavLink, useLocation } from "react-router";
import { NAV_ROUTES } from "../routes";

export function Wordmark() {
  return (
    <Link to="/" className="group flex items-center gap-2.5" aria-label="DrishtiSR home">
      <span aria-hidden className="relative grid size-7 place-items-center rounded-md border border-accent/40 bg-accent-dim">
        <span className="size-2 rounded-full bg-accent shadow-[0_0_10px_var(--color-accent)] transition-transform duration-300 group-hover:scale-125" />
        <span className="absolute inset-1 rounded-sm border border-accent/25" />
      </span>
      <span className="font-display text-[1.125rem] font-semibold tracking-tight text-fg">
        Drishti<span className="text-accent">SR</span>
      </span>
    </Link>
  );
}

/** Persistent sticky nav. Links come from NAV_ROUTES; the active one gets the accent pill. */
export function TopNav() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const { pathname } = useLocation();

  useEffect(() => setOpen(false), [pathname]);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const linkCls = ({ isActive }: { isActive: boolean }) =>
    `relative flex items-center gap-1.5 rounded-sm px-3 py-1.5 text-caption font-medium transition-colors ${
      isActive ? "bg-accent-dim text-accent-strong shadow-[inset_0_0_0_1px_rgb(34_211_238/0.25)]" : "text-fg-muted hover:bg-ink-800 hover:text-fg"
    }`;

  const links = NAV_ROUTES.map((r) => (
    <NavLink key={r.path} to={r.path} end={r.path === "/"} className={linkCls} data-testid={`nav-${r.label.toLowerCase()}`}>
      {r.index && <span className="num text-[10px] opacity-60">{r.index}</span>}
      {r.label}
    </NavLink>
  ));

  return (
    <header
      className={`sticky top-0 z-50 border-b backdrop-blur-xl transition-colors duration-300 ${
        scrolled || open ? "border-line bg-ink-950/80" : "border-transparent bg-ink-950/40"
      }`}
    >
      <nav aria-label="Primary" className="page-container flex h-nav items-center justify-between gap-6">
        <Wordmark />
        <div className="hidden items-center gap-1 lg:flex">{links}</div>
        <button
          type="button"
          className="grid size-9 place-items-center rounded-md border border-line text-fg-muted lg:hidden"
          aria-expanded={open}
          aria-controls="mobile-nav"
          aria-label={open ? "Close menu" : "Open menu"}
          onClick={() => setOpen((o) => !o)}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
            {open ? <path d="m4 4 8 8M12 4l-8 8" /> : <path d="M2.5 5h11M2.5 11h11" />}
          </svg>
        </button>
      </nav>
      {open && (
        <div id="mobile-nav" className="page-container flex animate-route-in flex-col gap-1 pb-4 lg:hidden">
          {links}
        </div>
      )}
    </header>
  );
}
