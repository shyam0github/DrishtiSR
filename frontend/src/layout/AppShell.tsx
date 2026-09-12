import { useEffect } from "react";
import { Outlet, ScrollRestoration, useLocation } from "react-router";
import { NAV_ROUTES } from "../routes";
import { TopNav, Wordmark } from "./TopNav";

/** Nav + route outlet + footer. The outlet is keyed by path so each route plays the entrance transition. */
export function AppShell() {
  const { pathname } = useLocation();
  useEffect(() => {
    const r = NAV_ROUTES.find((n) => n.path === pathname);
    document.title = r && r.path !== "/" ? `${r.title} · DrishtiSR` : "DrishtiSR — Sentinel-2 ×4 super-resolution";
  }, [pathname]);

  return (
    <div className="flex min-h-dvh flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-[60] focus:rounded-sm focus:bg-accent focus:px-3 focus:py-1.5 focus:text-ink-950">
        Skip to content
      </a>
      <TopNav />
      <main id="main" key={pathname} className="flex-1 animate-route-in">
        <Outlet />
      </main>
      <footer className="border-t border-line">
        <div className="page-container flex flex-col gap-4 py-10 text-caption text-fg-subtle sm:flex-row sm:items-center sm:justify-between">
          <Wordmark />
          <p>Sentinel-2 10 m → 2.5 m super-resolution · SIH26142</p>
        </div>
      </footer>
      <ScrollRestoration />
    </div>
  );
}
