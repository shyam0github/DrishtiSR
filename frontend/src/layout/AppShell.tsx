import { useEffect } from "react";
import { Link, Outlet, ScrollRestoration, useLocation } from "react-router";
import { Icon } from "../components/ui";
import { APP_CONFIG } from "../config";
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
      <footer className="border-t border-line" data-testid="site-footer">
        <div className="page-container flex flex-col gap-6 py-10 text-caption text-fg-subtle md:flex-row md:items-center md:justify-between">
          <div className="flex flex-col gap-2">
            <Wordmark />
            <p>
              Team <span className="font-medium text-fg-muted">{APP_CONFIG.teamName}</span> · Smart India Hackathon 2026 ·{" "}
              <span className="num text-fg-muted">SIH26142</span>
            </p>
          </div>
          <nav aria-label="Footer" className="flex flex-wrap items-center gap-1">
            <Link to="/about" className="rounded-sm px-3 py-1.5 font-medium text-fg-muted transition-colors hover:bg-ink-800 hover:text-fg">
              About
            </Link>
            <a
              href={APP_CONFIG.repoUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-sm px-3 py-1.5 font-medium text-fg-muted transition-colors hover:bg-ink-800 hover:text-fg"
            >
              <Icon name="github" />
              GitHub
            </a>
          </nav>
        </div>
      </footer>
      <ScrollRestoration />
    </div>
  );
}
