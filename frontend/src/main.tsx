import { lazy, StrictMode, Suspense, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router";
import "./index.css";
import { AppShell } from "./layout/AppShell";
import { AboutPage } from "./pages/AboutPage";
import { EfficiencyPage } from "./pages/EfficiencyPage";
import { HomePage } from "./pages/HomePage";
import { PlaceholderPage } from "./pages/PlaceholderPage";
import { SpectralPage } from "./pages/SpectralPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { DEMO_PATH, DESIGN_PATH, NAV_ROUTES } from "./routes";
import { ResultProvider } from "./state/ResultContext";

const DemoPage = lazy(() => import("./pages/DemoPage"));
const DesignSystemPage = lazy(() => import("./pages/DesignSystemPage"));
const ComparePage = lazy(() => import("./pages/ComparePage"));

/** Built pages by nav path; any nav route not listed here renders the placeholder. */
const PAGES: Record<string, ReactNode> = {
  "/": <HomePage />,
  "/novelty/spectral": <SpectralPage />,
  "/novelty/efficiency": <EfficiencyPage />,
  "/about": <AboutPage />,
  "/compare": <Suspense><ComparePage /></Suspense>,
};

const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      ...NAV_ROUTES.map((r) => ({ path: r.path, element: PAGES[r.path] ?? <PlaceholderPage route={r} /> })),
      { path: DESIGN_PATH, element: <Suspense><DesignSystemPage /></Suspense> },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
  // Full-screen map demo: outside the shell, it owns the whole viewport.
  { path: DEMO_PATH, element: <Suspense><DemoPage /></Suspense> },
]);

const root = document.getElementById("root");
if (!root) throw new Error("index.html has no #root element");

createRoot(root).render(
  <StrictMode>
    <ResultProvider>
      <RouterProvider router={router} />
    </ResultProvider>
  </StrictMode>,
);
