import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router";
import "./index.css";
import { AppShell } from "./layout/AppShell";
import { PlaceholderPage } from "./pages/PlaceholderPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { DEMO_PATH, DESIGN_PATH, NAV_ROUTES } from "./routes";

const DemoPage = lazy(() => import("./pages/DemoPage"));
const DesignSystemPage = lazy(() => import("./pages/DesignSystemPage"));

const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      ...NAV_ROUTES.map((r) => ({ path: r.path, element: <PlaceholderPage route={r} /> })),
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
    <RouterProvider router={router} />
  </StrictMode>,
);
