/**
 * The site map: one list drives both the router and the top nav, so a route
 * cannot exist without a nav entry (or the reverse) by accident.
 */
export interface NavRoute {
  path: string;
  /** Short nav label. */
  label: string;
  /** Full page title, used for document.title. */
  title: string;
  /** Optional small index shown before the label, e.g. "01" for the novelties. */
  index?: string;
}

export const NAV_ROUTES: readonly NavRoute[] = [
  { path: "/", label: "Home", title: "DrishtiSR" },
  { path: "/novelty/spectral", label: "Spectral", title: "Spectral consistency", index: "01" },
  { path: "/novelty/uncertainty", label: "Uncertainty", title: "Per-pixel uncertainty", index: "02" },
  { path: "/novelty/efficiency", label: "Efficiency", title: "CPU-only efficiency", index: "03" },
  { path: "/compare", label: "Compare", title: "Comparison" },
  { path: "/about", label: "About", title: "About" },
];

/** Routes outside the nav: the live map demo and the component reference. */
export const DEMO_PATH = "/demo";
export const DESIGN_PATH = "/design";
