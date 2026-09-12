import "maplibre-gl/dist/maplibre-gl.css";
import App from "../App";

/** The live map demo (the previous single-page app), unchanged, lazy-loaded so MapLibre stays off other routes. */
export default function DemoPage() {
  return <App />;
}
