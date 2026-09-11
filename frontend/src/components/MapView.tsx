import { useEffect, useRef, type ReactNode } from "react";
import { Map as MlMap, NavigationControl } from "maplibre-gl";
import { APP_CONFIG } from "../config";
import { basemapStyle } from "../map/layers";

interface Props {
  /** Called once the style has loaded and layers can be added. */
  onReady: (map: MlMap) => void;
  /** Overlays positioned over the map (the swipe view). */
  children?: ReactNode;
}

/** The interactive base map. Everything else on the map follows its camera. */
export function MapView({ onReady, children }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) throw new Error("MapView: container not mounted");
    const map = new MlMap({
      container,
      style: basemapStyle(),
      center: APP_CONFIG.map.center,
      zoom: APP_CONFIG.map.zoom,
      attributionControl: { compact: true },
    });
    map.addControl(new NavigationControl({ showCompass: false }), "top-left");
    map.on("error", (e) => console.error("MapLibre (base map):", e.error));
    let disposed = false;
    map.on("load", () => {
      if (!disposed) onReadyRef.current(map);
    });
    const resize = new ResizeObserver(() => map.resize());
    resize.observe(container);
    return () => {
      disposed = true;
      resize.disconnect();
      map.remove();
    };
  }, []);

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} style={MAP_CONTAINER_STYLE} />
      {children}
    </div>
  );
}

/**
 * Positioning for an element MapLibre takes over, as an INLINE style.
 *
 * MapLibre adds `.maplibregl-map { position: relative }` to its container. Its
 * stylesheet is unlayered, and Tailwind v4 utilities live in
 * `@layer utilities`, and unlayered rules beat layered ones whatever the
 * source order. So `className="absolute inset-0"` is silently overridden, the
 * container collapses to height 0, and the map paints nothing. Inline wins
 * over both.
 */
export const MAP_CONTAINER_STYLE = { position: "absolute", inset: 0 } as const;
