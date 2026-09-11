/**
 * Shared MapLibre plumbing: the basemap style and idempotent layer upserts.
 * Used by both maps of the swipe view, so the two always agree.
 */
import type { FeatureCollection } from "geojson";
import type { GeoJSONSource, ImageSource, Map as MlMap, StyleSpecification } from "maplibre-gl";
import { APP_CONFIG } from "../config";
import type { Bbox } from "../geo/aoi";

export const AOI_SOURCE_ID = "aoi";
const AOI_FILL_ID = "aoi-fill";
const AOI_LINE_ID = "aoi-line";

export function basemapStyle(): StyleSpecification {
  const b = APP_CONFIG.map.basemap;
  return {
    version: 8,
    sources: {
      basemap: { type: "raster", tiles: b.tiles, tileSize: b.tileSize, maxzoom: b.maxZoom, attribution: b.attribution },
    },
    layers: [{ id: "basemap", type: "raster", source: "basemap" }],
  };
}

/** MapLibre image-source corner order: top-left, top-right, bottom-right, bottom-left. */
function imageCoordinates([w, s, e, n]: Bbox): [[number, number], [number, number], [number, number], [number, number]] {
  return [[w, n], [e, n], [e, s], [w, s]];
}

/**
 * Add or update an image-backed raster layer, kept below the AOI outline.
 *
 * Resampling is `nearest`: a 10 m pixel is drawn as a 10 m square. Linear
 * resampling would smooth the input and make it look finer than it is, which
 * shrinks the very difference the swipe exists to show.
 */
export function upsertImageLayer(map: MlMap, id: string, url: string, footprint: Bbox, opacity = 1): void {
  const coordinates = imageCoordinates(footprint);
  const existing = map.getSource(id) as ImageSource | undefined;
  if (existing) {
    existing.updateImage({ url, coordinates });
    return;
  }
  map.addSource(id, { type: "image", url, coordinates });
  map.addLayer(
    {
      id,
      type: "raster",
      source: id,
      paint: { "raster-opacity": opacity, "raster-resampling": "nearest", "raster-fade-duration": 0 },
    },
    map.getLayer(AOI_FILL_ID) ? AOI_FILL_ID : undefined,
  );
}

export function removeLayer(map: MlMap, id: string): void {
  if (map.getLayer(id)) map.removeLayer(id);
  if (map.getSource(id)) map.removeSource(id);
}

/** Draw (or clear, with null) the AOI rectangle. Red when over the tile budget. */
export function setAoiLayer(map: MlMap, bbox: Bbox | null, overBudget: boolean): void {
  const data: FeatureCollection = {
    type: "FeatureCollection",
    features: bbox
      ? [
          {
            type: "Feature",
            properties: {},
            geometry: {
              type: "Polygon",
              coordinates: [[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]],
            },
          },
        ]
      : [],
  };
  const colour = overBudget ? "#dc2626" : "#2563eb";
  const source = map.getSource(AOI_SOURCE_ID) as GeoJSONSource | undefined;
  if (source) {
    source.setData(data);
  } else {
    map.addSource(AOI_SOURCE_ID, { type: "geojson", data });
    map.addLayer({ id: AOI_FILL_ID, type: "fill", source: AOI_SOURCE_ID, paint: { "fill-color": colour, "fill-opacity": 0.08 } });
    map.addLayer({ id: AOI_LINE_ID, type: "line", source: AOI_SOURCE_ID, paint: { "line-color": colour, "line-width": 2 } });
  }
  map.setPaintProperty(AOI_FILL_ID, "fill-color", colour);
  map.setPaintProperty(AOI_LINE_ID, "line-color", colour);
}
