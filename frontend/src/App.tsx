import { useCallback, useEffect, useMemo, useState } from "react";
import type { Map as MlMap } from "maplibre-gl";
import { DEMO_AOI, MOCK_MODE, runSr, type SrResponse } from "./api/client";
import { AoiPicker } from "./components/AoiPicker";
import { BackendBadge } from "./components/BackendBadge";
import { MapView } from "./components/MapView";
import { MetricsPanel } from "./components/MetricsPanel";
import { SwipeSlider } from "./components/SwipeSlider";
import { UncertaintyToggle } from "./components/UncertaintyToggle";
import type { Bbox } from "./geo/aoi";
import { removeLayer, upsertImageLayer } from "./map/layers";

const LR_LAYER_ID = "lr";
/** Screen padding when the camera fits to a new result, px. */
const FIT_PADDING_PX = 24;
const FIT_DURATION_MS = 800;
const DEFAULT_UNCERTAINTY_OPACITY = 0.6;

export default function App() {
  const [baseMap, setBaseMap] = useState<MlMap | null>(null);
  const [overlayMap, setOverlayMap] = useState<MlMap | null>(null);
  const [bbox, setBbox] = useState<Bbox | null>(DEMO_AOI);
  const [result, setResult] = useState<SrResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uncertaintyVisible, setUncertaintyVisible] = useState(false);
  const [uncertaintyOpacity, setUncertaintyOpacity] = useState(DEFAULT_UNCERTAINTY_OPACITY);

  const run = useCallback(async (aoi: Bbox) => {
    setRunning(true);
    setError(null);
    try {
      setResult(await runSr({ bbox: aoi }));
    } catch (exc) {
      console.error("runSr failed:", exc);
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setRunning(false);
    }
  }, []);

  // Open on the demo AOI, already super-resolved, so the swipe works on first load.
  useEffect(() => {
    if (baseMap) void run(DEMO_AOI);
  }, [baseMap, run]);

  // The input layer lives on the base map, under the swipe. Fit the camera to each result.
  useEffect(() => {
    if (!baseMap) return;
    if (!result) {
      removeLayer(baseMap, LR_LAYER_ID);
      return;
    }
    const lr = result.layers.lr;
    upsertImageLayer(baseMap, LR_LAYER_ID, lr.url, lr.footprint);
    const [w, s, e, n] = result.layers.sr.footprint;
    baseMap.fitBounds([[w, s], [e, n]], { padding: FIT_PADDING_PX, duration: FIT_DURATION_MS });
  }, [baseMap, result]);

  // Test hook for scripts/e2e_swipe.mjs; dev server only, never in a build.
  useEffect(() => {
    if (import.meta.env.DEV) (window as unknown as { __drishti?: object }).__drishti = { baseMap, overlayMap };
  }, [baseMap, overlayMap]);

  const renderOn = useMemo(() => [baseMap, overlayMap].filter((m): m is MlMap => m !== null), [baseMap, overlayMap]);

  return (
    <div className="flex h-dvh flex-col bg-white text-slate-900 md:flex-row">
      <main className="relative h-[60dvh] shrink-0 md:order-2 md:h-auto md:flex-1">
        <MapView onReady={setBaseMap}>
          {baseMap && (
            <SwipeSlider
              baseMap={baseMap}
              sr={result?.layers.sr ?? null}
              uncertainty={result?.layers.uncertainty ?? null}
              uncertaintyVisible={uncertaintyVisible}
              uncertaintyOpacity={uncertaintyOpacity}
              leftLabel={result?.layers.lr.label ?? null}
              onOverlayReady={setOverlayMap}
            />
          )}
        </MapView>
      </main>

      <aside className="min-h-0 flex-1 space-y-5 overflow-y-auto p-3 text-sm md:order-1 md:w-[30rem] md:flex-none md:border-r md:border-slate-200">
        <header className="space-y-1">
          <h1 className="text-base font-bold">DrishtiSR · Sentinel-2 10 m → 2.5 m</h1>
          <div className="flex flex-wrap items-center gap-2">
            <span data-testid="ai-label" className="rounded bg-amber-300 px-1.5 py-0.5 text-[11px] font-bold text-black ring-1 ring-black">
              AI-reconstructed imagery
            </span>
            <BackendBadge />
          </div>
          {MOCK_MODE && (
            <p data-testid="mock-banner" className="rounded bg-slate-800 px-2 py-1 text-xs text-white">
              MOCK MODE: fixtures and placeholder imagery. No model runs.
            </p>
          )}
        </header>

        {baseMap && (
          <AoiPicker
            map={baseMap}
            renderOn={renderOn}
            bbox={bbox}
            onChange={setBbox}
            onRun={(b) => void run(b)}
            running={running}
            runStatus={
              <>
                {error && (
                  <div role="alert" data-testid="sr-error" className="rounded border border-red-600 bg-red-50 p-2 text-xs text-red-800">
                    Super-resolution request failed: {error}
                  </div>
                )}
                {result && (
                  <p data-testid="provenance" data-request-id={result.request_id} className="text-xs text-slate-600">
                    {result.provenance.note}
                  </p>
                )}
              </>
            }
          />
        )}

        <section className="space-y-1" data-testid="compare">
          <h2 className="font-semibold">Compare</h2>
          <p className="text-xs text-slate-600">
            {result
              ? `Drag the divider on the map: left ${result.layers.lr.label}, right ${result.layers.sr.label}.`
              : "The swipe view appears on the map once a result is loaded."}
          </p>
        </section>

        <UncertaintyToggle
          layer={result?.layers.uncertainty ?? null}
          visible={uncertaintyVisible}
          opacity={uncertaintyOpacity}
          onVisibleChange={setUncertaintyVisible}
          onOpacityChange={setUncertaintyOpacity}
        />

        <MetricsPanel />
      </aside>
    </div>
  );
}
