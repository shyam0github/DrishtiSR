import { useEffect, useState } from "react";
import { getSamples, MOCK_MODE, upscale, type ImageResult } from "../api/client";

export type ExampleState =
  | { status: "loading" }
  | { status: "done"; result: ImageResult }
  /** No example exists: MOCK_MODE (no model), or the server lists no sample with a 2.5 m reference. */
  | { status: "unavailable"; reason: string }
  | { status: "error"; error: string };

/**
 * One server sample, super-resolved once per page load and shared by every
 * novelty page, so a page is never empty before the user uploads. It never
 * writes ResultContext: the example must not masquerade as "your image".
 */
let examplePromise: Promise<ExampleState> | null = null;

async function loadExample(): Promise<ExampleState> {
  if (MOCK_MODE) return { status: "unavailable", reason: "Mock mode: no model runs, so there is no real example to show." };
  const samples = await getSamples();
  const pick = samples.find((s) => s.has_gt) ?? samples[0];
  if (!pick) return { status: "unavailable", reason: "The server lists no samples (app/samples is empty)." };
  return { status: "done", result: await upscale({ sampleId: pick.id }, pick.label) };
}

/** The example is only fetched when `enabled` (i.e. no upload is in context). */
export function useExampleResult(enabled: boolean): ExampleState {
  const [state, setState] = useState<ExampleState>({ status: "loading" });
  useEffect(() => {
    if (!enabled) return;
    let live = true;
    examplePromise ??= loadExample().catch((exc: unknown) => {
      console.error("example upscale failed:", exc);
      examplePromise = null; // allow a retry on the next visit rather than caching the failure
      return { status: "error", error: exc instanceof Error ? exc.message : String(exc) } as ExampleState;
    });
    void examplePromise.then((s) => live && setState(s));
    return () => {
      live = false;
    };
  }, [enabled]);
  return state;
}
