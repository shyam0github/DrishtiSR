import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import { upscale, type ImageResult, type TtaChoice, type UpscaleInput } from "../api/client";

export type RunStatus = "idle" | "running" | "done" | "error";

/** What the user gave us: an uploaded file or a server sample. */
export type ResultInput = { kind: "file"; file: File; label: string } | { kind: "sample"; sampleId: string; label: string };

export interface ResultContextValue {
  input: ResultInput | null;
  /** The last completed result: images, SR output URLs and per-image metrics. */
  result: ImageResult | null;
  status: RunStatus;
  error: string | null;
  run: (input: UpscaleInput, label: string, tta?: TtaChoice) => Promise<void>;
  reset: () => void;
}

const ResultContext = createContext<ResultContextValue | null>(null);

/**
 * App-wide "this image's result". Wraps the router so Home, the novelty pages
 * and /compare all read the same upload without re-sending it. In memory only:
 * a reload starts empty, since the server's job files are the durable copy.
 */
export function ResultProvider({ children }: { children: ReactNode }) {
  const [input, setInput] = useState<ResultInput | null>(null);
  const [result, setResult] = useState<ImageResult | null>(null);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  // Only the latest request may write state: a slow earlier upload must not overwrite a newer one.
  const ticket = useRef(0);

  const run = useCallback(async (req: UpscaleInput, label: string, tta?: TtaChoice) => {
    const mine = ++ticket.current;
    setInput("file" in req ? { kind: "file", file: req.file, label } : { kind: "sample", sampleId: req.sampleId, label });
    setStatus("running");
    setError(null);
    try {
      const r = await upscale(req, label, tta);
      if (mine !== ticket.current) return;
      setResult(r);
      setStatus("done");
    } catch (exc) {
      if (mine !== ticket.current) return;
      console.error("upscale failed:", exc);
      setError(exc instanceof Error ? exc.message : String(exc));
      setStatus("error");
    }
  }, []);

  const reset = useCallback(() => {
    ticket.current++;
    setInput(null);
    setResult(null);
    setStatus("idle");
    setError(null);
  }, []);

  const value = useMemo(() => ({ input, result, status, error, run, reset }), [input, result, status, error, run, reset]);
  return <ResultContext.Provider value={value}>{children}</ResultContext.Provider>;
}

export function useResult(): ResultContextValue {
  const ctx = useContext(ResultContext);
  if (!ctx) throw new Error("useResult() needs a <ResultProvider> above it (src/main.tsx)");
  return ctx;
}
