import { useEffect, useState } from "react";
import { API_BASE, checkHealth, type HealthResult } from "../api/client";

/** "Backend: online/offline" from one GET /health on mount. No polling, no retries. */
export function BackendBadge() {
  const [health, setHealth] = useState<HealthResult | null>(null);

  useEffect(() => {
    let live = true;
    void checkHealth().then((h) => live && setHealth(h));
    return () => {
      live = false;
    };
  }, []);

  const state = health === null ? "checking" : health.online ? "online" : "offline";
  const style = {
    checking: "bg-slate-200 text-slate-700",
    online: "bg-emerald-600 text-white",
    offline: "bg-slate-500 text-white",
  }[state];
  const title = health && !health.online ? `${API_BASE}/health: ${health.reason}` : `${API_BASE}/health`;
  return (
    <span data-testid="backend-badge" data-state={state} title={title} className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${style}`}>
      Backend: {state === "checking" ? "checking…" : state}
    </span>
  );
}
