export const AI_BADGE_TEXT = "AI-RECONSTRUCTED — NOT MEASURED DATA";

/**
 * Provenance badge for every view that shows super-resolved pixels.
 *
 * Deliberately takes no props. Nothing can hide it, reword it or give it a
 * close button, and it ignores the pointer so it never blocks the map.
 */
export function AiBadge() {
  return (
    <div
      data-testid="ai-badge"
      role="note"
      className="pointer-events-none absolute right-2 top-2 z-20 select-none rounded bg-amber-300 px-2 py-1 text-xs font-bold tracking-wide text-black shadow ring-2 ring-black"
    >
      {AI_BADGE_TEXT}
    </div>
  );
}
