/**
 * Line icons on the same 16-unit grid and 1.6 stroke as the nav, slider and
 * lightbox glyphs, so every icon on a page reads as one set.
 */
const PATHS = {
  spectral: "M2 13.5h12M3.5 11V7M6.5 11V4M9.5 11V6M12.5 11V2.5",
  uncertainty: "M8 2a6 6 0 1 0 0 12A6 6 0 0 0 8 2ZM8 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6ZM8 7.4v1.2",
  cpu: "M4.5 4.5h7v7h-7zM6.5 6.5h3v3h-3zM6 2v2.5M10 2v2.5M6 11.5V14M10 11.5V14M2 6h2.5M2 10h2.5M11.5 6H14M11.5 10H14",
  "arrow-right": "M3 8h10M9 4l4 4-4 4",
  "arrow-down": "M8 3v10M4 9l4 4 4-4",
  upload: "M8 10.5V2.5M4.5 6 8 2.5 11.5 6M2.5 10.5v2a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-2",
  expand: "M9.5 2.5h4v4M6.5 13.5h-4v-4M13.5 2.5 9 7M2.5 13.5 7 9",
  github:
    "M6 13.5c-3 .9-3-1.5-4.2-1.8M10 15v-2.3c0-.7.1-1.1-.4-1.5 1.8-.2 3.4-.9 3.4-3.8a3 3 0 0 0-.8-2 2.7 2.7 0 0 0-.1-2s-.6-.2-2.1.8a7 7 0 0 0-3.6 0C4.9 3.2 4.3 3.4 4.3 3.4a2.7 2.7 0 0 0-.1 2 3 3 0 0 0-.8 2c0 2.9 1.6 3.6 3.4 3.8-.5.4-.5.9-.4 1.5V15",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16, className = "" }: { name: IconName; size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden className={className}>
      <path d={PATHS[name]} />
    </svg>
  );
}
