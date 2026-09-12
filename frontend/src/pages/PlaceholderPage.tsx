import { Badge, SectionHeader } from "../components/ui";
import type { NavRoute } from "../routes";

/** Stand-in for a route whose content lands in a later session. */
export function PlaceholderPage({ route }: { route: NavRoute }) {
  return (
    <section className="relative overflow-hidden" data-testid="placeholder-page">
      <div aria-hidden className="bg-graticule absolute inset-0" />
      <div className="page-container relative flex min-h-[60dvh] flex-col justify-center gap-8 py-section">
        <SectionHeader level="h1" eyebrow={route.index ? `Novelty ${route.index}` : route.label} title={route.title} />
        <div>
          <Badge variant="pending">Content pending</Badge>
        </div>
      </div>
    </section>
  );
}
