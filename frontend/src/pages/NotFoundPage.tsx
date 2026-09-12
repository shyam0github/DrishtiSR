import { Link } from "react-router";
import { SectionHeader } from "../components/ui";

export function NotFoundPage() {
  return (
    <section className="page-container flex min-h-[60dvh] flex-col justify-center gap-8 py-section">
      <SectionHeader level="h1" eyebrow="404" title="No imagery at these coordinates." />
      <Link to="/" className="text-accent hover:text-accent-strong">
        ← Back to home
      </Link>
    </section>
  );
}
