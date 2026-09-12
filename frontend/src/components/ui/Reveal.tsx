import { Children, type ElementType, type ReactNode } from "react";
import { MOTION, useInView } from "../../design/motion";

/**
 * Staggered fade + slide entrance. Each direct child is wrapped and delayed by
 * `stagger` ms times its index; the group reveals once, when it scrolls into view.
 * Honours prefers-reduced-motion via index.css.
 */
export function Reveal({
  children,
  as: Tag = "div",
  stagger = MOTION.staggerMs,
  className = "",
  itemClassName = "",
}: {
  children: ReactNode;
  as?: ElementType;
  stagger?: number;
  className?: string;
  itemClassName?: string;
}) {
  const [ref, inView] = useInView<HTMLElement>(0.12);
  return (
    <Tag ref={ref} data-revealed={inView} className={className}>
      {Children.toArray(children).map((child, i) => (
        <div key={i} className={`reveal-item ${itemClassName}`} style={{ ["--reveal-delay" as string]: `${i * stagger}ms` }}>
          {child}
        </div>
      ))}
    </Tag>
  );
}
