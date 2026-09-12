import { useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

export interface Tab {
  id: string;
  label: ReactNode;
  content: ReactNode;
}

/** WAI-ARIA tabs: arrow/Home/End keys, roving tabindex, sliding indicator. */
export function TabbedPanel({ tabs, defaultTab, onChange }: { tabs: Tab[]; defaultTab?: string; onChange?: (id: string) => void }) {
  if (tabs.length === 0) throw new Error("TabbedPanel: tabs must not be empty");
  const uid = useId();
  const [active, setActive] = useState(defaultTab ?? tabs[0]!.id);
  const listRef = useRef<HTMLDivElement>(null);
  const [indicator, setIndicator] = useState({ left: 0, width: 0 });

  useLayoutEffect(() => {
    const btn = listRef.current?.querySelector<HTMLButtonElement>(`[data-tab="${CSS.escape(active)}"]`);
    if (btn) setIndicator({ left: btn.offsetLeft, width: btn.offsetWidth });
  }, [active, tabs]);

  const select = (id: string, focus = false) => {
    setActive(id);
    onChange?.(id);
    if (focus) listRef.current?.querySelector<HTMLButtonElement>(`[data-tab="${CSS.escape(id)}"]`)?.focus();
  };

  const onKey = (e: KeyboardEvent) => {
    const i = tabs.findIndex((t) => t.id === active);
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    select(tabs[(next + tabs.length) % tabs.length]!.id, true);
  };

  const current = tabs.find((t) => t.id === active) ?? tabs[0]!;
  return (
    <div className="flex flex-col gap-6">
      <div className="overflow-x-auto">
        <div ref={listRef} role="tablist" onKeyDown={onKey} className="relative inline-flex gap-1 rounded-md border border-line bg-ink-900 p-1">
          <span
            aria-hidden
            className="absolute top-1 bottom-1 rounded-sm bg-ink-800 shadow-[inset_0_0_0_1px_var(--color-line-strong)] transition-all duration-300 ease-out-expo"
            style={{ left: indicator.left, width: indicator.width }}
          />
          {tabs.map((t) => {
            const on = t.id === active;
            return (
              <button
                key={t.id}
                data-tab={t.id}
                role="tab"
                id={`${uid}-tab-${t.id}`}
                aria-selected={on}
                aria-controls={`${uid}-panel-${t.id}`}
                tabIndex={on ? 0 : -1}
                onClick={() => select(t.id)}
                className={`relative z-10 whitespace-nowrap rounded-sm px-4 py-1.5 text-caption font-medium transition-colors ${on ? "text-fg" : "text-fg-muted hover:text-fg"}`}
              >
                {t.label}
              </button>
            );
          })}
        </div>
      </div>
      <div
        key={current.id}
        role="tabpanel"
        id={`${uid}-panel-${current.id}`}
        aria-labelledby={`${uid}-tab-${current.id}`}
        tabIndex={0}
        className="animate-route-in"
      >
        {current.content}
      </div>
    </div>
  );
}
