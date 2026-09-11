import type { RasterLayer } from "../api/client";

interface Props {
  layer: RasterLayer | null;
  visible: boolean;
  opacity: number;
  onVisibleChange: (visible: boolean) => void;
  onOpacityChange: (opacity: number) => void;
}

/** Show/hide and fade the per-pixel uncertainty layer drawn over the reconstruction. */
export function UncertaintyToggle({ layer, visible, opacity, onVisibleChange, onOpacityChange }: Props) {
  const disabled = layer === null;
  return (
    <section className="space-y-1" data-testid="uncertainty">
      <h2 className="font-semibold">Uncertainty overlay</h2>
      <label className="flex items-center gap-2">
        <input type="checkbox" data-testid="unc-toggle" checked={visible} disabled={disabled} onChange={(e) => onVisibleChange(e.target.checked)} />
        Show per-pixel σ over the reconstruction
      </label>
      <label className="flex items-center gap-2 text-xs">
        Opacity
        <input
          type="range"
          data-testid="unc-opacity"
          min={0}
          max={1}
          step={0.05}
          value={opacity}
          disabled={disabled || !visible}
          onChange={(e) => onOpacityChange(Number(e.target.value))}
          className="flex-1"
        />
        <span className="w-10 text-right tabular-nums">{Math.round(opacity * 100)}%</span>
      </label>
      <p className="text-xs text-slate-600">{layer ? layer.label : "No uncertainty layer loaded."}</p>
    </section>
  );
}
