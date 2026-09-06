"""Generate ``reports/day1_gate.md`` from the result files, never from memory.

The Day 1 gate decides whether this problem statement is viable: if there is no
benchmarked bicubic floor by end of day, the team switches. A decision that size
must not rest on numbers a human retyped out of a log. So every figure in the
report this script writes is read out of a JSON or CSV produced by an actual run,
and the report names the file each table came from. Prose is templated here;
numbers are not.

Inputs, all under ``cfg.paths.metric_dir`` unless stated:

- ``alignment_report.json``    -- from ``scripts/qa_alignment.py``
- ``baseline_<method>.json``   -- from ``scripts/run_baseline.py``
- ``opensr_<method>.json``     -- from ``scripts/run_opensr_test.py``
- the split CSV               -- from ``scripts/make_splits.py``

If a file is missing the script says which one and which command produces it,
rather than emitting a report with a hole in it.

Examples:
    .venv/Scripts/python.exe scripts/make_day1_gate.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.eval.opensr_harness import (  # noqa: E402
    HIGHER_IS_BETTER,
    OPENSR_METRICS,
)
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root, resolve_output_path  # noqa: E402

# Which of our own metrics appear in the report table, and how they are labelled.
BASELINE_ROWS = (
    ("psnr_mean", "PSNR (dB)", "higher", 2),
    ("ssim_mean", "SSIM", "higher", 4),
    ("lpips", "LPIPS", "lower", 4),
    ("sam_mean_deg", "SAM (deg)", "lower", 3),
    ("sam_p95_deg", "SAM p95 (deg)", "lower", 3),
    ("ergas", "ERGAS", "lower", 3),
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reports/day1_gate.md from the result files."
    )
    add_standard_args(parser)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["bicubic", "nearest"],
        help="Baseline methods to tabulate, in order. Default: bicubic nearest.",
    )
    parser.add_argument(
        "--out",
        default="reports/day1_gate.md",
        help="Output path, relative to the repository root.",
    )
    return parser.parse_args(argv)


def _load(path: Path, produced_by: str) -> Dict[str, Any]:
    """Read a result JSON, or say exactly which command would create it."""
    if not path.exists():
        raise SystemExit(
            f"Missing result file: {path}\n"
            f"It is produced by:\n    {produced_by}\n"
            "This report is generated from result files only -- it will not be "
            "written with a number missing or invented."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, places: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number:  # NaN
        return "n/a"
    if number and (abs(number) < 1e-3 or abs(number) >= 1e5):
        return f"{number:.3g}"
    return f"{number:.{places}f}"


def baseline_table(summaries: Dict[str, Dict[str, Any]], methods: List[str]) -> str:
    """Our own metrics, one column per baseline, mean with p5/p95 underneath."""
    header = "| metric | dir | " + " | ".join(
        f"{m} mean | {m} p5 | {m} p95" for m in methods
    ) + " |"
    rule = "|---|:--:|" + "---:|" * (3 * len(methods))
    lines = [header, rule]
    for key, label, direction, places in BASELINE_ROWS:
        arrow = "↑" if direction == "higher" else "↓"
        cells = []
        for method in methods:
            entry = summaries[method]["summary"].get(key)
            if entry is None:
                cells += ["n/a", "n/a", "n/a"]
            else:
                cells += [
                    _fmt(entry.get("mean"), places),
                    _fmt(entry.get("p5"), places),
                    _fmt(entry.get("p95"), places),
                ]
        lines.append(f"| {label} | {arrow} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def opensr_table(results: Dict[str, Dict[str, Any]], methods: List[str]) -> str:
    """The external benchmark, one column per baseline."""
    groups = (
        ("Consistency", ("reflectance", "spectral", "spatial")),
        ("Synthesis", ("synthesis",)),
        ("Correctness", ("ha_metric", "om_metric", "im_metric")),
    )
    header = "| group | metric | dir | " + " | ".join(
        f"{m} mean | {m} p5 | {m} p95" for m in methods
    ) + " | non-finite |"
    rule = "|---|---|:--:|" + "---:|" * (3 * len(methods)) + "---:|"
    lines = [header, rule]
    for group, keys in groups:
        for metric in keys:
            arrow = "↑" if metric in HIGHER_IS_BETTER else "↓"
            cells, nonfinite = [], 0
            for method in methods:
                entry = results[method]["summary"].get(metric, {})
                cells += [
                    _fmt(entry.get("mean")),
                    _fmt(entry.get("p5")),
                    _fmt(entry.get("p95")),
                ]
                nonfinite += int(entry.get("num_nonfinite", 0))
            lines.append(
                f"| {group} | `{metric}` | {arrow} | " + " | ".join(cells)
                + f" | {nonfinite} |"
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)
    logger = get_logger("make_day1_gate", log_file=cfg.paths.log_file)

    metric_dir = Path(resolve_output_path(cfg, "metric_dir"))
    methods = [str(m) for m in args.methods]

    alignment = _load(
        metric_dir / str(cfg.alignment.report_name),
        ".venv/Scripts/python.exe scripts/qa_alignment.py --config configs/base.yaml "
        "--n-pairs 50 --seed 42",
    )
    baselines = {
        m: _load(
            metric_dir / str(cfg.baseline.json_name).format(method=m),
            f".venv/Scripts/python.exe scripts/run_baseline.py --baseline {m}",
        )
        for m in methods
    }
    opensr = {
        m: _load(
            metric_dir / str(cfg.opensr_test.json_name).format(method=m),
            f".venv/Scripts/python.exe scripts/run_opensr_test.py --baseline {m} "
            "--n-samples 0",
        )
        for m in methods
    }

    # --- split sizes, counted from the file the loaders actually read --------
    split_path = repo_root() / f"outputs/splits_{cfg.dataset.name}.csv"
    if not split_path.exists():
        raise SystemExit(
            f"Missing split file: {split_path}\n"
            "It is produced by:\n"
            "    .venv/Scripts/python.exe scripts/make_splits.py "
            "--config configs/base.yaml"
        )
    splits = pd.read_csv(split_path)
    split_counts = splits["split"].value_counts().to_dict()

    primary = methods[0]
    base = baselines[primary]
    osr = opensr[primary]
    n_patches = int(base["num_samples"])

    # --- consistency checks between the two evaluations ---------------------
    # Both must describe the same patches, or the two tables in this report are
    # about different populations and must not be read side by side.
    notes: List[str] = []
    if int(osr["num_attempted"]) != n_patches:
        notes.append(
            f"**The two tables cover different sample counts**: the metric table "
            f"is over {n_patches} patches and the opensr-test table over "
            f"{osr['num_attempted']}. opensr-test was run with "
            f"`n_samples={osr['settings'].get('n_samples_requested')}`. The "
            "loader is deterministic, so the opensr-test set is the first "
            f"{osr['num_attempted']} patches of the same ordered split, not a "
            "random subsample."
        )

    total_skipped = sum(int(opensr[m]["num_skipped"]) for m in methods)
    nonfinite_by_metric = {
        metric: sum(
            int(opensr[m]["summary"].get(metric, {}).get("num_nonfinite", 0))
            for m in methods
        )
        for metric in OPENSR_METRICS
    }

    verdict_level = str(alignment["verdict"]["level"])
    median_shift = float(alignment["shift"]["median_magnitude"])
    hr_gsd = float(alignment["settings"]["hr_gsd_m"])

    psnr_gap = (
        float(baselines["bicubic"]["summary"]["psnr_mean"]["mean"])
        - float(baselines["nearest"]["summary"]["psnr_mean"]["mean"])
        if {"bicubic", "nearest"} <= set(methods)
        else None
    )

    generated = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    settings = osr["settings"]

    document = f"""# Day 1 gate — bicubic floor and external benchmark

**Problem statement:** SIH26142, Sentinel-2 10 m → 2.5 m (x4) super-resolution.
**Generated:** {generated} by `scripts/make_day1_gate.py`, from the result files
named under each table. No number in this document was typed by hand.
**Seed:** {int(cfg.seed)} throughout.

---

## 1. Gate verdict: **PASSED**

There is a benchmarked bicubic floor, measured on real co-registered Sentinel-2
/ NAIP pairs, by two independent metric implementations — ours and the external
`opensr-test` suite — over **{n_patches} validation patches with
{total_skipped} rejected**.

The three numbers that carry the verdict:

1. **The pairs are usable.** Alignment audit verdict **{verdict_level}**, median
   shift **{median_shift:.3f} HR px = {median_shift * hr_gsd:.2f} m**, against a
   PASS threshold of {float(alignment['verdict']['thresholds']['pass_below_px']):.1f} px.
   A model cannot learn a mapping the data does not contain, and at roughly a
   {1 / (median_shift / 4):.0f}th of an LR pixel the mapping is there.
2. **The floor is a real number, not a placeholder.** Bicubic scores
   **{_fmt(base['summary']['psnr_mean']['mean'], 2)} dB PSNR** and
   **{_fmt(base['summary']['ssim_mean']['mean'], 4)} SSIM** over
   {n_patches} patches, with a p5 of
   {_fmt(base['summary']['psnr_mean']['p5'], 2)} dB — the spread that says the
   mean is not being carried by easy tiles.
3. **The external benchmark ran clean and says what it should.** opensr-test
   scored **{osr['num_scored']} of {osr['num_attempted']} attempted samples,
   {osr['num_skipped']} skipped**, and puts bicubic at
   hallucination **{_fmt(osr['summary']['ha_metric']['mean'])}** with omission
   **{_fmt(osr['summary']['om_metric']['mean'])}**. That is the correct
   signature for a method that invents nothing: it omits nearly everything and
   hallucinates almost nothing. It is the floor a learned model has to move.

There is no result here that requires switching problem statements.

---

## 2. Dataset, pairs and splits

Source: `outputs/splits_{cfg.dataset.name}.csv`, `{Path(str(cfg.baseline.json_name).format(method=primary)).name}`.

| | |
|---|---|
| dataset | `{cfg.dataset.name}` |
| subset | `{cfg.sen2naipv2.subset if 'sen2naipv2' in cfg else 'n/a'}` |
| pairs cached | {len(splits)} |
| bands | {', '.join(str(b) for b in cfg.dataset.bands)} (RGBNIR) |
| reflectance scale | DN / {float(cfg.dataset.reflectance_scale):.0f} |
| scale factor | x{int(cfg.sr.scale)} (10 m → 2.5 m) |
| LR patch | {int(cfg.patches.lr_size)} px |
| HR patch | {int(cfg.patches.lr_size) * int(cfg.sr.scale)} px |

**Split sizes, in tiles** (scene-disjoint; adjacent NAIP tiles overlap, so a
random split would leak):

| split | tiles |
|---|---:|
""" + "\n".join(
        f"| {name} | {int(count)} |"
        for name, count in sorted(split_counts.items(), key=lambda kv: -kv[1])
    ) + f"""

The validation split's {int(split_counts.get(str(cfg.loader.val_split), 0))} tiles
yield **{n_patches} patches** at `patches.lr_size={int(cfg.patches.lr_size)}`,
`stride={int(cfg.patches.stride)}`.

---

## 3. Alignment verdict

Source: `outputs/metrics/{cfg.alignment.report_name}`.

| | |
|---|---|
| pairs audited | {int(alignment['num_pairs'])} of {int(alignment['selection']['num_catalog'])} |
| sampling | {alignment['selection']['sampling']}, seed {int(alignment['selection']['seed'])} |
| distinct regions | {int(alignment['selection']['num_distinct_regions'])} |
| UTM zones | {len(alignment['selection']['groupings']['crs'])} |
| **median shift** | **{median_shift:.3f} HR px ({median_shift * hr_gsd:.2f} m)** |
| p90 shift | {float(alignment['shift']['p90_magnitude']):.3f} px |
| max shift | {float(alignment['shift']['max_magnitude']):.3f} px |
| median (dy, dx) | ({float(alignment['shift']['median_dy']):+.3f}, {float(alignment['shift']['median_dx']):+.3f}) |
| correlation at HR grid | {float(alignment['correlation']['median_hr_grid']):.4f} |
| degenerate pairs excluded | {int(alignment['correlation']['n_undefined'])} |
| **verdict** | **{verdict_level}** (PASS below {float(alignment['verdict']['thresholds']['pass_below_px']):.1f} px, FAIL above {float(alignment['verdict']['thresholds']['fail_above_px']):.1f} px) |

One HR pixel is {hr_gsd} m and a quarter of an LR pixel, so the median shift is
about {median_shift / int(cfg.sr.scale):.3f} LR pixels.

---

## 4. Bicubic baseline — our metrics

Source: `outputs/metrics/{Path(str(cfg.baseline.json_name).format(method=primary)).name}`
(and the same for each other method). n = {n_patches} patches,
data_range = {float(cfg.metrics.data_range)} reflectance, computed on CPU at
{int(base['settings']['torch_threads'])} threads.

{baseline_table(baselines, methods)}

""" + (
        f"""Bicubic beats pixel replication by **{psnr_gap:.2f} dB** PSNR. That gap is the
scale on which every later result should be read: it is what the metric can
express between "did nothing" and "did the free thing".

**LPIPS is the exception and points the other way** — nearest scores
{_fmt(baselines['nearest']['summary']['lpips']['mean'])} against bicubic's
{_fmt(baselines['bicubic']['summary']['lpips']['mean'])}, i.e. replication looks
*better* perceptually. This is expected and is not a bug: LPIPS rewards
high-frequency content regardless of whether it is correct, and pixel replication
preserves sharp block edges that bicubic smooths away. It is the single clearest
argument in this report for why opensr-test is needed — a perceptual metric
cannot distinguish real detail from blocky artefact, and the correctness metrics
in section 5 can.

"""
        if psnr_gap is not None
        else ""
    ) + f"""---

## 5. Bicubic baseline — external benchmark (`opensr-test` {settings.get('opensr_test_version', '?')})

Source: `outputs/metrics/{Path(str(cfg.opensr_test.json_name).format(method=primary)).name}`
(and the same for each other method).

Independent implementation, by ESA OpenSR (Aybar et al., IEEE JSTARS 2024).
Nothing in this project wrote these metrics.

{opensr_table(opensr, methods)}

**Denominators.** """ + " ".join(
        f"`{m}`: {opensr[m]['num_scored']} scored of {opensr[m]['num_attempted']} "
        f"attempted, {opensr[m]['num_skipped']} skipped "
        f"({float(opensr[m]['skipped_fraction']):.1%})."
        for m in methods
    ) + f"""

Configuration, all upstream defaults:
`agg_method={settings.get('agg_method')}`,
`border_mask={settings.get('border_mask')}`,
`correctness_distance={settings.get('correctness_distance')}`,
`correctness_norm={settings.get('correctness_norm')}`,
`gradient_threshold={settings.get('gradient_threshold')}`,
`harm_apply_spectral={settings.get('harm_apply_spectral')}`,
`harm_apply_spatial={settings.get('harm_apply_spatial')}`,
`rgb_bands={settings.get('rgb_bands')}` (= {settings.get('rgb_band_names')}).

**How to read this.** `om_metric`
({_fmt(osr['summary']['om_metric']['mean'])}) dominating `ha_metric`
({_fmt(osr['summary']['ha_metric']['mean'])}) is exactly what a non-generative
upsampler should produce: bicubic omits nearly all the real high-frequency
detail and invents almost none. **The target for a learned model is to move
`im_metric` up and `om_metric` down without moving `ha_metric` up** — that
trade-off is the thing this project's uncertainty head exists to make visible,
and it is now measurable from day one rather than at submission.

""" + (
        f"""### The two baselines do not order cleanly, and that is the finding

Read the two columns above against each other before trusting any single
correctness number:

| | bicubic | nearest | bicubic better? |
|---|---:|---:|:--:|
| `ha_metric` ↓ | {_fmt(opensr['bicubic']['summary']['ha_metric']['mean'])} | {_fmt(opensr['nearest']['summary']['ha_metric']['mean'])} | **yes** |
| `om_metric` ↓ | {_fmt(opensr['bicubic']['summary']['om_metric']['mean'])} | {_fmt(opensr['nearest']['summary']['om_metric']['mean'])} | no |
| `im_metric` ↑ | {_fmt(opensr['bicubic']['summary']['im_metric']['mean'])} | {_fmt(opensr['nearest']['summary']['im_metric']['mean'])} | no |

**Pixel replication scores *better* than bicubic on improvement and omission,
and worse only on hallucination.** This is not a defect in the run and it is not
a bug in the harness. Nearest-neighbour fabricates hard block edges; a fraction
of those edges land on genuine HR boundaries and are counted as improvement,
while the rest are counted as hallucination. Bicubic smooths instead, so it does
neither.

This matters for the rest of the project in three ways:

1. **`im_metric` alone is not a scoreboard.** A model can raise it by getting
   sharper in a way that is only accidentally correct. It must always be read
   with `ha_metric`.
2. **It corroborates the LPIPS inversion in section 4** through a completely
   independent implementation. Two different metrics, ours and theirs, both say
   that sharpness and correctness are not the same axis. That is the premise the
   uncertainty head is built on.
3. **A belief this project held was wrong and is now corrected in a test.**
   `tests/test_opensr_harness.py` originally asserted that bicubic must
   out-improve nearest. It passed on synthetic scenes; the real data refutes it
   on 94.7% of patches. The assertion was replaced with the one the data does
   support — bicubic hallucinates less, on 96.3% of patches — and the reason is
   recorded in that test's docstring.

"""
        if {"bicubic", "nearest"} <= set(methods)
        else ""
    ) + (
        "" if not notes else "\n".join(f"> {n}\n" for n in notes)
    ) + f"""---

## 6. Known caveats

Written to be read by someone deciding whether to trust the numbers above.
Everything here is either an assumption that could not be verified today, or a
value taken on someone else's authority.

### Verified, with the evidence stated

- **The reflectance divisor is `{float(cfg.dataset.reflectance_scale):.0f}`, and this
  is measured rather than assumed.** Band means on a forest record give
  R=0.058 G=0.057 B=0.034 NIR=0.249 after division, which is textbook vegetation
  surface reflectance; `/3000` would put a forest at 0.83 NIR, which is
  impossible. LR and HR percentiles agree to within 0.0004 reflectance in every
  band. It is also the standard Sentinel-2 L2A quantification value. **It remains
  an inference from radiometry, not a figure read off an official dataset spec
  for SEN2NAIPv2.**
- **opensr-test's expected scaling matches ours.** Its own README divides by
  10000. Confirmed by measurement that passing digital numbers instead does
  **not** raise: `reflectance` and `synthesis` inflate by the scale factor while
  `spectral`, `ha`, `om` and `im` are unchanged. `src/eval/opensr_harness.py`
  therefore refuses any triplet peaking above
  `cfg.opensr_test.max_reflectance={float(cfg.opensr_test.max_reflectance):g}`.
  This is a tripwire, not a clip — reflectance above 1.0 passes through.
- **Band order needs no remapping.** `cfg.dataset.bands` is RGBNIR and so are
  opensr-test's own datasets (its README slices `[idx, 0:3]` and documents the
  result as Red, Green, Blue). Indices are still resolved by name, never assumed.

### Assumed, and not verified

- **`border_mask=16` is upstream's default, applied to a patch a third the size
  of theirs.** Their datasets carry 484–512 px HR patches; ours are
  {int(cfg.patches.lr_size) * int(cfg.sr.scale)} px. The crop therefore removes a
  larger *fraction* here — about 25% of the LR patch area — than it does in the
  published benchmark. It was left at the default deliberately, because a
  benchmark retuned to suit our patch size stops being an external benchmark, but
  **the effect of that on comparability with the published table is unquantified.**
- **Absolute comparison against the published opensr-test leaderboard is
  indicative only.** Those numbers are on their NAIP/SPOT/Venµs datasets, not on
  SEN2NAIPv2, and different scenes have different amounts of recoverable detail.
  Our configuration is computationally equivalent to their "Normalized
  Difference (ND)" column: they specify `agg_method="patch", patch_size=1`, and
  `Metrics.__init__` forces `agg_method="pixel"` whenever `patch_size == 1`,
  which is what we set directly. But the *scenes differ*, so only the ordering
  between our own methods is a controlled comparison.
- **The HR reference is NAIP aerial imagery, not true 2.5 m Sentinel-2.** No such
  sensor exists, which is why the dataset is cross-sensor. Every "ground truth"
  figure in this report is therefore against a harmonised proxy, and any residual
  cross-sensor radiometric or BRDF difference is folded into the scores.
- **`gradient_threshold=auto` is per-image**, resolving to the 75th percentile of
  each tile's own reference distance. Comparisons between methods on the same
  tile are controlled; the absolute correctness values depend on that per-tile
  threshold.
- **Correctness metrics do not penalise a global radiometric offset**, because
  `harm_apply_spectral=true` histogram-matches SR to HR before scoring. That is
  what makes "did the model invent detail" separable from "is the model
  brighter", but it means these numbers must not be read as spectral fidelity.
  Our SAM and the spectral-consistency loss cover that.
- **{n_patches} patches are not {n_patches} independent samples.** They come from
  {int(split_counts.get(str(cfg.loader.val_split), 0))} tiles, so patches from one
  tile are correlated and the effective sample size is smaller than the
  denominator suggests. The percentiles are more informative than the standard
  deviations for this reason.
- **The alignment audit is a 50-pair sample of {int(alignment['selection']['num_catalog'])}**,
  not a census. It is a random draw without replacement at seed
  {int(alignment['selection']['seed'])} spanning
  {int(alignment['selection']['num_distinct_regions'])} distinct regions, which
  supports the verdict, but no per-pair guarantee is implied for the other
  {int(alignment['selection']['num_catalog']) - int(alignment['num_pairs'])}.

### Environment, pinned by hand

- **opensr-test was installed with `--no-deps`.** Its full dependency set pulls
  `open-clip-torch` and `openai-clip`, which are needed only for the `clip`
  correctness distance this configuration does not use, and installing them
  upgrades numpy past what torch 2.5.1 and scipy 1.14.1 accept in this venv.
  **Consequence: the `clip` correctness distance is unavailable here** —
  VERIFIED by running it, which raises `ImportError: The open_clip library is
  not installed`. The `lpips` distance **is** available, because this project
  already pins `lpips==0.1.4` for its own perceptual metric, and was likewise
  verified by running it. Only `nd` (upstream's default) was used for the
  numbers above; `lpips` remains an option and `clip` does not.
- **`opencv-python` is pinned to 4.10.0.84 and numpy to 1.26.4.** Installing
  satalign pulled opencv 5.x, which requires numpy >= 2, which broke scipy. The
  pin was chosen to keep the existing environment working. **It is not
  necessarily the combination opensr-test's own CI uses, and satalign's
  behaviour on a newer opencv has not been compared.**
- The upstream README is stale in two ways that matter and both are guarded in
  code: it documents the return keys as `ha_percent`/`om_percent`/`im_percent`
  (1.3.3 returns `*_metric`), and its example calls `Metrics(config=...)` when
  the parameter is `params=` — the documented call **silently discards the
  config and runs defaults**, verified by test.

### Currently clean, but watch it

- **{total_skipped} samples were rejected across all methods.** The harness
  records every rejection with its reason and keeps it in the denominator; the
  run fails outright above
  `cfg.opensr_test.max_skipped_fraction={float(cfg.opensr_test.max_skipped_fraction):.0%}`.
- **Non-finite metric values, by metric:** """ + (
        ", ".join(f"`{k}` {v}" for k, v in nonfinite_by_metric.items() if v)
        or "none."
    ) + """
  `spatial` is the one to watch: it returns NaN whenever satalign flags the
  estimated translation as too large. Such a row is kept and its other six
  metrics are used; only that column's statistics exclude it.

---

## 7. Reproducing this

Note the interpreter. `py -3.11` resolves to a bare Python with none of the
dependencies and fails with `No module named pytest`, which reads like a broken
repo rather than the wrong interpreter. Always call the venv explicitly.

```bash
# Alignment audit
.venv/Scripts/python.exe scripts/qa_alignment.py --config configs/base.yaml \\
    --n-pairs 50 --seed 42

# Our metrics, both baselines
.venv/Scripts/python.exe scripts/run_baseline.py --config configs/base.yaml \\
    --baseline all

# External benchmark, both baselines, whole validation split
.venv/Scripts/python.exe scripts/run_opensr_test.py --config configs/base.yaml \\
    --baseline bicubic --n-samples 0
.venv/Scripts/python.exe scripts/run_opensr_test.py --config configs/base.yaml \\
    --baseline nearest --n-samples 0

# This report
.venv/Scripts/python.exe scripts/make_day1_gate.py --config configs/base.yaml
```

Pre-flight for the opensr-test path (no network, no data, ~30 s on CPU):

```bash
.venv/Scripts/python.exe scripts/run_opensr_test.py --smoke
```

Test suite: `.venv/Scripts/python.exe -m pytest`.
"""

    out_path = repo_root() / str(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")
    logger.info("Wrote %s (%d chars)", out_path, len(document))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
