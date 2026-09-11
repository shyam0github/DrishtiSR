# The irreducible spectral floor

**Day 3, 2026-09-09.** Measured before any spectral run is launched, because it
is the number that decides the weights.

Reproduce with:

```
.venv\Scripts\python.exe scripts/spectral_floor.py
```

Outputs: `outputs/metrics/spectral_floor.json`,
`outputs/metrics/spectral_floor_per_patch.csv`.

---

## The question

The spectral-consistency loss (`src/losses/spectral.py`) asks that degrading the
super-resolved output back to 10 m reproduces the Sentinel-2 input:

```
L_spec = lambda1 * L1(x_LR, D(y_SR)) + lambda2 * SAM(x_LR, D(y_SR))
```

A judge will ask what the achievable value of that is. If the answer were zero,
a run that plateaus at 0.006 would look like a failure.

It is not zero. Our LR is a real Sentinel-2 L2A acquisition; our HR is
NAIP-derived. Two sensors, two point-spread functions, two acquisition dates,
two atmospheric states. So `D(y_HR) != x_LR` even for the ground truth, and a
perfect super-resolver — one returning exactly the HR — still pays a
non-removable cost. That cost is a property of the dataset, not of any model,
and it is measurable: substitute the ground-truth HR for the SR output and run
the identical loss code over the whole validation split.

`cfg.degradation_audit` had already established the same fact from the other
direction — the best-fitting kernel reaches only ~44 dB median against the
stored LR, spread 13 dB across tiles, where a deterministic resampling would sit
near the 90.8 dB quantisation ceiling with under 0.1 dB of spread. This report
converts that finding into the units the loss is actually judged in.

## The measurement

All **1199 validation patches**, 300 scenes, `D = area` (exact 4×4 block mean),
`eps = 1e-8`, `cos_clamp = 1e-7`, `min_norm = 1e-6`.

| term | mean | std | p5 | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| `l1_spec` (reflectance) | **0.005733** | 0.003528 | 0.002025 | 0.004784 | 0.012790 | 0.028145 |
| `sam` (rad) | **0.022249** | 0.012147 | 0.008511 | 0.018971 | 0.048329 | 0.079000 |
| `sam` (deg) | **1.2748** | 0.6959 | 0.4877 | 1.0870 | 2.7691 | 4.5264 |

Two things worth reading off the spread. The floor is not one number: the p95
patch costs 2.2× the median in L1 and 2.5× in angle, so a run whose mean lands
at the floor is still far above it on some tiles and below it on others. And the
1.27° mean angle sits just under the 2.09° that bicubic scores against the HR
(`reports/day2_runA.json`) — the cross-sensor disagreement is of the same order
as the super-resolution error itself, which is precisely why it cannot be
ignored.

## What it means for the lambdas

Reference reconstruction loss: **L1(sr, hr) = 0.008784**, the median training L1
over Run A's final 10k iterations. (Run A was 64 features without augmentation,
so this sets the scale of the comparison rather than predicting Day 3's exact
converged value.)

At the candidate weights in `cfg.loss.spectral`:

| term | weighted floor | as % of L1(sr, hr) |
|---|---:|---:|
| `lambda1 = 0.5` × 0.005733 | 0.002867 | 32.6% |
| `lambda2 = 0.1` × 0.022249 | 0.002225 | 25.3% |
| **together** | **0.005092** | **58%** |

So at 0.5 / 0.1, **37% of the total objective at the floor is a constant no
model can remove**, and the spectral gradient is roughly half the magnitude of
the reconstruction gradient.

That matters because of what the spectral term is minimised by. **Taken alone,
it is minimised by an output that averages exactly to the LR** — which a blurry
bicubic-like upsample already does, scoring near 0, *far below* the 0.005733 an
HR-faithful output pays. The term does not merely regularise; at sufficient
weight it pulls towards doing no super-resolution at all, buying agreement with
the Sentinel-2 PSF at the price of the high-frequency detail the task exists to
produce.

**The floor is a reference point, not a lower bound.** Going below it is a
warning sign for over-smoothing, not an achievement. An output that averages to
the LR more closely than the ground truth does is closer to the Sentinel-2 PSF
than the real HR, and bicubic, which adds no detail at all, scores 0.21× the
floor. **Run B (B1) lands at L1_spec 0.001396 on the full validation split,
0.24× the floor**, next to bicubic's 0.21× (`reports/day3_results.md`). The
control A2 is also below it, at 0.48×, with the spectral term at zero, so an
L1-trained network sits below the floor by construction.

That is the reading the numbers support. The original version of this section
(below, struck through in substance) predated them.

**Recommendation (as written 2026-09-09, before any run).** Treat 0.5 / 0.1 as
the upper bracket, not a settled choice, and watch `val_l1_spec` against
0.005733 from the first validation onward:

- lands **near 0.0057** → the model has extracted everything the constraint
  contains. This is the claim we want, and it is now falsifiable.
- lands **below 0.0057** → ~~the spectral term has overridden the
  reconstruction L1~~. **Revised 2026-09-11:** every learned model lands here,
  the control included, so this outcome alone does not diagnose the spectral
  term. What it does flag is over-smoothing, to be checked against the blur
  diagnostic and PSNR/LPIPS against A2. For B1 that check came back as follows.
  Against A2, B1 is *not* blurrier: Sobel gradient +5.8%, high-frequency energy
  −1.3%, over the 400 in-loop validation patches at 12k. But both runs sit only
  ~2% of the way from bicubic to ground truth in the ×4 band (high-frequency
  energy frac 0.022 and 0.023). B1 pays 0.105 dB PSNR and +0.013 LPIPS against
  A2, both resolved.
- stays **well above 0.0057** → the constraint has not been learned; raising
  `lambda1` is justified.

## How the control makes this checkable

Both Day 3 runs log `val_l1_spec` and `val_sam` at every validation —
**including the control**, whose lambdas are 0.0. A control that did not measure
the term it is not optimising would leave "the spectral run drove it down" with
nothing to be down from. The columns are appended to `log.csv`
(`src/train.py::LOG_COLUMNS`); a Run A log is widened in place on resume rather
than corrupted by mixed row widths.

`src/train.py`'s `--spectral-lambda1` / `--spectral-lambda2` default to **0.0**,
recorded as such in `configs/frozen_day3.yaml`, so the control is this same file
with the flags left off and cannot acquire the objective by a forgotten
argument.

## Caveats

- The floor is measured with `D = area`. It is only comparable with a run that
  trains through the same operator; `tests/test_frozen_config.py` asserts the
  freeze and `configs/base.yaml` name the same one.
- The SAM term cannot reach 0 by construction: the cosine is clamped off 1.0 to
  keep `arccos` differentiable, putting a 4.88e-4 rad floor (float32) under it.
  That is 2.2% of the physical floor and applies to the training term and this
  measurement identically.
- The worst patch had 95.8% of its pixels with a defined spectral angle; the
  rest are nodata, excluded from the mean and counted, never scored as 0°.
- `reference_l1` comes from Run A. It should be re-read from Run A2's log once
  that run exists, and this table's percentages recomputed.
