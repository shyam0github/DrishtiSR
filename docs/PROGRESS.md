# DrishtiSR — progress and handoff log

SIH 2026, problem statement **SIH26142**: Sentinel-2 super-resolution 10 m → 2.5 m (x4).

One block per working day, newest last. Each block answers the same four
questions so a block can be read cold, without the conversation that produced
it: **what ran**, **what it produced**, **what is not yet true**, and **how to
reproduce it**. Numbers here are copied from the artefacts named beside them;
where a number was measured for this document rather than read out of a result
file, it says so.

> **Format note.** `docs/PROGRESS.md` did not exist before the Day 2 entry, and
> the repository contained no prior handoff document to copy. The structure
> below follows the house style of `reports/day1_gate.md` and
> `reports/day2_runA.md` — verdict first, provenance next to every number,
> caveats as blockquotes, reproduction commands last. Day 1 is reconstructed
> from the artefacts it left behind and cites them; it was not written on the
> day.

---

## Day 1 — the bicubic floor

**Source: `reports/day1_gate.md` (generated 2026-09-06 by `scripts/make_day1_gate.py`).**
Reconstructed here for continuity; that report is authoritative.

- **Gate: PASSED.** A benchmarked bicubic floor exists on real co-registered
  Sentinel-2 / NAIP pairs, measured twice — by `src.metrics.aggregate` and by
  the external `opensr-test` suite — over **1199 validation patches, 0 rejected**.
- Alignment audit **PASS**, median shift **0.596 HR px = 1.49 m** against a
  1.0 px threshold. The mapping the model has to learn is present in the data.
- Bicubic floor: **38.4717 dB PSNR**, **0.8825 SSIM**, p5 30.17 dB.
- opensr-test scored 1199/1199, putting bicubic at hallucination 0.0808 /
  omission 0.8620 — the correct signature for a method that invents nothing.
- Dataset `sen2naipv2-crosssensor`, 3000 pairs cached, bands B04/B03/B02/B08,
  reflectance = DN / 10000, LR patch 64 → HR patch 256.

---

## Day 2 — Run A trained, and put on real Delhi imagery

**Two separate results, and they must not be merged.** Run A passed its
validation gate on SEN2NAIP (`reports/day2_runA.md`), and Run A produces a
plausible, sharper-than-bicubic raster over central Delhi (this block). The
second is a qualitative check on out-of-distribution data with **no ground
truth**; it is evidence that the pipeline works end to end, not a measurement
of accuracy.

### 1. What ran

**Training (Kaggle T4, headless).** EDSR-baseline, 16 residual blocks, 64
features, 4-channel in/out (RGBN), x4 PixelShuffle, no batch norm. 40,000
iterations, L1 only, no uncertainty head, no spectral loss. Checkpoint
`runs/runA/best.pt`.

**Gate (`reports/day2_runA.md`).** PASSED on all three metrics over the full
1199-patch validation split: PSNR 38.4717 → **38.8260** (+0.3543), SSIM 0.8825
→ **0.8896** (+0.0072), LPIPS 0.4033 → **0.3219** (−0.0814).

**Inference on real Delhi imagery**, CPU-only, no AMP:

```
python -m drishtisr.infer.tiled --ckpt runs/runA/best.pt \
  --input data/delhi/delhi_A_20241030.tif --output outputs/delhi_A_sr.tif \
  --tile 256 --overlap 32 --scale 4 --out-dtype uint16 --device cpu --amp 0
```

Scene A is central Delhi (dense continuous urban fabric), STAC item
`S2A_MSIL2A_20241030T052941_R105_T43RGM_20241030T093050`, cloud cover 2.3e-05,
fetched by `scripts/fetch_delhi.py` and described in `data/delhi/manifest.json`.
36 tiles at 224 px stride, Hann-blended. **1.91 s per 256x256x4 tile on 6
threads** (measured for this document, not from the run) → ≈69 s of forward
pass; end-to-end wall clock was several minutes, dominated by the deflate write
of the 128 MB output rather than by the model.

> `python -m drishtisr.infer.tiled` works because `drishtisr/__init__.py`
> points its `__path__` at `src/`. The module's real home is
> `src/infer/tiled.py`; `drishtisr/` contains nothing but that alias.

### 2. What it produced

`outputs/delhi_A_sr.tif` — gitignored, 128 MB:

| | input | output |
|---|---|---|
| size (C x H x W) | 4 x 1243 x 1294 | 4 x 4972 x 5176 |
| pixel size | 10.0 m | **2.5 m** |
| CRS | EPSG:32643 | EPSG:32643 |
| origin (UTM 43N) | 710110.000000, 3172450.000000 | 710110.000000, 3172450.000000 |
| dtype | uint16 | uint16 |
| bands | B04, B03, B02, B08 | B04, B03, B02, B08 |
| DN range | — | 584 .. 9015 (reflectance 0.0584 .. 0.9015) |

**Validation — 10 of 10 checks PASS** (`scripts/validate_sr.py`, re-opening the
file with rasterio rather than trusting the writer's own report): shape is
exactly x4; pixel size 2.5 m matches `dataset.hr_gsd_m` and the input's 10 m / 4;
CRS identical; origin delta **(0, 0) m**; skew terms zero; 0 non-finite of
102,940,288 samples; 0 pixels at the uint16 floor or ceiling; DN range inside
`delhi.expect_dn_min/max` = [0, 12000]. The inference run emitted **no clipping
warning**, so nothing was lost to the uint16 write.

**Figures** (`scripts/smoke_view.py`): `outputs/day2_smoke.png` — three panels
(nearest x4 / bicubic x4 / EDSR SR) over a 256x256 LR crop at (x=830, y=286),
found automatically by an NDVI-plus-variance search and landing on the
Yamuna / ITO area — road bridges, dense blocks, a stadium, open water.
`outputs/day2_smoke_full.png` — the whole scene, LR and SR, decimated to ~900 px
with the crop footprint marked.

> **All three panels share ONE 2–98 percentile stretch, computed from the LR
> crop and applied unchanged to each.** This is the only thing that makes the
> comparison honest: a per-panel stretch hands the blurriest panel its own
> contrast boost and the figure stops meaning anything. The LR crop is the
> reference because it is the common ancestor of all three panels, so none is
> stretched to fit itself. The stretch is display-only and never touches data
> that reaches a loss, a metric or a raster.

### 3. Honest read of the figures

**It is genuinely sharper, not merely re-smoothed — and it is also inventing
texture.** Both halves of that sentence are supported:

- **Sharper.** Mean gradient magnitude over the crop is **1.30x bicubic** and
  1.38x nearest. At 1:1 zoom, road edges, building outlines and the stadium
  ring are crisply resolved where bicubic is blurred. This is not a stretch
  artefact — the stretch is identical across panels.
- **Overshoot / ringing.** **18.1%** of SR pixels fall outside the local (9x9)
  min–max envelope of the LR source, against **5.7%** for bicubic. Visible at
  zoom as dark haloes rimming bright roofs. Classic L1-trained-EDSR edge
  overshoot.
- **Texture invention in flat areas.** High-frequency energy is **1.71x**
  bicubic inside the flattest quartile of the crop, against 1.23x over the crop
  as a whole. The model amplifies *proportionally more* where there is least
  real structure — a mottled, wormy texture appears in dark vegetation and
  shadow that bicubic does not produce. This is amplified sensor noise
  shading into hallucination, and it is exactly the failure the Day 3
  uncertainty head is meant to expose.
- **Colour shift — small, systematic, real.** Degrading the SR back to 10 m by
  x4 area-average and differencing against the input gives MAE **0.0059
  reflectance (≈59 DN)** on crop means of 0.16–0.30, with a consistent sign:
  visible bands biased **darker** (B04 −0.0022, B03 −0.0039, B02 −0.0022) and
  NIR **brighter** (B08 +0.0031). A ~2% relative shift, invisible on screen but
  not nothing for a project whose headline contribution is spectral
  consistency.
- **No checkerboarding.** The x4 pixel-shuffle phase probe gives mean |gradient|
  of 0.00536 / 0.00508 / 0.00505 / 0.00536 across x-phases (≈6% spread) and a
  similar ≈5% across y-phases. A faint sub-pixel-convolution signature is
  measurable; nothing is visible at any zoom.
- **No tile seams.** Over the whole raster, the worst column seam sits at
  **+2.3σ** of the column-gradient profile, and **0.17%** of all columns exceed
  it; the worst row seam is +1.3σ, exceeded by 10.6% of all rows. The elevated
  columns cluster in the dense urban core rather than at tile joins. The Hann
  blend is doing its job.

### 4. What is NOT yet true

> **PARAMETER BUDGET EXCEEDED.** Run A is **1,518,724 parameters** against the
> 1,000,000 limit in `cfg.runtime.max_parameters` — **1.52x over**. Run A is a
> *reference measurement*, not a deliverable. Every number in this block
> inherits that caveat, and none of them may be quoted as a submission result
> without this sentence attached.

- **No uncertainty head.** Contribution 2 does not exist yet in code. The
  texture invention measured above is precisely what it needs to flag.
- **No spectral-consistency loss.** The 59 DN round-trip error above was
  measured, not optimised against. Contribution 1 is unimplemented.
- **No ONNX export, no INT8, no CPU latency benchmark.** Contribution 3 is
  untouched. The 1.91 s/tile above is eager float32 PyTorch, not the
  deployment path.
- **No ground truth over Delhi.** There is no 2.5 m reference for this scene.
  Every Delhi statement is qualitative or a self-consistency check. Accuracy
  numbers come only from the SEN2NAIP validation split.
- ~~**Open question — the BOA offset.**~~ **CLOSED 2026-09-08, empirically.**
  The offset WAS needed. See the Day 2 addendum below: every number in §3 above
  was measured on Delhi inputs a uniform +0.1 reflectance too bright, and §3 is
  superseded by the re-run recorded there.
- **One scene only.** `data/delhi/delhi_B_20241030.tif` (Gurugram edge, mixed
  peri-urban and agriculture) was fetched but not super-resolved. Farmland is a
  different texture and the texture-invention finding above should be re-checked
  there.

### 5. Reproducing this

```bash
# Inference (CPU, ~minutes)
.venv/Scripts/python.exe -m drishtisr.infer.tiled --ckpt runs/runA/best.pt \
    --input data/delhi/delhi_A_20241030.tif --output outputs/delhi_A_sr.tif \
    --tile 256 --overlap 32 --scale 4 --out-dtype uint16 --device cpu --amp 0

# Figures
.venv/Scripts/python.exe scripts/smoke_view.py

# GeoTIFF validation (exit 0 = all checks pass, 2 = a check failed)
.venv/Scripts/python.exe scripts/validate_sr.py
```

Pre-flight for both new scripts — offline, no data, seconds on CPU. They
compose: `smoke_view --smoke` fabricates an LR/SR GeoTIFF pair, and
`validate_sr --smoke` then validates that pair.

```bash
.venv/Scripts/python.exe scripts/smoke_view.py --smoke
.venv/Scripts/python.exe scripts/validate_sr.py --smoke
```

Test suite: `.venv/Scripts/python.exe -m pytest` — **623 passed**.

New config blocks: `smoke_view` and `validate_sr` in `configs/base.yaml`, each
with a matching entry under `smoke:`. `validate_sr` deliberately holds no
bounds of its own — expected pixel size is `dataset.hr_gsd_m`, DN bounds are
`delhi.expect_dn_min/max` — so it cannot keep passing after the config it
guards has moved.

### 6. Next

1. Add the heteroscedastic NLL uncertainty head and check its map against the
   flat-area texture invention measured in §3.
2. Add the spectral-consistency loss; re-measure the 59 DN round-trip error as
   a tracked metric rather than a one-off.
3. ~~Settle the BOA offset question above~~ — settled 2026-09-08, see the
   addendum below. Delhi numbers are re-measured with `--dn-offset 1000`.
4. Get under the 1,000,000-parameter budget — Run A's headroom problem is
   structural, not a matter of trimming.
5. Run scene B and repeat the artefact read on agricultural texture.

---

## Day 2 addendum — the BOA offset, settled empirically

**Verdict: the offset was needed. `reflectance = (dn − 1000) / 10000`, not
`dn / 10000`.** Every artefact number in Day 2 §3 was measured on Delhi inputs a
uniform **+0.1 reflectance too bright**, and is superseded by the table below.

### How it was decided

Not from documentation — Planetary Computer's `sentinel-2-l2a` items expose no
`s2:boa_add_offset` and no `raster:bands`, which is why `fetch_delhi.py` had to
derive the value from the processing baseline in the first place. The real
question was never what Delhi carries; it was **whether the SEN2NAIPv2 training
data is itself offset-corrected**, because if it were not, applying the offset
would be the error rather than the fix. That is answerable from the pixels.

Measured over **250 random `sen2naipv2-crosssensor` LR patches** — the exact
`(4, 130, 130)` uint16 arrays the dataset hands the model, nodata excluded,
unclipped — against `data/delhi/delhi_A_20241030.tif`:

| band | SEN2NAIP LR median | Delhi `dn/10000` | Delhi `(dn−1000)/10000` |
|---|---|---|---|
| B04 | 0.1104 | 0.1764 (**+0.0660**) | 0.0764 (−0.0340) |
| B03 | 0.0932 | 0.1750 (**+0.0818**) | 0.0750 (−0.0182) |
| B02 | 0.0664 | 0.1505 (**+0.0841**) | 0.0505 (−0.0159) |
| B08 | 0.2504 | 0.3077 (**+0.0573**) | 0.2077 (−0.0427) |

Sum of absolute median offsets: **0.2892 without, 0.1108 with**. The p5–p95
span is *identical* under both hypotheses (0.1361 / 0.1072 / 0.1066 / 0.1891) —
an additive shift cannot change a span — so the verdict rests on the medians and
on the floor below, never on the span or on min/max.

**The decisive evidence is the DN floor.** In raw DN, over the same 250 patches:

| cohort | B02 min | B02 p5 | B02 median | B02 pixels below DN 1000 |
|---|---|---|---|---|
| SEN2NAIP LR, NAIP date pre-2022 (n=177) | 0 | 260 | 652 | **82.1%** |
| SEN2NAIP LR, NAIP date 2022+ (n=73) | 0 | 284 | 692 | **81.0%** |
| Delhi `delhi_A`, raw DN | 0 | 1180 | 1505 | ~0% |

An uncorrected baseline ≥ 04.00 product **cannot** put 82% of a band below
DN 1000 — the offset shifts the whole distribution up by exactly that. And the
two SEN2NAIP cohorts are indistinguishable, which straddles ESA's baseline 04.00
cutover (2022-01-25): the correction is a property of **the dataset build**, not
of the acquisition date. Delhi, fetched as raw DN from baseline 05.11, shows the
uncorrected signature — a floor at ~1000 (B04 882, B03 1019, B08 1034).

> **On the acquisition-date cross-check.** SEN2NAIP sample ids carry the **NAIP**
> acquisition date (2019 ×37, 2020 ×54, 2021 ×86, 2022 ×73 in the sample), not
> the Sentinel-2 one, and the cached per-record JSON holds `crs`,
> `geotransform`, `data_split` and `correlation` — no S2 item id and no
> processing baseline. So the baseline is **not recoverable** from the cache and
> the date cohorts above are a proxy for it. That is exactly why the stratified
> DN floor was measured rather than reasoned about: it answers the question the
> metadata cannot.

### The three tables in full

Reproduce with `.venv/Scripts/python.exe scripts/boa_offset_audit.py`. All
values are surface reflectance, **unclipped**, nodata excluded. `span` is
p95 − p5.

**(1) Training — `sen2naipv2-crosssensor` LR patches, n=250, `dn / 10000`**

| band | min | p1 | p5 | p50 | p95 | p99 | max | mean | span |
|---|---|---|---|---|---|---|---|---|---|
| B04 | 0.0000 | 0.0236 | 0.0380 | 0.1104 | 0.2432 | 0.3216 | 1.1288 | 0.1215 | 0.2052 |
| B03 | 0.0000 | 0.0352 | 0.0500 | 0.0932 | 0.1880 | 0.2636 | 0.9656 | 0.1031 | 0.1380 |
| B02 | 0.0000 | 0.0164 | 0.0268 | 0.0664 | 0.1420 | 0.2116 | 0.8828 | 0.0733 | 0.1152 |
| B08 | 0.0000 | 0.1236 | 0.1672 | 0.2504 | 0.3880 | 0.4740 | 1.1968 | 0.2606 | 0.2208 |

**(H1) `delhi_A_20241030.tif`, `dn / 10000`**

| band | min | p1 | p5 | p50 | p95 | p99 | max | mean | span |
|---|---|---|---|---|---|---|---|---|---|
| B04 | 0.0882 | 0.1274 | 0.1349 | 0.1764 | 0.2710 | 0.3264 | 0.7364 | 0.1879 | 0.1361 |
| B03 | 0.1019 | 0.1322 | 0.1422 | 0.1750 | 0.2494 | 0.2974 | 0.7748 | 0.1838 | 0.1072 |
| B02 | 0.0000 | 0.1107 | 0.1180 | 0.1505 | 0.2246 | 0.2688 | 1.0912 | 0.1596 | 0.1066 |
| B08 | 0.1034 | 0.1359 | 0.2225 | 0.3077 | 0.4116 | 0.4575 | 0.7496 | 0.3107 | 0.1891 |

**(H2) `delhi_A_20241030.tif`, `(dn − 1000) / 10000`**

| band | min | p1 | p5 | p50 | p95 | p99 | max | mean | span |
|---|---|---|---|---|---|---|---|---|---|
| B04 | −0.0118 | 0.0274 | 0.0349 | 0.0764 | 0.1710 | 0.2264 | 0.6364 | 0.0879 | 0.1361 |
| B03 | 0.0019 | 0.0322 | 0.0422 | 0.0750 | 0.1494 | 0.1974 | 0.6748 | 0.0838 | 0.1072 |
| B02 | −0.1000 | 0.0107 | 0.0180 | 0.0505 | 0.1246 | 0.1688 | 0.9912 | 0.0596 | 0.1066 |
| B08 | 0.0034 | 0.0359 | 0.1225 | 0.2077 | 0.3116 | 0.3575 | 0.6496 | 0.2107 | 0.1891 |

Read the **p1 column of table 1 against table H1**: the training set has 1% of
its visible-band pixels below 0.016–0.035 reflectance, and `delhi_A` under H1
has *nothing* below 0.11. A distribution cannot lose its entire dark tail to
anything but an unsubtracted additive offset. Table H2 restores it. Note also
that the negative minima in H2 (B04 −0.0118, B02 −0.1000) are **not** an error
to clip away: offset-corrected dark water and deep shadow legitimately go
slightly negative, and `AGENTS.md` forbids silently clamping them.

### These numbers are now reproducible

Both measurements were made by throwaway scripts, which is exactly the failure
§3 of the Day 2 entry suffers from — its ringing and HF definitions were lost,
so the re-run below could not be compared to it on those two metrics. They are
entry points now:

| script | produces | smoke |
|---|---|---|
| `scripts/boa_offset_audit.py` | the three tables, the median verdict, the DN-floor cohorts | fabricated patches, no cache or network |
| `scripts/artefact_metrics.py` | the four artefact numbers, definitions written to `outputs/metrics/artefact_metrics.json` | scores the pair `smoke_view.py --smoke` fabricates |

Both re-derive every value quoted in this document exactly, including the
old-vs-new artefact table below.

### What changed

- `src/infer/tiled.py` gained `--dn-offset` / `run_file(dn_offset=...)`,
  **default 0.0**, so every existing caller is byte-identical. Forward:
  `reflectance = (dn − dn_offset) / reflect_div`. The uint16 write inverts the
  *same* transform (`dn = reflectance × div + offset`), so the SR raster keeps
  its input's DN convention and stays comparable to it band for band.
- `configs/base.yaml` gained `delhi.dn_offset: 1000.0`, with the measurement
  above recorded beside it. It is a separate key from `delhi.boa_offset_dn`
  (−1000) on purpose: one is what ESA publishes, the other is what a caller
  subtracts, and a sign error between them is the bug the block guards.

### Re-run of §3, old vs new

Both columns are the same crop (LR 256 px at x=830, y=286), the same checkpoint
and the same measurement code — only `--dn-offset` differs.

| metric | §3 as published (H1, no offset) | re-measured (H2, offset 1000) |
|---|---|---|
| gradient ratio vs bicubic | 1.30× | **1.10×** |
| flat-quartile HF energy ratio | 2.99× | **1.86×** |
| ringing, % outside the 9×9 LR envelope | 3.18% (bicubic 0.98%) | **0.79%** (bicubic 0.98%) |
| colour-shift MAE, reflectance | 0.0059 (58.6 DN) | **0.0049 (49.4 DN)** |
| per-band signed round-trip delta | B04 −0.0022, B03 −0.0039, B02 −0.0022, B08 +0.0031 | **B04 −0.0011, B03 −0.0017, B02 −0.0009, B08 +0.0016** |

**Every artefact shrank, and one changed its verdict.** Ringing goes from
**3.2× bicubic to below bicubic** — the "classic L1-EDSR edge overshoot" of §3
was substantially an out-of-distribution artefact, not a property of the model.
Flat-area texture invention drops by 38%, and the spectral round-trip error by
16% with every per-band bias roughly halved. The gradient ratio falls from 1.30×
to 1.10×, which is the same finding read honestly: part of what looked like
sharpness was overshoot.

> The gradient ratio (1.300×) and the colour-shift MAE (0.0059, and all four
> per-band signs) reproduce §3's published values exactly, which is what
> licenses the comparison. The ringing and HF-energy **absolute** values do not
> match §3's (18.1% / 5.7% and 1.71×) — those used a different envelope and
> high-pass definition, not recorded at the time. The SR:bicubic ringing *ratio*
> does reproduce §3's exactly (3.2×). Both columns above come from one
> definition applied to both rasters, so the old-vs-new comparison holds even
> though the absolutes are not comparable to §3's text.

> **A bug worth recording, found while making the above reproducible.** Metrics
> 1 and 2 (gradient, HF energy) compare SR against bicubic and are invariant to
> an additive offset. Metrics 3 and 4 (ringing, spectral round trip) compare SR
> against the LR **input** in absolute terms and are not. The first version of
> `artefact_metrics.py` read the input once with `dn_offset=0` and scored the
> corrected raster against it, reporting **88.5% ringing and 0.1005 MAE** for a
> raster whose real figures are 0.79% and 0.0049 — a number that looks like a
> catastrophic finding and is purely a units mismatch. The script now rebuilds
> the input crop and its baselines in each raster's own convention, and
> `tests/test_artefact_metrics.py` pins the failure.

`scripts/validate_sr.py` on the new raster: **10 of 10 checks pass**, DN range
[712, 8999], 0 saturated, origin delta (0, 0) m. Figures regenerated on the same
automatically-found crop (x=830, y=286, mean NDVI 0.215).

### Reproducing

```bash
.venv/Scripts/python.exe -m drishtisr.infer.tiled --ckpt runs/runA/best.pt \
    --input data/delhi/delhi_A_20241030.tif --output outputs/delhi_A_sr.tif \
    --tile 256 --overlap 32 --scale 4 --out-dtype uint16 --device cpu --amp 0 \
    --dn-offset 1000
.venv/Scripts/python.exe scripts/validate_sr.py
.venv/Scripts/python.exe scripts/smoke_view.py

# The verdict and the four artefact numbers, from the definitions in
# cfg.boa_offset_audit and cfg.artefact_metrics rather than from this file.
.venv/Scripts/python.exe scripts/boa_offset_audit.py
.venv/Scripts/python.exe scripts/artefact_metrics.py
```

> **Still not true.** Everything in Day 2 §4 stands: the checkpoint is still
> 1,518,724 parameters against a 1,000,000 budget, there is still no uncertainty
> head, no spectral-consistency loss, no ONNX/INT8 path, no ground truth over
> Delhi, and scene B is still not super-resolved. Correcting the offset makes
> the Delhi read *honest*; it does not make it an accuracy measurement.

---

## Day 2 addendum — Kaggle tooling, before Day 3 spends GPU-hours

### Run A, as actually measured (for sizing Day 3)

Read from the completed kernel log, `outputs/kaggle/runa/_logs/drishtisr-runa.log`:

| | |
|---|---|
| iterations | 40,000 (batch 16, 64 px LR patches, ×4, AMP on) |
| **sec/100it** | **median 27.7 s** (mean 27.9, min 25.5; the first window, 40.9 s, is warm-up) |
| **wall clock** | **186.0 min = 3.10 h** (`[done] iters=40000 elapsed=186.0 min`) |
| best val PSNR | 35.284 dB at it 8,000 — and *falling* thereafter (35.03 at 40k) |
| Kaggle Python | 3.12.13 |
| commit | `0dffbbb` |

**Three runs of this shape cost ≈9.3 GPU-hours** of the 30-hour weekly budget.
Note the validation curve: everything after it 8,000 was spent getting worse.
40k iterations is not the right length for a run of this size.

> **Which GPU: T4 — but the log does not say so, and that is now fixed.** The
> pushed `kernel-metadata.json` carries `"machine_shape": "NvidiaTeslaT4"`, and
> 40,000 CUDA iterations completed, which a P100 cannot do on this image (it
> dies on the first op with `cudaErrorNoKernelImageForDevice`). That is strong
> but *indirect*. The notebook template now prints
> `torch.cuda.get_device_name(0)`, its compute capability, its memory and the
> torch/CUDA versions in the header cell, so every future run records the card
> it actually got, next to its timings.

### Three fixes

1. **`logs` crashed on Windows (`charmap`).** MEASURED: 7 of Run A's 620 log
   lines contain U+2501, from pip's progress bars; printing them to a cp1252
   console raised `UnicodeEncodeError` mid-log, so `logs` failed on precisely
   the runs most worth reading. `make_console_unencodable_safe()` now sets
   `errors="backslashreplace"` on stdout/stderr at startup and leaves the
   encoding alone — forcing UTF-8 onto a cp1252 console would trade a crash for
   mojibake.
2. **The `train` job is deleted.** It named `entry: scripts/train.py`, which has
   never existed in this repository (the trainer is `src/train.py`, reached as
   `python -m drishtisr.train`), so it died in `generate_kernel`'s entry-point
   check on every attempt and was never pushed. Its `entry_args` were wrong too
   — `--config configs/base.yaml`, which `src/train.py` does not accept and
   which omits the `--data-module/--data-class/--data-root` it requires. Day 3's
   runs are defined by copying `runa`, which has actually run.
3. **The accelerator guard now exists.** T4-not-P100 previously lived in two
   comments, one config default and a pytest that reads the jobs file as
   committed — nothing refused a hand-edited `accelerator: NvidiaTeslaP100`, and
   the only job whose comments carried the rule was the unpushable `train`.
   `check_accelerator()` now runs inside `validate_job()`, i.e. at jobs-file
   **load** time, so it fires for every job on every command (`jobs`, `push`,
   `status`, `logs`) and *before* the entry-point check that used to stand in
   front of it. It refuses anything not on
   `cfg.kaggle_run.allowed_accelerators`, quoting P100's specific reason from
   `cfg.kaggle_run.forbidden_accelerators`.

Five new tests cover the guard (P100 refused, unknown value refused, CPU job not
checked, `train` cannot return without a real entry point) and the console fix.
