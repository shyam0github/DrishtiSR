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
- **Open question — the BOA offset.** `data/delhi/manifest.json` records that
  the on-disk pixels are **raw digital numbers, not offset-corrected**, with a
  derived `boa_add_offset_dn` of −1000 per band; `run_file` simply divides by
  10000. Whether the SEN2NAIPv2 rasters Run A trained on carry that offset was
  **not verified**. If they do, inference is running on a distribution shifted
  by 0.1 reflectance, which would plausibly explain part of the systematic
  colour bias in §3. Resolve before any Delhi number is reported as accuracy.
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
3. Settle the BOA offset question above before quoting any Delhi accuracy.
4. Get under the 1,000,000-parameter budget — Run A's headroom problem is
   structural, not a matter of trimming.
5. Run scene B and repeat the artefact read on agricultural texture.
