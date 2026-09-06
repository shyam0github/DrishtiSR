# Day 1 gate — bicubic floor and external benchmark

**Problem statement:** SIH26142, Sentinel-2 10 m → 2.5 m (x4) super-resolution.
**Generated:** 2026-09-06T16:40:19+00:00 by `scripts/make_day1_gate.py`, from the result files
named under each table. No number in this document was typed by hand.
**Seed:** 42 throughout.

---

## 1. Gate verdict: **PASSED**

There is a benchmarked bicubic floor, measured on real co-registered Sentinel-2
/ NAIP pairs, by two independent metric implementations — ours and the external
`opensr-test` suite — over **1199 validation patches with
0 rejected**.

The three numbers that carry the verdict:

1. **The pairs are usable.** Alignment audit verdict **PASS**, median
   shift **0.596 HR px = 1.49 m**, against a
   PASS threshold of 1.0 px.
   A model cannot learn a mapping the data does not contain, and at roughly a
   7th of an LR pixel the mapping is there.
2. **The floor is a real number, not a placeholder.** Bicubic scores
   **38.47 dB PSNR** and
   **0.8825 SSIM** over
   1199 patches, with a p5 of
   30.17 dB — the spread that says the
   mean is not being carried by easy tiles.
3. **The external benchmark ran clean and says what it should.** opensr-test
   scored **1199 of 1199 attempted samples,
   0 skipped**, and puts bicubic at
   hallucination **0.0808** with omission
   **0.8620**. That is the correct
   signature for a method that invents nothing: it omits nearly everything and
   hallucinates almost nothing. It is the floor a learned model has to move.

There is no result here that requires switching problem statements.

---

## 2. Dataset, pairs and splits

Source: `outputs/splits_sen2naipv2.csv`, `baseline_bicubic.json`.

| | |
|---|---|
| dataset | `sen2naipv2` |
| subset | `sen2naipv2-crosssensor` |
| pairs cached | 3000 |
| bands | B04, B03, B02, B08 (RGBNIR) |
| reflectance scale | DN / 10000 |
| scale factor | x4 (10 m → 2.5 m) |
| LR patch | 64 px |
| HR patch | 256 px |

**Split sizes, in tiles** (scene-disjoint; adjacent NAIP tiles overlap, so a
random split would leak):

| split | tiles |
|---|---:|
| train | 2400 |
| val | 300 |
| test | 300 |

The validation split's 300 tiles
yield **1199 patches** at `patches.lr_size=64`,
`stride=64`.

---

## 3. Alignment verdict

Source: `outputs/metrics/alignment_report.json`.

| | |
|---|---|
| pairs audited | 50 of 3000 |
| sampling | random_without_replacement, seed 42 |
| distinct regions | 50 |
| UTM zones | 5 |
| **median shift** | **0.596 HR px (1.49 m)** |
| p90 shift | 0.943 px |
| max shift | 1.315 px |
| median (dy, dx) | (-0.100, +0.050) |
| correlation at HR grid | 0.9126 |
| degenerate pairs excluded | 0 |
| **verdict** | **PASS** (PASS below 1.0 px, FAIL above 2.0 px) |

One HR pixel is 2.5 m and a quarter of an LR pixel, so the median shift is
about 0.149 LR pixels.

---

## 4. Bicubic baseline — our metrics

Source: `outputs/metrics/baseline_bicubic.json`
(and the same for each other method). n = 1199 patches,
data_range = 1.0 reflectance, computed on CPU at
6 threads.

| metric | dir | bicubic mean | bicubic p5 | bicubic p95 | nearest mean | nearest p5 | nearest p95 |
|---|:--:|---:|---:|---:|---:|---:|---:|
| PSNR (dB) | ↑ | 38.47 | 30.17 | 46.35 | 37.86 | 29.39 | 45.72 |
| SSIM | ↑ | 0.8825 | 0.7139 | 0.9771 | 0.8671 | 0.6824 | 0.9731 |
| LPIPS | ↓ | 0.4033 | 0.1529 | 0.6205 | 0.3291 | 0.1299 | 0.5156 |
| SAM (deg) | ↓ | 2.092 | 0.750 | 4.391 | 2.236 | 0.809 | 4.667 |
| SAM p95 (deg) | ↓ | 5.925 | 1.832 | 11.850 | 6.328 | 1.952 | 12.836 |
| ERGAS | ↓ | 3.050 | 1.067 | 6.070 | 3.281 | 1.137 | 6.580 |

Bicubic beats pixel replication by **0.61 dB** PSNR. That gap is the
scale on which every later result should be read: it is what the metric can
express between "did nothing" and "did the free thing".

**LPIPS is the exception and points the other way** — nearest scores
0.3291 against bicubic's
0.4033, i.e. replication looks
*better* perceptually. This is expected and is not a bug: LPIPS rewards
high-frequency content regardless of whether it is correct, and pixel replication
preserves sharp block edges that bicubic smooths away. It is the single clearest
argument in this report for why opensr-test is needed — a perceptual metric
cannot distinguish real detail from blocky artefact, and the correctness metrics
in section 5 can.

---

## 5. Bicubic baseline — external benchmark (`opensr-test` 1.3.3)

Source: `outputs/metrics/opensr_bicubic.json`
(and the same for each other method).

Independent implementation, by ESA OpenSR (Aybar et al., IEEE JSTARS 2024).
Nothing in this project wrote these metrics.

| group | metric | dir | bicubic mean | bicubic p5 | bicubic p95 | nearest mean | nearest p5 | nearest p95 | non-finite |
|---|---|:--:|---:|---:|---:|---:|---:|---:|---:|
| Consistency | `reflectance` | ↓ | 0.0022 | 0.000751 | 0.0053 | 0.0022 | 0.000755 | 0.0054 | 0 |
| Consistency | `spectral` | ↓ | 0.5152 | 0.1966 | 1.1906 | 0.5152 | 0.1943 | 1.1746 | 0 |
| Consistency | `spatial` | ↓ | 0.0062 | 0.0000 | 0.0200 | 0.0042 | 0.0000 | 0.0200 | 0 |
| Synthesis | `synthesis` | ↑ | 0.0026 | 0.000859 | 0.0063 | 0.0042 | 0.0014 | 0.0102 | 0 |
| Correctness | `ha_metric` | ↓ | 0.0808 | 0.0344 | 0.1911 | 0.1692 | 0.0867 | 0.3232 | 0 |
| Correctness | `om_metric` | ↓ | 0.8620 | 0.6935 | 0.9315 | 0.7408 | 0.5404 | 0.8554 | 0 |
| Correctness | `im_metric` | ↑ | 0.0572 | 0.0327 | 0.1133 | 0.0900 | 0.0554 | 0.1421 | 0 |

**Denominators.** `bicubic`: 1199 scored of 1199 attempted, 0 skipped (0.0%). `nearest`: 1199 scored of 1199 attempted, 0 skipped (0.0%).

Configuration, all upstream defaults:
`agg_method=pixel`,
`border_mask=16`,
`correctness_distance=nd`,
`correctness_norm=softmin`,
`gradient_threshold=auto`,
`harm_apply_spectral=True`,
`harm_apply_spatial=True`,
`rgb_bands=[0, 1, 2]` (= ['B04', 'B03', 'B02']).

**How to read this.** `om_metric`
(0.8620) dominating `ha_metric`
(0.0808) is exactly what a non-generative
upsampler should produce: bicubic omits nearly all the real high-frequency
detail and invents almost none. **The target for a learned model is to move
`im_metric` up and `om_metric` down without moving `ha_metric` up** — that
trade-off is the thing this project's uncertainty head exists to make visible,
and it is now measurable from day one rather than at submission.

### The two baselines do not order cleanly, and that is the finding

Read the two columns above against each other before trusting any single
correctness number:

| | bicubic | nearest | bicubic better? |
|---|---:|---:|:--:|
| `ha_metric` ↓ | 0.0808 | 0.1692 | **yes** |
| `om_metric` ↓ | 0.8620 | 0.7408 | no |
| `im_metric` ↑ | 0.0572 | 0.0900 | no |

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

---

## 6. Known caveats

Written to be read by someone deciding whether to trust the numbers above.
Everything here is either an assumption that could not be verified today, or a
value taken on someone else's authority.

### Verified, with the evidence stated

- **The reflectance divisor is `10000`, and this
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
  `cfg.opensr_test.max_reflectance=10`.
  This is a tripwire, not a clip — reflectance above 1.0 passes through.
- **Band order needs no remapping.** `cfg.dataset.bands` is RGBNIR and so are
  opensr-test's own datasets (its README slices `[idx, 0:3]` and documents the
  result as Red, Green, Blue). Indices are still resolved by name, never assumed.

### Assumed, and not verified

- **`border_mask=16` is upstream's default, applied to a patch a third the size
  of theirs.** Their datasets carry 484–512 px HR patches; ours are
  256 px. The crop therefore removes a
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
- **1199 patches are not 1199 independent samples.** They come from
  300 tiles, so patches from one
  tile are correlated and the effective sample size is smaller than the
  denominator suggests. The percentiles are more informative than the standard
  deviations for this reason.
- **The alignment audit is a 50-pair sample of 3000**,
  not a census. It is a random draw without replacement at seed
  42 spanning
  50 distinct regions, which
  supports the verdict, but no per-pair guarantee is implied for the other
  2950.

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

- **0 samples were rejected across all methods.** The harness
  records every rejection with its reason and keeps it in the denominator; the
  run fails outright above
  `cfg.opensr_test.max_skipped_fraction=10%`.
- **Non-finite metric values, by metric:** none.
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
.venv/Scripts/python.exe scripts/qa_alignment.py --config configs/base.yaml \
    --n-pairs 50 --seed 42

# Our metrics, both baselines
.venv/Scripts/python.exe scripts/run_baseline.py --config configs/base.yaml \
    --baseline all

# External benchmark, both baselines, whole validation split
.venv/Scripts/python.exe scripts/run_opensr_test.py --config configs/base.yaml \
    --baseline bicubic --n-samples 0
.venv/Scripts/python.exe scripts/run_opensr_test.py --config configs/base.yaml \
    --baseline nearest --n-samples 0

# This report
.venv/Scripts/python.exe scripts/make_day1_gate.py --config configs/base.yaml
```

Pre-flight for the opensr-test path (no network, no data, ~30 s on CPU):

```bash
.venv/Scripts/python.exe scripts/run_opensr_test.py --smoke
```

Test suite: `.venv/Scripts/python.exe -m pytest`.
