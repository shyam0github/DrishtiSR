# Day 1 gate — SEN2NAIPv2 data audit

**Problem statement:** SIH26142, Sentinel-2 10 m → 2.5 m (x4) super-resolution.
**Dataset:** `sen2naipv2-crosssensor`, 3,000 pairs cached, filtered at
`min_correlation >= 0.9` from 8,000 catalog rows (4,409 survive the filter).
**Date:** 2026-09-06. **Seed:** 42 throughout.

**Gate verdict: PASS.** The pairs are co-registered, the reflectance divisor is
correct, and no sample is lost to nodata. Training may proceed.

---

## 1. What changed before this audit ran

The index was amended before any number below was produced. Three of the changes
alter what the numbers mean, so they are recorded here rather than in a commit
message alone.

**The centre crop left the read path.** `SEN2NAIPv2Dataset.load_sample` used to
centre-crop every tile to `cfg.sr.lr_patch_size` before returning it. Every
statistic taken through it therefore described the middle of the tile. Nodata
sits at tile edges and bright targets sit anywhere but the middle, so the two
questions this audit exists to answer were both being asked of the wrong pixels.
`load_sample` now returns the full stored tile — LR `(4, 130, 130)`, HR
`(4, 520, 520)`. The deterministic crop moved to
`src/data/patches.centre_crop_pair` and is applied by the loader's **grid**
(validation) path only; training draws random crops from the whole tile.

*The cache was never affected.* `ensure_cached` always wrote `src.read()`
straight to disk with no crop, no band selection and no reflectance division, so
all 3,000 `.npz` files hold full raw `uint16` tiles. Nothing was re-fetched.

**Nodata is measured, not filled.** A pixel counts as nodata when **any**
selected band equals 65535 — one dead band makes the pixel unusable, and the
previous `mask.mean()` over `(C, H, W)` would have reported a quarter of the
true loss on a 4-band tile with one dead band. Nodata pixels are excluded from
every statistic rather than replaced with `nodata_fill` and averaged in as if
they were observations of a perfectly black surface.

**Percentiles are exact.** The source is integer digital numbers, so the index
accumulates a 65,536-bin count per band and reads percentiles off the inverse
CDF. No binning error, and memory is bounded regardless of archive size.

---

## 2. Index: partial (1,100) vs full (3,000)

Full outputs: [`index_summary_partial.txt`](index_summary_partial.txt),
[`index_summary_full.txt`](index_summary_full.txt).

The partial baseline is the **first 1,100 catalog records** — the download had
already completed by the time the amendment landed, so the partial set was
reconstructed by restricting the catalog (`sen2naipv2.num_samples=1100`) rather
than by cache state. Catalog order is download order, so these are the same
1,100 records that were on disk at the partial moment.

Both runs used `--force`. The manifest on disk had been written by the
pre-amendment `build_index` under the old schema; without `--force` the
incremental skip would have preserved all 3,000 of those rows and the comparison
would have been between two schemas rather than two sample sizes.

### Rejections

| | partial (1,100) | full (3,000) |
|---|---|---|
| accepted | 1,100 | 3,000 |
| rejected | **0** | **0** |

Zero rejections at `cfg.dataset.max_nodata_fraction = 0.02` (lowered from 0.05
as part of the amendment). **No tile in the cached archive contains a single
nodata pixel.** This is worth stating plainly because it is the reason the
threshold change cost nothing: it was tightened by more than half and still did
not bite.

### Reflectance above 1.2

| | partial | full |
|---|---|---|
| samples exceeding 1.2 | 3 of 1,100 | 9 of 3,000 |
| rate | **0.27 %** | **0.30 %** |

**No meaningful jump.** +0.03 percentage points across a near-tripling of the
sample. Both are an order of magnitude below the 5 % threshold at which the
index calls the divisor suspect. The 3 partial-set samples are a strict subset
of the 9 in the full set, which is what a stable bright tail looks like.

### Per-band 99th percentile, pooled over every indexed pixel

| band | p99 partial | p99 full | Δ | rel. |
|---|---:|---:|---:|---:|
| lr_B04 | 0.3336 | 0.3144 | −0.0192 | −5.8 % |
| lr_B03 | 0.2656 | 0.2476 | −0.0180 | −6.8 % |
| lr_B02 | 0.2128 | 0.1956 | −0.0172 | −8.1 % |
| lr_B08 | 0.4672 | 0.4708 | +0.0036 | +0.8 % |
| hr_B04 | 0.3336 | 0.3144 | −0.0192 | −5.8 % |
| hr_B03 | 0.2656 | 0.2472 | −0.0184 | −6.9 % |
| hr_B02 | 0.2128 | 0.1956 | −0.0172 | −8.1 % |
| hr_B08 | 0.4672 | 0.4708 | +0.0036 | +0.8 % |

**This is land cover, not a divisor fault**, on three independent grounds:

1. **Direction.** The visible bands went *down* and NIR went *up*. A wrong
   divisor is a single multiplicative constant applied to every band at once —
   it cannot move VIS and NIR in opposite directions. A shift toward more
   vegetation and less bright bare/urban surface does exactly this.
2. **Geography.** The partial 1,100 span UTM zones 10–12 (California to Utah);
   the full 3,000 reach zones 10–14, adding the higher-vegetation eastern half
   of the archive. The catalog is ordered geographically, so the extra 1,900
   records are a different landscape, not more of the same one.
3. **Magnitude.** 6–8 %. The divisor failure mode this check exists for is
   `/3000` instead of `/10000`, which is a factor of 3.33 — a 233 % move, not an
   8 % one.

The LR and HR percentiles agree to within 0.0004 reflectance in every band and
at every quantile. Two independently-acquired sources landing on the same
distribution is strong evidence that both `cfg.dataset.reflectance_scale = 10000`
and the `[B04, B03, B02, B08]` band order are right for the **whole** archive,
not just the part indexed first.

**Divisor verdict: sound.** No change to `reflectance_scale` or `bands`.

---

## 3. Alignment audit: partial vs full

Reports: `outputs/metrics/alignment_report_partial1100.json`,
`outputs/metrics/alignment_report.json`.

| | partial (1,100) | full (3,000) |
|---|---|---|
| pairs audited | 50 | 50 |
| **median shift** | **0.510 HR px** (1.27 m) | **0.596 HR px** (1.49 m) |
| p90 shift | 0.864 px | 0.943 px |
| max shift | 1.315 px | 1.315 px |
| median (dy, dx) | (+0.000, +0.100) | (−0.100, +0.050) |
| correlation at HR grid | 0.9014 | 0.9126 |
| degenerate pairs excluded | 0 | 0 |
| **verdict** | **PASS** | **PASS** |

Median shift rose by **0.086 HR px — 0.21 m, or 0.02 LR pixels**. Both sit
comfortably below the 1.0 px PASS threshold, and the maximum is identical
between the two sets. The estimator's self-test (2.0 px injected, 2.00 px
recovered, 0.00 px on the unshifted control) passes on every run, so these are
measurements rather than opinions.

One HR pixel is 2.5 m and a quarter of an LR pixel. A 0.6 px median means the
pairs are registered to roughly a seventh of an LR pixel — well inside what
super-resolution training tolerates.

### Coverage of the audit

Both audits drew a **random sample without replacement, seed 42**, from the
manifest rows that passed validation. Every selected sample id is listed in the
JSON report and printed by the script.

| | partial | full |
|---|---|---|
| distinct grid cells | **50 of 50 pairs** | **50 of 50 pairs** |
| UTM zones spanned | 3 (32610–32612) | 5 (32610–32614) |

No two audited pairs came from the same SEN2NAIP grid cell in either run. The
full audit spans **50 distinct regions across 5 UTM zones**, not 50 pairs from
one place — which is the claim the sampling change exists to make defensible.
Sequential selection would have returned one contiguous block of the catalog and
could not have supported it.

---

## 4. Standing limitations

- **`min_correlation >= 0.9` is a pre-filter on the catalog's own metric**, not
  an independent quality gate. 4,409 of 8,000 rows survive it and 3,000 are
  taken from the top of that list. The audit describes those 3,000.
- **No cloud mask.** SEN2NAIPv2 ships none. `cfg.patches.filters` uses a
  brightness proxy, which also catches snow and specular roofs, and is
  documented as a proxy in the config.
- **`crosssensor` is 100 % `train`.** The subset carries no validation or test
  split, so the held-out split must be cut geographically — adjacent NAIP tiles
  overlap and a random split would leak. See `src/data/splits.py`.
- **9 samples exceed 1.2 reflectance and are kept unclipped**, at up to 1.685.
  They are spectrally flat and bright across all four bands, which is the
  signature of cloud or a specular roof — real signal. Clipping them would
  destroy the radiometry the spectral-consistency objective depends on.
  `cfg.dataset.reflectance_valid_max = 2.0` catches decode faults without
  touching this tail.

## 5. Reproducing this

```bash
# Index (both summaries in reports/)
py -3.11 scripts/prepare_data.py --config configs/base.yaml --index-only --force \
    --no-preview --summary-out reports/index_summary_partial.txt \
    --set sen2naipv2.num_samples=1100
py -3.11 scripts/prepare_data.py --config configs/base.yaml --index-only --force \
    --no-preview --summary-out reports/index_summary_full.txt

# Alignment audit
py -3.11 scripts/qa_alignment.py --config configs/base.yaml --n-pairs 50 --seed 42

# Estimator self-test (CPU, no network, ~30 s)
py -3.11 scripts/qa_alignment.py --config configs/base.yaml --smoke
```

Test suite: **306 passing** on CPU, no data required (`py -3.11 -m pytest`).
