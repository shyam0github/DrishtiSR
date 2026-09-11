# Day 3 — data validity: why LR and HR per-band statistics match

_2026-09-11, local CPU. Evidence: `reports/day3_data_validity.json`._

**Verdict: (b).** The LR is real Sentinel-2 L2A. The HR is NAIP that the dataset authors radiometrically harmonised to that Sentinel-2 observation, so the per-band statistics match by design. This is not (a), a reporting bug: `verify_data_root.py` reads the `hr` and `lr` npz members separately. It is not (c), synthetically degraded NAIP: LR and area-downsampled HR correlate at r ≈ 0.97, not ≈ 1, and they are spatially offset.

## 1. Which variant was loaded

| source | what it says |
|---|---|
| `src/data/sen2naip.py::_load_catalog` | `tacoreader.v1.load(f"tacofoundation:{self.subset}")` |
| `configs/base.yaml` `sen2naipv2.subset` | `sen2naipv2-crosssensor` (the comment notes that unet and histmatch use synthetic LR) |
| Kaggle Day 3 `run.log` | `Loading taco catalog for sen2naipv2-crosssensor`; cache at `.../sen2naipv2-crosssensor` |
| Kaggle manifest `manifest_sen2naipv2.csv` | `subset` = `sen2naipv2-crosssensor` for 3000/3000 rows |
| cached sidecar JSON | carries `correlation` (0.900–0.983 across the manifest) |

## 2. Is `verify_data_root.py` reading one array twice? No

`describe_samples` loops over `sorted(arrays)`, which gives `hr` (4, 520, 520) and then `lr` (4, 130, 130). Each is printed from its own npz member. A rerun on 2026-09-11 reproduced the observation: min and max identical to 4 dp, and means that differ in the 4th decimal (for example B04 0.1346 vs 0.1347). No fix was needed and none was applied.

## 3. 50 cached pairs, measured independently

Reflectance is DN / 10000 with nodata masked. HR was 4×4 area-averaged onto the LR grid. Pairs were drawn with `default_rng(cfg.seed)`.

| band | LR mean / std | HR mean / std | per-pair r (median, min) | RMSE median | pairs with per-pair mean equal to 4 dp |
|---|---|---|---|---|---|
| B04 | 0.1054 / 0.0582 | 0.1054 / 0.0581 | 0.973, 0.938 | 0.0081 | 23/50 |
| B03 | 0.0936 / 0.0412 | 0.0935 / 0.0412 | 0.972, 0.928 | 0.0059 | 18/50 |
| B02 | 0.0655 / 0.0351 | 0.0655 / 0.0351 | 0.973, 0.938 | 0.0051 | 21/50 |
| B08 | 0.2578 / 0.0702 | 0.2578 / 0.0702 | 0.974, 0.926 | 0.0108 | 25/50 |

- **Separate arrays.** No pair shares memory. In 0/50 pairs does any LR band equal a subsample of its HR band.
- **Harmonisation fingerprint.** In 32/50 pairs, the per-band minimum and maximum DN are identical in all four bands. Area-averaging an HR tile to make a synthetic LR would shrink that range. Matching the HR's statistics to the LR produces exactly this.
- **Not a synthetic degradation.** A synthetic LR built from its own HR would correlate with the area-mean of that HR at r ≈ 1 with near-zero RMSE. The measured values are r 0.93–0.99 and RMSE 0.005–0.011 reflectance. Day 1's alignment audit also found a median residual shift of 0.596 HR px between LR and HR (`reports/day1_gate.md`), and a synthetic pair has none.
- **The `correlation` field is not what we can recompute.** On these 50 pairs it ranges 0.90–0.97, while the all-band Pearson r of LR against area-HR ranges 0.98–0.999, and the two correlate only 0.41 across pairs. Its exact definition is not recoverable from the data, so do not describe it as "the LR/HR Pearson r".

## 4. How to describe the training data

> Training pairs come from SEN2NAIPv2, "crosssensor" subset. The LR is real Sentinel-2 L2A surface reflectance at 10 m (B04, B03, B02, B08). The HR is co-registered NAIP aerial imagery at 2.5 m that the dataset authors radiometrically harmonised to the Sentinel-2 acquisition. The LR is not synthetically degraded. Because the HR is harmonised to the LR, their per-band reflectance statistics agree by construction: the HR's radiometry is Sentinel-2-referenced, and the mapping the model learns is spatial detail, not a radiometric transfer between sensors.

Training data and splits were not changed.
