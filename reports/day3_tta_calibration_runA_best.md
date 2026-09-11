# TTA-disagreement calibration — `runA_best`

**TTA disagreement IS monotone against error** on `runA_best`: MAE rises in every one of 19 steps across 20 equal-mass bins (Spearman 1.000); the most-uncertain bin's MAE is 11.5x the least-uncertain's.

Against the texture control (gradient magnitude of the SR): AUSE 0.2430 vs 0.2659 (random removal 0.7012; ratio to random 0.347 vs 0.379). TTA ranks error **better** than the edge-detector control.

_Written 2026-09-11T06:15:33+00:00._ Checkpoint `runs/runA/best.pt` (iteration 8000, 1,518,724 params, sha256 `9267674f3ba2`). Split `val`: 1199 patches, 314,310,656 pixel-band values. Error is |TTA mean − HR| in reflectance.

![calibration](day3_tta_calibration_runA_best.png)

## MAE per equal-mass bin of TTA std (pooled bands)

| bin | mass | TTA std range (refl.) | mean std | MAE | RMSE | RMSE / mean std | implied logvar ln(RMSE²) |
|---|---|---|---|---|---|---|---|
| 0 | 0.051 | 1.55e-05 – 2.69e-04 | 2.11e-04 | 0.00273 | 0.00396 | 18.75 | -11.06 |
| 1 | 0.047 | 2.69e-04 – 3.39e-04 | 3.05e-04 | 0.00334 | 0.00485 | 15.90 | -10.66 |
| 2 | 0.053 | 3.39e-04 – 4.07e-04 | 3.74e-04 | 0.00378 | 0.00552 | 14.78 | -10.40 |
| 3 | 0.048 | 4.07e-04 – 4.68e-04 | 4.38e-04 | 0.00420 | 0.00615 | 14.06 | -10.18 |
| 4 | 0.054 | 4.68e-04 – 5.37e-04 | 5.02e-04 | 0.00463 | 0.00678 | 13.50 | -9.99 |
| 5 | 0.049 | 5.37e-04 – 6.03e-04 | 5.69e-04 | 0.00506 | 0.00742 | 13.03 | -9.81 |
| 6 | 0.051 | 6.03e-04 – 6.76e-04 | 6.39e-04 | 0.00550 | 0.00806 | 12.62 | -9.64 |
| 7 | 0.052 | 6.76e-04 – 7.59e-04 | 7.17e-04 | 0.00598 | 0.00876 | 12.22 | -9.48 |
| 8 | 0.042 | 7.59e-04 – 8.32e-04 | 7.95e-04 | 0.00645 | 0.00942 | 11.86 | -9.33 |
| 9 | 0.052 | 8.32e-04 – 9.33e-04 | 8.81e-04 | 0.00697 | 0.01016 | 11.53 | -9.18 |
| 10 | 0.051 | 9.33e-04 – 1.05e-03 | 9.89e-04 | 0.00761 | 0.01107 | 11.19 | -9.01 |
| 11 | 0.049 | 1.05e-03 – 1.17e-03 | 1.11e-03 | 0.00832 | 0.01204 | 10.86 | -8.84 |
| 12 | 0.055 | 1.17e-03 – 1.35e-03 | 1.26e-03 | 0.00917 | 0.01322 | 10.50 | -8.65 |
| 13 | 0.050 | 1.35e-03 – 1.55e-03 | 1.44e-03 | 0.01018 | 0.01460 | 10.10 | -8.45 |
| 14 | 0.045 | 1.55e-03 – 1.78e-03 | 1.66e-03 | 0.01131 | 0.01613 | 9.72 | -8.25 |
| 15 | 0.052 | 1.78e-03 – 2.14e-03 | 1.95e-03 | 0.01275 | 0.01808 | 9.28 | -8.03 |
| 16 | 0.049 | 2.14e-03 – 2.63e-03 | 2.37e-03 | 0.01469 | 0.02069 | 8.74 | -7.76 |
| 17 | 0.052 | 2.63e-03 – 3.47e-03 | 3.01e-03 | 0.01731 | 0.02419 | 8.03 | -7.44 |
| 18 | 0.049 | 3.47e-03 – 5.01e-03 | 4.13e-03 | 0.02111 | 0.02927 | 7.08 | -7.06 |
| 19 | 0.050 | 5.01e-03 – 1.70e-01 | 8.31e-03 | 0.03152 | 0.04503 | 5.42 | -6.20 |

## Per band

| band | TTA monotone | inversions | Spearman | top/bottom | AUSE ratio TTA | AUSE ratio control |
|---|---|---|---|---|---|---|
| B04 | yes | 0 | 1.000 | 9.3x | 0.382 | 0.420 |
| B03 | yes | 0 | 1.000 | 10.1x | 0.376 | 0.392 |
| B02 | yes | 0 | 1.000 | 12.8x | 0.341 | 0.365 |
| B08 | yes | 0 | 1.000 | 6.9x | 0.427 | 0.494 |

Control monotonicity (pooled): 0 inversion(s), Spearman 1.000, top/bottom 9.5x.

## Side numbers

- **PSNR** (per-patch mean, all bands, data_range 1.0): TTA mean 38.8432 dB vs plain forward 38.8260 dB (+0.0172 dB). Diagnostic only; the headline table's numbers come from scripts/eval_all_ckpts.py.
- **Day 4 clamp check.** 19.9% of pixel mass sits in TTA bins whose implied log-variance ln(RMSE²) is below the head's clamp floor -10.0 (sigma 0.0067 refl.). Approximate -- within-bin spread makes the true conditional variances wider -- but it says how much of the raster a well-trained head would pin to the floor.

## Reproduce

```
D:\SIH\DrishtiSR\.venv\Scripts\python.exe scripts/calibrate_uncertainty.py --checkpoint runs/runA/best.pt
```
