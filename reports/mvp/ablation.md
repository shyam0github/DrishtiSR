# Ablation (VAL)

| Row | LPIPS↓ (vs bicubic) | Spec L1 opensr↓ | Spec L1 training op↓ (GT floor 0.005733) | SAM°↓ | HF ratio vs GT | Blur index | Sobel ratio vs GT | SSIM↑ | ERGAS↓ | PSNR↑ dB | Params | Split | n | cfg hash | git SHA | dirty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bicubic | 0.4033 | 0.002170 | 0.001210 | 2.092 | 0.0621 | — | 0.4232 | 0.8825 | 3.0503 | 38.472 | 0 | val | 1199 | — | — | — |
| Run A † | 0.3219 (-20.2%) | 0.003428 | 0.003114 | 2.022 | not measured | — | not measured | 0.8896 | 2.9036 | 38.826 | 1,518,724 | val | 1199 | — | — | — |
| A2 (control) | 0.3322 (-17.6%) | 0.003199 | 0.002761 | 1.972 | 0.0868 | — | 0.4488 | 0.8889 | 2.8966 | 38.871 | 855,652 | val | 1199 | e6082c69 | fb8659d | no |
| B1 (λ₁=0.1, λ₂=0.02) | 0.3452 (-14.4%); Δ vs A2 +3.9% | 0.002134 (Δ vs A2 -33.3%) | 0.001396 | 2.044 | 0.0830 | 0.847 ⚠ BLUR HAZARD | 0.4728 | 0.8881 | 2.9333 | 38.765 | 855,652 | val | 1199 | e6082c69 | fb8659d | no |
| B2 (λ₁=0.3, λ₂=0.06) | 0.3558 (-11.8%); Δ vs A2 +7.1% | 0.001344 (Δ vs A2 -58.0%) | 0.000200 | 2.110 | 0.0806 | 0.748 ⚠ BLUR HAZARD | 0.4970 | 0.8864 | 3.0080 | 38.559 | 855,652 | val | 1199 | e6082c69 | fb8659d | no |
| Best ★ (A2, Day 3 verdict) | 0.3322 (-17.6%) | 0.003199 | 0.002761 | 1.972 | 0.0868 | — | 0.4488 | 0.8889 | 2.8966 | 38.871 | 855,652 | val | 1199 | e6082c69 | fb8659d | no |
| A2 + consistency projection | 0.3521; Δ vs A2 +5.7% | not run | 0.000411 (Δ vs A2 -84.5%) | 1.968 | 0.0901 | — | not measured | 0.8907 | 2.8417 | 38.784 | 855,652 | val (subset) | 200 | — | 4978e05 | yes |

**Notes**

- † Run A is not a valid control. It differs from A2/B on four axes: architecture (1.52M vs 855,652 params, over the 1M budget), no dihedral augmentation, frozen crop origins (2,400 fixed crops for all 40k iterations due to infinite(train_dl) with persistent_workers), and dataset size. Its numbers carry "interim": true.
- A2, B1 and B2 executed at commit fb8659d (12,000 iterations each); their VAL numbers are taken from reports/day3_ab_metrics.csv (full 1,199-patch VAL) and not re-evaluated.
- ★ Best = A2, fixed by the pre-registered Day 3 verdict (training-time spectral loss did not help: B1 LPIPS +3.9% > 2% limit and blur index 0.847 < 0.90). Not re-selected across runs.
- Blur index = (HF_B − HF_bicubic) / (HF_A2 − HF_bicubic); rows below 0.90 are flagged BLUR HAZARD. The HF energy ratio governs; Sobel is reported alongside.
- Spec consistency L1 (opensr) = opensr-test `reflectance` measure (headline). Training-operator column = down4 area L1 vs LR; the GT floor is the same measure with HR in place of SR.
- Dirty working tree at run/eval time: A2 + consistency projection.
- A2 + consistency projection: n=200 VAL patches, NOT the full split; compare only with the A2 row of the same file; projection gate FAILED: not served; Δ spec L1 vs A2 uses the training operator (opensr measure not run); opensr consistency not run (projected 740s > 600s budget); Sobel ratio not measured by the projection gate.
- Run A †: HF/Sobel ratios not measured for Run A (Day 3 sharpness pass covered A2/B1/B2 only); Run A predates run_metadata.json: no cfg hash / git SHA / dirty flag recorded.
