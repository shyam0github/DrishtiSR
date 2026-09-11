# CHECKPOINT REPORT

## Overall status
✅ COMPLETE. All acceptance criteria are met. One metadata item is flagged: the runs executed at commit `fb8659d`, not `bf7fa0f`. The diff between the two touches no training code, so the comparison stays valid.

## Deliverables

| # | Deliverable | Status | Evidence I can check |
|---|---|---|---|
| 1 | Kaggle job state | ✅ COMPLETE (checked 2026-09-11 17:54) | `python -m kaggle kernels status shyamdwivedi0/drishtisr-day3` → `COMPLETE`; output is in `runs/kaggle/day3/`, and all 22 run files are byte-identical to the tracked `runs/day3/` |
| 2 | Run metadata validity | ⚠️ FLAGGED, but a valid controlled comparison | `reports/day3_ab_summary.json` → `metadata`, `metadata_flags`. Pass for all 3 runs: config hash `e6082c69…`, dirty=false, λ values as specified, 12,000 iterations, 855,652 params. **Git SHA is `fb8659d` for all 3, not `bf7fa0f`.** `git diff --stat bf7fa0f fb8659d` changes only `scripts/verify_data_root.py` and its test, so A2 and B share one code path. Kaggle `run.log`: `code: commit fb8659d… dirty=False` |
| 3 | A2-vs-B comparison table | ✅ | `reports/day3_ab_metrics.csv`, 1199/1199 pairs per row; selected and final rows are identical (both it12000) |
| 4 | Blur-hazard verdict | ✅ (B1 fails) | `reports/day3_ab_summary.json` → `verdicts`. **"The spectral loss did not help: consistency improved 33.3%, but LPIPS is 3.9% worse (above the 2% limit) and the blur index is 0.847, below 0.90 (BLUR HAZARD)."** |
| 5 | A2 overfitting check | ✅ controlled | `reports/day3_ab_summary.json` → `overfitting`; `reports/day3_val_curves.png` |
| 6 | Data-validity verdict | ✅ (b) | `reports/day3_data_validity.md`, `reports/day3_data_validity.json` |
| 7 | PROJECT_STATE + commit | ✅ | No `PROJECT_STATE.md` exists, so the repo's state document `docs/PROGRESS.md` got a Day 3 block. Results commit `0f9fd4f`; this report is committed on top of it |

**Comparison (full val split, 1199 patches, the pre-declared `cfg.eval_all_ckpts.selection` rule: minimum opensr consistency, then a CI tie with an LPIPS tie-break. It selected it12000 for every run, and it12000 is also the final iterate):**

| row | PSNR | SSIM | LPIPS | SAM° | ERGAS | consistency L1 (opensr) | L1_spec (area) | Sobel ÷ GT | HF ÷ GT | blur index | iter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bicubic | 38.472 | 0.8825 | 0.4033 | 2.092 | 3.050 | 0.00217 | 0.00121 | 0.423 | 0.062 | — | — |
| A2 (0/0) | 38.871 | 0.8889 | 0.3322 | 1.972 | 2.897 | 0.00320 | 0.00276 | 0.449 | 0.087 | — | 12,000 |
| B1 (0.1/0.02) | 38.765 | 0.8881 | 0.3452 | 2.044 | 2.933 | 0.00213 | 0.00140 | 0.473 | 0.083 | **0.847 ⚠** | 12,000 |
| B2 (0.3/0.06) | 38.559 | 0.8864 | 0.3558 | 2.110 | 3.008 | 0.00134 | 0.00020 | 0.497 | 0.081 | **0.748 ⚠** | 12,000 |

The two blur proxies disagree. HF energy says B is blurrier than A2; Sobel gradient says B is sharper. The blur test as specified uses HF energy.

**A2 overfitting (training-loop val, 400 patches):** best PSNR 35.303 at it 10,500, final 35.298 (−0.005 dB). Best LPIPS 0.3788 at it 11,500, final 0.3794 (+0.0006). Over it 8,000–12,000 the PSNR slope is +0.009 dB per 1k iterations and the LPIPS slope −0.00007 per 1k. No sustained decline, so overfitting is controlled. A 12k run cannot show 40k-scale late overfitting.

**Data validity: (b).** The LR is real Sentinel-2 L2A (`tacoreader.v1.load("tacofoundation:sen2naipv2-crosssensor")`; 3000/3000 manifest rows are crosssensor). The HR is NAIP, harmonised to Sentinel-2 by the dataset authors. `verify_data_root.py` reads separate arrays, so there is no bug and no fix was made. Across 50 pairs: per-band r median 0.973 (min 0.926), RMSE 0.005–0.011; per-band min/max DN identical in 32/50 pairs. A synthetic LR made from this HR would give r ≈ 1.

Wording for the project:
> "Real Sentinel-2 L2A 10 m LR (B04/B03/B02/B08) paired with co-registered 2.5 m NAIP HR that the SEN2NAIPv2 authors radiometrically harmonised to the Sentinel-2 acquisition (crosssensor subset); the LR is not synthetically degraded, and LR/HR per-band statistics agree by construction."

## Numbers / counts
- Runs found: **3/3** (A2, B1, B2)
- Runs with valid metadata: **3/3** on hash, dirty flag, λ values, iterations and params; **0/3** match the stated SHA `bf7fa0f` (all 3 are `fb8659d`, a non-training diff)
- Val patches evaluated: **1199/1199**
- B1 blur index: **0.847** (BLUR HAZARD)
- B1 LPIPS change vs A2: **+3.9%** (0.3322 → 0.3452)
- B1 consistency change vs A2: **−33.3%** (0.00320 → 0.00213; tile CI excludes 0, per `reports/day3_results.md`)
- Pairs checked in STEP 4: **50/50**

## Files/artifacts created or changed
- `scripts/day3_ab_report.py` (new): full-split Sobel/HF pass, blur index, verdict, overfitting check, figure; takes `--config`, `--smoke` and `--limit`
- `configs/base.yaml`: new `day3_ab_report` block plus its smoke override. `frozen_day3.yaml` is untouched
- `.gitignore`: added `runs/kaggle/` (the raw download duplicates `runs/day3/`; checkpoints were already ignored by `*.pt`)
- `reports/day3_ab_metrics.csv`, `reports/day3_ab_summary.json`, `reports/day3_val_curves.png`
- `reports/day3_data_validity.md`, `reports/day3_data_validity.json`
- `docs/PROGRESS.md`: Day 3 results and data-validity block
- `reports/checkpoints/day3_p1_results.md` (this file)
- Local only, not committed: `runs/kaggle/day3/`, `outputs/metrics/day3_ab/`

## Verification performed
- Checked the Kaggle job once: complete. Downloaded its output and confirmed every run file matches the copy already in the repo.
- Read each run's metadata and compared hash, commit, dirty flag, λ values, iterations and params against the plan. The commit differs from the plan, so I diffed the two commits: only the data-check script and its test changed.
- Reused the existing full-split scores (PSNR, SSIM, LPIPS, SAM, ERGAS, consistency) and re-applied the existing selection rule. Measured only what was missing, the two sharpness measures. Timed 32 patches first (0.82 s/patch, about 16 min projected), then ran all 1199 on CPU.
- Ran the new script's `--smoke` path end to end. It first crashed on the missing LPIPS column; after the fix it runs clean. The full test suite passes: 956 tests.
- Data validity: re-ran `verify_data_root.py` and read which arrays it prints. Independently loaded 50 cached pairs and computed per-band statistics, correlation and error. Confirmed the dataset variant from the code, the Kaggle log and the manifest.
- Looked at the figure: labels readable, no overlaps.

## Problems/blockers
- **Commit mismatch.** Problem: the runs executed at `fb8659d`, not `bf7fa0f`. Attempted: diffed the two commits. Status: resolved as a non-confound (verify script only). Next: cite `fb8659d` in write-ups.
- **Few checkpoints.** Problem: only 2 checkpoints per run survived the old trainer (10,500/12,000; B2 5,500/12,000), so selection had 2 candidates per run. Attempted: none possible without retraining. Status: known limitation; the trainer was fixed in `891c8e0`. Next: future runs keep every checkpoint.
- **Smoke crash.** Problem: `--smoke` crashed because LPIPS is off in smoke. Attempted: made the allowed-missing columns configurable (none for real runs). Status: fixed.
- **Blur proxies disagree.** Problem: Sobel and HF energy disagree on B1. Status: open interpretation question; the verdict uses HF energy as specified.

## What I should verify manually
- □ Open `reports/day3_val_curves.png`. The orange B1 LPIPS line should sit above blue A2 from about it 1,500 onward.
- □ Open `reports/day3_ab_metrics.csv`. B1 `blur_index` should be 0.847 and `lpips` 0.3452 vs A2's 0.3322.
- □ Accept that the Day 3 runs executed at commit `fb8659d` (Kaggle `run.log` line 1: `code: commit fb8659d… dirty=False`).
- □ Approve the training-data wording in `reports/day3_data_validity.md` §4 for the submission.
- □ Decide whether HF energy stays the governing blur measure, given that Sobel rates B1 sharper.

## Next action
Start P2. It was waiting on this result. Carry in the verdict "spectral loss did not help as specified (LPIPS +3.9%, blur index 0.847)", and do not tune λ values today.
