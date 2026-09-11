# DrishtiSR — project state (current truth)

Read this first. Dated history lives in `docs/PROGRESS.md`; this file holds only
what is true *now*. Last full rewrite: 2026-09-11 (P0a), at main `1e7986e`.

## 1. Project in one paragraph

DrishtiSR is a Smart India Hackathon 2026 entry for problem statement SIH26142
(repo github.com/shyam0github/DrishtiSR): deep-learning super-resolution of
Sentinel-2 RGBN (B04, B03, B02, B08) from 10 m to 2.5 m (×4), trained on
SEN2NAIPv2 and demonstrated on Delhi. Team name "Turing Testers" —
Verification needed: the name appears nowhere in the repo. Three contributions:
1. **Spectral consistency** — the SR output degraded back to 10 m must reproduce the LR reflectance.
2. **Per-pixel uncertainty** — heteroscedastic log-variance head, with TTA disagreement as the fallback.
3. **CPU-only deployment** — ≤1,000,000 params, INT8 ONNX, 6-thread inference (AGENTS.md §1).

## 2. Frozen decisions & rules (do NOT change)

- AGENTS.md is the working agreement; it overrides everything here.
- Training config `configs/frozen_day3.yaml`, hash `e6082c69…` (re-verified 2026-09-11 via `load_frozen()`). Never edit; copy and re-freeze for a new run.
- Val/test ids are fixed (300 tiles each, `outputs/splits_sen2naipv2.csv`); the Day 3 cache expansion extended train only.
- Reflectance = DN / 10000 for SEN2NAIP; raw Delhi scenes need `--dn-offset 1000` (PROGRESS.md, Day 2 addendum). No ImageNet normalisation, no silent clipping.
- Kaggle GPU is **T4; P100 is refused** (`configs/base.yaml` `kaggle_run.forbidden_accelerators`). Always `python -m kaggle` via the venv; `--data-root auto` (the mount is `/kaggle/input/datasets/<owner>/<slug>`).
- Checkpoints are chosen post hoc by the pre-registered rule (`cfg.eval_all_ckpts.selection`, `src/eval/ckpt_selection.py`), never by `best.pt`.
- Every run writes `run_metadata.json` with config hash, git SHA, dirty flag and λ values (see `runs/day3/*/run_metadata.json`).
- One inference path: `src/infer/tiled.py`, invoked as `python -m drishtisr.infer.tiled`.
- The metrics panel leads with LPIPS and spectral consistency. Consistency headlines use the opensr measure.
- The governing blur measure is the HF-energy ratio (pre-registered); Sobel is reported alongside it.
- Approved data wording (verdict b): "Real Sentinel-2 L2A 10 m inputs (B04, B03, B02, B08) paired with 2.5 m NAIP targets that the SEN2NAIPv2 authors co-registered and radiometrically harmonised to Sentinel-2 (crosssensor subset). Inputs are not synthetically degraded; matching per-band statistics between inputs and targets are expected from this harmonisation."
- Decisions of 2026-09-11 (session brief): Day 3 P2 is cancelled, superseded by the night MVP build. Day 4 "Run C" = a post-hoc heteroscedastic Laplace head on the FROZEN A2 backbone, with its config frozen before training, CPU training, and no spectral term. Novelty 1 is carried by consistency maps/metrics plus a gated inference-time consistency projection. The MVP runs on CPU; Kaggle is a fallback only.
- Git: stage explicit paths only; no AI attribution in commits; never switch the branch of the shared tree (AGENTS.md §4).

## 3. Day-by-day status

| Day | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Data + alignment | ✅ | `reports/day1_gate.md` (alignment PASS, median shift 1.49 m) |
| 1 | Bicubic baseline | ✅ | `reports/day1_gate.md` (38.4717 dB, 1199 val patches) |
| 1 | Benchmark (opensr-test) | ✅ | `reports/day1_gate.md`, commit `06a7e14` |
| 2 | Model beats bicubic | ✅ (Run A is 1.52M params, over budget) | `reports/day2_runA.md` |
| 2 | Saved model | ✅ local only | `runs/runA/best.pt` (gitignored); `reports/hf_model_card.md`; HF push blocked |
| 2 | Delhi Sentinel-2 test | ✅ qualitative, no ground truth | `docs/PROGRESS.md` Day 2 + BOA addendum |
| 2 | GeoTIFF inference | ✅ | `scripts/validate_sr.py` 10/10 (PROGRESS.md Day 2 §2) |
| 3 | Overfitting fix | ✅ | 12k runs + `set_epoch` fix; `reports/day3_ab_summary.json` → `overfitting` |
| 3 | Freeze config | ✅ | `configs/frozen_day3.yaml` (hash re-verified by C1) |
| 3 | Cache expansion | ✅ time-boxed, 3,261 of 4,409 | commit `ccf6d0f`; C2 below |
| 3 | opensr-test | ✅ | `reports/day3_opensr_derisk.md`; `requirements.txt` opensr-test==1.3.3 |
| 3 | Spectral loss | ✅ built; did not help | `src/losses/spectral.py`; `reports/day3_results.md` |
| 3 | A2 vs B comparison | ✅ winner A2 | `reports/day3_ab_metrics.csv`, `reports/checkpoints/day3_p1_results.md` |
| 3 | Uncertainty/TTA start | 🟡 in progress on mvp/night-build | commits `c183fa6`, `891c8e0`; `reports/day3_tta_calibration_runA_best.md` |
| 3 | React/MapLibre skeleton | ✅ mocked API | commits `9c6f164`, `34e80f2`; `reports/checkpoints/day3_p3_web.md` |
| 4 | Run C | ⏭️ | — |
| 4 | Collapse watch | ⏭️ | — |
| 4 | ONNX export | ⏭️ | `src/export/` holds only `model_card.py` |
| 4 | CPU inference (deployment path) | ⏭️ | eager PyTorch CPU only (Day 2: 1.91 s/tile) |
| 4 | Uncertainty validation | ⏭️ | TTA calibration exists for Run A only, not Run C |
| 4 | FastAPI | ⏭️ | no backend in repo; `frontend/src/api/client.ts` is mocked |
| 4 | Frontend connection | ⏭️ | — |
| 5 | Delhi AOI end to end | ⏭️ | — |
| 5 | Swipe | ⏭️ | slider exists against mock data (`dd9420f`) |
| 5 | Uncertainty overlay | ⏭️ | — |
| 5 | GeoTIFF download | ⏭️ | — |
| 5 | QGIS check | ⏭️ | — |
| 5 | Metrics panel | ⏭️ | panel placeholder only (`9c6f164`) |
| 6 | Final eval bicubic/A/A2/B/C | ⏭️ | — |
| 6 | Params + CPU latency | ⏭️ | — |
| 6 | Ringing / flat-quartile HF | ⏭️ | tooling exists: `scripts/artefact_metrics.py` |
| 6 | US→Delhi domain shift | ⏭️ | — |
| 6 | Failure cases | ⏭️ | — |
| 6 | Cloud gating / OOM fallback / AI warning | ⏭️ | — |
| 7 | Write-up, README, cleanup | ⏭️ | — |
| 7 | Backup video, judge Q&A, 110-s rehearsal | ⏭️ | — |

## 4. Key numbers (confirmed)

- Full val split = 300 tiles → 1,199 patches (`reports/day1_gate.md`).
- Bicubic: PSNR 38.472, SSIM 0.8825, LPIPS 0.4033 (`reports/day3_ab_metrics.csv`).
- Run A (EDSR 16×64, 1,518,724 params, 40k it): PSNR 38.826 (+0.354), SSIM 0.8896, LPIPS 0.3219 (−20%) (`reports/day2_runA.md`).
- Day 3 model: 16 blocks × 48 features, 855,652 params (`configs/frozen_day3.yaml`).
- The table below uses the full 1,199-patch val split, it12000 for every run (`reports/day3_ab_metrics.csv`).

  | row | PSNR | LPIPS | consistency L1 (opensr) | L1_spec (area) | blur index |
  |---|---|---|---|---|---|
  | bicubic | 38.472 | 0.4033 | 0.00217 | 0.00121 | — |
  | A2 (0/0) | 38.871 | 0.3322 | 0.00320 | 0.00276 | — |
  | B1 (0.1/0.02) | 38.765 | 0.3452 | 0.00213 | 0.00140 | 0.847 |
  | B2 (0.3/0.06) | 38.559 | 0.3558 | 0.00134 | 0.00020 | 0.748 |

- A2 in-loop val (400 patches; never compare with the full split): best PSNR 35.303 @10.5k, final 35.298 (`reports/day3_ab_summary.json`).
- Day 3 runs: commit `fb8659d`, clean tree, hash `e6082c69…` (`runs/day3/*/run_metadata.json`).
- Cache: 3,261 pairs; `min_correlation` 0.9 → ceiling 4,409 of 8,000 (`configs/base.yaml` `sen2naipv2`).

## 5. Key files

| path | purpose |
|---|---|
| `AGENTS.md` | Working agreement (hard constraints, coding rules, worktrees) |
| `docs/PROGRESS.md` | Dated, append-only progress log |
| `configs/base.yaml` | Single source of defaults |
| `configs/frozen_day3.yaml` | Frozen training config, hash-verified by `src/config.py::load_frozen` |
| `configs/kaggle_jobs.yaml` | Kaggle job definitions (runa, day3) for `scripts/kaggle_run.py` |
| `src/train.py` | Trainer (`python -m drishtisr.train`) |
| `src/models/edsr.py` | EDSR backbone |
| `src/uncertainty.py`, `src/metrics/logvar.py` | Uncertainty head, NLL, TTA fallback, calibration |
| `src/losses/spectral.py` | Spectral-consistency loss (area ×4 downsampler) |
| `src/eval/ckpt_selection.py` | Pre-registered post-hoc checkpoint selection |
| `src/infer/tiled.py` | The one inference path (tiled, Hann-blended, GeoTIFF) |
| `outputs/splits_sen2naipv2.csv` | Train/val/test ids (gitignored, local) |
| `runs/day3/` | A2/B1/B2 metadata, logs, summaries (weights gitignored) |
| `reports/day3_results.md`, `reports/day3_ab_metrics.csv` | Day 3 headline results |
| `reports/day3_data_validity.md` | Data-validity verdict (b) |
| `reports/checkpoints/` | Per-session checkpoint reports |
| `frontend/` | Vite + React + MapLibre demo UI (mocked API) |

## 6. Open questions & risks

- **C1 (2026-09-11):** `load_frozen()` verified the hash `e6082c69bba3…fc0a`, so the frozen config is intact.
- **C2 (2026-09-11):** the cache holds 3,261 pairs; `outputs/manifest_sen2naipv2.csv` has 3,000 rows. The manifest is dated 2026-09-07 and was not regenerated after the 2026-09-09 expansion. Its 3,000 rows equal the first 3,000 rows of the splits file (2,400 train / 300 val / 300 test), and all 3,000 are cached. The other 261 cached pairs are:
  - 259 expansion pairs, all train (splits rows 3,000–4,408). 1,150 split ids are still uncached.
  - 2 stray pairs at catalog rows 0–1, with correlation 0.887 and 0.893. They are below 0.9 and not in the splits file, so unused.

  Usable cached train = 2,659. The fix, not applied, is to re-run the manifest.
- **Operator question (partly answered).** The training spectral loss uses `area` (an exact 4×4 block mean), per `frozen_day3.yaml` `loss.spectral_downsample`, `runs/day3/b2/args.json` and `src/losses/spectral.py`. B2's area L1 of 0.00020 against its opensr 0.00134 fits "B fits its own operator". Verification needed: opensr-test's internal degradation operator is not recorded in the repo.
- **Blur-measure disagreement.** HF energy says B is blurrier (0.087 → 0.083 → 0.081 × GT); Sobel says it is sharper (0.449 → 0.473 → 0.497). This may be edge overshoot, to be tested by the Day 6 ringing measurement.
- Only 2 checkpoints per Day 3 run survived; the trainer was fixed in `891c8e0` (`reports/checkpoints/day3_p1_results.md`).
- HF push is blocked: `scripts/push_to_hf.py` reads `HF_TOKEN`, which is unset in this shell.
- The 1.52M-param Run A is over budget and is not a controlled comparison.
- **Contradictions with the session brief (the repo wins):**
  - The GPU is T4, not P100 (`configs/kaggle_jobs.yaml` day3 `accelerator: NvidiaTeslaT4`). AGENTS.md §1 still says P100.
  - Val is 300 tiles / 1,199 patches, fixed since Day 1, not "frozen at Day 2".
  - `src/drishtisr/infer/tiled.py` does not exist; the path is `src/infer/tiled.py`.
  - The repo's Run A confounds (`reports/day3_results.md` fn 1, `src/train.py:626`) are architecture, no augmentation, weight decay 1e-4 vs 0, and frozen crops. It does not list dataset size. Verification needed: A2's train-set size.
  - `reports/day3_data_validity.md` §4 words verdict (b) differently from the approved wording in §2 above.
- Verification needed:
  - The Kaggle cache size of ≈3.62 GB. The local `.npz` total is 4.22 GB (3.93 GiB).
  - INT8 ONNX ≈1 MB. Nothing is exported yet.
  - The Kaggle quota reset at 2026-09-12 05:28 IST. It is not in the repo.

## 7. Next action

Night MVP build in progress on branch mvp/night-build; P12 merges it into main.

## 8. How to update this file

- Re-read this file right before editing; other sessions edit it too.
- Edit only the rows and bullets your task touched; keep every section.
- Stay ≤150 lines. Cite evidence paths or commits, never paste logs.
- Dated narrative goes in `docs/PROGRESS.md`, not here.
