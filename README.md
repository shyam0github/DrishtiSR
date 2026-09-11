# DrishtiSR

Deep-learning super-resolution of Sentinel-2 imagery, **10 m → 2.5 m (x4)**.

Smart India Hackathon 2026, problem statement **SIH26142**.

---

## What makes this submission different

Three things, and every architectural decision in the repo is subordinate to them:

1. **Spectral consistency.** Degrading the super-resolved output back to 10 m must
   reproduce the input reflectance. Reflectance stays a physical quantity end to
   end — no ImageNet normalisation anywhere, no silent clipping. Bright targets
   (cloud, snow, specular water, roofs) legitimately exceed 1.0 and are allowed to.
2. **Per-pixel uncertainty.** A heteroscedastic NLL head emits a hallucination map
   alongside the SR output. It is a deliverable, not a diagnostic, and it survives
   ONNX export and INT8 quantisation.
3. **CPU-only efficient deployment.** ≤ 1M parameters, INT8 ONNX, benchmarked on
   6 CPU threads. Latency and quantised accuracy are measured and reported, never
   asserted.

## Constraints this repo is built under

| | |
|---|---|
| Training | Kaggle P100 notebook, 30 GPU-hours/week. Runs must be resumable. |
| Local machine | Windows 11, Python 3.11, **no CUDA**. Everything local runs on CPU. |
| Parameter budget | ≤ 1,000,000 total, asserted in code against `cfg.runtime.max_parameters`. |
| Deployment | INT8-quantised ONNX, ONNX Runtime, 6 CPU threads. |
| Data root | Resolved *only* through `src/utils/paths.resolve_data_root` — works identically on a local `D:` path and on `/kaggle/input`. |

The full working agreement lives in the repository root.

---

## Layout

```
configs/       OmegaConf YAML. base.yaml is the single source of defaults.
src/data/      Dataset construction, degradation pipeline, patch sampling.
src/models/    SR backbones and the heteroscedastic uncertainty head.
src/losses/    Reconstruction, spectral-consistency, and NLL objectives.
src/metrics/   Reflectance-domain quality and spectral-fidelity metrics.
src/eval/      Evaluation harnesses, ablations, figure generation.
src/export/    ONNX export, INT8 quantisation, CPU benchmarking.
src/utils/     paths.py, seed.py, logging.py, config.py.
scripts/       Entry points. Each takes --config and --smoke.
notebooks/     Kaggle notebooks. Import from src/ only — no logic in cells.
notebooks/templates/  Sources for generated notebooks. Never edit a generated one.
docs/          Operational guides, e.g. kaggle_workflow.md.
tests/         pytest. Runs on CPU, no data required.
reports/       Write-ups and submission material.
frontend/      Demo UI: Vite + React + MapLibre. See frontend/README.md.
outputs/       Gitignored: figures/, metrics/, checkpoints/, cache/, run.log.
```

## Setup

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate          # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running

Every entry point takes `--config` (a YAML merged over `configs/base.yaml`) and
`--smoke` (merges the `smoke` block; completes end to end in a couple of minutes
on CPU, so every code path is exercised before a GPU-hour is spent).

```bash
# Pre-flight on synthetic data -- no network, no dataset needed.
python scripts/prepare_data.py --config configs/base.yaml --smoke

# Populate the sample cache and write an inspectable manifest.
python scripts/prepare_data.py --config configs/base.yaml

# Geographic train/val/test split, verified against leakage before it is written.
python scripts/make_splits.py --config configs/base.yaml

# Co-registration audit. Ends in PASS / WARN / FAIL.
python scripts/qa_alignment.py --config configs/base.yaml

# The number every model result is quoted against.
python scripts/run_baseline.py --config configs/base.yaml
```

Kaggle runs are driven from the terminal — see
[docs/kaggle_workflow.md](docs/kaggle_workflow.md):

```bash
python scripts/kaggle_run.py jobs                      # what is defined
python scripts/kaggle_run.py push   --job train        # generate, push, start
python scripts/kaggle_run.py status --job train --watch
python scripts/kaggle_run.py logs   --job train        # tail-first
python scripts/kaggle_run.py fetch  --job train        # outputs/kaggle/<job>/<ts>/
```

Tests, from the repository root (453 tests, CPU-only, no data required):

```bash
python -m pytest          # inside the activated .venv
```

---

## Status

**Day 1 gate: PASS** — see [reports/day1_gate.md](reports/day1_gate.md).
3,000 `sen2naipv2-crosssensor` pairs cached and filtered at
`min_correlation >= 0.9`. Pairs are co-registered, the reflectance divisor is
correct, and no sample is lost to nodata.

Non-learned baselines over the validation split (1,199 patches, 4 bands
B04/B03/B02/B08, CPU, 6 threads):

| method | PSNR ↑ | SSIM ↑ | SAM° ↓ | ERGAS ↓ | LPIPS ↓ |
|---|---|---|---|---|---|
| bicubic | 38.47 | 0.8825 | 2.09 | 3.05 | 0.403 |
| nearest | 37.86 | 0.8671 | 2.24 | 3.28 | 0.329 |

Bicubic is the bar. A model that does not beat it convincingly has not earned its
place in the submission.

**Reproduced on Kaggle, headless, at a pinned commit.** `push --job baseline`
ran the same entry point on a fresh CPU session against the mounted cache and
agreed with the local numbers to ten decimal places — bicubic 38.471676 vs
38.471676, SSIM 0.882467, SAM 2.092455°, ERGAS 3.050261, LPIPS 0.403318, over
the same 1,199 validation patches. `nearest` came back bit-identical. The
residual on bicubic is ~1e-10, which is float non-determinism, not an
environment difference.

That comparison is only meaningful because the run **read** the split rather
than recomputing one: `Split read from .../splits_sen2naipv2.csv (3000 samples
assigned)`, with the manifest and split CSVs staged out of the mounted dataset
byte-for-byte identical to the local copies. See
[docs/kaggle_workflow.md](docs/kaggle_workflow.md) — that staging was an open
gap until this run, and a missing split file does not fail, it silently
recomputes.

**Benchmarked against `opensr-test` too, so the floor is not only our own
arithmetic.** ESA OpenSR's suite (Aybar et al., IEEE JSTARS 2024) is an
independent implementation that asks whether the detail a super-resolver adds is
actually present in the reference. Full validation split, 1,199 patches scored,
0 rejected:

| method | reflectance ↓ | spectral° ↓ | spatial ↓ | synthesis ↑ | halluc. ↓ | omission ↓ | improv. ↑ |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0022 | 0.515 | 0.006 | 0.0026 | **0.0808** | 0.8620 | 0.0572 |
| nearest | 0.0022 | 0.515 | 0.004 | 0.0042 | 0.1692 | **0.7408** | **0.0900** |

**Pixel replication scores better than bicubic on improvement and omission, and
worse only on hallucination** — its block edges partly coincide with real HR
boundaries and are counted as recovered detail. LPIPS above ranks the two the
same way, for the same reason. Two independent metric suites agreeing that
sharpness and correctness are different axes is the measured case for the
uncertainty head, and it means `im_metric` must never be read without
`ha_metric` next to it.

Getting there required guarding two upstream defects, both pinned by test:
feeding the library digital numbers instead of reflectance does **not** raise (it
inflates `reflectance` and `synthesis` by 10,000× while the other five are
bit-identical), and its README's own `Metrics(config=...)` example silently
discards the configuration and runs defaults.

**The LR is a real second sensor, not a downsample of the HR — checked, because
the numbers looked wrong.** `verify_data_root.py` reported LR and HR reflectance
extrema agreeing to four decimals on all four bands; across all 3,000 manifest
rows the LR and HR maxima are identical in 100% of cases. That is not a
reporting fault, so `scripts/audit_degradation.py` tested the pairs directly over
200 samples:

| test | synthetic LR would give | measured |
|---|---|---|
| best kernel fit vs stored LR | ≈ 90.8 dB (uint16 rounding ceiling) | **45.38 dB** median, 0/200 exact |
| per-tile spread, p5–p95 | ~0 dB (a fixed kernel is deterministic) | **13.15 dB** |
| std(LR) / std(HR) | < 1 (averaging destroys variance) | **1.0004** |

Verdict **CROSS_SENSOR**. The matching extrema come from the HR having been
radiometrically harmonised onto the LR — LR/HR quantiles agree to ~4 DN
uniformly, the signature of histogram matching, not of resampling. Provenance
confirmed against the live TACO catalogue: every record resolves to
`sen2naipv2-crosssensor.taco`, with `days_between` ∈ {−1, 0, +1}, i.e. two real
acquisitions within a day of each other. Unrelated tiles floor at 24.83 dB.

Two caveats recorded rather than buried: the catalogue's own `correlation` field
(median 0.927) does not match the measured LR/HR agreement (0.9966), so it is
computed on something else and should not be quoted as validation; and because
the HR radiometry was matched to the LR, part of the spectral-consistency
objective is satisfied by dataset construction rather than by the model.
`--smoke` runs the audit against the synthetic stub, whose LR *is* an exact block
mean, as a standing self-test that the detector can still detect a degradation.

**Kaggle jobs run headless.** `scripts/kaggle_run.py` generates a notebook pinned
to a git SHA, pushes it, watches it, and pulls the outputs — no browser. Two
guards are non-negotiable: a push is refused from a dirty or unpushed working
tree (Kaggle clones from GitHub, so uncommitted work would silently not be in the
run), and every GPU job runs the data-root check before the expensive work and
aborts if the mount is unreadable. GPU jobs pin a **T4**, never a P100: the
default image's cu128 PyTorch has no Pascal `sm_60` kernels, so a P100 session
reports `cuda.is_available() == True` and then dies on the first CUDA op.

**Demo UI skeleton, mocked end to end.** `frontend/` (Vite + React + MapLibre)
has the swipe comparison, an AOI picker with a tile budget, the uncertainty
overlay, and the Day 3 metrics table. All of it runs against fixtures behind one
typed client (`src/api/client.ts`, `MOCK_MODE`); Day 4 points that client at
FastAPI and changes nothing else. `npm run test:e2e` drives a real Chrome with a
mouse on a desktop viewport and with touch on a phone viewport, then reads back
the painted pixels. The 2.5 m side of the swipe carries **2.25×** the fine
detail of the 10 m side. That check exists because the first version passed on
DOM state over a blank map. Every SR view carries a non-dismissible
"AI-RECONSTRUCTED — NOT MEASURED DATA" badge.

Caveats, stated:
- The imagery is a procedural placeholder. Its 10 m layer is the exact 4×4
  block mean of the 2.5 m one, so it shows what 4× GSD means, not what the model
  does.
- The σ layer is an edge map, not the model's σ. Both are labelled on screen.
- The 36-tile AOI budget is sized to Delhi AOI A, not to a measured latency: the
  INT8 CPU benchmark does not exist yet.

**The fallback uncertainty raster is already shippable: TTA disagreement is
monotone against error.** Before training the heteroscedastic head (Run C), the
fallback was measured on Run A's checkpoint: run the model under all 8 dihedral
transforms, invert each exactly, and take the per-pixel std. Over the full
validation split (1,199 patches, 314 M pixel-band values), binned into 20
equal-mass bins of that std, MAE against the HR rises in **every one of 19
steps, in every band** (Spearman 1.000; the most-uncertain bin's MAE is 11.5×
the least's). See [reports/day3_tta_calibration_runA_best.md](reports/day3_tta_calibration_runA_best.md).

| predictor | monotone | top/bottom MAE | AUSE ↓ | AUSE / random ↓ |
|---|---|---|---|---|
| TTA std | 19/19 steps | 11.5× | **0.243** | **0.347** |
| SR gradient magnitude (control) | 19/19 steps | 9.5× | 0.266 | 0.379 |

Stated rather than buried: **most of that ranking is texture.** A free edge
detector on the SR output is also monotone and gets within 9% of TTA's AUSE, so
TTA's added information beyond "edges are hard" is real but modest. And the std
is not a calibrated sigma: RMSE runs 5–19× the std, with the ratio falling as
the std rises, so a single scale factor will not calibrate it. What it gives is
a ranking. 19.9% of pixel mass has an implied log-variance below the NLL head's
−10 clamp floor, so a healthy Run C head will put about a fifth of the raster
on that bound, and that alone is not collapse.

The head itself (`--uncertainty 1`), the NLL with its warmup, and the
sigma-collapse monitor are merged to `main`. They are off by default, and Day 3
checkpoints load into the head-enabled model bit-identically on the SR output.

**B1's −33% consistency gain is not bought with blur against the control, but
neither model adds much ×4-band detail.** Both logs measure the Sobel gradient
and the high-frequency energy fraction on the same 400 in-loop validation
patches as the GT and bicubic reference lines. frac = 0 means bicubic, 1 means
GT.

| run @ 12k | Sobel frac | HF-energy frac | full-split L1_spec ÷ floor |
|---|---|---|---|
| A2 (control) | 0.014 | 0.023 | 0.48× |
| B1 (λ 0.1/0.02) | **0.060** | 0.022 | 0.24× |
| bicubic | 0 | 0 | 0.21× |

B1 is above A2 on Sobel at 23 of 23 validations from 1k on, and 1.3% below on
HF energy. Neither curve drifts toward bicubic while `l1_spec` falls. But both
runs sit about 2% of the way from bicubic to GT in the ×4 band. B1 also pays
0.105 dB PSNR and +0.013 LPIPS against A2, both resolved. The spectral floor is
a reference point, not a lower bound: going below it is a warning sign for
over-smoothing, not an achievement (see
[reports/day3_spectral_floor.md](reports/day3_spectral_floor.md)).

Stated plainly (see [reports/day3_results.md](reports/day3_results.md)):
- Only 2 of 12 checkpoints per Day 3 run survived a trainer bug that wrote each
  one over `last.pt`.
- The selection rule chose between those two.
- A2 and B1 both selected the final iterate, so neither had converged at 12k.

**Pre-Day-4 fixes, in place and tested:**
- Every `--ckpt-every` checkpoint now persists as `ckpt_it<NNNNNN>.pt`.
- The NLL gradient stops at the SR output by default (`--nll-detach-sr 1`), at
  `--nll-weight 0.1`. The trainer logs the measured NLL-to-L1 gradient ratio at
  the SR output every validation: raw (the ~90×) and applied.
- `calibrate_uncertainty.py` reports every predictor's AUSE, a head's
  included, only beside the SR-gradient control and the delta.
- Parallel sessions work in git worktrees; see AGENTS.md section 4.

**Next:** Run C (heteroscedastic head), then ONNX export and INT8 CPU
benchmarking of both output rasters.
