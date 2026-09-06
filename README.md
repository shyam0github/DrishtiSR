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

**Kaggle jobs run headless.** `scripts/kaggle_run.py` generates a notebook pinned
to a git SHA, pushes it, watches it, and pulls the outputs — no browser. Two
guards are non-negotiable: a push is refused from a dirty or unpushed working
tree (Kaggle clones from GitHub, so uncommitted work would silently not be in the
run), and every GPU job runs the data-root check before the expensive work and
aborts if the mount is unreadable. GPU jobs pin a **T4**, never a P100: the
default image's cu128 PyTorch has no Pascal `sm_60` kernels, so a P100 session
reports `cuda.is_available() == True` and then dies on the first CUDA op.

**Next:** SR backbone + heteroscedastic uncertainty head under the 1M-parameter
budget, then ONNX export and INT8 CPU benchmarking.
