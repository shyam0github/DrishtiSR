# DrishtiSR — working agreement

Smart India Hackathon 2026, problem statement **SIH26142**: deep-learning
super-resolution of Sentinel-2 imagery from 10 m to 2.5 m (x4).

These constraints apply to **every file in this repository**. They are not
suggestions and they are not negotiable per-file. If a change cannot satisfy
them, say so instead of working around them.

---

## 1. Hard constraints

### Compute

- **Training happens only on a Kaggle P100 notebook.** Budget: 30 GPU-hours per
  week. Assume every training run is expensive and irreversible; make runs
  resumable and log enough that a run never has to be repeated to answer a
  question.
- **The local machine has no working CUDA.** All local execution is CPU-only,
  Windows 11. Anything that must run locally — tests, data inspection, export,
  benchmarking, figure generation — must complete on CPU.
- Code must never assume a GPU exists. Select the device from availability, and
  make CPU the working default path, not a degraded fallback.

### Python versions — a known local/remote divergence, watch for it

| where | version | notes |
|---|---|---|
| **local project venv** `D:\SIH\DrishtiSR\.venv` | **3.11.9** | The only working local environment. |
| Kaggle notebooks | **3.11** | Matches the venv at `MAJOR.MINOR`. |
| local system `py -3.11` | 3.11.9 | **Bare.** No torch, omegaconf, pytest, or kaggle. |
| local `py -3.14` | 3.14.3 | Has torch but **not** omegaconf. Not a project env. |

MEASURED on this machine, and the reason this table exists:

- `py -3.11 -m pytest` — which earlier versions of this file prescribed — **does
  not work**. That resolves to
  `C:\Users\shayamji\AppData\Local\Python\pythoncore-3.11-64\python.exe`, which
  has none of the dependencies. It fails with `No module named pytest`, which
  reads like a broken repo rather than the wrong interpreter.
- Three interpreters answer to some form of "python" here. Two of them are
  traps. Package installs land in whichever one was invoked, which is how
  `kaggle` ended up installed under 3.14 while the project venv could not
  import it.

**Therefore, always invoke the venv interpreter explicitly**, and never a bare
`python`, `py`, or a console script from `Scripts/` that may belong to a
different interpreter:

```
D:\SIH\DrishtiSR\.venv\Scripts\python.exe -m pytest
D:\SIH\DrishtiSR\.venv\Scripts\python.exe -m pip install <package>
D:\SIH\DrishtiSR\.venv\Scripts\python.exe scripts/<script>.py
```

The same rule applies inside the code: subprocess calls to Python tooling use
`[sys.executable, "-m", "<module>"]`, never a bare executable name. See
`scripts/kaggle_upload.py::kaggle_command` — the bare `kaggle` executable is not
on PATH here, and `python -m kaggle` through `sys.executable` is what makes the
tool work regardless.

The version gap that actually matters is small (both sides are 3.11), so
language-level incompatibility is not the risk. **The risk is running local code
with the wrong local interpreter** and concluding the code is broken.

### Model budget

- **≤ 1,000,000 parameters.** Total, including every head. Assert it in code
  rather than trusting a mental count; `cfg.runtime.max_parameters` holds the
  limit.
- **Deployment target is INT8-quantised ONNX inference on 6 CPU threads.** Every
  architectural choice is subject to this. Operators that do not quantise
  cleanly, or that ONNX Runtime does not accelerate on CPU, are disqualified no
  matter how good the PSNR is.

### The three technical contributions — protect these architecturally

1. **Spectral consistency.** Degrading the SR output back to 10 m must reproduce
   the input reflectance. Therefore:
   - **Reflectance values stay physically meaningful end to end.**
   - **Never normalise with ImageNet statistics.** Not for the backbone, not for
     a perceptual loss, not "just to stabilise training". Reflectance is a
     physical quantity, not an RGB photograph.
   - **Never clip silently.** Surface reflectance legitimately exceeds 1.0 over
     cloud, snow, specular water, and bright roofs. If a clip is genuinely
     required, it must be explicit, configurable, logged, and justified in the
     docstring.
   - Scaling between digital number and reflectance uses
     `cfg.dataset.reflectance_scale`; the nominal range in config is for
     assertions and reporting, **not** for clamping.
2. **Per-pixel uncertainty / hallucination map**, via a heteroscedastic NLL head.
   The uncertainty output is a first-class deliverable, not a diagnostic — it
   must survive export and quantisation alongside the SR output.
3. **CPU-only efficient deployment.** Latency, thread count, and quantised
   accuracy are reported results, measured, not asserted.

### Portability

- **The repo must work identically whether the data root is a local `D:` path or
  `/kaggle/input`.** Resolution goes through `src/utils/paths.resolve_data_root`
  — nothing else may decide where data lives.

---

## 2. Coding rules

- **Never write logic inside notebooks.** Notebooks only import from `src/` and
  call functions. If you find yourself writing a loop, a model definition, or a
  transform in a cell, it belongs in `src/` and the cell calls it. Notebooks are
  disposable; `src/` is the project.
- **Every entry-point script accepts `--config` and `--smoke`.** `--config`
  points at a YAML merged over `configs/base.yaml`. `--smoke` merges the
  `smoke` block and must make the script complete end to end in a couple of
  minutes on CPU, so every path is exercised before a GPU-hour is spent.
- **Every function that touches imagery documents, in its docstring:**
  - expected **dtype** (`uint16`, `float32`, ...),
  - expected **shape**, stating the axis order explicitly — `(C, H, W)` vs
    `(H, W, C)` — never just "an image array",
  - expected **value range**, and where reflectance is involved, **stated
    explicitly as reflectance** (e.g. "surface reflectance, float32, nominally
    [0, 1] but unclipped; bright targets exceed 1.0") rather than "normalised".
  - The same applies to what it returns.
- **No silent exception handling. Fail loudly.** No bare `except:`, no
  `except Exception: pass`, no returning `None`/zeros/an empty tensor on error,
  no "skip the bad tile and carry on" without an explicit, logged, configured
  policy. A crash costs minutes; a silently corrupted training set costs the
  submission. When re-raising, chain with `raise ... from exc`.
- **No hardcoded paths anywhere.** Every path comes from config, resolved
  against `repo_root()` or `resolve_data_root()`.
- **No hardcoded hyperparameters, band names, or scale factors in code.** They
  live in `configs/base.yaml`.
- **Seed from config** via `seed_everything(cfg.seed)` at the top of every entry
  point. Do not pass literals.
- **Log through `get_logger()`**, not `print`, in anything under `src/` or
  `scripts/`.

---

## 3. Layout

```
configs/       OmegaConf YAML. base.yaml is the single source of defaults.
src/data/      Dataset construction, degradation pipeline, patch sampling.
src/models/    SR backbones and the heteroscedastic uncertainty head.
src/losses/    Reconstruction, spectral-consistency, and NLL objectives.
src/metrics/   Reflectance-domain quality and spectral-fidelity metrics.
src/eval/      Evaluation harnesses, ablations, figure generation.
src/export/    ONNX export, INT8 quantisation, CPU benchmarking.
src/utils/     paths.py, seed.py, logging.py.
scripts/       Entry points. Each takes --config and --smoke.
notebooks/     Kaggle notebooks. Import from src/ only.
tests/         pytest. Runs on CPU, no data required.
reports/       Write-ups and submission material.
outputs/       Gitignored: figures/, metrics/, checkpoints/, run.log.
```

Run tests with `D:\SIH\DrishtiSR\.venv\Scripts\python.exe -m pytest` from the
repository root. NOT `py -3.11 -m pytest` -- see the Python-versions table
above; that interpreter exists but has none of the dependencies.
