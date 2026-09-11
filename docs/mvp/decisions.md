# MVP decision log

prompt | decision | reason
---|---|---
P0b | Package paths `src/drishtisr/<pkg>` read as `src/<pkg>`; predictor written to `src/infer/predictor.py` | The repo has no `src/drishtisr/`; `drishtisr` is a root alias package mapping to `src/` (drishtisr/__init__.py).
P0b | Python mechanism = venv interpreter + cwd/PYTHONPATH = worktree root; no `scripts/mvp/_py.py` | `PYTHONPATH=<worktree>/src` cannot import `drishtisr` (alias lives at root); no editable install exists, so the root path already resolves inside the worktree.
P0b | TorchPredictor runs `tiled.sr_array`; inputs ≤ tile use one tile of their own size | Eval never calls tiled.py (it calls `model(lr)`); a single unpadded tile makes the predictor bit-identical to eval (measured max diff 0.0), larger inputs tile with cfg.frontend.tile (256/32).
P0b | `info.model_bytes` = fp32 weight bytes (3,422,608 for A2), not the .pt file size | The .pt also stores optimiser/scaler state (10.3 MB); weight bytes are comparable to an ONNX file size.
P0b | Scale-head models: run sr_array twice (mean, then scale); b = sqrt(exp(logvar)/2) | tiled.py sizes its output from the input channels and cannot carry 8 channels; editing it is forbidden. Conversion matches a Laplace to the Gaussian head's variance; P6 may register its own predictor.
P0b | Smoke VAL patch = first cached VAL id in split-CSV order, 64-px centre crop, read from the npz cache directly | The dataset catalog load needs the network (tacoreader); cfg.patches.lr_size is 64.
P0b | Main-tree root found in tests via `git rev-parse --git-common-dir`; tests SKIP (with path) when artefacts are absent | AGENTS.md forbids hardcoded paths and requires tests to run without data.
P0b | Fixture metrics: with_gt from Day 3 CSV split means (A2, bicubic); other numbers placeholders; a FIXTURE warning string is included | The contract has no fixture flag; the warning keeps a UI screenshot from being mistaken for a measurement.
P0b | Fixture refs unc_display_max = cons_display_max = 0.02 reflectance | Placeholder above the 0.0057 consistency floor; P10/P5 own the real values.
