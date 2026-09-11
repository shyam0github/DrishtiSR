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
P5 | VAL patch pool = all 300 VAL tiles (split-CSV order) × 2×2 grid of 64-px LR patches in the 128-px centre crop (cfg.sr.lr_patch_size, cfg.patches stride 64); draws = default_rng(26142).choice without replacement; npz cache read directly | Mirrors the val grid (≈4 patches/tile) without the network-bound catalog; one pool shared by calibration and projection eval.
P5 | TTA identity member reuses the single-pass SR; the other transforms are batched per transformed shape | Same deterministic computation; saves one forward pass. timings_ms.uncertainty covers only the extra passes + std.
P5 | spec_l1 / spec_sam_deg scalars from src.losses.spectral.spectral_terms; per-pixel cons_sam_deg map uses its numerics (eps inside norm, cos clamp), 0 where a spectrum norm < DEFAULT_MIN_NORM | No second operator; map needs per-pixel values that spectral_terms only reduces.
P5 | HF ratio = src.metrics.sharpness.hf_energy_ratio (cutoff 0.25); FFT fallback not needed | Found in inventory §7.
P5 | Sobel baseline map = per-pixel band-mean Sobel magnitude of bicubic using sharpness.SOBEL_X/Y + replicate pad | gradient_magnitude only returns the image mean; kernels imported, not redefined.
P5 | learned_laplace: timings_ms.uncertainty = 0.0; projection time (project_iters>0) folded into timings_ms.sr, no new key | Contract runtime_ms has exactly sr/uncertainty/total.
P5 | project_consistency + compute_trust(project_iters=0) shipped in the step-10 commit; eval/test/gate committed separately after | Default 0 keeps behaviour identical; splitting the function out would only churn trust.py.
P5 | Projection gate HF = mean over patches of hf_energy_ratio(x)/hf_energy_ratio(HR); gate on 200-patch VAL means | Same per-patch definition as Day 3 hf_ratio_vs_gt; the gate is relative between variants.
P5 | TTA timing on the 128-px centre crop of a VAL tile, median of 3 after warm-up; CPU load measured just before (13% → provisional=false) | RULES Evidence section.
P5 | No Co-Authored-By trailer on P5 commits | RULES.md Git section forbids it.
P5 | Projection gate FAILED at iters 1/2/3 (LPIPS ×1.057–1.069 > 1.01, HF ×0.90–0.91 < 0.95; spec_l1 ×0.16–0.02 passes); not served. opensr measure skipped (projected 740 s > 600 s budget) | Pre-registered gate, not tuned; RULES: projection served only if passed=true.
