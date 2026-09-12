# MVP build rules (binding for P4–P12)

## Scope
- Work only inside D:\SIH\DrishtiSR-mvp. Never write to D:\SIH\DrishtiSR (P12 excepted).
- Main-tree artifacts (checkpoints, data cache, runs, Kaggle outputs) are READ-ONLY; use absolute paths from docs/mvp/inventory.md.
- Create/edit only files listed in your prompt's "Owns" line. Forbidden to edit: training loop/scripts, data pipeline, losses, metrics module, models/edsr.py, infer/tiled.py, configs/frozen_day3.yaml, split ID files, scripts/kaggle_run.py, gitmeta, kernel folders. Import them instead. If an edit seems unavoidable, write it as a patch under docs/mvp/patches/ and work around it.

## Data discipline
- Val/test ID lists are frozen. Anything fitted (quantisation calibration, init statistics, training) uses TRAIN IDs only. VAL may be used for measurement, monitoring and reporting. TEST is touched only by P9 sample building (display only) and by P7 with --final.
- Degradation operator, bicubic, metrics, normalisation, HF/Sobel ratios: import the project's existing implementations from the inventory; never create a second version. If NOT FOUND, use the default in your prompt and log it.

## Python & packages
- Run D:\SIH\DrishtiSR\.venv\Scripts\python.exe (the shared venv, Python 3.11.9) with the working directory D:\SIH\DrishtiSR-mvp and PYTHONPATH=D:\SIH\DrishtiSR-mvp — the worktree ROOT, not <worktree>\src: the importable packages are `src` and the root-level alias `drishtisr` (drishtisr/__init__.py points its __path__ at src/), so src/drishtisr/... in prompts means src/... here. There is no editable install, so no runner shim is needed (verified: drishtisr.__file__ = D:\SIH\DrishtiSR-mvp\drishtisr\__init__.py). Every python invocation uses it. Verify once per session that drishtisr.__file__ is inside D:\SIH\DrishtiSR-mvp.
- Before any pip install, run it with --dry-run; abort that install and report if it would upgrade/downgrade torch, numpy, scipy, onnx, onnxruntime, lpips or opensr-test. Never uninstall.
- CPU only. torch.set_num_threads(6); onnxruntime intra_op_num_threads=6, inter_op 1. DataLoader num_workers=0 on Windows.

## Git
- Stage explicit paths only (never `add -A` / `add .`). Commit message prefix "[mvp/Pn]". No Claude attribution, no Co-Authored-By trailers.
- If .git index.lock exists: wait 10 s, retry, max 6 times, then report.
- No pushes to GitHub, Kaggle or Hugging Face (P12 has a conditional Kaggle exception). Always `python -m kaggle`, never bare `kaggle`.

## Evidence
- Every reported number is also written to JSON under reports/mvp/ with checkpoint_id, git SHA, split, n_samples, timestamp.
- Numbers from the Run A checkpoint carry "interim": true.
- Latency numbers carry "provisional": true unless CPU load was < 20% before benchmarking (record the load).

## Autonomy & reporting
- Don't stop to ask. Ambiguity → the default stated in your prompt; if none, the most conservative option; log one line in docs/mvp/decisions.md (prompt | decision | reason) and continue.
- No code echoes in chat. Final report ≤ 15 lines: files created, tests passed/failed, key numbers, decisions logged, blockers.

# Day 3 outcome (binding; overrides any conflicting line in P4–P12)

## Facts
- Day 3 is COMPLETE. Runs A2 (λ 0/0), B1 (0.1/0.02), B2 (0.3/0.06): 12,000 iterations, 855,652 params, executed at commit fb8659d. Evidence (full 1,199-patch VAL): reports/day3_ab_metrics.csv, reports/day3_ab_summary.json, reports/day3_val_curves.png.
- Pre-registered verdict: the training-time spectral loss did NOT help. B1 LPIPS +3.9% vs A2 (limit 2%), blur index 0.847 (< 0.90); B2 blur index 0.748. Blur index = (HF_B − HF_bicubic) / (HF_A2 − HF_bicubic); the HF energy ratio governs, Sobel is reported alongside.
- Winner = A2, checkpoint D:\SIH\DrishtiSR\runs\day3\a2\last.pt. Do not re-select across runs.
- Data verdict (b). Use exactly this sentence wherever data provenance is described: "Real Sentinel-2 L2A 10 m inputs (B04, B03, B02, B08) paired with 2.5 m NAIP targets that the SEN2NAIPv2 authors co-registered and radiometrically harmonised to Sentinel-2 (crosssensor subset). Inputs are not synthetically degraded; matching per-band statistics between inputs and targets are expected from this harmonisation."

## Substitutions
- Wherever a prompt says "Run A", "interim checkpoint", "--interim" or "interim": true for the served or evaluated model, use A2 with interim=false. Run A appears only as the † table row and in the loader test.
- Within-run checkpoint selection uses the repo's existing pre-registered rule (lowest spectral-consistency error, tie-break LPIPS). Do NOT apply sel-v1's 0.85 vs-GT ratio thresholds: measured HF/GT ratios are 0.06–0.09 for every model including bicubic, so that guard rejects everything.
- Bicubic/A2/B1/B2 VAL table numbers come from reports/day3_ab_metrics.csv; do not re-evaluate them. Cite fb8659d as the run commit.
- Headline spectral-consistency numbers cite the opensr-test measure in that CSV. On-screen consistency maps use the training degradation operator and are labelled "(training operator)".
- A2 already is the 16-block/48-feature architecture: skip any random-init "arch16x48" export or benchmark.

## Plan changes
- P8 is cancelled (superseded by reports/day3_data_validity.md). Day 3 P2 is cancelled (superseded by P5 + P6).
- Day 4 "Run C" = post-hoc heteroscedastic Laplace head on the FROZEN A2 backbone (P6). No spectral term in training.
- Novelty 1 on screen = consistency map + card, plus P5's consistency projection, served only if reports/mvp/projection_gate.json has passed=true.
- The MVP must not depend on Kaggle. Only P12 may push: to Kaggle (after 2026-09-12 05:28 IST, head-training fallback only) and to GitHub main (after the merge and passing tests).
- frontend/ (React + MapLibre skeleton from Day 3) is out of scope tonight: never modify it. P10 may reuse its labels and colours.
- If an optional or add-on step would push your session past ~75 minutes, skip it, log it in docs/mvp/decisions.md, and finish the core deliverables.
