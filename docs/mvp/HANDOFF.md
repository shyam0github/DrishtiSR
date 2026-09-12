# MVP handoff (P11 → P12), 2026-09-12

## State
- Worktree `D:\SIH\DrishtiSR-mvp`, branch `mvp/night-build` (main tree `D:\SIH\DrishtiSR` on `main`, untouched).
- Python: `D:\SIH\DrishtiSR\.venv\Scripts\python.exe`, cwd and `PYTHONPATH` = worktree root.
- Launch: `D:\SIH\DrishtiSR\.venv\Scripts\python.exe scripts/serve.py`. It uses the first free port from 8000 and prints the backend, checkpoint_id and interim flag.
- Served model: A2 `a2-last-dce224ec` (`D:\SIH\DrishtiSR\runs\day3\a2\last.pt`, 855,652 params), backend **onnx-fp32**, interim=false, projection off, uncertainty = TTA-4 by default.
- Tests: `tests/mvp/test_e2e.py` 7/7 pass (real server over HTTP; live responses match `app/fixtures/*.json` in keys, types and nullability). `test_api.py` + `test_ui_static.py` 23 pass (latency test deselected, so `api_latency.json` was not overwritten).

## Done: commit per prompt
- P0 `69d6e5f`: interfaces, inventory, rules, API contract
- P5 `90e0efa`: trust maps, rendering, display scales · `c5c2d52`: projection add-on, gate FAILED
- P10 `4978e05`: web UI + mock fixtures
- P4 `e251e0c`: ONNX FP32 export, INT8 ladder (all 4 rungs FAILED), CPU bench
- P7 `eef735b`: post-hoc selection, tables, headline.json
- P6 `941a509`: learned Laplace head machinery · `00aa18b`: training c1 (2000 it, watch **passed**), VAL calibration
- P9 `97966b3`: FastAPI backend, samples builder
- P11: the `[mvp/P11]` commit after `00aa18b` (see `git log`): serve.py, E2E test, README MVP section, HF Space files, demo script, this file, the DRISHTI_UNC_CKPT opt-in, VAL tables rerun

## Numbers and their status
- Final (full VAL, n=1199, run commit fb8659d, from `reports/day3_ab_metrics.csv`): A2 LPIPS 0.3322 vs bicubic 0.4033 (−17.6 %). opensr consistency: A2 0.00320, bicubic 0.00217, HR reference 0.00478.
- Uncertainty (VAL, n=128, `reports/mvp/uncertainty_table.md`): AUSE/ρ are TTA-4 0.0019/0.525, TTA-8 0.0018/0.542, learned Laplace 0.0027/0.379. The learned head's coverage is calibrated (50 % → 0.533, 90 % → 0.904), but its ranking is worse than TTA's.
- Latency: `bench_a2-last-dce224ec.json` has provisional=false (FP32 median 1220 ms, 256² Delhi window, 6 threads). **Provisional:** `api_latency.json` (CPU load 47 % before the run, provisional=true).
- TEST-split numbers: none exist (`--final` never run). Test tiles appear only as display samples.

## Gates
- Projection gate: **FAILED** at iters 1/2/3 (LPIPS ×1.057–1.069 > 1.01; HF ×0.90–0.91 < 0.95). Not served.
- INT8 quant gate: **FAILED** on all 4 rungs → onnx-fp32 is served. The 981 KB INT8 graph is benched but never served.
- P6 collapse watch: **passed** (ρ 0.21 → 0.37, frac_floor 0, spatial CV 0.46 at 2000 it). Head: `runs/mvp/unc_c1/head_last.pt` (`unc_c1-head_last-19c2bd0a`, gitignored).

## Next: what P12 must decide
- **Default uncertainty: TTA-4 or learned head.** The opt-in `DRISHTI_UNC_CKPT=D:\SIH\DrishtiSR-mvp\runs\mvp\unc_c1\head_last.pt` works (contract-valid, 0 ms extra, torch-fp32 only). But the AUSE/ρ ranking is worse than TTA's, and the overlay saturates, because `unc_display_max` (0.0037) is the TTA-8 std scale while the head's b averages ~0.009 (p95 0.018). If the head becomes the default: add a per-method display max (a contract change, or a second trust_scales key) and an ONNX export of `forward_packed`.
- **UI copy:** the "What's novel" panel (app/static/index.html) still says "INT8", which is not served. Change it to "ONNX, CPU-only", or pass the INT8 gate first.
- Merge `mvp/night-build` → main after the full `tests/mvp` run. Clear `reports/mvp/p0_smoke.json` (dirty since P0, not P11's), then push per the RULES exception.
- HF Space: publish by hand per `deploy/hf_space/README.md` (needs HF_TOKEN). Nothing was pushed.

## Gotchas
- `app/samples/` (30.7 MB) is gitignored. A fresh clone must run `scripts/mvp/build_samples.py`, which needs the main-tree cache and `data/delhi`. Without it, E2E/API tests SKIP.
- `checkpoint_id` hashes the `.pt`, and OnnxPredictor locates the graph by that id. The ONNX backend therefore still needs the `.pt` file present (the Space image stages it at `ckpt/a2/last.pt`).
- `artifacts/` and `runs/` are gitignored. The ONNX graphs and the head checkpoint live only in this worktree.
- The server is same-origin only (no CORS). A POST whose Origin host ≠ Host gets 400, so behind a proxy the Host header must be forwarded.
- One job at a time (Engine lock). TTA-8 on 128² takes ~2.5 s at 6 threads, so expect ~3× that on a 2-vCPU Space.
- Delhi samples use `(dn − 1000)/10000`, because they are raw L2A DN with the baseline-05.11 offset. For uploads of the same kind, pick "DN ÷ 10000 with −1000 offset".
- Docker: the Space image is **not build-tested**. The staging steps in `deploy/hf_space/README.md` were run (47.9 MB staged), but the Docker Desktop daemon was not running. Before publishing, run `docker build` + `docker run -p 7860:7860` once.
