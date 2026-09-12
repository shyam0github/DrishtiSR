# CHECKPOINT REPORT
## Overall status   ✅ COMPLETE

## Deliverables
| # | Deliverable | Status | Evidence I can check |
|---|---|---|---|
| 1 | Manual launch diagnosed and fixed | ✅ | `scripts/serve.py` exists. A bare `python` resolves to `D:\SIH\DrishtiSR\.venv-1\Scripts\python.exe` (3.14.3), which has no fastapi/uvicorn/onnxruntime/rasterio/torch. The launch failed with `ModuleNotFoundError: No module named 'uvicorn'` (serve.py:66). The project `.venv` (3.11.9) has every dependency, and `drishtisr.__file__` is `D:\SIH\DrishtiSR\drishtisr\__init__.py`. serve.py adds the repo root to sys.path itself and accepts `--host/--port/--no-browser`; the env vars are optional. ONNX artifacts and `app/samples/` are present. Fix: call the venv interpreter explicitly. No code change, no install. |
| 2 | docs/RUN_DEMO.md + run_demo.ps1 tested | ✅ | The `run_demo.ps1` fresh-process test served `/api/health` (onnx-fp32) on :8000 and the port was free after the stop. The port lookup and `Stop-Process -Id` lines were run on a server I owned. |
| 3 | UI wording fixed | ✅ | `app/static/index.html` "What's novel": the INT8 claim is replaced; the honest "not shipped" line is added. |
| 4 | PROJECT_STATE corrected | ✅ | §1 item 3; §3 Run C / Collapse watch rows; §6: the "Nothing is exported yet" line removed, and Export + Launch-trap bullets added. 130 lines. |
| 5 | Port check | ✅ | 8000, 8001 and 8002 were all free at the start (PID 16916 is gone). Used 8000. |
| 6 | End-to-end run | ✅ | `reports/mvp/e2e_p13.json` |
| 7 | Screenshot | ✅ (headless Chrome, already installed; Playwright absent) | `reports/checkpoints/p13_ui.png`: test-low-1 rendered with `?autorun=1&tta=0` |
| 8 | Tests + commit + push | ✅ | `tests/mvp` 81 passed in 72 s. The commit is the one adding this file (`git log -1 -- reports/checkpoints/p13_fix.md`). |

## Numbers / counts
- INT8 strings replaced: 1 (the only one in `app/static/`). The remaining "INT8" is the added honest line.
- Health: onnx-fp32, a2-last-dce224ec, 855,652 params, 3,441,761 B, 6 threads, interim=false. Startup 5.0 s.
- Samples found: 8. has_gt true 6, false 2 (delhi-urban, delhi-mixed).
- Upscale test-low-1, tta=4: HTTP 200, 6.60 s wall. SR, bicubic, uncertainty (tta4) and consistency images are present, and so is the SR GeoTIFF (GET 200, 2.80 MB). LPIPS SR 0.0455 vs bicubic 0.0525. Overlays present: 2/2.
- Upscale delhi-urban (no GT): HTTP 200, 6.94 s, with_gt null.
- Upscale test-low-1, tta=0: HTTP 200, 1.74 s (faster).
- Tests 81/81.

## Files/artifacts created or changed
`app/static/index.html`, `PROJECT_STATE.md`, `docs/RUN_DEMO.md`, `scripts/run_demo.ps1`, `reports/mvp/e2e_p13.json`, `reports/checkpoints/p13_ui.png`, `reports/checkpoints/p13_fix.md`. The test run rewrote `reports/mvp/api_latency.json` and `p0_smoke.json`; both were restored to HEAD rather than committed.

## Verification performed
I reproduced the failing launch exactly as the user runs it and traced it to the wrong interpreter. I started the demo through the new wrapper in a separate PowerShell process and confirmed it answered. I then ran the live API end to end: health, samples, three upscales and a GeoTIFF download. I rendered the real page in headless Chrome and looked at the screenshot. I stopped every server I started and confirmed ports 8000–8002 were free. Finally I ran the full MVP test suite.

## Problems/blockers
None. Note: the venv `python.exe` is a redirector, so the listening PID is its child, not the PID `Start-Process` returns. RUN_DEMO therefore finds the PID from the port.

## What I should verify manually
- □ In a fresh PowerShell window: `Set-Location D:\SIH\DrishtiSR` then `powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1`, and open http://127.0.0.1:8000/
- □ Open "What's novel" and confirm the ONNX / CPU-only wording.
- □ Check the uncertainty overlay and GeoTIFF download in the browser (open in QGIS if possible).
- □ Decide where the `runs/mvp` logs live, then run: `Copy-Item -Recurse D:\SIH\DrishtiSR-mvp\runs\mvp D:\SIH\DrishtiSR\runs\; git -C D:\SIH\DrishtiSR worktree remove --force D:\SIH\DrishtiSR-mvp`
- □ Confirm origin/main on GitHub shows the P13 commit.

## Next action
Build and run the Hugging Face Space image once (`docker build` + `docker run -p 7860:7860`, per `deploy/hf_space/README.md`) before publishing the demo.
