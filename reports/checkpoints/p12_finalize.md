# CHECKPOINT REPORT
## Overall status   🟡 PARTIAL
The merge, tests, push and state files are done. Two items were blocked or skipped: the idle-benchmark rerun (a permission refusal) and the worktree folder removal.

## Deliverables
| # | Deliverable | Status | Evidence I can check |
|---|---|---|---|
| 1 | Preconditions | ✅ | Main was clean at `5500a65`. The worktree had `reports/mvp/p0_smoke.json` modified (committed in P12) and `runs/mvp/` untracked. Commits present: P0, P4, P5, P6, P7, P9, P10, P11. P1–P3 have no `[mvp/Pn]` commit; P8 was cancelled. No training process was running; one idle `scripts/serve.py` (PID 16916/15028) was on :8000. |
| 2 | Learned-head decision | ✅ TTA kept | Watch "passed", but AUSE learned 0.00271 > TTA-8 0.00182 (`reports/mvp/unc_eval_*.json`). TTA-4 stays the default in `app/server.py` (unchanged). No packed export. The reason is logged in `docs/mvp/decisions.md` (P12). |
| 3 | Kaggle fallback | N/A | `reports/mvp/unc_cpu_speed.json`: needs_kaggle=false, and CPU training c1 finished 2000 it. |
| 4 | Idle benchmark + tables | 🟡 | The bench rerun was refused by the session's permission check, and so was stopping the leftover server. `bench_a2-last-dce224ec.json` was kept (provisional=false, load 0 %, 2026-09-11). `make_tables.py --split val` ran: ablation 7/7 filled, 0 pending. |
| 5 | Worktree tests | ✅ | `tests/mvp` 81 passed. Commit `209edb7` "[mvp/P12] finalize reports". |
| 6 | Merge + main E2E + push | 🟡 | `afa86b7` merged with no conflicts. `drishtisr.__file__` is `D:\SIH\DrishtiSR\drishtisr\__init__.py`. E2E 7/7 in main. `/api/health` returned ok (onnx-fp32, a2-last-dce224ec) on :8001. Pushed `5500a65..afa86b7`. The worktree was NOT removed: git refuses because of the untracked `runs/mvp` logs. The branch is kept. |
| 7 | PROJECT_STATE / PROGRESS | ✅ | Rows updated for Day 3 uncertainty, Day 4 ONNX/CPU/uncertainty/FastAPI, Day 5 swipe/overlay/GeoTIFF/metrics and Day 7 README/demo script, plus the 2026-09-12 decisions bullet and Next action. PROGRESS.md has a 2026-09-12 block. Both are in the P12 state commit. |

## Numbers / counts
- Tests: worktree 81/81 (tests/mvp); main `test_e2e.py` 7/7. The full tests/mvp suite was not rerun in main.
- INT8 graph: 981 KB (981,131 B). It is benched but not served, because the gate failed on all 4 rungs. The served model is ONNX FP32 at 3,441,761 B.
- Median latency, 256² LR, 6 threads: onnx-fp32 1220 ms, provisional=false (INT8, not served: 782 ms).
- LPIPS vs bicubic: 0.3322 vs 0.4033, a 17.6 % reduction (VAL n=1199, run commit fb8659d).
- Projection gate: FAILED, so projection is not served.
- Served uncertainty: TTA-4. The learned head is opt-in via `DRISHTI_UNC_CKPT`.
- AUSE: learned 0.00271 vs TTA-8 0.00182 (TTA-4 0.00191); Spearman ρ 0.379 vs 0.542.
- SHAs: worktree `209edb7`; merge `afa86b7`; the P12 state commit is the one that adds this file (`git log -1 -- reports/checkpoints/p12_finalize.md`).

## Files/artifacts created or changed
- Worktree commit: `docs/mvp/decisions.md`, `reports/mvp/{ablation,headline,p0_smoke}.json`.
- Main: the merge of `mvp/night-build`; `PROJECT_STATE.md`, `docs/PROGRESS.md`, `reports/checkpoints/p12_finalize.md`.
- Copied from the worktree into main, all gitignored: `artifacts/onnx/a2-last-dce224ec/*` (hash-verified), `app/samples/*`, `runs/mvp/unc_c1/head_last.pt`.
- Not created: the `drishtisr-day4` kernel folder, because it was not needed.

## Verification performed
I checked main's status and the worktree's commits and processes, and read the decision inputs from the report JSONs. I regenerated the VAL tables and ran all MVP tests in the worktree, then committed and merged. After the merge I confirmed that main imports its own package and that the end-to-end test passes. I also started the server, confirmed that health reports the ONNX FP32 model, and stopped it. Finally I pushed and checked that main's status stayed clean after copying the runtime files.

## Problems/blockers
- Stopping the leftover `serve.py` on :8000 and rerunning `bench_cpu.py` were both refused by the auto-mode permission check. The old server may still hold files in `D:\SIH\DrishtiSR-mvp`.
- Worktree `D:\SIH\DrishtiSR-mvp` remains. Its `runs/mvp` logs (unc_c1/smoke/speedtest `log.jsonl` and `run_metadata.json`) are untracked in main. Forcing the removal would delete them.
- `app/static/index.html` still says "INT8" in the "What's novel" panel, but INT8 is not served. The file was outside P12's owned files, so it is not fixed.
- PROJECT_STATE §6 still says "INT8 ONNX ≈1 MB. Nothing is exported yet." That line is now stale, but it was outside the rows this prompt allowed me to edit.

## What I should verify manually
- □ Stop the old demo server on :8000 (PID 16916) and, if you want a fresh number, run `scripts/mvp/bench_cpu.py --checkpoint D:\SIH\DrishtiSR\runs\day3\a2\last.pt`.
- □ Decide where `runs/mvp` logs live (copy them to main and ignore them, or keep them), then `git worktree remove D:\SIH\DrishtiSR-mvp`.
- □ Open the demo (`scripts/serve.py`) and check swipe, overlays and the GeoTIFF download in a browser.
- □ Change the "INT8" copy in the "What's novel" panel to "ONNX, CPU-only".
- □ Confirm `origin/main` on GitHub shows the P12 state commit.

## Next action
Build and run the Hugging Face Space image once (`docker build` + `docker run -p 7860:7860`, per `deploy/hf_space/README.md`) before publishing the demo.
