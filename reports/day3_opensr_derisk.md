# opensr-test de-risking — Day 3

**Verdict: PATH A.** `opensr-test` installs, imports, and emits finite numbers on
real validation data, and the metrics discriminate between bicubic and a trained
model. The fallback — reimplementing consistency ourselves in
`src/metrics_consistency.py` — is **not** needed and was not taken.

This check was run deliberately early, before opensr-test became load-bearing for
the submission. A broken external dependency discovered at hour 6 of the build is
not a recoverable position; discovered now, it costs an afternoon.

---

## 1. Install — pinned, not floating

Already pinned in `requirements.txt` and verified resolved in the local venv:

| package | pinned | resolved |
|---|---|---|
| `opensr-test` | 1.3.3 | 1.3.3 |
| `satalign` | 0.1.17 | 0.1.17 |
| `mpltern` | 1.0.5 | 1.0.5 |
| `opencv-python` | 4.10.0.84 | 4.10.0.84 |
| `numpy` | 1.26.4 | 1.26.4 |

Every one is an exact `==` pin. The `opencv`/`numpy` pair must be installed in a
single command — see the block comment in `requirements.txt`; `satalign` pulls an
opencv that requires `numpy>=2`, which breaks `scipy` 1.14.1 and the torch 2.5.1
build if allowed to float.

`--no-deps` remains correct: the `clip` correctness distance is unavailable
(needs `open-clip-torch`) and we do not use it. `nd`, the upstream default, is
what we run.

## 2. The API, read from the installed source

Read out of `.venv/Lib/site-packages/opensr_test/`, not from memory or the README.

**Entry point.** `opensr_test.Metrics(params=Config(...))`, then
`.compute(lr, sr, hr, gradient_threshold="auto")`, which returns a flat dict of
seven floats.

**Layout and dtype.** `(C, H, W)` torch tensors — **CHW, not HWC**, and a single
sample, not a batch: `(B, C, H, W)` fails inside `interpolate`. float32 or
float64; `uint16` raises. `sr.requires_grad` must be `False` (explicit
`ValueError`). `C >= 3` — with `C == 1` a `.squeeze()` in `apply_upsampling` eats
the channel axis.

**Scale — the dangerous one.** The library takes **surface reflectance,
nominally [0, 1]**, and does *not* check it. `reflectance` and `synthesis` are
absolute L1 distances and scale linearly with the input, so passing digital
numbers inflates those two by 10000× while leaving the other five untouched, and
the run completes with no warning. `cfg.opensr_test.max_reflectance` is the
tripwire for this. It is a tripwire, **not a clip** — values above 1.0 pass
through, per the project's no-silent-clipping rule.

**Band order.** `rgb_bands` defaults to `[0, 1, 2]`, and upstream's own datasets
are RGBNIR — red first — which matches our `cfg.dataset.bands`
(`[B04, B03, B02, B08]`). The indices are still resolved **by name**, because the
coincidence holds only for the current band list and the failure would be
invisible: a spatial PCC on a blue/green/red composite returns a plausible number.

**Triplet or pair.** Both, and this matters:

| call | needs | returns |
|---|---|---|
| `.consistency(lr, sr)` | **LR/SR only** | `reflectance`, `spectral`, `spatial` |
| `.synthesis(lr, sr, hr)` | triplet (HR optional) | `synthesis` |
| `.correctness(lr, sr, hr)` | **triplet** | `ha_metric`, `om_metric`, `im_metric` |
| `.compute(lr, sr, hr)` | triplet | all seven |

So the **consistency metrics — the ones that speak to our spectral-consistency
contribution — need no ground truth at all.** They compare the input LR against
the SR bilinearly-antialias-downsampled back to LR. That means they can be
computed on real Sentinel-2 scenes where no HR reference exists, which is exactly
the Delhi demo case.

**A trap in the upstream README.** It shows `Metrics(config=config)`. The real
signature is `Metrics(params=None, **kwargs)`, and `Config` is a pydantic model
that silently ignores unknown fields — so that call **discards the config and
runs defaults**. `build_metrics` reads every field back off the constructed
object and asserts it took.

## 3. The adapter

`src.eval.opensr_harness.score_arrays(lr, sr, hr, cfg) -> dict[str, float]`.

Takes numpy arrays (or tensors) in **our** convention — `(C, H, W)`, float32
surface reflectance, unclipped, `cfg.dataset.bands` order — and returns the flat
seven-metric dict. All layout, dtype, device and detachment conversion happens
inside it and nowhere else. `scripts/eval_opensr.py` drives it and never
transposes, rescales or casts.

`tests/test_opensr_harness.py::test_score_arrays_matches_the_two_step_path`
pins it to the existing two-step path, so the two can never silently diverge.

## 4 & 5. Measured — bicubic vs Run A, same 50 validation pairs

```
.venv/Scripts/python.exe scripts/eval_opensr.py --n-samples 50 --set train.num_workers=0
```

50 pairs per method, one shared `Metrics` object, same pairs in the same order.
**100/100 rows scored, 0 skipped, 0 non-finite values.** Run twice; identical to
four decimals.

| metric | dir | bicubic | edsr_runA | separation |
|---|:--:|---:|---:|---:|
| `reflectance` | ↓ | 0.0036 | 0.0052 | **−0.0016** |
| `spectral` | ↓ | 0.9799 | 1.4529 | **−0.4730** |
| `spatial` | ↓ | 0.0048 | 0.0248 | **−0.0200** |
| `synthesis` | ↑ | 0.0040 | 0.0070 | **+0.0030** |
| `ha_metric` | ↓ | 0.0859 | 0.1612 | **−0.0752** |
| `om_metric` | ↓ | 0.8564 | 0.6991 | **+0.1573** |
| `im_metric` | ↑ | 0.0577 | 0.1398 | **+0.0821** |

Separation is signed so that **positive means Run A is better**. Every one of the
seven moves, and none is near zero: the metrics discriminate.

### Reading these honestly

**Run A is worse than bicubic on all three consistency metrics, and that is not a
bug.** Bicubic downsampled back to LR reproduces the input almost exactly by
construction — it is a smooth interpolation and invents nothing, so it sets a
consistency floor that is very hard to beat. A model that adds real
high-frequency detail necessarily drifts from that floor unless it is explicitly
constrained not to.

That constraint is contribution #1. This table is therefore the **headroom
measurement for the spectral-consistency loss**, quantified by a third-party
implementation we did not write: Run A leaves 0.47° of spectral angle and
0.0016 reflectance L1 on the table against a zero-parameter baseline. Closing
that gap while keeping the correctness gains below is the result to report.

**On correctness, Run A is genuinely better, with the expected trade.** Omission
falls sharply (0.856 → 0.699) and improvement more than doubles (0.058 → 0.140):
the model recovers real detail bicubic misses. Hallucination also rises
(0.086 → 0.161) — it invents some detail that is not in the reference. That trade
is exactly the quantity the heteroscedastic uncertainty head exists to expose,
and we now have an external number for it.

Note Run A is **1,518,724 parameters, 1.52× over the 1,000,000 budget**. It is a
reference measurement, not a deliverable; the submitted model must fit.

## Operational notes

- **`--set train.num_workers=0` was needed for this run.** Windows spawns
  dataloader workers, which re-import `src.data.sen2naip` from disk; a second
  terminal was mid-write on that file (+576 lines), so a worker imported a
  partially-written module and died with
  `AttributeError: 'SEN2NAIPv2Dataset' object has no attribute '_manifest_index'`.
  This is a collision with concurrent editing, **not a defect** in the data
  layer. Use `num_workers=0` while another terminal is writing `src/data/`.
- Cost: ~0.25 s/sample for bicubic, ~0.45 s/sample for the model, on 6 CPU
  threads. A full 1199-patch validation pass is roughly 8 minutes per method.
- Artefacts: `outputs/metrics/opensr_pair_bicubic.{json,csv}` and
  `opensr_pair_edsr_runA.{json,csv}`; console log in
  `outputs/opensr_pair_run.log`.
