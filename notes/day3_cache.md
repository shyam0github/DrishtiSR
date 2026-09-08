# Day 3, task 2 — expanding the SEN2NAIPv2 cache

**Date:** 2026-09-09
**Time box:** 45 minutes wall clock, enforced in code (`--time-budget-sec 2700`).
**Outcome:** the box was enforced and the run ended on it. See *Results*.

---

## The headline, before the numbers

**The 8–10k target was never reachable, for two independent reasons.** Both were
measurable before the run and neither is a failure of the pipeline.

1. **The catalog does not contain 8–10k eligible pairs.**
   `cfg.sen2naipv2.min_correlation: 0.9` keeps **4,409** of the 8,000
   `sen2naipv2-crosssensor` rows. MEASURED this run, and it matches the figure
   already recorded in `configs/base.yaml`. With 3,002 pairs already cached, the
   most the cache can ever reach at this threshold is **4,409** — about 1,400
   more, not 6,000 more. Getting to 8k means dropping `min_correlation` to
   ~0.87, which admits the worst-registered pairs in the dataset. That is a
   data-quality decision, not a plumbing one, and it is not made here.

2. **Wall clock.** The fetch rate is ~11 s/pair, MEASURED across this run and
   consistent with the ~10.4 s/pair recorded on Day 1. 2,700 s therefore buys
   **~245 pairs**, whatever the target says. Reaching even 4,409 would take
   ~4.3 hours of continuous download.

Both facts were stated before the run started rather than discovered in the
post-mortem. The time box was spent anyway, because ~245 additional training
tiles is worth 45 minutes and the harness that makes the box enforceable is
reusable for every later expansion.

---

## What was built

The requirement was that the box be *enforceable, not a promise*. Four
properties do that, each tested rather than asserted
(`tests/test_cache_expand.py`, `tests/test_cache_manifest.py`).

### 1. A manifest, not a directory glob — `src/data/cache_manifest.py`

JSONL at the cache root, one appended line per pair: `taco_id`, `lr_path`,
`hr_path`, `lr_shape`, `hr_shape`, `bands`, `cached_at`, plus the scaling facts
(`dtype`, `nodata_value`, `reflectance_scale`) and the measured `lr_dn_p999`.

A glob answers "which files exist". The question is "which pairs are complete,
validated, and safe to train on", and a filename cannot answer it — it counts a
truncated `.npz`, and it cannot tell a 130/520 pair from a 130/260 one.

Paths are stored **relative** to the manifest directory, so the cache stays
portable between `D:\SIH\DrishtiSR\outputs\cache` and `/kaggle/input/...`.

### 2. Resumability

The manifest is read first; cached taco ids are skipped; new records are
appended. A cache that predates the manifest is **backfilled** by reading each
NPZ member's *header* — shape and dtype without inflating the array. The 3,002
Day 2 pairs were adopted this way in 231 s rather than re-downloaded over nine
hours.

`is_cached()` now consults the manifest first. Side effect worth recording:
building the training split went from decompressing 2,400 NPZs to a **0.38 s**
lookup.

### 3. Atomic writes

Each pair is written to `<stem>.npz.tmp` / `<stem>.json.tmp`, then **re-read from
disk** and validated, then `os.replace`d into place, and only then does the
manifest line get appended with an `fsync`. The manifest append is the commit.

- A kill mid-write leaves temporaries and nothing under the real name.
- A kill between the replace and the append leaves a complete but unrecorded
  pair, which the next run's backfill adopts for free. The failure mode is a
  re-validation, never a re-download.

Validation reads the file **on disk**, not the in-memory arrays, so a truncated
or mis-encoded write is caught rather than trusted.

### 4. Per-pair validation before commit

HR exactly `4 x` LR in both spatial dims (not "about 4x" — that would mean a
resampling step crept in and the pair is no longer co-registered); band count;
dtype identical to the existing cache (`uint16` raw DN — a float array means the
reflectance divisor was already applied and the pair would be divided twice);
no all-zero and no all-NaN tile; and a reflectance-scaling check.

The scaling check bounds the 99.9th percentile of valid LR digital numbers.
MEASURED over a seeded 300-pair sample of the existing cache: p99.9 ran
**2,152 to 8,303**, median 4,156. The configured bounds are `[50, 40000]` —
more than an order of magnitude wider on each side, deliberately. It exists to
catch a pair delivered under a different quantification value, **not** to filter
bright scenes: DN 40,000 is reflectance 4.0, and snow, cloud and specular water
must pass unclipped.

Failures are counted by reason and logged at ERROR; they never crash the run and
are never silently skipped.

---

## Keeping Run A in the results table

This was the constraint with teeth. Re-running `scripts/make_splits.py` on a
larger dataset **reassigns everything** — the greedy largest-first pass sees
different group sizes and tiles move between splits. Every Day 2 number would
silently become a number about a validation set that no longer exists.

So the split is not recomputed. `scripts/make_splits.py --extend`
(`src/data/split_extend.py`) freezes every existing assignment verbatim and
places only the new samples.

New samples cannot simply be dumped into train, because the separation guarantee
is symmetric: a new tile 2 km from a *validation* tile leaks whether it arrived
first or last, and the leak is invisible in training loss. A new sample is
refused from train when either it is within `cfg.splits.min_separation_km` of a
val/test sample, or it shares a NAIP quarter-quad with one (overlapping crops of
one aerial scene, at any distance). Refused samples get the split name
`excluded`, which no loader selects.

The script **fails and writes nothing** if val or test membership changes by a
single id, and re-derives the separation guarantee from the coordinates before
writing.

### Verified independently

Not taken on trust from the script that produced it:

- val: **300 ids, identical set** to `outputs/splits_sen2naipv2.csv`.
- test: **300 ids, identical set**.
- Nearest new sample to any held-out tile: **11.27 km** — comfortably beyond the
  5 km guarantee, which is why 0 of 1,409 candidates were excluded. That zero is
  a genuine finding, not a guard that failed to fire.

---

## Results

Run: `scripts/prepare_data.py --target-pairs 4500 --time-budget-sec 2700`,
2026-09-09 00:30:17 → 01:15:10. Exit code **0**.
Machine-readable summary: `outputs/metrics/day3_cache_expansion.json`.

### Cached pairs

| | before | after | delta |
|---|---|---|---|
| pairs in cache | **3,002** | **3,261** | **+259** |

- Elapsed **2,705.0 s** against a 2,700 s budget. The 5 s overrun is the pair in
  flight finishing, being validated, and being committed — the budget is checked
  at the top of an iteration precisely so that a started pair always lands whole.
- Rate **0.0957 pairs/s** (10.4 s/pair), stable across the whole run.
- `stopped_because: time_budget`. Exited 0, as specified.
- Backfill on entry: 3,002 already recorded, **0 unreadable, 0 structurally
  rejected** — every Day 2 pair is a clean `(4,130,130)`/`(4,520,520)` uint16
  pair.
- **0 stray `.tmp` files** after the run.

### Rejects by reason

**Zero, across all eight reasons.** Listed in full because a zero is evidence
the check ran and passed, not evidence it was skipped:

| reason | count |
|---|---|
| `fetch_failed` | 0 |
| `scale_mismatch` | 0 |
| `band_count_mismatch` | 0 |
| `dtype_mismatch` | 0 |
| `all_zero` | 0 |
| `all_nan` | 0 |
| `dn_range_implausible` | 0 |
| `shape_degenerate` | 0 |

No retries were needed and no HTTP 429s were seen — a quieter network than the
Day 1 run that died after 121 samples.

### Train / val split

`configs/base.yaml` `sen2naipv2.num_samples` was raised 3000 → 4409 so the new
pairs are visible to the catalog, and the split was **extended, not recut**.

| split | Day 2 assigned | Day 3 assigned | Day 2 trainable | Day 3 trainable |
|---|---|---|---|---|
| train | 2,400 | 3,809 | **2,400** | **2,659** (+259) |
| val | 300 | 300 | **300** | **300** (unchanged) |
| test | 300 | 300 | **300** | **300** (unchanged) |
| excluded | — | 0 | — | — |

"Assigned" counts catalog rows; "trainable" counts rows whose pair is actually
in the local cache (`cfg.loader.cached_only: true`). The 1,150-row gap in train
is catalog entries not yet downloaded — the loader logs that count at WARNING on
every build rather than absorbing it silently.

**All 259 new pairs went to train.** Verified against a copy of the Day 2 file:

- val: 300 ids, **set-identical**.
- test: 300 ids, **set-identical**.
- **0** previously-assigned samples changed split.
- Nearest cross-split pair: 5.00 km, guarantee re-derived from coordinates.

Run A stays in the results table.

## A bug this work found

`is_cached()` caught `(OSError, ValueError, EOFError)` when probing a cached
NPZ. A truncated NPZ — the signature of an interrupted download, and the single
most likely damage in this cache — raises `zipfile.BadZipFile`, which is a plain
`Exception` and none of those. The handler written to survive an interrupted
download would itself have crashed on one. Now centralised as
`cache_manifest.UNREADABLE_CACHE_ERRORS` and used in both the probe and the
backfill. Caught by `tests/test_cache_expand.py`, not by inspection.
