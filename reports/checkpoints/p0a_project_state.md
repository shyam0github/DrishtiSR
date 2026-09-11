# CHECKPOINT REPORT

## Overall status
✅ COMPLETE. `PROJECT_STATE.md` exists with all 8 sections in 148 lines. C1–C3 are done, with evidence. Five contradictions between the seed facts and the repo are recorded; the repo wins in each.

## Deliverables

| # | Deliverable | Status | Evidence I can check |
|---|---|---|---|
| 1 | PROJECT_STATE.md created | ✅ | `PROJECT_STATE.md` at the repo root, 8 sections, 148 lines |
| 2 | Seed facts verified | ✅ 45/55 confirmed | `PROJECT_STATE.md` §6 lists every contradiction and every "Verification needed" item |
| 3 | C1 frozen config hash | ✅ intact | `load_frozen()` with verify=True returned `e6082c69bba3086aaf71d0a413e64d621d860e5506da507ce2d76edb92f1fc0a`. The last commit to touch `configs/frozen_day3.yaml` is `bf7fa0f` |
| 4 | C2 data count reconciliation | ✅ explained, nothing fixed | See "Numbers / counts" below; files are `outputs/manifest_sen2naipv2.csv`, `outputs/splits_sen2naipv2.csv` and `outputs/cache/sen2naipv2-crosssensor/` |
| 5 | C3 .gitignore fix | ✅ | `.gitignore` now has `!reports/checkpoints/` right after `checkpoints/`. `git check-ignore -v --no-index` output is below |
| 6 | Pointer lines | ✅ | `docs/PROGRESS.md` line 1 and `AGENTS.md` line 3. `CLAUDE.md` only imports `@AGENTS.md`, so it was left unchanged |
| 7 | Commit + push | ✅ | The commit containing this file, on `origin/main` |

## Numbers / counts
- **Seed facts:** 45 of 55 confirmed; 5 need verification; 5 contradicted.
- **"Verification needed" items (6):**
  - the team name;
  - the opensr-test degradation operator;
  - A2's train-set size;
  - the Kaggle cache size of ≈3.62 GB;
  - INT8 ONNX at ≈1 MB;
  - the quota reset time.
- **Contradictions (5):**
  - The GPU is T4, and P100 is forbidden in config.
  - Val is 300 tiles / 1,199 patches, fixed since Day 1.
  - The inference path is `src/infer/tiled.py`, not `src/drishtisr/infer/tiled.py`.
  - The repo's Run A confounds include weight decay and do not list dataset size.
  - The verdict (b) wording in `reports/day3_data_validity.md` §4 differs from the approved text.
- **Status rows:** ✅ 14 · 🟡 1 · ❌ 0 · ⏭️ 20 (35 rows).
- **PROJECT_STATE.md:** 148 lines.
- **Cached pairs vs manifest rows: 3,261 vs 3,000.**
  - `manifest_sen2naipv2.csv` is dated 2026-09-07. It predates the 2026-09-09 cache expansion and was never regenerated. Its 3,000 rows are exactly the first 3,000 rows of the splits file (2,400 train / 300 val / 300 test), and all 3,000 are cached.
  - The extra 261 cached pairs are:
    - 259 expansion pairs, all train (splits rows 3,000–4,408);
    - 2 stray pairs at catalog rows 0–1, with correlation 0.887 and 0.893. They are below `min_correlation` 0.9 and absent from the splits file, so no split loads them.
  - The splits file has 4,409 rows (3,809 / 300 / 300); 1,150 of those ids are uncached.
  - Cached train = 2,659.
  - The cache `.npz` total is 4.22 GB (3.93 GiB) on disk.
- **C3 proof** (`git check-ignore -v --no-index --non-matching`):
  - `reports/checkpoints/p0a_project_state.md` gave `::`, which means NOT ignored. Before the fix it matched `.gitignore:14:checkpoints/`.
  - `runs/day3/a2/best.pt` gave `.gitignore:10:*.pt`, so it is ignored.
  - `runs/runC/checkpoints/ckpt_it001000.pt` gave `.gitignore:14:checkpoints/`, so it is ignored.
  - `reports/checkpoints/x.pt` gave `.gitignore:10:*.pt`, so weights stay ignored even inside the reports folder.

## Files/artifacts created or changed
- `PROJECT_STATE.md` (new)
- `reports/checkpoints/p0a_project_state.md` (new, this file; staged without `-f`)
- `.gitignore`: one negation plus a one-line comment
- `docs/PROGRESS.md`: one pointer line at the top
- `AGENTS.md`: one pointer line under the title
- Scratch only, not committed: three read-only check scripts in the session scratchpad

## Verification performed
- Read AGENTS.md, all of docs/PROGRESS.md, the Day 3 P1 checkpoint report, the A/B metrics CSV and the data-validity report. Checked git status, the last 30 commits, the branches and the worktrees.
- Loaded the frozen config through the repo's own loader with hash checking on. It passed, and the hash starts `e6082c69`.
- Counted the cache files, the manifest rows, the split rows and the cache index lines. Cross-matched the ids between them, compared file dates, and opened the metadata of the 2 pairs that are not in the splits file.
- Checked each seed fact against its source file: config values, run metadata, the training loss downsampler, the Kaggle job accelerator, file paths, requirements pins, the augmentation block and the report numbers.
- Ran git's ignore check before and after the `.gitignore` edit, and staged this report without force-adding it.
- Ran no training, evaluation, tests or builds, and touched no code, config, data or frontend.

## Problems/blockers
- **Seed/repo contradictions (5).** Recorded in `PROJECT_STATE.md` §6 and not resolved; each one needs a human call on which wording or value is canonical.
- **Stale manifest.** Problem: the manifest undercounts the cache by 261 pairs. Status: recorded only, as instructed.
- **opensr operator unknown.** Problem: the question stays open. Next: read the opensr-test 1.3.3 source for its `consistency()` degradation.
- **HF push.** Still blocked on `HF_TOKEN`.

## What I should verify manually
- □ AGENTS.md §1 still says "Kaggle P100"; the config and the day3 job say T4. Decide whether to correct AGENTS.md.
- □ Pick the canonical verdict (b) wording: the brief's version (now in PROJECT_STATE §2) or `reports/day3_data_validity.md` §4.
- □ Confirm the team name "Turing Testers" and the 2026-09-12 05:28 IST quota reset. Neither is in the repo.
- □ Check the Kaggle dataset size (≈3.62 GB claimed; the local npz total is 4.22 GB).
- □ Skim the `PROJECT_STATE.md` status table (Day 3 cache expansion is marked ✅ time-boxed at 3,261 of 4,409).

## Next action
Night MVP build in progress on branch mvp/night-build; P12 merges it into main.
