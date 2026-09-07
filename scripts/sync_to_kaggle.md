# Getting this repo from `D:` onto Kaggle

Two ways exist. This document compares them, recommends one, and gives the exact
commands. The recommendation is not a matter of taste — over a week of many
re-syncs the two options differ by roughly an order of magnitude in the time they
cost you, and one of them can silently run code you did not write.

**Recommendation: git clone.** Push to GitHub, let the Kaggle session clone. Skip
to [The recommended loop](#the-recommended-loop) if you do not want the argument.

Related: [`docs/kaggle_workflow.md`](../docs/kaggle_workflow.md) covers driving
Kaggle *jobs* headlessly once the code is there. This document is only about
getting the code there.

---

## The two options

### Option A — git clone (recommended)

The Kaggle notebook runs `git clone` in its first cell. Your `D:` drive is never
involved; the transport is GitHub.

```
D:  --git push-->  GitHub  --git clone-->  Kaggle session
```

The repository must be **publicly cloneable**. The session clones anonymously —
no SSH key, no credential helper, no token — so a private repo or an
`git@github.com:` remote fails inside the run with an auth prompt that never
gets answered.

### Option B — upload the repo as a Kaggle Dataset

You zip the working tree, create or version a Kaggle Dataset from it, attach the
dataset to the notebook, and the code appears read-only under `/kaggle/input/`.

```
D:  --kaggle datasets version-->  Kaggle Dataset  --mount-->  /kaggle/input/<slug>
```

---

## The comparison

| | **A: git clone** | **B: repo as a Dataset** |
|---|---|---|
| time per re-sync | `git push`, seconds | zip + upload + Kaggle unpacks + dataset version becomes available: **minutes**, and you wait on the last step |
| what the session runs | a SHA you can `git checkout` | whatever was in your working tree when you zipped it |
| provenance in the log | the commit hash, printed by cell 1 | a dataset version number that maps to nothing |
| working-tree drift | impossible — uncommitted work simply is not there | **silent** — you can and will upload a half-finished edit |
| picking up a fix mid-session | re-run cell 1, seconds | re-upload, wait, then detach/reattach the dataset in the sidebar |
| writability | the clone is writable; `outputs/` is written in place | mount is **read-only**; you must copy the tree before anything writes |
| needs internet in the session | yes (also needed for pip) | no |
| needs a public repo | **yes** | no |
| quota | none | counts against your Kaggle Dataset storage |
| what it is genuinely good at | everything here | shipping *data*, which is what the cache dataset already does |

### The three that actually decide it

**1. Re-sync latency, multiplied by a week.** You said many re-syncs. Option A
costs a `git push` and a re-run of one cell. Option B costs a zip, an upload, and
then Kaggle's server-side unpack, which is not instant and is occasionally backed
up. At ten syncs a day the difference is minutes versus most of an hour, spent
watching a progress bar rather than reading results.

**2. You cannot tell what Option B ran.** This is the serious one. A dataset
version is a number; it does not tell you which of your edits were saved when you
zipped. With a clone, the session prints the commit and refuses to be anything
else, and `scripts/kaggle_run.py push` already **refuses to push from a dirty or
unpushed tree** for exactly this reason — that mistake has cost this project a
session once already. Option B has no equivalent guard and cannot have one.

**3. The mount is read-only.** `/kaggle/input` cannot be written. Every script
here writes under `outputs/`, resolved against `repo_root()`, so a repo mounted
at `/kaggle/input/<slug>` would need copying to `/kaggle/working` before the
first stage — reintroducing a copy step *and* the question of which copy is
current.

### When Option B is the right answer

- **The repo must stay private** and you will not make a public mirror. This is
  the only genuinely disqualifying constraint on Option A.
- **The session has no internet.** Then a clone is impossible — but so is
  `pip install`, so the Kaggle image would have to already carry every dependency
  in `requirements.txt`, which it does not (`omegaconf`, `tacoreader`,
  `opensr-test` are all missing).

Note what Option B is *not* needed for: **the data**. The ~3.9 GB SEN2NAIPv2
cache is already published as its own Kaggle Dataset by
`scripts/kaggle_upload.py`, which is the correct use of the mechanism — large,
rarely-changing, read-only. Code is small and changes constantly. Use each
transport for what it is good at.

---

## The recommended loop

### Once, ever

**A public GitHub remote with an HTTPS URL.**

```
git remote -v
```

It must read `https://github.com/<you>/DrishtiSR.git`, not `git@github.com:...`.
Fix an SSH remote with:

```
git remote set-url origin https://github.com/<you>/DrishtiSR.git
```

Then confirm the repo is public — GitHub → the repo → *Settings* → *General* →
*Danger Zone* → *Change repository visibility*. Anonymous cloneability is the
requirement; verify it the way Kaggle will:

```
git ls-remote https://github.com/<you>/DrishtiSR.git HEAD
```

If that works from a shell with no credentials, Kaggle can clone it.

**A Kaggle API token**, if you also want to drive jobs headlessly:
<https://www.kaggle.com/settings/account> → API → *Create New Token*, saved at
`~/.kaggle/kaggle.json`. Not needed to run a notebook by hand in the browser.

### Every sync

```
D:\SIH\DrishtiSR\.venv\Scripts\python.exe -m pytest
git add -A
git commit -m "..."
git push
```

Tests first, and not as a courtesy: a Kaggle session that dies on an import error
has still consumed setup time and, on a GPU job, budget.

Then, in the Kaggle notebook, **re-run cell 1**. It fetches and checks out
`origin/main` and prints the commit it landed on.

### The one thing that will catch you out

**Re-running cell 1 does not reload already-imported `src/` modules.** Python
caches them. The new code is on disk in the session and the old code is still in
memory, and nothing says so — the symptom is a fix that "did not work".

After any push that changes `src/`:

> *Run → Restart & clear cell outputs*, then run cell 1.

Changes to `scripts/` and `configs/` do **not** need a restart: every stage runs
its entry point in a fresh subprocess (`run_stage` → `sys.executable`), which
reads both from disk each time. Only `src/` modules imported into the notebook
process are cached.

---

## Two ways to run, once the code is there

**By hand, in a browser tab** — [`notebooks/01_day1_baseline.ipynb`](../notebooks/01_day1_baseline.ipynb).
Upload it once via Kaggle's *File → Import Notebook*, attach the cache dataset,
and run cells. It clones the repo itself, so re-syncing is a cell re-run. Right
when you want to watch stages finish and react between them.

**Headless** — `scripts/kaggle_run.py`, documented in
[`docs/kaggle_workflow.md`](../docs/kaggle_workflow.md). It generates a notebook
from `notebooks/templates/kaggle_job.py`, bakes in one exact commit, pushes,
polls, and fetches the outputs. Right for anything long, and for anything you
want reproducible from a SHA.

```
python scripts/kaggle_run.py push   --job baseline --smoke   # generate, do not send
python scripts/kaggle_run.py push   --job baseline
python scripts/kaggle_run.py status --job baseline --watch
python scripts/kaggle_run.py fetch  --job baseline
```

Both clone from the same GitHub remote. Neither uploads your working tree.

---

## If you have to take Option B anyway

For the private-repo case, in full, so it is not reinvented under pressure.

1. **Never zip the working tree wholesale.** `outputs/` holds the multi-gigabyte
   sample cache and the upload staging copies; `.venv/` is larger still and is
   Windows-specific binaries that will not run on Linux. Export the tracked
   files only:

   ```
   git archive --format=zip -o D:\SIH\drishtisr_src.zip HEAD
   ```

   `git archive HEAD` also makes the drift problem visible rather than silent —
   it exports the last commit, so anything uncommitted is conspicuously missing
   instead of quietly included.

2. **Create the dataset once**, then version it on each sync:

   ```
   python -m kaggle datasets init  -p <folder>
   python -m kaggle datasets create -p <folder>
   python -m kaggle datasets version -p <folder> -m "sync <short sha>"
   ```

   `python -m kaggle`, never a bare `kaggle` — the console script is not on PATH
   here and resolves to a different interpreter when it is (see the
   Python-versions table in `AGENTS.md`).

3. **Put the commit SHA in the version message.** It is the only provenance this
   route has. Without it the run log cannot be traced to any code at all.

4. **Copy before running.** The mount is read-only:

   ```python
   shutil.copytree("/kaggle/input/<slug>", "/kaggle/working/DrishtiSR")
   ```

   Then `os.chdir` into the copy, so `repo_root()` resolves under
   `/kaggle/working` and `outputs/` is saved as kernel output.

5. **Re-attaching is manual.** A new dataset version does not reach a running
   session. Detach and re-attach the dataset in the sidebar, then restart the
   kernel.
