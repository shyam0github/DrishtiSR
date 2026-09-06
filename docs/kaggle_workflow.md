# Driving Kaggle from VS Code

Training happens on a Kaggle P100-budget account; the code lives here. This
document is the loop that connects the two without opening a browser, and — just
as importantly — the short list of things that still need one.

The tool is [`scripts/kaggle_run.py`](../scripts/kaggle_run.py). Run everything
with the project interpreter:

```
D:\SIH\DrishtiSR\.venv\Scripts\python.exe scripts/kaggle_run.py <subcommand>
```

(Shortened to `python scripts/kaggle_run.py` below. It is never a bare `python`
in practice — see the Python-versions table in `AGENTS.md`.)

---

## The idea in one paragraph

A **job** is one notebook that clones this repository at one specific commit,
installs the two or three packages Kaggle lacks, checks that it can see its data,
and runs one script from `scripts/`. Nothing else. The notebook is generated from
a template every time you push, so there is no notebook to keep in sync and no
possibility of code existing only inside a browser tab. What ran is always a SHA
you can `git checkout`.

Jobs are defined in [`configs/kaggle_jobs.yaml`](../configs/kaggle_jobs.yaml).
Three exist to start with:

| job | accelerator | runs | roughly |
|---|---|---|---|
| `verify` | CPU | `scripts/verify_data_root.py` | 5 min |
| `baseline` | CPU | `scripts/run_baseline.py` | 45 min |
| `train` | **T4 GPU** | `scripts/train.py` | 8 h |

---

## The loop

### 0. Once, ever

Get a Kaggle API token: <https://www.kaggle.com/settings/account> → API →
**Create New Token**. It downloads `kaggle.json`; put it at `~/.kaggle/kaggle.json`.
Every subcommand checks for it first and tells you this if it is missing.

Your GitHub repository must be **publicly cloneable**. The Kaggle session clones
anonymously — no SSH key, no credential helper — so a private repo or an SSH
remote URL fails inside the run. The tool refuses an SSH URL up front rather than
letting you discover it on Kaggle.

### 1. See what is defined

```
python scripts/kaggle_run.py jobs
```

Prints the configured jobs with their accelerator, whether the data guard is on,
and expected runtime; then each one's current state on Kaggle; then every kernel
on your account.

### 2. Commit and push your work — first, not last

```
git add -A && git commit -m "..." && git push
```

This is not politeness. **Kaggle clones from GitHub. It cannot see your working
tree.** Pushing a job from a dirty checkout starts a run of the *last committed*
code while you sit there believing it is running what is on screen, and nothing
in the log says otherwise. That has already cost one session on this project, so
`push` now refuses: it checks the tree is clean, fetches, and confirms your commit
is actually on a remote branch. There is no override flag.

### 3. Push the job

```
python scripts/kaggle_run.py push --job verify
```

It prints the commit it is about to run, generates `verify.ipynb` and
`kernel-metadata.json` into `outputs/kaggle_kernels/verify/`, and pushes. The run
starts immediately; the URL is printed.

Add `--smoke` for a dry run: every local check happens, the notebook is generated
for real and you can read it, and nothing is sent. That is the right thing to do
after editing the template or a job definition.

### 4. Watch it

```
python scripts/kaggle_run.py status --job verify --watch
```

Polls every 60 seconds and prints each state change (`queued` → `running` →
`complete`). It exits non-zero if the run failed, so it can gate a shell
sequence. Ctrl-C stops watching; it does **not** stop the run on Kaggle.

### 5. Read the log

```
python scripts/kaggle_run.py logs --job verify            # last 200 lines
python scripts/kaggle_run.py logs --job verify --lines 50
python scripts/kaggle_run.py logs --job verify --full
```

Tail-first, because the failure is at the end of the log and scrolling through
eight hours of training output to reach it is the wrong default. Kaggle stores
logs as JSON records rather than text; this renders them, marking `stderr` lines.
Exits non-zero if the log contains a traceback.

### 6. Fetch the results

```
python scripts/kaggle_run.py fetch --job baseline
```

Downloads into `outputs/kaggle/<job>/<timestamp>/` and prints what arrived,
grouped into checkpoints, metrics, figures and logs — and says explicitly when a
group is empty, because "the run succeeded and produced no checkpoints" is a
result worth noticing immediately.

`outputs/` is gitignored. Copy anything that belongs in the submission into
`reports/`.

---

## The two guards, and why they are not optional

**Unpushed code.** Covered above. `push` refuses a dirty or unpushed tree.

**The data guard.** Every job except `verify` itself runs
`scripts/verify_data_root.py` *before* the real work, and aborts the whole run if
it fails. This exists because a Kaggle session that cannot see its mounted
dataset does not crash. It resolves an empty tree, reports zero samples, "trains"
in nine seconds, and finishes green. On the `train` job that is an hour of a
30-hour weekly budget spent on nothing, with a log that looks fine. The guard
opens real sample files and checks their shapes and reflectance ranges, so
passing it means the data is genuinely readable — not merely that a directory
exists.

If the guard fails, the message tells you which paths were checked and why each
was rejected. It is nearly always a dataset that is not attached: fix
`dataset_sources` for that job in `configs/kaggle_jobs.yaml` and push again.

---

## Three things measured on the live API, not assumed

These cost four CPU runs to find on 2026-09-06. They are recorded here because
each one had a plausible wrong answer already written down somewhere in this
repo.

**1. The dataset mounts under the owner.** A dataset attached through
`kernel-metadata.json`'s `dataset_sources` appears at

```
/kaggle/input/datasets/<owner>/<slug>
```

**with** the owner segment — not at `/kaggle/input/<slug>`, which is what
`configs/base.yaml` and a docstring in `src/utils/paths.py` both asserted. Under
the old assumption no Kaggle run could ever see its data. Both layouts are now
tried, in the order given by `paths.kaggle_mount_patterns`, with the owner as a
glob so no username enters version control.

**2. Kaggle names the kernel from the title, not the id.** Pushing metadata with
`id: shyamdwivedi0/drishtisr-verify` and `title: DrishtiSR verify data root`
created the kernel at `drishtisr-verify-data-root`. The push reported success;
`status`, `logs` and `fetch` then all failed with a permission error that reads
as though the kernel were private. A run that works and cannot be reached.

Job titles are therefore required to slugify to the kernel slug, checked before
any push. That is why titles in `configs/kaggle_jobs.yaml` are terse — the
readable text lives in `description`, which the generated notebook prints in its
header.

**3. "Never pushed" is reported as a permission error.** Asking about a kernel
that does not exist returns

```
Cannot access kernel 'owner/slug' (Permission 'kernels.get' was denied).
```

not a 404. The wording says permission; the cause is almost always that it has
not been pushed yet.

---

## T4, never P100

GPU jobs pin `accelerator: NvidiaTeslaT4`, and the tests enforce it.

Kaggle's default image ships a PyTorch built for cu128, and that build contains
no Pascal (`sm_60`) kernels. On a P100 the failure is worse than having no GPU at
all: `torch` imports cleanly, `torch.cuda.is_available()` returns `True`, the
device reports itself as a Tesla P100 — and then the *first actual CUDA op* dies
with `cudaErrorNoKernelImageForDevice`, after the session has already queued,
booted, cloned and installed. Every cheap check passes and only the expensive one
fails. The T4 is `sm_75`, which the same build covers.

---

## Adding a job

Add an entry to `configs/kaggle_jobs.yaml`. No code changes:

```yaml
  export:
    title: DrishtiSR ONNX export
    enable_gpu: false
    entry: scripts/export_onnx.py
    entry_args: ["--config", "configs/base.yaml"]
    expected_runtime_min: 15
```

Everything else — the template, the clone directory, the mounted dataset, the
data guard, the pip list — comes from the `defaults:` block at the top of that
file. Then `push --job export --smoke` to read the notebook it would generate.

Two rules worth knowing:

- **`workdir` must be under `/kaggle/working`.** Only that directory is saved as
  kernel output. A job that clones elsewhere runs perfectly and then loses
  everything it wrote when the session ends.
- **Do not add `-r requirements.txt` to `pip_packages`.** That file pins numpy,
  pandas, scipy and pillow; installing those pins on Kaggle downgrades packages
  the preinstalled PyTorch was compiled against. Name only what the image lacks.

To change what *every* job does, edit
[`notebooks/templates/kaggle_job.py`](../notebooks/templates/kaggle_job.py) or
the functions it calls in
[`src/utils/kaggle_session.py`](../src/utils/kaggle_session.py). Never edit a
generated `.ipynb` — it is a build artefact under `outputs/` and the next push
overwrites it.

---

## What still needs a browser

Honestly, not much, but it is not zero:

1. **Getting the API token**, once. Account settings → API → Create New Token.
2. **Phone verification.** Kaggle requires a verified account before a kernel may
   use the internet or a GPU. Without it, `push` fails with a 403 that the tool
   translates for you, but the fix is on the website.
3. **Your real GPU quota.** The Kaggle public API exposes no quota endpoint —
   there is no CLI command and no documented route that returns hours used or
   remaining. `push` therefore prints a *floor*: the GPU time this tool has
   started since the quota week began (Saturday 00:00 UTC), from a local ledger
   at `outputs/kaggle/gpu_ledger.json`. Anything you launch from the browser or
   another machine is invisible to it, and it counts each job's *declared*
   `expected_runtime_min`, not measured time. Before a long training run, check
   the real figure at <https://www.kaggle.com/settings>.
4. **Live output while a run is in progress.** The CLI retrieves a run's log and
   outputs once the run has finished; there is no streaming endpoint. `status
   --watch` tells you it is alive and what state it is in, but to watch training
   loss scroll past in real time, open the kernel page.
5. **Cancelling a run.** There is no CLI command for it. Open the kernel and stop
   it there. (`status --watch` Ctrl-C only stops your watching.)
6. **Anything about the artefact rather than the run**: making a kernel public,
   editing its description, accepting a dataset's terms, or browsing rendered
   figures inline instead of downloading them.

---

## When something goes wrong

Every Kaggle failure is translated into what happened and what to do; the raw
client output is always appended, so an unrecognised error is never swallowed.
The usual ones:

| symptom | meaning |
|---|---|
| `401 Unauthorized` | `kaggle.json` missing, expired, or from another account. Creating a new token invalidates the old one. |
| `403 Forbidden` | Account not phone-verified — required for GPU and internet kernels. |
| `404 Not found` on `logs`/`fetch` | The kernel has never been pushed, or its first run has not finished, so there is no output yet. |
| Run fails at the clone step | The commit is not on GitHub, or the repo is private. |
| `DATA GUARD FAILED` | The cache dataset is not attached, is attached under a different name, or is empty. |
| Run "succeeds" in seconds with no outputs | Almost certainly the guard was off and the data root was empty. Turn `guard_data_root` on. |
| `Cannot access kernel ... permission denied` | Usually means the kernel has never been pushed — not that it is private. |

## Closed: the manifest and splits are now staged automatically

This used to be an open gap. `verify` passed and the 3,002 cached samples were
readable, but `outputs/manifest_sen2naipv2.csv` and
`outputs/splits_sen2naipv2.csv` were *not* in the working directory — they sat
in the mounted dataset and nothing copied them across. Harmless for `verify`,
which only reports it; **not** harmless for `baseline` or `train`, because
without the split CSV the loader recomputes a split in-process. That is
reproducible and completely wrong: adjacent NAIP tiles overlap, so a recomputed
split is not the geographic split the baseline was measured on, and every
number downstream is quietly incomparable.

Cell 3 of every generated notebook now calls
`src.utils.kaggle_session.stage_supporting_files`, which:

- resolves the mount through `src/utils/paths.py::kaggle_mount_path` — no
  `/kaggle/input` path is composed in the notebook or in the staging function,
  because that layout has already been wrong once here;
- derives both file names from config (`cfg.dataset.name`,
  `cfg.splits.output_name`), so switching datasets moves them automatically;
- copies both into `cfg.paths.manifest_dir`, and **aborts the run** if either is
  absent, naming the mount searched and listing what is actually in it.

It is in the template rather than in any job, so every job inherits it, and it
runs *before* the data guard — `scripts/verify_data_root.py` now treats a
missing manifest or split CSV as a FAILURE alongside its sample check, not as
the advisory note it used to print. A readable cache with no split file is no
longer a session that may start training.

Should the abort fire, the dataset was uploaded without the CSVs. Fix it at the
source rather than by hand-copying in a notebook:

```bash
python scripts/make_splits.py --config configs/base.yaml
python scripts/kaggle_upload.py stage
python scripts/kaggle_upload.py push
```

Related tooling: [`scripts/kaggle_upload.py`](../scripts/kaggle_upload.py)
publishes the SEN2NAIPv2 sample cache as the Kaggle Dataset every job mounts.
`verify` is the job to run after re-uploading it.
