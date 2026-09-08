"""Drive Kaggle GPU/CPU notebook jobs from the terminal, without a browser.

Why this exists
---------------
Training happens on Kaggle; the code lives here. Without this script the loop is
"edit locally, open a browser, paste or re-upload a notebook, click Run, watch a
web page, click Download" -- which is slow, unrepeatable, and above all *silent
about which code actually ran*. A notebook edited in the browser is code that
exists nowhere else.

So a job here is exactly one thing: **a notebook that clones this repository at
one specific commit and calls one entry point in ``scripts/``**. Notebooks are
generated from ``notebooks/templates/kaggle_job.py``, never hand-edited, and the
logic they call lives in ``src/utils/kaggle_session.py``. What ran is always a
SHA you can check out.

The two guards that pay for the whole script
--------------------------------------------
1. **``push`` refuses a dirty or unpushed working tree.** Kaggle clones from
   GitHub, so uncommitted work is simply not in the run -- the session succeeds
   while testing code you did not write. This failure has already cost a session
   on this project; it now costs a one-line refusal instead.
2. **Every GPU job runs ``scripts/verify_data_root.py`` before the expensive
   work, and aborts the run if it fails.** A session that cannot see its mounted
   dataset does not crash: it reports zero samples and finishes green. The
   budget is 30 GPU-hours a week and that mistake costs an hour of it.

The loop
--------
    python scripts/kaggle_run.py jobs                    # what is defined
    python scripts/kaggle_run.py push   --job train      # generate + push + start
    python scripts/kaggle_run.py status --job train --watch
    python scripts/kaggle_run.py logs   --job train      # tail-first
    python scripts/kaggle_run.py fetch  --job train      # outputs/kaggle/train/<ts>/

Jobs are defined in ``configs/kaggle_jobs.yaml``; the runner's own settings are
the ``kaggle_run`` block of ``configs/base.yaml``. See ``docs/kaggle_workflow.md``
for the plain-English version, including the few things that still need a
browser.

Requires the Kaggle API token at ``cfg.kaggle.credentials_file``
(``~/.kaggle/kaggle.json``), the same one ``scripts/kaggle_upload.py`` uses.
Every subcommand checks for it first and explains how to get one if it is
missing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Sibling script, imported for the pieces that are genuinely shared with the
# dataset uploader: how the Kaggle CLI is invoked, how credentials are read, and
# the output formatting. Duplicating those would let them drift, and the one
# that matters most -- "always [sys.executable, '-m', 'kaggle']" -- must never
# drift.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from omegaconf import DictConfig, OmegaConf  # noqa: E402

from kaggle_upload import (  # noqa: E402
    BAD,
    INFO,
    OK,
    SLUG_PATTERN,
    human_duration,
    kaggle_command,
    next_steps,
    preflight,
    rule,
)
from src.utils.config import (  # noqa: E402
    DEFAULT_CONFIG,
    add_standard_args,
    load_config,
)
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

# Kaggle's kernel-slug rules, checked locally so a bad name costs a second
# rather than a rejected push.
KERNEL_SLUG_MIN, KERNEL_SLUG_MAX = 5, 60
KERNEL_TITLE_MIN, KERNEL_TITLE_MAX = 5, 60

# The states the Kaggle kernels API reports. Anything not listed here is treated
# as still running rather than as success -- an unknown state must never be
# mistaken for "done, and fine".
STATE_SUCCESS = "complete"
STATE_FAILURES = ("error", "cancelacknowledged", "cancelrequested")
STATE_RUNNING = ("queued", "running")

# MEASURED against the live API: the reply is
#     shyamdwivedi0/drishtisr-verify has status "KernelWorkerStatus.RUNNING"
# not the bare word the older client returned. So the pattern accepts a dotted,
# underscored enum name as well as a plain word, and :func:`normalise_state`
# reduces both to the same token. Getting this wrong is not a crash: an
# unrecognised state is treated as "still running", so `--watch` would poll a
# finished job until its timeout and never report the result.
_STATE_RE = re.compile(r'status\s+"?([A-Za-z_.]+)"?', re.IGNORECASE)
_TOKEN_RE = re.compile(r"\{\{[A-Z_]+\}\}")


class RunError(RuntimeError):
    """A failure already translated into plain English.

    Raised with a message that says what happened, why, and what to do about it.
    ``main`` prints it without a traceback: a stack trace is the right output for
    a bug in this script and the wrong output for an unpushed commit.
    """


# -- config -----------------------------------------------------------------


def repo_path(value: Any) -> Path:
    """Resolve a config path against the repository root.

    Args:
        value: A path from config. Absolute values are returned unchanged.

    Returns:
        An absolute path. Nothing is created here.
    """
    path = Path(str(value))
    return path if path.is_absolute() else repo_root() / path


def load_jobs(cfg: Any) -> DictConfig:
    """Load ``configs/kaggle_jobs.yaml`` and merge ``defaults`` into each job.

    Merging happens here rather than in the template so that adding a job is one
    edit to one file. Note that list-valued keys (``dataset_sources``,
    ``pip_packages``, ``entry_args``) are REPLACED by a job that sets them, not
    concatenated -- a job stating a partial mount list and silently inheriting
    the rest would be the kind of surprise this project cannot afford.

    Args:
        cfg: The loaded base config; reads ``kaggle_run.jobs_file``.

    Returns:
        A mapping of job name to its fully-resolved definition.

    Raises:
        RunError: The file is missing, unparseable, or defines no jobs.
    """
    path = repo_path(cfg.kaggle_run.jobs_file)
    if not path.is_file():
        raise RunError(
            f"Job definitions not found at {path}.\n"
            "This file lists every Kaggle job (verify, baseline, train, ...). "
            "It is version-controlled; if it is missing, restore it with "
            "`git checkout configs/kaggle_jobs.yaml`."
        )
    try:
        raw = OmegaConf.load(path)
    except Exception as exc:  # OmegaConf raises several YAML error types
        raise RunError(
            f"Could not parse {path}: {exc}\n"
            "It must be valid YAML with a top-level 'jobs' mapping."
        ) from exc

    if "jobs" not in raw or not raw.jobs:
        raise RunError(
            f"{path} defines no jobs. It needs a top-level 'jobs:' mapping with "
            "at least one entry, e.g. 'verify'."
        )
    defaults = raw.get("defaults") or OmegaConf.create({})
    merged = OmegaConf.create(
        {name: OmegaConf.merge(defaults, job) for name, job in raw.jobs.items()}
    )
    for name, job in merged.items():
        validate_job(name, job, path, kernel_slug(cfg, name), cfg)
    return merged


# Keys every job must end up with, whether from 'defaults' or its own block.
# 'title', 'entry' and 'enable_gpu' deliberately have NO default: a job that
# does not say whether it wants a GPU is a job whose cost is unknown, and
# guessing either way is the wrong thing to do with a 30-hour weekly budget.
REQUIRED_JOB_KEYS = (
    "title", "enable_gpu", "enable_internet",
    "template", "workdir", "guard_script", "guard_data_root",
)

# A job names its entry point EITHER as a script path (`entry`) or as an
# importable module run with -m (`entry_module`). Exactly one, checked in
# validate_job(): a job with both is ambiguous about what actually runs, and a
# job with neither renders a notebook whose last cell does nothing.
ENTRY_KEYS = ("entry", "entry_module")


def slugify(text: str) -> str:
    """Reduce text to a Kaggle slug the way Kaggle does.

    Lowercase, every run of non-alphanumeric characters becomes a single hyphen,
    and hyphens are trimmed from the ends. ``"DrishtiSR verify data root"``
    becomes ``"drishtisr-verify-data-root"``.
    """
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def check_accelerator(name: str, job: Any, path: Path, cfg: Any) -> None:
    """Refuse a GPU job that asks for an accelerator we know does not work.

    Called from :func:`validate_job`, which runs at jobs-file LOAD time. That
    placement is the whole point. The T4-not-P100 rule previously lived in a
    pytest, in two code comments and in one config default, and nothing checked
    an actual job definition: a job overriding ``accelerator: NvidiaTeslaP100``
    would have been generated and pushed without a word. The one job that
    carried the rule in its comments, ``train``, named an entry point
    (``scripts/train.py``) that has never existed, so it could not be pushed and
    the rule was never reached at all.

    Load time also means this fires for ``jobs``, ``status`` and ``logs``, not
    only for ``push`` -- and before :func:`generate_kernel`'s entry-point check,
    which is what used to stand in front of it.

    Args:
        name: The job's key in the jobs file.
        job: The merged job definition.
        path: The jobs file, for the error message.
        cfg: The loaded base config; reads ``kaggle_run.allowed_accelerators``
            and ``kaggle_run.forbidden_accelerators``.

    Raises:
        RunError: ``enable_gpu`` is true and the requested accelerator is not
            on the allowlist. A CPU job's accelerator is ignored rather than
            checked, because :func:`kernel_metadata` never writes one.
    """
    if not job.get("enable_gpu"):
        return

    requested = str(job.get("accelerator") or cfg.kaggle_run.accelerator)
    allowed = [str(value) for value in cfg.kaggle_run.allowed_accelerators]
    if requested in allowed:
        return

    forbidden = cfg.kaggle_run.get("forbidden_accelerators") or {}
    reason = str(forbidden.get(requested, "")).strip()
    raise RunError(
        f"The job {name!r} in {path} asks for accelerator {requested!r}, which "
        "is not allowed.\n"
        + (f"\n  {reason}\n" if reason else "")
        + "\n"
        f"Allowed: {', '.join(allowed)}.\n"
        "\n"
        "These are kagglesdk ApiSaveKernelRequest.machine_shape values. Kaggle "
        "does not reject a machine_shape it fails to recognise -- it silently "
        "falls back to whatever GPU it feels like, which is how a run A session "
        "landed on a P100 and died on its first CUDA op. So an unrecognised "
        "value is refused here rather than sent.\n"
        "\n"
        f"Fix: drop the job's 'accelerator' key to inherit "
        f"{str(cfg.kaggle_run.accelerator)!r}, or set it to one of the allowed "
        "values. To permit a new one, add it to "
        "cfg.kaggle_run.allowed_accelerators -- deliberately, with a note."
    )


def validate_job(
    name: str, job: Any, path: Path, expected_slug: str, cfg: Any
) -> None:
    """Check one job definition has everything the template needs, is named
    consistently, and asks for an accelerator that exists.

    Runs at load time so a half-written job produces one plain sentence here,
    rather than an OmegaConf attribute error from somewhere inside template
    rendering, or -- worse -- a pushed kernel missing a setting.

    Args:
        name: The job's key in the jobs file.
        job: The merged job definition.
        path: The jobs file, for the error message.
        expected_slug: The bare kernel slug this job must resolve to.
        cfg: The loaded base config, for the accelerator allowlist.

    Raises:
        RunError: A required key is absent, the title does not slugify to the
            kernel slug, or a GPU job asks for an accelerator that is not on
            ``cfg.kaggle_run.allowed_accelerators``. None of the three is
            cosmetic -- see the notes at each check.
    """
    declared = [key for key in ENTRY_KEYS if job.get(key)]
    if len(declared) != 1:
        raise RunError(
            f"The job {name!r} in {path} must set exactly one of "
            f"{' or '.join(ENTRY_KEYS)}; it sets "
            + (", ".join(declared) if declared else "neither")
            + "." + "\n"
            "Use 'entry' for a script run by path (scripts/run_baseline.py) and "
            "'entry_module' for one run as `python -m <module>` "
            "(drishtisr.train). The module form is what code importing both "
            "`drishtisr.*` and `src.*` needs, because only -m puts the clone "
            "root on sys.path."
        )

    missing = [key for key in REQUIRED_JOB_KEYS if key not in job]
    if missing:
        raise RunError(
            f"The job {name!r} in {path} is missing: {', '.join(missing)}.\n"
            "Keys shared by every job belong in the 'defaults:' block at the top "
            "of that file; 'title', the entry point and 'enable_gpu' are per-job "
            "and deliberately have no default."
        )

    # THE DRIFT GUARD. MEASURED, not assumed: pushing a kernel whose
    # kernel-metadata.json says id "shyamdwivedi0/drishtisr-verify" while its
    # title is "DrishtiSR verify data root" creates the kernel at
    # "shyamdwivedi0/drishtisr-verify-data-root". Kaggle slugifies the TITLE and
    # the id loses; the CLI prints a warning about it and pushes anyway.
    #
    # The consequence is not cosmetic. The push succeeds, and then `status`,
    # `logs` and `fetch` -- which all address the id -- fail with a permission
    # error that reads like the kernel is private. The run itself is fine and
    # completely unreachable from this tool.
    #
    # So the two are required to agree here, before anything is pushed. Which
    # field Kaggle honours then stops mattering: both name the same kernel.
    check_accelerator(name, job, path, cfg)

    actual = slugify(str(job.title))
    if actual != expected_slug:
        raise RunError(
            f"The job {name!r} in {path} has a title that does not match its "
            "kernel slug.\n"
            f"  title:          {str(job.title)!r}\n"
            f"  slugifies to:   {actual!r}\n"
            f"  kernel slug:    {expected_slug!r}\n"
            "\n"
            "Kaggle derives the kernel's URL from the TITLE, not from the 'id' in "
            "kernel-metadata.json. If they disagree, the push succeeds and lands "
            "at the title's slug, while status/logs/fetch look for the id's slug "
            "and report a permission error -- a run that works and cannot be "
            "reached.\n"
            "\n"
            f"Fix: set this job's title to something that slugifies to "
            f"{expected_slug!r}, e.g. {expected_slug.replace('-', ' ').title()!r}, "
            "or change cfg.kaggle_run.slug_prefix."
        )


def resolve_job(cfg: Any, name: str) -> DictConfig:
    """Look up one job by name, or explain what the valid names are.

    Args:
        cfg: The loaded base config.
        name: The ``--job`` value.

    Returns:
        The job definition with defaults merged in.

    Raises:
        RunError: No job by that name.
    """
    jobs = load_jobs(cfg)
    if name not in jobs:
        available = ", ".join(sorted(jobs.keys()))
        raise RunError(
            f"No job named {name!r}. Defined jobs: {available}.\n"
            f"Add one by editing {cfg.kaggle_run.jobs_file}, or run "
            "`python scripts/kaggle_run.py jobs` to see them with their current "
            "state on Kaggle."
        )
    return jobs[name]


def validate_kernel_slug(slug: str) -> str:
    """Check a kernel slug against Kaggle's rules, locally.

    Args:
        slug: The bare kernel name, with no owner prefix.

    Returns:
        The slug unchanged.

    Raises:
        RunError: Wrong length, or characters Kaggle rejects.
    """
    if not KERNEL_SLUG_MIN <= len(slug) <= KERNEL_SLUG_MAX:
        raise RunError(
            f"The kernel slug {slug!r} is {len(slug)} characters; Kaggle requires "
            f"{KERNEL_SLUG_MIN}-{KERNEL_SLUG_MAX}. It is built as "
            "cfg.kaggle_run.slug_prefix + the job name, so shorten or lengthen "
            "one of those."
        )
    if not SLUG_PATTERN.match(slug):
        raise RunError(
            f"The kernel slug {slug!r} would be rejected by Kaggle. Use lowercase "
            "letters, digits, and single hyphens between them -- no spaces, "
            "underscores, capitals, or leading/trailing hyphens. It is built as "
            "cfg.kaggle_run.slug_prefix + the job name."
        )
    return slug


def kernel_slug(cfg: Any, job_name: str) -> str:
    """The bare kernel slug for a job, e.g. ``drishtisr-train``."""
    return validate_kernel_slug(f"{cfg.kaggle_run.slug_prefix}{job_name}")


def kernel_id(cfg: Any, username: str, job_name: str) -> str:
    """The fully-qualified kernel id Kaggle addresses, ``<username>/<slug>``.

    The owner comes from ``kaggle.json``, never from a config file, so the repo
    carries no username and works for anyone who clones it.
    """
    return f"{username}/{kernel_slug(cfg, job_name)}"


def validate_title(title: str) -> str:
    """Check a kernel title's length locally.

    Raises:
        RunError: Kaggle rejects titles outside 5-60 characters.
    """
    if not KERNEL_TITLE_MIN <= len(title) <= KERNEL_TITLE_MAX:
        raise RunError(
            f"The job title {title!r} is {len(title)} characters; Kaggle requires "
            f"{KERNEL_TITLE_MIN}-{KERNEL_TITLE_MAX}. Edit the job's 'title' in "
            "configs/kaggle_jobs.yaml."
        )
    return title


# -- git: the guard that stops a run of the wrong code ----------------------


def git(*args: str, what: str = "a git command") -> str:
    """Run git in the repository and return its stdout, stripped.

    Args:
        *args: Arguments after ``git``.
        what: Description used in the error message.

    Returns:
        Captured stdout with trailing whitespace removed.

    Raises:
        RunError: git is missing, or exited non-zero. The raw stderr is included
            -- git's own messages are usually the clearest explanation available.
    """
    command = ["git", "-C", str(repo_root()), *args]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except FileNotFoundError as exc:
        raise RunError(
            "git is not on PATH, so this script cannot tell which commit Kaggle "
            "would run. Install git, or run this from a shell that has it."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RunError(f"{what} timed out after 120 s: {' '.join(command)}") from exc

    if completed.returncode != 0:
        raise RunError(
            f"{what} failed (exit code {completed.returncode}):\n"
            f"  {' '.join(command)}\n\n"
            f"{(completed.stderr or completed.stdout or '(no output)').strip()}"
        )
    return completed.stdout.strip()


def remote_url(cfg: Any) -> str:
    """The URL the generated notebook clones from.

    ``cfg.kaggle_run.repo_url`` wins when set; otherwise the ``origin`` remote is
    read from git, which is right for every ordinary checkout.

    Raises:
        RunError: No origin remote, or an SSH URL. Kaggle clones anonymously with
            no keys and no credential helper, so ``git@github.com:...`` hangs the
            session on an auth prompt instead of failing fast.
    """
    configured = cfg.kaggle_run.get("repo_url")
    url = str(configured).strip() if configured else git(
        "remote", "get-url", "origin", what="reading the origin remote URL"
    )
    if url.startswith("git@") or url.startswith("ssh://"):
        raise RunError(
            f"The clone URL is {url!r}, which is an SSH URL.\n"
            "The Kaggle session clones anonymously -- it has no SSH key and no "
            "credential helper -- so this would hang or fail inside the run.\n"
            "Fix: use the HTTPS URL, either by changing the remote\n"
            "  git remote set-url origin https://github.com/<you>/<repo>.git\n"
            "or by setting cfg.kaggle_run.repo_url in configs/base.yaml."
        )
    return url


def require_publishable_tree(
    cfg: Any, logger: Any, enforce: bool = True, allow_dirty: bool = False
) -> str:
    """Refuse to push unless the exact local code is on the remote.

    This is the check that exists because of a real lost session. Kaggle clones
    from GitHub at a SHA; it cannot see the working tree. So a run started from a
    dirty checkout silently tests the last committed code, succeeds, and produces
    numbers for code that is not the code being edited. There is no symptom.

    Three things must hold:

    1. the working tree has no modified or untracked files;
    2. ``git fetch`` succeeds, so the remote refs are current (the check is
       worthless against a week-old idea of what the remote has);
    3. HEAD is reachable from at least one remote branch.

    ``git fetch`` is the only mutating call, and it touches only remote-tracking
    refs -- never the working tree, never a local branch.

    Args:
        cfg: The loaded base config.
        logger: Logger.
        enforce: False under ``--smoke``, where the same checks run and report
            but a dirty tree is a warning. A smoke run contacts no network, so it
            cannot start a run of the wrong code.
        allow_dirty: Downgrade check 1 -- and ONLY check 1 -- to a printed
            warning. Checks 2 and 3 still hold, so what runs is still a commit
            that is provably on the remote.

            This exists because more than one session edits this repository at
            once, and check 1 is a whole-tree check: another session's work in
            progress blocks a launch that has nothing to do with it, and the
            only ways out are committing someone else's half-finished files or
            stashing their live work. Both are worse than the risk here.

            It does NOT weaken the guarantee the check was written for. That
            guarantee is "you cannot be unaware that Kaggle runs a commit rather
            than your screen", and the flag is explicit, prints every dirty
            path, and prints the SHA actually being run. What it costs is the
            automatic proof that the dirty paths are irrelevant to this job --
            so the caller is the one asserting that, deliberately.

    Returns:
        The 40-character HEAD SHA.

    Raises:
        RunError: Any check failed and ``enforce`` is True.
    """
    def refuse(message: str) -> None:
        if enforce:
            raise RunError(message)
        print(f"{BAD} --smoke: would REFUSE to push. {message.splitlines()[0]}")
        logger.warning("smoke: publishable-tree check would have refused")

    sha = git("rev-parse", "HEAD", what="reading the current commit")
    branch = git("rev-parse", "--abbrev-ref", "HEAD", what="reading the branch name")

    if not enforce:
        # --smoke must not touch the network, and `git fetch` is a network call.
        # So the local half of the check runs for real and the remote half is
        # skipped, loudly. Nothing can start a wrong-code run from here anyway:
        # a smoke push never reaches `kaggle kernels push`.
        print(f"{INFO}--smoke: skipping `git fetch` and the remote-branch check;")
        print(f"{INFO}smoke mode contacts no network. A real push runs both.")

    dirty = git("status", "--porcelain", what="checking the working tree")
    if dirty and allow_dirty:
        listing = "\n".join(f"      {line}" for line in dirty.splitlines()[:20])
        more = "" if len(dirty.splitlines()) <= 20 else "\n      ..."
        print(f"{BAD} --allow-dirty: the working tree is NOT clean.")
        print(f"{INFO}Kaggle will run commit {sha[:12]}, not the files below.")
        print(listing + more)
        print(f"{INFO}You are asserting these paths do not affect this job.")
        logger.warning(
            "--allow-dirty: pushing %s with %d dirty path(s) in the tree",
            sha[:12],
            len(dirty.splitlines()),
        )
    elif dirty:
        listing = "\n".join(f"      {line}" for line in dirty.splitlines()[:20])
        more = "" if len(dirty.splitlines()) <= 20 else "\n      ..."
        refuse(
            "Your working tree has uncommitted changes, so the Kaggle run would "
            "NOT contain them.\n"
            "\n"
            "Kaggle clones this repository from GitHub at a commit. It cannot "
            "see your local edits. Pushing now would start a run of the last "
            "committed code while you believe it is running what is on screen -- "
            "and nothing in the log would say so.\n"
            "\n"
            f"{listing}{more}\n"
            "\n"
            "Fix: commit and push, then push the job.\n"
            "  git add -A && git commit -m \"...\" && git push\n"
            "\n"
            "If those paths belong to another session and have nothing to do "
            "with this job, re-run with --allow-dirty. The run still uses a "
            "commit that is on the remote; you are asserting the dirty paths "
            "do not affect it."
        )
    else:
        print(f"{OK} working tree is clean")

    if not enforce:
        print(f"{OK} commit {sha}")
        print(f"{INFO}branch:  {branch}")
        return sha

    try:
        git("fetch", "--quiet", "origin", what="fetching the remote refs")
        print(f"{OK} fetched origin (remote-tracking refs only)")
    except RunError as exc:
        refuse(
            "Could not fetch from origin, so this script cannot prove your "
            "commit is on the remote -- and Kaggle can only run what IS on the "
            "remote.\n"
            "\n"
            "Check your network and that the remote is reachable, then retry.\n"
            "\n"
            f"{exc}"
        )

    contains = git(
        "branch", "--remotes", "--contains", sha, what="checking whether the commit is pushed"
    )
    if not contains.strip():
        refuse(
            f"Commit {sha[:12]} is not on any remote branch, so Kaggle cannot "
            "clone it.\n"
            "\n"
            f"You are on branch {branch!r}. The run would fail at the clone step, "
            "or -- worse, if a stale kernel already exists -- Kaggle would simply "
            "run whatever was pushed last.\n"
            "\n"
            "Fix:\n"
            f"  git push origin {branch}"
        )
    else:
        remotes = ", ".join(line.strip() for line in contains.splitlines()[:4])
        print(f"{OK} commit is on the remote: {remotes}")

    subject = git("log", "-1", "--pretty=%s", what="reading the commit subject")
    print(f"{OK} commit {sha}")
    print(f"{INFO}branch:  {branch}")
    print(f"{INFO}subject: {subject}")
    logger.info("Push will run commit %s from branch %s", sha, branch)
    return sha


# -- notebook generation ----------------------------------------------------


def render_template(template_text: str, tokens: Dict[str, str]) -> str:
    """Substitute ``{{TOKEN}}`` placeholders and verify none are left.

    Substitution is plain text replacement rather than ``str.format`` because the
    template is Python source full of braces, and a format string would have to
    escape every one of them -- a rule that is broken the first time someone adds
    a dict literal.

    Args:
        template_text: The raw template file contents.
        tokens: Mapping of token name (without braces) to replacement text.
            Values are inserted verbatim, so anything that must be a Python
            literal in the generated notebook is passed already ``repr``'d.

    Returns:
        The rendered source.

    Raises:
        RunError: A ``{{TOKEN}}`` survived substitution. That would be a notebook
            with a syntax error, discovered on Kaggle after the queue wait rather
            than here.
    """
    rendered = template_text
    for name, value in tokens.items():
        rendered = rendered.replace("{{" + name + "}}", value)

    # Scan only from the first cell marker onward. Everything above it is the
    # template's own header comment, which is dropped by :func:`to_notebook` and
    # is allowed to write "{{TOKEN}}" while explaining what a token is.
    marker = rendered.find("# %%")
    body = rendered[marker:] if marker >= 0 else rendered
    leftover = sorted(set(_TOKEN_RE.findall(body)))
    if leftover:
        raise RunError(
            "The notebook template has placeholders this script does not fill: "
            f"{', '.join(leftover)}.\n"
            "Either the template gained a token, or a token was renamed. Both are "
            "fixed in notebooks/templates/kaggle_job.py and in "
            "scripts/kaggle_run.py::build_tokens -- they must agree."
        )
    return rendered


def to_notebook(source: str) -> Dict[str, Any]:
    """Split rendered template source into an ``.ipynb`` document.

    Cells are delimited by lines starting with ``# %%``; ``# %% [markdown]``
    starts a markdown cell, whose following ``# `` comment prefixes are stripped.
    Anything before the first marker is the template's own header comment and is
    dropped -- it documents the template, not the run.

    Args:
        source: Rendered template text.

    Returns:
        An nbformat 4 notebook as a plain dict, ready for ``json.dump``.

    Raises:
        RunError: The template contained no cell markers, which would produce an
            empty notebook that Kaggle accepts and runs to completion doing
            nothing.
    """
    cells: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    lines: List[str] = []

    def flush() -> None:
        if current is None:
            return
        body = "\n".join(lines).strip("\n")
        if not body:
            return
        if current["cell_type"] == "markdown":
            body = "\n".join(
                line[2:] if line.startswith("# ") else ("" if line.strip() == "#" else line)
                for line in body.splitlines()
            )
        cell: Dict[str, Any] = {
            "cell_type": current["cell_type"],
            "metadata": {},
            "source": body.splitlines(keepends=True),
        }
        if current["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    for line in source.splitlines():
        if line.startswith("# %%"):
            flush()
            current = {"cell_type": "markdown" if "[markdown]" in line else "code"}
            lines = []
            continue
        if current is not None:
            lines.append(line)
    flush()

    if not cells:
        raise RunError(
            "The notebook template produced no cells. It must contain '# %%' cell "
            "markers; see notebooks/templates/kaggle_job.py."
        )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def config_path_for(job: Any) -> str:
    """The config path a job's cells should use, taken from the job's own args.

    Derived rather than declared, so a job that runs against a different config
    cannot end up staging its supporting files from the wrong one. The job's
    ``entry_args`` are the authority: whatever ``--config`` the entry point is
    given is what the notebook's staging cell is given too.

    Args:
        job: The merged job definition.

    Returns:
        The ``--config`` value from ``entry_args``, falling back to
        ``guard_args``, and finally to the repository default. The fallback
        matters for a job whose entry point takes no ``--config`` at all.
    """
    for key in ("entry_args", "guard_args"):
        args = [str(arg) for arg in (job.get(key) or [])]
        for index, arg in enumerate(args):
            if arg == "--config" and index + 1 < len(args):
                return args[index + 1]
            if arg.startswith("--config="):
                return arg.split("=", 1)[1]
    return DEFAULT_CONFIG


def build_tokens(
    cfg: Any, job_name: str, job: Any, sha: str, url: str
) -> Dict[str, str]:
    """Assemble the template substitutions for one job.

    Values destined for Python code are ``repr``'d here, so the template can
    write ``{{ENTRY_CALL}}`` and get a correctly quoted
    string with no escaping rules of its own.

    Args:
        cfg: The loaded base config.
        job_name: The job's key in the jobs file.
        job: The merged job definition.
        sha: The commit the notebook will check out.
        url: The clone URL.

    Returns:
        Token name to replacement text.
    """
    entry = str(job.get("entry") or "")
    entry_module = str(job.get("entry_module") or "")
    entry_args = [str(arg) for arg in (job.get("entry_args") or [])]
    guard_args = [str(arg) for arg in (job.get("guard_args") or [])]
    runtime_min = int(job.get("expected_runtime_min") or 0)
    accelerator = str(job.get("accelerator") or cfg.kaggle_run.accelerator)

    return {
        "JOB_NAME": job_name,
        # Kaggle slugifies the title into the kernel URL, so titles are terse and
        # this is where a job says what it actually does. Collapsed to one line:
        # it lands inside a markdown cell built from comment lines.
        "DESCRIPTION": " ".join(str(job.get("description") or "").split())
        or "(no description set for this job)",
        "GENERATED_AT": _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "GIT_SHA": sha,
        "GIT_SHA_LITERAL": repr(sha),
        "REPO_URL": repr(url),
        "WORKDIR": repr(str(job.workdir)),
        "PIP_PACKAGES": repr([str(pkg) for pkg in (job.get("pip_packages") or [])]),
        # Installed in a second pass with --no-deps, for packages whose
        # declared dependencies would re-resolve something the Kaggle image
        # already has correctly -- lpips would drag in a torch of its own.
        "PIP_PACKAGES_NO_DEPS": repr(
            [str(pkg) for pkg in (job.get("pip_packages_no_deps") or [])]
        ),
        "GUARD_SCRIPT": repr(str(job.guard_script)),
        "GUARD_ARGS": repr(guard_args),
        "GUARD_ENABLED": repr(bool(job.guard_data_root)),
        # The config the staging cell loads to resolve the mount and work out the
        # manifest/split file names. Taken from the job's own arguments by
        # config_path_for(), so it cannot disagree with the entry point.
        "CONFIG_PATH_LITERAL": repr(config_path_for(job)),
        "ENTRY_ARGS": repr(entry_args),
        # The generated notebook calls ONE of run_entry/run_module. Deciding it
        # here rather than with an `if` in the template keeps the template free
        # of job-shaped logic, which is the rule this file exists to enforce.
        "ENTRY_CALL": (
            f"run_module({entry_module!r}, args={entry_args!r})"
            if entry_module
            else f"run_entry({entry!r}, args={entry_args!r})"
        ),
        "ENTRY_DISPLAY": " ".join(
            ([f"python -m {entry_module}"] if entry_module else [entry]) + entry_args
        ),
        "OUTPUT_DIRS": repr([str(item) for item in (job.get("output_dirs") or [])]),
        "ACCELERATOR_DISPLAY": accelerator if job.enable_gpu else "CPU only",
        "EXPECTED_RUNTIME_DISPLAY": (
            human_duration(runtime_min * 60) if runtime_min else "unknown"
        ),
    }


def kernel_metadata(
    cfg: Any, username: str, job_name: str, job: Any
) -> Dict[str, Any]:
    """Build ``kernel-metadata.json`` for one job.

    Generated rather than hand-maintained, for the same reason the notebook is:
    the id, the accelerator, and the mounted datasets are facts about the job,
    and a file edited by hand drifts from the job definition without saying so.

    The accelerator deserves its own note. See ``ACCELERATOR_NOTE`` below -- in
    short, **T4 and never P100**.

    Args:
        cfg: The loaded base config.
        username: Kaggle username, read from ``kaggle.json``.
        job_name: The job's key.
        job: The merged job definition.

    Returns:
        The metadata document as a dict.
    """
    sources = []
    for entry in job.get("dataset_sources") or []:
        text = str(entry).strip()
        # Bare slugs get the owner prefix from the credentials, so the jobs file
        # carries no username and works for anyone who clones this repo.
        sources.append(text if "/" in text else f"{username}/{text}")

    metadata: Dict[str, Any] = {
        "id": kernel_id(cfg, username, job_name),
        "title": validate_title(str(job.title)),
        "code_file": f"{job_name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": bool(job.get("is_private", True)),
        "enable_gpu": bool(job.enable_gpu),
        "enable_internet": bool(job.enable_internet),
        "dataset_sources": sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if metadata["enable_gpu"]:
        # ------------------------------------------------------------------
        # ACCELERATOR_NOTE -- T4, NEVER P100.
        #
        # Kaggle's default image ships a PyTorch built for cu128, and that build
        # contains no Pascal (sm_60) kernels. On a P100 the resulting failure is
        # far worse than "no GPU": torch imports cleanly, torch.cuda.is_available()
        # returns True, the device reports itself as a Tesla P100, and then the
        # FIRST actual CUDA op dies with cudaErrorNoKernelImageForDevice -- after
        # the session has already queued, booted, cloned, and installed. Every
        # cheap pre-check passes and the expensive one fails.
        #
        # T4 is sm_75, which that same build covers. So GPU jobs pin
        # cfg.kaggle_run.accelerator = NvidiaTeslaT4 and a job may override it
        # only deliberately.
        #
        # THE KEY IS "machine_shape", NOT "accelerator". MEASURED 2026-09-07,
        # the expensive way: the first runA job was pushed with
        # "accelerator": "NvidiaTeslaT4" and Kaggle ran it on a Tesla
        # P100-PCIE-16GB anyway, which died on the first CUDA op exactly as the
        # note above predicts. The value was right and the key was wrong --
        # kaggle_api_extended.py does
        #     request.machine_shape = acc or get_or_default(meta, "machine_shape")
        # so an "accelerator" key is read by nothing, no error is raised, and
        # the session silently falls back to Kaggle's default GPU. A wrong
        # accelerator is not a warning here; it is the whole run.
        #
        # Valid values, from kagglesdk ApiSaveKernelRequest.machine_shape:
        # NvidiaTeslaT4, NvidiaTeslaP100, Tpu1VmV38.
        # ------------------------------------------------------------------
        metadata["machine_shape"] = str(
            job.get("accelerator") or cfg.kaggle_run.accelerator
        )
    return metadata


def entry_point_path(job: Any) -> Path:
    """The file a job's entry point resolves to, for the pre-flight check.

    Both entry forms are checkable here, in milliseconds, instead of on Kaggle
    after the queue wait and the pip installs. A module is mapped to a file
    through the same rule the ``drishtisr`` alias package uses -- its
    ``__path__`` points at ``src/`` -- so ``drishtisr.train`` is ``src/train.py``.
    Any other top-level package is resolved as a plain dotted path from the
    repository root.

    Args:
        job: The merged job definition.

    Returns:
        The absolute path the entry point is expected to live at. Existence is
        the caller's to check, so it can raise with its own message.
    """
    entry = str(job.get("entry") or "")
    if entry:
        return repo_path(entry)

    parts = str(job.get("entry_module") or "").split(".")
    if parts and parts[0] == "drishtisr":
        parts = ["src"] + parts[1:]
    return repo_path("/".join(parts) + ".py")


def generate_kernel(
    cfg: Any, username: str, job_name: str, job: Any, sha: str, logger: Any
) -> Tuple[Path, Dict[str, Any]]:
    """Write the notebook and ``kernel-metadata.json`` for a job.

    Args:
        cfg: The loaded base config.
        username: Kaggle username.
        job_name: The job's key.
        job: The merged job definition.
        sha: The commit the notebook will check out.
        logger: Logger.

    Returns:
        ``(build_directory, metadata_dict)``. The directory is what
        ``kaggle kernels push -p`` is pointed at.

    Raises:
        RunError: The template is missing, or a placeholder was left unfilled.
    """
    template_path = repo_path(job.template)
    if not template_path.is_file():
        raise RunError(
            f"Notebook template not found at {template_path}.\n"
            "It is version-controlled; restore it with "
            f"`git checkout {job.template}`, or point the job's 'template' key "
            "at the right file."
        )

    entry_path = entry_point_path(job)
    if not entry_path.is_file():
        raise RunError(
            "This job's entry point does not exist: "
            f"{job.get('entry') or '-m ' + str(job.get('entry_module'))}\n"
            f"  Looked at: {entry_path}\n"
            "\n"
            "The Kaggle session would clone the repo, install packages, pass the "
            "data guard, and only then fail on a missing file -- minutes of "
            "session time to learn something checkable here in milliseconds.\n"
            "\n"
            "Either write that script, or point the job's 'entry' key at one that "
            "exists. Defined entry points: "
            + ", ".join(sorted(p.name for p in (repo_root() / "scripts").glob("*.py")))
        )

    url = remote_url(cfg)
    tokens = build_tokens(cfg, job_name, job, sha, url)
    notebook = to_notebook(render_template(template_path.read_text(encoding="utf-8"), tokens))
    metadata = kernel_metadata(cfg, username, job_name, job)

    build_dir = repo_path(cfg.kaggle_run.kernel_build_dir) / job_name
    build_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = build_dir / metadata["code_file"]
    notebook_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (build_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    logger.info(
        "Generated %s (%d cells) and kernel-metadata.json in %s",
        notebook_path.name,
        len(notebook["cells"]),
        build_dir,
    )
    print(f"{OK} generated {notebook_path.name} ({len(notebook['cells'])} cells)")
    print(f"{OK} generated kernel-metadata.json")
    print(f"{INFO}build folder: {build_dir}")
    return build_dir, metadata


# -- the kaggle CLI ---------------------------------------------------------


def run_kaggle(
    args: Sequence[str],
    timeout_s: int,
    logger: Any,
    what: str,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess:
    """Run a Kaggle CLI command, translating any failure into plain English.

    Always invoked as ``[sys.executable, "-m", "kaggle"]`` via
    :func:`kaggle_upload.kaggle_command`: the ``kaggle`` console script is not on
    PATH on this machine, and going through ``-m`` guarantees the package
    resolves inside the same interpreter that is running this script.

    Args:
        args: Arguments after the ``kaggle`` prefix.
        timeout_s: Seconds before the call is treated as hung.
        logger: Logger for the command actually executed.
        what: Short description used in error messages, e.g. ``"the push"``.
        allow_failure: Return the failed process instead of raising. Used only by
            callers that interpret the failure themselves -- ``jobs``, where "no
            such kernel" means "never pushed", not an error.

    Returns:
        The completed process.

    Raises:
        RunError: The call failed and ``allow_failure`` is False.
    """
    command = kaggle_command() + list(args)
    logger.info("Running: %s", " ".join(str(part) for part in command))
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        raise RunError(
            f"{what.capitalize()} timed out after {human_duration(timeout_s)} with "
            "no response from Kaggle.\n"
            "  - Check https://status.kaggle.com\n"
            "  - Raise cfg.kaggle_run.cli_timeout_s (or fetch_timeout_s for "
            "downloads) if the operation is genuinely this slow.\n"
            "A timed-out status or logs call changes nothing on Kaggle; a "
            "timed-out push may or may not have landed, so check with "
            "`kaggle_run.py jobs` before pushing again."
        ) from exc
    except OSError as exc:
        raise RunError(
            f"Could not start the Kaggle client for {what}: {exc}\n"
            f"Install it with `{sys.executable} -m pip install kaggle`."
        ) from exc

    if completed.returncode != 0 and not allow_failure:
        raise RunError(translate_kaggle_failure(completed, what))
    return completed


def translate_kaggle_failure(completed: subprocess.CompletedProcess, what: str) -> str:
    """Turn a Kaggle CLI failure into an explanation and a fix.

    Handled specifically: 401 and 403 credential problems, 404 (a kernel that was
    never pushed, or an output pull before the first run), the GPU-quota
    rejection, dataset-source mistakes, and network errors.

    Args:
        completed: The failed process, with output captured.
        what: Short description, for the first line.

    Returns:
        A multi-line message. Always ends with the raw output, so an unrecognised
        failure is fully reported rather than swallowed.
    """
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}".strip()
    lowered = output.lower()

    if "401" in lowered or "unauthorized" in lowered:
        explanation = (
            "Kaggle rejected your credentials (401 Unauthorized).\n"
            "  - Fix: https://www.kaggle.com/settings/account -> API -> 'Create "
            "New Token', and replace ~/.kaggle/kaggle.json with the download. "
            "Creating a new token invalidates the old one, so a copy on another "
            "machine stops working."
        )
    elif "403" in lowered or "forbidden" in lowered:
        explanation = (
            "Kaggle accepted your token but refused the action (403 Forbidden).\n"
            "  - Running kernels with internet or a GPU requires a "
            "phone-verified account. Sign in, open your profile settings, and "
            "complete verification.\n"
            "  - It can also mean the kernel belongs to someone else."
        )
    elif (
        "404" in lowered or "not found" in lowered or "cannot access kernel" in lowered
    ):
        explanation = (
            "Kaggle could not find that kernel, or would not let you see it.\n"
            "  - MEASURED: a kernel that has never been pushed answers "
            "\"Cannot access kernel ... (Permission 'kernels.get' was denied)\", "
            "not a 404. The wording says permission; the cause is almost always "
            "that it does not exist yet.\n"
            "  - Most likely it has never been pushed. Run "
            "`python scripts/kaggle_run.py push --job <name>` first.\n"
            "  - `logs` and `fetch` also 404 while a kernel exists but has not "
            "finished its first run: there is no output yet.\n"
            "  - Otherwise check cfg.kaggle_run.slug_prefix -- the kernel id is "
            "that prefix plus the job name."
        )
    elif "quota" in lowered or "gpu" in lowered and "exceed" in lowered:
        explanation = (
            "Kaggle refused the run on quota grounds.\n"
            "  - The weekly GPU allowance (30 hours) resets Saturday 00:00 UTC. "
            "Check the exact figure at https://www.kaggle.com/settings\n"
            "  - Until it resets, push the job with enable_gpu: false in "
            "configs/kaggle_jobs.yaml if it can make progress on CPU."
        )
    elif "dataset" in lowered and ("source" in lowered or "invalid" in lowered):
        explanation = (
            "Kaggle rejected one of the dataset sources.\n"
            "  - They must be 'owner/slug' and must exist and be visible to you. "
            "This script prefixes bare slugs with your username, so a dataset "
            "owned by someone else must be written out in full.\n"
            "  - Check the job's 'dataset_sources' in configs/kaggle_jobs.yaml "
            "against your dataset list at https://www.kaggle.com/datasets"
        )
    elif any(
        token in lowered
        for token in (
            "connectionerror", "timed out", "timeout", "temporary failure",
            "name resolution", "ssl", "max retries", "connection aborted",
        )
    ):
        explanation = (
            "The connection to Kaggle failed.\n"
            "  - Check your internet connection and https://status.kaggle.com, "
            "then re-run the same command. Every subcommand here is safe to "
            "repeat except `push`, which starts a new run each time."
        )
    else:
        explanation = (
            f"The Kaggle client failed during {what} and the error was not one "
            "this script recognises. The full output is below -- the useful line "
            "is usually the last one."
        )

    return (
        f"{what.capitalize()} failed (exit code {completed.returncode}).\n\n"
        f"{explanation}\n\n"
        "----- raw output from the Kaggle client -----\n"
        f"{output or '(no output)'}"
    )


def looks_absent(output: str) -> bool:
    """True when Kaggle's failure means "no such kernel" rather than a real error.

    MEASURED: asking for a kernel that has never been pushed does **not** return
    a 404. It returns

        Cannot access kernel 'owner/slug' (Permission 'kernels.get' was denied).

    which is the same message a genuinely private kernel belonging to someone
    else produces. Matching only on "404" therefore made ``jobs`` fail outright
    as soon as one configured job had not been pushed yet -- which is its normal
    state, and precisely the case it exists to report.

    Args:
        output: Combined stdout and stderr from the failed call.

    Returns:
        True when the kernel appears not to exist or not to be reachable. The
        caller decides what that means; only ``missing_ok`` callers consult it,
        so a real permissions problem still surfaces everywhere else.
    """
    lowered = output.lower()
    return any(
        token in lowered
        for token in ("404", "not found", "cannot access kernel", "was denied")
    )


def kernel_state(
    cfg: Any, identifier: str, logger: Any, missing_ok: bool = False
) -> Optional[str]:
    """Ask Kaggle for a kernel's current state.

    Args:
        cfg: The loaded base config.
        identifier: ``<username>/<slug>``.
        logger: Logger.
        missing_ok: Return None instead of raising when the kernel does not
            exist. Used by ``jobs``, where "never pushed" is information rather
            than a failure.

    Returns:
        The state, lowercased -- ``queued``, ``running``, ``complete``,
        ``error``, ``cancelacknowledged`` -- or None when the kernel is unknown
        and ``missing_ok`` is set.

    Raises:
        RunError: The call failed, or Kaggle answered something with no state in
            it. An unparseable answer is never reported as success.
    """
    completed = run_kaggle(
        ["kernels", "status", identifier],
        int(cfg.kaggle_run.cli_timeout_s),
        logger,
        f"the status check for {identifier}",
        allow_failure=missing_ok,
    )
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}".strip()
    if completed.returncode != 0:
        if missing_ok and looks_absent(output):
            return None
        raise RunError(translate_kaggle_failure(completed, f"the status check for {identifier}"))

    match = _STATE_RE.search(output)
    if match is None:
        raise RunError(
            f"Could not read a state out of Kaggle's reply for {identifier}.\n"
            "This script will not guess: an unreadable status must never be "
            "reported as success.\n\n"
            "----- raw output from the Kaggle client -----\n"
            f"{output or '(no output)'}"
        )

    state = normalise_state(match.group(1))
    if state in STATE_FAILURES:
        # Kaggle appends the exception text after a failed state. It is the most
        # useful line available without downloading the whole log, so it is
        # surfaced here rather than discarded.
        detail = state_message(output)
        if detail:
            print(f"{BAD} Kaggle reports: {detail}")
            logger.error("%s failed: %s", identifier, detail)
    return state


def normalise_state(raw: str) -> str:
    """Reduce whatever Kaggle called the state to one comparable token.

    Handles both forms the API has been seen to return: the bare word
    (``complete``) and the qualified enum name
    (``KernelWorkerStatus.CANCEL_ACKNOWLEDGED``). The enum's qualifier is
    dropped, underscores are removed, and the result is lowercased, so
    ``CANCEL_ACKNOWLEDGED`` and ``cancelAcknowledged`` both become
    ``cancelacknowledged``.

    Args:
        raw: The captured text after ``status``.

    Returns:
        The lowercased, undecorated state token.
    """
    return raw.rsplit(".", 1)[-1].replace("_", "").lower()


def state_message(output: str) -> str:
    """Extract the failure detail Kaggle appends after an errored state, if any.

    Kaggle reports a failed kernel as ``... has status "error"`` followed by the
    exception text. That text is the single most useful line available without
    downloading the log, so it is surfaced rather than discarded.
    """
    marker = re.search(r'status\s+"?[A-Za-z_.]+"?\s*[.:]?\s*(.*)', output, re.DOTALL)
    detail = (marker.group(1).strip() if marker else "").strip()
    return detail


# -- GPU quota --------------------------------------------------------------


def week_start(cfg: Any) -> _dt.datetime:
    """The start of the current Kaggle quota week, in UTC.

    Kaggle's GPU allowance resets weekly; ``cfg.kaggle_run.quota_week_starts_on``
    says which weekday that is (``datetime.weekday()`` numbering, Monday 0).
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    target = int(cfg.kaggle_run.quota_week_starts_on)
    days_since = (now.weekday() - target) % 7
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight - _dt.timedelta(days=days_since)


def read_ledger(cfg: Any) -> List[Dict[str, Any]]:
    """Read the local record of GPU runs this tool has started.

    Returns:
        The list of run records, oldest first. An absent ledger is an empty list
        -- that is the correct state before the first GPU push, not an error.

    Raises:
        RunError: The file exists but is not readable JSON. Silently starting
            over would erase the only local record of GPU time spent.
    """
    path = repo_path(cfg.kaggle_run.gpu_ledger)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(
            f"Could not read the GPU ledger at {path}: {exc}\n"
            "It is a small JSON file this script appends to on every GPU push. "
            "Fix or delete it -- deleting loses only the local estimate of GPU "
            "hours used, which Kaggle tracks authoritatively anyway at "
            f"{cfg.kaggle_run.quota_url}."
        ) from exc
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise RunError(
            f"The GPU ledger at {path} has no 'runs' list. Delete it to start a "
            "fresh record."
        )
    return runs


def append_ledger(cfg: Any, record: Dict[str, Any]) -> None:
    """Append one GPU run to the local ledger, creating it if needed."""
    path = repo_path(cfg.kaggle_run.gpu_ledger)
    runs = read_ledger(cfg)
    runs.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8")


def report_gpu_quota(cfg: Any, job: Any, logger: Any) -> None:
    """Say what is known about the remaining weekly GPU allowance.

    **The Kaggle public API exposes no quota endpoint.** There is no
    ``kaggle`` CLI command, and no documented REST route, that returns hours
    used or remaining; the figure exists only on the account settings page. So
    this cannot print the real number and does not pretend to.

    What it prints instead is a floor: the GPU time this tool has *started* since
    the quota week began, from a local ledger. Anything launched from the browser,
    from another machine, or before the ledger existed is invisible to it, and
    the estimate uses each job's declared ``expected_runtime_min`` rather than
    measured duration. It is a sanity check against pushing an eight-hour job
    with one hour left -- not an accounting record.

    Args:
        cfg: The loaded base config.
        job: The job about to be pushed.
        logger: Logger.
    """
    rule("GPU QUOTA")
    if not bool(job.enable_gpu):
        print(f"{OK} this job is CPU-only; it does not touch the GPU quota.")
        return

    budget_hours = float(cfg.kaggle_run.weekly_gpu_hours)
    start = week_start(cfg)
    started_minutes = 0.0
    counted = 0
    for record in read_ledger(cfg):
        stamp = record.get("started_utc")
        if not stamp:
            continue
        when = _dt.datetime.fromisoformat(str(stamp))
        if when >= start:
            started_minutes += float(record.get("expected_runtime_min") or 0)
            counted += 1

    about_to_add = float(job.get("expected_runtime_min") or 0) / 60.0
    print(f"{BAD} Kaggle's API does not expose your remaining GPU quota.")
    print(f"{INFO}No CLI command or documented endpoint returns it. The real")
    print(f"{INFO}figure is on {cfg.kaggle_run.quota_url} -- check it there.")
    print()
    print(f"{INFO}Local estimate, from this tool's own ledger only:")
    print(f"{INFO}  quota week began   {start.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{INFO}  GPU runs started   {counted}")
    print(f"{INFO}  their declared time {started_minutes / 60:.1f} h of {budget_hours:.0f} h")
    print(f"{INFO}  this job declares   {about_to_add:.1f} h")
    print(f"{INFO}This is a FLOOR, not your quota: browser-launched sessions and")
    print(f"{INFO}runs from other machines are not in it, and the figures are the")
    print(f"{INFO}declared expected_runtime_min, not measured GPU time.")

    if started_minutes / 60 + about_to_add > budget_hours:
        print()
        print(
            f"{BAD} Even this incomplete count exceeds the {budget_hours:.0f} h "
            "weekly budget."
        )
        print(f"{INFO}Check {cfg.kaggle_run.quota_url} before pushing.")
    logger.info(
        "GPU ledger: %d runs, %.1f h declared since %s",
        counted,
        started_minutes / 60,
        start.isoformat(),
    )


# -- subcommands ------------------------------------------------------------


def refuse_network_in_smoke(command: str, detail: str) -> None:
    """Stop a subcommand under ``--smoke``, saying what did and did not happen.

    ``--smoke`` must never touch the network. For ``push`` that is a genuinely
    useful preflight: every local check runs, the notebook and metadata are
    generated for real, and only the transfer is skipped. For the read-only
    subcommands there is nothing left to do offline, and saying so is better than
    pretending to have checked something.

    Args:
        command: The subcommand name, for the banner.
        detail: What actually ran, in this script's own terms.
    """
    print()
    print("=" * 78)
    print(f"  --smoke: STOPPING BEFORE THE NETWORK CALL for `{command}`.")
    print("=" * 78)
    print()
    for line in detail.splitlines():
        print(f"  {line}" if line else "")


def cmd_push(args, cfg, logger) -> int:
    """Generate the notebook and metadata, then push and start the run."""
    username = preflight(cfg, logger)
    job = resolve_job(cfg, args.job)
    identifier = kernel_id(cfg, username, args.job)

    rule("CODE THAT WILL RUN")
    sha = require_publishable_tree(
        cfg, logger, enforce=not args.smoke, allow_dirty=bool(getattr(args, "allow_dirty", False))
    )

    rule("JOB")
    print(f"{INFO}job          {args.job}")
    print(f"{INFO}kernel       {identifier}")
    entry_args = " ".join(str(arg) for arg in (job.get("entry_args") or []))
    shown = job.get("entry") or f"python -m {job.get('entry_module')}"
    print(f"{INFO}entry point  {shown} {entry_args}".rstrip())
    accel = str(job.get('accelerator') or cfg.kaggle_run.accelerator)
    print(f"{INFO}accelerator  {accel if job.enable_gpu else 'CPU only (enable_gpu: false)'}")
    print(f"{INFO}data guard   {'ON -- aborts the run if the mount fails' if job.guard_data_root else 'off (the entry point IS the check)'}")
    runtime_min = int(job.get("expected_runtime_min") or 0)
    print(f"{INFO}expected     {human_duration(runtime_min * 60) if runtime_min else 'unknown'}")

    report_gpu_quota(cfg, job, logger)

    rule("GENERATED FILES")
    build_dir, metadata = generate_kernel(cfg, username, args.job, job, sha, logger)
    print(f"{INFO}datasets mounted: {', '.join(metadata['dataset_sources']) or '(none)'}")

    if args.smoke:
        refuse_network_in_smoke(
            "push",
            "Everything above ran for real: the job definition was resolved and\n"
            "validated, the working tree was inspected, the notebook was\n"
            "generated from the template and its placeholders checked, and\n"
            "kernel-metadata.json was written with the slug, title and dataset\n"
            "sources Kaggle will see.\n"
            "\n"
            "Skipped, because smoke mode contacts no network:\n"
            "  - `git fetch` and the check that your commit is on the remote;\n"
            "  - the push itself, which is what starts the run.\n"
            "\n"
            "Read the generated notebook -- it is exactly what Kaggle would run.\n"
            "\n"
            "Re-run without --smoke to actually push.",
        )
        next_steps(
            "Inspect the generated notebook: " + str(build_dir),
            "Commit and push any outstanding work.",
            f"Run it for real: python scripts/kaggle_run.py push --job {args.job}",
        )
        return 0

    rule("PUSHING")
    completed = run_kaggle(
        ["kernels", "push", "-p", str(build_dir)],
        int(cfg.kaggle_run.cli_timeout_s),
        logger,
        "the kernel push",
    )
    for line in (completed.stdout or "").splitlines():
        if line.strip():
            print(f"{INFO}{line.strip()}")

    if bool(job.enable_gpu):
        append_ledger(
            cfg,
            {
                "job": args.job,
                "kernel": identifier,
                "sha": sha,
                "accelerator": accel,
                "expected_runtime_min": runtime_min,
                "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            },
        )
        print(f"{OK} recorded in the local GPU ledger")

    print(f"{OK} pushed and queued: https://www.kaggle.com/{identifier}")
    logger.info("Pushed %s at commit %s", identifier, sha)
    next_steps(
        f"Watch it: python scripts/kaggle_run.py status --job {args.job} --watch",
        f"Read the log: python scripts/kaggle_run.py logs --job {args.job}",
        f"Download outputs when it finishes: python scripts/kaggle_run.py fetch --job {args.job}",
    )
    return 0


def cmd_status(args, cfg, logger) -> int:
    """Report the kernel's state; with ``--watch``, follow it to a terminal one.

    Returns:
        0 when the last observed state is success, 1 when it is a failure. With
        ``--watch`` the exit code is the run's verdict, which is what makes this
        usable as the gate in a shell sequence.
    """
    username = preflight(cfg, logger)
    resolve_job(cfg, args.job)  # validates the name before any network call
    identifier = kernel_id(cfg, username, args.job)

    rule(f"STATUS: {identifier}")
    if not args.watch:
        state = kernel_state(cfg, identifier, logger)
        return report_terminal_state(state, identifier, args.job, logger)

    interval = int(cfg.kaggle_run.poll_interval_s)
    deadline = _dt.datetime.now() + _dt.timedelta(seconds=int(cfg.kaggle_run.watch_timeout_s))
    print(f"{INFO}polling every {interval} s; Ctrl-C stops watching without")
    print(f"{INFO}stopping the run on Kaggle.")
    print()

    previous: Optional[str] = None
    started = _dt.datetime.now()
    while _dt.datetime.now() < deadline:
        state = kernel_state(cfg, identifier, logger)
        elapsed = (_dt.datetime.now() - started).total_seconds()
        if state != previous:
            stamp = _dt.datetime.now().strftime("%H:%M:%S")
            print(f"  {stamp}  {state:20s} (+{human_duration(elapsed)})", flush=True)
            logger.info("%s -> %s after %.0f s", identifier, state, elapsed)
            previous = state
        if state == STATE_SUCCESS or state in STATE_FAILURES:
            print()
            return report_terminal_state(state, identifier, args.job, logger)
        if state not in STATE_RUNNING:
            # An unrecognised state is followed, not assumed to be terminal.
            logger.warning("Unrecognised kernel state %r; still polling.", state)
        _sleep(interval)

    raise RunError(
        f"Stopped watching {identifier} after "
        f"{human_duration(int(cfg.kaggle_run.watch_timeout_s))}; it was last seen "
        f"in state {previous!r}.\n"
        "The run itself is unaffected -- this is only the watcher giving up.\n"
        "Check it with `python scripts/kaggle_run.py status --job "
        f"{args.job}`, or raise cfg.kaggle_run.watch_timeout_s."
    )


def _sleep(seconds: int) -> None:
    """Sleep between polls. Factored out so tests can replace it."""
    import time

    time.sleep(seconds)


def report_terminal_state(state: Optional[str], identifier: str, job_name: str, logger: Any) -> int:
    """Print the verdict for a state and return the process exit code."""
    if state == STATE_SUCCESS:
        print(f"{OK} {identifier} finished successfully.")
        logger.info("%s complete", identifier)
        next_steps(
            f"Download the outputs: python scripts/kaggle_run.py fetch --job {job_name}",
            f"Read the log: python scripts/kaggle_run.py logs --job {job_name}",
        )
        return 0
    if state in STATE_FAILURES:
        print(f"{BAD} {identifier} finished in state {state!r}.")
        logger.error("%s ended in state %s", identifier, state)
        next_steps(
            f"Read the failure: python scripts/kaggle_run.py logs --job {job_name}",
            "Fix it locally and run the same entry point with --smoke before "
            "pushing again.",
            f"Push the fix: python scripts/kaggle_run.py push --job {job_name}",
        )
        return 1
    print(f"{INFO}{identifier} is {state!r} -- still going.")
    next_steps(
        f"Follow it: python scripts/kaggle_run.py status --job {job_name} --watch",
    )
    return 0


def format_log(text: str) -> List[str]:
    """Turn a Kaggle kernel log file into printable lines.

    Kaggle writes the log as a JSON array of ``{"stream_name", "time", "data"}``
    records rather than as plain text. That format is unreadable raw, so it is
    rendered here; a log that is not JSON (the format has changed before) is
    returned as its own lines rather than discarded.

    Args:
        text: The downloaded log file's contents.

    Returns:
        One string per output line, in chronological order.
    """
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        return text.splitlines()
    if not isinstance(records, list):
        return text.splitlines()

    lines: List[str] = []
    for record in records:
        if not isinstance(record, dict):
            lines.append(str(record))
            continue
        stream = str(record.get("stream_name", "")).strip()
        data = str(record.get("data", "")).rstrip("\n")
        marker = "" if stream in ("stdout", "") else f"[{stream}] "
        lines.extend(f"{marker}{line}" for line in data.splitlines() or [""])
    return lines


def download_output(cfg: Any, identifier: str, destination: Path, logger: Any) -> Path:
    """Download a kernel's output files into ``destination``.

    Args:
        cfg: The loaded base config.
        identifier: ``<username>/<slug>``.
        destination: Directory to download into; created if needed.
        logger: Logger.

    Returns:
        The destination directory.

    Raises:
        RunError: The download failed, or produced nothing. An empty download is
            an error rather than an empty success: it means either the run has
            not finished or it wrote outside ``/kaggle/working``, and both need
            saying.
    """
    destination.mkdir(parents=True, exist_ok=True)
    run_kaggle(
        ["kernels", "output", identifier, "-p", str(destination)],
        int(cfg.kaggle_run.fetch_timeout_s),
        logger,
        f"the output download for {identifier}",
    )
    files = [path for path in destination.rglob("*") if path.is_file()]
    if not files:
        raise RunError(
            f"Kaggle returned no output files for {identifier}.\n"
            "Either the run has not finished (check "
            "`python scripts/kaggle_run.py status --job <name>`), or it wrote "
            "outside /kaggle/working -- only that directory is saved as kernel "
            "output. The job's 'workdir' in configs/kaggle_jobs.yaml must be "
            "under /kaggle/working for this to work."
        )
    return destination


def cmd_logs(args, cfg, logger) -> int:
    """Print the run log, tail-first.

    Tail-first means the newest lines, in order, are what you get: a failure is
    at the end of a log, and scrolling up through hours of training output to
    reach it is the wrong default. ``--full`` prints everything.
    """
    username = preflight(cfg, logger)
    resolve_job(cfg, args.job)
    identifier = kernel_id(cfg, username, args.job)

    destination = repo_path(cfg.kaggle_run.run_dir) / args.job / "_logs"
    rule(f"LOG: {identifier}")

    # Emptied first, so a log left by an earlier run cannot be printed as though
    # it belonged to this one. Scoped deliberately: only this job's own "_logs"
    # scratch folder, never the timestamped folders `fetch` writes.
    if destination.exists():
        if destination.name != "_logs":
            raise RunError(
                f"Refusing to clear {destination}: it is not a '_logs' scratch "
                "folder. Check cfg.kaggle_run.run_dir."
            )
        shutil.rmtree(destination)

    download_output(cfg, identifier, destination, logger)

    log_files = sorted(destination.rglob("*.log"))
    if not log_files:
        raise RunError(
            f"No .log file came back for {identifier}, although other output "
            f"did. Kaggle names it after the kernel slug and puts it alongside "
            f"the outputs; look in {destination} to see what arrived."
        )

    exit_code = 0
    for log_file in log_files:
        lines = format_log(log_file.read_text(encoding="utf-8", errors="replace"))
        tail = int(args.lines if args.lines else cfg.kaggle_run.log_tail_lines)
        shown = lines if args.full else lines[-tail:]
        header = (
            f"{log_file.name}: all {len(lines)} lines"
            if args.full
            else f"{log_file.name}: last {len(shown)} of {len(lines)} lines "
            f"(--full for everything, --lines N for more)"
        )
        print()
        print(header)
        print("-" * 78)
        for line in shown:
            print(line)
        if any("Traceback (most recent call last)" in line for line in lines):
            exit_code = 1

    if exit_code:
        print()
        print(f"{BAD} this log contains a traceback.")
    logger.info("Printed %d log file(s) for %s", len(log_files), identifier)
    return exit_code


def cmd_fetch(args, cfg, logger) -> int:
    """Download the run's outputs into a timestamped folder and inventory them."""
    username = preflight(cfg, logger)
    resolve_job(cfg, args.job)
    identifier = kernel_id(cfg, username, args.job)

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = repo_path(cfg.kaggle_run.run_dir) / args.job / stamp

    rule(f"FETCH: {identifier}")
    print(f"{INFO}into {destination}")
    download_output(cfg, identifier, destination, logger)

    files = sorted(path for path in destination.rglob("*") if path.is_file())
    groups: Dict[str, List[Path]] = {
        "checkpoints": [], "metrics": [], "figures": [], "logs": [], "other": [],
    }
    suffix_map = {
        ".pt": "checkpoints", ".pth": "checkpoints", ".ckpt": "checkpoints",
        ".onnx": "checkpoints", ".safetensors": "checkpoints",
        ".json": "metrics", ".csv": "metrics", ".md": "metrics",
        ".png": "figures", ".jpg": "figures", ".jpeg": "figures",
        ".pdf": "figures", ".svg": "figures",
        ".log": "logs", ".txt": "logs",
    }
    for path in files:
        groups[suffix_map.get(path.suffix.lower(), "other")].append(path)

    print()
    total_bytes = 0
    for label, members in groups.items():
        if not members:
            continue
        size = sum(path.stat().st_size for path in members)
        total_bytes += size
        print(f"  {label.upper()}  ({len(members)} file(s), {size / 1024 / 1024:.2f} MB)")
        for path in members[:12]:
            relative = path.relative_to(destination)
            print(f"      {str(relative):<58} {path.stat().st_size / 1024:>10.1f} KB")
        if len(members) > 12:
            print(f"      ... and {len(members) - 12} more")
        print()

    print(f"{OK} {len(files)} file(s), {total_bytes / 1024 / 1024:.2f} MB in {destination}")
    for label in ("checkpoints", "metrics", "figures"):
        if not groups[label]:
            print(f"{INFO}no {label} in this run's output")
    logger.info("Fetched %d files for %s into %s", len(files), identifier, destination)

    next_steps(
        "Inspect what arrived: " + str(destination),
        "Outputs are gitignored on purpose; copy anything that belongs in the "
        "report into reports/.",
    )
    return 0


def cmd_jobs(args, cfg, logger) -> int:
    """List the configured jobs with their current state, then all your kernels."""
    username = preflight(cfg, logger)
    jobs = load_jobs(cfg)

    rule("CONFIGURED JOBS")
    print(f"  {'job':<12} {'accel':<16} {'guard':<6} {'expected':<12} entry")
    print("  " + "-" * 74)
    for name, job in jobs.items():
        accel = (
            str(job.get("accelerator") or cfg.kaggle_run.accelerator)
            if job.enable_gpu
            else "CPU"
        )
        runtime_min = int(job.get("expected_runtime_min") or 0)
        expected = human_duration(runtime_min * 60) if runtime_min else "unknown"
        guard = "on" if job.guard_data_root else "off"
        shown = job.get("entry") or f'-m {job.get("entry_module")}'
        print(f"  {name:<12} {accel:<16} {guard:<6} {expected:<12} {shown}")

    rule("STATE ON KAGGLE")
    for name in jobs:
        identifier = kernel_id(cfg, username, name)
        state = kernel_state(cfg, identifier, logger, missing_ok=True)
        if state is None:
            # Deliberately hedged: Kaggle gives the same "permission denied"
            # answer for a kernel that does not exist and one you cannot see.
            print(f"  {name:<12} not pushed / no access  {identifier}")
        else:
            marker = OK.strip() if state == STATE_SUCCESS else (
                BAD.strip() if state in STATE_FAILURES else "  ..  "
            )
            print(f"  {name:<12} {state:<20} {identifier}   {marker}")

    rule("ALL KERNELS ON YOUR ACCOUNT")
    completed = run_kaggle(
        ["kernels", "list", "--mine", "--page-size", str(args.limit)],
        int(cfg.kaggle_run.cli_timeout_s),
        logger,
        "the kernel listing",
        allow_failure=True,
    )
    if completed.returncode != 0:
        print(f"{BAD} could not list your kernels; the configured jobs above are")
        print(f"{INFO}still accurate. Raw output:")
        print((completed.stdout or completed.stderr or "").strip())
    else:
        for line in (completed.stdout or "").splitlines():
            print(f"  {line}")

    next_steps(
        "Push one: python scripts/kaggle_run.py push --job <name>",
        "Define a new one by adding an entry to " + str(cfg.kaggle_run.jobs_file),
    )
    return 0


# -- entry point ------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drive Kaggle notebook jobs from the terminal: generate a notebook "
            "pinned to a commit, push it, watch it, and pull the results."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical loop:\n"
            "  python scripts/kaggle_run.py jobs\n"
            "  python scripts/kaggle_run.py push   --job verify\n"
            "  python scripts/kaggle_run.py status --job verify --watch\n"
            "  python scripts/kaggle_run.py logs   --job verify\n"
            "  python scripts/kaggle_run.py fetch  --job verify\n"
            "\n"
            "Jobs are defined in configs/kaggle_jobs.yaml.\n"
            "See docs/kaggle_workflow.md for the whole loop in plain English.\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    definitions = [
        ("push", "Generate the notebook, push it, and start the run."),
        ("status", "Report the kernel's state; --watch follows it to the end."),
        ("logs", "Download and print the run log, tail-first."),
        ("fetch", "Download the run's outputs into outputs/kaggle/<job>/<timestamp>/."),
        ("jobs", "List the configured jobs and your kernels' last run state."),
    ]
    for name, help_text in definitions:
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        add_standard_args(sub)
        if name != "jobs":
            sub.add_argument(
                "--job",
                required=True,
                help="Job name, as defined in configs/kaggle_jobs.yaml "
                "(verify, baseline, train, ...).",
            )
        if name == "push":
            sub.add_argument(
                "--allow-dirty",
                action="store_true",
                help="Push even though the working tree has uncommitted changes. "
                "The run still uses a commit proven to be on the remote; every "
                "dirty path and the SHA being run are printed. For a repository "
                "several sessions edit at once, where another session's work in "
                "progress would otherwise block an unrelated launch.",
            )
        if name == "status":
            sub.add_argument(
                "--watch",
                action="store_true",
                help="Poll until the run reaches a terminal state, printing every "
                "state change. Exits non-zero if the run failed, so it can gate a "
                "shell sequence. Ctrl-C stops watching; it does not stop the run.",
            )
        if name == "logs":
            sub.add_argument(
                "--lines",
                type=int,
                default=None,
                help="How many trailing lines to print "
                "(default: cfg.kaggle_run.log_tail_lines).",
            )
            sub.add_argument(
                "--full",
                action="store_true",
                help="Print the whole log instead of the tail.",
            )
        if name == "jobs":
            sub.add_argument(
                "--limit",
                type=int,
                default=20,
                help="How many of your kernels to list (default: 20).",
            )
    return parser.parse_args(argv)


HANDLERS = {
    "push": cmd_push,
    "status": cmd_status,
    "logs": cmd_logs,
    "fetch": cmd_fetch,
    "jobs": cmd_jobs,
}


def make_console_unencodable_safe() -> None:
    """Stop a Kaggle log from killing this script on a legacy Windows console.

    MEASURED 2026-09-08 against the real Run A log, ``drishtisr-runa.log``:
    7 of its 620 lines contain U+2501 (heavy box drawing) -- pip's progress
    bars from the session's package install. Printing
    those on a Windows console whose stdout encoding is cp1252 -- the default
    for a non-UTF-8 code page, which is what ``py``/``python.exe`` gets here --
    raises ``UnicodeEncodeError: 'charmap' codec can't encode character
    '━'``. The traceback lands in the middle of the log, so ``logs`` fails
    on precisely the runs whose output is most worth reading.

    The encoding is left ALONE and only the error handler is changed: forcing
    UTF-8 onto a cp1252 console would print mojibake for every box character
    instead of crashing on some, which is a different way of being unreadable.
    ``backslashreplace`` renders the unencodable character as its escape, so
    the line survives and says what it lost.

    A stream that cannot be reconfigured (a pipe wrapper, a captured stream
    under pytest) is left as it is; this is a display convenience and must not
    become a reason the script fails to start.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        reconfigure(errors="backslashreplace")


def main(argv=None) -> int:
    make_console_unencodable_safe()
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("kaggle_run", log_file=cfg.paths.log_file)
    seed_everything(cfg.seed)

    if args.smoke and args.command in ("status", "logs", "fetch", "jobs"):
        refuse_network_in_smoke(
            args.command,
            f"`{args.command}` does nothing but read state from Kaggle, so there\n"
            "is nothing for it to do offline and nothing was checked here.\n"
            "\n"
            "The preflight worth running is `push --smoke`: it performs every\n"
            "local check and generates the notebook, then stops before the push.",
        )
        return 0

    try:
        return HANDLERS[args.command](args, cfg, logger)
    except RunError as exc:
        print()
        print("=" * 78)
        print("  STOPPED")
        print("=" * 78)
        print()
        for line in str(exc).splitlines():
            print(f"  {line}" if line else "")
        print()
        logger.error("%s failed: %s", args.command, str(exc).splitlines()[0])
        return 1
    except KeyboardInterrupt:
        print()
        print(
            "  Stopped watching. The Kaggle run is unaffected -- it keeps going "
            "on their machines.\n"
            "  Check on it with `python scripts/kaggle_run.py status --job "
            "<name>`."
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
