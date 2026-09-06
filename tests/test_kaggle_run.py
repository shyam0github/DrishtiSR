"""Tests for the headless Kaggle job runner.

The expensive failures this tool can produce are not crashes:

1. **A run of code that is not the code on screen.** Kaggle clones from GitHub,
   so an uncommitted or unpushed working tree means the session tests something
   else entirely -- successfully, and silently.
2. **A GPU session that cannot see its data.** It resolves an empty tree,
   reports zero samples, finishes green, and costs an hour of a 30-hour week.
3. **A P100 session.** ``torch.cuda.is_available()`` returns True and the first
   real CUDA op dies, after the setup time has already been spent.

So the tests concentrate on the guards against those three, on the notebook
being generated rather than hand-written, and on every Kaggle CLI failure
becoming an explanation instead of a traceback.

Nothing here touches the network, Kaggle, or the real cache.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.config import load_config
from src.utils.kaggle_session import SessionAborted, guard_data_root, run_entry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import kaggle_run as kr  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def jobs(cfg):
    return kr.load_jobs(cfg)


# -- the accelerator guard -------------------------------------------------


def test_gpu_jobs_request_t4_and_never_p100(cfg, jobs):
    """P100 is the trap this project must not fall into.

    Kaggle's default image ships a cu128 PyTorch with no Pascal (sm_60) kernels.
    On a P100 that fails *late*: torch imports, ``torch.cuda.is_available()``
    returns True, the device names itself correctly, and the first real CUDA op
    raises ``cudaErrorNoKernelImageForDevice`` -- after the session has queued,
    booted, cloned and installed. T4 is sm_75 and covered by the same build.
    """
    assert str(cfg.kaggle_run.accelerator) == "NvidiaTeslaT4"
    for name, job in jobs.items():
        if not job.enable_gpu:
            continue
        accelerator = str(job.get("accelerator") or cfg.kaggle_run.accelerator)
        assert accelerator == "NvidiaTeslaT4", (
            f"job {name!r} requests {accelerator!r}"
        )
        assert "P100" not in accelerator


def test_metadata_carries_the_accelerator_only_for_gpu_jobs(cfg, jobs):
    """A CPU job must not name an accelerator; Kaggle would allocate one."""
    gpu = kr.kernel_metadata(cfg, "someone", "train", jobs["train"])
    assert gpu["enable_gpu"] is True
    assert gpu["accelerator"] == "NvidiaTeslaT4"

    cpu = kr.kernel_metadata(cfg, "someone", "verify", jobs["verify"])
    assert cpu["enable_gpu"] is False
    assert "accelerator" not in cpu


# -- the data guard --------------------------------------------------------


def test_every_gpu_job_runs_the_data_guard(jobs):
    """No GPU-hour may be spent by a session that has not proved it sees data."""
    for name, job in jobs.items():
        if job.enable_gpu:
            assert bool(job.guard_data_root), f"GPU job {name!r} has no data guard"


def test_the_only_job_without_the_guard_is_the_verifier_itself(jobs):
    """`verify` may skip it, because its entry point IS the check."""
    for name, job in jobs.items():
        if not job.guard_data_root:
            assert str(job.entry) == str(job.guard_script), (
                f"job {name!r} skips the data guard without being the guard"
            )


def test_guard_failure_aborts_the_session(tmp_path):
    """A failed guard raises, so the Kaggle kernel run is marked failed.

    Returning a status code here would let the notebook carry on into the
    expensive cell, which is the exact behaviour the guard exists to prevent.
    """
    script = tmp_path / "failing_check.py"
    script.write_text("raise SystemExit(2)\n", encoding="utf-8")
    with pytest.raises(SessionAborted) as excinfo:
        guard_data_root("failing_check.py", repo_dir=str(tmp_path))
    message = str(excinfo.value)
    assert "DATA GUARD FAILED" in message
    assert "No GPU time has been spent" in message


def test_guard_can_be_disabled_only_explicitly(tmp_path):
    """``enabled=False`` is a no-op that says so rather than a silent skip."""
    guard_data_root("does_not_exist.py", enabled=False, repo_dir=str(tmp_path))


def test_a_failing_entry_point_aborts_rather_than_returning(tmp_path):
    script = tmp_path / "job.py"
    script.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    with pytest.raises(SessionAborted) as excinfo:
        run_entry("job.py", repo_dir=str(tmp_path))
    assert "exited with code" in str(excinfo.value)


# -- notebook generation ---------------------------------------------------


def test_generated_notebook_code_is_valid_python(cfg, jobs):
    """Every job's notebook must parse. A syntax error would be discovered on
    Kaggle, after the queue wait, rather than here in milliseconds."""
    for name, job in jobs.items():
        tokens = kr.build_tokens(cfg, name, job, "a" * 40, "https://example.com/r.git")
        source = kr.render_template(
            (kr.repo_root() / str(job.template)).read_text(encoding="utf-8"), tokens
        )
        notebook = kr.to_notebook(source)
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        ast.parse(code)  # raises SyntaxError if the template rendered badly


def test_generated_notebook_pins_the_commit(cfg, jobs):
    """The SHA must appear in the notebook: it is the whole provenance claim."""
    sha = "0123456789abcdef0123456789abcdef01234567"
    tokens = kr.build_tokens(cfg, "train", jobs["train"], sha, "https://example.com/r.git")
    source = kr.render_template(
        (kr.repo_root() / str(jobs["train"].template)).read_text(encoding="utf-8"), tokens
    )
    assert source.count(sha) >= 2  # the header table and the checkout literal
    assert "checkout" in source and "--detach" in source


def test_unfilled_placeholders_are_refused(cfg, jobs):
    """A missing token would generate a notebook that cannot even parse."""
    template = "# %%\nvalue = {{NOT_A_REAL_TOKEN}}\n"
    with pytest.raises(kr.RunError, match="placeholders this script does not fill"):
        kr.render_template(template, {"JOB_NAME": "x"})


def test_a_template_with_no_cells_is_refused():
    """An empty notebook is accepted by Kaggle and runs to completion doing
    nothing -- a green result for a run that did not happen."""
    with pytest.raises(kr.RunError, match="produced no cells"):
        kr.to_notebook("print('no cell markers here')\n")


def test_markdown_cells_lose_their_comment_prefix():
    notebook = kr.to_notebook("# %% [markdown]\n# # Title\n# body\n\n# %%\nx = 1\n")
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert "".join(notebook["cells"][0]["source"]).startswith("# Title")
    assert notebook["cells"][1]["cell_type"] == "code"


def test_notebook_is_written_next_to_its_metadata(cfg, jobs, tmp_path, monkeypatch):
    """`kaggle kernels push -p <dir>` needs both files in one folder, and the
    metadata's ``code_file`` must name the notebook that is actually there."""
    monkeypatch.setitem(cfg.kaggle_run, "kernel_build_dir", str(tmp_path / "build"))
    logger = _silent_logger()
    build_dir, metadata = kr.generate_kernel(
        cfg, "someone", "verify", jobs["verify"], "b" * 40, logger
    )
    assert (build_dir / "kernel-metadata.json").is_file()
    assert (build_dir / metadata["code_file"]).is_file()
    written = json.loads((build_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert written["id"] == "someone/drishtisr-verify"
    assert written["kernel_type"] == "notebook"
    assert written["enable_internet"] is True
    json.loads((build_dir / metadata["code_file"]).read_text(encoding="utf-8"))


def test_missing_entry_point_is_caught_before_the_push(cfg, jobs, tmp_path, monkeypatch):
    """Cheap here; minutes of session time to discover on Kaggle."""
    monkeypatch.setitem(cfg.kaggle_run, "kernel_build_dir", str(tmp_path / "build"))
    job = OmegaConf.merge(jobs["verify"], {"entry": "scripts/not_a_real_script.py"})
    with pytest.raises(kr.RunError, match="entry point does not exist"):
        kr.generate_kernel(cfg, "someone", "verify", job, "c" * 40, _silent_logger())


# -- job definitions -------------------------------------------------------


def test_the_three_starting_jobs_exist(jobs):
    assert {"verify", "baseline", "train"} <= set(jobs.keys())


def test_defaults_are_merged_into_every_job(jobs):
    for name, job in jobs.items():
        for key in kr.REQUIRED_JOB_KEYS:
            assert key in job, f"job {name!r} lacks {key!r}"


def test_jobs_mount_the_cache_dataset_the_resolver_looks_for(cfg, jobs):
    """The mount path is composed from ``paths.kaggle_dataset_dir``. A job that
    mounts a different dataset produces an empty data root and a run that
    reports zero samples."""
    expected = str(cfg.paths.kaggle_dataset_dir)
    for name, job in jobs.items():
        slugs = [str(entry).split("/")[-1] for entry in job.dataset_sources]
        assert expected in slugs, f"job {name!r} does not mount {expected}"


def test_workdir_is_saved_as_kernel_output(jobs):
    """Only /kaggle/working is retained as kernel output. A job cloning
    elsewhere runs fine and then loses everything it wrote."""
    for name, job in jobs.items():
        assert str(job.workdir).startswith("/kaggle/working"), (
            f"job {name!r} clones to {job.workdir}, whose outputs are discarded"
        )


def test_jobs_do_not_install_the_pinned_requirements_file(jobs):
    """Installing requirements.txt on Kaggle downgrades numpy/pandas/scipy under
    the preinstalled PyTorch. Jobs name only what the image lacks."""
    for name, job in jobs.items():
        for package in job.pip_packages:
            assert "requirements" not in str(package), f"job {name!r}: {package}"


def test_an_unknown_job_lists_the_known_ones(cfg):
    with pytest.raises(kr.RunError, match="Defined jobs: baseline, train, verify"):
        kr.resolve_job(cfg, "no-such-job")


def test_an_incomplete_job_is_refused_by_name(tmp_path):
    path = tmp_path / "jobs.yaml"
    path.write_text("jobs:\n  half:\n    title: A job\n", encoding="utf-8")
    with pytest.raises(kr.RunError, match="'half'.*is missing"):
        kr.validate_job("half", OmegaConf.create({"title": "A job"}), path, "half")


# -- the naming drift guard ------------------------------------------------
#
# MEASURED against the live API on 2026-09-06, which is why these tests exist
# rather than the assumption they replace. Pushing kernel-metadata.json with
#     id    = "shyamdwivedi0/drishtisr-verify"
#     title = "DrishtiSR verify data root"
# created the kernel at "shyamdwivedi0/drishtisr-verify-data-root". Kaggle
# slugifies the TITLE and the id loses. The push reported success; every
# subsequent status/logs/fetch failed with a permission error that reads like the
# kernel is private. The run was fine and unreachable.


def test_every_job_title_slugifies_to_its_kernel_slug(cfg, jobs):
    """The push path and the read path must name the same kernel.

    This is the kernel-side twin of the dataset drift guard in
    test_kaggle_upload.py: there, the upload slug and the mount directory must
    agree; here, the title Kaggle turns into a URL and the id this tool addresses
    must agree.
    """
    for name, job in jobs.items():
        assert kr.slugify(str(job.title)) == kr.kernel_slug(cfg, name), (
            f"job {name!r}: title {str(job.title)!r} would create a kernel at "
            f"{kr.slugify(str(job.title))!r}, but this tool addresses "
            f"{kr.kernel_slug(cfg, name)!r}"
        )


def test_a_title_that_would_land_elsewhere_is_refused(tmp_path):
    """The exact configuration that produced the unreachable kernel."""
    job = OmegaConf.create(
        {key: "x" for key in kr.REQUIRED_JOB_KEYS} | {"title": "DrishtiSR verify data root"}
    )
    with pytest.raises(kr.RunError, match="does not match its kernel slug"):
        kr.validate_job("verify", job, tmp_path / "jobs.yaml", "drishtisr-verify")


@pytest.mark.parametrize(
    "title, slug",
    [
        ("DrishtiSR verify", "drishtisr-verify"),
        ("DrishtiSR verify data root", "drishtisr-verify-data-root"),
        ("  Mixed  CASE__and   punctuation!  ", "mixed-case-and-punctuation"),
    ],
)
def test_slugify_matches_kaggles_rule(title, slug):
    assert kr.slugify(title) == slug


# -- state parsing ---------------------------------------------------------
#
# Also MEASURED: the live API answers `has status "KernelWorkerStatus.RUNNING"`,
# not the bare word. Failing to parse that is not a crash -- an unrecognised
# state is treated as "still running", so --watch polls a finished job until its
# timeout and never reports the result.


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("complete", "complete"),
        ("KernelWorkerStatus.RUNNING", "running"),
        ("KernelWorkerStatus.COMPLETE", "complete"),
        ("KernelWorkerStatus.ERROR", "error"),
        ("KernelWorkerStatus.CANCEL_ACKNOWLEDGED", "cancelacknowledged"),
        ("cancelAcknowledged", "cancelacknowledged"),
    ],
)
def test_enum_and_bare_states_normalise_to_the_same_token(raw, expected):
    assert kr.normalise_state(raw) == expected


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('x/y has status "KernelWorkerStatus.RUNNING"', "running"),
        ('x/y has status "KernelWorkerStatus.COMPLETE"', "complete"),
        ('x/y has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', "cancelacknowledged"),
    ],
)
def test_enum_replies_are_recognised_as_real_states(reply, expected):
    """The regex and the normaliser together must land on a token the runner
    actually knows, or a finished run is invisible to --watch."""
    match = kr._STATE_RE.search(reply)
    assert match is not None
    state = kr.normalise_state(match.group(1))
    assert state == expected
    assert state == kr.STATE_SUCCESS or state in kr.STATE_FAILURES + kr.STATE_RUNNING


# -- slugs and ids ---------------------------------------------------------


def test_kernel_id_takes_its_owner_from_the_credentials(cfg):
    """No username in version control: the repo works for anyone who clones it."""
    assert kr.kernel_id(cfg, "someone", "train") == "someone/drishtisr-train"
    assert "someone" not in OmegaConf.to_yaml(cfg.kaggle_run)


def test_bare_dataset_slugs_gain_the_owner_prefix(cfg, jobs):
    metadata = kr.kernel_metadata(cfg, "someone", "train", jobs["train"])
    assert metadata["dataset_sources"] == ["someone/drishtisr-sen2naipv2-cache"]


def test_fully_qualified_dataset_sources_are_left_alone(cfg, jobs):
    job = OmegaConf.merge(jobs["train"], {"dataset_sources": ["elsewhere/public-data"]})
    metadata = kr.kernel_metadata(cfg, "someone", "train", job)
    assert metadata["dataset_sources"] == ["elsewhere/public-data"]


@pytest.mark.parametrize("slug", ["ab", "Has-Capitals", "under_scores", "-leading", "a" * 61])
def test_bad_kernel_slugs_are_refused_locally(slug):
    with pytest.raises(kr.RunError):
        kr.validate_kernel_slug(slug)


def test_every_configured_job_produces_a_valid_slug(cfg, jobs):
    for name in jobs:
        kr.kernel_slug(cfg, name)


def test_titles_satisfy_kaggle_length_rules(jobs):
    for name, job in jobs.items():
        kr.validate_title(str(job.title))


# -- Kaggle CLI handling ---------------------------------------------------


def test_the_cli_is_invoked_through_this_interpreter():
    """The bare ``kaggle`` executable is not on PATH on this machine, and a
    ``kaggle.exe`` belonging to another Python would report a different set of
    installed packages."""
    assert kr.kaggle_command() == [sys.executable, "-m", "kaggle"]


@pytest.mark.parametrize(
    "output, expected",
    [
        ("401 - Unauthorized", "Create New Token"),
        ("403 Forbidden", "phone"),
        ("404 - Not found", "never been pushed"),
        ("GPU quota exceeded", "Saturday"),
        ("Max retries exceeded with url", "status.kaggle.com"),
    ],
)
def test_cli_failures_become_explanations(output, expected):
    completed = subprocess.CompletedProcess(["kaggle"], 1, stdout="", stderr=output)
    message = kr.translate_kaggle_failure(completed, "the push")
    assert expected in message
    assert output in message  # the raw output is always preserved


def test_an_unrecognised_failure_still_carries_its_output():
    completed = subprocess.CompletedProcess(
        ["kaggle"], 7, stdout="something entirely new", stderr=""
    )
    message = kr.translate_kaggle_failure(completed, "the push")
    assert "not one this script recognises" in message
    assert "something entirely new" in message


# -- states ----------------------------------------------------------------


@pytest.mark.parametrize(
    "reply, state",
    [
        ('someone/drishtisr-train has status "complete"', "complete"),
        ('someone/drishtisr-train has status "running"', "running"),
        ('someone/drishtisr-train has status "error"', "error"),
        ("has status queued", "queued"),
    ],
)
def test_states_are_read_out_of_kaggles_reply(reply, state):
    match = kr._STATE_RE.search(reply)
    assert match is not None and match.group(1).lower() == state


def test_success_is_exactly_one_state():
    """Anything unrecognised must be treated as 'still running', never as done.
    A status this script cannot read is reported as an error, not a pass."""
    assert kr.STATE_SUCCESS == "complete"
    assert kr.STATE_SUCCESS not in kr.STATE_FAILURES
    assert kr.STATE_SUCCESS not in kr.STATE_RUNNING


def test_terminal_state_exit_codes():
    logger = _silent_logger()
    assert kr.report_terminal_state("complete", "u/k", "train", logger) == 0
    assert kr.report_terminal_state("error", "u/k", "train", logger) == 1
    assert kr.report_terminal_state("cancelacknowledged", "u/k", "train", logger) == 1
    assert kr.report_terminal_state("running", "u/k", "train", logger) == 0


# -- logs ------------------------------------------------------------------


def test_kaggle_json_logs_are_rendered_as_text():
    payload = json.dumps(
        [
            {"stream_name": "stdout", "time": 0.1, "data": "line one\nline two\n"},
            {"stream_name": "stderr", "time": 0.2, "data": "a warning\n"},
        ]
    )
    lines = kr.format_log(payload)
    assert lines[:2] == ["line one", "line two"]
    assert lines[2] == "[stderr] a warning"


def test_a_plain_text_log_is_passed_through():
    """The log format has changed before; an unrecognised one is printed rather
    than discarded."""
    assert kr.format_log("just\ntext\n") == ["just", "text"]


# -- git guards ------------------------------------------------------------


def test_ssh_clone_urls_are_refused(cfg, monkeypatch):
    """The Kaggle session clones anonymously: no key, no credential helper."""
    monkeypatch.setitem(cfg.kaggle_run, "repo_url", "git@github.com:someone/repo.git")
    with pytest.raises(kr.RunError, match="SSH URL"):
        kr.remote_url(cfg)


def test_a_dirty_tree_refuses_the_push(cfg, monkeypatch):
    """The failure this whole check exists for: Kaggle clones from GitHub, so
    uncommitted work is simply not in the run, and nothing says so."""
    monkeypatch.setattr(kr, "git", _fake_git({"status": " M src/models/sr.py"}))
    with pytest.raises(kr.RunError, match="uncommitted changes"):
        kr.require_publishable_tree(cfg, _silent_logger(), enforce=True)


def test_an_unpushed_commit_refuses_the_push(cfg, monkeypatch):
    monkeypatch.setattr(kr, "git", _fake_git({"status": "", "branch": ""}))
    with pytest.raises(kr.RunError, match="not on any remote branch"):
        kr.require_publishable_tree(cfg, _silent_logger(), enforce=True)


def test_a_clean_pushed_tree_returns_the_sha(cfg, monkeypatch):
    sha = "d" * 40
    monkeypatch.setattr(
        kr, "git", _fake_git({"status": "", "branch": "  origin/main", "rev-parse": sha})
    )
    assert kr.require_publishable_tree(cfg, _silent_logger(), enforce=True) == sha


def test_smoke_mode_neither_fetches_nor_refuses(cfg, monkeypatch):
    """``--smoke`` must not touch the network, so it checks the working tree,
    warns, and stops -- it cannot start a wrong-code run in any case."""
    calls = []

    def recording_git(*args, **kwargs):
        calls.append(args)
        return {"status": " M x.py"}.get(args[0], "e" * 40)

    monkeypatch.setattr(kr, "git", recording_git)
    kr.require_publishable_tree(cfg, _silent_logger(), enforce=False)
    assert not any(args[0] == "fetch" for args in calls)


# -- GPU ledger ------------------------------------------------------------


def test_the_quota_week_starts_on_the_configured_day(cfg):
    start = kr.week_start(cfg)
    assert start.weekday() == int(cfg.kaggle_run.quota_week_starts_on)
    assert (start.hour, start.minute, start.second) == (0, 0, 0)


def test_a_missing_ledger_is_not_an_error(cfg, monkeypatch, tmp_path):
    monkeypatch.setitem(cfg.kaggle_run, "gpu_ledger", str(tmp_path / "none.json"))
    assert kr.read_ledger(cfg) == []


def test_a_corrupt_ledger_is_reported_not_overwritten(cfg, monkeypatch, tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setitem(cfg.kaggle_run, "gpu_ledger", str(path))
    with pytest.raises(kr.RunError, match="Could not read the GPU ledger"):
        kr.read_ledger(cfg)


def test_ledger_entries_round_trip(cfg, monkeypatch, tmp_path):
    monkeypatch.setitem(cfg.kaggle_run, "gpu_ledger", str(tmp_path / "ledger.json"))
    kr.append_ledger(cfg, {"job": "train", "expected_runtime_min": 60})
    kr.append_ledger(cfg, {"job": "train", "expected_runtime_min": 30})
    assert [entry["expected_runtime_min"] for entry in kr.read_ledger(cfg)] == [60, 30]


# -- helpers ---------------------------------------------------------------


def _fake_git(answers):
    """A stand-in for :func:`kaggle_run.git` keyed on the first argument.

    Returns a 40-character SHA for anything unspecified, so tests only state the
    answer they care about.
    """

    def fake(*args, **kwargs):
        return answers.get(args[0], "f" * 40)

    return fake


def _silent_logger():
    import logging

    logger = logging.getLogger("test_kaggle_run")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger
