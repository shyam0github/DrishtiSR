"""Tests for staging the manifest and split CSVs out of the mounted dataset.

THE FAILURE BEING GUARDED. The Kaggle Dataset carries the pixels *and* two small
CSVs saying which samples were validated and which are train/val/test. Nothing
copied them into the working directory, so a job ran with the imagery present
and the split file absent -- and a missing split file does not crash. The loader
recomputes a split in-process, which is reproducible and completely wrong:
adjacent NAIP tiles overlap, so a recomputed split is not the geographic split
the baseline was measured on, and every number downstream is quietly
incomparable.

The tests therefore care about three things, in order of what would actually
cost a session:

1. Missing CSVs **abort**. A silently recomputed split is the whole point.
2. The mount path is **resolved**, never composed from a literal. That layout
   has already been wrong once here.
3. Every job inherits the staging cell, because it lives in the one template.

CPU-only, no network, no real Kaggle mount.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.config import load_config
from src.utils.kaggle_session import (
    SessionAborted,
    stage_supporting_files,
    supporting_file_names,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import kaggle_run as kr  # noqa: E402

MANIFEST = "manifest_sen2naipv2.csv"
SPLITS = "splits_sen2naipv2.csv"


@pytest.fixture(autouse=True)
def real_outputs_are_untouched():
    """Tripwire: no test in this module may write to the real ``outputs/``.

    Not hypothetical. The first version of ``stage_supporting_files`` resolved
    its destination through ``resolve_output_path``, which anchors relative
    paths at ``repo_root()`` -- derived from the installed module's own location
    and therefore blind to the ``repo_dir`` argument. Running these tests
    truncated the real 3,000-row ``outputs/manifest_sen2naipv2.csv`` and
    ``outputs/splits_sen2naipv2.csv`` to a two-line fixture, which would have
    silently changed the split every later run used.

    Fingerprinting size and mtime around every test turns a repeat of that into
    a failure here rather than a mystery three commits later.
    """
    watched = [
        kr.repo_root() / "outputs" / MANIFEST,
        kr.repo_root() / "outputs" / SPLITS,
    ]
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in watched
        if path.is_file()
    }
    yield
    for path, fingerprint in before.items():
        assert path.is_file(), f"a test deleted the real {path.name}"
        now = (path.stat().st_size, path.stat().st_mtime_ns)
        assert now == fingerprint, (
            f"a test overwrote the real {path}. Restore it from "
            f"outputs/kaggle_staging/{path.name}, which is the copy that was "
            "uploaded to Kaggle, then fix the test to pass repo_dir."
        )


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def jobs(cfg):
    return kr.load_jobs(cfg)


def make_repo(tmp_path: Path, mount_name: str = "cache", write: tuple = ()) -> Path:
    """Build a throwaway repo + mount and return the repo root.

    Args:
        tmp_path: pytest temporary directory.
        mount_name: Directory name the "dataset" is mounted under.
        write: File names to create inside the mount.

    Returns:
        The repo root, containing ``configs/base.yaml`` pointed at the mount.
    """
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    mount_root = tmp_path / "input"
    mount = mount_root / mount_name
    mount.mkdir(parents=True)
    for name in write:
        (mount / name).write_text("sample_id,split\nx,train\n", encoding="utf-8")

    config = {
        "seed": 42,
        "paths": {
            "kaggle_mount_root": str(mount_root).replace("\\", "/"),
            "kaggle_dataset_dir": mount_name,
            "kaggle_mount_patterns": ["{root}/{name}"],
            "manifest_dir": "outputs",
            "output_root": "outputs",
        },
        "dataset": {"name": "sen2naipv2"},
        "splits": {"output_name": "splits_{dataset}.csv"},
    }
    OmegaConf.save(OmegaConf.create(config), repo / "configs" / "base.yaml")
    return repo


# -- file names come from config, not from a literal -----------------------


def test_file_names_are_derived_from_the_configured_dataset(cfg):
    assert supporting_file_names(cfg) == [MANIFEST, SPLITS]


def test_changing_the_dataset_moves_both_file_names(cfg):
    switched = OmegaConf.merge(cfg, OmegaConf.create({"dataset": {"name": "worldstrat"}}))
    assert supporting_file_names(switched) == [
        "manifest_worldstrat.csv",
        "splits_worldstrat.csv",
    ]


def test_the_names_match_what_the_uploader_stages(cfg):
    """Upload and staging must agree, or the files are published under names
    the notebook will not look for."""
    import kaggle_upload as ku  # noqa: E402

    source = (kr.repo_root() / "scripts" / "kaggle_upload.py").read_text(encoding="utf-8")
    assert 'manifest_{dataset_name}.csv' in source
    assert 'cfg.splits.output_name' in source
    assert ku is not None


# -- the happy path --------------------------------------------------------


def test_both_files_are_copied_into_outputs(tmp_path):
    repo = make_repo(tmp_path, write=(MANIFEST, SPLITS))
    staged = stage_supporting_files("configs/base.yaml", repo_dir=str(repo))

    assert staged == [MANIFEST, SPLITS]
    assert (repo / "outputs" / MANIFEST).is_file()
    assert (repo / "outputs" / SPLITS).is_file()


def test_contents_are_copied_verbatim(tmp_path):
    """A truncated or rewritten CSV would change the split without saying so."""
    repo = make_repo(tmp_path, write=(MANIFEST, SPLITS))
    mount = tmp_path / "input" / "cache"
    (mount / SPLITS).write_text("sample_id,split\na,train\nb,val\n", encoding="utf-8")

    stage_supporting_files("configs/base.yaml", repo_dir=str(repo))
    assert (repo / "outputs" / SPLITS).read_text(encoding="utf-8") == (
        "sample_id,split\na,train\nb,val\n"
    )


def test_restaging_over_an_existing_file_succeeds(tmp_path):
    """A re-run must not fail on a file the previous run already copied."""
    repo = make_repo(tmp_path, write=(MANIFEST, SPLITS))
    stage_supporting_files("configs/base.yaml", repo_dir=str(repo))
    staged = stage_supporting_files("configs/base.yaml", repo_dir=str(repo))
    assert staged == [MANIFEST, SPLITS]


# -- the failures that must abort -----------------------------------------


def test_a_missing_split_file_aborts_the_run(tmp_path):
    """The headline case: pixels present, split absent. Must NOT proceed."""
    repo = make_repo(tmp_path, write=(MANIFEST,))
    with pytest.raises(SessionAborted) as excinfo:
        stage_supporting_files("configs/base.yaml", repo_dir=str(repo))
    message = str(excinfo.value)
    assert SPLITS in message
    assert "recomputes a split" in message


def test_a_missing_manifest_aborts_the_run(tmp_path):
    repo = make_repo(tmp_path, write=(SPLITS,))
    with pytest.raises(SessionAborted, match=MANIFEST):
        stage_supporting_files("configs/base.yaml", repo_dir=str(repo))


def test_both_missing_are_named_together(tmp_path):
    repo = make_repo(tmp_path, write=())
    with pytest.raises(SessionAborted) as excinfo:
        stage_supporting_files("configs/base.yaml", repo_dir=str(repo))
    message = str(excinfo.value)
    assert MANIFEST in message and SPLITS in message


def test_the_failure_lists_what_is_actually_in_the_mount(tmp_path):
    """"Not found" without this is not diagnosable from a Kaggle log."""
    repo = make_repo(tmp_path, write=("something_else.csv",))
    with pytest.raises(SessionAborted) as excinfo:
        stage_supporting_files("configs/base.yaml", repo_dir=str(repo))
    assert "something_else.csv" in str(excinfo.value)


def test_a_missing_mount_directory_aborts_with_the_path_named(tmp_path):
    repo = make_repo(tmp_path, write=(MANIFEST, SPLITS))
    config = OmegaConf.load(repo / "configs" / "base.yaml")
    config.paths.kaggle_dataset_dir = "not-attached"
    OmegaConf.save(config, repo / "configs" / "base.yaml")

    with pytest.raises(SessionAborted, match="not-attached"):
        stage_supporting_files("configs/base.yaml", repo_dir=str(repo))


def test_no_configured_mount_aborts_rather_than_guessing(tmp_path):
    repo = make_repo(tmp_path, write=(MANIFEST, SPLITS))
    config = OmegaConf.load(repo / "configs" / "base.yaml")
    del config.paths["kaggle_mount_root"]
    del config.paths["kaggle_dataset_dir"]
    OmegaConf.save(config, repo / "configs" / "base.yaml")

    with pytest.raises(SessionAborted, match="No Kaggle mount is configured"):
        stage_supporting_files("configs/base.yaml", repo_dir=str(repo))


# -- the mount layout is resolved, never hardcoded -------------------------


def test_the_owner_nested_mount_layout_is_found(tmp_path):
    """MEASURED on Kaggle: the real layout is /kaggle/input/datasets/<owner>/<slug>.

    This project assumed the flat one and every run found no data. Staging must
    go through the same resolver, so a glob layout resolves here too.
    """
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    mount_root = tmp_path / "input"
    nested = mount_root / "datasets" / "someowner" / "cache"
    nested.mkdir(parents=True)
    for name in (MANIFEST, SPLITS):
        (nested / name).write_text("a,b\n1,2\n", encoding="utf-8")

    OmegaConf.save(
        OmegaConf.create(
            {
                "seed": 42,
                "paths": {
                    "kaggle_mount_root": str(mount_root).replace("\\", "/"),
                    "kaggle_dataset_dir": "cache",
                    "kaggle_mount_patterns": [
                        "{root}/datasets/*/{name}",
                        "{root}/{name}",
                    ],
                    "manifest_dir": "outputs",
                    "output_root": "outputs",
                },
                "dataset": {"name": "sen2naipv2"},
                "splits": {"output_name": "splits_{dataset}.csv"},
            }
        ),
        repo / "configs" / "base.yaml",
    )

    assert stage_supporting_files("configs/base.yaml", repo_dir=str(repo)) == [
        MANIFEST,
        SPLITS,
    ]


def code_strings(source: str, function: str = "") -> list:
    """Every string literal in executable code, excluding docstrings and comments.

    Prose is allowed to name ``/kaggle/input`` -- explaining that the layout was
    wrong once is exactly why these comments exist. What must never appear is a
    mount path in an *expression*, because that is a second copy of the
    assumption the resolver exists to hold.

    Args:
        source: Python source text.
        function: Restrict to this function's body, or "" for the whole module.

    Returns:
        The string literal values, docstrings removed.
    """
    tree = ast.parse(source)
    if function:
        tree = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == function
        )
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    ]


def test_the_template_composes_no_mount_path():
    """The mount layout has been wrong once. A second literal is how it recurs."""
    template = (kr.repo_root() / "notebooks" / "templates" / "kaggle_job.py").read_text(
        encoding="utf-8"
    )
    for line in template.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # prose may name the path it is warning about
        assert "/kaggle/input" not in line, line


def test_the_staging_function_composes_no_mount_path():
    source = (kr.repo_root() / "src" / "utils" / "kaggle_session.py").read_text(
        encoding="utf-8"
    )
    for value in code_strings(source, "stage_supporting_files"):
        assert "/kaggle/input" not in value, value
    body = source.split("def stage_supporting_files", 1)[1].split("\ndef ", 1)[0]
    assert "kaggle_mount_path" in body


def test_staging_never_writes_outside_the_repo_it_was_given(tmp_path):
    """The bug this test was written for: the destination ignored repo_dir and
    resolved against the installed module's repo root, so a test with a
    temporary repo overwrote the real outputs/manifest_sen2naipv2.csv."""
    repo = make_repo(tmp_path, write=(MANIFEST, SPLITS))
    stage_supporting_files("configs/base.yaml", repo_dir=str(repo))

    assert (repo / "outputs" / MANIFEST).is_file()
    # Nothing landed in the real repository.
    real = kr.repo_root() / "outputs" / MANIFEST
    if real.is_file():
        assert real.stat().st_size > 1000, (
            "the real manifest looks like it was overwritten by a test fixture"
        )


# -- every job inherits it, because it is in the template ------------------


def test_every_generated_notebook_stages_before_the_entry_point(cfg, jobs):
    """In the template, not per-job: that is what makes it inherited."""
    for name, job in jobs.items():
        tokens = kr.build_tokens(cfg, name, job, "a" * 40, "https://example.com/r.git")
        source = kr.render_template(
            (kr.repo_root() / str(job.template)).read_text(encoding="utf-8"), tokens
        )
        assert "stage_supporting_files(" in source, name

        staging = source.index("stage_supporting_files(")
        guard = source.index("guard_data_root(")
        entry = source.index("run_entry(")
        assert staging < guard < entry, (
            f"job {name!r}: staging must precede the guard, which must precede "
            "the entry point"
        )


def test_generated_notebooks_still_parse(cfg, jobs):
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
        ast.parse(code)


# -- the config the staging cell is handed ---------------------------------


def test_the_config_path_comes_from_the_jobs_own_entry_args(jobs):
    assert kr.config_path_for(jobs["baseline"]) == "configs/base.yaml"


def test_a_job_with_a_different_config_stages_from_that_config():
    job = OmegaConf.create(
        {"entry_args": ["--config", "configs/experiment.yaml"], "guard_args": []}
    )
    assert kr.config_path_for(job) == "configs/experiment.yaml"


def test_the_equals_form_is_understood():
    job = OmegaConf.create({"entry_args": ["--config=configs/other.yaml"]})
    assert kr.config_path_for(job) == "configs/other.yaml"


def test_a_job_with_no_config_arg_falls_back_to_the_default():
    job = OmegaConf.create({"entry_args": ["--smoke"], "guard_args": []})
    assert kr.config_path_for(job) == "configs/base.yaml"


def test_the_guard_args_are_used_when_the_entry_takes_no_config(jobs):
    job = OmegaConf.create(
        {"entry_args": [], "guard_args": ["--config", "configs/guard.yaml"]}
    )
    assert kr.config_path_for(job) == "configs/guard.yaml"
