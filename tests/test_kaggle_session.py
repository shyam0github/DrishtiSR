"""Tests for the functions the Day 1 notebook calls.

``notebooks/01_day1_baseline.ipynb`` contains no logic, which is only worth
anything if the logic it calls instead is tested. These run on CPU in under a
second, need no data and no network, and cover the parts whose failure would be
expensive on Kaggle rather than merely annoying: the archive that must not
contain itself, the fallback hint that must name the right edit, and the stage
runner that must stop the notebook rather than let four more stages run against
data that was never prepared.
"""

import inspect
import sys
import zipfile
from pathlib import Path

import pytest

from src.utils.kaggle_session import (
    SessionAborted,
    _elapsed,
    _free_space,
    _package_version,
    archive_outputs,
    dataset_fallback_hint,
    run_stage,
    stage_args,
)


# --- stage_args -----------------------------------------------------------


def test_stage_args_is_just_the_config_by_default():
    assert stage_args("configs/base.yaml") == ["--config", "configs/base.yaml"]


def test_stage_args_carries_the_dataset_override_as_a_config_set():
    """The fallback must reach every entry point, including the ones with no
    --dataset flag of their own (scripts/make_day1_gate.py)."""
    assert stage_args("c.yaml", dataset="worldstrat") == [
        "--config",
        "c.yaml",
        "--set",
        "dataset.name=worldstrat",
    ]


def test_stage_args_smoke_and_override_compose():
    assert stage_args("c.yaml", dataset="worldstrat", smoke=True) == [
        "--config",
        "c.yaml",
        "--smoke",
        "--set",
        "dataset.name=worldstrat",
    ]


# --- dataset_fallback_hint ------------------------------------------------


def test_fallback_hint_names_the_variable_the_value_and_the_cell():
    """The hint is the whole fallback procedure. If it does not name the exact
    edit, the reader has to already know it, and then it is not a fallback."""
    hint = dataset_fallback_hint("worldstrat")
    assert 'DATASET = "worldstrat"' in hint
    assert "dataset.name=worldstrat" in hint
    assert "configuration cell" in hint.lower()


def test_fallback_hint_warns_that_a_missing_mount_is_not_fixed_by_switching():
    hint = dataset_fallback_hint("worldstrat")
    assert "empty data root" in hint


# --- run_stage ------------------------------------------------------------


def test_run_stage_runs_every_step_in_order(capsys):
    elapsed = run_stage(
        "test stage",
        steps=[
            ("-c", ["print('first')"]),
            ("-c", ["print('second')"]),
        ],
    )
    out = capsys.readouterr().out
    assert out.index("first") < out.index("second")
    assert "step 1/2 ok" in out and "step 2/2 ok" in out
    assert "COMPLETE" in out
    assert elapsed >= 0.0


def test_run_stage_raises_on_a_failing_step_and_does_not_run_the_next(capsys):
    with pytest.raises(SessionAborted, match="exited with code 3"):
        run_stage(
            "test stage",
            steps=[
                ("-c", ["raise SystemExit(3)"]),
                ("-c", ["print('MUST NOT RUN')"]),
            ],
        )
    assert "MUST NOT RUN" not in capsys.readouterr().out


def test_run_stage_prints_the_failure_hint_before_raising(capsys):
    with pytest.raises(SessionAborted):
        run_stage(
            "test stage",
            steps=[("-c", ["raise SystemExit(1)"])],
            on_failure="SWITCH TO WORLDSTRAT",
        )
    assert "SWITCH TO WORLDSTRAT" in capsys.readouterr().out


def test_run_stage_stamps_a_start_time(capsys):
    run_stage("test stage", steps=[("-c", ["pass"])])
    assert "started 20" in capsys.readouterr().out


# --- archive_outputs ------------------------------------------------------


@pytest.fixture()
def run_tree(tmp_path):
    """A miniature of what a Day 1 run leaves behind, cache included."""
    (tmp_path / "outputs" / "metrics").mkdir(parents=True)
    (tmp_path / "outputs" / "metrics" / "baseline_bicubic.json").write_text("{}")
    (tmp_path / "outputs" / "figures").mkdir()
    (tmp_path / "outputs" / "figures" / "panel.png").write_bytes(b"\x89PNG")
    (tmp_path / "outputs" / "cache").mkdir()
    (tmp_path / "outputs" / "cache" / "huge.npz").write_bytes(b"0" * 4096)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "day1_gate.md").write_text("# gate")
    return tmp_path


def test_archive_packs_every_directory_with_relative_names(run_tree):
    archive = archive_outputs(["outputs", "reports"], "bundle.zip", repo_dir=str(run_tree))
    names = sorted(zipfile.ZipFile(archive).namelist())
    assert names == [
        "outputs/cache/huge.npz",
        "outputs/figures/panel.png",
        "outputs/metrics/baseline_bicubic.json",
        "reports/day1_gate.md",
    ]


def test_archive_skip_excludes_a_subtree(run_tree, capsys):
    archive = archive_outputs(
        ["outputs", "reports"],
        "bundle.zip",
        repo_dir=str(run_tree),
        skip=["outputs/cache"],
    )
    names = zipfile.ZipFile(archive).namelist()
    assert not any(name.startswith("outputs/cache/") for name in names)
    assert "outputs/metrics/baseline_bicubic.json" in names
    assert "1 excluded by `skip`" in capsys.readouterr().out


def test_archive_refuses_to_contain_itself(run_tree):
    """A zip written inside a directory it is packing grows until the disk does
    not, at the end of a long session."""
    with pytest.raises(SessionAborted, match="would contain itself"):
        archive_outputs(["outputs"], "outputs/bundle.zip", repo_dir=str(run_tree))


def test_archive_skips_a_missing_directory_but_still_packs_the_rest(run_tree, capsys):
    archive = archive_outputs(
        ["outputs", "never_ran"], "bundle.zip", repo_dir=str(run_tree)
    )
    assert "skipping never_ran" in capsys.readouterr().out
    assert zipfile.ZipFile(archive).namelist()


def test_archive_refuses_to_leave_an_empty_zip(tmp_path):
    """An empty archive downloads, opens, and looks like a result."""
    (tmp_path / "outputs").mkdir()
    with pytest.raises(SessionAborted, match="Nothing to archive"):
        archive_outputs(["outputs"], "bundle.zip", repo_dir=str(tmp_path))
    assert not (tmp_path / "bundle.zip").exists()


# --- helpers --------------------------------------------------------------


def test_free_space_walks_up_to_an_existing_ancestor(tmp_path):
    measured, free, total = _free_space(tmp_path / "does" / "not" / "exist")
    assert measured == tmp_path
    assert free is not None and total is not None


def test_free_space_reports_absent_rather_than_the_root_filesystem():
    """MEASURED: /kaggle/temp on the local Windows box walked all the way to the
    drive root and reported its free space under the label /kaggle/temp -- a real
    number about a directory that is not there."""
    if Path("/kaggle").is_dir():  # pragma: no cover -- only true on Kaggle
        pytest.skip("running on Kaggle, where /kaggle/temp genuinely exists")
    measured, free, total = _free_space(Path("/kaggle/temp/drishtisr_cache"))
    assert (measured, free, total) == (None, None, None)


@pytest.mark.parametrize(
    "seconds, expected",
    [(0, "00:00"), (9.4, "00:09"), (59.6, "01:00"), (600, "10:00"), (3725, "1:02:05")],
)
def test_elapsed_formatting(seconds, expected):
    assert _elapsed(seconds) == expected


def test_package_version_reports_absence_as_none():
    assert _package_version("definitely-not-a-real-distribution-name") is None
    assert _package_version("pytest") == pytest.__version__


def test_reported_packages_are_asked_for_by_distribution_name():
    """importlib.metadata wants the distribution name, not the import name --
    'opencv-python', not 'cv2'. A typo here reports NOT INSTALLED forever."""
    from src.utils.kaggle_session import _REPORTED_PACKAGES

    assert "cv2" not in _REPORTED_PACKAGES
    assert "opencv-python" in _REPORTED_PACKAGES
    assert _package_version("numpy") is not None


def test_run_stage_uses_this_interpreter(capsys):
    """Never a bare `python`: three interpreters answer to that name on the local
    box and two of them have none of the dependencies (see AGENTS.md)."""
    run_stage("test stage", steps=[("-c", ["pass"])])
    assert sys.executable in capsys.readouterr().out


# --- supporting_file_names ------------------------------------------------
#
# These pin the agreement between the two halves of the same contract:
# scripts/kaggle_upload.py::copy_supporting_files decides which names get
# uploaded INTO the Kaggle Dataset, and
# src/utils/kaggle_session.py::supporting_file_names decides which names are
# looked for coming OUT of it. If they drift, the upload succeeds, the mount
# looks right, and the job aborts on a file that was staged under another name.


def _config():
    from src.utils.config import load_config

    return load_config("configs/base.yaml")


def test_supporting_file_names_are_the_manifest_and_the_split():
    from src.utils.kaggle_session import supporting_file_names

    cfg = _config()
    names = supporting_file_names(cfg)
    assert names == [
        f"manifest_{cfg.dataset.name}.csv",
        str(cfg.splits.output_name).format(dataset=cfg.dataset.name),
    ]


def test_supporting_file_names_match_what_kaggle_upload_stages():
    """The reader's list must equal the writer's list, name for name."""
    import scripts.kaggle_upload as upload  # noqa: PLC0415
    from src.utils.kaggle_session import supporting_file_names

    cfg = _config()
    source = inspect.getsource(upload.copy_supporting_files)
    for name in supporting_file_names(cfg):
        stem = name.split("_")[0]
        assert stem in source, (
            f"{name!r} is staged for by supporting_file_names() but nothing in "
            "copy_supporting_files() writes a file of that shape."
        )
    # And the composition rules themselves, which is what actually drifts.
    assert 'f"manifest_{dataset_name}.csv"' in source
    assert "cfg.splits.output_name" in source


def test_supporting_file_names_follow_the_dataset_override():
    """The fallback changes dataset.name, and both file names must move with it
    -- a staged manifest_sen2naipv2.csv is not the worldstrat manifest."""
    from src.utils.kaggle_session import supporting_file_names

    cfg = _config()
    cfg.dataset.name = "worldstrat"
    assert supporting_file_names(cfg) == [
        "manifest_worldstrat.csv",
        "splits_worldstrat.csv",
    ]
