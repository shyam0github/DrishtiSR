"""Tests for src.utils.paths.resolve_data_root.

These use plain dicts rather than OmegaConf so the resolver's contract is pinned
independently of the config library, and tmp_path stands in for /kaggle/input.
"""

import pytest

from src.utils.paths import repo_root, resolve_data_root


def _cfg(**paths):
    base = {"data_root": None, "kaggle_data_root": None, "local_data_root": None}
    base.update(paths)
    return {"paths": base}


def test_prefers_kaggle_when_both_exist(tmp_path):
    kaggle = tmp_path / "kaggle" / "input"
    local = tmp_path / "local" / "data"
    kaggle.mkdir(parents=True)
    local.mkdir(parents=True)

    cfg = _cfg(kaggle_data_root=str(kaggle), local_data_root=str(local))

    assert resolve_data_root(cfg) == kaggle.resolve()


def test_falls_back_to_local_when_kaggle_absent(tmp_path):
    local = tmp_path / "local" / "data"
    local.mkdir(parents=True)

    cfg = _cfg(
        kaggle_data_root=str(tmp_path / "no" / "such" / "kaggle"),
        local_data_root=str(local),
    )

    assert resolve_data_root(cfg) == local.resolve()


def test_explicit_override_wins_over_kaggle(tmp_path):
    kaggle = tmp_path / "kaggle" / "input"
    override = tmp_path / "override"
    kaggle.mkdir(parents=True)
    override.mkdir()

    cfg = _cfg(data_root=str(override), kaggle_data_root=str(kaggle))

    assert resolve_data_root(cfg) == override.resolve()


def test_missing_override_raises_rather_than_falling_back(tmp_path):
    """A typo'd override must fail loudly, not quietly train on the local mirror."""
    kaggle = tmp_path / "kaggle" / "input"
    kaggle.mkdir(parents=True)

    cfg = _cfg(data_root=str(tmp_path / "typo"), kaggle_data_root=str(kaggle))

    with pytest.raises(NotADirectoryError):
        resolve_data_root(cfg)


def test_returns_absolute_resolved_path(tmp_path, monkeypatch):
    local = tmp_path / "data"
    local.mkdir()
    monkeypatch.chdir(tmp_path)

    cfg = _cfg(kaggle_data_root="nope", local_data_root="data")
    result = resolve_data_root(cfg)

    assert result.is_absolute()
    assert result == local.resolve()


def test_no_candidate_exists_raises_file_not_found(tmp_path):
    cfg = _cfg(
        kaggle_data_root=str(tmp_path / "a"),
        local_data_root=str(tmp_path / "b"),
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_data_root(cfg)

    # The message must name both attempts, or the failure is undiagnosable.
    assert "kaggle_data_root" in str(excinfo.value)
    assert "local_data_root" in str(excinfo.value)


def test_file_is_not_accepted_as_data_root(tmp_path):
    not_a_dir = tmp_path / "data.tif"
    not_a_dir.write_bytes(b"")

    cfg = _cfg(kaggle_data_root=str(not_a_dir), local_data_root=str(tmp_path / "b"))

    with pytest.raises(FileNotFoundError):
        resolve_data_root(cfg)


def test_does_not_create_the_data_root(tmp_path):
    missing = tmp_path / "missing"
    cfg = _cfg(kaggle_data_root=str(missing), local_data_root=str(tmp_path / "b"))

    with pytest.raises(FileNotFoundError):
        resolve_data_root(cfg)
    assert not missing.exists()


def test_missing_paths_section_raises_key_error():
    with pytest.raises(KeyError):
        resolve_data_root({"seed": 42})


def test_missing_candidate_key_raises_key_error(tmp_path):
    with pytest.raises(KeyError):
        resolve_data_root({"paths": {"local_data_root": str(tmp_path)}})


def test_blank_candidate_raises_value_error(tmp_path):
    cfg = _cfg(kaggle_data_root="   ", local_data_root=str(tmp_path))

    with pytest.raises(ValueError):
        resolve_data_root(cfg)


def test_repo_root_contains_the_project():
    root = repo_root()

    assert (root / "src" / "utils" / "paths.py").is_file()
    assert (root / "configs" / "base.yaml").is_file()
