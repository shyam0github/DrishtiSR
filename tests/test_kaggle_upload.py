"""Tests for the Kaggle upload tool and the Kaggle-mount path resolution.

The expensive failure this tool can produce is not a crash -- it is a 3.9 GB
upload that lands somewhere the notebook does not look, or a notebook that reads
an empty mount and trains on nothing while reporting success. So the tests
concentrate on:

1. **Slug/title validation happens locally**, before any transfer, for every
   rule Kaggle enforces remotely.
2. **The upload path and the read path agree.** ``paths.kaggle_dataset_dir`` and
   ``kaggle.dataset_slug`` must be the same string; if they drift, the data is
   pushed to one place and looked for in another.
3. **An empty mount is rejected, loudly**, and the error names every path that
   was checked.
4. **Every Kaggle CLI failure becomes an explanation**, and an unrecognised one
   still carries the raw output rather than swallowing it.

Nothing here touches the network or the real cache.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.config import load_config
from src.utils.paths import kaggle_mount_path, resolve_cache_dir, resolve_data_root

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import kaggle_upload as ku  # noqa: E402


# -- the drift guard -------------------------------------------------------


def test_mount_directory_matches_the_upload_slug():
    """The single most dangerous inconsistency in this tool.

    ``kaggle.dataset_slug`` decides where the data is PUSHED;
    ``paths.kaggle_dataset_dir`` decides where the notebook LOOKS. Kaggle mounts
    a dataset at ``/kaggle/input/<name>``, so the two must be the same string. If
    they drift, the upload succeeds, the notebook finds nothing, and the only
    symptom is a training run with zero samples.
    """
    cfg = load_config()
    assert str(cfg.paths.kaggle_dataset_dir) == str(cfg.kaggle.dataset_slug)


def test_configured_slug_and_title_satisfy_kaggle_rules():
    """The shipped defaults must be uploadable without editing anything."""
    cfg = load_config()
    ku.validate_slug(str(cfg.kaggle.dataset_slug), "kaggle.dataset_slug")
    ku.validate_slug(str(cfg.kaggle.dryrun_slug), "kaggle.dryrun_slug")
    ku.validate_title(str(cfg.kaggle.dataset_title), "kaggle.dataset_title")
    ku.validate_title(str(cfg.kaggle.dryrun_title), "kaggle.dryrun_title")


def test_dryrun_never_shares_a_slug_or_folder_with_the_real_upload():
    """A 5-sample test payload must not be able to overwrite the real dataset."""
    cfg = load_config()
    assert cfg.kaggle.dryrun_slug != cfg.kaggle.dataset_slug
    assert cfg.kaggle.dryrun_staging_dir != cfg.kaggle.staging_dir


def test_smoke_staging_is_separate_from_real_staging():
    """--smoke fabricates fake samples; they must not land in the real folder."""
    real = load_config()
    smoke = load_config(smoke=True)
    assert smoke.kaggle.staging_dir != real.kaggle.staging_dir
    assert smoke.kaggle.dryrun_staging_dir != real.kaggle.dryrun_staging_dir


# -- slug and title validation --------------------------------------------


@pytest.mark.parametrize(
    "slug, expected_hint",
    [
        ("myname/my-cache", "NAME only"),
        ("short", "6-50 characters"),
        ("x" * 51, "6-50 characters"),
        ("My_Cache_Data", "lowercase"),
        ("has spaces here", "lowercase"),
        ("-leading-hyphen", "lowercase"),
        ("trailing-hyphen-", "lowercase"),
        ("double--hyphen", "lowercase"),
    ],
)
def test_bad_slugs_are_rejected_before_any_upload(slug, expected_hint):
    with pytest.raises(ku.UploadError, match=expected_hint):
        ku.validate_slug(slug, "kaggle.dataset_slug")


@pytest.mark.parametrize(
    "slug", ["valid-slug", "drishtisr-sen2naipv2-cache", "abc123", "a1-b2-c3"]
)
def test_good_slugs_pass(slug):
    assert ku.validate_slug(slug, "kaggle.dataset_slug") == slug


def test_slug_suggestions_are_themselves_valid():
    """An error hint that suggests another invalid name is worse than no hint."""
    for text in ["My_Cache", "a", "SEN2NAIP v2 Cache!", "---", "x"]:
        suggestion = ku.suggest_slug(text)
        ku.validate_slug(suggestion, "suggestion")


@pytest.mark.parametrize("title", ["Cache", "abcde", ""])
def test_short_titles_are_rejected_before_any_upload(title):
    with pytest.raises(ku.UploadError, match="6-50 characters"):
        ku.validate_title(title, "kaggle.dataset_title")


def test_long_titles_are_rejected():
    with pytest.raises(ku.UploadError, match="6-50 characters"):
        ku.validate_title("T" * 51, "kaggle.dataset_title")


# -- credentials -----------------------------------------------------------


def _cfg_with_credentials(tmp_path, payload=None, write=True):
    cfg = load_config()
    path = tmp_path / "kaggle.json"
    if write:
        path.write_text(json.dumps(payload), encoding="utf-8")
    cfg.kaggle.credentials_file = str(path)
    return cfg


def test_missing_token_explains_how_to_get_one(tmp_path):
    cfg = _cfg_with_credentials(tmp_path, write=False)
    with pytest.raises(ku.UploadError) as excinfo:
        ku.read_kaggle_username(cfg)
    message = str(excinfo.value)
    assert "401" in message
    assert "kaggle.com/settings/account" in message
    assert "Create New Token" in message
    assert str(tmp_path / "kaggle.json") in message


def test_token_without_a_username_is_rejected(tmp_path):
    cfg = _cfg_with_credentials(tmp_path, {"key": "abc"})
    with pytest.raises(ku.UploadError, match="username"):
        ku.read_kaggle_username(cfg)


def test_malformed_token_file_is_rejected(tmp_path):
    cfg = _cfg_with_credentials(tmp_path, write=False)
    (tmp_path / "kaggle.json").write_text("not json at all", encoding="utf-8")
    with pytest.raises(ku.UploadError, match="Could not read"):
        ku.read_kaggle_username(cfg)


def test_username_is_read_from_the_token(tmp_path):
    cfg = _cfg_with_credentials(tmp_path, {"username": "someone", "key": "abc"})
    assert ku.read_kaggle_username(cfg) == "someone"


def test_metadata_is_generated_with_the_owner_prefixed_id(tmp_path):
    """The user never edits dataset-metadata.json; it is derived from the token."""
    path = ku.write_metadata(tmp_path, "someone", "my-dataset", "My Dataset Title")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["id"] == "someone/my-dataset"
    assert payload["title"] == "My Dataset Title"
    assert path.name == "dataset-metadata.json"


# -- Kaggle CLI failure translation ---------------------------------------


def _failure(stderr, code=1):
    return subprocess.CompletedProcess(
        args=["kaggle"], returncode=code, stdout="", stderr=stderr
    )


@pytest.mark.parametrize(
    "stderr, must_contain",
    [
        ("401 - Unauthorized", "Create New Token"),
        ("403 - Forbidden", "phone verification"),
        ("409 Conflict: dataset already exists", "version -m"),
        ("Error: title must be at least 6 characters", "dataset_title"),
        ("requests.exceptions.ConnectionError: Max retries exceeded", "NOT resumable"),
        ("404 - Not Found", "upload` first"),
    ],
)
def test_known_failures_become_plain_english(stderr, must_contain):
    message = ku.translate_kaggle_failure(_failure(stderr), "the upload")
    assert must_contain in message
    # The raw text is always kept: a translation must never lose evidence.
    assert stderr in message


def test_unknown_failure_still_shows_the_raw_output():
    """Never swallow an error just because it was not anticipated."""
    message = ku.translate_kaggle_failure(
        _failure("something nobody predicted"), "the upload"
    )
    assert "not one this script recognises" in message
    assert "something nobody predicted" in message


def test_failure_message_names_the_operation_and_exit_code():
    message = ku.translate_kaggle_failure(_failure("boom", code=7), "the version upload")
    assert "The version upload failed" in message
    assert "exit code 7" in message


# -- disk space ------------------------------------------------------------


def test_staging_refuses_to_start_without_room(tmp_path):
    """Refuse before copying, not halfway through."""
    with pytest.raises(ku.UploadError) as excinfo:
        ku.check_free_space(tmp_path, needed_bytes=10**18, margin_gb=2.0)
    message = str(excinfo.value)
    assert "Short by:" in message
    assert "staging_dir" in message


def test_staging_proceeds_when_there_is_room(tmp_path, capsys):
    ku.check_free_space(tmp_path, needed_bytes=1024, margin_gb=0.0)
    assert "OK --" in capsys.readouterr().out


# -- checksums -------------------------------------------------------------


def test_checksums_cover_every_staged_file_except_themselves(tmp_path):
    import csv

    cfg = load_config()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.npz").write_bytes(b"first")
    (tmp_path / "sub" / "b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")

    path, count = ku.write_checksums(tmp_path, cfg, logger=_NullLogger())

    assert count == 3
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    listed = {row["relative_path"] for row in rows}
    assert listed == {"sub/a.npz", "sub/b.json", "README.md"}
    # checksums.csv must not list itself -- it cannot contain its own hash.
    assert "checksums.csv" not in listed
    assert all(len(row[cfg.kaggle.checksum_algorithm]) > 0 for row in rows)


def test_checksums_change_when_a_file_changes(tmp_path):
    cfg = load_config()
    target = tmp_path / "a.npz"
    target.write_bytes(b"before")
    first = ku.hash_file(target, "blake2b", 16, 4096)
    target.write_bytes(b"after!")
    assert ku.hash_file(target, "blake2b", 16, 4096) != first


def test_unknown_hash_algorithm_fails_loudly(tmp_path):
    target = tmp_path / "a.npz"
    target.write_bytes(b"x")
    with pytest.raises(ku.UploadError, match="not a hash"):
        ku.hash_file(target, "definitely-not-a-hash", 16, 4096)


# -- missing cache ---------------------------------------------------------


def test_survey_refuses_and_names_the_path_it_looked_for(tmp_path):
    cfg = load_config()
    cfg.paths.cache_dir = str(tmp_path / "no-such-cache")
    with pytest.raises(ku.UploadError) as excinfo:
        ku.require_cache(cfg)
    message = str(excinfo.value)
    assert "no-such-cache" in message
    assert "prepare_data.py" in message


def test_empty_cache_directory_is_refused(tmp_path):
    cfg = load_config()
    subset = tmp_path / "cache" / str(cfg.sen2naipv2.subset)
    subset.mkdir(parents=True)
    cfg.paths.cache_dir = str(tmp_path / "cache")
    with pytest.raises(ku.UploadError, match="empty"):
        ku.require_cache(cfg)


# -- formatting ------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(0, "0 B"), (1536, "1.50 KB"), (1024**3, "1.00 GB"), (3 * 1024**4, "3.00 TB")],
)
def test_human_bytes(value, expected):
    assert ku.human_bytes(value) == expected


@pytest.mark.parametrize(
    "seconds, expected", [(12, "12 s"), (245, "4 min 5 s"), (5000, "1 h 23 min")]
)
def test_human_duration(seconds, expected):
    assert ku.human_duration(seconds) == expected


def test_upload_estimates_cover_every_configured_speed():
    cfg = load_config()
    lines = ku.upload_estimates(cfg, 1024**3)
    assert len(lines) == len(cfg.kaggle.upload_speeds_mbps)
    assert all("Mbps upload" in line for line in lines)


def test_public_flag_defaults_to_private():
    cfg = load_config()
    assert cfg.kaggle.is_private is True
    assert ku.public_flag(cfg) == []
    cfg.kaggle.is_private = False
    assert ku.public_flag(cfg) == ["-u"]


# -- the Kaggle mount, in src/utils/paths ---------------------------------


def _paths_cfg(**overrides):
    base = {
        "data_root": None,
        "kaggle_data_root": None,
        "local_data_root": None,
        "kaggle_mount_root": None,
        "kaggle_dataset_dir": None,
    }
    base.update(overrides)
    return {"paths": base}


def test_mount_path_is_composed_from_config_not_hardcoded():
    cfg = _paths_cfg(kaggle_mount_root="/somewhere/else", kaggle_dataset_dir="my-data")
    assert kaggle_mount_path(cfg) == Path("/somewhere/else/my-data")


def test_mount_path_is_none_when_not_configured():
    assert kaggle_mount_path(_paths_cfg()) is None
    assert kaggle_mount_path(_paths_cfg(kaggle_mount_root="/kaggle/input")) is None


def test_populated_mount_wins_over_every_other_candidate(tmp_path):
    mount_root, local = tmp_path / "input", tmp_path / "local"
    mount = mount_root / "my-data"
    mount.mkdir(parents=True)
    (mount / "sample.npz").write_bytes(b"x")
    local.mkdir()

    cfg = _paths_cfg(
        kaggle_mount_root=str(mount_root),
        kaggle_dataset_dir="my-data",
        kaggle_data_root=str(mount_root),
        local_data_root=str(local),
    )
    assert resolve_data_root(cfg) == mount.resolve()


def test_an_empty_mount_is_never_silently_accepted(tmp_path):
    """THE failure mode this design exists to prevent.

    /kaggle/input exists on every notebook. An attached-but-empty (or
    wrongly-named) dataset must not resolve, or training starts on zero samples
    and reports success.
    """
    mount_root = tmp_path / "input"
    (mount_root / "my-data").mkdir(parents=True)

    cfg = _paths_cfg(
        kaggle_mount_root=str(mount_root),
        kaggle_dataset_dir="my-data",
        kaggle_data_root=str(tmp_path / "absent"),
        local_data_root=str(tmp_path / "also-absent"),
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_data_root(cfg)

    message = str(excinfo.value)
    assert "EMPTY" in message
    assert "my-data" in message


def test_resolution_failure_names_every_path_it_checked(tmp_path):
    mount_root = tmp_path / "input"
    cfg = _paths_cfg(
        kaggle_mount_root=str(mount_root),
        kaggle_dataset_dir="my-data",
        kaggle_data_root=str(tmp_path / "no-kaggle"),
        local_data_root=str(tmp_path / "no-local"),
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_data_root(cfg)

    message = str(excinfo.value)
    for fragment in ("my-data", "no-kaggle", "no-local"):
        assert fragment in message
    assert "verify_data_root.py" in message
    assert "Add Input" in message


def test_cache_dir_resolves_to_a_populated_mount(tmp_path):
    """The whole point of the upload: the loader reads the mount, not a scratch dir."""
    mount_root = tmp_path / "input"
    mount = mount_root / "my-data"
    mount.mkdir(parents=True)
    (mount / "subset").mkdir()

    cfg = {
        "paths": {
            "kaggle_mount_root": str(mount_root),
            "kaggle_dataset_dir": "my-data",
            "cache_dir": str(tmp_path / "local-cache"),
            "kaggle_cache_dir": None,
        }
    }
    assert resolve_cache_dir(cfg) == mount.resolve()


def test_cache_dir_falls_back_to_local_when_the_mount_is_empty(tmp_path):
    mount_root = tmp_path / "input"
    (mount_root / "my-data").mkdir(parents=True)
    local = tmp_path / "local-cache"

    cfg = {
        "paths": {
            "kaggle_mount_root": str(mount_root),
            "kaggle_dataset_dir": "my-data",
            "cache_dir": str(local),
            "kaggle_cache_dir": None,
        }
    }
    assert resolve_cache_dir(cfg) == local.resolve()


def test_cache_dir_never_creates_the_read_only_mount(tmp_path):
    """A mount is Kaggle's to provide; creating it would fabricate a valid-looking
    empty dataset."""
    mount_root = tmp_path / "input"
    cfg = {
        "paths": {
            "kaggle_mount_root": str(mount_root),
            "kaggle_dataset_dir": "my-data",
            "cache_dir": str(tmp_path / "local-cache"),
            "kaggle_cache_dir": None,
        }
    }
    resolve_cache_dir(cfg)
    assert not (mount_root / "my-data").exists()


# -- CLI surface -----------------------------------------------------------


@pytest.mark.parametrize(
    "command", ["survey", "stage", "dryrun", "upload", "version"]
)
def test_every_subcommand_accepts_config_and_smoke(command):
    """CLAUDE.md: every entry point takes --config and --smoke."""
    argv = [command, "--config", "configs/base.yaml", "--smoke"]
    if command == "version":
        argv += ["-m", "test message"]
    args = ku.parse_args(argv)
    assert args.command == command
    assert args.smoke is True
    assert args.config == "configs/base.yaml"


def test_version_requires_a_message():
    with pytest.raises(SystemExit):
        ku.parse_args(["version"])


def test_repeated_set_flags_all_apply():
    """Two --set flags must accumulate, not silently discard the earlier one."""
    args = ku.parse_args(
        ["survey", "--set", "a.b=1", "--set", "c.d=2"]
    )
    assert args.overrides == ["a.b=1", "c.d=2"]


class _NullLogger:
    """Swallows log calls in tests that exercise functions needing a logger."""

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


# -- FIX 1: never invoke the bare executable ------------------------------


def test_kaggle_is_always_invoked_through_this_interpreter():
    """The bare `kaggle` executable is not on PATH on this machine.

    PowerShell reports "The term 'kaggle' is not recognized...". Going through
    ``sys.executable -m kaggle`` resolves the package inside the same interpreter
    that is running the script -- the only one guaranteed to have both the Kaggle
    client and the project's dependencies.
    """
    assert ku.kaggle_command() == [sys.executable, "-m", "kaggle"]


def test_no_bare_executable_lookup_remains_in_the_source():
    """A regression guard: shutil.which('kaggle') must never come back."""
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "kaggle_upload.py"
    ).read_text(encoding="utf-8")
    assert 'shutil.which("kaggle")' not in source


# -- FIX 2: preflight ------------------------------------------------------


def test_preflight_returns_the_username_and_is_fast(tmp_path, capsys):
    import time

    cfg = _cfg_with_credentials(tmp_path, {"username": "someone", "key": "k"})
    started = time.perf_counter()
    assert ku.preflight(cfg, _NullLogger()) == "someone"
    assert time.perf_counter() - started < 2.0

    out = capsys.readouterr().out
    assert "PREFLIGHT" in out
    assert "kaggle package importable" in out
    assert "someone" in out


def test_preflight_fails_on_a_missing_token_before_anything_expensive(tmp_path):
    cfg = _cfg_with_credentials(tmp_path, write=False)
    with pytest.raises(ku.UploadError) as excinfo:
        ku.preflight(cfg, _NullLogger())
    assert "Create New Token" in str(excinfo.value)


def test_preflight_names_the_interpreter_when_kaggle_is_missing(tmp_path, monkeypatch):
    """The remedy must point at THIS interpreter, not a generic `pip install`."""
    cfg = _cfg_with_credentials(tmp_path, {"username": "someone", "key": "k"})
    monkeypatch.setattr(ku.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ku.UploadError) as excinfo:
        ku.preflight(cfg, _NullLogger())
    message = str(excinfo.value)
    assert sys.executable in message
    assert "-m pip install kaggle" in message
    assert "does not help" in message


def test_preflight_does_not_import_kaggle(tmp_path, monkeypatch):
    """Importing the Kaggle client authenticates over the network at import time.

    A two-second local preflight must not do that, so it uses find_spec.
    """
    cfg = _cfg_with_credentials(tmp_path, {"username": "someone", "key": "k"})
    monkeypatch.delitem(sys.modules, "kaggle", raising=False)
    ku.preflight(cfg, _NullLogger())
    assert "kaggle" not in sys.modules


# -- dry-run layout verification ------------------------------------------


def _files_csv(names):
    body = "\n".join(f"{n},1024,2024-01-01" for n in names)
    return f"name,size,creationDate\n{body}\n"


def _fake_run(stdout, returncode=0):
    def runner(command, capture_output=True, text=True, timeout=None):
        return subprocess.CompletedProcess(command, returncode, stdout, "")

    return runner


def test_remote_listing_parses_the_csv(monkeypatch):
    monkeypatch.setattr(
        ku.subprocess, "run", _fake_run(_files_csv(["a/b.npz", "README.md"]))
    )
    names = ku.list_remote_files(load_config(), _NullLogger(), "me/ds")
    assert names == ["a/b.npz", "README.md"]


def test_remote_listing_returns_none_while_still_processing(monkeypatch):
    """No CSV header yet means 'ask again', not 'no files'."""
    monkeypatch.setattr(
        ku.subprocess, "run", _fake_run("Dataset is still being processed")
    )
    assert ku.list_remote_files(load_config(), _NullLogger(), "me/ds") is None


def test_remote_listing_returns_none_on_a_failed_call(monkeypatch):
    monkeypatch.setattr(ku.subprocess, "run", _fake_run("", returncode=1))
    assert ku.list_remote_files(load_config(), _NullLogger(), "me/ds") is None


def test_layout_verification_passes_when_samples_are_nested(monkeypatch, capsys):
    cfg = load_config()
    nested = [f"subset/s{i}.npz" for i in range(5)] + ["README.md", "checksums.csv"]
    monkeypatch.setattr(ku, "list_remote_files", lambda *a, **k: nested)

    assert ku.verify_remote_layout(cfg, _NullLogger(), "me/ds", "subset") is True
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "survived the round trip" in out


def test_layout_verification_fails_when_samples_are_flat(monkeypatch, capsys):
    """The failure the dry run exists to catch, at 6 MB instead of 3.9 GB."""
    cfg = load_config()
    flat = [f"s{i}.npz" for i in range(5)] + ["README.md"]
    monkeypatch.setattr(ku, "list_remote_files", lambda *a, **k: flat)

    assert ku.verify_remote_layout(cfg, _NullLogger(), "me/ds", "subset") is False
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "--dir-mode zip did not apply" in out
    assert "inherit exactly the same flaw" in out


def test_layout_verification_fails_when_a_zip_was_not_unpacked(monkeypatch, capsys):
    cfg = load_config()
    monkeypatch.setattr(
        ku, "list_remote_files", lambda *a, **k: ["subset.zip", "README.md"]
    )
    assert ku.verify_remote_layout(cfg, _NullLogger(), "me/ds", "subset") is False
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "not unpacked" in out or "did not expand" in out


def test_layout_verification_times_out_with_a_clear_message(monkeypatch, capsys):
    cfg = load_config()
    cfg.kaggle.verify_timeout_s = 1
    cfg.kaggle.verify_poll_interval_s = 0
    monkeypatch.setattr(ku, "list_remote_files", lambda *a, **k: None)

    assert ku.verify_remote_layout(cfg, _NullLogger(), "me/ds", "subset") is False
    out = capsys.readouterr().out
    assert "TIMED OUT" in out
    assert "NOT necessarily a failed upload" in out
    assert "verify_timeout_s" in out


# -- unattended upload -----------------------------------------------------


def test_upload_and_version_accept_yes_for_unattended_runs():
    assert ku.parse_args(["upload", "--yes"]).yes is True
    assert ku.parse_args(["version", "-m", "m", "--yes"]).yes is True


def test_confirmation_is_required_by_default():
    """--yes must be opt-in: a multi-GB irreversible transfer is not a default."""
    assert ku.parse_args(["upload"]).yes is False
    assert ku.parse_args(["version", "-m", "m"]).yes is False


def test_survey_and_stage_have_no_yes_flag():
    """Only the two network-writing commands take it."""
    for command in ("survey", "stage", "dryrun"):
        assert not hasattr(ku.parse_args([command]), "yes")


def test_upload_log_path_is_configured_not_hardcoded():
    cfg = load_config()
    assert str(cfg.paths.upload_log).endswith("upload.log")


def test_tabular_conversion_is_disabled_on_every_upload():
    """Kaggle converts tabular files to CSV by default, which would rewrite the
    manifest and split CSVs and invalidate every hash in checksums.csv."""
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "kaggle_upload.py"
    ).read_text(encoding="utf-8")
    assert source.count('"--keep-tabular"') == 3  # dryrun, upload, version


# -- verify-remote ---------------------------------------------------------


def test_verify_remote_is_a_registered_subcommand():
    args = ku.parse_args(["verify-remote", "--config", "configs/base.yaml"])
    assert args.command == "verify-remote"
    assert "verify-remote" in ku.HANDLERS


def test_listing_follows_every_page(monkeypatch):
    """Kaggle pages at 200. Reading one page would report 6008 files as 200."""
    pages = [
        ([{"name": f"f{i}.npz", "size": 10} for i in range(200)], "tok1"),
        ([{"name": f"g{i}.npz", "size": 10} for i in range(200)], "tok2"),
        ([{"name": "last.npz", "size": 10}], None),
    ]
    calls = []

    def fake_page(cfg, logger, slug, token):
        calls.append(token)
        return pages[len(calls) - 1]

    monkeypatch.setattr(ku, "_files_page", fake_page)
    names = ku.list_remote_files(load_config(), _NullLogger(), "me/ds")
    assert len(names) == 401
    assert calls == [None, "tok1", "tok2"]


def test_listing_returns_sizes_when_asked(monkeypatch):
    monkeypatch.setattr(
        ku,
        "_files_page",
        lambda *a: ([{"name": "a.npz", "size": 4096}], None),
    )
    assert ku.list_remote_files(
        load_config(), _NullLogger(), "me/ds", with_sizes=True
    ) == {"a.npz": 4096}


def test_listing_stops_at_the_configured_page_ceiling(monkeypatch):
    """A runaway pager must stop and be logged, never loop forever."""
    cfg = load_config()
    cfg.kaggle.max_listing_pages = 3
    pages = []

    def endless(c, lg, s, t):
        pages.append(t)
        # A distinct name per page, so the count reflects pages fetched rather
        # than being collapsed by the name-keyed accumulator.
        return [{"name": f"f{len(pages)}.npz", "size": 1}], "always-more"

    monkeypatch.setattr(ku, "_files_page", endless)
    names = ku.list_remote_files(cfg, _NullLogger(), "me/ds")
    assert len(pages) == 3, "must stop at the ceiling, not loop forever"
    assert len(names) == 3


def _staging_with_checksums(tmp_path, entries):
    import csv as _csv

    staging = tmp_path / "staging"
    staging.mkdir()
    with (staging / "checksums.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh)
        writer.writerow(["relative_path", "size_bytes", "blake2b"])
        for name, size in entries.items():
            writer.writerow([name, size, "deadbeef"])
    return staging


def _verify_cfg(tmp_path, entries, credentials=True):
    cfg = load_config()
    staging = _staging_with_checksums(tmp_path, entries)
    cfg.kaggle.staging_dir = str(staging.relative_to(staging.parents[1]))
    token = tmp_path / "kaggle.json"
    if credentials:
        token.write_text(json.dumps({"username": "me", "key": "k"}), encoding="utf-8")
    cfg.kaggle.credentials_file = str(token)
    return cfg, staging


def test_verify_remote_passes_when_everything_matches(tmp_path, monkeypatch, capsys):
    subset = load_config().sen2naipv2.subset
    entries = {
        f"{subset}/a.npz": 100,
        f"{subset}/a.json": 10,
        "README.md": 20,
        "dataset-metadata.json": 5,
    }
    cfg, staging = _verify_cfg(tmp_path, entries)
    monkeypatch.setattr(ku, "repo_root", lambda: staging.parents[1])

    remote = {k: v for k, v in entries.items() if k != "dataset-metadata.json"}
    remote["checksums.csv"] = 999
    monkeypatch.setattr(ku, "list_remote_files", lambda *a, **k: remote)

    args = ku.parse_args(["verify-remote"])
    assert ku.cmd_verify_remote(args, cfg, _NullLogger()) == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "complete and intact" in out


def test_verify_remote_detects_a_truncated_file(tmp_path, monkeypatch, capsys):
    """Sizes are what catch truncation -- the file name would still be there."""
    subset = load_config().sen2naipv2.subset
    entries = {f"{subset}/a.npz": 1_000_000, "README.md": 20}
    cfg, staging = _verify_cfg(tmp_path, entries)
    monkeypatch.setattr(ku, "repo_root", lambda: staging.parents[1])
    monkeypatch.setattr(
        ku,
        "list_remote_files",
        lambda *a, **k: {f"{subset}/a.npz": 512, "README.md": 20},
    )

    args = ku.parse_args(["verify-remote"])
    assert ku.cmd_verify_remote(args, cfg, _NullLogger()) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "SIZE MISMATCH" in out
    assert "truncated" in out.lower()


def test_verify_remote_detects_missing_files(tmp_path, monkeypatch, capsys):
    subset = load_config().sen2naipv2.subset
    entries = {f"{subset}/a.npz": 10, f"{subset}/b.npz": 10, "README.md": 20}
    cfg, staging = _verify_cfg(tmp_path, entries)
    monkeypatch.setattr(ku, "repo_root", lambda: staging.parents[1])
    monkeypatch.setattr(
        ku, "list_remote_files", lambda *a, **k: {f"{subset}/a.npz": 10, "README.md": 20}
    )

    args = ku.parse_args(["verify-remote"])
    assert ku.cmd_verify_remote(args, cfg, _NullLogger()) == 1
    out = capsys.readouterr().out
    assert "MISSING" in out
    assert "version -m" in out


def test_verify_remote_needs_a_staging_folder(tmp_path, monkeypatch):
    cfg, staging = _verify_cfg(tmp_path, {"a": 1})
    (staging / "checksums.csv").unlink()
    monkeypatch.setattr(ku, "repo_root", lambda: staging.parents[1])
    with pytest.raises(ku.UploadError, match="stage"):
        ku.cmd_verify_remote(ku.parse_args(["verify-remote"]), cfg, _NullLogger())


def test_verify_remote_explains_an_unqueryable_dataset(tmp_path, monkeypatch):
    cfg, staging = _verify_cfg(tmp_path, {"a": 1})
    monkeypatch.setattr(ku, "repo_root", lambda: staging.parents[1])
    monkeypatch.setattr(ku, "list_remote_files", lambda *a, **k: None)
    with pytest.raises(ku.UploadError, match="still being processed"):
        ku.cmd_verify_remote(ku.parse_args(["verify-remote"]), cfg, _NullLogger())
