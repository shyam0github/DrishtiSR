"""The pre-flight guard, and the check that was missing from it.

MEASURED 2026-09-09. The `day3` Kaggle job passed ``verify_data_root.py`` and
died seven seconds into training:

    SplitError: Split file outputs/splits_sen2naipv2.csv covers 3000 samples
    but 1409 of the dataset's 4409 samples are absent from it

The cache had been expanded locally and the split regenerated; the Kaggle
dataset still carried the pre-expansion CSV. Every check the guard ran passed,
because every one of them asked whether the file was THERE -- it was, readable,
with the right columns and sensible per-split counts. None asked whether it was
CURRENT.

So the guard now runs the loader's own ``resolve_split_assignments``. These
tests hold that in place: it must pass on a covering split, FAIL on a stale one,
and stay agnostic when there is no split file at all (which section 4 already
fails the run for, and reporting it twice would hide which check caught it).

CPU only, synthetic stub, no network, no cache.
"""

from __future__ import annotations

import csv

from scripts.verify_data_root import check_split_covers_catalog
from src.data.registry import get_dataset
from src.utils.config import load_config
from src.utils.logging import get_logger


def _cfg(tmp_path):
    return load_config(
        "configs/base.yaml",
        smoke=True,
        overrides=[
            f"paths.manifest_dir={tmp_path.as_posix()}",
            f"paths.log_file={(tmp_path / 'run.log').as_posix()}",
            "loader.cached_only=false",
        ],
    )


def _logger(tmp_path):
    return get_logger("test.verify", log_file=tmp_path / "run.log")


def _split_path(cfg, tmp_path):
    from pathlib import Path

    name = str(cfg.splits.output_name).format(dataset=cfg.dataset.name)
    return Path(tmp_path) / name


def _sample_ids(cfg):
    """Every sample id the split has to cover.

    Through the loader's own _split_records, which is what
    resolve_split_assignments enumerates: it reads dataset.catalog when
    there is one and falls back to sample metadata otherwise, and the synthetic
    stub takes the second path.
    """
    from src.data.loader import _split_records

    return [str(record["sample_id"]) for record in _split_records(get_dataset(cfg))]


def _write_split(path, sample_ids):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "split"])
        for index, sample_id in enumerate(sample_ids):
            writer.writerow([sample_id, "val" if index % 4 == 0 else "train"])


def test_a_covering_split_passes(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    _write_split(_split_path(cfg, tmp_path), _sample_ids(cfg))

    assert check_split_covers_catalog(cfg, _logger(tmp_path)) is True
    assert "covers every" in capsys.readouterr().out


def test_a_stale_split_fails(tmp_path, capsys):
    """THE regression test. A split missing later samples must not certify a run.

    This is the Day 3 failure in miniature: the CSV is present, readable and
    well-formed, and describes a catalog smaller than the one on disk.
    """
    cfg = _cfg(tmp_path)
    ids = _sample_ids(cfg)
    assert len(ids) > 2, "the stub must hold enough samples to drop one"
    _write_split(_split_path(cfg, tmp_path), ids[:-1])

    assert check_split_covers_catalog(cfg, _logger(tmp_path)) is False

    out = capsys.readouterr().out
    assert "does NOT cover" in out
    # The message has to carry the fix, not just the diagnosis: this is read on
    # Kaggle, by someone who has just lost a queue wait.
    assert "make_splits.py" in out
    assert "kaggle_upload.py" in out


def test_no_split_file_is_undefined_not_failed(tmp_path, capsys):
    """Section 4 already fails the run for this; reporting it twice hides the cause."""
    cfg = _cfg(tmp_path)
    assert not _split_path(cfg, tmp_path).exists()

    assert check_split_covers_catalog(cfg, _logger(tmp_path)) is None
    assert "undefined" in capsys.readouterr().out


def test_the_guard_uses_the_loaders_own_function(tmp_path):
    """Not a reimplementation -- a second opinion could drift from the first.

    Asserted by patching the loader's ``resolve_split_assignments`` and checking
    the guard's verdict follows it. If the guard ever grew its own count
    comparison, this would keep passing while the guard and the trainer quietly
    disagreed, so it also pins the SOURCE of the answer.
    """
    from src.data import loader as loader_module

    cfg = _cfg(tmp_path)
    _write_split(_split_path(cfg, tmp_path), _sample_ids(cfg))

    calls = []
    original = loader_module.resolve_split_assignments

    def spy(cfg_arg, dataset, logger):
        calls.append(dataset)
        return original(cfg_arg, dataset, logger)

    loader_module.resolve_split_assignments = spy
    try:
        assert check_split_covers_catalog(cfg, _logger(tmp_path)) is True
    finally:
        loader_module.resolve_split_assignments = original

    assert len(calls) == 1, (
        "the guard did not call src.data.loader.resolve_split_assignments, so "
        "it is answering the coverage question with its own logic and can drift "
        "from what src/train.py actually does."
    )
