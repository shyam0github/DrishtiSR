"""Retry behaviour for the SEN2NAIPv2 download.

Regression tests for a measured failure: an unattended 3000-sample fetch died
after 121 samples with ``FileNotFoundError`` on the .taco URL. fsspec maps every
non-OK HTTP status to ``FileNotFoundError``, so HuggingFace rate limiting is
indistinguishable from a missing file at that layer, and one blip aborted an
~8.7 hour job at 4% completion.

These tests exercise the retry logic directly on an uninitialised instance,
because constructing the real dataset requires network access to fetch the taco
catalog and these must run offline.
"""

import logging

import pytest

from src.data.sen2naip import SEN2NAIPv2Dataset


def _bare_dataset(max_retries=4, n_samples=3):
    """A SEN2NAIPv2Dataset with only the attributes the retry path touches."""
    obj = SEN2NAIPv2Dataset.__new__(SEN2NAIPv2Dataset)
    obj.max_retries = max_retries
    obj.retry_backoff_s = 0.0  # no real sleeping in tests
    obj.request_delay_s = 0.0
    obj.subset = "sen2naipv2-crosssensor"
    obj.cache_dir = "<test>"
    obj.logger = logging.getLogger("test.sen2naip")
    obj._catalog = [{"sample_id": f"s{i}"} for i in range(n_samples)]
    return obj


def test_retries_transient_failure_then_succeeds():
    dataset = _bare_dataset(max_retries=4)
    attempts = []

    def flaky(idx):
        attempts.append(idx)
        if len(attempts) < 3:
            raise FileNotFoundError("https://huggingface.co/... (really a 429)")
        return "cached"

    dataset.ensure_cached = flaky

    assert SEN2NAIPv2Dataset._fetch_with_retry(dataset, 0) is True
    assert len(attempts) == 3, "should have retried twice before succeeding"


def test_gives_up_after_max_retries_and_reports_false():
    dataset = _bare_dataset(max_retries=3)
    attempts = []

    def always_fails(idx):
        attempts.append(idx)
        raise FileNotFoundError("persistent 404")

    dataset.ensure_cached = always_fails

    assert SEN2NAIPv2Dataset._fetch_with_retry(dataset, 0) is False
    assert len(attempts) == 3, "should attempt exactly max_retries times"


def test_backoff_is_exponential(monkeypatch):
    dataset = _bare_dataset(max_retries=5)
    dataset.retry_backoff_s = 15.0
    delays = []
    monkeypatch.setattr("src.data.sen2naip.time.sleep", delays.append)
    dataset.ensure_cached = lambda idx: (_ for _ in ()).throw(OSError("boom"))

    SEN2NAIPv2Dataset._fetch_with_retry(dataset, 0)

    assert delays == [15.0, 30.0, 60.0, 120.0]


def test_prepare_continues_past_a_failed_sample_and_counts_it():
    """One dead record must not discard hours of completed downloads."""
    dataset = _bare_dataset(max_retries=2, n_samples=3)
    dataset.is_cached = lambda idx: False

    def fetch(idx):
        if idx == 1:
            raise FileNotFoundError("gone")
        return "cached"

    dataset.ensure_cached = fetch

    counts = SEN2NAIPv2Dataset.prepare(dataset)

    assert counts == {"cached": 0, "fetched": 2, "failed": 1}


def test_prepare_skips_already_cached_samples():
    dataset = _bare_dataset(n_samples=3)
    dataset.is_cached = lambda idx: idx < 2
    fetched = []
    dataset.ensure_cached = lambda idx: fetched.append(idx)

    counts = SEN2NAIPv2Dataset.prepare(dataset)

    assert counts == {"cached": 2, "fetched": 1, "failed": 0}
    assert fetched == [2], "cached samples must not be re-fetched"


def test_prepare_honours_limit():
    dataset = _bare_dataset(n_samples=3)
    dataset.is_cached = lambda idx: False
    dataset.ensure_cached = lambda idx: None

    counts = SEN2NAIPv2Dataset.prepare(dataset, limit=2)

    assert counts["fetched"] == 2


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("fsspec maps 429/404 to this"),
        OSError("connection reset"),
        TimeoutError("read timed out"),
        RuntimeError("nested record missing lr/hr"),
    ],
)
def test_retry_net_covers_the_failure_modes_seen_in_practice(exc):
    dataset = _bare_dataset(max_retries=2)
    attempts = []

    def fail_once(idx):
        attempts.append(idx)
        if len(attempts) == 1:
            raise exc
        return "cached"

    dataset.ensure_cached = fail_once

    assert SEN2NAIPv2Dataset._fetch_with_retry(dataset, 0) is True
