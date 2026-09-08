"""The cache manifest and per-pair validation. CPU-only, no data required.

These tests exist because the expansion path's guarantees are all
failure-mode guarantees -- "an interrupted run leaves no half pair", "a
re-run never re-downloads" -- and a guarantee about failure is only real if
the failure is exercised.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from src.data.cache_manifest import (
    CacheValidationError,
    REJECTION_REASONS,
    append_record,
    format_rejection_tally,
    make_record,
    manifest_path,
    npz_member_shape,
    read_manifest,
    validate_pair,
)

# The cache's shape contract, mirrored from the SEN2NAIPv2 measurements.
BANDS = ("B04", "B03", "B02", "B08")
SCALE = 4
LR_HW = 130
DN_RANGE = (50.0, 40000.0)


def _pair(lr_hw=LR_HW, bands=len(BANDS), scale=SCALE, dtype=np.uint16, fill=3000):
    """A synthetic (lr, hr) pair in the stored domain: raw uint16 DN, (C, H, W)."""
    lr = np.full((bands, lr_hw, lr_hw), fill, dtype=dtype)
    hr = np.full((bands, lr_hw * scale, lr_hw * scale), fill, dtype=dtype)
    return lr, hr


def _validate(lr, hr, **kwargs):
    defaults = dict(
        scale=SCALE,
        expected_bands=len(BANDS),
        expected_dtype="uint16",
        nodata_value=65535,
        dn_plausible_range=DN_RANGE,
        taco_id="test_pair",
    )
    defaults.update(kwargs)
    return validate_pair(lr, hr, **defaults)


# -- validate_pair ---------------------------------------------------------


def test_valid_pair_passes_and_reports_its_statistics():
    lr, hr = _pair()
    stats = _validate(lr, hr)
    assert stats["lr_shape"] == [4, 130, 130]
    assert stats["hr_shape"] == [4, 520, 520]
    assert stats["dtype"] == "uint16"
    assert stats["lr_dn_p999"] == pytest.approx(3000.0)
    assert stats["nodata_fraction"] == 0.0


def test_hr_that_is_not_exactly_four_times_lr_is_rejected():
    """Not 'about 4x'. A pair off by one pixel is no longer co-registered."""
    lr, hr = _pair()
    with pytest.raises(CacheValidationError) as excinfo:
        _validate(lr, hr[:, :-1, :])
    assert excinfo.value.reason == "scale_mismatch"


def test_a_pair_at_the_wrong_scale_entirely_is_rejected():
    lr = np.full((4, 130, 130), 3000, dtype=np.uint16)
    hr = np.full((4, 260, 260), 3000, dtype=np.uint16)  # 2x, not 4x
    with pytest.raises(CacheValidationError) as excinfo:
        _validate(lr, hr)
    assert excinfo.value.reason == "scale_mismatch"


def test_band_count_mismatch_is_rejected():
    lr, hr = _pair()
    with pytest.raises(CacheValidationError) as excinfo:
        _validate(lr[:3], hr)
    assert excinfo.value.reason == "band_count_mismatch"


def test_float_arrays_are_rejected_because_the_cache_stores_digital_numbers():
    """A float array means the divisor was already applied. Dividing twice is
    silent and catastrophic -- reflectance would come out 10000x too small."""
    lr, hr = _pair(dtype=np.float32, fill=0.3)
    with pytest.raises(CacheValidationError) as excinfo:
        _validate(lr, hr)
    assert excinfo.value.reason == "dtype_mismatch"


def test_all_zero_tile_is_rejected():
    lr, hr = _pair(fill=0)
    with pytest.raises(CacheValidationError) as excinfo:
        _validate(lr, hr)
    assert excinfo.value.reason == "all_zero"


def test_entirely_nodata_tile_is_rejected():
    lr, hr = _pair(fill=65535)
    with pytest.raises(CacheValidationError) as excinfo:
        _validate(lr, hr)
    assert excinfo.value.reason == "all_nan"


def test_all_nan_float_tile_is_rejected_before_the_dtype_check_would_matter():
    lr, hr = _pair(dtype=np.float32, fill=np.nan)
    with pytest.raises(CacheValidationError) as excinfo:
        _validate(lr, hr, expected_dtype="float32")
    assert excinfo.value.reason == "all_nan"


def test_a_pair_under_a_different_quantification_value_is_rejected():
    """The reflectance-scaling check. A pair delivered at /3000 rather than
    /10000 sits an order of magnitude outside the band the cache occupies."""
    lr, hr = _pair(fill=60000)
    with pytest.raises(CacheValidationError) as excinfo:
        _validate(lr, hr, nodata_value=None)
    assert excinfo.value.reason == "dn_range_implausible"


def test_a_bright_target_above_reflectance_one_is_NOT_rejected():
    """Snow, cloud and specular water legitimately exceed reflectance 1.0. The
    DN check is a scaling check, and must never act as a brightness clip."""
    lr, hr = _pair(fill=15000)  # reflectance 1.5
    stats = _validate(lr, hr)
    assert stats["lr_dn_p999"] == pytest.approx(15000.0)


def test_nodata_pixels_are_excluded_from_the_statistics():
    lr, hr = _pair()
    lr[:, :13, :] = 65535  # 10% of rows dead in every band
    stats = _validate(lr, hr)
    assert stats["nodata_fraction"] == pytest.approx(0.1)
    # The p99.9 is over VALID pixels only, so nodata does not drag it to 65535.
    assert stats["lr_dn_p999"] == pytest.approx(3000.0)


def test_a_pixel_with_one_dead_band_counts_as_nodata():
    """ANY-band, not ALL-band: one dead band makes the pixel unusable, and
    averaging over channels would report a quarter of the true loss."""
    lr, hr = _pair()
    lr[0, :13, :] = 65535  # one band only
    stats = _validate(lr, hr)
    assert stats["nodata_fraction"] == pytest.approx(0.1)


def test_degenerate_shapes_are_rejected():
    lr = np.zeros((4, 0, 130), dtype=np.uint16)
    hr = np.zeros((4, 0, 520), dtype=np.uint16)
    with pytest.raises(CacheValidationError) as excinfo:
        _validate(lr, hr)
    assert excinfo.value.reason == "shape_degenerate"


def test_two_dimensional_input_is_rejected_rather_than_broadcast():
    lr = np.full((130, 130), 3000, dtype=np.uint16)
    hr = np.full((520, 520), 3000, dtype=np.uint16)
    with pytest.raises(CacheValidationError) as excinfo:
        _validate(lr, hr)
    assert excinfo.value.reason == "shape_degenerate"


def test_every_reason_raised_is_a_declared_reason():
    """The tally is keyed on REJECTION_REASONS, so an unlisted reason would be
    counted under nothing."""
    with pytest.raises(ValueError, match="Unknown rejection reason"):
        CacheValidationError("something_new", "message")


# -- the manifest ----------------------------------------------------------


def test_missing_manifest_reads_as_an_empty_cache():
    assert read_manifest("/no/such/manifest.jsonl") == {}


def test_records_round_trip(tmp_path):
    path = manifest_path(tmp_path)
    record = make_record(
        taco_id="NA_001",
        lr_path="NA_001.npz",
        hr_path="NA_001.npz",
        lr_shape=(4, 130, 130),
        hr_shape=(4, 520, 520),
        bands=BANDS,
        dtype="uint16",
        nodata_value=65535,
        reflectance_scale=10000.0,
        validated="full",
    )
    append_record(path, record)
    back = read_manifest(path)
    assert list(back) == ["NA_001"]
    assert back["NA_001"]["lr_shape"] == [4, 130, 130]
    assert back["NA_001"]["bands"] == list(BANDS)
    assert back["NA_001"]["reflectance_scale"] == 10000.0
    assert back["NA_001"]["cached_at"].endswith("Z")


def test_appending_never_rewrites_earlier_lines(tmp_path):
    path = manifest_path(tmp_path)
    for i in range(3):
        append_record(path, make_record(
            f"id_{i}", f"id_{i}.npz", f"id_{i}.npz",
            (4, 130, 130), (4, 520, 520), BANDS, "uint16", 65535, 10000.0,
        ))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    assert list(read_manifest(path)) == ["id_0", "id_1", "id_2"]


def test_a_repeated_id_keeps_the_later_record(tmp_path):
    """Re-validating a pair supersedes its earlier line without a file rewrite."""
    path = manifest_path(tmp_path)
    append_record(path, make_record(
        "dup", "dup.npz", "dup.npz", (4, 130, 130), (4, 520, 520),
        BANDS, "uint16", 65535, 10000.0, validated="structural",
    ))
    append_record(path, make_record(
        "dup", "dup.npz", "dup.npz", (4, 130, 130), (4, 520, 520),
        BANDS, "uint16", 65535, 10000.0, validated="full",
    ))
    back = read_manifest(path)
    assert len(back) == 1
    assert back["dup"]["validated"] == "full"


def test_a_torn_final_line_is_dropped_not_raised(tmp_path):
    """The one tolerated corruption: a process killed mid-append. The pair it
    described is simply re-validated on the next run."""
    path = manifest_path(tmp_path)
    append_record(path, make_record(
        "good", "good.npz", "good.npz", (4, 130, 130), (4, 520, 520),
        BANDS, "uint16", 65535, 10000.0,
    ))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"taco_id": "torn", "lr_sh')
    back = read_manifest(path)
    assert list(back) == ["good"]


def test_a_corrupt_line_in_the_middle_raises(tmp_path):
    """Not the interrupted-append signature. Something rewrote the file, and
    skipping the row would quietly shrink the training set."""
    path = manifest_path(tmp_path)
    append_record(path, make_record(
        "a", "a.npz", "a.npz", (4, 130, 130), (4, 520, 520),
        BANDS, "uint16", 65535, 10000.0,
    ))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json at all\n")
    append_record(path, make_record(
        "c", "c.npz", "c.npz", (4, 130, 130), (4, 520, 520),
        BANDS, "uint16", 65535, 10000.0,
    ))
    with pytest.raises(ValueError, match="not valid JSON"):
        read_manifest(path)


def test_a_record_without_a_taco_id_raises(tmp_path):
    path = manifest_path(tmp_path)
    path.write_text(json.dumps({"lr_path": "x.npz"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'taco_id'"):
        read_manifest(path)


def test_blank_lines_are_ignored(tmp_path):
    path = manifest_path(tmp_path)
    append_record(path, make_record(
        "a", "a.npz", "a.npz", (4, 130, 130), (4, 520, 520),
        BANDS, "uint16", 65535, 10000.0,
    ))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")
    assert list(read_manifest(path)) == ["a"]


def test_manifest_paths_are_relative_so_the_cache_stays_portable(tmp_path):
    """A cache built at D:\\...\\outputs\\cache must work when mounted at
    /kaggle/input/<name>. An absolute path baked into the manifest would not."""
    record = make_record(
        "id", "id.npz", "id.npz", (4, 130, 130), (4, 520, 520),
        BANDS, "uint16", 65535, 10000.0,
    )
    assert "/" not in record["lr_path"] and "\\" not in record["lr_path"]


# -- npz_member_shape ------------------------------------------------------


def test_member_shape_matches_a_full_load(tmp_path):
    path = tmp_path / "pair.npz"
    lr, hr = _pair(lr_hw=8)
    with path.open("wb") as handle:
        np.savez_compressed(handle, lr=lr, hr=hr)
    assert npz_member_shape(path, "lr") == ((4, 8, 8), "uint16")
    assert npz_member_shape(path, "hr") == ((4, 32, 32), "uint16")


def test_member_shape_on_a_missing_member_raises_keyerror(tmp_path):
    path = tmp_path / "pair.npz"
    with path.open("wb") as handle:
        np.savez_compressed(handle, lr=np.zeros((4, 8, 8), np.uint16))
    with pytest.raises(KeyError, match="hr.npy"):
        npz_member_shape(path, "hr")


def test_member_shape_on_a_truncated_archive_raises(tmp_path):
    """The interrupted-download signature. It must raise so the caller can
    decline to adopt the pair, not return a plausible-looking shape."""
    path = tmp_path / "pair.npz"
    with path.open("wb") as handle:
        np.savez_compressed(handle, lr=np.zeros((4, 8, 8), np.uint16))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises((OSError, ValueError, EOFError, zipfile.BadZipFile)):
        npz_member_shape(path, "lr")


# -- the tally -------------------------------------------------------------


def test_the_tally_lists_reasons_that_did_not_fire():
    """A zero is evidence the check ran and passed, so it is printed."""
    lines = format_rejection_tally({"scale_mismatch": 2})
    counts = {
        parts[0]: int(parts[1])
        for parts in (line.split() for line in lines)
        if len(parts) == 2
    }
    assert set(counts) == set(REJECTION_REASONS)
    assert counts["scale_mismatch"] == 2
    assert counts["all_zero"] == 0


def test_the_tally_surfaces_an_unlisted_reason_rather_than_dropping_it():
    lines = format_rejection_tally({"invented": 1})
    assert any("UNLISTED REASON" in line for line in lines)
