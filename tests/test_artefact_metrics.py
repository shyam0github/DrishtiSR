"""Tests for the two Day 2 measurement entry points.

These exist because both scripts were written to stop a specific failure from
recurring -- a headline number recorded in a report with no way to re-derive it
-- and a measurement script nobody can run is that failure wearing a different
hat. Everything here runs on CPU in seconds and needs no imagery: rasters are
fabricated in memory or written to ``tmp_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.artefact_metrics as am  # noqa: E402
import scripts.boa_offset_audit as boa  # noqa: E402


# -- boa_offset_audit ------------------------------------------------------


def test_summarise_reports_every_requested_percentile():
    values = np.linspace(0.0, 1.0, 10001, dtype=np.float32)
    out = boa.summarise(values, [1, 5, 50, 95, 99])
    for key in ("min", "p1", "p5", "p50", "p95", "p99", "max", "mean", "span"):
        assert key in out
    assert out["p50"] == pytest.approx(0.5, abs=1e-3)
    assert out["span"] == pytest.approx(out["p95"] - out["p5"])


def test_summarise_refuses_an_empty_band_rather_than_reporting_zeros():
    """A fully-masked band is a data fault, not a band whose statistics are 0."""
    with pytest.raises(ValueError, match="No valid samples"):
        boa.summarise(np.array([], dtype=np.float32), [50])


def test_an_additive_offset_cannot_change_the_span():
    """The load-bearing reason the verdict is read off medians, not spans.

    If this ever failed, the audit's own warning that identical spans are
    expected -- and therefore are not evidence -- would be wrong.
    """
    rng = np.random.default_rng(0)
    values = rng.normal(0.2, 0.05, 50_000).astype(np.float32)
    plain = boa.summarise(values, [5, 50, 95])
    shifted = boa.summarise(values - np.float32(0.1), [5, 50, 95])
    assert shifted["span"] == pytest.approx(plain["span"], abs=1e-6)
    assert shifted["p50"] == pytest.approx(plain["p50"] - 0.1, abs=1e-6)


def test_pool_bands_excludes_nodata_from_every_band():
    """65535 / 10000 = 6.55 reflectance would dominate any statistic it reached.

    Masking is across channels: a pixel nodata in ANY band is dropped from all
    of them, so the bands stay pixel-aligned.
    """
    patch = np.full((2, 4, 4), 1200, dtype=np.uint16)
    patch[0, 0, 0] = boa.NODATA
    pooled = boa.pool_bands([patch], n_bands=2, scale=10000.0, offset=0.0)
    assert all(band.size == 15 for band in pooled)
    assert float(max(band.max() for band in pooled)) == pytest.approx(0.12)


def test_pool_bands_applies_the_offset_before_dividing():
    patch = np.full((1, 2, 2), 1500, dtype=np.uint16)
    (pooled,) = boa.pool_bands([patch], n_bands=1, scale=10000.0, offset=1000.0)
    assert pooled.min() == pytest.approx(0.05)


def test_pool_bands_refuses_a_band_axis_that_does_not_match_the_config():
    patch = np.zeros((3, 2, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="channels, expected 4"):
        boa.pool_bands([patch], n_bands=4, scale=10000.0, offset=0.0)


# -- artefact_metrics ------------------------------------------------------


def _write_raster(path: Path, chw: np.ndarray, pixel_size: float) -> Path:
    """Write ``(C, H, W)`` uint16 to a small GeoTIFF with a real UTM transform."""
    import rasterio
    from rasterio.transform import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    transform = Affine(pixel_size, 0.0, 700000.0, 0.0, -pixel_size, 3170000.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=chw.shape[1], width=chw.shape[2],
        count=chw.shape[0], dtype="uint16", crs="EPSG:32643", transform=transform,
    ) as dst:
        dst.write(chw)
    return path


def test_gradient_and_hf_are_invariant_to_an_additive_offset():
    rng = np.random.default_rng(1)
    image = rng.random((2, 32, 32), dtype=np.float32)
    assert am.gradient_magnitude(image - 0.1) == pytest.approx(
        am.gradient_magnitude(image), rel=1e-5
    )
    assert am.hf_energy(image - 0.1, 5) == pytest.approx(
        am.hf_energy(image, 5), rel=1e-5
    )


def test_ringing_is_NOT_invariant_to_a_mismatched_offset(tmp_path):
    """The bug this script was rewritten to prevent, pinned as a test.

    MEASURED 2026-09-08: scoring the offset-corrected Delhi raster against an
    uncorrected input crop reported 88% ringing for a raster whose real figure
    is 0.79%. Metrics 3 and 4 compare SR against the input in ABSOLUTE terms,
    so both sides must share one DN convention.
    """
    # A SMOOTH scene, because that is what makes the envelope locally tight and
    # the failure severe -- the same reason it showed up on real imagery. White
    # noise gives a 9x9 envelope spanning the whole value range, which absorbs a
    # 0.1 shift and understates the problem.
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 16, dtype=np.float32),
        np.linspace(0.0, 1.0, 16, dtype=np.float32),
        indexing="ij",
    )
    lr = (0.20 + 0.02 * np.sin(6.0 * xx) * np.cos(6.0 * yy))[None].astype(np.float32)
    sr = am.upsample(lr, 4, "bicubic")

    matched = am.envelope_violation_fraction(sr, lr, 9, 4)
    mismatched = am.envelope_violation_fraction(sr - 0.1, lr, 9, 4)
    assert matched < 0.05, f"a faithful bicubic should barely ring, got {matched}"
    assert mismatched > 0.9, (
        "a 0.1 reflectance convention mismatch must blow through a local "
        f"envelope, got {mismatched}"
    )


def test_a_bicubic_upsample_scores_close_to_one_against_itself():
    """Sanity floor: the baseline compared to itself is 1.0 by construction."""
    rng = np.random.default_rng(3)
    lr = rng.random((2, 24, 24), dtype=np.float32)
    bicubic = am.upsample(lr, 4, "bicubic")
    assert am.gradient_magnitude(bicubic) / am.gradient_magnitude(bicubic) == 1.0
    assert am.hf_energy(bicubic, 5) / am.hf_energy(bicubic, 5) == 1.0


def test_the_flat_mask_finds_the_flat_half_and_lands_on_the_hr_grid():
    """Half flat, half textured: everything selected must be in the flat half.

    The threshold is ``variance <= quantile(variance, q)``, so exact ties are
    all kept. A perfectly flat region has variance 0 for every pixel, and 0 is
    also the 25th percentile here -- so the mask comes out LARGER than the
    nominal quartile (43.75% of this fixture, not 25%). That is correct and is
    pinned rather than tuned away: on real imagery variance is continuous and
    the fraction lands on the quantile, but a synthetic flat region must not
    make the mask arbitrary.
    """
    lr = np.zeros((1, 32, 32), dtype=np.float32)
    rng = np.random.default_rng(4)
    lr[:, :, 16:] = rng.random((1, 32, 16), dtype=np.float32)

    mask = am.flat_mask(lr, window=5, quantile=0.25, scale=4)
    assert mask.shape == (128, 128)
    # Every selected pixel is in the flat (left) half of the HR grid.
    assert not mask[:, 64:].any()
    assert mask.sum() >= 0.25 * mask.size


def test_the_flat_mask_selects_the_requested_fraction_on_continuous_variance():
    """With no ties, the lowest quartile is the lowest quartile."""
    rng = np.random.default_rng(6)
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)[None, None, :]
    lr = (rng.random((1, 64, 64), dtype=np.float32) * ramp).astype(np.float32)

    mask = am.flat_mask(lr, window=5, quantile=0.25, scale=4)
    assert mask.sum() == pytest.approx(0.25 * mask.size, rel=0.05)


def test_hf_energy_refuses_a_mask_that_selects_nothing():
    image = np.zeros((1, 8, 8), dtype=np.float32)
    empty = np.zeros((8, 8), dtype=bool)
    with pytest.raises(ValueError, match="selected no pixels"):
        am.hf_energy(image, 3, empty)


def test_spectral_round_trip_is_zero_for_an_exact_area_average(tmp_path):
    """A nearest-upsampled raster degrades back to its input exactly."""
    rng = np.random.default_rng(5)
    lr = rng.random((4, 8, 8), dtype=np.float32)
    sr = am.upsample(lr, 4, "nearest")
    out = am.spectral_round_trip(sr, lr, 4)
    assert out["mae_reflectance"] == pytest.approx(0.0, abs=1e-6)
    assert all(abs(v) < 1e-6 for v in out["per_band_signed_mean"])


def test_spectral_round_trip_reports_the_sign_of_a_per_band_bias():
    """The sign is the finding: a consistent bias is a colour shift."""
    lr = np.full((2, 8, 8), 0.2, dtype=np.float32)
    sr = am.upsample(lr, 4, "nearest")
    sr[0] += 0.01
    sr[1] -= 0.02
    out = am.spectral_round_trip(sr, lr, 4)
    assert out["per_band_signed_mean"][0] == pytest.approx(+0.01, abs=1e-6)
    assert out["per_band_signed_mean"][1] == pytest.approx(-0.02, abs=1e-6)


def test_read_crop_applies_the_offset_and_refuses_a_window_off_the_edge(tmp_path):
    dn = np.full((4, 16, 16), 1500, dtype=np.uint16)
    path = _write_raster(tmp_path / "scene.tif", dn, 10.0)

    crop = am.read_crop(path, 0, 0, 8, 8, 10000.0, 1000.0)
    assert crop.shape == (4, 8, 8)
    assert float(crop.mean()) == pytest.approx(0.05)

    with pytest.raises(ValueError, match="does not fit inside"):
        am.read_crop(path, 12, 12, 8, 8, 10000.0, 0.0)


def test_read_crop_does_not_clip_a_negative_reflectance(tmp_path):
    """Offset correction legitimately puts dark water below zero. Never clamp it."""
    dn = np.full((1, 8, 8), 800, dtype=np.uint16)
    path = _write_raster(tmp_path / "dark.tif", dn, 10.0)
    crop = am.read_crop(path, 0, 0, 8, 8, 10000.0, 1000.0)
    assert float(crop.min()) == pytest.approx(-0.02)
