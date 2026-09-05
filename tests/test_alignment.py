"""Tests for the LR/HR alignment audit.

The headline test is :func:`test_estimator_recovers_an_injected_shift`: a known
translation is injected into the HR image and the estimator must recover it. Until
that holds, the PASS/WARN/FAIL verdict this module prints on real data is an
opinion rather than a measurement, so it is asserted here rather than eyeballed
in a smoke run.
"""

import numpy as np
import pytest

from src.eval.alignment import (
    FAIL,
    PASS,
    WARN,
    audit_alignment,
    bicubic_upsample,
    describe_selection,
    format_selection_block,
    grid_cell_of,
    read_manifest_records,
    select_audit_indices,
    block_mean_downsample,
    estimate_pair_shift,
    inject_shift,
    pair_correlations,
    reference_band_index,
    reflectance_range_overlap,
    self_test_result,
    verdict,
    write_report,
)
from src.utils.config import load_config

SCALE = 4
BANDS = 4
HR_SIZE = 256
TOLERANCE_PX = 0.35


def _scene(seed=0, size=HR_SIZE, bands=BANDS):
    """A smooth, textured HR scene and its exact block-mean LR counterpart.

    Smoothed noise rather than white noise: phase correlation needs structure
    that survives a 4x block mean, which is exactly what real land cover has and
    white noise does not.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    hr = np.empty((bands, size, size), dtype=np.float32)
    for band in range(bands):
        field = gaussian_filter(rng.random((size, size)), sigma=3.0)
        field = (field - field.min()) / (field.max() - field.min() + 1e-9)
        hr[band] = (0.05 + 0.35 * field).astype(np.float32)
    lr = (
        hr.reshape(bands, size // SCALE, SCALE, size // SCALE, SCALE)
        .mean(axis=(2, 4))
        .astype(np.float32)
    )
    return lr, hr


def _pairs(count=4, inject=None, seed=0):
    out = []
    for i in range(count):
        lr, hr = _scene(seed=seed + i)
        if inject is not None:
            hr = inject_shift(hr, inject[0], inject[1])
        out.append(
            {
                "index": i,
                "sample_id": f"scene_{i}",
                "lr": lr,
                "hr": hr,
                "injected_shift": list(inject) if inject is not None else None,
            }
        )
    return out


# -- THE test: does the estimator measure what it claims to measure? -------


@pytest.mark.parametrize(
    "dy,dx", [(2.0, 2.0), (2.0, 0.0), (0.0, 2.0), (-2.0, 3.0), (1.5, -0.5)]
)
def test_estimator_recovers_an_injected_shift(dy, dx):
    """Inject a known translation into HR; the estimate must come back at it."""
    errors = []
    for seed in range(4):
        lr, hr = _scene(seed=seed)
        estimate = estimate_pair_shift(
            lr,
            inject_shift(hr, dy, dx),
            scale=SCALE,
            band_index=0,
            upsample_factor=10,
            border_crop_px=8,
        )
        errors.append((estimate["dy"] - dy, estimate["dx"] - dx))
    errors = np.array(errors)
    assert np.abs(errors).max() <= TOLERANCE_PX, (
        f"injected ({dy}, {dx}) but the estimator was off by up to "
        f"{np.abs(errors).max():.2f} HR px"
    )


def test_estimator_reports_no_shift_on_an_aligned_pair():
    """The other half of the proof: it must not invent a shift."""
    for seed in range(4):
        lr, hr = _scene(seed=seed)
        estimate = estimate_pair_shift(
            lr, hr, scale=SCALE, band_index=0, upsample_factor=10, border_crop_px=8
        )
        assert estimate["shift_magnitude"] <= TOLERANCE_PX


def test_shift_sign_convention_is_documented_correctly():
    """Positive dy must mean "the upsampled LR moves down to meet the HR"."""
    lr, hr = _scene(seed=11)
    estimate = estimate_pair_shift(
        lr, inject_shift(hr, 3.0, 0.0), scale=SCALE, band_index=0,
        upsample_factor=10, border_crop_px=8,
    )
    assert estimate["dy"] > 2.5
    assert abs(estimate["dx"]) < TOLERANCE_PX


def test_sub_pixel_resolution_follows_the_upsample_factor():
    lr, hr = _scene(seed=3)
    shifted = inject_shift(hr, 0.4, 0.0)
    coarse = estimate_pair_shift(
        lr, shifted, scale=SCALE, band_index=0, upsample_factor=1, border_crop_px=8
    )
    fine = estimate_pair_shift(
        lr, shifted, scale=SCALE, band_index=0, upsample_factor=10, border_crop_px=8
    )
    assert coarse["dy"] == pytest.approx(round(coarse["dy"]))  # whole pixels only
    assert abs(fine["dy"] - 0.4) < abs(coarse["dy"] - 0.4) + 1e-9


def test_degenerate_constant_image_raises_rather_than_reporting_zero():
    """A flat tile has no measurable shift; reporting 0.0 would bias the median."""
    lr = np.full((BANDS, 64, 64), 0.2, dtype=np.float32)
    hr = np.full((BANDS, 256, 256), 0.2, dtype=np.float32)
    with pytest.raises(ValueError, match="constant"):
        estimate_pair_shift(lr, hr, scale=SCALE, band_index=0, upsample_factor=10)


# -- resampling helpers ----------------------------------------------------


def test_bicubic_upsample_shape_and_dtype():
    lr, _ = _scene()
    up = bicubic_upsample(lr, SCALE)
    assert up.shape == (BANDS, HR_SIZE, HR_SIZE)
    assert up.dtype == np.float32


def test_bicubic_upsample_does_not_clip_bright_reflectance():
    """Surface reflectance above 1.0 is real over cloud and snow; keep it."""
    lr = np.full((1, 16, 16), 0.2, dtype=np.float32)
    lr[0, 8, 8] = 1.6
    up = bicubic_upsample(lr, SCALE)
    assert up.max() > 1.2


def test_block_mean_downsample_inverts_the_synthetic_degradation():
    lr, hr = _scene()
    assert np.allclose(block_mean_downsample(hr, SCALE), lr, atol=1e-6)


def test_block_mean_downsample_rejects_an_indivisible_shape():
    with pytest.raises(ValueError, match="divisible"):
        block_mean_downsample(np.zeros((1, 10, 10), np.float32), SCALE)


def test_correlations_are_high_when_aligned_and_drop_at_hr_when_shifted():
    lr, hr = _scene(seed=5)
    aligned = pair_correlations(lr, hr, SCALE, 0)
    shifted = pair_correlations(lr, inject_shift(hr, 4.0, 4.0), SCALE, 0)
    assert aligned["correlation_hr_grid"] > 0.9
    assert shifted["correlation_hr_grid"] < aligned["correlation_hr_grid"]


def test_reference_band_index_matches_the_config():
    cfg = load_config("configs/base.yaml")
    assert reference_band_index(cfg) == list(cfg.dataset.bands).index(
        cfg.alignment.reference_band
    )


def test_unknown_reference_band_is_rejected():
    cfg = load_config("configs/base.yaml", overrides=["alignment.reference_band=B12"])
    with pytest.raises(KeyError, match="reference_band"):
        reference_band_index(cfg)


# -- radiometry ------------------------------------------------------------


def test_identical_sources_have_full_range_overlap():
    lr, hr = _scene()
    pairs = [{"lr": lr, "hr": block_mean_downsample(hr, 1)}]
    result = reflectance_range_overlap(
        pairs, [f"b{i}" for i in range(BANDS)], (1.0, 99.0), 0.5
    )
    assert result["ok"]
    assert result["min_overlap"] > 0.5


def test_disjoint_sources_do_not_overlap():
    lr, hr = _scene()
    pairs = [{"lr": lr, "hr": hr + 10.0}]
    result = reflectance_range_overlap(
        pairs, [f"b{i}" for i in range(BANDS)], (1.0, 99.0), 0.5
    )
    assert not result["ok"]
    assert result["min_overlap"] == pytest.approx(0.0, abs=1e-6)


# -- the verdict -----------------------------------------------------------


@pytest.mark.parametrize(
    "median,expected", [(0.0, PASS), (0.99, PASS), (1.0, WARN), (2.0, WARN), (2.01, FAIL), (9.0, FAIL)]
)
def test_verdict_thresholds(median, expected):
    assert verdict(median, True, 1.0, 2.0)["level"] == expected


def test_a_good_shift_with_bad_radiometry_is_downgraded_to_warn():
    result = verdict(0.2, False, 1.0, 2.0)
    assert result["level"] == WARN
    assert any("radiometrically" in reason for reason in result["reasons"])


def test_a_failing_shift_is_not_softened_by_good_radiometry():
    result = verdict(3.0, True, 1.0, 2.0)
    assert result["level"] == FAIL


def test_verdict_refuses_a_non_finite_median():
    with pytest.raises(ValueError, match="not finite"):
        verdict(float("nan"), True, 1.0, 2.0)


def test_verdict_requires_ordered_thresholds():
    with pytest.raises(ValueError, match="must be below"):
        verdict(0.5, True, 2.0, 1.0)


# -- the audit as a whole --------------------------------------------------


def _audit_cfg():
    return load_config(
        "configs/base.yaml",
        overrides=["dataset.name=synthetic_stub", "alignment.border_crop_px=8"],
    )


def test_audit_passes_on_aligned_pairs():
    report = audit_alignment(_pairs(count=3), _audit_cfg(), label="control")
    assert report["verdict"]["level"] == PASS
    assert report["shift"]["median_magnitude"] <= TOLERANCE_PX
    assert report["num_pairs"] == 3


def test_audit_fails_on_pairs_shifted_by_two_pixels():
    """The acceptance criterion, as an assertion: 2 px in must read as FAIL."""
    report = audit_alignment(_pairs(count=3, inject=(2.0, 2.0)), _audit_cfg())
    assert report["shift"]["median_dy"] == pytest.approx(2.0, abs=TOLERANCE_PX)
    assert report["shift"]["median_dx"] == pytest.approx(2.0, abs=TOLERANCE_PX)
    assert report["verdict"]["level"] == FAIL


def test_audit_excludes_degenerate_pairs_instead_of_scoring_them_zero():
    pairs = _pairs(count=2, inject=(2.0, 2.0))
    pairs.append(
        {
            "index": 99,
            "sample_id": "flat",
            "lr": np.full((BANDS, 64, 64), 0.2, np.float32),
            "hr": np.full((BANDS, HR_SIZE, HR_SIZE), 0.2, np.float32),
            "injected_shift": None,
        }
    )
    report = audit_alignment(pairs, _audit_cfg())
    assert report["num_pairs"] == 2
    assert len(report["degenerate"]) == 1
    assert report["degenerate"][0]["sample_id"] == "flat"
    # The flat pair must not have dragged the median toward zero.
    assert report["shift"]["median_magnitude"] > 2.0


def test_audit_rejects_an_empty_pair_list():
    with pytest.raises(ValueError, match="no pairs"):
        audit_alignment([], _audit_cfg())


def test_audit_raises_when_every_pair_is_degenerate():
    flat = [
        {
            "index": i,
            "sample_id": f"flat_{i}",
            "lr": np.full((BANDS, 64, 64), 0.2, np.float32),
            "hr": np.full((BANDS, HR_SIZE, HR_SIZE), 0.2, np.float32),
            "injected_shift": None,
        }
        for i in range(2)
    ]
    with pytest.raises(ValueError, match="degenerate"):
        audit_alignment(flat, _audit_cfg())


def test_report_round_trips_through_json(tmp_path):
    import json

    report = audit_alignment(_pairs(count=2), _audit_cfg())
    path = write_report(report, tmp_path / "metrics" / "alignment_report.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["verdict"]["level"] == report["verdict"]["level"]
    assert len(loaded["pairs"]) == 2


# -- the self-test scoring -------------------------------------------------


def test_self_test_passes_when_the_shift_is_recovered():
    cfg = _audit_cfg()
    control = audit_alignment(_pairs(count=3), cfg, label="control")
    injected = audit_alignment(_pairs(count=3, inject=(2.0, 2.0)), cfg, label="injected")
    result = self_test_result(injected, control, (2.0, 2.0), TOLERANCE_PX)
    assert result["passed"]
    assert result["recovered"][0] == pytest.approx(2.0, abs=TOLERANCE_PX)


def test_self_test_fails_when_the_estimator_misses_the_shift():
    cfg = _audit_cfg()
    injected = audit_alignment(_pairs(count=3, inject=(2.0, 2.0)), cfg, label="injected")
    result = self_test_result(injected, None, (5.0, 5.0), TOLERANCE_PX)
    assert not result["passed"]
    assert result["notes"]


def test_self_test_fails_when_the_control_shows_a_phantom_shift():
    cfg = _audit_cfg()
    control = audit_alignment(_pairs(count=3, inject=(2.0, 2.0)), cfg, label="control")
    injected = audit_alignment(_pairs(count=3, inject=(2.0, 2.0)), cfg, label="injected")
    result = self_test_result(injected, control, (2.0, 2.0), TOLERANCE_PX)
    assert not result["passed"]
    assert any("control" in note for note in result["notes"])


# -- pair selection --------------------------------------------------------
#
# The audit's verdict is only as good as the pairs it saw. SEN2NAIPv2 records
# are ordered geographically, so "the first 50" is one grid cell under one
# sensor geometry. These tests pin the sampling behaviour that stops the audit
# quietly re-measuring the same corner of the archive.


class _FakeDataset:
    """Catalog-backed stand-in with SEN2NAIP-shaped ids and cache state.

    Args:
        cells: Grid-cell prefix per sample, e.g. ``["NA5120_E1183N0757", ...]``.
        cached: Indices that report as cached. None means all of them.
        crs: CRS string per sample, or None to omit the column.
    """

    def __init__(self, cells, cached=None, crs=None):
        self.catalog = [
            {
                "sample_id": f"{cell}__m_{3900000 + i}_nw_10_060_20220710",
                "crs": (crs[i] if crs is not None else "EPSG:32610"),
            }
            for i, cell in enumerate(cells)
        ]
        self._cached = set(range(len(cells))) if cached is None else set(cached)

    def __len__(self):
        return len(self.catalog)

    def is_cached(self, idx):
        return int(idx) in self._cached


def _cells(n_cells, per_cell):
    """``n_cells`` distinct grid cells, ``per_cell`` consecutive records each.

    Consecutive on purpose: this is the clustering that makes taking the first
    N a geographic error rather than merely an arbitrary one.
    """
    return [f"NA5120_E{1100 + c:04d}N0757" for c in range(n_cells) for _ in range(per_cell)]


def _write_manifest(path, dataset, failing=()):
    """Write a build_index-shaped manifest for ``dataset``.

    Args:
        path: Destination CSV.
        dataset: A :class:`_FakeDataset`.
        failing: Indices to mark with a validation_error.
    """
    import csv

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["index", "sample_id", "validation_error"]
        )
        writer.writeheader()
        for idx, entry in enumerate(dataset.catalog):
            writer.writerow(
                {
                    "index": idx,
                    "sample_id": entry["sample_id"],
                    "validation_error": (
                        "RuntimeError: 31.0% nodata" if idx in set(failing) else ""
                    ),
                }
            )
    return path


def test_selection_is_not_the_first_n_records():
    """The audited indices must not be ``range(num_pairs)``.

    This is the regression the whole change exists for: a sequential selection
    over a geographically ordered catalog audits one region and reports it as
    if it described the archive.
    """
    dataset = _FakeDataset(_cells(20, 10))
    indices = select_audit_indices(dataset, num_pairs=50, seed=42)

    assert indices != list(range(50))
    assert max(indices) > 50, "the sample never reached past the first 50 records"


def test_selection_spans_many_grid_cells():
    dataset = _FakeDataset(_cells(20, 10))
    indices = select_audit_indices(dataset, num_pairs=50, seed=42)
    selection = describe_selection(dataset, indices, seed=42)

    assert selection["primary_grouping"] == "grid_cell"
    # 50 draws from 200 records over 20 cells: a sequential draw would touch 5.
    assert selection["num_distinct_regions"] >= 15
    assert sum(selection["groupings"]["grid_cell"].values()) == 50


def test_selection_is_reproducible_from_the_seed():
    dataset = _FakeDataset(_cells(20, 10))
    first = select_audit_indices(dataset, num_pairs=30, seed=7)
    again = select_audit_indices(dataset, num_pairs=30, seed=7)
    other = select_audit_indices(dataset, num_pairs=30, seed=8)

    assert first == again
    assert first != other


def test_selection_is_without_replacement_and_sorted():
    dataset = _FakeDataset(_cells(10, 10))
    indices = select_audit_indices(dataset, num_pairs=40, seed=3)

    assert len(indices) == len(set(indices)) == 40
    assert indices == sorted(indices)


def test_selection_is_capped_at_what_is_cached():
    dataset = _FakeDataset(_cells(10, 10), cached=range(25))
    indices = select_audit_indices(dataset, num_pairs=40, seed=3, cached_only=True)

    assert len(indices) == 25
    assert max(indices) < 25


def test_uncached_samples_are_included_when_cached_only_is_false():
    dataset = _FakeDataset(_cells(10, 10), cached=range(25))
    indices = select_audit_indices(dataset, num_pairs=40, seed=3, cached_only=False)

    assert len(indices) == 40
    assert max(indices) >= 25


def test_manifest_filter_excludes_rows_that_failed_validation(tmp_path):
    dataset = _FakeDataset(_cells(10, 10))
    # The failures are the first 40 records, i.e. exactly the block a
    # first-N selection would have audited.
    failing = set(range(40))
    manifest = _write_manifest(tmp_path / "manifest.csv", dataset, failing=failing)

    indices = select_audit_indices(
        dataset, num_pairs=40, seed=5, manifest_path=manifest
    )

    assert indices, "nothing was selected"
    assert not (set(indices) & failing)


def test_missing_manifest_raises_rather_than_silently_widening_the_pool(tmp_path):
    dataset = _FakeDataset(_cells(4, 4))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        select_audit_indices(
            dataset, num_pairs=4, seed=0, manifest_path=tmp_path / "absent.csv"
        )


def test_manifest_with_every_row_failing_raises(tmp_path):
    dataset = _FakeDataset(_cells(4, 4))
    manifest = _write_manifest(
        tmp_path / "manifest.csv", dataset, failing=range(len(dataset))
    )
    with pytest.raises(RuntimeError, match="validation_error"):
        select_audit_indices(dataset, num_pairs=4, seed=0, manifest_path=manifest)


def test_manifest_without_the_expected_columns_raises(tmp_path):
    path = tmp_path / "not_a_manifest.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing column"):
        read_manifest_records(path)


def test_empty_pool_raises():
    dataset = _FakeDataset(_cells(4, 4), cached=[])
    with pytest.raises(RuntimeError, match="No samples available"):
        select_audit_indices(dataset, num_pairs=4, seed=0, cached_only=True)


def test_grid_cell_of_splits_on_the_double_underscore():
    assert (
        grid_cell_of("NA5120_E1183N0757__m_3912321_nw_10_060_20220710")
        == "NA5120_E1183N0757"
    )


def test_grid_cell_of_returns_none_when_the_id_has_no_cell():
    assert grid_cell_of("stub_0000") is None


def test_describe_selection_records_provenance(tmp_path):
    dataset = _FakeDataset(_cells(8, 5))
    manifest = _write_manifest(tmp_path / "manifest.csv", dataset)
    indices = select_audit_indices(
        dataset, num_pairs=12, seed=11, manifest_path=manifest
    )
    selection = describe_selection(
        dataset, indices, seed=11, manifest_path=manifest
    )

    assert selection["sampling"] == "random_without_replacement"
    assert selection["seed"] == 11
    assert selection["manifest_path"] == str(manifest)
    assert selection["num_catalog"] == 40
    assert selection["num_selected"] == 12
    assert selection["indices"] == indices
    assert len(selection["sample_ids"]) == 12
    assert selection["sample_ids"] == [
        dataset.catalog[i]["sample_id"] for i in indices
    ]


def test_describe_selection_reports_no_grouping_rather_than_inventing_one():
    """A dataset whose ids carry no cell must say so, not fake a region."""

    class _NoCells:
        catalog = [{"sample_id": f"stub_{i:04d}"} for i in range(8)]

        def __len__(self):
            return len(self.catalog)

    selection = describe_selection(_NoCells(), [0, 1, 2], seed=0)
    assert selection["primary_grouping"] is None
    assert selection["num_distinct_regions"] is None


def test_selection_block_lists_every_sample_id():
    dataset = _FakeDataset(_cells(8, 5))
    indices = select_audit_indices(dataset, num_pairs=10, seed=1)
    selection = describe_selection(dataset, indices, seed=1)
    block = format_selection_block(selection)

    for sample_id in selection["sample_ids"]:
        assert sample_id in block
    assert "distribution by grid_cell (primary)" in block
    assert "seed 1" in block


def test_selection_block_flags_an_unfiltered_pool():
    dataset = _FakeDataset(_cells(4, 4))
    selection = describe_selection(dataset, [0, 1], seed=0, manifest_path=None)
    assert "validation NOT enforced" in format_selection_block(selection)


def test_selection_survives_a_json_round_trip(tmp_path):
    """The selection has to reach the report -- that is where it gets quoted."""
    import json

    dataset = _FakeDataset(_cells(6, 4))
    indices = select_audit_indices(dataset, num_pairs=8, seed=2)
    selection = describe_selection(dataset, indices, seed=2)

    path = write_report({"selection": selection}, tmp_path / "report.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))["selection"]
    assert loaded["sample_ids"] == selection["sample_ids"]
    assert loaded["groupings"]["grid_cell"] == selection["groupings"]["grid_cell"]
