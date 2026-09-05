"""The dataset contract, enforced against every registered implementation.

These tests are the reason a dataset swap is a config edit. They run the same
assertions over every dataset that can be built without network or mounted data,
plus a deliberately minimal in-memory implementation written here. If a fallback
dataset is dropped in under time pressure, these tests are what say whether it
actually satisfies the interface training code assumes.

Datasets that need network (sen2naipv2) or a Kaggle mount (worldstrat) are
checked structurally -- registered, importable, correct base class -- without
being instantiated.
"""

from typing import Any, Dict

import numpy as np
import pytest
import torch

from src.data.base import SRPairDataset, SampleValidationError
from src.data.registry import (
    available_datasets,
    get_dataset,
    get_dataset_class,
    register_dataset,
)
from src.utils.config import load_config


@register_dataset("fake_tiny")
class FakeTinyDataset(SRPairDataset):
    """A minimal, deliberately different implementation of the interface.

    Uses its own patch size (8 -> 32) rather than the config's, so the contract
    tests cannot pass merely because every dataset shares one shape.
    """

    SOURCE_NAME = "fake_tiny"
    LR_SIZE = 8

    def __init__(self, cfg: Any, validate: bool = True) -> None:
        self.BANDS = tuple(cfg["dataset"]["bands"])
        super().__init__(cfg, validate=validate)

    def __len__(self) -> int:
        return 3

    def load_sample(self, idx: int) -> Dict[str, Any]:
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        rng = np.random.default_rng(idx)
        c = len(self.BANDS)
        hr_size = self.LR_SIZE * self.scale
        hr = rng.uniform(0.02, 0.35, (c, hr_size, hr_size)).astype(np.float32)
        lr = (
            hr.reshape(c, self.LR_SIZE, self.scale, self.LR_SIZE, self.scale)
            .mean(axis=(2, 4))
            .astype(np.float32)
        )
        return {
            "lr": torch.from_numpy(lr),
            "hr": torch.from_numpy(hr),
            "meta": {
                "sample_id": f"fake_{idx}",
                "source_dataset": self.SOURCE_NAME,
                "crs": "EPSG:4326",
                "geotransform": None,
            },
        }


# Datasets that can be constructed with no network and no mounted data.
CONSTRUCTIBLE = ["synthetic_stub", "fake_tiny"]
# Datasets that need external resources; checked structurally only.
EXTERNAL = ["sen2naipv2", "worldstrat"]


@pytest.fixture(scope="module")
def cfg():
    """The real base config in smoke mode -- not a hand-built stub."""
    return load_config("configs/base.yaml", smoke=True)


@pytest.fixture(params=CONSTRUCTIBLE)
def dataset(request, cfg):
    return get_dataset(cfg, name=request.param)


# -- the contract ---------------------------------------------------------


def test_is_srpairdataset(dataset):
    assert isinstance(dataset, SRPairDataset)


def test_has_bands_and_source(dataset):
    assert len(dataset.BANDS) > 0
    assert dataset.SOURCE_NAME


def test_len_is_positive(dataset):
    assert len(dataset) > 0


def test_getitem_returns_required_keys(dataset):
    sample = dataset[0]
    assert set(sample) >= {"lr", "hr", "meta"}


def test_meta_carries_required_fields(dataset):
    meta = dataset[0]["meta"]
    assert meta["sample_id"]
    assert meta["source_dataset"] == dataset.SOURCE_NAME
    assert "crs" in meta
    assert "geotransform" in meta


def test_tensors_are_float32_chw(dataset):
    sample = dataset[0]
    for key in ("lr", "hr"):
        tensor = sample[key]
        assert torch.is_tensor(tensor), key
        assert tensor.dtype == torch.float32, key
        assert tensor.ndim == 3, key
        assert tensor.shape[0] == len(dataset.BANDS), key


def test_shape_ratio_equals_scale(dataset):
    sample = dataset[0]
    lr, hr = sample["lr"], sample["hr"]
    assert hr.shape[-2] == lr.shape[-2] * dataset.scale
    assert hr.shape[-1] == lr.shape[-1] * dataset.scale


def test_reflectance_in_valid_range(dataset):
    sample = dataset[0]
    for key in ("lr", "hr"):
        tensor = sample[key]
        assert float(tensor.min()) >= dataset.reflectance_valid_min
        assert float(tensor.max()) <= dataset.reflectance_valid_max


def test_no_nan_or_inf(dataset):
    sample = dataset[0]
    for key in ("lr", "hr"):
        assert bool(torch.isfinite(sample[key]).all()), key


def test_every_sample_satisfies_the_contract(dataset):
    for idx in range(len(dataset)):
        dataset.validate_sample(dataset[idx], idx=idx)


def test_deterministic_across_reads(dataset):
    torch.testing.assert_close(dataset[0]["lr"], dataset[0]["lr"])
    torch.testing.assert_close(dataset[0]["hr"], dataset[0]["hr"])


def test_out_of_range_index_raises(dataset):
    with pytest.raises((IndexError, KeyError)):
        dataset[len(dataset)]


def test_iteration_yields_every_sample(dataset):
    assert sum(1 for _ in dataset) == len(dataset)


# -- manifest -------------------------------------------------------------


def test_manifest_rows_cover_every_sample(dataset):
    rows = dataset.manifest_rows()
    assert len(rows) == len(dataset)
    assert all(r["validation_error"] == "" for r in rows)
    assert all(r["scale_ok"] for r in rows)


def test_manifest_reports_per_band_ranges(dataset):
    row = dataset.manifest_rows()[0]
    for band_idx, band in enumerate(dataset.BANDS):
        assert f"lr_b{band_idx}_{band}_min" in row
        assert f"hr_b{band_idx}_{band}_max" in row


def test_build_index_writes_csv(dataset, tmp_path):
    import csv

    out = dataset.build_index(tmp_path / "m.csv")
    assert out.is_file()
    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(dataset)
    assert rows[0]["sample_id"]


# -- registry -------------------------------------------------------------


@pytest.mark.parametrize("name", EXTERNAL)
def test_external_datasets_are_registered_and_conform(name):
    """They must satisfy the interface structurally without being constructed."""
    cls = get_dataset_class(name)
    assert issubclass(cls, SRPairDataset)
    assert cls.SOURCE_NAME
    # load_sample must be concrete, or the class is still abstract.
    assert getattr(cls.load_sample, "__isabstractmethod__", False) is False


def test_all_registered_names_resolve():
    for name in available_datasets():
        assert issubclass(get_dataset_class(name), SRPairDataset)


def test_unknown_dataset_raises_with_options():
    with pytest.raises(KeyError) as excinfo:
        get_dataset_class("no_such_dataset")
    assert "synthetic_stub" in str(excinfo.value)


def test_registering_non_dataset_raises():
    with pytest.raises(TypeError):
        register_dataset("bad")(object)


def test_get_dataset_uses_config_name(cfg):
    # smoke mode sets dataset.name to the stub.
    assert get_dataset(cfg).SOURCE_NAME == "synthetic_stub"


# -- validation actually rejects bad data ---------------------------------


class _Broken(FakeTinyDataset):
    SOURCE_NAME = "broken"
    BREAKAGE = ""

    def load_sample(self, idx):
        sample = super().load_sample(idx)
        if self.BREAKAGE == "nan":
            sample["lr"][0, 0, 0] = float("nan")
        elif self.BREAKAGE == "ratio":
            sample["hr"] = sample["hr"][:, :-4, :-4]
        elif self.BREAKAGE == "range":
            sample["lr"][0, 0, 0] = 6.5535  # 65535 / 10000: unmasked nodata
        elif self.BREAKAGE == "negative":
            sample["lr"][0, 0, 0] = -0.1
        elif self.BREAKAGE == "dtype":
            sample["lr"] = sample["lr"].double()
        elif self.BREAKAGE == "hwc":
            sample["lr"] = sample["lr"].permute(1, 2, 0).contiguous()
        elif self.BREAKAGE == "meta":
            sample["meta"].pop("crs")
        elif self.BREAKAGE == "missing_key":
            sample.pop("hr")
        return sample


@pytest.mark.parametrize(
    "breakage",
    ["nan", "ratio", "range", "negative", "dtype", "hwc", "meta", "missing_key"],
)
def test_contract_violations_raise(cfg, breakage):
    cls = type(f"Broken_{breakage}", (_Broken,), {"BREAKAGE": breakage})
    dataset = cls(cfg)
    with pytest.raises(SampleValidationError):
        dataset[0]


def test_unmasked_nodata_is_caught_specifically(cfg):
    """65535 nodata divided by 10000 must fail loudly, not train as reflectance 6.55."""
    cls = type("Broken_nodata", (_Broken,), {"BREAKAGE": "range"})
    with pytest.raises(SampleValidationError) as excinfo:
        cls(cfg)[0]
    message = str(excinfo.value)
    assert "nodata" in message.lower()
    # The error must steer away from the tempting wrong fix.
    assert "clip" in message.lower()


def test_manifest_records_failures_instead_of_raising(cfg):
    """build_index must survive bad samples -- that is what it exists to show."""
    cls = type("Broken_manifest", (_Broken,), {"BREAKAGE": "nan"})
    rows = cls(cfg).manifest_rows()
    assert len(rows) == 3
    assert all(r["validation_error"] for r in rows)
    assert any(r["n_nan"] for r in rows)


def test_validate_false_skips_validation(cfg):
    cls = type("Broken_off", (_Broken,), {"BREAKAGE": "nan"})
    sample = cls(cfg, validate=False)[0]
    assert torch.isnan(sample["lr"]).any()


# -- reflectance conversion -----------------------------------------------


def test_to_reflectance_masks_nodata_before_dividing(dataset):
    raw = np.full((len(dataset.BANDS), 4, 4), 5000, dtype=np.uint16)
    raw[0, 0, 0] = 65535
    ref, mask = dataset.to_reflectance(raw, nodata_value=65535, nodata_fill=0.0)

    assert ref.dtype == np.float32
    assert mask[0, 0, 0]
    assert ref[0, 0, 0] == 0.0, "nodata must not become 6.5535 reflectance"
    assert ref[0, 0, 1] == pytest.approx(0.5)


def test_to_reflectance_does_not_clip_above_one(dataset):
    """Bright specular targets legitimately exceed 1.0 and must survive."""
    raw = np.full((len(dataset.BANDS), 2, 2), 11000, dtype=np.uint16)
    ref, _ = dataset.to_reflectance(raw, nodata_value=65535)
    assert ref.max() == pytest.approx(1.1)


def test_to_reflectance_rejects_hwc(dataset):
    with pytest.raises(ValueError):
        dataset.to_reflectance(np.zeros((4, 4, 3), dtype=np.uint16)[0])
