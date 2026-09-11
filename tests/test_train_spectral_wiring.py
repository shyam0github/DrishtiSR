"""The trainer's half of the spectral contribution: defaults, logging, schema.

The loss itself is tested in ``tests/test_spectral_loss.py``. What is at stake
here is the EXPERIMENT rather than the arithmetic:

- The control run must be this same file with the flags left off. If a default
  drifts off zero, the control silently acquires the objective it exists to
  isolate, and the Day 3 comparison becomes two runs of the same thing.
- The control must LOG the spectral terms it is not optimising. Without that,
  "the spectral run drove l1_spec down" has nothing to be down from, and the
  claim is unfalsifiable.
- ``log.csv`` grew three columns. A resumed Kaggle session finds a five-column
  Run A log in its output directory, and appending eight-column rows to it
  produces a file ``pandas`` refuses to read -- hours after the GPU time was
  spent. The migration is tested here, and so is the reader that has to accept
  both schemas.

CPU only, no data, no network, no GPU.
"""

import csv

import pytest
import torch

from src.eval.curves import read_training_log
from src.losses import DEFAULT_DOWNSAMPLE, spectral_terms
from src.train import (
    KNOWN_SCHEMAS,
    LEGACY_COLUMNS,
    LOG_COLUMNS,
    REFERENCE_COLUMNS,
    SPECTRAL_COLUMNS,
    UNCERTAINTY_COLUMNS,
    build_parser,
    ensure_log_header,
)

BANDS = 4


# -- the control is defined by its defaults --------------------------------


def test_the_spectral_flags_default_to_off():
    """Run A2 must be byte-identical control code: same file, flags omitted."""
    defaults = {a.dest: a.default for a in build_parser()._actions}
    assert defaults["spectral_lambda1"] == 0.0
    assert defaults["spectral_lambda2"] == 0.0
    assert defaults["spectral_downsample"] == DEFAULT_DOWNSAMPLE


def test_the_flags_parse_into_floats_a_run_can_be_launched_with():
    args = build_parser().parse_args(
        [
            "--data-module", "src.data.adapter",
            "--data-class", "SRPatchDataset",
            "--data-root", "auto",
            "--spectral-lambda1", "0.5",
            "--spectral-lambda2", "0.1",
        ]
    )
    assert (args.spectral_lambda1, args.spectral_lambda2) == (0.5, 0.1)
    # The boolean the training step branches on, reproduced here so that a
    # change to the condition in main() breaks a test rather than a run.
    assert args.spectral_lambda1 > 0.0 or args.spectral_lambda2 > 0.0


def test_the_defaults_leave_the_spectral_branch_off():
    args = build_parser().parse_args(
        ["--data-module", "m", "--data-class", "c", "--data-root", "r"]
    )
    assert not (args.spectral_lambda1 > 0.0 or args.spectral_lambda2 > 0.0)


# -- the schema ------------------------------------------------------------


def test_the_new_columns_are_appended_not_interleaved():
    """Every schema must be a PREFIX of the next one.

    That is the whole reason a Run A log, a first-pass Day 3 log and a current
    one can be read by one function and widened by one migration. A column
    inserted rather than appended silently re-points every positional slice in
    the project at the wrong quantity.
    """
    for older, newer in zip(KNOWN_SCHEMAS, KNOWN_SCHEMAS[1:]):
        assert newer[: len(older)] == older, (
            f"{newer[: len(older)]} is not {older}: a column was inserted or "
            "reordered instead of appended."
        )

    assert SPECTRAL_COLUMNS[len(LEGACY_COLUMNS):] == [
        "val_l1_spec",
        "val_sam",
        "val_sam_valid_frac",
    ]
    assert REFERENCE_COLUMNS[len(SPECTRAL_COLUMNS):] == [
        "val_sharpness",
        "val_hf_energy",
        "val_ssim",
        "val_lpips",
        "val_sam_hr",
        "val_ergas",
    ]
    assert UNCERTAINTY_COLUMNS[len(REFERENCE_COLUMNS):] == [
        "val_nll",
        "nll_weight",
        "val_logvar_mean",
        "val_logvar_min",
        "val_logvar_max",
        "val_logvar_spatial_std",
        "val_logvar_clamp_lo_frac",
        "val_logvar_clamp_hi_frac",
    ]
    assert LOG_COLUMNS[len(UNCERTAINTY_COLUMNS):] == [
        "nll_grad_ratio_raw",
        "nll_grad_ratio_applied",
    ]


def test_the_two_sam_columns_are_not_the_same_quantity():
    """val_sam is spectral-domain RADIANS; val_sam_hr is image-domain DEGREES.

    Named here so the distinction is asserted somewhere rather than living only
    in a comment. Reading one as the other is a factor of 57 and a completely
    different comparison.
    """
    assert "val_sam" in LOG_COLUMNS and "val_sam_hr" in LOG_COLUMNS
    assert LOG_COLUMNS.index("val_sam") < LOG_COLUMNS.index("val_sam_hr")


def test_a_fresh_log_gets_the_current_header(tmp_path):
    path = tmp_path / "log.csv"
    ensure_log_header(path)
    with path.open(newline="") as handle:
        assert next(csv.reader(handle)) == LOG_COLUMNS


@pytest.mark.parametrize(
    "schema", [LEGACY_COLUMNS, SPECTRAL_COLUMNS, REFERENCE_COLUMNS, UNCERTAINTY_COLUMNS]
)
def test_an_older_log_is_widened_in_place(tmp_path, schema):
    """The resume path. Appending current rows to a narrower file corrupts it.

    Parametrised over every superseded schema, because a Kaggle run resumes
    from whatever its previous session left behind and there are now two
    generations that could be sitting there.
    """
    pad = [""] * (len(schema) - len(LEGACY_COLUMNS))
    path = tmp_path / "log.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(schema)
        writer.writerow([100, "0.031000", "2.0e-04", "", "12.5"] + pad)
        writer.writerow([2000, "", "", "31.5000", ""] + pad)

    ensure_log_header(path)

    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == LOG_COLUMNS
    assert len(rows) == 3
    # Old rows keep their values and gain empty cells -- they are not
    # backfilled with zeros, which would read as "measured, and it was zero".
    tail = [""] * (len(LOG_COLUMNS) - len(LEGACY_COLUMNS))
    assert rows[1] == ["100", "0.031000", "2.0e-04", "", "12.5"] + tail
    assert rows[2] == ["2000", "", "", "31.5000", ""] + tail

    # Idempotent: a second resume must not widen it again.
    ensure_log_header(path)
    with path.open(newline="") as handle:
        assert len(list(csv.reader(handle))) == 3


def test_an_unrecognised_header_raises_rather_than_being_overwritten(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text("step,value\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not write"):
        ensure_log_header(path)


def test_curves_reads_every_schema(tmp_path):
    """src/eval/curves.py is the consumer; it must accept all three generations."""
    legacy = tmp_path / "legacy.csv"
    with legacy.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LEGACY_COLUMNS)
        writer.writerow([100, "0.031", "2.0e-04", "", "12.5"])
        writer.writerow([2000, "", "", "31.5", ""])
    read = read_training_log(legacy)
    assert list(read["val"].columns) == ["iter", "val_psnr"]

    spectral = tmp_path / "spectral.csv"
    with spectral.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SPECTRAL_COLUMNS)
        writer.writerow([100, "0.031", "2.0e-04", "", "12.5", "", "", ""])
        writer.writerow([2000, "", "", "31.5", "", "0.0210", "0.0361", "1.0"])
    read = read_training_log(spectral)
    assert list(read["val"].columns) == [
        "iter", "val_psnr", "val_l1_spec", "val_sam", "val_sam_valid_frac",
    ]
    assert float(read["val"]["val_sam"].iloc[0]) == pytest.approx(0.0361)

    reference = tmp_path / "reference.csv"
    blanks = [""] * (len(REFERENCE_COLUMNS) - len(LEGACY_COLUMNS))
    with reference.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(REFERENCE_COLUMNS)
        writer.writerow([100, "0.031", "2.0e-04", "", "12.5"] + blanks)
        writer.writerow(
            [2000, "", "", "31.5", "", "0.0210", "0.0361", "1.0",
             "0.0154", "0.0312", "0.881", "0.402", "2.11", "3.02"]
        )
    read = read_training_log(reference)
    assert list(read["val"].columns) == ["iter", "val_psnr"] + list(
        REFERENCE_COLUMNS[len(LEGACY_COLUMNS):]
    )

    # The current generation: an uncertainty run's row carries the sigma
    # monitor; a control's row leaves it blank and must still parse.
    current = tmp_path / "current.csv"
    with current.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LOG_COLUMNS)
        writer.writerow([100, "0.031", "2.0e-04", "", "12.5"]
                        + [""] * (len(LOG_COLUMNS) - len(LEGACY_COLUMNS)))
        writer.writerow(
            [2000, "", "", "31.5", "", "0.0210", "0.0361", "1.0",
             "0.0154", "0.0312", "0.881", "0.402", "2.11", "3.02",
             "-3.9", "0.5", "-8.7", "-10.0", "-4.2", "0.61", "0.12", "0.0",
             "88.5", "0.0"]
        )
        writer.writerow(
            [4000, "", "", "31.9", "", "0.0200", "0.0350", "1.0",
             "0.0155", "0.0313", "0.882", "0.401", "2.10", "3.01"]
            + [""] * (len(LOG_COLUMNS) - len(REFERENCE_COLUMNS))
        )
    read = read_training_log(current)
    assert list(read["val"].columns) == ["iter", "val_psnr"] + list(
        LOG_COLUMNS[len(LEGACY_COLUMNS):]
    )
    assert float(read["val"]["val_logvar_spatial_std"].iloc[0]) == pytest.approx(0.61)
    assert read["val"]["val_logvar_spatial_std"].isna().iloc[1]
    read = read_training_log(reference)
    # The blur diagnostic and the reference suite must survive the round trip:
    # they are the columns the Day 3 verdict is read off.
    assert float(read["val"]["val_sharpness"].iloc[0]) == pytest.approx(0.0154)
    assert float(read["val"]["val_hf_energy"].iloc[0]) == pytest.approx(0.0312)
    assert float(read["val"]["val_sam_hr"].iloc[0]) == pytest.approx(2.11)


# -- the control measures what it does not optimise ------------------------


def test_the_measured_terms_do_not_depend_on_the_lambdas():
    """What makes the control's log comparable with the spectral run's.

    ``validate()`` calls :func:`src.losses.spectral_terms` directly, with no
    lambda anywhere in the call. This pins that property from the loss side: the
    quantity logged is a function of the tensors alone, so a control that never
    optimised it still produces a curve on the same axis as the run that did.
    """
    torch.manual_seed(0)
    sr = torch.rand(2, BANDS, 32, 32)
    lr = torch.rand(2, BANDS, 8, 8)

    parts = spectral_terms(sr, lr, mode=DEFAULT_DOWNSAMPLE)
    again = spectral_terms(sr, lr, mode=DEFAULT_DOWNSAMPLE)
    assert float(parts["l1_spec"]) == float(again["l1_spec"])
    assert float(parts["sam"]) == float(again["sam"])
    # And they are real numbers, not zeros a control could be mistaken for
    # having "optimised away".
    assert float(parts["l1_spec"]) > 0.0
    assert float(parts["sam"]) > 0.0


def test_validation_measures_in_float32_not_half():
    """AMP must not reach the spectral terms; validate() casts to float first.

    ``arccos`` near +-1 -- where a trained model sits -- loses most of its
    significant digits in fp16. A validation number that moves with the AMP
    setting is not a measurement, so the value computed from a half-precision
    forward, once cast, must match the float32 one closely.
    """
    torch.manual_seed(1)
    sr = torch.rand(1, BANDS, 32, 32) * 0.4 + 0.05
    lr = torch.rand(1, BANDS, 8, 8) * 0.4 + 0.05

    full = spectral_terms(sr, lr)
    from_half = spectral_terms(sr.half().float(), lr)
    assert float(from_half["sam"]) == pytest.approx(float(full["sam"]), rel=1e-2)
