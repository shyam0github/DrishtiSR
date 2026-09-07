"""The Run A gate and the curve reading, on fabricated logs and summaries.

Two pieces of logic decide what happens to a training run, and neither needs a
checkpoint, a dataset, or a GPU to test:

- ``apply_gate`` -- does the model beat bicubic on every criterion? The cases
  that matter are the ones where it *nearly* does: two of three metrics, a
  metric that moved the wrong way by a hair, and a metric that was never
  computed at all. The last is the dangerous one, because an absent number is
  the easiest thing in the world to read as a pass.
- ``curve_verdict`` -- was validation still improving when the run stopped?
  A wrong answer here is what makes a team spend another 8 GPU-hours on a run
  that peaked at 20% and has been flat since.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.eval.curves import curve_verdict, read_training_log

# Same convention as tests/test_kaggle_run.py: scripts/ is not a package, so the
# entry point is imported by putting that directory on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_runA import apply_gate  # noqa: E402


class _Result(SimpleNamespace):
    """Stand-in for EvaluationResult: apply_gate only reads ``summary``."""


def _cfg(gate: dict) -> dict:
    return {"eval_runA": {"gate": gate}}


def _results(bicubic: dict, model: dict) -> dict:
    return {
        "bicubic": _Result(summary={k: {"mean": v} for k, v in bicubic.items()}),
        "edsr_runA": _Result(summary={k: {"mean": v} for k, v in model.items()}),
    }


_GATE = {"psnr_mean": "higher", "ssim_mean": "higher", "lpips": "lower"}


def _apply(bicubic: dict, model: dict, gate: dict = None, logger=None) -> dict:
    return apply_gate(
        _cfg(gate or _GATE),
        _results(bicubic, model),
        "edsr_runA",
        "bicubic",
        logger or _SilentLogger(),
    )


class _SilentLogger:
    def info(self, *args, **kwargs) -> None: ...
    def error(self, *args, **kwargs) -> None: ...
    def warning(self, *args, **kwargs) -> None: ...


# -- the gate --------------------------------------------------------------


def test_gate_passes_only_when_every_metric_improves():
    gate = _apply(
        bicubic={"psnr_mean": 30.0, "ssim_mean": 0.80, "lpips": 0.40},
        model={"psnr_mean": 31.0, "ssim_mean": 0.85, "lpips": 0.30},
    )
    assert gate["passed"] is True
    assert gate["evaluable"] is True
    assert all(entry["passed"] for entry in gate["metrics"])


def test_two_of_three_is_a_failure():
    """The whole point of a hard gate: a model that wins on PSNR and SSIM but
    loses on LPIPS has not passed. Averaging the three, or counting a majority,
    is exactly the softening this test exists to prevent."""
    gate = _apply(
        bicubic={"psnr_mean": 30.0, "ssim_mean": 0.80, "lpips": 0.40},
        model={"psnr_mean": 31.0, "ssim_mean": 0.85, "lpips": 0.45},
    )
    assert gate["passed"] is False
    failed = [e["metric"] for e in gate["metrics"] if not e["passed"]]
    assert failed == ["lpips"]
    assert "lpips" in gate["metrics"][2]["reason"]


def test_an_exact_tie_does_not_pass():
    """Equal to bicubic is not better than bicubic."""
    gate = _apply(
        bicubic={"psnr_mean": 30.0, "ssim_mean": 0.80, "lpips": 0.40},
        model={"psnr_mean": 30.0, "ssim_mean": 0.85, "lpips": 0.30},
    )
    assert gate["passed"] is False
    assert gate["metrics"][0]["passed"] is False
    assert gate["metrics"][0]["delta_model_minus_bicubic"] == 0.0


def test_lpips_direction_is_lower_is_better():
    """A smaller LPIPS must count as an improvement; a larger one must not.
    Getting this backwards would publish a model that looks worse to the
    perceptual metric while the table says PASS."""
    better = _apply(
        bicubic={"psnr_mean": 30.0, "ssim_mean": 0.80, "lpips": 0.40},
        model={"psnr_mean": 31.0, "ssim_mean": 0.85, "lpips": 0.39},
    )
    assert better["metrics"][2]["passed"] is True

    worse = _apply(
        bicubic={"psnr_mean": 30.0, "ssim_mean": 0.80, "lpips": 0.40},
        model={"psnr_mean": 31.0, "ssim_mean": 0.85, "lpips": 0.41},
    )
    assert worse["metrics"][2]["passed"] is False


def test_an_uncomputed_metric_fails_the_gate_rather_than_being_skipped():
    """An unmeasured criterion is not a satisfied one.

    This is the failure mode the gate is most likely to hit in practice --
    cfg.metrics.lpips.enabled is false on any run without internet, and a gate
    that quietly judged the two metrics it *did* have would report PASS on
    two-thirds of the evidence.
    """
    gate = _apply(
        bicubic={"psnr_mean": 30.0, "ssim_mean": 0.80},
        model={"psnr_mean": 31.0, "ssim_mean": 0.85},
    )
    assert gate["passed"] is False
    assert gate["evaluable"] is False
    lpips_entry = next(e for e in gate["metrics"] if e["metric"] == "lpips")
    assert lpips_entry["passed"] is False
    assert lpips_entry["model"] is None
    assert "not computed" in lpips_entry["reason"]


def test_an_unknown_direction_raises():
    with pytest.raises(ValueError, match="higher.*lower"):
        _apply(
            bicubic={"psnr_mean": 30.0},
            model={"psnr_mean": 31.0},
            gate={"psnr_mean": "bigger"},
        )


# -- the curve reading -----------------------------------------------------


def _val(points) -> pd.DataFrame:
    return pd.DataFrame(points, columns=["iter", "val_psnr"])


def test_a_curve_that_peaks_early_and_decays_is_not_undertrained():
    """Run A's actual shape: peak at 8k of 40k, then a slow slide.

    Values here are the real ones from runs/runA/log.csv, thinned. The verdict
    must be "not still improving" -- reading this curve as undertrained is what
    would send another 8 GPU-hours after a run that has been going backwards
    for 32,000 iterations.
    """
    verdict = curve_verdict(
        _val([
            (2000, 34.9790), (8000, 35.2842), (16000, 35.2461),
            (24000, 35.1330), (32000, 35.0626), (40000, 35.0328),
        ])
    )
    assert verdict["still_improving"] is False
    assert verdict["best_iter"] == 8000
    assert verdict["delta_final_minus_best"] < 0
    assert verdict["tail_slope_db_per_1k"] < 0
    assert "NOT still improving" in verdict["verdict"]
    assert "UNDERTRAINED" not in verdict["verdict"]


def test_a_curve_still_climbing_at_the_end_is_undertrained():
    verdict = curve_verdict(
        _val([
            (2000, 30.0), (10000, 32.0), (20000, 33.5),
            (30000, 34.4), (40000, 35.0),
        ])
    )
    assert verdict["still_improving"] is True
    assert verdict["best_iter"] == 40000
    assert verdict["tail_slope_db_per_1k"] > 0
    assert "UNDERTRAINED" in verdict["verdict"]


def test_a_plateau_is_not_still_improving():
    """The peak lands in the final quarter by luck of the noise, but the tail is
    flat. 'Best point is last' alone must not be enough to call it undertrained,
    or every noisy plateau reads as a reason to keep spending GPU-hours."""
    verdict = curve_verdict(
        _val([
            (10000, 35.00), (20000, 35.02), (30000, 35.01),
            (36000, 35.00), (40000, 35.02),
        ])
    )
    assert verdict["still_improving"] is False


def test_one_validation_point_cannot_support_a_trend():
    with pytest.raises(ValueError, match="at least 2"):
        curve_verdict(_val([(40000, 35.0)]))


# -- reading the log -------------------------------------------------------


def _write_log(path, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iter", "loss", "lr", "val_psnr", "sec_per_100it"])
        writer.writerows(rows)


def test_the_two_row_kinds_are_separated_by_their_empty_cells(tmp_path):
    path = tmp_path / "log.csv"
    _write_log(path, [
        [100, 0.05, 2e-4, "", 27.0],
        [200, 0.04, 2e-4, "", 27.0],
        [200, "", "", 34.5, ""],
        [300, 0.03, 1e-4, "", 27.0],
        [400, "", "", 34.9, ""],
    ])
    frames = read_training_log(path)
    assert list(frames["train"]["iter"]) == [100, 200, 300]
    assert list(frames["val"]["iter"]) == [200, 400]
    assert frames["val"]["val_psnr"].iloc[-1] == pytest.approx(34.9)


def test_a_resumed_run_keeps_the_last_value_per_iteration(tmp_path):
    """A resumed run re-logs iterations it already logged. Keeping both would
    draw a sawtooth and could put the 'best' point on a superseded value."""
    path = tmp_path / "log.csv"
    _write_log(path, [
        [100, 0.05, 2e-4, "", 27.0],
        [200, 0.04, 2e-4, "", 27.0],
        [200, "", "", 30.0, ""],
        # session 2 resumes from 100 and re-logs 200 with the real value
        [100, 0.049, 2e-4, "", 26.0],
        [200, 0.039, 2e-4, "", 26.0],
        [200, "", "", 34.5, ""],
    ])
    frames = read_training_log(path)
    assert list(frames["train"]["iter"]) == [100, 200]
    assert frames["train"]["loss"].iloc[0] == pytest.approx(0.049)
    assert len(frames["val"]) == 1
    assert frames["val"]["val_psnr"].iloc[0] == pytest.approx(34.5)


def test_a_log_with_no_validation_rows_raises(tmp_path):
    """Silently drawing a blank validation panel makes 'never validated' and
    'validation was flat' look identical."""
    path = tmp_path / "log.csv"
    _write_log(path, [[100, 0.05, 2e-4, "", 27.0], [200, 0.04, 2e-4, "", 27.0]])
    with pytest.raises(ValueError, match="no validation rows"):
        read_training_log(path)


def test_a_foreign_csv_is_rejected_rather_than_misread(tmp_path):
    path = tmp_path / "log.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "training_loss"])
        writer.writerow([1, 0.5])
    with pytest.raises(ValueError, match="expected"):
        read_training_log(path)


def test_a_missing_log_names_how_to_get_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle_run.py fetch"):
        read_training_log(tmp_path / "absent.csv")
