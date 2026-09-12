"""P7 reporting: wrapped selection rule, table rendering, test-split lock.

Synthetic fixtures only (tests/mvp/fixtures/runs); no checkpoints or data.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from src.reporting import tables as T
from src.reporting.selection import RULE_NAME, load_cached_candidates, select_run

FIX = Path(__file__).resolve().parent / "fixtures" / "runs"
REPO = Path(__file__).resolve().parents[2]
RULE = {"metric": "reflectance", "tie_metric": "lpips", "tie_better": "lower", "n_boot": 2000, "ci": 0.95}


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    shutil.copytree(FIX, tmp_path / "fx")
    return tmp_path / "fx"


def _run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "rundir"
    d.mkdir()
    (d / "best.pt").write_bytes(b"")
    (d / "last.pt").write_bytes(b"")
    return d


def _reader(path: Path) -> int:
    return {"best.pt": 100, "last.pt": 200}[path.name]


def test_rule_picks_lowest_consistency_first(tmp_path):
    rec = select_run("sep", _run_dir(tmp_path), FIX / "eval_all_ckpts", RULE, seed=42, iteration_reader=_reader)
    assert rec["rule"] == RULE_NAME
    assert rec["chosen"]["label"] == "it00200"  # lower consistency error despite worse LPIPS
    assert rec["chosen"]["path"].endswith("last.pt")
    assert rec["best_pt"] == {**rec["best_pt"], "iteration": 100, "coincides": False}
    assert {c["label"] for c in rec["candidates"]} == {"it00100", "it00200"}
    assert rec["split"] == "val" and rec["n_pairs"] == 40


def test_rule_breaks_ci_tie_on_lpips(tmp_path):
    rec = select_run("tie", _run_dir(tmp_path), FIX / "eval_all_ckpts", RULE, seed=42, iteration_reader=_reader)
    assert rec["minimum"] == "it00200"
    assert set(rec["tied_set"]) == {"it00100", "it00200"}
    assert rec["chosen"]["label"] == "it00100"  # tied on consistency -> lower LPIPS wins
    assert rec["best_pt"]["coincides"] is True


def test_selection_refuses_non_val(tmp_path):
    src = FIX / "eval_all_ckpts" / "sep" / "it00100" / "per_pair.csv"
    dst = tmp_path / "c" / "x" / "it00100" / "per_pair.csv"
    dst.parent.mkdir(parents=True)
    dst.write_text(src.read_text().replace(",val,", ",test,"))
    with pytest.raises(ValueError, match="VAL only"):
        load_cached_candidates(tmp_path / "c", "x")


def _inputs(tree: Path, split: str = "val") -> T.ReportInputs:
    rep = tree / "reports"
    return T.ReportInputs(runs_root=tree / "runs", eval_cache=tree / "eval_all_ckpts",
                          ab_csv=rep / "day3_ab_metrics.csv", ab_summary=rep / "day3_ab_summary.json",
                          day3_results=rep / "day3_results.json", run_a_report=rep / "day2_runA.json",
                          mvp_dir=rep / "mvp", split=split)


def test_pending_rows_render(tree):
    abl = T.build_ablation(_inputs(tree))
    rows = {r["key"]: r for r in abl["rows"]}
    assert rows["b2"]["status"] == "pending"
    assert rows["a2"]["status"] == rows["runA"]["status"] == rows["best"]["status"] == "filled"
    md = T.render_ablation_md(abl)
    assert "| B2 (λ₁=0.3, λ₂=0.06) | pending |" in md
    assert "GT floor 0.005733" in md
    dep = T.build_deployment(tree / "reports" / "mvp")
    assert all(r["status"] == "pending" for r in dep["rows"])
    assert "| TTA-8 | pending |" in T.render_uncertainty_md(T.build_uncertainty(tree / "reports" / "mvp"))


def test_blur_index_below_threshold_is_flagged(tree):
    abl = T.build_ablation(_inputs(tree))
    rows = {r["key"]: r for r in abl["rows"]}
    assert rows["b1"]["blur_flag"] is True and rows["b1"]["blur_index"] == pytest.approx(0.85)
    assert rows["a2"]["blur_flag"] is False and rows["a2"]["blur_index"] is None
    assert "0.850 ⚠ BLUR HAZARD" in T.render_ablation_md(abl)
    assert rows["b1"]["d_lpips_vs_a2_pct"] == pytest.approx(100 * (0.345 / 0.33 - 1))


def test_dagger_footnote_and_run_a_interim(tree):
    abl = T.build_ablation(_inputs(tree))
    md = T.render_ablation_md(abl)
    assert "† Run A is not a valid control" in md
    for axis in ("1.52M vs 855,652", "no dihedral augmentation", "frozen crop origins", "dataset size"):
        assert axis in md
    assert "fb8659d" in md
    run_a = next(r for r in abl["rows"] if r["key"] == "runA")
    assert run_a["interim"] is True and run_a["params"] == 1518724 and run_a["hf_ratio_vs_gt"] is None


def test_test_split_lock(tmp_path):
    lock, dec = tmp_path / "TEST_EVAL_LOCK", tmp_path / "decisions.md"
    assert T.guard_test_split("val", False, False, lock, dec)["action"] == "none"
    with pytest.raises(T.TestSplitLocked, match="--final"):
        T.guard_test_split("test", False, False, lock, dec)
    assert not lock.exists()
    assert T.guard_test_split("test", True, False, lock, dec)["action"] == "created_lock"
    assert lock.exists()
    with pytest.raises(T.TestSplitLocked, match="exists"):
        T.guard_test_split("test", True, False, lock, dec)
    assert not dec.exists()
    assert T.guard_test_split("test", True, True, lock, dec)["action"] == "forced_override"
    assert "TEST_EVAL_LOCK overridden" in dec.read_text(encoding="utf-8")


def _make_tables():
    spec = importlib.util.spec_from_file_location("p7_make_tables", REPO / "scripts" / "mvp" / "make_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_tables_end_to_end(tree):
    mt = _make_tables()
    common = ["--runs-root", str(tree / "runs"), "--eval-cache", str(tree / "eval_all_ckpts"),
              "--reports-dir", str(tree / "reports"), "--decisions", str(tree / "decisions.md")]
    assert mt.main(common) == 0
    mvp = tree / "reports" / "mvp"
    for name in ("ablation.md", "ablation.csv", "ablation.json", "deploy_table.md", "uncertainty_table.md", "headline.json"):
        assert (mvp / name).is_file(), name
    assert mt.main(common + ["--split", "test"]) == 2  # no --final
    assert not (mvp / "TEST_EVAL_LOCK").exists()
    assert mt.main(common + ["--split", "test", "--final"]) == 0
    assert (mvp / "TEST_EVAL_LOCK").exists() and (mvp / "ablation_test.md").is_file()
    assert "| A2 (control) | pending |" in (mvp / "ablation_test.md").read_text(encoding="utf-8")
    assert mt.main(common + ["--split", "test", "--final"]) == 2
    assert mt.main(common + ["--split", "test", "--final", "--force"]) == 0
    assert "overridden" in (tree / "decisions.md").read_text(encoding="utf-8")
