"""P0 smoke: the shared Predictor interface against the real Day 3 checkpoints.

Needs the main tree's gitignored artefacts (runs/, outputs/cache, outputs/splits
CSV). The main tree is located through git -- the parent of the common git dir
-- rather than a hardcoded path, so this runs from any worktree. When the
artefacts are absent (CI, Kaggle, a fresh clone) the checkpoint tests SKIP with
the missing path named; the fixture/contract test always runs.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import drishtisr
from src.data.patches import centre_crop_pair
from src.data.sen2naip import SEN2NAIPv2Dataset
from src.eval.baselines import bicubic_upsample
from src.infer.predictor import (
    INFO_KEYS,
    TorchPredictor,
    build_model_from_checkpoint,
    infer_architecture,
    make_predictor,
)
from src.metrics.image_quality import psnr
from src.utils.config import load_config
from src.utils.gitmeta import git_metadata
from src.utils.paths import repo_root

WORKTREE = repo_root()


def _main_tree() -> Path:
    common = subprocess.run(
        ["git", "-C", str(WORKTREE), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return Path(common).parent


MAIN = _main_tree()
# Day 3 winner: A2's it12000 checkpoint (reports/day3_ab_summary.json
# selection.a2.selected), which is last.pt (it=11999, 1-based 12000); best.pt
# in the same directory is it10500. See docs/mvp/inventory.md section 2.
A2_CKPT = MAIN / "runs" / "day3" / "a2" / "last.pt"
RUNA_CKPT = MAIN / "runs" / "runA" / "best.pt"


def _require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"main-tree artefacts absent: {missing}")


def _val_pair(cfg):
    """First VAL id (frozen split CSV order) that is cached; its centre LR/HR crop.

    Returns (sample_id, lr (1, C, s, s), hr (1, C, 4s, 4s)), float32 surface
    reflectance, unclipped, band order cfg.dataset.bands, s = cfg.patches.lr_size.
    The catalog (network) is never touched: the npz is read directly and
    converted by the dataset's own to_reflectance with its own nodata policy.
    """
    split_csv = MAIN / "outputs" / str(cfg.splits.output_name).format(dataset=cfg.dataset.name)
    cache_root = MAIN / "outputs" / "cache"
    _require(split_csv, cache_root)
    cfg.paths.cache_dir = str(cache_root)
    ds = SEN2NAIPv2Dataset(cfg, validate=False)
    import pandas as pd

    ids = pd.read_csv(split_csv, dtype={"sample_id": str})
    val_ids = ids.loc[ids["split"] == str(cfg.loader.val_split), "sample_id"]
    for sid in val_ids:
        npz, _ = ds._cache_paths(sid)
        if npz.is_file():
            break
    else:
        pytest.skip("no VAL sample is cached in the main tree")
    with np.load(npz) as handle:
        lr_raw, hr_raw = handle["lr"][ds.band_indices], handle["hr"][ds.band_indices]
    lr, _ = ds.to_reflectance(lr_raw, cfg.dataset.nodata_value, float(cfg.dataset.nodata_fill))
    hr, _ = ds.to_reflectance(hr_raw, cfg.dataset.nodata_value, float(cfg.dataset.nodata_fill))
    crop = centre_crop_pair(lr, hr, lr_size=int(cfg.patches.lr_size), scale=int(cfg.sr.scale))
    return sid, np.ascontiguousarray(crop["lr"])[None], np.ascontiguousarray(crop["hr"])[None]


def test_imports_resolve_inside_worktree():
    assert Path(drishtisr.__file__).resolve().is_relative_to(WORKTREE.resolve()), drishtisr.__file__


def test_run_a_loads_and_state_dict_inference_agrees():
    _require(RUNA_CKPT)
    model, meta = build_model_from_checkpoint(RUNA_CKPT)
    assert meta["arch_source"] == "args"
    assert (meta["arch"]["n_resblocks"], meta["arch"]["n_feats"]) == (16, 64)
    state = torch.load(RUNA_CKPT, map_location="cpu", weights_only=False)["model"]
    inferred = infer_architecture(state)
    assert {k: inferred[k] for k in ("scale", "n_resblocks", "n_feats", "in_ch")} == \
           {k: int(meta["arch"][k]) for k in ("scale", "n_resblocks", "n_feats", "in_ch")}


def test_a2_smoke_on_val_patch():
    _require(A2_CKPT)
    cfg = load_config()
    sid, lr, hr = _val_pair(cfg)
    scale = int(cfg.sr.scale)

    pred = make_predictor("torch", checkpoint_path=A2_CKPT, threads=6, interim=False)
    assert isinstance(pred, TorchPredictor)
    assert set(pred.info) == set(INFO_KEYS)
    assert pred.info["params"] == 855_652 and pred.info["has_scale_head"] is False

    sr = pred(lr)
    n, c, h, w = lr.shape
    assert sr.shape == (1, 4, scale * h, scale * w) and sr.dtype == np.float32
    assert np.isfinite(sr).all()

    # Eval calls model(lr) directly on reflectance; the predictor must agree.
    with torch.no_grad():
        direct = pred.model(torch.from_numpy(lr)).numpy()
    assert np.allclose(sr, direct, atol=1e-5), float(np.abs(sr - direct).max())

    data_range = float(cfg.metrics.data_range)
    bic = bicubic_upsample(lr, scale, antialias=bool(cfg.baseline.antialias))
    psnr_sr = float(psnr(sr[0], hr[0], data_range=data_range).mean)
    psnr_bic = float(psnr(bic[0], hr[0], data_range=data_range).mean)

    git = git_metadata(WORKTREE)
    out = WORKTREE / "reports" / "mvp" / "p0_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "what": "P0 predictor smoke: one VAL patch through A2 via TorchPredictor",
        "checkpoint_id": pred.info["checkpoint_id"],
        "checkpoint_path": str(A2_CKPT),
        "git_sha": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "split": str(cfg.loader.val_split),
        "n_samples": 1,
        "sample_id": sid,
        "lr_shape": list(lr.shape),
        "crop": f"centre_crop_pair lr_size={int(cfg.patches.lr_size)}",
        "psnr_sr_db": psnr_sr,
        "psnr_bicubic_db": psnr_bic,
        "psnr_data_range": data_range,
        "bicubic_antialias": bool(cfg.baseline.antialias),
        "max_abs_diff_vs_direct_forward": float(np.abs(sr - direct).max()),
        "model": pred.info,
        "interim": False,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")


def test_fixtures_match_contract():
    fx = WORKTREE / "app" / "fixtures"
    resp = json.loads((fx / "sample_response.json").read_text(encoding="utf-8"))
    health = json.loads((fx / "health.json").read_text(encoding="utf-8"))
    samples = json.loads((fx / "samples.json").read_text(encoding="utf-8"))

    assert health["status"] == "ok" and set(health["model"]) == set(INFO_KEYS)
    assert set(resp) == {"job_id", "input", "model", "uncertainty_method", "images",
                         "downloads", "metrics", "refs", "warnings"}
    assert set(resp["model"]) == set(INFO_KEYS) and resp["model"]["interim"] is False
    assert resp["uncertainty_method"] == "tta4"
    names = ["lr_rgb", "lr_fcc", "bicubic_rgb", "bicubic_fcc", "sr_rgb", "sr_fcc",
             "hr_rgb", "hr_fcc", "uncertainty", "consistency"]
    assert set(resp["images"]) == set(names)
    assert all(resp["images"][k] == f"/fixtures/{k}.png" for k in names)
    rf = resp["metrics"]["reference_free"]
    assert set(rf) == {"spec_l1", "spec_sam_deg", "spec_l1_bicubic", "spec_sam_bicubic_deg",
                       "hf_ratio_vs_bicubic", "unc_mean", "unc_p95", "runtime_ms"}
    gt = resp["metrics"]["with_gt"]
    assert gt is not None and set(gt["sr"]) == set(gt["bicubic"]) == {"lpips", "ssim", "psnr", "sam_deg", "ergas"}
    h, w = resp["input"]["lr_size"]
    assert resp["input"]["sr_size"] == [4 * h, 4 * w] and 32 <= h <= 512 and 32 <= w <= 512
    assert resp["refs"]["spec_l1_gt_floor"] == 0.005733 and resp["refs"]["sharpness_warn_below"] == 1.05
    for s in samples["samples"]:
        assert set(s) == {"id", "label", "thumb_url", "has_gt", "lr_size"}
