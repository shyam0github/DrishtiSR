"""Build the API's display samples into app/samples/<id>/ (P9).

    python scripts/mvp/build_samples.py        (cwd + PYTHONPATH = repo root)

TEST pairs, DISPLAY ONLY -- never used for any selection or tuning. Deterministic:
a seeded 200-id subset of the cached TEST ids (frozen split CSV order), each
centre-cropped to 128 px LR (``centre_crop_pair``), ranked by HR HF energy
(``src.metrics.sharpness.hf_energy_ratio`` via ``trust.hf_energy``); two per
tercile, at 1/3 and 2/3 of each tercile. Plus two Day 2 Delhi crops (no GT)
when the main tree has them, converted as ``(dn - cfg.delhi.dn_offset) / 10000``
(the Day 2 ``tiled.run_file --dn-offset 1000`` formula).

Per sample: lr.npy, hr.npy (if GT), lr.tif (float32 reflectance, real
geotransform/CRS), thumb.png (LR RGB, the shared 2-98 % LR stretch) and
manifest.json. Arrays are gitignored when the total is >= 25 MB.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.io_geotiff import profile_to_json, write_tif  # noqa: E402
from src.data.patches import centre_crop_pair  # noqa: E402
from src.data.sen2naip import SEN2NAIPv2Dataset  # noqa: E402
from src.infer import render, trust  # noqa: E402
from src.utils.config import load_config  # noqa: E402

from rasterio.crs import CRS  # noqa: E402

SEED = 26142
N_SUBSET = 200
LR_SIZE = 128
DELHI_SIZE = 256
OUT = ROOT / "app" / "samples"
COMMIT_LIMIT_BYTES = 25 * 1024 * 1024
DELHI_LABEL = "Delhi · Sentinel-2 L2A · no ground truth"


def main_tree() -> Path:
    common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute",
                             "--git-common-dir"], check=True, capture_output=True, text=True).stdout.strip()
    return Path(common).parent


def save_sample(sid: str, lr: np.ndarray, hr, profile, manifest: dict) -> int:
    d = OUT / sid
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    np.save(d / "lr.npy", np.ascontiguousarray(lr, dtype=np.float32))
    if hr is not None:
        np.save(d / "hr.npy", np.ascontiguousarray(hr, dtype=np.float32))
    write_tif(d / "lr.tif", lr, profile)
    render.save_png(render.to_png_array(lr, render.stretch_params(lr), "rgb"), d / "thumb.png")
    manifest = {"id": sid, **manifest, "lr_size": [int(lr.shape[1]), int(lr.shape[2])],
                "has_gt": hr is not None, "profile": profile_to_json(profile),
                "units": "surface reflectance, float32, unclipped; bands B04,B03,B02,B08"}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return sum(f.stat().st_size for f in d.iterdir())


def build_test(cfg, main: Path) -> int:
    split_csv = main / "outputs" / str(cfg.splits.output_name).format(dataset=cfg.dataset.name)
    cfg.paths.cache_dir = str(main / "outputs" / "cache")
    ds = SEN2NAIPv2Dataset(cfg, validate=False)
    ids = pd.read_csv(split_csv, dtype={"sample_id": str})
    test_ids = [s for s in ids.loc[ids["split"] == "test", "sample_id"] if ds._cache_paths(s)[0].is_file()]
    rng = np.random.default_rng(SEED)
    subset = [test_ids[i] for i in sorted(rng.choice(len(test_ids), size=min(N_SUBSET, len(test_ids)),
                                                     replace=False))]
    scale = int(cfg.sr.scale)

    def load(sid):
        npz, side = ds._cache_paths(sid)
        with np.load(npz) as f:
            lr_raw, hr_raw = f["lr"][ds.band_indices], f["hr"][ds.band_indices]
        lr, _ = ds.to_reflectance(lr_raw, cfg.dataset.nodata_value, float(cfg.dataset.nodata_fill))
        hr, _ = ds.to_reflectance(hr_raw, cfg.dataset.nodata_value, float(cfg.dataset.nodata_fill))
        crop = centre_crop_pair(lr, hr, lr_size=LR_SIZE, scale=scale)
        return crop, json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}

    energies = []
    for sid in subset:
        crop, _ = load(sid)
        energies.append(trust.hf_energy(np.ascontiguousarray(crop["hr"])))
    order = np.argsort(np.asarray(energies), kind="stable")
    total = 0
    for t_idx, (tier, chunk) in enumerate(zip(("low", "mid", "high"), np.array_split(order, 3))):
        for k, pos in enumerate((len(chunk) // 3, (2 * len(chunk)) // 3), start=1):
            i = int(chunk[pos])
            sid = subset[i]
            crop, side = load(sid)
            c = crop["coords"]
            lrp = (side.get("profile") or {}).get("lr") or {}
            profile = None
            if lrp.get("crs") and lrp.get("transform"):
                profile = {"crs": CRS.from_user_input(lrp["crs"]),
                           "transform": Affine(*lrp["transform"][:6]) @ Affine.translation(c.lr_col, c.lr_row)}
            total += save_sample(f"test-{tier}-{k}", np.ascontiguousarray(crop["lr"]),
                                 np.ascontiguousarray(crop["hr"]), profile, {
                "label": f"Test tile · {tier} texture {k}", "order": t_idx * 2 + k,
                "source_id": sid, "split": "test", "display_only": True,
                "hr_hf_energy": float(energies[i]), "tercile": tier,
                "selection": f"seed {SEED}, {len(subset)}-id TEST subset, HR HF-energy tercile, rank {pos}",
                "crop": {"lr_row": int(c.lr_row), "lr_col": int(c.lr_col), "lr_size": LR_SIZE},
            })
            print(f"test-{tier}-{k}: {sid} hf={energies[i]:.5f}")
    return total


def build_delhi(cfg, main: Path) -> int:
    offset = float(cfg["delhi"]["dn_offset"])
    div = float(cfg["dataset"]["reflectance_scale"])
    specs = [("delhi-urban", "delhi_A_20241030.tif", "dense urban", "centre"),
             ("delhi-mixed", "delhi_B_20241030.tif", "mixed / agricultural", "mixed")]
    total = 0
    for order, (sid, fname, kind, rule) in enumerate(specs, start=10):
        path = main / "data" / "delhi" / fname
        if not path.is_file():
            print(f"skip {sid}: {path} absent")
            continue
        with rasterio.open(path) as src:
            full = (src.read().astype(np.float32) - np.float32(offset)) / np.float32(div)
            crs, transform = src.crs, src.transform
        _, H, W = full.shape
        if rule == "centre":
            r0, c0 = (H - DELHI_SIZE) // 2, (W - DELHI_SIZE) // 2
            why = "centre window"
        else:
            # Most "mixed" window: vegetated fraction (NDVI > 0.3) closest to 0.5,
            # among windows free of bright outliers (p99 reflectance <= 1).
            best = None
            for r in range(0, H - DELHI_SIZE + 1, 128):
                for c in range(0, W - DELHI_SIZE + 1, 128):
                    x = full[:, r:r + DELHI_SIZE, c:c + DELHI_SIZE]
                    if np.percentile(x, 99) > 1.0:
                        continue
                    ndvi = (x[3] - x[0]) / np.maximum(x[3] + x[0], 1e-6)
                    score = abs(float((ndvi > 0.3).mean()) - 0.5)
                    if best is None or score < best[0]:
                        best = (score, r, c)
            _, r0, c0 = best
            why = "stride-128 window with NDVI>0.3 fraction closest to 0.5 (p99 <= 1)"
        lr = np.ascontiguousarray(full[:, r0:r0 + DELHI_SIZE, c0:c0 + DELHI_SIZE])
        win_tf = rasterio.windows.transform(Window(c0, r0, DELHI_SIZE, DELHI_SIZE), transform)
        total += save_sample(sid, lr, None, {"crs": crs, "transform": win_tf}, {
            "label": f"{DELHI_LABEL} ({kind})", "order": order, "source_id": fname,
            "window": {"row": int(r0), "col": int(c0), "size": DELHI_SIZE, "rule": why},
            "conversion": f"(dn - {offset:g}) / {div:g}", "processing_baseline": "05.11",
        })
        print(f"{sid}: {fname} window r{r0} c{c0}")
    return total


def main() -> None:
    cfg = load_config()
    main_root = main_tree()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in OUT.iterdir():
        if d.is_dir():
            shutil.rmtree(d)
    total = build_test(cfg, main_root) + build_delhi(cfg, main_root)
    gi = OUT / ".gitignore"
    if total >= COMMIT_LIMIT_BYTES:
        gi.write_text("# Built by scripts/mvp/build_samples.py; too large to commit.\n*\n!.gitignore\n",
                      encoding="utf-8")
    elif gi.exists():
        gi.unlink()
    print(f"samples total {total / 2**20:.1f} MB -> {'gitignored' if total >= COMMIT_LIMIT_BYTES else 'commit'}")


if __name__ == "__main__":
    main()
