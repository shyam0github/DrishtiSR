"""P5: calibrate the fixed display scales of the trust overlays on VAL.

64 VAL patches (seed 26142), A2 via TorchPredictor, TTA-8 uncertainty and
training-operator consistency. Writes reports/mvp/trust_scales.json with the
p99 display maxima, the rank correlation of TTA-8 uncertainty with the actual
|SR - HR| error, and the same correlation for a trivial baseline (Sobel edge
magnitude of the bicubic image), plus per-image timing for tta 0/4/8 on one
128x128 LR.

VAL patch pool: every VAL tile (frozen split CSV order), centre-cropped to the
dataset's 128-px LR tile (cfg.sr.lr_patch_size), cut on the validation grid
(cfg.patches.lr_size 64, stride 64) -> 4 patches per tile. Patches are drawn
without replacement with numpy default_rng(seed). Read from the npz cache
directly (the catalog needs the network).

Run: python scripts/mvp/calibrate_trust_scales.py
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from src.data.patches import centre_crop_pair, crop_pair, patch_grid  # noqa: E402
from src.data.sen2naip import SEN2NAIPv2Dataset  # noqa: E402
from src.infer.predictor import TorchPredictor  # noqa: E402
from src.infer.trust import compute_trust  # noqa: E402
from src.metrics.sharpness import SOBEL_X, SOBEL_Y  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.gitmeta import git_metadata  # noqa: E402

SEED = 26142
MAIN = Path(r"D:\SIH\DrishtiSR")
A2_CKPT = MAIN / "runs" / "day3" / "a2" / "last.pt"
OUT = WORKTREE / "reports" / "mvp" / "trust_scales.json"


# ------------------------------------------------------------ data
class ValPatches:
    """Seeded VAL patch draws from the npz cache (see module docstring)."""

    def __init__(self) -> None:
        import pandas as pd

        cfg = load_config()
        cfg.paths.cache_dir = str(MAIN / "outputs" / "cache")
        self.cfg = cfg
        self.ds = SEN2NAIPv2Dataset(cfg, validate=False)
        self.scale = int(cfg.sr.scale)
        self.tile = int(cfg.sr.lr_patch_size)
        self.size = int(cfg.patches.lr_size)
        split_csv = MAIN / "outputs" / str(cfg.splits.output_name).format(dataset=cfg.dataset.name)
        ids = pd.read_csv(split_csv, dtype={"sample_id": str})
        self.ids = [s for s in ids.loc[ids["split"] == str(cfg.loader.val_split), "sample_id"]
                    if self.ds._cache_paths(s)[0].is_file()]
        self.origins = patch_grid(self.tile, self.tile, self.size, int(cfg.patches.stride))
        self.pool = [(sid, o) for sid in self.ids for o in self.origins]

    def load_tile(self, sid: str) -> Tuple[np.ndarray, np.ndarray]:
        """Tile-sized (cfg.sr.lr_patch_size) centre crop of one VAL tile, reflectance."""
        npz, _ = self.ds._cache_paths(sid)
        with np.load(npz) as h:
            lr_raw, hr_raw = h["lr"][self.ds.band_indices], h["hr"][self.ds.band_indices]
        nod, fill = self.cfg.dataset.nodata_value, float(self.cfg.dataset.nodata_fill)
        lr, _ = self.ds.to_reflectance(lr_raw, nod, fill)
        hr, _ = self.ds.to_reflectance(hr_raw, nod, fill)
        c = centre_crop_pair(lr, hr, lr_size=self.tile, scale=self.scale)
        return np.ascontiguousarray(c["lr"], np.float32), np.ascontiguousarray(c["hr"], np.float32)

    def draw(self, n: int, seed: int = SEED) -> List[Dict[str, Any]]:
        idx = np.random.default_rng(seed).choice(len(self.pool), size=n, replace=False)
        cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        out = []
        for i in idx:
            sid, (r, c) = self.pool[int(i)]
            if sid not in cache:
                cache[sid] = self.load_tile(sid)
            lr, hr = cache[sid]
            p = crop_pair(lr, hr, r, c, self.size, self.scale)
            out.append({"sample_id": sid, "origin": [int(r), int(c)],
                        "lr": np.ascontiguousarray(p["lr"], np.float32),
                        "hr": np.ascontiguousarray(p["hr"], np.float32)})
        return out


def cpu_load_percent() -> float | None:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_Processor | Measure-Object -Property "
                            "LoadPercentage -Average).Average"],
                           capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip())
    except Exception:  # noqa: BLE001 -- load is metadata; absence is recorded as None
        return None


def provenance(pred: Any) -> Dict[str, Any]:
    git = git_metadata(WORKTREE)
    return {"checkpoint_id": pred.info["checkpoint_id"], "checkpoint_path": str(A2_CKPT),
            "git_sha": git.get("commit"), "git_dirty": git.get("dirty"), "split": "val",
            "seed": SEED, "interim": False,
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}


# ------------------------------------------------------------ helpers
def sobel_map(img: np.ndarray) -> np.ndarray:
    """Per-pixel band-mean Sobel magnitude (project kernels, replicate padding)."""
    x = torch.from_numpy(np.ascontiguousarray(img, np.float32))[None]
    c = x.shape[1]
    k = torch.tensor([SOBEL_X, SOBEL_Y], dtype=x.dtype).unsqueeze(1).repeat(c, 1, 1, 1)
    g = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate"),
                                   k, groups=c).view(1, c, 2, *x.shape[-2:])
    return torch.sqrt(g[:, :, 0] ** 2 + g[:, :, 1] ** 2).mean(dim=1)[0].numpy()


def time_tta(pred: Any, lr: np.ndarray, repeats: int = 3) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    compute_trust(lr, pred, tta=0)  # warm-up
    for tta in (0, 4, 8):
        runs = [compute_trust(lr, pred, tta=tta).timings_ms for _ in range(repeats)]
        out[f"tta{tta}"] = {k: (float(np.median([r[k] for r in runs])) if runs[0][k] is not None
                                else None) for k in ("sr", "uncertainty", "total")}
    return out


def main() -> None:
    torch.set_num_threads(6)
    n = 64
    pred = TorchPredictor(A2_CKPT, threads=6, interim=False)
    val = ValPatches()
    patches = val.draw(n)

    unc, err, sob, cons = [], [], [], []
    spec_sr, spec_bic, spec_hr, hf = [], [], [], []
    t0 = time.perf_counter()
    for p in patches:
        res = compute_trust(p["lr"], pred, tta=8, hr=p["hr"])
        unc.append(res.unc.ravel())
        err.append(np.abs(res.sr - p["hr"]).mean(axis=0).ravel())
        sob.append(sobel_map(res.bicubic).ravel())
        cons.append(res.cons_l1.ravel())
        s = res.scalars
        spec_sr.append(s["spec_l1"]); spec_bic.append(s["spec_l1_bicubic"])
        spec_hr.append(s["spec_l1_hr"]); hf.append(s["hf_ratio_vs_bicubic"])
    elapsed = time.perf_counter() - t0
    unc_a, err_a, sob_a, cons_a = (np.concatenate(v) for v in (unc, err, sob, cons))

    from scipy.stats import spearmanr

    n_px = 200_000
    pick = np.random.default_rng(SEED).choice(unc_a.size, size=n_px, replace=False)
    rho_unc = float(spearmanr(unc_a[pick], err_a[pick]).statistic)
    rho_sob = float(spearmanr(sob_a[pick], err_a[pick]).statistic)
    rho_unc_sob = float(spearmanr(unc_a[pick], sob_a[pick]).statistic)

    load = cpu_load_percent()
    timing_lr, _ = val.load_tile(patches[0]["sample_id"])  # 128x128 LR
    timing = time_tta(pred, timing_lr)

    report = {
        "what": "P5 trust overlay display scales: TTA-8 uncertainty and training-operator "
                "consistency on VAL, A2 via TorchPredictor",
        "unc_display_max": float(np.percentile(unc_a, 99)),
        "cons_display_max": float(np.percentile(cons_a, 99)),
        "spearman_tta8_unc_vs_abs_err": rho_unc,
        "spearman_sobel_bicubic_vs_abs_err": rho_sob,
        "spearman_tta8_unc_vs_sobel_bicubic": rho_unc_sob,
        "spearman_n_pixels": n_px,
        "mean_spec_l1_sr": float(np.mean(spec_sr)),
        "mean_spec_l1_bicubic": float(np.mean(spec_bic)),
        "mean_spec_l1_hr": float(np.mean(spec_hr)),
        "mean_hf_ratio_vs_bicubic": float(np.mean(hf)),
        "units": {"unc": "reflectance (band-mean per-band std across TTA-8)",
                  "cons": "reflectance (band-mean |down4_area(SR) - LR|, training operator)",
                  "abs_err": "reflectance, band-mean |SR - HR|"},
        "n": n,
        "n_samples": n,
        "patch_pool": f"{len(val.ids)} VAL tiles x {len(val.origins)} grid patches "
                      f"(centre {val.tile}px LR tile, {val.size}px patches)",
        "patches": [{"sample_id": p["sample_id"], "origin_lr": p["origin"]} for p in patches],
        "calibration_seconds": elapsed,
        "timing_ms_per_image": {
            "lr_size": list(timing_lr.shape[1:]), "backend": pred.info["backend"],
            "threads": pred.info["threads"], "repeats": 3, "statistic": "median",
            "cpu_load_percent_before": load,
            "provisional": load is None or load >= 20,
            **timing},
        **provenance(pred),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "patches"}, indent=2))


if __name__ == "__main__":
    main()
