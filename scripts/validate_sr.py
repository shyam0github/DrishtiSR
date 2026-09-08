"""Validate a super-resolved GeoTIFF against the input it was derived from.

``drishtisr.infer.tiled`` reports what it wrote from its own variables. This
script re-opens the file with rasterio and checks the claim against the source
raster, which is the only version of the answer that survives the process
exiting. Six checks, each pass/fail on its own line:

1. **Shape** -- exactly ``scale`` times the input in both axes.
2. **Pixel size** -- ``cfg.dataset.hr_gsd_m`` (2.5 m), and consistent with the
   input's own pixel size divided by ``scale``.
3. **CRS** -- identical to the input's.
4. **Origin** -- the top-left corner is bit-identical to the input's. Only the
   pixel size may change; a shifted origin means the SR raster describes
   different ground while looking perfectly well-formed in a viewer.
5. **No NaN** -- and, because uint16 cannot represent a NaN and a "no NaN" pass
   on an integer raster is vacuous, the count of pixels pinned at the storage
   ceiling or floor is reported alongside it. That is the failure mode a uint16
   write actually has.
6. **Value range** -- within ``cfg.delhi.expect_dn_min`` .. ``expect_dn_max``
   digital numbers, the same bounds the fetch script reports against.

Nothing here clips, repairs or rewrites the raster. A failure is reported and
exits non-zero; deciding what to do about it is a person's job.

Exit codes: ``0`` all checks passed, ``2`` at least one failed, ``1`` the run
itself broke.

Examples:
    # Validate the fabricated pair from `smoke_view.py --smoke`, seconds, CPU.
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/validate_sr.py --smoke

    # The real thing.
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/validate_sr.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

LOGGER = get_logger("validate_sr")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-open a super-resolved GeoTIFF with rasterio and check its "
            "geometry, CRS, origin and value range against the input raster."
        ),
    )
    add_standard_args(parser)
    return parser.parse_args(argv)


class Checks:
    """Collects pass/fail results so every check runs before anything exits.

    A bare ``assert`` would stop at the first failure and hide the rest, which
    turns one debugging pass into several. Every check is evaluated; the exit
    code is decided once at the end.
    """

    def __init__(self) -> None:
        self.rows: List[Tuple[bool, str, str]] = []

    def add(self, ok: bool, name: str, detail: str) -> bool:
        self.rows.append((bool(ok), name, detail))
        LOGGER.info("%-14s %s -- %s", "PASS" if ok else "FAIL", name, detail)
        return bool(ok)

    @property
    def failed(self) -> List[Tuple[bool, str, str]]:
        return [r for r in self.rows if not r[0]]


def band_stats(path: Path) -> dict:
    """Scan every band block-wise and return min, max, NaN and saturation counts.

    Reads by internal block rather than whole-array so a 4-band 4972x5176
    raster is never fully resident. Returns raw DIGITAL NUMBERS as stored --
    no reflectance conversion, because the bounds this feeds are DN bounds.

    Returns:
        ``{"min", "max", "nan", "n_low", "n_high", "count", "dtype"}`` with
        counts as ints and min/max as floats.
    """
    import rasterio

    lo, hi = np.inf, -np.inf
    n_nan = n_low = n_high = total = 0
    with rasterio.open(path) as src:
        dt = np.dtype(src.dtypes[0])
        is_int = np.issubdtype(dt, np.integer)
        floor, ceil = (
            (np.iinfo(dt).min, np.iinfo(dt).max) if is_int else (None, None)
        )
        for _, window in src.block_windows(1):
            arr = src.read(window=window)
            total += arr.size
            if np.issubdtype(arr.dtype, np.floating):
                n_nan += int(np.count_nonzero(~np.isfinite(arr)))
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    lo = min(lo, float(finite.min()))
                    hi = max(hi, float(finite.max()))
            else:
                lo = min(lo, float(arr.min()))
                hi = max(hi, float(arr.max()))
                n_low += int(np.count_nonzero(arr == floor))
                n_high += int(np.count_nonzero(arr == ceil))
    return {
        "min": lo, "max": hi, "nan": n_nan, "n_low": n_low,
        "n_high": n_high, "count": total, "dtype": str(dt),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)
    seed_everything(cfg.seed)

    import rasterio

    vs = cfg.validate_sr
    root = repo_root()
    lr_path = root / str(vs.input)
    sr_path = root / str(vs.sr)
    scale = int(vs.scale)
    expect_px = float(cfg.dataset.hr_gsd_m)
    px_tol = float(vs.pixel_size_tol_m)
    org_tol = float(vs.origin_tol_m)
    dn_min = float(cfg.delhi.expect_dn_min)
    dn_max = float(cfg.delhi.expect_dn_max)

    for label, path in (("input", lr_path), ("sr", sr_path)):
        if not path.is_file():
            raise FileNotFoundError(
                f"validate_sr.{label} does not exist: {path}. Produce the SR "
                "raster with `python -m drishtisr.infer.tiled` first, or run "
                "--smoke against the fabricated pair."
            )

    with rasterio.open(lr_path) as s:
        lr = {"h": s.height, "w": s.width, "crs": s.crs, "tf": s.transform,
              "res": s.res, "count": s.count}
    with rasterio.open(sr_path) as s:
        sr = {"h": s.height, "w": s.width, "crs": s.crs, "tf": s.transform,
              "res": s.res, "count": s.count, "dtype": s.dtypes[0],
              "nodata": s.nodata}

    LOGGER.info(
        "input %s: %d x %d x %d @ %g m, %s",
        lr_path.name, lr["count"], lr["h"], lr["w"], lr["res"][0], lr["crs"],
    )
    LOGGER.info(
        "sr    %s: %d x %d x %d @ %g m, %s, %s",
        sr_path.name, sr["count"], sr["h"], sr["w"], sr["res"][0],
        sr["crs"], sr["dtype"],
    )

    c = Checks()

    want_h, want_w = lr["h"] * scale, lr["w"] * scale
    c.add(
        (sr["h"], sr["w"]) == (want_h, want_w),
        "shape",
        f"{sr['h']}x{sr['w']} vs required {want_h}x{want_w} "
        f"({lr['h']}x{lr['w']} x{scale})",
    )
    c.add(
        sr["count"] == lr["count"], "band count",
        f"{sr['count']} vs input {lr['count']}",
    )

    px_x, px_y = abs(sr["tf"].a), abs(sr["tf"].e)
    c.add(
        abs(px_x - expect_px) <= px_tol and abs(px_y - expect_px) <= px_tol,
        "pixel size",
        f"{px_x:.10g} x {px_y:.10g} m vs dataset.hr_gsd_m {expect_px:g} "
        f"(tol {px_tol:g})",
    )
    derived = abs(lr["tf"].a) / scale
    c.add(
        abs(px_x - derived) <= px_tol,
        "pixel size vs input",
        f"{px_x:.10g} m vs input {abs(lr['tf'].a):g} / {scale} = {derived:.10g}",
    )

    c.add(sr["crs"] == lr["crs"], "CRS", f"{sr['crs']} vs input {lr['crs']}")

    dx, dy = abs(sr["tf"].c - lr["tf"].c), abs(sr["tf"].f - lr["tf"].f)
    c.add(
        dx <= org_tol and dy <= org_tol,
        "origin",
        f"({sr['tf'].c:.6f}, {sr['tf'].f:.6f}) vs input "
        f"({lr['tf'].c:.6f}, {lr['tf'].f:.6f}); delta ({dx:g}, {dy:g}) m",
    )
    # Rotation/skew terms: a non-zero b or d would mean the SR grid is not
    # axis-aligned with the input's, which no amount of matching origin and
    # pixel size would reveal.
    c.add(
        sr["tf"].b == 0.0 and sr["tf"].d == 0.0,
        "no rotation",
        f"skew terms b={sr['tf'].b:g}, d={sr['tf'].d:g}",
    )

    st = band_stats(sr_path)
    c.add(
        st["nan"] == 0, "no NaN",
        f"{st['nan']} non-finite of {st['count']} samples"
        + (
            f" (dtype {st['dtype']} cannot represent NaN; see saturation)"
            if not str(st["dtype"]).startswith("float") else ""
        ),
    )

    sat = (st["n_low"] + st["n_high"]) / max(1, st["count"])
    c.add(
        sat <= float(vs.saturation_frac_max), "not saturated",
        f"{st['n_low']} at dtype floor + {st['n_high']} at ceiling "
        f"= {100.0 * sat:.4f}% (max {100.0 * float(vs.saturation_frac_max):.4f}%)",
    )

    c.add(
        st["min"] >= dn_min and st["max"] <= dn_max, "value range",
        f"DN [{st['min']:g}, {st['max']:g}] vs expected "
        f"[{dn_min:g}, {dn_max:g}] "
        f"(reflectance [{st['min'] / float(cfg.dataset.reflectance_scale):.4f}, "
        f"{st['max'] / float(cfg.dataset.reflectance_scale):.4f}])",
    )

    n_fail = len(c.failed)
    if n_fail:
        LOGGER.error(
            "%d of %d checks FAILED: %s", n_fail, len(c.rows),
            ", ".join(r[1] for r in c.failed),
        )
        return 2
    LOGGER.info("all %d checks passed for %s", len(c.rows), sr_path.name)
    if args.smoke:
        LOGGER.info(
            "--smoke validated a fabricated raster pair; the verdict means "
            "the checks execute, not that any model output is correct."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
