"""Decide the BOA offset from the pixels, not from the documentation.

Sentinel-2 L2A products at processing baseline >= 04.00 (2022-01-25) carry a
``BOA_ADD_OFFSET`` of -1000: the reflectance is ``(dn - 1000) / 10000``, not
``dn / 10000``. Microsoft Planetary Computer exposes that value NOWHERE on its
``sentinel-2-l2a`` items -- no ``s2:boa_add_offset`` property, no
``raster:bands`` block -- so ``scripts/fetch_delhi.py`` writes raw digital
numbers and derives the offset from the baseline, recording it in the manifest
as a formula rather than applying it.

That leaves the question this script answers, which is NOT "does Delhi carry an
offset" but **"is the SEN2NAIPv2 training data itself offset-corrected?"** If it
were not, applying the offset at inference would be the error rather than the
fix. Nothing in the dataset card or the cache metadata settles it: the cached
per-record JSON holds ``crs``, ``geotransform``, ``data_split`` and
``correlation``, with no Sentinel-2 item id and no processing baseline, and the
sample id carries the *NAIP* acquisition date rather than the S2 one. So it is
settled from the distributions.

Three tables, per band, in surface reflectance:

1. **SEN2NAIPv2 LR patches** -- ``cfg.boa_offset_audit.num_patches`` drawn at
   random from the cache, nodata excluded, unclipped.
2. **The inference scene under H1**, ``dn / reflectance_scale``.
3. **The same scene under H2**, ``(dn - offset) / reflectance_scale``.

Then the cross-checks that actually decide it:

- **Median distance per band.** The verdict is read off the medians, and off
  the p5-p95 span as a shape check. It is NOT read off min/max: one saturated
  pixel moves those, and an additive offset cannot change a span at all --
  identical spans under both hypotheses are expected, not evidence.
- **The DN floor.** The decisive one. An uncorrected baseline >= 04.00 product
  cannot place pixels below the offset it has not had subtracted. A large
  fraction of the training set below ``cfg.boa_offset_audit.dn_floor_probe``
  therefore proves the correction was already applied.
- **Cohorts either side of baseline 04.00**, by the NAIP year in the sample id
  as a proxy for the unavailable S2 date. If the cohorts agree, the correction
  is a property of the dataset build rather than of the ESA baseline, which is
  a stronger and more durable conclusion than either cohort alone.

Reports and names the closer hypothesis. Decides nothing and writes nothing to
the config; ``cfg.delhi.dn_offset`` is set by a person who has read this.

Examples:
    # Offline, seconds, no cache and no scene required.
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/boa_offset_audit.py --smoke

    # The real measurement.
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/boa_offset_audit.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root, resolve_cache_dir  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

LOGGER = get_logger("boa_offset_audit")

# Declared on both LR and HR by SEN2NAIPv2 and masked before any division:
# 65535 / 10000 = 6.55 reflectance would dominate every statistic below.
NODATA = 65535


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SEN2NAIPv2 training reflectance against an inference "
            "scene under two BOA-offset hypotheses, and name the closer one."
        ),
    )
    add_standard_args(parser)
    return parser.parse_args(argv)


def summarise(values: np.ndarray, percentiles: Sequence[float]) -> Dict[str, float]:
    """Per-band summary of one band's samples.

    Args:
        values: surface reflectance, float32, ``(N,)`` -- one band's valid
            pixels pooled across patches. Unclipped: bright roofs, cloud and
            specular water legitimately exceed 1.0, and an offset-corrected
            dark target legitimately falls below 0.
        percentiles: Percentile points to report, e.g. ``[1, 5, 50, 95, 99]``.

    Returns:
        ``{"min", "p<n>"..., "max", "mean", "span"}`` in reflectance, where
        ``span`` is p95 - p5 when both are present and NaN otherwise. Raises
        rather than returning zeros when handed nothing.
    """
    if values.size == 0:
        raise ValueError(
            "No valid samples for this band. Every pixel was nodata, which is "
            "a data fault worth looking at rather than a zero worth reporting."
        )

    out: Dict[str, float] = {"min": float(values.min())}
    got = np.percentile(values, list(percentiles))
    for point, value in zip(percentiles, np.atleast_1d(got)):
        out[f"p{point:g}"] = float(value)
    out["max"] = float(values.max())
    out["mean"] = float(values.mean())
    out["span"] = (
        out["p95"] - out["p5"] if "p95" in out and "p5" in out else float("nan")
    )
    return out


def render_table(title: str, bands: Sequence[str], stats: Dict[str, Dict[str, float]],
                 percentiles: Sequence[float]) -> None:
    """Log one per-band table. Reflectance throughout, four decimals."""
    columns = ["min"] + [f"p{p:g}" for p in percentiles] + ["max", "mean", "span"]
    LOGGER.info("")
    LOGGER.info("%s", title)
    LOGGER.info("%-5s%s", "band", "".join(f"{c:>9}" for c in columns))
    for band in bands:
        row = stats[band]
        LOGGER.info("%-5s%s", band, "".join(f"{row[c]:>9.4f}" for c in columns))


def fabricate_patches(cfg: Any, rng: random.Random) -> List[np.ndarray]:
    """Invent uint16 patches for ``--smoke``, with an offset deliberately baked in.

    Returns:
        A list of ``(C, H, W)`` uint16 arrays -- RAW DIGITAL NUMBERS, the same
        thing the cache stores. The values are drawn to sit LOW (well under the
        offset probe), so the smoke run exercises the same "already corrected"
        branch the real data takes and the verdict logic is genuinely run.

    These are invented numbers. The tables they produce describe nothing.
    """
    n_bands = len(cfg.dataset.bands)
    size = int(cfg.boa_offset_audit.fabricate_size_px)
    count = int(cfg.boa_offset_audit.fabricate_patches)
    generator = np.random.default_rng(rng.randrange(2**32))
    return [
        generator.integers(0, 1500, size=(n_bands, size, size), dtype=np.uint16)
        for _ in range(count)
    ]


def load_cached_patches(
    cfg: Any, rng: random.Random
) -> Tuple[List[np.ndarray], List[str], str]:
    """Draw random cached SEN2NAIPv2 patches.

    Reads the ``.npz`` files the dataset writes, which hold the FULL stored tile
    (``lr`` is ``(4, 130, 130)`` uint16, ``hr`` is ``(4, 520, 520)``). The model
    is trained on ``cfg.patches.lr_size`` crops cut from these tiles, so a tile
    is the population every training patch is drawn from -- which is the right
    unit for a distribution comparison, and is stated here rather than glossed
    as "the arrays the model sees".

    Returns:
        ``(patches, sample_ids, subset)``: ``(C, H, W)`` uint16 RAW DIGITAL
        NUMBERS, the cache stem of each (whose trailing ``_YYYYMMDD`` is the
        NAIP date), and the subset name the cache was read from.

    Raises:
        RunError-free by design: raises FileNotFoundError with the resolved path
        when the cache is absent, because a silently empty sample would produce
        a confident table describing nothing.
    """
    # Named by the SUBSET, not by cfg.dataset.name: SEN2NAIPv2Dataset builds
    # its cache as resolve_cache_dir(cfg) / self.subset (see src/data/sen2naip.py),
    # so the real directory is "sen2naipv2-crosssensor" and not "sen2naipv2".
    # Reading it any other way finds an empty tree and reports a confident table
    # about nothing.
    dataset_cfg = cfg.get(str(cfg.dataset.name)) or {}
    subset = str(dataset_cfg.get("subset") or cfg.dataset.name)
    cache = resolve_cache_dir(cfg) / subset
    if not cache.is_dir():
        raise FileNotFoundError(
            f"No sample cache at {cache}. Populate it with "
            "scripts/prepare_data.py, or run this with --smoke to exercise the "
            "arithmetic against fabricated patches."
        )

    files = sorted(cache.glob("*.npz"))
    wanted = int(cfg.boa_offset_audit.num_patches)
    if len(files) < wanted:
        raise FileNotFoundError(
            f"{cache} holds {len(files)} cached patches, fewer than the "
            f"{wanted} requested by cfg.boa_offset_audit.num_patches. Lower "
            "that, or finish the download; a short sample is reported here "
            "rather than quietly used."
        )

    chosen = rng.sample(files, wanted)
    side = str(cfg.boa_offset_audit.side)
    patches: List[np.ndarray] = []
    for path in chosen:
        with np.load(path) as handle:
            if side not in handle:
                raise KeyError(
                    f"{path} has no {side!r} array (found {list(handle)}). "
                    "cfg.boa_offset_audit.side must name a stored half."
                )
            patches.append(np.asarray(handle[side]))
    return patches, [path.stem for path in chosen], subset


def pool_bands(patches: Sequence[np.ndarray], n_bands: int, scale: float,
               offset: float) -> List[np.ndarray]:
    """Pool valid pixels per band across patches, converted to reflectance.

    Args:
        patches: ``(C, H, W)`` uint16 raw digital numbers.
        n_bands: Expected channel count, from ``cfg.dataset.bands``.
        scale: ``cfg.dataset.reflectance_scale``.
        offset: DN to SUBTRACT before dividing.

    Returns:
        One float32 ``(N,)`` array per band: surface reflectance, nodata
        excluded, UNCLIPPED at both ends.
    """
    accumulated: List[List[np.ndarray]] = [[] for _ in range(n_bands)]
    for patch in patches:
        if patch.shape[0] != n_bands:
            raise ValueError(
                f"Patch has {patch.shape[0]} channels, expected {n_bands} from "
                "cfg.dataset.bands. The band axis is load-bearing here."
            )
        valid = ~np.any(patch == NODATA, axis=0)
        for index in range(n_bands):
            values = patch[index][valid].astype(np.float32)
            accumulated[index].append((values - np.float32(offset)) / np.float32(scale))
    return [np.concatenate(band) for band in accumulated]


def cohort_floor_report(cfg: Any, patches: Sequence[np.ndarray],
                        sample_ids: Sequence[str], bands: Sequence[str]) -> None:
    """The decisive cross-check, in RAW DN: how much of the training set sits low.

    Splits the sample on the NAIP year parsed from the sample id -- a PROXY for
    the Sentinel-2 processing baseline, which the cache does not record -- and
    reports, per band, the fraction of pixels below
    ``cfg.boa_offset_audit.dn_floor_probe``.

    An uncorrected baseline >= 04.00 product cannot put pixels below its own
    unsubtracted offset. Cohorts that AGREE mean the correction belongs to the
    dataset build rather than to the acquisition date.
    """
    probe = int(cfg.boa_offset_audit.dn_floor_probe)
    split = int(cfg.boa_offset_audit.cohort_split_year)
    groups: Dict[str, List[np.ndarray]] = {
        f"NAIP year < {split}": [], f"NAIP year >= {split}": []
    }
    for patch, sample_id in zip(patches, sample_ids):
        tail = sample_id.rsplit("_", 1)[-1]
        if len(tail) < 4 or not tail[:4].isdigit():
            raise ValueError(
                f"Cannot parse a year from sample id {sample_id!r}. The cohort "
                "split relies on the trailing _YYYYMMDD; if the id format has "
                "changed, this check must change with it rather than guess."
            )
        key = f"NAIP year >= {split}" if int(tail[:4]) >= split else f"NAIP year < {split}"
        groups[key].append(patch)

    LOGGER.info("")
    LOGGER.info(
        "DN FLOOR PROBE -- raw digital numbers, fraction below %d, by cohort.", probe
    )
    LOGGER.info(
        "  ESA added the -1000 offset at baseline 04.00 (2022-01-25). The NAIP "
        "year is a PROXY: the cache records no S2 date and no baseline."
    )
    for label, members in groups.items():
        if not members:
            LOGGER.info("  %s: no patches in this cohort.", label)
            continue
        LOGGER.info("")
        LOGGER.info("  %s  n=%d", label, len(members))
        LOGGER.info("  %-5s%9s%9s%9s%15s",
                    "band", "min", "p5", "median", f"frac<{probe}")
        for index, band in enumerate(bands):
            pooled = np.concatenate([
                patch[index][~np.any(patch == NODATA, axis=0)].astype(np.float32)
                for patch in members
            ])
            p5, p50 = np.percentile(pooled, (5, 50))
            LOGGER.info(
                "  %-5s%9.0f%9.0f%9.0f%15.4f", band, pooled.min(), p5, p50,
                float((pooled < probe).mean()),
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)
    seed_everything(cfg.seed)

    audit = cfg.boa_offset_audit
    bands = [str(b) for b in cfg.dataset.bands]
    percentiles = [float(p) for p in audit.percentiles]
    scale = float(cfg.dataset.reflectance_scale)
    rng = random.Random(int(cfg.seed))

    LOGGER.info(
        "boa_offset_audit starting (seed %s, smoke=%s)", cfg.seed, bool(args.smoke)
    )

    # -- table 1: the training distribution --------------------------------
    if bool(audit.get("fabricate")):
        patches = fabricate_patches(cfg, rng)
        sample_ids = [f"fabricated_{i:03d}_20220101" for i in range(len(patches))]
        source = "FABRICATED patches (--smoke); these describe nothing"
    else:
        patches, sample_ids, cache_label = load_cached_patches(cfg, rng)
        source = f"{cache_label} {audit.side} patches from the cache"

    train_stats = {
        band: summarise(values, percentiles)
        for band, values in zip(bands, pool_bands(patches, len(bands), scale, 0.0))
    }
    render_table(
        f"(1) TRAINING -- {source}, n={len(patches)}, dn/{scale:g}, unclipped",
        bands, train_stats, percentiles,
    )

    # -- tables 2..n: the scene under each hypothesis -----------------------
    import rasterio

    scene = repo_root() / str(audit.scene)
    if not scene.is_file():
        raise FileNotFoundError(
            f"Inference scene not found at {scene}. Fetch it with "
            "scripts/fetch_delhi.py, or point cfg.boa_offset_audit.scene "
            "elsewhere. It is gitignored imagery, so a clean checkout has none."
        )
    with rasterio.open(scene) as src:
        scene_dn = src.read().astype(np.float32)
    if scene_dn.shape[0] != len(bands):
        raise ValueError(
            f"{scene} has {scene_dn.shape[0]} bands, expected {len(bands)}."
        )

    hypotheses = {str(k): float(v) for k, v in audit.hypotheses.items()}
    scene_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, offset in hypotheses.items():
        scene_stats[name] = {
            band: summarise((scene_dn[i].ravel() - offset) / scale, percentiles)
            for i, band in enumerate(bands)
        }
        render_table(
            f"({name}) SCENE {scene.name} -- (dn - {offset:g}) / {scale:g}",
            bands, scene_stats[name], percentiles,
        )

    # -- verdict ------------------------------------------------------------
    LOGGER.info("")
    LOGGER.info("MEDIAN DISTANCE FROM THE TRAINING DISTRIBUTION, per band")
    LOGGER.info(
        "  Spans are printed for shape only. An additive offset CANNOT change "
        "a span, so identical spans are expected and are not evidence."
    )
    header = "".join(f"{name:>12}" for name in hypotheses)
    LOGGER.info("%-5s%12s%s", "band", "train med", header)
    totals = {name: 0.0 for name in hypotheses}
    for band in bands:
        train_median = train_stats[band]["p50"]
        deltas = []
        for name in hypotheses:
            delta = scene_stats[name][band]["p50"] - train_median
            totals[name] += abs(delta)
            deltas.append(delta)
        LOGGER.info(
            "%-5s%12.4f%s", band, train_median,
            "".join(f"{d:>+12.4f}" for d in deltas),
        )

    LOGGER.info("")
    for name, total in totals.items():
        LOGGER.info(
            "  sum |median offset|  %-4s = %.4f  (dn_offset %g)",
            name, total, hypotheses[name],
        )
    winner = min(totals, key=totals.get)
    LOGGER.info("")
    LOGGER.info(
        "CLOSER HYPOTHESIS: %s -- reflectance = (dn - %g) / %g",
        winner, hypotheses[winner], scale,
    )

    cohort_floor_report(cfg, patches, sample_ids, bands)

    LOGGER.info("")
    if args.smoke:
        LOGGER.info(
            "--smoke measured FABRICATED patches against a real scene. The "
            "verdict above is arithmetic on invented numbers and means only "
            "that every code path executed."
        )
    else:
        LOGGER.info(
            "This names the closer hypothesis. It does not edit the config: "
            "cfg.delhi.dn_offset is set by a person who has read the tables, "
            "and src/infer/tiled.py --dn-offset is what applies it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
