"""Prove that this machine can actually read the data. One command, no arguments.

Run this **first** in every fresh Kaggle session, before anything that costs GPU
time. It answers the only question that matters at that moment -- "can this
session see the ~3000 sample pairs, or is it about to train on nothing?" -- and
it answers it by opening real files, not by checking that a directory exists.

What it checks, in order, stopping at the first real failure:

1. **Path resolution.** Where does ``resolve_data_root`` land, where does
   ``resolve_cache_dir`` land, and is the Kaggle mount composed from config
   present? Every candidate path is printed whether or not it was chosen, so a
   wrong answer is diagnosable without re-running.
2. **The cache.** How many ``.npz`` sample files are actually there.
3. **The manifest.** Row count, and whether it agrees with the cache.
4. **The splits.** Row count per split, because a missing split file means the
   loader silently recomputes one, and a recomputed split is a different split.
5. **The split COVERS the catalog.** Present is not the same as current. MEASURED
   2026-09-09: the Day 3 `day3` job passed this guard and died seven seconds
   into training with "Split file covers 3000 samples but 1409 of the dataset's
   4409 samples are absent from it". The cache had been expanded locally and the
   split regenerated; the Kaggle dataset still carried the pre-expansion CSV.
   The guard checked the file existed and stopped there, so it certified a
   session that could not train. It now runs the trainer's OWN
   ``resolve_split_assignments`` -- not a reimplementation of it -- so the guard
   cannot pass while the loader would raise.
6. **Real samples.** Three random pairs are loaded and their shapes, dtypes, and
   **per-band reflectance ranges** printed. This is the part that catches a
   wrong reflectance divisor, a transposed channel axis, or truncated files --
   none of which a file count would notice.

Exit code 0 means the session can read the data. Any non-zero exit means do not
start training.

Examples:
    python scripts/verify_data_root.py
    python scripts/verify_data_root.py --samples 10
    python scripts/verify_data_root.py --smoke     # synthetic stub, no data needed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import (  # noqa: E402
    kaggle_mount_path,
    repo_root,
    resolve_cache_dir,
    resolve_data_root,
    resolve_output_path,
)
from src.utils.seed import seed_everything  # noqa: E402

OK = "  [OK]  "
BAD = "  [!!]  "
INFO = "         "


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove that this machine (local or Kaggle) can read the sample "
            "cache: resolve the paths, load the manifest, and open real "
            "samples."
        ),
    )
    add_standard_args(parser)
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="How many random samples to open and describe (default: 3).",
    )
    return parser.parse_args(argv)


def shallow_tree(root: Path, max_depth: int = 3, max_entries: int = 40) -> List[str]:
    """List a few levels of a directory, for diagnosing a missing mount.

    One level is not enough. MEASURED on Kaggle: when the expected mount was
    absent, ``/kaggle/input`` contained a single entry named ``datasets`` -- which
    says the attachment exists but is nested somewhere else, and nothing about
    where. Three levels is enough to reach ``<root>/datasets/<owner>/<slug>``
    while staying short enough to read in a log.

    Args:
        root: Directory to list.
        max_depth: How many levels below ``root`` to descend.
        max_entries: Stop after this many lines, so a mount holding thousands of
            files cannot bury the rest of the report.

    Returns:
        Indented ``"- name"`` lines, deepest paths last. Directories are marked
        with a trailing ``/``. Empty when ``root`` has no children.
    """
    lines: List[str] = []

    def walk(directory: Path, depth: int) -> None:
        if depth > max_depth or len(lines) >= max_entries:
            return
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if len(lines) >= max_entries:
                lines.append(f"{'  ' * (depth - 1)}... (listing truncated)")
                return
            is_dir = child.is_dir()
            lines.append(f"{'  ' * (depth - 1)}- {child.name}{'/' if is_dir else ''}")
            if is_dir:
                walk(child, depth + 1)

    walk(root, 1)
    return lines


def describe_paths(cfg: Any) -> dict:
    """Print every path the resolver considered, and what it chose.

    Args:
        cfg: The loaded config.

    Returns:
        ``{"data_root": Path|None, "cache_dir": Path|None, "mount": Path|None,
        "failure": str|None}``. Resolution failure is captured rather than
        raised so the remaining checks can still report what they see.
    """
    print("1. PATH RESOLUTION")
    print()
    print(f"{INFO}repository root:  {repo_root()}")

    mount = kaggle_mount_path(cfg)
    if mount is None:
        print(
            f"{INFO}Kaggle mount:     not configured "
            "(paths.kaggle_mount_root / paths.kaggle_dataset_dir)"
        )
    else:
        if mount.is_dir():
            entries = list(mount.iterdir())
            if entries:
                print(f"{OK}Kaggle mount:     {mount}  ({len(entries)} entries)")
            else:
                print(f"{BAD}Kaggle mount:     {mount}  EXISTS BUT IS EMPTY")
        else:
            on_kaggle = Path("/kaggle").is_dir()
            marker = BAD if on_kaggle else INFO
            suffix = (
                "  NOT PRESENT -- nothing is attached under this name"
                if on_kaggle
                else "  (not present -- normal off Kaggle)"
            )
            print(f"{marker}Kaggle mount:     {mount}{suffix}")
            # List what IS mounted. Without this the log says only that the
            # expected path is missing, which cannot distinguish "no dataset was
            # attached" from "a dataset was attached under a different name" --
            # and those have different fixes. MEASURED: a run that reached this
            # line cost a full session to diagnose because the log did not say
            # what was actually there.
            if on_kaggle and mount.parent.is_dir():
                listing = shallow_tree(mount.parent)
                if listing:
                    print(f"{INFO}  {mount.parent} actually contains:")
                    for line in listing:
                        print(f"{INFO}    {line}")
                    print(
                        f"{INFO}  If the cache is in there under another path, "
                        "point paths.kaggle_mount_root and"
                    )
                    print(f"{INFO}  paths.kaggle_dataset_dir at it.")
                else:
                    print(
                        f"{INFO}  {mount.parent} is EMPTY: this session has no "
                        "inputs attached at all."
                    )

    result: dict = {"mount": mount, "failure": None, "data_root": None, "cache_dir": None}

    try:
        data_root = resolve_data_root(cfg)
        result["data_root"] = data_root
        print(f"{OK}data root:        {data_root}")
    except (FileNotFoundError, NotADirectoryError, KeyError, ValueError) as exc:
        result["failure"] = str(exc)
        print(f"{BAD}data root:        COULD NOT BE RESOLVED")
        print()
        for line in str(exc).splitlines():
            print(f"{INFO}{line}")

    cache_dir = Path(resolve_cache_dir(cfg))
    result["cache_dir"] = cache_dir
    print(f"{OK}cache directory:  {cache_dir}")
    return result


def cache_subset_dir(cfg: Any, cache_dir: Path) -> Path:
    """The subset folder inside the cache, matching what the loader computes."""
    name = str(cfg.dataset.name)
    section = cfg.get(name) if hasattr(cfg, "get") else None
    subset = None if section is None else section.get("subset")
    return cache_dir / str(subset or name)


def check_cache(cfg: Any, cache_dir: Path) -> tuple:
    """Count the cached sample files.

    Returns:
        ``(subset_dir, npz_paths)``. ``npz_paths`` is empty when nothing is
        cached, which the caller treats as a failure.
    """
    print()
    print("2. SAMPLE CACHE")
    print()
    subset_dir = cache_subset_dir(cfg, cache_dir)
    print(f"{INFO}looking in: {subset_dir}")

    if not subset_dir.is_dir():
        print(f"{BAD}That directory does not exist.")
        print(f"{INFO}On Kaggle this means the dataset is not attached, or is")
        print(f"{INFO}attached under a name that differs from")
        print(f"{INFO}paths.kaggle_dataset_dir={cfg.paths.kaggle_dataset_dir!r}.")
        print(f"{INFO}Locally it means the cache has not been downloaded yet:")
        print(f"{INFO}  python scripts/prepare_data.py --config configs/base.yaml")
        return subset_dir, []

    npz_paths = sorted(subset_dir.rglob("*.npz"))
    sidecars = sorted(subset_dir.rglob("*.json"))
    if not npz_paths:
        print(f"{BAD}No .npz sample files found.")
        return subset_dir, []

    total = sum(p.stat().st_size for p in npz_paths)
    print(f"{OK}{len(npz_paths):,} .npz sample files  ({total / 1024**3:,.2f} GB)")
    print(f"{INFO}{len(sidecars):,} .json sidecars")
    if len(sidecars) != len(npz_paths):
        print(
            f"{BAD}Sidecar count does not match sample count -- an interrupted "
            "download or a partial upload."
        )
    return subset_dir, npz_paths


def check_manifest(cfg: Any, cached_count: int) -> Optional[List[dict]]:
    """Load the manifest and report its row count.

    Returns:
        The rows, or ``None`` when the manifest is absent. Absence is a warning
        rather than a hard failure: the cache is still readable, but the split
        provenance is not.
    """
    print()
    print("3. MANIFEST")
    print()
    manifest_dir = Path(resolve_output_path(cfg, "manifest_dir"))
    path = manifest_dir / f"manifest_{cfg.dataset.name}.csv"
    print(f"{INFO}looking for: {path}")

    if not path.is_file():
        print(f"{BAD}Not found.")
        mount = kaggle_mount_path(cfg)
        if mount is not None and (mount / path.name).is_file():
            print(f"{INFO}It IS present in the mounted dataset. Copy it across:")
            print(f"{INFO}  !mkdir -p outputs && cp {mount}/*.csv outputs/")
        else:
            print(f"{INFO}Generate it with:")
            print(f"{INFO}  python scripts/prepare_data.py --config configs/base.yaml")
        return None

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"{OK}{len(rows):,} rows")

    rejected = sum(1 for r in rows if str(r.get("rejected", "")).lower() == "true")
    errors = sum(1 for r in rows if r.get("validation_error"))
    print(f"{INFO}{rejected:,} rejected by policy, {errors:,} with a validation error")

    if cached_count and len(rows) != cached_count:
        print(
            f"{BAD}Manifest has {len(rows):,} rows but {cached_count:,} samples "
            "are cached -- these describe different sets of data."
        )
    return rows


def check_splits(cfg: Any) -> Optional[dict]:
    """Load the split CSV and report the per-split counts.

    Returns:
        ``{split_name: count}``, or ``None`` when the file is absent.
    """
    print()
    print("4. TRAIN / VAL / TEST SPLIT")
    print()
    manifest_dir = Path(resolve_output_path(cfg, "manifest_dir"))
    path = manifest_dir / str(cfg.splits.output_name).format(dataset=cfg.dataset.name)
    print(f"{INFO}looking for: {path}")

    if not path.is_file():
        print(f"{BAD}Not found.")
        mount = kaggle_mount_path(cfg)
        if mount is not None and (mount / path.name).is_file():
            print(f"{INFO}It IS present in the mounted dataset. Copy it across:")
            print(f"{INFO}  !mkdir -p outputs && cp {mount}/*.csv outputs/")
        else:
            print(f"{INFO}Generate it with:")
            print(f"{INFO}  python scripts/make_splits.py --config configs/base.yaml")
        print(
            f"{INFO}Without it the loader recomputes a split in-process. That is "
            "reproducible,"
        )
        print(
            f"{INFO}but it is not necessarily the split your existing results "
            "were measured on."
        )
        return None

    with path.open("r", newline="", encoding="utf-8") as handle:
        counts = Counter(row["split"] for row in csv.DictReader(handle))
    print(f"{OK}{sum(counts.values()):,} samples assigned")
    for name, count in sorted(counts.items()):
        print(f"{INFO}  {name:<8} {count:,}")
    return dict(counts)


def check_split_covers_catalog(cfg: Any, logger: Any) -> Optional[bool]:
    """Run the trainer's own split resolution and report whether it would raise.

    THE CHECK THAT WOULD HAVE SAVED THE DAY 3 SESSION. Every check above this
    one asks whether a file is THERE. This one asks whether it is CURRENT, which
    is a different question and the one that actually failed: a split CSV
    covering 3000 samples is perfectly readable, has the right columns, and
    reports sensible per-split counts, while the catalog it is supposed to
    describe has since grown to 4409. ``resolve_split_assignments`` refuses that
    -- correctly, because training on the overlap would silently change the
    val set the baseline was measured on -- and it refuses it AFTER the queue
    wait, the pip installs and the mount.

    Deliberately calls the loader's function rather than comparing counts here.
    A reimplementation would be a second opinion that can drift from the first,
    and the only useful guarantee is "the guard passes exactly when the trainer
    would get past this point".

    Cost: this builds the dataset, which loads the TACO catalog. MEASURED on
    Kaggle 2026-09-09: about two seconds. That is the price of the guard being
    predictive instead of decorative.

    Args:
        cfg: The loaded config.
        logger: Logger, passed through to the loader for its provenance line.

    Returns:
        True when the split file covers the catalog; False when it exists but
        does not; None when there is no split file at all, which the previous
        check already reported and which makes coverage undefined rather than
        failed.
    """
    print()
    print("5. DOES THE SPLIT COVER THE CATALOG?")
    print()

    from src.data.loader import SplitError, resolve_split_assignments
    from src.data.registry import get_dataset

    manifest_dir = Path(resolve_output_path(cfg, "manifest_dir"))
    split_path = manifest_dir / str(cfg.splits.output_name).format(
        dataset=cfg.dataset.name
    )
    if not split_path.is_file():
        print(f"{INFO}No split file, so coverage is undefined -- see section 4.")
        return None

    print(f"{INFO}building the dataset catalog (this reads the TACO index)...")
    try:
        dataset = get_dataset(cfg)
    except ImportError as exc:
        # tacoreader missing is a job-definition problem, not a data problem,
        # and it has its own failure mode elsewhere. Say which it is.
        print(f"{BAD}Cannot build the catalog: {exc}")
        print(f"{INFO}This is a missing dependency, not a bad split. Add it to")
        print(f"{INFO}pip_packages in configs/kaggle_jobs.yaml.")
        return False

    try:
        assignments, source = resolve_split_assignments(cfg, dataset, logger)
    except SplitError as exc:
        print(f"{BAD}The split does NOT cover the catalog.")
        for line in str(exc).splitlines():
            print(f"{INFO}  {line}")
        print()
        print(f"{INFO}This is the exact exception src/train.py raises, from the")
        print(f"{INFO}same function. A session started now dies seconds into the")
        print(f"{INFO}first dataset build, after the queue wait and the installs.")
        print(f"{INFO}Fix, in order:")
        print(f"{INFO}  python scripts/make_splits.py --config configs/base.yaml")
        print(f"{INFO}  python scripts/kaggle_upload.py version   # ship it")
        return False

    print(f"{OK}{len(assignments):,} samples resolved from {source}")
    print(f"{INFO}The split is READ, not recomputed, and it covers every")
    print(f"{INFO}catalog sample. src/train.py will get past this point.")
    return True


def describe_samples(cfg: Any, npz_paths: List[Path], count: int, seed: int) -> bool:
    """Open random samples and print shapes and per-band reflectance ranges.

    This is the check that cannot be faked by a directory listing. It converts
    raw digital numbers to reflectance exactly as the loader does -- mask nodata,
    then divide by ``cfg.dataset.reflectance_scale`` -- and prints the resulting
    range per band. A wrong divisor, a transposed channel axis, or a truncated
    file all show up here as obviously wrong numbers.

    Args:
        cfg: The loaded config.
        npz_paths: Candidate sample files.
        count: How many to open.
        seed: Seed for the random choice, so the check is reproducible.

    Returns:
        True when every opened sample looked physically plausible.
    """
    print()
    print(f"6. REAL SAMPLES ({min(count, len(npz_paths))} chosen at random)")

    bands = [str(b) for b in cfg.dataset.bands]
    scale = float(cfg.dataset.reflectance_scale)
    nodata = int(cfg.dataset.nodata_value)
    valid_max = float(cfg.dataset.reflectance_valid_max)

    rng = np.random.default_rng(seed)
    chosen = [
        npz_paths[i]
        for i in rng.choice(
            len(npz_paths), size=min(count, len(npz_paths)), replace=False
        )
    ]

    all_good = True
    for path in chosen:
        print()
        print(f"{INFO}{path.name}")
        try:
            with np.load(path) as payload:
                arrays = {key: payload[key] for key in payload.files}
        except (OSError, ValueError) as exc:
            print(f"{BAD}Could not read it: {exc}")
            print(f"{INFO}A truncated file -- the upload or download was cut short.")
            all_good = False
            continue

        for key in sorted(arrays):
            array = arrays[key]
            print(f"{INFO}  {key}: shape {tuple(array.shape)}, dtype {array.dtype}")
            if array.ndim != 3 or array.shape[0] != len(bands):
                print(
                    f"{BAD}  Expected (C, H, W) with C={len(bands)} to match "
                    f"cfg.dataset.bands. Got {tuple(array.shape)}."
                )
                all_good = False
                continue

            masked = np.where(array == nodata, np.nan, array.astype(np.float32))
            reflectance = masked / scale
            nodata_pixels = int(np.isnan(reflectance).sum())

            for index, band in enumerate(bands):
                plane = reflectance[index]
                finite = plane[np.isfinite(plane)]
                if finite.size == 0:
                    print(f"{BAD}    {band}: entirely nodata")
                    all_good = False
                    continue
                low, high, mean = finite.min(), finite.max(), finite.mean()
                flag = ""
                if high > valid_max:
                    flag = f"  <-- ABOVE cfg.dataset.reflectance_valid_max={valid_max:g}"
                    all_good = False
                elif high > 1.0:
                    flag = "  (bright target; above 1.0 is legal and unclipped)"
                print(
                    f"{INFO}    {band}: reflectance "
                    f"[{low:.4f}, {high:.4f}]  mean {mean:.4f}{flag}"
                )
            if nodata_pixels:
                fraction = nodata_pixels / reflectance.size
                print(
                    f"{INFO}    {nodata_pixels:,} nodata pixels "
                    f"({fraction:.2%}), masked before scaling"
                )

    print()
    print(
        f"{INFO}Reflectance is digital_number / {scale:g} with {nodata} masked "
        "first."
    )
    print(
        f"{INFO}Band order is {', '.join(bands)} -- red first, NIR last. Typical "
        "vegetation"
    )
    print(f"{INFO}reads about R 0.06, G 0.06, B 0.03, NIR 0.25.")
    return all_good


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("verify_data_root", log_file=cfg.paths.log_file)
    seed = seed_everything(cfg.seed)

    print()
    print("=" * 78)
    print("  CAN THIS SESSION READ THE DATA?")
    print("=" * 78)
    print()
    print(f"{INFO}dataset: {cfg.dataset.name}")
    print(f"{INFO}config:  {args.config}")
    print()

    resolved = describe_paths(cfg)
    subset_dir, npz_paths = check_cache(cfg, resolved["cache_dir"])
    manifest = check_manifest(cfg, len(npz_paths))
    splits = check_splits(cfg)
    coverage_ok = check_split_covers_catalog(cfg, logger)

    samples_ok = False
    if npz_paths:
        samples_ok = describe_samples(cfg, npz_paths, int(args.samples), seed)

    # The manifest and the split CSV are REQUIRED, not advisory, and this is the
    # guard the generated Kaggle notebook runs before every job. They used to be
    # reported and shrugged off, which is how a job could run with the imagery
    # present and the split file absent -- and a missing split file does not
    # fail, it makes the loader recompute a split in-process. Adjacent NAIP
    # tiles overlap, so a recomputed split is not the geographic split the
    # baseline was measured on, and every number downstream is then quietly
    # incomparable. A readable cache without them is not a session that may
    # start training.
    #
    # `coverage_ok is not False` rather than `coverage_ok is True`: None means
    # there was no split file to check, which section 4 has already failed the
    # run for. Folding None into the failure here would report the same problem
    # twice and hide which check actually caught it.
    supporting_ok = (
        manifest is not None and splits is not None and coverage_ok is not False
    )

    print()
    print("=" * 78)
    if npz_paths and samples_ok and supporting_ok:
        print("  VERDICT: the data is readable. Safe to start training.")
        print("=" * 78)
        print()
        print(f"{INFO}{len(npz_paths):,} samples at {subset_dir}")
        print(f"{INFO}Manifest and split CSV both present: the split is READ,")
        print(f"{INFO}not recomputed, and it covers every catalog sample.")
        print()
        print("WHAT HAPPENS NEXT")
        print("  1. Run the baseline to confirm the numbers reproduce here:")
        print("     python scripts/run_baseline.py --config configs/base.yaml "
              "--baseline bicubic")
        print("  2. Then start training.")
        print()
        logger.info("Data root verified: %d samples at %s", len(npz_paths), subset_dir)
        return 0

    print("  VERDICT: THE DATA IS NOT READABLE. Do not start training.")
    print("=" * 78)
    print()
    if not npz_paths:
        print(f"{INFO}No samples were found. The numbered sections above name")
        print(f"{INFO}every path that was checked; fix the first one marked [!!].")
    elif not samples_ok:
        print(f"{INFO}Samples were found but did not look right. See section 5.")
    else:
        # The pixels are fine; what is missing is the provenance. Said
        # separately because the fix is completely different -- nothing is wrong
        # with the mount, a file simply was not staged out of it.
        absent = [
            name
            for name, value in (("manifest", manifest), ("split CSV", splits))
            if value is None
        ]
        print(f"{INFO}The samples are readable, but the {' and '.join(absent)} "
              f"{'is' if len(absent) == 1 else 'are'} missing.")
        print(f"{INFO}That is a FAILURE, not a warning: without the split CSV the")
        print(f"{INFO}loader recomputes a split in-process. It is reproducible, but")
        print(f"{INFO}it is not the geographic split the baseline was measured on,")
        print(f"{INFO}so any number produced from it is quietly incomparable.")
    print()
    print("WHAT HAPPENS NEXT")
    print("  1. On Kaggle: sidebar -> '+ Add Input' -> attach the dataset named")
    print(f"     {cfg.paths.kaggle_dataset_dir!r}, then re-run this script.")
    print("     If the samples were readable and only the CSVs were missing, the")
    print("     dataset was uploaded without them -- re-run make_splits.py, then")
    print("     kaggle_upload.py stage and push.")
    print("  2. Locally: python scripts/prepare_data.py --config configs/base.yaml")
    print("     then:    python scripts/make_splits.py --config configs/base.yaml")
    print("  3. If the dataset has never been uploaded:")
    print("     python scripts/kaggle_upload.py survey")
    print()
    logger.error("Data root verification FAILED (cache=%s)", subset_dir)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
