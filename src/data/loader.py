"""DataLoader construction: scene-level splits, patch datasets, seeded workers.

Three properties this module is responsible for, in order of how expensive they
are to get wrong:

1. **The split is by scene, never by patch.** Patches from one tile -- and tiles
   from one NAIP quarter-quad, and tiles whose centroids are within
   ``cfg.splits.min_separation_km`` -- all land in the same split. Splitting at
   patch level would put neighbouring pixels of the same scene in both train and
   validation, and super-resolution is the task where that leaks hardest: the
   thing being predicted is exactly the high-frequency detail the model would
   have memorised. The split assignment therefore happens over *samples*, before
   any patch exists, and patch indices are built inside each split afterwards.

2. **It is deterministic.** The assignment comes from the CSV written by
   ``scripts/make_splits.py`` when that file exists, and otherwise is recomputed
   in-process by :func:`src.data.splits.geographic_split` with the same seed and
   the same rules. Two calls to :func:`build_dataloaders` on the same config give
   the same split, the same validation patch order, and the same training
   sequence.

3. **Worker seeding is reproducible.** Each worker's NumPy/Python/torch RNGs are
   seeded from ``cfg.seed``, the worker id, and the epoch, so a run can be
   reproduced exactly and two workers never draw the same "random" crops.

The dataset is not fully downloaded yet. ``cfg.loader.cached_only`` restricts the
loaders to samples already in the local cache so every downstream stage can be
exercised on a partial mirror; the number of samples skipped is logged loudly on
every build, never silently absorbed.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.patches import (
    PatchFilter,
    centre_crop_pair,
    crop_pair,
    extract_patches,
    patch_params_from_cfg,
)
from src.data.registry import get_dataset
from src.data.splits import geographic_split, scene_group_key
from src.utils.logging import get_logger
from src.utils.paths import resolve_output_path

__all__ = [
    "PatchDataset",
    "build_dataloaders",
    "resolve_split_assignments",
    "select_indices",
    "worker_init_fn",
    "SplitError",
]


class SplitError(RuntimeError):
    """A split could not be built, or is unusable for training.

    Raised rather than warned. Every case it covers (an empty split, a split that
    disagrees with the dataset, a scene appearing on both sides) produces numbers
    that look fine and mean nothing.
    """


# -- split assignment ------------------------------------------------------


def _read_split_file(path: Path) -> Dict[str, str]:
    """Read ``sample_id -> split`` from the CSV ``scripts/make_splits.py`` writes.

    Args:
        path: The CSV. Must have ``sample_id`` and ``split`` columns.

    Returns:
        Mapping of sample id to split name.

    Raises:
        SplitError: The file is missing the required columns or is empty. A
            malformed split file is not silently ignored in favour of
            recomputing -- that would hide the fact that the file the user
            believes is in force is not.
    """
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SplitError(f"Split file {path} is empty.")
    missing = {"sample_id", "split"} - set(rows[0])
    if missing:
        raise SplitError(
            f"Split file {path} lacks required column(s) {sorted(missing)}. "
            "Regenerate it with scripts/make_splits.py."
        )
    return {row["sample_id"]: row["split"] for row in rows}


def resolve_split_assignments(
    cfg: Any,
    dataset: Any,
    logger: Any,
) -> Tuple[Dict[str, str], str]:
    """Get ``sample_id -> split`` for every sample in ``dataset``.

    Prefers the CSV written by ``scripts/make_splits.py`` (it is the one that was
    verified against ``cfg.splits.min_separation_km`` before being written). When
    that file is absent, the same geographic split is recomputed in-process from
    the sample centroids with ``cfg.splits.seed``. There is no third option: a
    random split is never a fallback, because it leaks and does so invisibly.

    Args:
        cfg: The loaded config.
        dataset: An :class:`~src.data.base.SRPairDataset`. Its ``catalog`` is
            used when present (no imagery is decoded); otherwise sample metadata
            is read, which does decode.
        logger: Logger for the provenance line.

    Returns:
        ``(assignments, source)`` where ``assignments`` maps sample id to split
        name and ``source`` is ``"file:<path>"`` or ``"computed"``.

    Raises:
        SplitError: The split file exists but does not cover the dataset, or a
            sample has no parseable centroid so no geographic split is possible.
    """
    loader_cfg = cfg["loader"]
    split_cfg = cfg["splits"]
    dataset_name = str(cfg["dataset"]["name"])

    configured = loader_cfg["split_file"]
    if configured is not None:
        split_path = Path(str(configured))
        if not split_path.is_absolute():
            split_path = Path(resolve_output_path(cfg, "manifest_dir")) / split_path
    else:
        split_path = Path(resolve_output_path(cfg, "manifest_dir")) / str(
            split_cfg["output_name"]
        ).format(dataset=dataset_name)

    records = _split_records(dataset)
    sample_ids = [record["sample_id"] for record in records]

    if split_path.is_file():
        assignments = _read_split_file(split_path)
        uncovered = [sid for sid in sample_ids if sid not in assignments]
        if uncovered:
            raise SplitError(
                f"Split file {split_path} covers {len(assignments)} samples but "
                f"{len(uncovered)} of the dataset's {len(sample_ids)} samples "
                f"are absent from it (first: {uncovered[0]!r}). The split and "
                "the dataset disagree -- regenerate it with "
                "scripts/make_splits.py rather than training on the overlap."
            )
        logger.info(
            "Split read from %s (%d samples assigned).", split_path, len(sample_ids)
        )
        return {sid: assignments[sid] for sid in sample_ids}, f"file:{split_path}"

    logger.warning(
        "No split file at %s; recomputing the geographic split in-process with "
        "seed=%s and min_separation_km=%s. Run scripts/make_splits.py to write "
        "and verify it once instead of re-deriving it on every build.",
        split_path,
        split_cfg["seed"],
        split_cfg["min_separation_km"],
    )
    result = geographic_split(
        records,
        fractions={k: float(v) for k, v in split_cfg["fractions"].items()},
        min_separation_km=float(split_cfg["min_separation_km"]),
        seed=int(split_cfg["seed"]),
        group_by_scene=bool(split_cfg["group_by_scene"]),
    )
    return dict(zip(sample_ids, result["assignments"])), "computed"


def _split_records(dataset: Any) -> List[Dict[str, Any]]:
    """Collect ``sample_id`` + ``centroid_lonlat`` per sample, cheaply if possible."""
    catalog = getattr(dataset, "catalog", None)
    if catalog is not None:
        return [
            {
                "sample_id": entry["sample_id"],
                "centroid_lonlat": entry.get("centroid_lonlat"),
            }
            for entry in catalog
        ]
    records = []
    for idx in range(len(dataset)):
        meta = dataset[idx]["meta"]
        records.append(
            {
                "sample_id": meta["sample_id"],
                "centroid_lonlat": meta.get("centroid_lonlat"),
            }
        )
    return records


def select_indices(
    cfg: Any,
    dataset: Any,
    assignments: Mapping[str, str],
    split_name: str,
    logger: Any,
) -> List[int]:
    """Dataset indices belonging to one split, optionally cache-restricted.

    Args:
        cfg: The loaded config.
        dataset: The dataset being split.
        assignments: ``sample_id -> split name``.
        split_name: The split to select, e.g. ``cfg.loader.train_split``.
        logger: Logger for the skip counts.

    Returns:
        Sorted dataset indices, in catalog order, so the selection is
        deterministic.

    Raises:
        SplitError: The split has fewer than ``cfg.loader.min_samples_per_split``
            samples after filtering.
    """
    cached_only = bool(cfg["loader"]["cached_only"])
    minimum = int(cfg["loader"]["min_samples_per_split"])
    records = _split_records(dataset)
    is_cached = getattr(dataset, "is_cached", None)

    in_split, skipped_uncached = [], 0
    for idx, record in enumerate(records):
        if assignments.get(record["sample_id"]) != split_name:
            continue
        if cached_only and callable(is_cached) and not is_cached(idx):
            skipped_uncached += 1
            continue
        in_split.append(idx)

    if skipped_uncached:
        logger.warning(
            "Split %r: %d of %d samples are not in the local cache and were "
            "EXCLUDED (cfg.loader.cached_only=true). This is the partial-download "
            "path -- numbers computed here describe %d samples, not the full "
            "split. Set cfg.loader.cached_only=false once the download finishes.",
            split_name,
            skipped_uncached,
            len(in_split) + skipped_uncached,
            len(in_split),
        )

    if len(in_split) < minimum:
        raise SplitError(
            f"Split {split_name!r} has {len(in_split)} samples, below "
            f"cfg.loader.min_samples_per_split={minimum}"
            + (
                f" ({skipped_uncached} were excluded as not yet cached)."
                if skipped_uncached
                else "."
            )
            + " Check cfg.splits.fractions, the split file, and the cache."
        )

    logger.info("Split %r: %d samples.", split_name, len(in_split))
    return in_split


# -- patch dataset ---------------------------------------------------------


class PatchDataset(Dataset):
    """Patches cut from one split's tiles.

    In ``"grid"`` mode the patch list is enumerated once at construction, so
    ``len()`` is the true number of patches and the order is fixed -- validation
    over the same config is byte-for-byte comparable between runs. In
    ``"random"`` mode ``len()`` is ``num_samples * random_per_sample`` and each
    item is a fresh crop whose origin comes from an RNG seeded by
    ``(seed, epoch, item index)``: reproducible, different every epoch, and
    independent of how many workers happen to be running.

    A rejected patch is not skipped silently. In grid mode rejected patches are
    dropped at construction and the counts are logged; in random mode the crop is
    re-drawn up to ``max_filter_retries`` times and, if every draw is rejected,
    the last one is returned with ``filter_reason`` set on the item, so the
    training loop can see the rate rather than the dataset quietly biasing
    itself.
    """

    def __init__(
        self,
        cfg: Any,
        dataset: Any,
        indices: Sequence[int],
        mode: str,
        split_name: str,
        logger: Any = None,
        max_filter_retries: int = 8,
    ) -> None:
        """
        Args:
            cfg: The loaded config; patch geometry comes from
                :func:`~src.data.patches.patch_params_from_cfg`.
            dataset: The underlying :class:`~src.data.base.SRPairDataset`.
            indices: Dataset indices this split owns.
            mode: ``"grid"`` (deterministic) or ``"random"``.
            split_name: For logs and for the per-item ``split`` field.
            logger: Logger; created if None.
            max_filter_retries: Random-mode redraws before giving up on a tile.

        Raises:
            SplitError: Grid mode rejected every patch in the split.
        """
        self.cfg = cfg
        self.dataset = dataset
        self.indices = list(int(i) for i in indices)
        self.mode = str(mode)
        self.split_name = str(split_name)
        self.logger = logger or get_logger("data.loader", log_file=cfg["paths"]["log_file"])
        self.max_filter_retries = int(max_filter_retries)

        params = patch_params_from_cfg(cfg)
        self.lr_size = params["lr_size"]
        self.hr_size = params["hr_size"]
        self.scale = params["scale"]
        self.stride = params["stride"]
        self.random_per_sample = params["random_per_sample"]
        self.drop_incomplete_edge = params["drop_incomplete_edge"]

        self.seed = int(cfg["seed"])
        self.epoch = 0
        self.patch_filter = PatchFilter.from_cfg(cfg)

        # The LR tile size the GRID PATH validates over. The dataset emits the
        # full stored tile (130 px for SEN2NAIPv2); grid mode centre-crops to
        # this size in _tile() so the validation set is a fixed set of pixels,
        # and random mode ignores it and draws from the whole tile. Grid
        # enumeration needs it before any imagery is read, and it is re-checked
        # against the actual tile on every load.
        self.tile_lr_size = int(cfg["sr"]["lr_patch_size"])

        if self.mode == "grid":
            self._index = self._build_grid_index()
            self.patch_filter.log_summary(self.logger)
            if not self._index:
                raise SplitError(
                    f"Split {self.split_name!r}: the patch filter rejected every "
                    f"one of the {self.patch_filter.counts.get('examined', 0)} "
                    "candidate patches. Relax cfg.patches.filters or check the "
                    "data -- an empty validation set cannot be trained against."
                )
        else:
            self._index = None

    # -- grid mode ---------------------------------------------------------

    def _build_grid_index(self) -> List[Tuple[int, int, int]]:
        """Enumerate ``(dataset_idx, lr_row, lr_col)`` for every surviving patch.

        Reads every tile once, which is the price of knowing the true patch count
        and rejection rate up front rather than discovering mid-epoch that half
        the validation set is cloud.
        """
        index: List[Tuple[int, int, int]] = []
        for dataset_idx in self.indices:
            sample = self.dataset[dataset_idx]
            lr, hr = self._tile(sample, dataset_idx)
            patches = extract_patches(
                lr,
                hr,
                lr_size=self.lr_size,
                scale=self.scale,
                stride=self.stride,
                mode="grid",
                patch_filter=self.patch_filter,
            )
            for patch in patches:
                coords = patch["coords"]
                index.append((dataset_idx, coords.lr_row, coords.lr_col))
        self.logger.info(
            "Split %r: %d patches from %d tiles (grid, lr_size=%d stride=%d).",
            self.split_name,
            len(index),
            len(self.indices),
            self.lr_size,
            self.stride,
        )
        return index

    def _tile(self, sample: Mapping[str, Any], idx: int):
        """The LR/HR tile this split cuts patches from, after the mode transform.

        The dataset emits the FULL stored tile (130x130 LR for SEN2NAIPv2), and
        the two modes want different things from it:

        - **grid** (deterministic, validation) centre-crops to
          ``cfg.sr.lr_patch_size`` first. That crop IS the deterministic
          validation transform: it fixes which pixels a validation number
          describes, so the same config gives the same number on every machine
          and every run.
        - **random** (training) cuts from the full tile. Centre-cropping first
          would throw away the tile periphery for every epoch of training, which
          is both a waste of downloaded data and a quiet bias toward scene
          centres.

        Args:
            sample: A sample dict from the underlying dataset.
            idx: Dataset index, for error messages.

        Returns:
            ``(lr, hr)`` float32 surface reflectance, ``(C, H, W)`` and
            ``(C, H*scale, W*scale)``, nominally ``[0, 1]`` and unclipped.

        Raises:
            SplitError: The tile is too small to cut from.
        """
        lr, hr = sample["lr"], sample["hr"]
        self._check_tile(sample, lr, hr, idx)
        if self.mode != "grid":
            return lr, hr
        cropped = centre_crop_pair(lr, hr, self.tile_lr_size, self.scale)
        return cropped["lr"], cropped["hr"]

    def _check_tile(self, sample: Mapping[str, Any], lr: Any, hr: Any, idx: int) -> None:
        """Fail loudly if a tile is too small for the patches planned against it.

        The check is a LOWER BOUND, not equality: the dataset emits whole stored
        tiles (130 px LR for SEN2NAIPv2) and the grid path centre-crops them to
        ``cfg.sr.lr_patch_size`` in :meth:`_tile`. Requiring exact equality here
        would force the dataset to crop in its read path, which is the thing the
        index amendment removed.
        """
        need = self.tile_lr_size if self.mode == "grid" else self.lr_size
        if lr.shape[-2] < need or lr.shape[-1] < need:
            raise SplitError(
                f"Sample {sample['meta']['sample_id']!r} (index {idx}) emitted an "
                f"LR tile of {tuple(int(d) for d in lr.shape)}, smaller than the "
                f"{need} px this split cuts from "
                f"(mode={self.mode!r}, cfg.sr.lr_patch_size={self.tile_lr_size}, "
                f"cfg.patches.lr_size={self.lr_size}). The patch index was "
                "planned against that size, so a smaller tile would silently "
                "change which pixels are validated."
            )
        if hr.shape[-2] != lr.shape[-2] * self.scale:
            raise SplitError(
                f"Sample {sample['meta']['sample_id']!r} (index {idx}): HR tile "
                f"{tuple(int(d) for d in hr.shape)} is not {self.scale}x the LR "
                f"tile {tuple(int(d) for d in lr.shape)}."
            )

    # -- torch Dataset interface ------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch that seeds random-mode crops.

        Call once per epoch from the training loop. Without it every epoch draws
        the same crops, which turns a random-crop augmentation into a fixed one.
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self._index is not None:
            return len(self._index)
        return len(self.indices) * self.random_per_sample

    def _rng_for(self, item: int) -> np.random.Generator:
        """Per-item generator seeded by ``(seed, epoch, item)``.

        Seeding per item rather than per worker is what makes the crop sequence
        independent of ``cfg.train.num_workers``: the same config gives the same
        crops on 0 workers and on 6.
        """
        return np.random.default_rng([self.seed, self.epoch, int(item)])

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Return one patch pair.

        Args:
            item: Index in ``[0, len(self))``.

        Returns:
            A dict with ``lr`` (``torch.float32``, ``(C, lr_size, lr_size)``),
            ``hr`` (``torch.float32``, ``(C, lr_size*scale, lr_size*scale)``) --
            both surface reflectance in the same units and band order as the
            underlying dataset, nominally ``[0, 1]`` and unclipped -- plus
            ``sample_id`` (str), ``dataset_index`` (int), ``lr_row``/``lr_col``
            (int, LR pixels), ``hr_row``/``hr_col`` (int, HR pixels, always
            ``scale`` times the LR values), ``split`` (str), and
            ``filter_reason`` (str, empty when the patch passed the filter).

        Raises:
            IndexError: ``item`` is out of range.
        """
        if not 0 <= item < len(self):
            raise IndexError(
                f"{type(self).__name__}[{self.split_name}] index {item} out of "
                f"range for {len(self)} patches."
            )

        if self._index is not None:
            dataset_idx, row, col = self._index[item]
            sample = self.dataset[dataset_idx]
            lr, hr = self._tile(sample, dataset_idx)
            # Sliced through the same crop_pair() the index was built with, so
            # the HR-equals-scale-times-LR assertion runs on this read too.
            patch = crop_pair(lr, hr, row, col, self.lr_size, self.scale)
            return self._item(patch, sample, dataset_idx, "")

        dataset_idx = self.indices[item // self.random_per_sample]
        sample = self.dataset[dataset_idx]
        lr, hr = self._tile(sample, dataset_idx)
        rng = self._rng_for(item)

        accepted = None
        last = None
        reason = ""
        for _ in range(self.max_filter_retries):
            drawn = extract_patches(
                lr,
                hr,
                lr_size=self.lr_size,
                scale=self.scale,
                mode="random",
                num_patches=1,
                rng=rng,
                patch_filter=None,
            )[0]
            verdict = self.patch_filter(drawn["lr"], drawn["hr"])
            if verdict is None:
                accepted = drawn
                break
            last, reason = drawn, verdict

        if accepted is None:
            # Every draw was rejected. Return the last candidate labelled with
            # the rule that rejected it rather than raising: one difficult tile
            # must not kill a training run, and the label plus the filter's
            # counters make the rate visible rather than hiding it.
            accepted = last
            self.logger.warning(
                "Split %r sample %d: %d random crops in a row were rejected "
                "(last rule: %s); returning the last one labelled with it.",
                self.split_name,
                dataset_idx,
                self.max_filter_retries,
                reason,
            )
        else:
            reason = ""
        return self._item(accepted, sample, dataset_idx, reason)

    def _item(
        self,
        patch: Mapping[str, Any],
        sample: Mapping[str, Any],
        dataset_idx: int,
        filter_reason: str,
    ) -> Dict[str, Any]:
        coords = patch["coords"]
        lr_patch, hr_patch = patch["lr"], patch["hr"]
        if not torch.is_tensor(lr_patch):
            lr_patch = torch.from_numpy(np.ascontiguousarray(lr_patch))
            hr_patch = torch.from_numpy(np.ascontiguousarray(hr_patch))
        else:
            lr_patch = lr_patch.contiguous()
            hr_patch = hr_patch.contiguous()
        return {
            "lr": lr_patch,
            "hr": hr_patch,
            "sample_id": str(sample["meta"]["sample_id"]),
            "dataset_index": int(dataset_idx),
            "lr_row": coords.lr_row,
            "lr_col": coords.lr_col,
            "hr_row": coords.hr_row,
            "hr_col": coords.hr_col,
            "split": self.split_name,
            "filter_reason": filter_reason,
        }


# -- worker seeding --------------------------------------------------------


def worker_init_fn(worker_id: int) -> None:
    """Seed a DataLoader worker's RNGs reproducibly.

    ``torch`` already gives each worker a distinct ``initial_seed()`` derived
    from the loader's generator and the epoch, but it seeds only torch. NumPy and
    Python's ``random`` are left on whatever state was inherited by ``fork`` --
    or, on Windows ``spawn``, on a fresh entropy-seeded state that differs
    between runs. Either way, anything that draws from them is unreproducible
    unless it is seeded here.

    Args:
        worker_id: Index of this worker, supplied by torch.
    """
    base_seed = torch.initial_seed() % (2**32)
    seed = (base_seed + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# -- the entry point -------------------------------------------------------


def build_dataloaders(
    cfg: Any,
    dataset: Any = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """Build the train and validation loaders for ``cfg``.

    Args:
        cfg: The loaded config. Reads ``dataset``, ``splits``, ``patches``,
            ``loader``, ``train``, and ``seed``.
        dataset: An already-constructed dataset. Defaults to
            :func:`~src.data.registry.get_dataset`; pass one to avoid rebuilding
            a catalog that has already been loaded.
        logger: Logger; created if None.

    Returns:
        ``{"train": DataLoader, "val": DataLoader, "datasets": {...},
        "indices": {...}, "split_source": str}``. The ``datasets`` entry holds
        the :class:`PatchDataset` objects so a training loop can call
        ``set_epoch``; ``indices`` holds the dataset indices per split, which is
        what a report needs to state how much data a number came from.

    Raises:
        SplitError: A split is empty, the split file disagrees with the dataset,
            or -- the check that matters -- a scene id appears in more than one
            split.
    """
    logger = logger or get_logger("data.loader", log_file=cfg["paths"]["log_file"])
    dataset = dataset if dataset is not None else get_dataset(cfg)

    assignments, source = resolve_split_assignments(cfg, dataset, logger)
    _assert_scene_disjoint(assignments, logger)

    loader_cfg = cfg["loader"]
    train_split = str(loader_cfg["train_split"])
    val_split = str(loader_cfg["val_split"])

    train_indices = select_indices(cfg, dataset, assignments, train_split, logger)
    val_indices = select_indices(cfg, dataset, assignments, val_split, logger)

    overlap = set(train_indices) & set(val_indices)
    if overlap:
        raise SplitError(
            f"{len(overlap)} sample indices are in both {train_split!r} and "
            f"{val_split!r}: {sorted(overlap)[:5]}."
        )

    train_patches = PatchDataset(
        cfg, dataset, train_indices, mode="random", split_name=train_split, logger=logger
    )
    val_patches = PatchDataset(
        cfg, dataset, val_indices, mode="grid", split_name=val_split, logger=logger
    )

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    generator = torch.Generator()
    generator.manual_seed(int(cfg["seed"]))

    common = {
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg["pin_memory"]),
        "worker_init_fn": worker_init_fn,
        # persistent_workers requires num_workers > 0; torch raises otherwise.
        "persistent_workers": bool(loader_cfg["persistent_workers"]) and num_workers > 0,
    }

    train_loader = DataLoader(
        train_patches,
        batch_size=batch_size,
        shuffle=bool(loader_cfg["shuffle_train"]),
        drop_last=bool(loader_cfg["drop_last"]),
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_patches,
        batch_size=batch_size,
        # Never shuffled and never truncated: the validation number must cover
        # the same patches in the same order on every run.
        shuffle=False,
        drop_last=False,
        **common,
    )

    logger.info(
        "Loaders built from %s: train %d patches / %d tiles, val %d patches / "
        "%d tiles, batch_size=%d num_workers=%d.",
        source,
        len(train_patches),
        len(train_indices),
        len(val_patches),
        len(val_indices),
        batch_size,
        num_workers,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "datasets": {"train": train_patches, "val": val_patches},
        "indices": {"train": train_indices, "val": val_indices},
        "split_source": source,
    }


def _assert_scene_disjoint(assignments: Mapping[str, str], logger: Any) -> None:
    """Assert no scene id appears in two splits.

    This re-derives the property from the sample ids rather than trusting the
    grouping that produced them, so it is a real check on the file on disk --
    including a hand-edited one.

    Raises:
        SplitError: A scene straddles two splits.
    """
    scenes: Dict[str, str] = {}
    straddling: Dict[str, set] = {}
    for sample_id, split in assignments.items():
        scene = scene_group_key(sample_id)
        if scene is None:
            continue
        if scene in scenes and scenes[scene] != split:
            straddling.setdefault(scene, {scenes[scene]}).add(split)
        scenes.setdefault(scene, split)

    if straddling:
        examples = list(straddling.items())[:5]
        raise SplitError(
            f"{len(straddling)} scene(s) appear in more than one split, so "
            "patches from one aerial scene are in both train and validation. "
            f"Examples: {examples}. Regenerate the split with "
            "scripts/make_splits.py (cfg.splits.group_by_scene must be true)."
        )
    logger.info(
        "Scene check: %d distinct scenes, none straddling a split boundary.",
        len(scenes),
    )
