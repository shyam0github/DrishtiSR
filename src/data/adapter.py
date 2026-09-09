"""Simple-constructor adapter over the Day 1 patch pipeline.

WHY THIS FILE EXISTS. ``src/train.py`` loads its dataset by name and calls it
with a four-argument constructor::

    DS(root=<str>, split="train"|"val", patch_lr=<int>, scale=<int>)
    DS[i] -> {"lr": FloatTensor[C, h, w], "hr": FloatTensor[C, h*scale, w*scale]}

The Day 1 dataset does not have that shape and should not be bent into it. Its
constructor takes the whole composed config (``SEN2NAIPv2Dataset(cfg)``), it
emits FULL stored tiles rather than patches, and the split lives outside it in
``src/data/loader.py``. All three are deliberate, so this module WRAPS that
stack instead of changing it:

    SRPatchDataset(root, split, patch_lr, scale)
      -> load_config() with the four arguments applied as overrides
      -> SEN2NAIPv2Dataset(cfg)                        (tiles, reflectance)
      -> resolve_split_assignments(...)                (the split CSV)
      -> select_indices(..., split)                    (this split's tiles)
      -> PatchDataset(cfg, ..., mode)                  (patches, filters)

**The split is the one the Day 1 bicubic baseline was measured on**, because it
is read from the same file by the same function: ``resolve_split_assignments``
prefers ``outputs/splits_sen2naipv2.csv``, written once by
``scripts/make_splits.py`` from a geographic, scene-grouped split with
``cfg.splits.seed`` (42). Nothing here re-derives, re-seeds, or re-shuffles it.
Note that ``--seed 1337`` in ``src/train.py`` seeds torch/numpy/random for
initialisation and batch order; it is NOT a split seed and does not touch split
membership.

Reflectance is passed through untouched: surface reflectance, float32,
nominally [0, 1] and UNCLIPPED -- bright targets legitimately exceed 1.0. This
adapter does not normalise, standardise, or clamp.

The one thing it does add is GEOMETRIC augmentation, on the training split
only: a dihedral transform drawn once per item and applied to the LR and the HR
by the same call, via :mod:`src.data.augment`. It moves pixels, never their
values, so the reflectance guarantee above survives it intact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.data.augment import augment_pair
from src.data.loader import PatchDataset, resolve_split_assignments, select_indices
from src.data.registry import get_dataset_class
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.paths import resolve_cache_dir
from src.utils.seed import torch_generator

__all__ = ["SRPatchDataset"]

# Grid mode is deterministic and is what the validation metrics -- including the
# Day 1 bicubic baseline -- are computed over. Random crops are for training
# only. Any split not named here is read deterministically.
_RANDOM_SPLITS = ("train",)

# Splits that may be augmented. Same list, stated separately because the two
# decisions are separate: "random" is about WHERE the crop comes from, augment
# is about what happens to it afterwards. Both must stay off for val and test.
_AUGMENTED_SPLITS = ("train",)


class SRPatchDataset(Dataset):
    """Day 1 tiles + split + patching behind the four-argument constructor.

    Items are ``{"lr", "hr"}`` only. The Day 1 ``PatchDataset`` also returns
    ``sample_id``, patch coordinates and a nullable ``filter_reason``; those are
    dropped here because ``torch.utils.data.default_collate`` raises on a
    ``None`` field, which would turn a logged filter miss into a crash inside
    the training loop.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_lr: int = 64,
        scale: int = 4,
        config: Any = "configs/base.yaml",
        dataset_name: str = "sen2naipv2",
        mode: Optional[str] = None,
        validate: bool = True,
        augment: bool = True,
    ) -> None:
        """
        Args:
            root: The sample cache root -- the directory that CONTAINS the
                subset directory (e.g. ``outputs/cache``, which holds
                ``sen2naipv2-crosssensor/``). The subset directory itself is
                also accepted and its parent used, since that is the path a
                caller is most likely to have to hand. The literal string
                ``"auto"`` resolves the location through
                ``src.utils.paths.resolve_cache_dir`` instead, which is the
                only thing in this project allowed to decide where data lives;
                use it from Kaggle jobs, whose mount path is not the one a
                dataset slug suggests.
            split: ``"train"``, ``"val"`` or ``"test"``; matched against the
                ``split`` column of the split CSV.
            patch_lr: LR patch edge in pixels. Overrides ``cfg.patches.lr_size``.
                The HR patch is exactly ``patch_lr * scale``.
            scale: SR factor. Overrides ``cfg.sr.scale``; ``cfg.sr.hr_patch_size``
                is kept consistent at ``lr_patch_size * scale``.
            config: Config path merged over the defaults.
            dataset_name: Registry name of the Day 1 dataset to wrap.
            mode: ``"grid"`` or ``"random"``. Defaults to ``"random"`` for the
                training split and ``"grid"`` for every other split.
            validate: Passed to the underlying dataset's sample validation.
            augment: Draw a random dihedral transform per item and apply it to
                the LR and the HR together (see :mod:`src.data.augment`).
                IGNORED AND FORCED OFF for every split but ``"train"``: a
                validation number is only comparable to the Day 1 bicubic
                baseline if it is measured on the same fixed patches, and an
                augmented val set would move that reference every time the RNG
                advanced. The resolved value is on ``self.augment``, and it is
                logged, so a run's metadata records what was actually applied
                rather than what was asked for.

        Raises:
            FileNotFoundError: ``root`` does not exist, or holds no subset
                directory for the configured subset. From ``"auto"``, raised by
                ``resolve_cache_dir`` when no candidate mount is present.
            SplitError: The split is empty, or the split file does not cover the
                dataset.
        """
        self.split = str(split)
        self.patch_lr = int(patch_lr)
        self.scale = int(scale)

        cache_root = self._resolve_cache_root(root, config, dataset_name)

        overrides = [
            f"paths.cache_dir={cache_root.as_posix()}",
            f"patches.lr_size={self.patch_lr}",
            f"sr.scale={self.scale}",
        ]
        probe = load_config(config)
        # hr_patch_size must equal lr_patch_size * scale (configs/base.yaml).
        # scale is an argument here, so re-derive it rather than let the two
        # disagree silently when scale != the config default.
        tile_lr_size = int(probe["sr"]["lr_patch_size"])
        cfg = load_config(
            config,
            overrides=overrides + [f"sr.hr_patch_size={tile_lr_size * self.scale}"],
        )

        resolved_mode = mode or ("random" if self.split in _RANDOM_SPLITS else "grid")
        cfg["patches"]["mode"] = resolved_mode
        self.mode = resolved_mode
        self.cfg = cfg

        # Forced, not merely defaulted: the caller cannot switch augmentation on
        # for a split whose whole job is to stay fixed.
        self.augment = bool(augment) and self.split in _AUGMENTED_SPLITS

        self.logger = get_logger("data.adapter", log_file=cfg["paths"]["log_file"])
        self.logger.info(
            "SRPatchDataset(split=%r, patch_lr=%d, scale=%d, mode=%r, "
            "augment=%r, cache=%s)",
            self.split,
            self.patch_lr,
            self.scale,
            resolved_mode,
            self.augment,
            cache_root,
        )
        if bool(augment) and not self.augment:
            self.logger.info(
                "augment=True was requested for split %r but only %s is "
                "augmented; augmentation is OFF for this dataset.",
                self.split,
                " / ".join(repr(s) for s in _AUGMENTED_SPLITS),
            )

        dataset_cls = get_dataset_class(dataset_name)
        self.source = dataset_cls(cfg, validate=validate)

        assignments, split_source = resolve_split_assignments(
            cfg, self.source, self.logger
        )
        self.split_source = split_source
        self.logger.info("Split assignments from %s", split_source)

        self.indices: List[int] = list(
            select_indices(cfg, self.source, assignments, self.split, self.logger)
        )
        self.patches = PatchDataset(
            cfg,
            self.source,
            self.indices,
            mode=resolved_mode,
            split_name=self.split,
            logger=self.logger,
        )

        # The run seed, for the per-sample augmentation stream. Read from the
        # config, never a literal, and the SAME value PatchDataset seeds crops
        # from -- the two decisions share the triple and differ only by stream
        # name, so a sample's crop and its flip move together across runs.
        self.seed = int(cfg["seed"])

        self.bands = tuple(cfg["dataset"]["bands"])

    @staticmethod
    def _resolve_cache_root(root: str, config: Any, dataset_name: str) -> Path:
        """Accept either the cache root or the subset directory inside it.

        Args:
            root: Path as given by the caller.
            config: Config path, read for the subset name.
            dataset_name: Registry name, used only in the error message.

        Returns:
            The directory ``resolve_cache_dir`` should return, i.e. the parent
            of the subset directory.

        Raises:
            FileNotFoundError: Neither interpretation of ``root`` exists.
        """
        subset = str(load_config(config)["sen2naipv2"]["subset"])

        # "auto" defers to the project's single path authority instead of
        # trusting a literal the caller typed. MEASURED 2026-09-06: the live
        # Kaggle mount is /kaggle/input/datasets/<owner>/<slug>, NOT the
        # /kaggle/input/<slug> that a job definition would naturally spell --
        # so a hardcoded --data-root is wrong on exactly the machine that
        # costs GPU-hours to be wrong on. resolve_cache_dir() tries both
        # layouts (paths.kaggle_mount_patterns) and carries no username.
        if str(root).strip().lower() == "auto":
            resolved = Path(resolve_cache_dir(load_config(config)))
            # resolve_cache_dir returns the directory CONTAINING the subset on
            # a mount, but the local cache root itself otherwise; normalise to
            # the same "parent of subset" contract as the explicit branches.
            if resolved.name == subset:
                return resolved.parent
            return resolved

        path = Path(str(root)).expanduser()

        if (path / subset).is_dir():
            return path
        if path.name == subset and path.is_dir():
            return path.parent
        raise FileNotFoundError(
            f"root={str(root)!r} is not a usable cache location for "
            f"{dataset_name!r}: expected either a directory containing "
            f"{subset!r}, or {subset!r} itself. Checked {path / subset} and "
            f"{path}."
        )

    def set_epoch(self, epoch: int) -> None:
        """Advance the crop AND augmentation streams to ``epoch``.

        Must be called once per epoch, before that epoch's DataLoader iterator
        is created -- see the epoch loop in ``src/train.py``. Both streams read
        ``self.patches.epoch``, so one call moves them together and there is no
        state to keep in sync.

        Refused in grid mode by :meth:`PatchDataset.set_epoch`, which pins val
        and test at epoch 0 forever. Because the augmentation stream reads the
        same attribute, that pin covers augmentation too: a deterministic split
        cannot be made non-deterministic from here.

        Args:
            epoch: Epoch number, 0-based.

        Raises:
            ValueError: ``epoch`` is negative.
        """
        self.patches.set_epoch(int(epoch))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one LR/HR patch pair.

        Args:
            index: Index in ``[0, len(self))``.

        Returns:
            ``{"lr": float32 (C, patch_lr, patch_lr),
            "hr": float32 (C, patch_lr*scale, patch_lr*scale)}``. Surface
            reflectance in both, channel order ``cfg.dataset.bands``, nominally
            [0, 1] and UNCLIPPED above 1.0. When ``self.augment`` is set, both
            tensors have had the SAME randomly drawn dihedral transform applied
            (:func:`src.data.augment.augment_pair`); shapes, dtypes and
            reflectance values are unaffected, only the spatial layout.
        """
        item = self.patches[index]
        lr, hr = item["lr"], item["hr"]
        if self.augment:
            # Seeded from the SAME (seed, epoch, index) triple as the crop, on
            # a separate stream. Not the global RNG: that advances per worker
            # and would make the transform a function of num_workers and batch
            # order, so Run A2 and Run B could not be guaranteed the identical
            # data stream the comparison depends on.
            lr, hr = augment_pair(
                lr,
                hr,
                generator=torch_generator(
                    self.seed, self.patches.epoch, int(index), "augment"
                ),
            )
        return {"lr": lr, "hr": hr}
