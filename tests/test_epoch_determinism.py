"""The frozen-crop bug, and the A2-vs-B guarantee that depends on it staying fixed.

WHAT WENT WRONG. ``src/train.py`` iterated ``while True: for batch in loader``
and never advanced an epoch counter. ``PatchDataset`` seeds crop origins from
``(seed, epoch, index)``, so with ``epoch`` pinned at 0 all 40k iterations of
Run A trained on ONE fixed set of ~2400 crops. Nothing raised. The loss fell.
The model memorised, and val PSNR peaked at 8k and declined to 40k.

WHAT THESE TESTS HOLD IN PLACE.

  (a) train crops MOVE between epochs -- the bug, caught directly;
  (b) val tensors are BYTE-IDENTICAL between epochs -- a val set that resampled
      would make every validation point a measurement of different data, and
      the training curve would stop being a curve;
  (c) two independent "runs" at one seed produce identical crops for epochs
      0..3 -- this is the Run A2 (control) vs Run B (fix) guarantee. If the
      crop stream diverges between them, the control is worthless and the
      headline comparison dies with it;
  (c2) the same, for BATCH COMPOSITION. Which crops land in a batch together
      was a separate hole: `shuffle=True` without an explicit generator draws
      from the GLOBAL torch RNG, so batch order depended on how many random
      numbers weight init and the objective happened to consume. A2 and B1
      match today only because neither L1 nor the spectral term touches the
      RNG -- an accident of the current losses, and the NLL head is next. The
      generator threaded through ``epoch_stream`` makes the guarantee
      structural instead of inherited;
  (d) crop origins stay in bounds for the smallest tile in the split.

CPU only, synthetic stub, no network, no cache.
"""

import numpy as np
import pytest
import torch

from src.data.loader import PatchDataset
from src.data.registry import get_dataset
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import derive_seed, numpy_generator, torch_generator


def _cfg(tmp_path, *overrides):
    base = [
        f"paths.manifest_dir={tmp_path.as_posix()}",
        f"paths.log_file={(tmp_path / 'run.log').as_posix()}",
        "loader.cached_only=false",
    ]
    return load_config("configs/base.yaml", smoke=True, overrides=base + list(overrides))


def _patches(tmp_path, mode, split, indices=(0, 1)):
    cfg = _cfg(tmp_path)
    return cfg, PatchDataset(
        cfg,
        get_dataset(cfg),
        list(indices),
        mode,
        split,
        logger=get_logger("test.epoch", log_file=tmp_path / "run.log"),
    )


def _origins(patches):
    """Crop origins for a whole pass, as ``(lr_row, lr_col)`` per item."""
    return [(patches[i]["lr_row"], patches[i]["lr_col"]) for i in range(len(patches))]


# -- (a) the bug -----------------------------------------------------------


def test_train_crops_move_between_epochs(tmp_path):
    """The regression test for the frozen-crop bug itself."""
    _, patches = _patches(tmp_path, "random", "train")

    patches.set_epoch(0)
    first = _origins(patches)
    patches.set_epoch(0)
    assert _origins(patches) == first, "the same epoch must reproduce its crops"

    patches.set_epoch(1)
    second = _origins(patches)
    assert second != first, (
        "epoch 1 drew the same crops as epoch 0 -- crops are frozen and the "
        "model will see one fixed dataset forever, which is what Run A did."
    )

    # And it keeps moving, rather than alternating between two states.
    patches.set_epoch(2)
    assert _origins(patches) not in (first, second)


def test_the_train_loop_actually_calls_set_epoch(tmp_path):
    """The wiring, not just the capability. Run A had the capability."""
    from src.train import epoch_stream

    seen = []

    class _Spy:
        def set_epoch(self, epoch):
            seen.append(epoch)

    loader = [{"lr": torch.zeros(1)}, {"lr": torch.zeros(1)}]  # 2 batches/epoch
    stream = epoch_stream(loader, _Spy(), start_epoch=0)
    epochs = [next(stream)[0] for _ in range(6)]

    assert epochs == [0, 0, 1, 1, 2, 2]
    assert seen == [0, 1, 2], "set_epoch must fire once per epoch, before its batches"


def test_resume_rejoins_the_same_epoch(tmp_path):
    """A resumed run must continue the crop sequence, not restart it."""
    from src.train import epoch_stream

    seen = []

    class _Spy:
        def set_epoch(self, epoch):
            seen.append(epoch)

    # 150 it/epoch, resuming at iteration 1500 -> epoch 10.
    start_epoch = 1500 // 150
    stream = epoch_stream([0] * 150, _Spy(), start_epoch=start_epoch)
    next(stream)
    assert seen == [10]


def test_a_dataset_that_cannot_advance_is_an_error(tmp_path):
    """Fail loudly: no silent getattr fallback that would refreeze the crops."""
    from src.train import epoch_stream

    stream = epoch_stream([0], object(), start_epoch=0)
    with pytest.raises(AttributeError):
        next(stream)


# -- (b) val and test must not resample ------------------------------------


def test_val_tensors_are_identical_across_epochs(tmp_path):
    """Byte-identical, not merely similar -- compared as tensors, not origins."""
    _, patches = _patches(tmp_path, "grid", "val")

    patches.set_epoch(0)
    before = [(patches[i]["lr"].clone(), patches[i]["hr"].clone()) for i in range(len(patches))]

    for epoch in (1, 2, 17):
        patches.set_epoch(epoch)
        assert patches.epoch == 0, (
            f"set_epoch({epoch}) moved a grid-mode split off epoch 0; the "
            f"validation set would resample and every val point would measure "
            f"different data."
        )
        for i, (lr0, hr0) in enumerate(before):
            item = patches[i]
            assert torch.equal(item["lr"], lr0)
            assert torch.equal(item["hr"], hr0)


def test_val_origins_are_fixed_across_epochs(tmp_path):
    _, patches = _patches(tmp_path, "grid", "val")
    patches.set_epoch(0)
    first = _origins(patches)
    patches.set_epoch(9)
    assert _origins(patches) == first


def test_a_negative_epoch_is_an_error(tmp_path):
    _, patches = _patches(tmp_path, "random", "train")
    with pytest.raises(ValueError, match="epoch must be >= 0"):
        patches.set_epoch(-1)


# -- (c) the A2-vs-B guarantee ---------------------------------------------


def test_two_runs_at_one_seed_produce_identical_crops_for_epochs_0_to_3(tmp_path):
    """THE control-validity test. Run A2 and Run B must see the same data stream.

    Two independently constructed datasets stand in for two runs. If their crop
    origins diverge at any epoch, then any difference between A2's and B's
    curves could be the data rather than the change under test, and the
    headline "augmentation and capacity fixed the overfit" claim is unsupported.
    """
    _, run_a2 = _patches(tmp_path / "a2", "random", "train")
    _, run_b = _patches(tmp_path / "b", "random", "train")

    for epoch in range(4):
        run_a2.set_epoch(epoch)
        run_b.set_epoch(epoch)
        assert _origins(run_a2) == _origins(run_b), (
            f"crop origins diverged at epoch {epoch} between two runs at the "
            f"same seed. The A2-vs-B comparison is invalid."
        )


# -- (c2) batch composition, not just crop content -------------------------


class _IndexDataset(torch.utils.data.Dataset):
    """A dataset whose items ARE their indices, so a batch shows its own order.

    Stands in for ``SRPatchDataset`` in the two properties that matter here: it
    exposes ``set_epoch`` and it carries a ``seed`` attribute holding the DATA
    seed (``cfg.seed``), which is what ``src/train.py`` derives the shuffle
    stream from.
    """

    def __init__(self, length: int = 64, seed: int = 42) -> None:
        self.length = int(length)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return int(index)


def _batch_indices(epochs=4, length=64, batch=8, use_generator=True):
    """The index sequence a run would train on, as ``[(epoch, [indices]), ...]``.

    Built through the REAL ``epoch_stream`` and a real ``DataLoader``, because
    the property under test is a property of that wiring, not of ``derive_seed``
    -- which already has its own test.
    """
    from torch.utils.data import DataLoader

    from src.train import epoch_stream

    dataset = _IndexDataset(length)
    generator = torch.Generator() if use_generator else None
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )
    stream = epoch_stream(
        loader,
        dataset,
        0,
        shuffle_generator=generator,
        shuffle_seed=dataset.seed if use_generator else None,
    )
    return [
        (epoch, [int(i) for i in batch_tensor])
        for epoch, batch_tensor in (
            next(stream) for _ in range(epochs * (length // batch))
        )
    ]


def test_two_runs_at_one_seed_produce_identical_batches_for_epochs_0_to_3():
    """THE determinism-hardening test. A2 and B1 must compose batches identically."""
    assert _batch_indices() == _batch_indices()


def test_batch_composition_does_not_depend_on_the_global_rng():
    """The actual hole this closed.

    Two runs whose global torch RNG is in different states -- which is what
    happens the moment an objective consumes a random number, as the
    heteroscedastic-NLL head will -- must still see the same batches. Without
    the explicit generator this fails, which is the point: the guarantee is now
    structural rather than inherited from "neither loss samples".
    """
    torch.manual_seed(0)
    first = _batch_indices()
    torch.manual_seed(999)
    torch.rand(17)  # advance the global stream by an arbitrary amount
    second = _batch_indices()
    assert first == second


def test_without_the_generator_the_global_rng_does_change_the_batches():
    """Proves the test above has teeth rather than passing vacuously."""
    torch.manual_seed(0)
    first = _batch_indices(use_generator=False)
    torch.manual_seed(999)
    second = _batch_indices(use_generator=False)
    assert first != second, (
        "batch order was unaffected by the global RNG even without an explicit "
        "generator, so test_batch_composition_does_not_depend_on_the_global_rng "
        "proves nothing. Check that shuffle=True is still in effect."
    )


def test_batches_are_reshuffled_every_epoch():
    """Deterministic must not mean frozen -- that was the Run A failure."""
    sequence = _batch_indices()
    by_epoch = {}
    for epoch, indices in sequence:
        by_epoch.setdefault(epoch, []).extend(indices)
    assert sorted(by_epoch) == [0, 1, 2, 3]
    orders = [tuple(by_epoch[e]) for e in sorted(by_epoch)]
    assert len(set(orders)) == 4, "an epoch repeated another epoch's batch order"
    for order in orders:
        assert sorted(order) == list(range(64)), "an epoch did not cover the split"


def test_a_generator_without_a_seed_is_refused():
    """Fail loudly: a generator seeded from nothing looks like determinism."""
    from src.train import epoch_stream

    stream = epoch_stream(
        [0], _IndexDataset(), 0, shuffle_generator=torch.Generator(), shuffle_seed=None
    )
    with pytest.raises(ValueError, match="shuffle_seed"):
        next(stream)


def test_the_shuffle_stream_is_independent_of_the_crop_stream():
    """Streams are separated by name, so changing one cannot shift the other."""
    assert derive_seed(42, 0, 0, "shuffle") != derive_seed(42, 0, 0, "crop")
    assert derive_seed(42, 0, 0, "shuffle") != derive_seed(42, 0, 0, "augment")
    assert derive_seed(42, 0, 0, "shuffle") != derive_seed(42, 1, 0, "shuffle")


def test_crops_do_not_depend_on_worker_count(tmp_path):
    """Same guarantee, stated against the thing most likely to break it.

    The seed is a pure function of (seed, epoch, index), so it cannot depend on
    which worker drew the item or in what order. Checked by drawing the items
    out of order, which is what a multi-worker loader does.
    """
    _, patches = _patches(tmp_path, "random", "train")
    patches.set_epoch(3)
    in_order = _origins(patches)

    shuffled = list(range(len(patches)))[::-1]
    out_of_order = {i: (patches[i]["lr_row"], patches[i]["lr_col"]) for i in shuffled}
    assert [out_of_order[i] for i in range(len(patches))] == in_order


def test_derive_seed_is_a_pure_function_of_the_triple():
    """No PYTHONHASHSEED salt, no process state, and streams stay independent."""
    assert derive_seed(1337, 2, 5, "crop") == derive_seed(1337, 2, 5, "crop")
    assert derive_seed(1337, 2, 5, "crop") != derive_seed(1337, 2, 5, "augment")
    assert derive_seed(1337, 2, 5, "crop") != derive_seed(1337, 3, 5, "crop")
    assert derive_seed(1337, 2, 5, "crop") != derive_seed(1337, 2, 6, "crop")
    assert derive_seed(1337, 2, 5, "crop") != derive_seed(42, 2, 5, "crop")
    # Must fit torch's signed 64-bit manual_seed and numpy's unsigned range.
    for triple in ((1337, 0, 0), (2**31, 10**6, 10**6)):
        value = derive_seed(*triple, "crop")
        assert 0 <= value < 2**63
        torch_generator(*triple, "crop")
        numpy_generator(*triple, "crop")


def test_augmentation_draws_are_reproducible_and_epoch_dependent():
    """The augmentation stream carries the same guarantee as the crop stream."""
    from src.data.augment import augment_pair

    lr = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(4, 8, 8)
    hr = torch.arange(4 * 32 * 32, dtype=torch.float32).reshape(4, 32, 32)

    def draw(epoch, index):
        out, _ = augment_pair(lr, hr, generator=torch_generator(1337, epoch, index, "augment"))
        return out

    assert torch.equal(draw(0, 5), draw(0, 5)), "same triple must give the same transform"
    over_epochs = [draw(e, 5) for e in range(16)]
    assert any(not torch.equal(over_epochs[0], x) for x in over_epochs[1:]), (
        "the transform never changed across 16 epochs; augmentation is frozen."
    )


# -- (d) bounds ------------------------------------------------------------


def test_crop_origins_stay_in_bounds_for_the_smallest_tile(tmp_path):
    """Origins must leave a full patch inside the tile, at every epoch.

    Checked against the SMALLEST tile the split actually holds, not the
    configured size: an off-by-one here reads past the array on exactly the
    tile that is hardest to notice, and the HR read is 4x further out.
    """
    cfg, patches = _patches(tmp_path, "random", "train")
    source = get_dataset(cfg)
    smallest = min(
        min(int(source[i]["lr"].shape[-2]), int(source[i]["lr"].shape[-1]))
        for i in patches.indices
    )
    lr_size, scale = patches.lr_size, patches.scale
    assert smallest >= lr_size, "the stub must emit tiles at least one patch wide"

    for epoch in range(6):
        patches.set_epoch(epoch)
        for i in range(len(patches)):
            item = patches[i]
            row, col = item["lr_row"], item["lr_col"]
            assert 0 <= row <= smallest - lr_size, f"epoch {epoch} item {i}: row {row}"
            assert 0 <= col <= smallest - lr_size, f"epoch {epoch} item {i}: col {col}"
            assert item["hr_row"] == row * scale and item["hr_col"] == col * scale
            assert item["lr"].shape[-2:] == (lr_size, lr_size)
            assert item["hr"].shape[-2:] == (lr_size * scale, lr_size * scale)


def test_crop_origins_cover_more_than_one_position_over_many_epochs(tmp_path):
    """Bounds must not be satisfied by always returning the same corner."""
    _, patches = _patches(tmp_path, "random", "train")
    seen = set()
    for epoch in range(24):
        patches.set_epoch(epoch)
        seen.update(_origins(patches))
    assert len(seen) > 8, f"only {len(seen)} distinct crop origins over 24 epochs"


# -- provenance ------------------------------------------------------------


def _args(**overrides):
    """Parsed args from the REAL parser, so a new flag cannot be forgotten here.

    Building a bare ``Namespace`` was how this test used to work, and it broke
    the moment ``write_run_metadata`` started reading a field the Namespace did
    not carry -- a test failure standing in for what would otherwise have been
    a crash on Kaggle after the queue wait.
    """
    from src.train import build_parser

    args = build_parser().parse_args(
        ["--data-module", "m", "--data-class", "C", "--data-root", "auto"]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_run_metadata_records_config_hash_and_commit(tmp_path):
    """Behaviour now depends on code as well as config; both must be recorded."""
    from src.config import FROZEN_CONFIG, HASH_FIELD, load_frozen
    from src.train import write_run_metadata

    meta = write_run_metadata(tmp_path, _args(seed=1337, out=str(tmp_path)))

    assert (tmp_path / "run_metadata.json").is_file()
    assert meta["config_hash"] == str(load_frozen(FROZEN_CONFIG)[HASH_FIELD])
    assert meta["git"]["commit"] != ""
    assert "dirty" in meta["git"]
    assert set(meta) >= {"config_hash", "git", "seed", "started_utc", "torch"}


def test_run_metadata_records_the_lambdas_prominently(tmp_path):
    """The ONLY record of what B1 and B2 optimised.

    ``configs/frozen_day3.yaml`` carries 0.0/0.0 because it describes the
    CONTROL, so ``config_hash`` is byte-identical across A2, B1 and B2 and
    cannot tell them apart. If the lambdas are not in the metadata, nothing
    anywhere says which run was which.
    """
    import json

    from src.train import write_run_metadata

    meta = write_run_metadata(
        tmp_path, _args(spectral_lambda1=0.1, spectral_lambda2=0.02)
    )
    assert meta["lambdas"] == {"spectral_lambda1": 0.1, "spectral_lambda2": 0.02}
    assert meta["spectral_on"] is True
    assert meta["git_sha"] == meta["git"]["commit"]

    on_disk = json.loads((tmp_path / "run_metadata.json").read_text())
    assert on_disk["lambdas"]["spectral_lambda1"] == 0.1

    control = write_run_metadata(tmp_path, _args())
    assert control["lambdas"] == {"spectral_lambda1": 0.0, "spectral_lambda2": 0.0}
    assert control["spectral_on"] is False, (
        "the control must record that it was a control, not merely omit the "
        "lambdas -- an absent key is answerable only by reading argparse."
    )


def test_dirty_file_paths_are_not_truncated(tmp_path):
    """The first dirty path must survive intact.

    ``git status --porcelain`` is "XY<space><path>", so an unstaged change's
    line STARTS with a space. Stripping the command's whole output removed that
    space from the first line only, and the fixed ``line[3:]`` slice then ate
    the first character of the first filename -- "onfigs/base.yaml". It never
    failed anything; it just quietly put a filename that does not exist into the
    provenance record of every run started from a dirty tree.
    """
    import subprocess

    from src.utils.gitmeta import git_metadata

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    for name in ("aaa.txt", "bbb.txt"):
        (tmp_path / name).write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
    )

    # Unstaged modifications: the " M path" form that triggered the bug.
    for name in ("aaa.txt", "bbb.txt"):
        (tmp_path / name).write_text("two\n", encoding="utf-8")

    meta = git_metadata(root=tmp_path)
    assert meta["dirty"] is True
    assert meta["dirty_files"] == ["aaa.txt", "bbb.txt"], (
        "a dirty path was truncated; the porcelain leading space was stripped."
    )


def test_git_metadata_never_reports_a_clean_tree_it_could_not_check(tmp_path):
    """"Unknown" and "clean" are different facts; the comparison depends on it."""
    from src.utils.gitmeta import NO_GIT, git_metadata

    meta = git_metadata(root=tmp_path)  # not a repository
    assert meta["commit"] == NO_GIT
    assert meta["dirty"] is None, "an unknown tree must not be recorded as clean"
