"""src/train.py keeps every --ckpt-every checkpoint, end to end, on CPU.

Day 3's trainer wrote each scheduled checkpoint to the same ``last.pt``, so A2,
B1 and B2 kept 2 of their 12 and the post-hoc selection chose between two
candidates (reports/day3_results.md). Run C's sigma-collapse diagnosis needs the
trajectory, so this runs the real ``main()`` for a few iterations on a tiny
in-memory dataset and counts the files.

CPU only, no data, no network: the dataset is synthetic and LPIPS -- whose
backbone weights would be a download -- is stubbed; nothing here is about LPIPS.
"""

import csv
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import src.train as train
from src.utils.config import load_config

ITERS, EVERY = 6, 2


class _TinyPairs(torch.utils.data.Dataset):
    """Satisfies src/train.py's dataset contract with 8 synthetic pairs.

    ``hr``: float32 ``(4, 32, 32)`` surface reflectance in [0.05, 0.45];
    ``lr``: its exact 4x4 block mean, float32 ``(4, 8, 8)``.
    """

    def __init__(self, root, split, patch_lr, scale):
        self.cfg = load_config("configs/base.yaml")
        self.seed = int(self.cfg.seed)
        g = torch.Generator().manual_seed(0 if split == "train" else 1)
        size = patch_lr * scale
        self.hr = torch.rand(8, 4, size, size, generator=g) * 0.4 + 0.05
        self.lr = F.avg_pool2d(self.hr, scale)

    def set_epoch(self, epoch):
        pass

    def __len__(self):
        return self.hr.shape[0]

    def __getitem__(self, i):
        return {"lr": self.lr[i], "hr": self.hr[i]}


def _run(monkeypatch, out, *extra):
    monkeypatch.setattr(train, "load_dataset_cls", lambda module, cls: _TinyPairs)
    monkeypatch.setattr(
        train, "lpips",
        lambda sr, hr, **kw: SimpleNamespace(distance=np.zeros(sr.shape[0])),
    )
    monkeypatch.setattr(sys, "argv", [
        "train", "--data-module", "unused", "--data-class", "unused", "--data-root", "unused",
        "--out", str(out), "--iters", str(ITERS), "--batch", "2", "--patch-lr", "8",
        "--n-resblocks", "1", "--n-feats", "8", "--warmup", "1", "--workers", "0",
        "--val-every", str(EVERY), "--val-batches", "1", "--metric-batches", "1",
        "--ckpt-every", str(EVERY), "--log-every", str(EVERY), "--amp", "0",
        "--resume", "none", *extra,
    ])
    train.main()


def _weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_every_scheduled_checkpoint_persists_under_its_own_name(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path)

    expected = [train.checkpoint_name(i) for i in range(EVERY, ITERS + 1, EVERY)]
    assert expected == ["ckpt_it000002.pt", "ckpt_it000004.pt", "ckpt_it000006.pt"]
    found = sorted(p.name for p in tmp_path.glob("ckpt_it*.pt"))
    assert found == expected, f"expected {len(expected)} distinct checkpoints, found {found}"

    payloads = [_weights(tmp_path / name) for name in expected]
    # Each file is the checkpoint of ITS iteration, not a copy of the last one.
    assert [p["it"] + 1 for p in payloads] == [2, 4, 6]
    assert [p["val_metrics_iter"] for p in payloads] == [2, 4, 6]
    for a, b in zip(payloads, payloads[1:]):
        assert any(not torch.equal(a["model"][k], b["model"][k]) for k in a["model"]), (
            "two scheduled checkpoints hold identical weights"
        )

    # last.pt is still the resume pointer, and equals the final checkpoint.
    last = _weights(tmp_path / "last.pt")
    assert last["it"] == payloads[-1]["it"]
    assert all(torch.equal(last["model"][k], payloads[-1]["model"][k]) for k in last["model"])


def test_a_time_budget_stop_is_stamped_with_the_iteration_reached(monkeypatch, tmp_path):
    """Regression: the post-loop save used to claim args.iters - 1 after a break."""
    _run(monkeypatch, tmp_path, "--max-hours", "0")
    assert _weights(tmp_path / "last.pt")["it"] == 0


@pytest.mark.parametrize("detach, joint_expected", [("1", False), ("0", True)])
def test_the_nll_gradient_ratio_is_logged_in_both_modes(monkeypatch, tmp_path, detach, joint_expected):
    _run(monkeypatch, tmp_path, "--uncertainty", "1", "--var-feats", "4",
         "--nll-warmup", "0", "--nll-ramp", "0", "--nll-weight", "0.1",
         "--nll-detach-sr", detach)
    with (tmp_path / "log.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["val_psnr"]]
    assert [int(r["iter"]) for r in rows] == [2, 4, 6]
    for row in rows:
        raw, applied = float(row["nll_grad_ratio_raw"]), float(row["nll_grad_ratio_applied"])
        assert raw > 0.0
        if joint_expected:
            assert applied == pytest.approx(float(row["nll_weight"]) * raw, rel=1e-4)
        else:
            assert applied == 0.0
    meta = _weights(tmp_path / "last.pt")["uncertainty"]
    assert meta["nll_detach_sr"] is (detach == "1")
