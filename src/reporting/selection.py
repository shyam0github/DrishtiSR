"""Within-run checkpoint selection: the repo's pre-registered rule, wrapped.

The rule itself is NOT reimplemented here. It is
:func:`src.eval.ckpt_selection.select_checkpoint` with the parameters of
``configs/base.yaml`` ``eval_all_ckpts.selection``: lowest mean opensr-test
``reflectance`` consistency error; checkpoints whose paired bootstrap CI on
``(candidate - minimum)`` contains 0 are tied with it; LPIPS (lower) breaks the
tie. This module only

* finds the per-pair VAL tables that ``scripts/eval_all_ckpts.py`` cached for a
  run (``<cache>/<run>/itNNNNN/per_pair.csv``),
* refuses anything that is not VAL (selection never touches test),
* maps the chosen iteration back to a checkpoint file, and
* records whether ``best.pt`` (the training loop's PSNR pick) coincides.

The superseded "sel-v1" rule (0.85 HF / Sobel ratio vs GT guard, 0.002 LPIPS
band) is deliberately absent: measured HF/GT ratios are 0.06-0.09 for every
model including bicubic, so that guard rejects everything (RULES.md, Day 3
addendum). Only its output layout is kept.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import pandas as pd

from src.eval.ckpt_selection import select_checkpoint

__all__ = [
    "RULE_NAME",
    "ITER_DIR",
    "load_cached_candidates",
    "checkpoint_iterations",
    "select_run",
]

#: Name written into every selection record; the parameters travel alongside.
RULE_NAME = "day3-preregistered"

#: Cache directory name of one evaluated checkpoint, e.g. ``it12000``.
ITER_DIR = re.compile(r"^it(\d+)$")


def load_cached_candidates(cache_root: Path, run: str) -> Dict[str, pd.DataFrame]:
    """Read every cached per-pair VAL table of one run.

    Args:
        cache_root: The ``eval_all_ckpts`` cache root (holds one folder per run).
        run: Run label, e.g. ``"a2"``.

    Returns:
        ``{"itNNNNN": per-pair frame}`` ordered by iteration. Frames hold one row
        per VAL patch (metric columns are float64 means per patch, reflectance
        domain where applicable).

    Raises:
        FileNotFoundError: The run has no cached tables.
        ValueError: A table holds rows from a split other than ``val``.
    """
    run_dir = Path(cache_root) / run
    found = []
    if run_dir.is_dir():
        for child in run_dir.iterdir():
            match = ITER_DIR.match(child.name)
            if match and (child / "per_pair.csv").is_file():
                found.append((int(match.group(1)), child))
    if not found:
        raise FileNotFoundError(f"no cached per_pair.csv tables for run {run!r} under {run_dir}")
    frames: Dict[str, pd.DataFrame] = {}
    for _, child in sorted(found):
        frame = pd.read_csv(child / "per_pair.csv")
        if "split" in frame.columns:
            splits = set(frame["split"].astype(str))
            if splits != {"val"}:
                raise ValueError(
                    f"{child / 'per_pair.csv'} holds splits {sorted(splits)}; checkpoint "
                    "selection runs on VAL only and never on test."
                )
        frames[child.name] = frame
    return frames


def _torch_iteration(path: Path) -> int:
    """1-based iteration stored in a training checkpoint (``payload['it'] + 1``)."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "it" not in payload:
        raise KeyError(f"{path} has no 'it' entry; cannot map it to an iteration.")
    return int(payload["it"]) + 1


def checkpoint_iterations(
    run_dir: Path, reader: Optional[Callable[[Path], int]] = None
) -> Dict[str, int]:
    """Map each ``*.pt`` in a run directory to its 1-based iteration.

    Args:
        run_dir: Main-tree run directory (read only).
        reader: Returns the iteration of one file; defaults to reading the
            checkpoint's ``it`` field (0-based in ``src/train.py``, hence +1).

    Returns:
        ``{file name: iteration}``; empty when the directory has no checkpoints.
    """
    reader = reader or _torch_iteration
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return {}
    return {p.name: reader(p) for p in sorted(run_dir.glob("*.pt"))}


def select_run(
    run: str,
    run_dir: Path,
    cache_root: Path,
    rule_cfg: Mapping[str, Any],
    seed: int,
    iteration_reader: Optional[Callable[[Path], int]] = None,
) -> Dict[str, Any]:
    """Apply the pre-registered rule to one run and build its selection record.

    Args:
        run: Run label (cache folder name), e.g. ``"a2"``.
        run_dir: Run directory holding ``best.pt`` / ``last.pt`` (read only).
        cache_root: ``eval_all_ckpts`` per-pair cache root.
        rule_cfg: ``cfg.eval_all_ckpts.selection`` (``metric``, ``tie_metric``,
            ``tie_better``, ``n_boot``, ``ci``, optional ``provenance``).
        seed: Bootstrap seed (``cfg.seed``).
        iteration_reader: Override for :func:`checkpoint_iterations` (tests).

    Returns:
        JSON-ready record: ``run, rule, rule_params, source, split, n_pairs,
        candidates[...], chosen{label, iteration, path}, best_pt{...}``.
    """
    frames = load_cached_candidates(cache_root, run)
    result = select_checkpoint(
        frames,
        metric=str(rule_cfg["metric"]),
        tie_metric=str(rule_cfg["tie_metric"]),
        tie_better=str(rule_cfg["tie_better"]),
        n_boot=int(rule_cfg["n_boot"]),
        ci=float(rule_cfg["ci"]),
        seed=int(seed),
    )
    iters = checkpoint_iterations(run_dir, iteration_reader)
    by_iter: Dict[int, str] = {}
    for name, it in iters.items():
        by_iter.setdefault(it, name)

    candidates = []
    for row in result["rows"]:
        it = int(ITER_DIR.match(row["label"]).group(1))
        file_name = by_iter.get(it)
        candidates.append(
            {
                **row,
                "iteration": it,
                "path": str(Path(run_dir) / file_name) if file_name else None,
                "n_pairs": int(len(frames[row["label"]])),
            }
        )
    chosen = next(c for c in candidates if c["selected"])
    best_it = iters.get("best.pt")
    return {
        "what": "P7 within-run checkpoint selection (pre-registered Day 3 rule, wrapped)",
        "run": run,
        "rule": RULE_NAME,
        "rule_params": {
            **result["rule"],
            "n_boot": int(rule_cfg["n_boot"]),
            "ci": float(rule_cfg["ci"]),
            "seed": int(seed),
            "config_block": "configs/base.yaml eval_all_ckpts.selection",
            "implementation": "src.eval.ckpt_selection.select_checkpoint",
            "provenance": str(rule_cfg.get("provenance", "")),
        },
        "superseded": "sel-v1 (0.85 HF/Sobel vs GT guard, 0.002 LPIPS band) not applied: RULES.md Day 3 addendum",
        "source": str(Path(cache_root) / run),
        "split": "val",
        "n_pairs": chosen["n_pairs"],
        "minimum": result["minimum"],
        "tied_set": result["tied_set"],
        "candidates": candidates,
        "chosen": {"label": chosen["label"], "iteration": chosen["iteration"], "path": chosen["path"]},
        "best_pt": {
            "path": str(Path(run_dir) / "best.pt") if best_it is not None else None,
            "iteration": best_it,
            "selected_by": "training loop, val PSNR on 25-batch AMP subsample",
            "coincides": bool(best_it is not None and best_it == chosen["iteration"]),
        },
        "unmapped_iterations": sorted(c["iteration"] for c in candidates if c["path"] is None),
    }
