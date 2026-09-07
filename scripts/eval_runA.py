"""Score the Run A EDSR checkpoint against the bicubic floor, on the full val split.

Run A is the EDSR-baseline L1 reference run: 40k iterations, no uncertainty
head, no spectral loss. This script decides whether it earned its place, and the
decision is a **hard gate** -- the model must beat bicubic on PSNR (up), SSIM
(up) and LPIPS (down), all three, on the whole validation split. Two of three is
a fail. The gate criterion lives in ``cfg.eval_runA.gate`` so it is visible next
to the numbers it judges rather than buried in an ``if``.

Why bicubic is re-scored here instead of quoted
-----------------------------------------------
``outputs/metrics/baseline_bicubic.json`` already holds a bicubic number from
Day 1. This script ignores it and computes bicubic again, in the same pass, from
the same loader, with the same :class:`~src.metrics.aggregate.Evaluator`. The
stored number and a fresh model number would only be comparable if the split,
the patch grid, the band order, the reflectance scale, the SSIM window and the
LPIPS backbone had all held still in between -- and nothing in a JSON file can
prove that. Recomputing costs about two minutes of CPU and removes the entire
question. The two bicubic numbers are printed side by side at the end, so a
drift shows up as a discrepancy rather than as a wrong verdict.

What "the same val split" means, concretely
-------------------------------------------
:func:`src.data.loader.build_dataloaders` reads ``outputs/splits_sen2naipv2.csv``
(written once by ``scripts/make_splits.py``, geographic and scene-grouped) and
builds the validation ``PatchDataset`` in ``grid`` mode -- deterministic,
unshuffled, undropped. That is the identical call ``scripts/run_baseline.py``
makes, and ``src/data/adapter.SRPatchDataset`` -- what Run A trained against --
reaches the same split through the same ``resolve_split_assignments``. So the
patches scored here are the patches bicubic was measured on and the patches the
model never trained on, by construction rather than by care.

Metrics, and where they are computed
------------------------------------
PSNR and SSIM are computed over **all four bands** (RGBN) and reported as the
band mean; the per-band columns are in the CSV. LPIPS runs on the **RGB bands
only** (``cfg.metrics.lpips.rgb_bands``, resolved by name), scaled to the
``[-1, 1]`` domain its AlexNet backbone expects. That scaling requires the one
clip in the metrics module, which is counted and reported -- see
``src.metrics.image_quality.lpips``. LPIPS is a perceptual proxy here, not an
authority: it ranks nearest-neighbour replication above bicubic on this very
split. It is in the gate because it was asked for, and because a model that
loses to bicubic on it while winning on PSNR is blurring; it must not be read as
the headline.

Outputs (names from ``cfg.eval_runA``):

- ``reports/day2_runA.md``   -- the table, the gate verdict, the curve reading.
- ``reports/day2_runA.json`` -- every raw number, plus the settings behind them.
- ``reports/day2_curves.png`` -- loss and validation curves (src/eval/curves.py).
- ``outputs/metrics/day2_runA_<method>.csv`` -- per-patch rows, both methods.

Exit codes: ``0`` gate passed, ``2`` gate failed or could not be evaluated,
``1`` the run itself broke. ``--smoke`` always exits ``0``; its verdict
describes the synthetic stub and a random-weight model, and means nothing.

Examples:
    # Pre-flight: fabricates a run directory, exercises every path, ~2 min, CPU.
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/eval_runA.py --smoke

    # The real thing: full validation split, both methods.
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/eval_runA.py

    # Score last.pt instead of best.pt.
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/eval_runA.py \\
        --set eval_runA.checkpoint_name=last.pt
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data.loader import build_dataloaders  # noqa: E402
from src.data.registry import get_dataset  # noqa: E402
from src.eval.baselines import get_baseline  # noqa: E402
from src.eval.curves import plot_training_curves  # noqa: E402
from src.metrics.aggregate import Evaluator  # noqa: E402
from src.metrics.image_quality import LPIPS_CAVEAT  # noqa: E402
from src.models.edsr import build_model, count_params  # noqa: E402
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root, resolve_output_path  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

# Architecture keys read out of the checkpoint's own "args" dict. The checkpoint
# records what the run actually used; a config value here would be a second,
# unverifiable claim about a file that already carries the truth.
_ARCH_KEYS = ("scale", "n_resblocks", "n_feats", "in_ch")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Run A EDSR checkpoint and bicubic over the full "
            "validation split, apply the hard gate, and write the Day 2 report."
        ),
    )
    add_standard_args(parser)
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Override cfg.eval_runA.run_dir -- the directory holding best.pt, "
            "args.json and log.csv."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Stop after this many validation batches. DEBUGGING ONLY: the gate "
            "then judges those batches, not the split, so the outputs are "
            "written to suffixed paths and can never overwrite the real report."
        ),
    )
    return parser.parse_args(argv)


def resolve_run_dir(cfg: Any, override: Optional[str]) -> Path:
    """Locate the run directory and verify it holds what this script needs.

    Args:
        cfg: Loaded config; reads ``eval_runA.run_dir`` and the three file names.
        override: ``--run-dir`` value, or None to use the config.

    Returns:
        The directory, resolved against the repository root when relative.

    Raises:
        FileNotFoundError: The directory or any of the three required files is
            missing. Checked up front, together, so a missing log.csv is not
            discovered after a full evaluation pass has already been paid for.
    """
    block = cfg["eval_runA"]
    raw = Path(str(override if override is not None else block["run_dir"]))
    run_dir = raw if raw.is_absolute() else repo_root() / raw

    required = [
        run_dir / str(block["checkpoint_name"]),
        run_dir / str(block["args_name"]),
        run_dir / str(block["log_csv_name"]),
    ]
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory {run_dir} does not exist. Fetch the Kaggle output "
            "with 'python scripts/kaggle_run.py fetch --job runa' and copy "
            f"best.pt, args.json and log.csv into {run_dir}."
        )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{run_dir} is missing {[p.name for p in missing]}. This script "
            "needs the checkpoint, the run's args.json, and log.csv; they are "
            "all in the fetched Kaggle output under runs/runA/."
        )
    return run_dir


def fabricate_run(cfg: Any, run_dir: Path, logger: Any) -> Path:
    """Write a complete, fake run directory for ``--smoke``.

    There is no trained checkpoint for the synthetic stub. Rather than special-
    case the loading path -- the part most likely to break when a checkpoint
    format changes, and so the part least worth skipping -- smoke writes a real
    checkpoint of a freshly initialised model, a real ``args.json``, and a real
    ``log.csv``, then reads them back through the ordinary code path.

    Args:
        cfg: Loaded config. Reads the smoke-merged ``eval_runA`` block, plus
            ``sr.scale`` and ``dataset.bands`` for the fabricated architecture.
        run_dir: Directory to create and fill.
        logger: Logger.

    Returns:
        ``run_dir``.

    Raises:
        RuntimeError: Called when ``cfg.eval_runA.fabricate_run`` is not set,
            which would mean a real run was about to be overwritten with noise.
    """
    block = cfg["eval_runA"]
    if not bool(block.get("fabricate_run", False)):
        raise RuntimeError(
            "fabricate_run() called but cfg.eval_runA.fabricate_run is not "
            "true. This function writes random weights; it must never run "
            "against a real run directory."
        )

    logger.warning(
        "SMOKE: fabricating a run directory at %s with RANDOM weights. Every "
        "number this run produces describes the synthetic stub and an "
        "untrained model.",
        run_dir,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    scale = int(cfg["sr"]["scale"])
    channels = len(cfg["dataset"]["bands"])
    # Deliberately tiny: smoke proves the path, and a 1.5 M-parameter forward
    # pass over the stub's patches is minutes that buy nothing.
    args = {
        "scale": scale,
        "n_resblocks": 2,
        "n_feats": 16,
        "in_ch": channels,
        "iters": int(block["fabricate_iters"]),
        "batch": int(cfg["train"]["batch_size"]),
        "lr": 2e-4,
        "seed": int(cfg["seed"]),
        "fabricated_by": "scripts/eval_runA.py --smoke",
    }
    model = build_model(
        str(block["model_name"]),
        scale=scale,
        n_resblocks=args["n_resblocks"],
        n_feats=args["n_feats"],
        in_ch=channels,
        out_ch=channels,
    )
    torch.save(
        {"model": model.state_dict(), "it": args["iters"] - 1, "best": 0.0,
         "args": args},
        run_dir / str(block["checkpoint_name"]),
    )
    (run_dir / str(block["args_name"])).write_text(
        json.dumps(args, indent=2), encoding="utf-8"
    )

    # A log with both row kinds and a peak that is NOT at the end, so the curve
    # reading and its "still improving" branch are both exercised.
    lines = ["iter,loss,lr,val_psnr,sec_per_100it"]
    total = int(block["fabricate_iters"])
    for step in range(100, total + 1, 100):
        decay = 0.05 * (0.9 ** (step / 200.0))
        lines.append(f"{step},{decay:.6f},{2e-4 * (1 - step / total):.3e},,12.0")
        if step % 300 == 0:
            peak = 20.0 + 2.0 * (step / total) - 3.0 * (step / total) ** 2
            lines.append(f"{step},,,{peak:.4f},")
    (run_dir / str(block["log_csv_name"])).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return run_dir


def load_checkpoint_model(cfg: Any, checkpoint_path: Path, logger: Any):
    """Rebuild the trained model from a checkpoint written by ``src/train.py``.

    Args:
        cfg: Loaded config. Reads ``eval_runA.model_name``, ``dataset.bands``
            and ``runtime.max_parameters``.
        checkpoint_path: The ``.pt`` file.
        logger: Logger.

    Returns:
        ``(model, info)``. ``model`` is in ``eval()`` mode on the CPU, its
        weights loaded strictly. ``info`` records the checkpoint's iteration,
        its recorded best validation PSNR, the architecture arguments it was
        built from, the parameter count, and whether that count exceeds
        ``cfg.runtime.max_parameters``.

    Raises:
        KeyError: The checkpoint has no ``model`` or no ``args`` entry, or its
            ``args`` omits an architecture key. Raised rather than defaulted:
            guessing ``n_feats`` produces a model that loads nothing and scores
            like noise, which is a much more expensive failure than a crash.
        RuntimeError: The state dict does not fit the rebuilt architecture.
    """
    # weights_only=False: this checkpoint is our own Kaggle output and its "args"
    # entry is a plain dict that the weights-only unpickler will not restore.
    # Stated rather than left to a warning, because the flag flips default in a
    # future torch and this call must keep working.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    for key in ("model", "args"):
        if key not in payload:
            raise KeyError(
                f"{checkpoint_path} has no {key!r} entry (keys: "
                f"{sorted(payload)}). It was not written by src/train.py's "
                "save(), so this script cannot know what architecture to build."
            )
    saved_args = dict(payload["args"])
    missing = [key for key in _ARCH_KEYS if key not in saved_args]
    if missing:
        raise KeyError(
            f"{checkpoint_path}'s args is missing {missing}, so the "
            "architecture cannot be reconstructed from the run's own record. "
            "Refusing to guess: a wrong n_feats loads no weights and scores "
            "like noise."
        )

    channels = int(saved_args["in_ch"])
    expected_channels = len(cfg["dataset"]["bands"])
    if channels != expected_channels:
        raise ValueError(
            f"The checkpoint was trained on {channels} channels but "
            f"cfg.dataset.bands has {expected_channels} "
            f"({list(cfg['dataset']['bands'])}). Scoring one against the other "
            "would compare different band sets."
        )

    model = build_model(
        str(cfg["eval_runA"]["model_name"]),
        scale=int(saved_args["scale"]),
        n_resblocks=int(saved_args["n_resblocks"]),
        n_feats=int(saved_args["n_feats"]),
        in_ch=channels,
        out_ch=channels,
    )
    # strict=True is the default and is the point: a silently partial load is a
    # model that scores badly for a reason no metric can show.
    model.load_state_dict(payload["model"])
    model.eval()

    params = count_params(model)
    budget = int(cfg["runtime"]["max_parameters"])
    over_budget = params > budget

    logger.info(
        "Loaded %s: iteration %s, recorded best val PSNR %.4f dB, arch %s, "
        "%d parameters.",
        checkpoint_path,
        payload.get("it", "?"),
        float(payload.get("best", float("nan"))),
        {key: saved_args[key] for key in _ARCH_KEYS},
        params,
    )
    if over_budget:
        # NOT an exception. Run A is the EDSR reference point the submission
        # model is measured against, and refusing to score it would leave the
        # comparison unmeasured. But it does not ship at this size, so the
        # overage is logged here, recorded in the JSON, and printed in bold in
        # the report -- it must not reach a reader as a footnote.
        logger.warning(
            "PARAMETER BUDGET EXCEEDED: %d parameters against a limit of %d "
            "(cfg.runtime.max_parameters), %.2fx over. Run A is a reference "
            "measurement, not a deliverable; the submitted model must fit the "
            "budget.",
            params,
            budget,
            params / budget,
        )

    return model, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(payload["it"]) if "it" in payload else None,
        "checkpoint_recorded_best_val_psnr": (
            float(payload["best"]) if "best" in payload else None
        ),
        "architecture": {key: saved_args[key] for key in _ARCH_KEYS},
        "train_args": saved_args,
        "parameters": int(params),
        "parameter_budget": budget,
        "exceeds_parameter_budget": bool(over_budget),
    }


def make_model_sr_fn(model):
    """Wrap a model as the ``sr_fn(lr) -> sr`` the Evaluator expects.

    Args:
        model: A module in ``eval()`` mode taking ``(B, C, h, w)`` float32
            surface reflectance -- nominally ``[0, 1]`` and UNCLIPPED, exactly
            what the loader emits and what training consumed -- and returning
            ``(B, C, h*scale, w*scale)`` in the same units and band order.

    Returns:
        The callable. It does not normalise, standardise, or clip: the model was
        trained on raw reflectance and the spectral-consistency objective needs
        those units intact end to end. The Evaluator already calls it inside
        ``torch.no_grad()``.
    """

    def sr_fn(lr: torch.Tensor) -> torch.Tensor:
        return model(lr)

    sr_fn.__name__ = "edsr_runA"
    sr_fn.__doc__ = (
        "Run A EDSR forward pass. Takes and returns float32 surface "
        "reflectance, (B, C, H, W), unclipped."
    )
    return sr_fn


def apply_gate(cfg: Any, results: Dict[str, Any], model_label: str,
               bicubic_label: str, logger: Any) -> Dict[str, Any]:
    """Compare the model to bicubic on every gate metric and decide pass/fail.

    Args:
        cfg: Loaded config; reads ``eval_runA.gate``, a mapping of summary
            column name to ``"higher"`` or ``"lower"``.
        results: ``{label: EvaluationResult}`` for both methods.
        model_label: Key of the model result.
        bicubic_label: Key of the baseline result.
        logger: Logger.

    Returns:
        ``{"passed": bool, "evaluable": bool, "metrics": [...]}``. Each entry in
        ``metrics`` has the column, the direction, both means, the delta
        (model minus bicubic), whether that metric passed, and -- when a metric
        could not be computed at all -- a ``reason``.

        ``passed`` is True only when every gate metric was computed AND improved.
        A metric that could not be computed makes ``evaluable`` False and
        ``passed`` False: an unmeasured criterion is not a satisfied one.

    Raises:
        ValueError: A gate direction is neither ``"higher"`` nor ``"lower"``.
    """
    gate_cfg = cfg["eval_runA"]["gate"]
    model_summary = results[model_label].summary
    bicubic_summary = results[bicubic_label].summary

    entries: List[Dict[str, Any]] = []
    evaluable = True
    passed = True

    for column, direction in gate_cfg.items():
        column, direction = str(column), str(direction)
        if direction not in ("higher", "lower"):
            raise ValueError(
                f"cfg.eval_runA.gate[{column!r}] is {direction!r}; it must be "
                "'higher' or 'lower'."
            )

        entry: Dict[str, Any] = {
            "metric": column,
            "better": direction,
            "bicubic": None,
            "model": None,
            "delta_model_minus_bicubic": None,
            "passed": False,
            "reason": None,
        }

        if column not in model_summary or column not in bicubic_summary:
            entry["reason"] = (
                f"{column!r} is absent from the summary -- the metric was not "
                "computed (check cfg.metrics.enabled and, for lpips, "
                "cfg.metrics.lpips.enabled)."
            )
            logger.error(
                "GATE METRIC %r NOT COMPUTED. The gate cannot pass: an "
                "unmeasured criterion is not a satisfied one.", column
            )
            evaluable = False
            passed = False
            entries.append(entry)
            continue

        model_value = float(model_summary[column]["mean"])
        bicubic_value = float(bicubic_summary[column]["mean"])
        delta = model_value - bicubic_value
        improved = delta > 0.0 if direction == "higher" else delta < 0.0

        entry.update(
            {
                "bicubic": bicubic_value,
                "model": model_value,
                "delta_model_minus_bicubic": delta,
                "passed": bool(improved),
            }
        )
        if not improved:
            passed = False
            entry["reason"] = (
                f"{column} {model_value:.6g} vs bicubic {bicubic_value:.6g}: "
                f"{delta:+.6g}, and {direction} is better."
            )
        entries.append(entry)

    logger.info(
        "Gate %s (%d/%d metrics improved).",
        "PASSED" if passed else "FAILED",
        sum(1 for e in entries if e["passed"]),
        len(entries),
    )
    return {"passed": bool(passed and evaluable), "evaluable": evaluable,
            "metrics": entries}


def gate_table(gate: Dict[str, Any], model_label: str) -> str:
    """Render the gate comparison as a markdown table.

    Args:
        gate: :func:`apply_gate` output.
        model_label: Column heading for the model.

    Returns:
        A markdown table: metric, direction, bicubic, model, delta, verdict.
    """
    lines = [
        f"| metric | better | bicubic | {model_label} | delta | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for entry in gate["metrics"]:
        if entry["model"] is None:
            lines.append(
                f"| `{entry['metric']}` | {entry['better']} | - | - | - | "
                "**NOT COMPUTED** |"
            )
            continue
        lines.append(
            f"| `{entry['metric']}` | {entry['better']} | "
            f"{entry['bicubic']:.4f} | {entry['model']:.4f} | "
            f"{entry['delta_model_minus_bicubic']:+.4f} | "
            f"{'PASS' if entry['passed'] else '**FAIL**'} |"
        )
    return "\n".join(lines)


def build_report(cfg: Any, results: Dict[str, Any], gate: Dict[str, Any],
                 model_info: Dict[str, Any], curve: Dict[str, Any],
                 context: Dict[str, Any]) -> str:
    """Assemble the markdown report.

    Args:
        cfg: Loaded config.
        results: ``{label: EvaluationResult}``.
        gate: :func:`apply_gate` output.
        model_info: :func:`load_checkpoint_model` info dict.
        curve: :func:`src.eval.curves.plot_training_curves` output.
        context: Split, dataset, patch count, paths, and the stored Day 1
            bicubic number for the drift check.

    Returns:
        The full markdown document, verdict first.
    """
    block = cfg["eval_runA"]
    model_label = str(block["model_label"])
    bicubic_label = str(block["bicubic_label"])
    passed = gate["passed"]

    verdict_line = (
        f"**GATE {'PASSED' if passed else 'FAILED'}** -- "
        + (
            f"{model_label} beats bicubic on all "
            f"{len(gate['metrics'])} gate metrics "
            "(PSNR up, SSIM up, LPIPS down) over the full validation split."
            if passed
            else (
                f"{model_label} does not beat bicubic on "
                + ", ".join(
                    f"`{e['metric']}`" for e in gate["metrics"] if not e["passed"]
                )
                + ". It does not ship."
            )
        )
    )

    parts: List[str] = [
        "# Day 2 — Run A (EDSR-baseline, L1) against the bicubic floor",
        "",
        verdict_line,
        "",
        f"_Written {context['written_utc']}._ "
        f"Dataset `{context['dataset']}`, split `{context['split']}`, "
        f"**{context['num_patches']} patches** from {context['num_tiles']} tiles, "
        f"x{context['scale']}. Split source: `{context['split_source_name']}`. "
        f"Evaluated on {context['device']} with "
        f"{context['torch_threads']} torch threads.",
        "",
        "---",
        "",
        "## 1. The gate",
        "",
        "The criterion, from `cfg.eval_runA.gate`: the model must improve on "
        "bicubic on **every** one of PSNR (higher), SSIM (higher) and LPIPS "
        "(lower). Two of three is a fail.",
        "",
        gate_table(gate, model_label),
        "",
    ]

    if not gate["evaluable"]:
        parts += [
            "> **A gate metric was not computed**, so the gate could not be "
            "evaluated and is recorded as a failure. An unmeasured criterion "
            "is not a satisfied one.",
            "",
        ]

    parts += [
        "## 2. Full metric tables",
        "",
        "Both methods scored by the same `src.metrics.aggregate.Evaluator`, in "
        "the same pass, over the same loader. PSNR and SSIM are means over all "
        f"{context['num_bands']} bands ({', '.join(context['bands'])}); the "
        "per-band columns are in the CSVs. LPIPS is computed on the RGB bands "
        f"only ({', '.join(context['lpips_rgb_bands'])}), scaled to [-1, 1].",
        "",
    ]
    for label in (bicubic_label, model_label):
        parts += [results[label].to_markdown(), ""]

    parts += [
        "## 3. The model",
        "",
        f"- **Architecture** — EDSR-baseline, "
        f"{model_info['architecture']['n_resblocks']} residual blocks, "
        f"{model_info['architecture']['n_feats']} features, "
        f"{model_info['architecture']['in_ch']}-channel in and out (RGBN), "
        f"x{model_info['architecture']['scale']} PixelShuffle upsampler, no "
        "batch norm.",
        f"- **Parameters** — {model_info['parameters']:,}.",
        f"- **Checkpoint** — `{Path(model_info['checkpoint']).as_posix()}`, "
        f"saved at iteration "
        f"{(model_info['checkpoint_iteration'] or 0) + 1:,} with a recorded "
        f"best validation PSNR of "
        f"{model_info['checkpoint_recorded_best_val_psnr']:.4f} dB.",
        "",
    ]

    if model_info["exceeds_parameter_budget"]:
        parts += [
            f"> **PARAMETER BUDGET EXCEEDED.** {model_info['parameters']:,} "
            f"parameters against the "
            f"{model_info['parameter_budget']:,} limit in "
            f"`cfg.runtime.max_parameters` — "
            f"{model_info['parameters'] / model_info['parameter_budget']:.2f}x "
            "over. Run A is a **reference measurement**, not a deliverable: it "
            "establishes what a conventional EDSR achieves on this split so "
            "the budgeted model can be compared to something real. The "
            "submitted model must fit the budget, and this number must not be "
            "quoted as a result without this sentence attached.",
            "",
        ]

    parts += [
        "> **Note on the checkpoint's own validation number.** The "
        f"{model_info['checkpoint_recorded_best_val_psnr']:.4f} dB recorded in "
        "the checkpoint comes from the training loop's periodic validation, "
        f"which covers only `--val-batches` "
        f"({model_info['train_args'].get('val_batches', '?')}) batches and runs "
        "under AMP. The table above is the full split in float32 and is the "
        "number that counts; they are not expected to agree exactly.",
        "",
        "## 4. Training curves",
        "",
        f"![Run A training curves]({context['curves_relative']})",
        "",
        f"**Was it still improving at "
        f"{model_info['train_args'].get('iters', '?')}k?** {curve['verdict']}",
        "",
        f"- Peak validation PSNR **{curve['best_psnr']:.4f} dB** at iteration "
        f"**{curve['best_iter']:,}**; final **{curve['final_psnr']:.4f} dB** at "
        f"**{curve['final_iter']:,}** "
        f"({curve['delta_final_minus_best']:+.4f} dB).",
        f"- Slope over the last {curve['tail_fraction']:.0%} of the run: "
        f"**{curve['tail_slope_db_per_1k']:+.4f} dB per 1k iterations**.",
        "",
        "## 5. Reproducing this",
        "",
        "```",
        "D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/eval_runA.py",
        "```",
        "",
        f"Bicubic was **re-scored in this pass**, not quoted from Day 1. The "
        f"Day 1 stored value is "
        f"{context['day1_bicubic_psnr']} dB PSNR against "
        f"{results[bicubic_label].summary['psnr_mean']['mean']:.4f} dB here"
        + (
            "; they agree, so the split and the metric settings have not moved."
            if context["day1_agrees"]
            else " — **THEY DISAGREE**, so something in the split, the patch "
            "grid, or the metric settings has changed since Day 1. Resolve "
            "that before quoting either number."
        ),
        "",
        f"Per-patch rows: `{context['bicubic_csv']}`, `{context['model_csv']}`.",
        "",
        f"> **LPIPS caveat.** {LPIPS_CAVEAT}",
        "",
    ]
    return "\n".join(parts)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("eval_runA", log_file=cfg.paths.log_file)
    seed = seed_everything(cfg.seed)
    logger.info("Seeded with %d", seed)

    torch.set_num_threads(int(cfg.runtime.num_threads))
    logger.info(
        "torch threads set to cfg.runtime.num_threads=%d (deployment target); "
        "CUDA available: %s.",
        int(cfg.runtime.num_threads),
        torch.cuda.is_available(),
    )

    block = cfg["eval_runA"]
    model_label = str(block["model_label"])
    bicubic_label = str(block["bicubic_label"])

    # -- the run directory ---------------------------------------------------
    if args.smoke and bool(block.get("fabricate_run", False)):
        raw = Path(str(args.run_dir if args.run_dir is not None else block["run_dir"]))
        smoke_dir = raw if raw.is_absolute() else repo_root() / raw
        fabricate_run(cfg, smoke_dir, logger)
    run_dir = resolve_run_dir(cfg, args.run_dir)
    logger.info("Run directory: %s", run_dir)

    # -- a truncated run must never land on the real report's path -----------
    suffix = "" if args.max_batches is None else f"_truncated{args.max_batches}"
    if suffix:
        logger.warning(
            "--max-batches %d: outputs are suffixed %r and the gate verdict "
            "describes those batches, NOT the validation split.",
            args.max_batches,
            suffix,
        )

    def _report_path(key: str) -> Path:
        raw = Path(str(block[key]))
        stem = f"{raw.stem}{suffix}{raw.suffix}"
        path = raw.with_name(stem)
        return path if path.is_absolute() else repo_root() / path

    report_md = _report_path("report_md")
    report_json = _report_path("report_json")
    curves_figure = _report_path("curves_figure")

    # -- data ----------------------------------------------------------------
    dataset = get_dataset(cfg)
    loaders = build_dataloaders(cfg, dataset=dataset, logger=logger)
    val_loader = loaders["val"]
    val_split = str(cfg.loader.val_split)
    num_patches = len(loaders["datasets"]["val"])
    num_tiles = len(loaders["indices"]["val"])
    logger.info(
        "Validation split %r: %d patches from %d tiles (split source: %s). This "
        "is the same call scripts/run_baseline.py makes.",
        val_split,
        num_patches,
        num_tiles,
        loaders["split_source"],
    )

    # -- model ---------------------------------------------------------------
    model, model_info = load_checkpoint_model(
        cfg, run_dir / str(block["checkpoint_name"]), logger
    )

    # -- evaluate both methods in one pass -----------------------------------
    evaluator = Evaluator(cfg, logger=logger, device=torch.device("cpu"))
    model.to(evaluator.device)
    metric_dir = Path(resolve_output_path(cfg, "metric_dir"))

    sr_fns = {
        bicubic_label: get_baseline("bicubic", int(cfg.sr.scale)),
        model_label: make_model_sr_fn(model),
    }
    results: Dict[str, Any] = {}
    csv_paths: Dict[str, Path] = {}
    for label, sr_fn in sr_fns.items():
        logger.info("Evaluating %r over the full %r split.", label, val_split)
        results[label] = evaluator.run(
            val_loader,
            sr_fn,
            name=label,
            max_batches=args.max_batches,
            collect_samples=0,
            split=val_split,
        )
        name = str(block["per_sample_csv_name"]).format(method=label)
        stem = Path(name)
        csv_paths[label] = results[label].to_csv(
            metric_dir / f"{stem.stem}{suffix}{stem.suffix}"
        )
        logger.info("Per-sample metrics for %r written to %s", label, csv_paths[label])

    # -- the gate ------------------------------------------------------------
    gate = apply_gate(cfg, results, model_label, bicubic_label, logger)

    # -- curves --------------------------------------------------------------
    curve = plot_training_curves(
        run_dir / str(block["log_csv_name"]),
        cfg,
        curves_figure,
        title=f"Run A — EDSR-baseline x{cfg.sr.scale}, L1, "
              f"{model_info['train_args'].get('iters', '?')} iterations",
        best_checkpoint_iter=(
            None if model_info["checkpoint_iteration"] is None
            else model_info["checkpoint_iteration"] + 1
        ),
    )
    logger.info("Curves written to %s", curve["figure_path"])

    # -- the Day 1 drift check ----------------------------------------------
    stored_path = metric_dir / str(cfg.baseline.json_name).format(method="bicubic")
    day1_psnr: Any = "unavailable"
    day1_agrees = True
    fresh_psnr = float(results[bicubic_label].summary["psnr_mean"]["mean"])
    if stored_path.is_file():
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        stored_value = float(stored["summary"]["psnr_mean"]["mean"])
        day1_psnr = f"{stored_value:.4f}"
        # 0.01 dB: far below anything that changes a conclusion, far above
        # float noise. A larger gap means the split or a metric setting moved.
        day1_agrees = abs(stored_value - fresh_psnr) < 0.01
        (logger.info if day1_agrees else logger.error)(
            "Day 1 stored bicubic PSNR %.4f dB vs re-scored %.4f dB (%+0.4f). %s",
            stored_value,
            fresh_psnr,
            fresh_psnr - stored_value,
            "Consistent." if day1_agrees else
            "INCONSISTENT -- the split or the metric settings have moved.",
        )
    else:
        logger.warning(
            "No stored Day 1 baseline at %s, so the drift check was skipped. "
            "The comparison in this report is still internally consistent "
            "(both methods scored in this pass); what cannot be checked is "
            "whether it matches Day 1.", stored_path
        )

    # -- write the report ----------------------------------------------------
    written_utc = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    context = {
        "written_utc": written_utc,
        "dataset": str(cfg.dataset.name),
        "split": val_split,
        "num_patches": len(results[model_label].per_sample),
        "num_tiles": num_tiles,
        "scale": int(cfg.sr.scale),
        # The full value carries this machine's absolute path (e.g.
        # "file:D:\\SIH\\DrishtiSR\\outputs\\splits_sen2naipv2.csv"). The JSON
        # keeps it -- provenance is the point of that file -- but the report,
        # which is submission material, names the file rather than the drive.
        "split_source": loaders["split_source"],
        "split_source_name": Path(str(loaders["split_source"])).name,
        "device": str(evaluator.device),
        "torch_threads": torch.get_num_threads(),
        "bands": [str(b) for b in cfg.dataset.bands],
        "num_bands": len(cfg.dataset.bands),
        "lpips_rgb_bands": [str(b) for b in cfg.metrics.lpips.rgb_bands],
        "curves_relative": curves_figure.name,
        "bicubic_csv": csv_paths[bicubic_label].as_posix(),
        "model_csv": csv_paths[model_label].as_posix(),
        "day1_bicubic_psnr": day1_psnr,
        "day1_agrees": day1_agrees,
    }

    markdown = build_report(cfg, results, gate, model_info, curve, context)
    if args.smoke:
        markdown = (
            "> **THESE ARE SMOKE NUMBERS on the synthetic stub, produced by a "
            "RANDOMLY INITIALISED model. They are not results and the gate "
            "verdict below means nothing.** This run exists to prove the "
            "evaluation path executes end to end.\n\n" + markdown
        )
    if suffix:
        markdown += (
            f"\n> Truncated to --max-batches {args.max_batches}: everything "
            "above describes those batches, not the validation split.\n"
        )

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(markdown, encoding="utf-8")
    logger.info("Report written to %s", report_md)

    payload = {
        "written_utc": written_utc,
        "smoke": bool(args.smoke),
        "max_batches": args.max_batches,
        "gate": gate,
        "model": model_info,
        "curve": curve,
        "context": context,
        "results": {
            label: {
                "num_samples": int(len(result.per_sample)),
                "settings": result.settings,
                "summary": result.summary,
                "per_sample_csv": csv_paths[label].as_posix(),
            }
            for label, result in results.items()
        },
        "lpips_caveat": LPIPS_CAVEAT,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Raw numbers written to %s", report_json)

    # -- stdout --------------------------------------------------------------
    print()
    print(f"## Run A vs bicubic -- {cfg.dataset.name}, {val_split} split, "
          f"{context['num_patches']} patches")
    print()
    print(gate_table(gate, model_label))
    print()
    print(f"GATE: {'PASSED' if gate['passed'] else 'FAILED'}")
    print(f"Curve: {curve['verdict']}")
    print()
    print(f"  report:  {report_md}")
    print(f"  json:    {report_json}")
    print(f"  figure:  {curves_figure}")
    print()

    if args.smoke:
        print(
            "> **SMOKE RUN on the synthetic stub with a randomly initialised "
            "model.** The gate verdict above describes noise, not a model. "
            "Exit code is 0 regardless."
        )
        return 0

    if not gate["passed"]:
        print(
            "The gate FAILED, so the checkpoint must not be published. Read "
            f"{report_md} for the per-metric numbers."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
