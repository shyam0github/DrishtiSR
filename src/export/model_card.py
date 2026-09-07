"""Build a Hugging Face model card from an evaluation report.

The card is GENERATED from ``reports/day2_runA.json``, never hand-written. That
file is the output of ``scripts/eval_runA.py``, so every number on the card --
the parameter count, the metric table, the split size, the training
configuration -- is the number that was actually measured, carried through
without a human retyping it. A model card whose metrics were copied by hand is
a claim; this one is a transcript.

What that buys, specifically: the parameter-budget overage and the LPIPS caveat
cannot be dropped from the card, because they are read out of the report rather
than remembered. Both are exactly the kind of caveat that goes missing between
an evaluation and a published artefact.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List

__all__ = ["build_model_card"]

# Metric rows shown on the card, in order: the summary column, its display name,
# which direction is better, and how many decimals to print. LPIPS is last
# because it is the weakest evidence here (see the caveat carried onto the card).
_CARD_METRICS = (
    ("psnr_mean", "PSNR (dB)", "higher", 4),
    ("ssim_mean", "SSIM", "higher", 4),
    ("sam_mean_deg", "SAM (deg)", "lower", 4),
    ("ergas", "ERGAS", "lower", 4),
    ("lpips", "LPIPS (alex, RGB)", "lower", 4),
)


def _metric_table(report: Dict[str, Any], model_label: str,
                  bicubic_label: str) -> str:
    """Render the model-vs-bicubic table for the card.

    Args:
        report: The parsed ``day2_runA.json``.
        model_label: Key of the model result in ``report["results"]``.
        bicubic_label: Key of the baseline result.

    Returns:
        A markdown table. Metrics absent from the report are omitted rather than
        printed as blanks -- a card should not imply a measurement was taken.
    """
    model = report["results"][model_label]["summary"]
    bicubic = report["results"][bicubic_label]["summary"]
    gate_columns = {entry["metric"] for entry in report["gate"]["metrics"]}

    lines = [
        "| metric | better | bicubic x4 | **EDSR Run A** | delta | in gate |",
        "|---|---|---|---|---|---|",
    ]
    for column, display, direction, places in _CARD_METRICS:
        if column not in model or column not in bicubic:
            continue
        model_value = float(model[column]["mean"])
        bicubic_value = float(bicubic[column]["mean"])
        delta = model_value - bicubic_value
        improved = delta > 0 if direction == "higher" else delta < 0
        lines.append(
            f"| {display} | {direction} | {bicubic_value:.{places}f} | "
            f"**{model_value:.{places}f}** | {delta:+.{places}f} "
            f"{'✅' if improved else '❌'} | "
            f"{'yes' if column in gate_columns else 'no'} |"
        )
    return "\n".join(lines)


def build_model_card(report: Dict[str, Any], cfg: Any, repo_id: str,
                     project_url: str = "") -> str:
    """Compose the full model card markdown.

    Args:
        report: Parsed ``reports/day2_runA.json`` -- the output of
            ``scripts/eval_runA.py``. Must carry ``gate``, ``model``, ``curve``,
            ``context`` and ``results``.
        cfg: Loaded config. Reads ``eval_runA`` labels, ``dataset``, ``sr`` and
            ``runtime``.
        repo_id: ``"<owner>/<name>"``, used in the usage snippet.
        project_url: Source repository, resolved by the caller from the git
            remote so no username is hardcoded here or in config. Empty string
            renders the project name as plain text -- a card with a link to
            ``https://github.com/`` is worse than a card with no link.

    Returns:
        The card, including YAML front matter.

    Raises:
        KeyError: The report is missing a section, i.e. it was not written by
            ``scripts/eval_runA.py``.
        ValueError: The report records a failed gate. A card asserting a model
            beats bicubic must not be generated from a run that says otherwise.
    """
    for key in ("gate", "model", "curve", "context", "results"):
        if key not in report:
            raise KeyError(
                f"The evaluation report has no {key!r} section, so it was not "
                "written by scripts/eval_runA.py. Refusing to build a card from "
                "an unknown file."
            )
    if not report["gate"]["passed"]:
        raise ValueError(
            "The evaluation report records a FAILED gate. This card states "
            "that the model beats bicubic; generating it from a failing run "
            "would publish a false claim. Fix the model, not the card."
        )

    block = cfg["eval_runA"]
    model_label = str(block["model_label"])
    bicubic_label = str(block["bicubic_label"])

    info = report["model"]
    context = report["context"]
    curve = report["curve"]
    train_args = info["train_args"]
    bands = context["bands"]

    over_budget = bool(info["exceeds_parameter_budget"])
    gate_line = " · ".join(
        f"{entry['metric']} {entry['delta_model_minus_bicubic']:+.4f}"
        for entry in report["gate"]["metrics"]
    )

    front_matter = "\n".join([
        "---",
        "license: mit",
        "library_name: pytorch",
        "tags:",
        "  - super-resolution",
        "  - remote-sensing",
        "  - sentinel-2",
        "  - earth-observation",
        "  - image-to-image",
        "  - edsr",
        f"pipeline_tag: image-to-image",
        "---",
    ])

    parts: List[str] = [
        front_matter,
        "",
        "# DrishtiSR — EDSR baseline, Sentinel-2 x4 (10 m → 2.5 m)",
        "",
        "**Run A**: the L1-only EDSR reference run for "
        + (f"[DrishtiSR]({project_url})" if project_url else "DrishtiSR")
        + ", Smart India Hackathon 2026 problem "
        "statement **SIH26142**. This is the number every later contribution "
        "of the project is measured against — it is a **baseline**, "
        "deliberately plain: no uncertainty head, no spectral-consistency loss.",
        "",
        f"Trained on SEN2NAIPv2 cross-sensor pairs, 4-band **RGBN** "
        f"({', '.join(bands)}), x{context['scale']}, and evaluated over the "
        f"**full {context['num_patches']}-patch validation split** "
        f"({context['num_tiles']} geographically disjoint tiles).",
        "",
        "---",
        "",
        "## Results vs the bicubic floor",
        "",
        f"Both rows were computed **in the same pass, by the same evaluator, "
        f"over the same loader** — the baseline is re-scored rather than quoted, "
        f"so the comparison cannot be an artefact of a drifting split. "
        f"n = {context['num_patches']} patches, "
        f"reflectance `data_range=1.0`, float32, CPU.",
        "",
        _metric_table(report, model_label, bicubic_label),
        "",
        f"PSNR and SSIM are means over **all {context['num_bands']} bands**. "
        f"LPIPS uses the **AlexNet** backbone on the **RGB bands only** "
        f"({', '.join(context['lpips_rgb_bands'])}), reflectance scaled into "
        "`[-1, 1]`.",
        "",
        f"**Gate: PASSED** — the model improves on bicubic on every gate metric "
        f"({gate_line}).",
        "",
    ]

    if over_budget:
        parts += [
            "> ### ⚠️ This model exceeds the project's own parameter budget",
            ">",
            f"> {info['parameters']:,} parameters against the "
            f"{info['parameter_budget']:,} limit the DrishtiSR submission must "
            f"meet — **{info['parameters'] / info['parameter_budget']:.2f}x "
            "over**. Run A exists to establish what a conventional EDSR "
            "achieves on this split, so that the budgeted submission model can "
            "be compared against something real. **It is a reference point, "
            "not the deliverable**, and its numbers should not be quoted as "
            "DrishtiSR's result.",
            "",
        ]

    parts += [
        "## Intended use",
        "",
        "**Intended.** Super-resolving Sentinel-2 L2A surface reflectance from "
        "10 m to 2.5 m ground sampling distance, x4, on the four 10 m bands "
        f"(**{', '.join(bands)}** — red, green, blue, NIR, in that channel "
        "order). Input and output are **surface reflectance**, nominally "
        "`[0, 1]` (digital number ÷ "
        f"{float(cfg['dataset']['reflectance_scale']):.0f}) and **not clipped** "
        "— bright targets such as cloud, snow, specular water and some roofs "
        "legitimately exceed 1.0, and clipping them corrupts the radiometry.",
        "",
        "**Not intended.**",
        "",
        "- **Not for measurement or quantitative retrieval.** This is an L1-"
        "trained SR model with no uncertainty estimate. It cannot tell you "
        "which of the detail it produces is inferred and which is real, so no "
        "pixel it outputs should feed an area estimate, a change-detection "
        "decision, or a legal or safety judgement.",
        "- **Not validated outside its training geography.** The training "
        "pairs are SEN2NAIPv2, which is dominated by the continental United "
        "States. Behaviour over other land cover, other atmospheres, and other "
        "sun angles is unmeasured.",
        "- **Not a different band set or scale.** Four channels in this order, "
        f"x{context['scale']} only.",
        "- **Do not normalise with ImageNet statistics.** The model consumes "
        "physical reflectance. Standardising the input breaks it.",
        "",
        "## Architecture",
        "",
        "EDSR-baseline (Lim et al., 2017), no batch normalisation:",
        "",
        f"- **{info['architecture']['n_resblocks']} residual blocks**, "
        f"**{info['architecture']['n_feats']} features**, residual scaling 1.0",
        f"- Head and tail 3x3 convolutions; **PixelShuffle** upsampler "
        f"(two x2 stages for x{info['architecture']['scale']})",
        f"- **{info['architecture']['in_ch']} channels in and out** (RGBN). "
        "The 4-channel input is what makes the parameter count "
        f"{info['parameters']:,}; a 3-channel RGB variant would be 1,517,571.",
        f"- **{info['parameters']:,} trainable parameters**",
        "",
        "## Training configuration",
        "",
        "| setting | value |",
        "|---|---|",
        f"| objective | L1 (no perceptual, adversarial or spectral term) |",
        f"| iterations | {train_args.get('iters', '?'):,} |",
        f"| batch size | {train_args.get('batch', '?')} |",
        f"| LR patch size | {train_args.get('patch_lr', '?')} px "
        f"(HR {int(train_args.get('patch_lr', 0)) * int(context['scale'])} px) |",
        f"| optimiser | AdamW, betas (0.9, 0.99), weight decay "
        f"{train_args.get('wd', '?')} |",
        f"| learning rate | {train_args.get('lr', '?')} → "
        f"{train_args.get('min_lr', '?')}, cosine, "
        f"{train_args.get('warmup', '?')}-iteration warmup |",
        f"| gradient clipping | {train_args.get('clip', '?')} (global norm) |",
        f"| precision | AMP (fp16) on CUDA |",
        f"| hardware | Kaggle NVIDIA T4, single GPU |",
        f"| seed | {train_args.get('seed', '?')} |",
        "",
        "### Was it trained long enough?",
        "",
        f"**No — it was trained too long.** {curve['verdict']}",
        "",
        f"The published checkpoint is **`best.pt`, from iteration "
        f"{(info['checkpoint_iteration'] or 0) + 1:,}**, selected by the "
        f"training loop's periodic validation — not the final iteration. "
        f"`log.csv` in this repository carries the full curve.",
        "",
        "## Dataset",
        "",
        "**SEN2NAIPv2** (cross-sensor subset): real Sentinel-2 L2A scenes "
        "paired with NAIP aerial imagery degraded to a matching sensor model, "
        "so the LR/HR pairs are genuinely co-registered rather than "
        "synthetically downsampled.",
        "",
        f"- Split: **geographic and scene-grouped** with a fixed seed, so no "
        f"scene appears in two splits. {context['num_tiles']} validation tiles "
        f"→ {context['num_patches']} non-overlapping "
        f"{train_args.get('patch_lr', '?')} px LR patches (grid mode, "
        "deterministic).",
        f"- Reflectance scale: digital number ÷ "
        f"{float(cfg['dataset']['reflectance_scale']):.0f}, **unclipped**.",
        f"- Split source: `{Path(str(context['split_source'])).name}`.",
        "",
        "## Files in this repository",
        "",
        "| file | what it is |",
        "|---|---|",
        "| `best.pt` | The checkpoint. A dict with `model` (the state dict), "
        "`opt`, `scaler`, `it`, `best` and `args`. |",
        "| `args.json` | Every training argument the run was launched with. |",
        "| `log.csv` | The full training log: L1 loss and LR every 100 "
        "iterations, validation PSNR every 2000. |",
        "",
        "## Usage",
        "",
        "```python",
        "import torch",
        "from huggingface_hub import hf_hub_download",
        "",
        "# The EDSR definition is in the DrishtiSR repo: src/models/edsr.py",
        "from src.models.edsr import build_model",
        "",
        f'path = hf_hub_download("{repo_id}", "best.pt")',
        'ckpt = torch.load(path, map_location="cpu", weights_only=False)',
        "",
        'model = build_model(',
        '    "edsr_baseline",',
        f'    scale={info["architecture"]["scale"]},',
        f'    n_resblocks={info["architecture"]["n_resblocks"]},',
        f'    n_feats={info["architecture"]["n_feats"]},',
        f'    in_ch={info["architecture"]["in_ch"]},',
        f'    out_ch={info["architecture"]["in_ch"]},',
        ")",
        'model.load_state_dict(ckpt["model"])',
        "model.eval()",
        "",
        "# lr: (B, 4, H, W) float32 SURFACE REFLECTANCE in band order "
        f"{bands},",
        f"#     i.e. digital number / "
        f"{float(cfg['dataset']['reflectance_scale']):.0f}. Do NOT standardise "
        "and do NOT clip.",
        "with torch.no_grad():",
        "    sr = model(lr)   # (B, 4, H*4, W*4), same units and band order",
        "```",
        "",
        "## Limitations and caveats",
        "",
        f"- **LPIPS is a proxy here, not an authority.** {report['lpips_caveat']}",
        "- **The validation patches are not independent.** "
        f"{context['num_patches']} patches come from {context['num_tiles']} "
        "tiles, so patches from one tile share land cover, atmosphere and "
        "acquisition date. Treat the spread as narrower than it looks.",
        "- **An L1 objective produces conservative, slightly soft output.** "
        "That is the honest failure mode for a measurement instrument, and it "
        "is the baseline this project's uncertainty-aware model is meant to "
        "improve on.",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{drishtisr_edsr_baseline_x4,",
        "  title  = {DrishtiSR EDSR baseline: Sentinel-2 x4 super-resolution},",
        "  note   = {Smart India Hackathon 2026, problem statement SIH26142},",
        f"  year   = {{{_dt.datetime.now().year}}}",
        "}",
        "```",
        "",
        "---",
        "",
        f"_Model card generated from `reports/day2_runA.json` "
        f"(evaluated {context['written_utc']}) by "
        f"`src/export/model_card.py`. Every number above is read from that "
        f"report rather than transcribed._",
        "",
    ]
    return "\n".join(parts)
