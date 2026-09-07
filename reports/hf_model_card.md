---
license: mit
library_name: pytorch
tags:
  - super-resolution
  - remote-sensing
  - sentinel-2
  - earth-observation
  - image-to-image
  - edsr
pipeline_tag: image-to-image
---

# DrishtiSR — EDSR baseline, Sentinel-2 x4 (10 m → 2.5 m)

**Run A**: the L1-only EDSR reference run for [DrishtiSR](https://github.com/shyam0github/DrishtiSR), Smart India Hackathon 2026 problem statement **SIH26142**. This is the number every later contribution of the project is measured against — it is a **baseline**, deliberately plain: no uncertainty head, no spectral-consistency loss.

Trained on SEN2NAIPv2 cross-sensor pairs, 4-band **RGBN** (B04, B03, B02, B08), x4, and evaluated over the **full 1199-patch validation split** (300 geographically disjoint tiles).

---

## Results vs the bicubic floor

Both rows were computed **in the same pass, by the same evaluator, over the same loader** — the baseline is re-scored rather than quoted, so the comparison cannot be an artefact of a drifting split. n = 1199 patches, reflectance `data_range=1.0`, float32, CPU.

| metric | better | bicubic x4 | **EDSR Run A** | delta | in gate |
|---|---|---|---|---|---|
| PSNR (dB) | higher | 38.4717 | **38.8260** | +0.3543 ✅ | yes |
| SSIM | higher | 0.8825 | **0.8896** | +0.0072 ✅ | yes |
| SAM (deg) | lower | 2.0925 | **2.0219** | -0.0705 ✅ | no |
| ERGAS | lower | 3.0503 | **2.9036** | -0.1466 ✅ | no |
| LPIPS (alex, RGB) | lower | 0.4033 | **0.3219** | -0.0814 ✅ | yes |

PSNR and SSIM are means over **all 4 bands**. LPIPS uses the **AlexNet** backbone on the **RGB bands only** (B04, B03, B02), reflectance scaled into `[-1, 1]`.

**Gate: PASSED** — the model improves on bicubic on every gate metric (psnr_mean +0.3543 · ssim_mean +0.0072 · lpips -0.0814).

> ### ⚠️ This model exceeds the project's own parameter budget
>
> 1,518,724 parameters against the 1,000,000 limit the DrishtiSR submission must meet — **1.52x over**. Run A exists to establish what a conventional EDSR achieves on this split, so that the budgeted submission model can be compared against something real. **It is a reference point, not the deliverable**, and its numbers should not be quoted as DrishtiSR's result.

## Intended use

**Intended.** Super-resolving Sentinel-2 L2A surface reflectance from 10 m to 2.5 m ground sampling distance, x4, on the four 10 m bands (**B04, B03, B02, B08** — red, green, blue, NIR, in that channel order). Input and output are **surface reflectance**, nominally `[0, 1]` (digital number ÷ 10000) and **not clipped** — bright targets such as cloud, snow, specular water and some roofs legitimately exceed 1.0, and clipping them corrupts the radiometry.

**Not intended.**

- **Not for measurement or quantitative retrieval.** This is an L1-trained SR model with no uncertainty estimate. It cannot tell you which of the detail it produces is inferred and which is real, so no pixel it outputs should feed an area estimate, a change-detection decision, or a legal or safety judgement.
- **Not validated outside its training geography.** The training pairs are SEN2NAIPv2, which is dominated by the continental United States. Behaviour over other land cover, other atmospheres, and other sun angles is unmeasured.
- **Not a different band set or scale.** Four channels in this order, x4 only.
- **Do not normalise with ImageNet statistics.** The model consumes physical reflectance. Standardising the input breaks it.

## Architecture

EDSR-baseline (Lim et al., 2017), no batch normalisation:

- **16 residual blocks**, **64 features**, residual scaling 1.0
- Head and tail 3x3 convolutions; **PixelShuffle** upsampler (two x2 stages for x4)
- **4 channels in and out** (RGBN). The 4-channel input is what makes the parameter count 1,518,724; a 3-channel RGB variant would be 1,517,571.
- **1,518,724 trainable parameters**

## Training configuration

| setting | value |
|---|---|
| objective | L1 (no perceptual, adversarial or spectral term) |
| iterations | 40,000 |
| batch size | 16 |
| LR patch size | 64 px (HR 256 px) |
| optimiser | AdamW, betas (0.9, 0.99), weight decay 0.0001 |
| learning rate | 0.0002 → 1e-06, cosine, 500-iteration warmup |
| gradient clipping | 1.0 (global norm) |
| precision | AMP (fp16) on CUDA |
| hardware | Kaggle NVIDIA T4, single GPU |
| seed | 1337 |

### Was it trained long enough?

**No — it was trained too long.** Validation was NOT still improving at 40,000 iterations: it peaked at 35.284 dB at iteration 8,000 (16% of the way through) and ended at 35.033 dB, -0.251 dB from the peak, with the final 25% sloping -0.0033 dB per 1k iterations. The run is NOT undertrained; the extra iterations after the peak bought nothing.

The published checkpoint is **`best.pt`, from iteration 8,000**, selected by the training loop's periodic validation — not the final iteration. `log.csv` in this repository carries the full curve.

## Dataset

**SEN2NAIPv2** (cross-sensor subset): real Sentinel-2 L2A scenes paired with NAIP aerial imagery degraded to a matching sensor model, so the LR/HR pairs are genuinely co-registered rather than synthetically downsampled.

- Split: **geographic and scene-grouped** with a fixed seed, so no scene appears in two splits. 300 validation tiles → 1199 non-overlapping 64 px LR patches (grid mode, deterministic).
- Reflectance scale: digital number ÷ 10000, **unclipped**.
- Split source: `splits_sen2naipv2.csv`.

## Files in this repository

| file | what it is |
|---|---|
| `best.pt` | The checkpoint. A dict with `model` (the state dict), `opt`, `scaler`, `it`, `best` and `args`. |
| `args.json` | Every training argument the run was launched with. |
| `log.csv` | The full training log: L1 loss and LR every 100 iterations, validation PSNR every 2000. |

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# The EDSR definition is in the DrishtiSR repo: src/models/edsr.py
from src.models.edsr import build_model

path = hf_hub_download("<your-hf-username>/drishtisr-edsr-baseline-x4", "best.pt")
ckpt = torch.load(path, map_location="cpu", weights_only=False)

model = build_model(
    "edsr_baseline",
    scale=4,
    n_resblocks=16,
    n_feats=64,
    in_ch=4,
    out_ch=4,
)
model.load_state_dict(ckpt["model"])
model.eval()

# lr: (B, 4, H, W) float32 SURFACE REFLECTANCE in band order ['B04', 'B03', 'B02', 'B08'],
#     i.e. digital number / 10000. Do NOT standardise and do NOT clip.
with torch.no_grad():
    sr = model(lr)   # (B, 4, H*4, W*4), same units and band order
```

## Limitations and caveats

- **LPIPS is a proxy here, not an authority.** LPIPS is a perceptual proxy: its backbone was trained on natural RGB photographs, not on multispectral surface reflectance. On satellite data it is indicative of perceptual sharpness, not authoritative. It sees only the RGB bands, ignores NIR entirely, and requires reflectance to be clipped into [0, 1] before use. Rank models by PSNR/SSIM/SAM/ERGAS; read LPIPS as a tie-breaker on apparent detail.
- **The validation patches are not independent.** 1199 patches come from 300 tiles, so patches from one tile share land cover, atmosphere and acquisition date. Treat the spread as narrower than it looks.
- **An L1 objective produces conservative, slightly soft output.** That is the honest failure mode for a measurement instrument, and it is the baseline this project's uncertainty-aware model is meant to improve on.

## Citation

```bibtex
@misc{drishtisr_edsr_baseline_x4,
  title  = {DrishtiSR EDSR baseline: Sentinel-2 x4 super-resolution},
  note   = {Smart India Hackathon 2026, problem statement SIH26142},
  year   = {2026}
}
```

---

_Model card generated from `reports/day2_runA.json` (evaluated 2026-09-07T16:18:43+00:00) by `src/export/model_card.py`. Every number above is read from that report rather than transcribed._
