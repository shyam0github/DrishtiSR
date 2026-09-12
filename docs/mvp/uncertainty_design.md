# P6 learned uncertainty — design (Day 4 "Run C")

**What.** A post-hoc heteroscedastic **Laplace** scale head trained on the **frozen** A2 backbone (`a2-last-dce224ec`, interim=false). The mean path is untouched: the backbone runs under `no_grad` with `requires_grad=False`, the NLL stop-grads `mu`, and a test checks that `mu` is bit-identical after training. So every A2 mean metric (`reports/day3_ab_metrics.csv`) still holds, and the ablation stays clean.

**Why Laplace.** The backbone was trained with L1, which is the Laplace MLE of the mean. A Laplace scale `b` is the matching likelihood, and `b` is in the loss's own units (reflectance). A Gaussian log-variance head would pair an L2-family likelihood with an L1-trained mean.

**Head.** Input: the LR-resolution body output, i.e. the tensor entering the upsampler `tail.0`, captured by a forward pre-hook (`models/edsr.py` is not edited). Layers: conv3×3 48→32, ReLU, conv3×3 32→64 (4 bands × 16), PixelShuffle(4). Output: `b = 1e-4 + softplus(raw)`. Running at LR resolution keeps it cheap: 32,352 params, for 888,004 total with the 855,652-param backbone (< 1 M). The last conv is zero-initialised with its bias set so that `b ≡ b0` at start, where `b0` = mean |SR − HR| over 128 TRAIN patches (the constant-scale Laplace MLE). Training therefore starts at a calibrated constant and only learns spatial structure.

**Loss.** `mean(w·(log 2b + |y − sg(mu)|/b))` with `w = sg(b)^β`, β = 0. β-NLL matters only when the mean is trained jointly; the option is kept.

**Tradeoff (accepted).** A frozen mean gets no loss-attenuation benefit, i.e. no down-weighting of hard pixels in the mean. We accept that for ablation cleanliness and for CPU cost: only the head trains, so ~2k iterations fit on CPU tonight.

**Serving.** `forward_packed` returns 8 channels: `[mu, b]`. `tiled.sr_array` sizes its buffer from the input band count, so `ScalePredictor` passes the two 4-channel halves through it (`ChannelSlice`). `b` is blended with the same linear Hann weights; tiled.py is unchanged. The cost is a second backbone pass per tile, and the API only sees the `info` contract (`has_scale_head: true`). A single-pass packed tiler would need a tiled.py patch.

**Collapse watch (pre-registered, not tuned).** Runs every 250 iterations up to 2,000 on a fixed 128-patch VAL subset (seed 26142, monitoring only). It tracks three statistics:
- Spearman ρ of band-mean b vs band-mean |err| (200k pixels)
- the share of per-band values with b < 2e-4
- the mean per-image spatial CV of b

Abort rule at iteration 2,000: ρ < 0.10, OR floor share > 0.20, OR CV < 0.05. If any holds, the verdict is "collapsed", training stops, and TTA is the fallback. The frozen backbone outputs on that subset are cached once, so each check re-runs only the head.

**Calibration (method-agnostic).** Every method is scored on the same metrics:
- AUSE on band-mean |err|: 20 removal fractions, against the oracle curve
- Spearman ρ
- Laplace coverage of |err| ≤ b·ln2 (nominal 50 %) and b·ln10 (nominal 90 %)

TTA std is not a Laplace scale, so its coverage is N/A. Eval patches are VAL, drawn with seed 26143 and disjoint from the watch subset.

**Compatibility.** The `src/uncertainty/` package shadows the older `src/uncertainty.py` (TTA helpers). `__init__` loads that file by path and re-exports `tta_predict`, `sr_output` and `TTAResult`, so the older callers still work.
