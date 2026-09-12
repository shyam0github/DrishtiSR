# opensr-test README benchmark, ND table (literature source)

Transcribed verbatim on 2026-09-12 from
https://raw.githubusercontent.com/ESAOpenSR/opensr-test/main/README.md,
section "Benchmark" -> "ND". The /compare page quotes the rows below and
`frontend/tests/compare.test.ts` checks each quoted number against this file.

These are **reported, not measured by us**. They are not comparable with
DrishtiSR's rows:

- **Different test data.** The README table covers the opensr-test benchmark
  datasets (NAIP / SPOT / Venus / SPAIN, 9-62 images each) and does not say
  which subset or pooling produced it. DrishtiSR rows are SEN2NAIPv2 val,
  1,199 pairs.
- **Different aggregation.** README: `agg_method: "patch", patch_size: 1`,
  `device: cuda`, library version unstated. Ours: `agg_method: "pixel"`,
  opensr-test 1.3.3, CPU (`outputs/metrics/eval_all_ckpts/*/meta.json`).
- Same distance family (`correctness_distance: "nd"`, reflectance L1,
  spectral SAD), which is why these rows are quoted for rough positioning at all.

README text above the table: "We use the normalized difference (ND) distance to
measure the distance between the LR, SR, and HR images. [...] The parameters of
the experiments are: `{"device": "cuda", "agg_method": "patch", "patch_size": 1,
"correctness_distance": "nd"}`."

|    model     | reflectance ↓   | spectral ↓       | spatial ↓       | synthesis ↑     | ha_metric ↓     | om_metric ↓     | im_metric ↑     |
|:-------------|:----------------|:-----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| ldm_baseline | 0.0505 ± 0.0161 | 9.6923 ± 2.1742  | 0.0715 ± 0.0679 | 0.0285 ± 0.0307 | 0.6067 ± 0.2172 | 0.3088 ± 0.1786 | 0.0845 ± 0.0428 |
| opensrmodel  | 0.0031 ± 0.0018 | 1.2632 ± 0.5878  | 0.0114 ± 0.0111 | 0.0068 ± 0.0044 | 0.3431 ± 0.0738 | 0.4593 ± 0.0781 | 0.1976 ± 0.0328 |
| satlas       | 0.0489 ± 0.0086 | 12.1231 ± 3.1529 | 0.2742 ± 0.0748 | 0.0227 ± 0.0107 | 0.8004 ± 0.0641 | 0.1073 ± 0.0393 | 0.0923 ± 0.0266 |
| sr4rs        | 0.0396 ± 0.0198 | 3.4044 ± 1.6882  | 1.0037 ± 0.1520 | 0.0177 ± 0.0083 | 0.7274 ± 0.0840 | 0.1637 ± 0.0572 | 0.1089 ± 0.0292 |
| superimage   | 0.0029 ± 0.0009 | 1.5672 ± 1.0692  | 0.0132 ± 0.1131 | 0.0046 ± 0.0027 | 0.2026 ± 0.0692 | 0.6288 ± 0.0754 | 0.1686 ± 0.0302 |

Paper: Aybar et al., "A Comprehensive Benchmark for Optical Remote Sensing
Image Super-Resolution", IEEE GRSL 2024, https://ieeexplore.ieee.org/document/10530998.
The README notes the table is "an improved version of the one presented in the
opensr-test paper".

Not in this table, so every opensr-test cell for them is N/A on /compare:
SwinIR / HAT (transformer) and Real-ESRGAN (GAN). Their papers report PSNR/SSIM
on synthetic-bicubic photo benchmarks, not these seven metrics.
