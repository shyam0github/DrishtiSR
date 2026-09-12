# Deployment (a2-last-dce224ec, 256² LR, 6 threads)

| Backend | Params | Bytes | Median ms | p90 ms | ΔPSNR dB | ΔSAM° | ΔLPIPS | Provisional | Notes |
|---|---|---|---|---|---|---|---|---|---|
| torch-fp32 | 855,652 | 3,422,608 | 1217.1 | 1262.2 | +0.000 | +0.000 | +0.0000 | no | reference |
| onnx-fp32 | 855,652 | 3,441,761 | 1220.1 | 1249.1 | +0.000 | +0.000 | +0.0000 | no | parity max abs diff 4.5e-07 reflectance vs torch (metric Δ ≈ 0) |
| onnx-int8 | 855,652 | 981,131 | 781.9 | 815.2 | -0.329 | +0.066 | -0.0054 | no | attempt C FAILED the quality gate; timed for information, not served; quant attempt C, n=64 VAL; INT8 gate FAILED; served backend = onnx-fp32 |

- Input: data/delhi/delhi_A_20241030.tif window rows/cols (400, 400) 256x256, (dn-1000)/10000; CPU load before benchmark 0.0%.
- Δ = backend − FP32 on the quant-gate VAL set; gate {'d_psnr_db': 0.1, 'd_sam_deg': 0.05, 'd_lpips': 0.005}, passed=False, served backend onnx-fp32.
