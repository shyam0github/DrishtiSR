# DrishtiSR demo: 90-second click path

Launch beforehand (the model loads in about 10 s):
`D:\SIH\DrishtiSR\.venv\Scripts\python.exe scripts/serve.py` (cwd and PYTHONPATH =
`D:\SIH\DrishtiSR-mvp`). Keep `app/samples/test-mid-1/lr.tif` on the desktop for the
upload step. Leave TTA at **4 passes**.

| t (s) | Click | Say |
|---|---|---|
| 0–10 | Header: point at the model badge (ONNX FP32 · 855,652 params · ~3.4 MB · 6 threads) | "Under a million parameters, running on this laptop's CPU. No GPU." |
| 10–20 | Sample list → the **high-texture** TEST sample (`test-high-1`) → **Run super-resolution** | "Real Sentinel-2 at 10 m in, 2.5 m out, in about a second. TEST tile, shown for display only." |
| 20–30 | Drag the **slider**. Left layer: LR, then Bicubic, then HR 2.5 m reference. | "The right side is always the AI output, and it's labelled as a reconstruction, not a measurement." |
| 30–42 | Overlay → **Uncertainty**, opacity ~65 % | "Where the four test-time passes disagree. Fixed scale, never stretched per image, so bright means unsure on every image." |
| 42–55 | Overlay → **Spectral consistency**. Metrics panel → consistency card: SR value, the **bicubic** marker, the **HR floor** marker (0.0057). | "We re-degrade our output with the training operator and compare it with the 10 m input. Real HR sits at this floor, and bicubic sits near zero because it is blurry." |
| 55–62 | Point at the **HF ratio vs bicubic** row. Show the blur-guard banner if it is visible. | "A low spectral error can also mean blur. If our output isn't at least 5 % sharper than bicubic, the UI warns. That's why we rejected the training-time spectral loss: it bought consistency with blur." |
| 62–72 | Sample list → **Delhi · dense urban** → Run | "Our own Sentinel-2 scene over Delhi. There's no ground truth, so the panel shows trust maps only: consistency, sharpness, uncertainty." |
| 72–84 | Drop `test-mid-1/lr.tif` on the upload zone → Run | "Any 4-band GeoTIFF up to 512 px. The DN scaling is detected automatically." |
| 84–90 | Downloads → **SR GeoTIFF (2.5 m)** | "The georeferenced 2.5 m GeoTIFF opens directly in QGIS." |

Fallbacks: if the API badge is not green, open `/?mock=1` (the fixture UI; the
"MOCK DATA" chip is shown). If the Delhi samples are missing, skip that row; they
are built by `scripts/mvp/build_samples.py`.
