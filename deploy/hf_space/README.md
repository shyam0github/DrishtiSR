---
title: DrishtiSR
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# DrishtiSR: Sentinel-2 10 m → 2.5 m on CPU

This is a CPU demo of a compact EDSR (855,652 parameters, ONNX FP32). It shows a
per-pixel uncertainty map, a reference-free spectral-consistency map and a blur
guard. It runs on the free 2-vCPU Space hardware (`DRISHTI_THREADS=2`).

Real Sentinel-2 L2A 10 m inputs (B04, B03, B02, B08) paired with 2.5 m NAIP targets that the SEN2NAIPv2 authors co-registered and radiometrically harmonised to Sentinel-2 (crosssensor subset). Inputs are not synthetically degraded; matching per-band statistics between inputs and targets are expected from this harmonisation.

---

## How to publish this Space by hand

Nothing here was pushed automatically. Publishing needs a Hugging Face account
and a **write** token.

1. **Get a token.** On huggingface.co go to Settings → Access Tokens and create
   a token with the "Write" role. In PowerShell, keep it in an environment
   variable only; never paste it into a file:
   `$env:HF_TOKEN = "hf_..."`
2. **Create an empty Space** on the website: New → Space, pick a name (for
   example `drishtisr`), choose **Docker** as the SDK and **CPU basic** as the
   hardware.
3. **Build the upload folder.** A Space holds only the files it needs, so copy
   them into a fresh staging folder that mirrors the repository layout. Run
   this from `D:\SIH\DrishtiSR-mvp` (the samples must be built first with
   `scripts/mvp/build_samples.py`):

   ```powershell
   $S = "D:\SIH\hf_space_stage"; $W = "D:\SIH\DrishtiSR-mvp"; $O = "a2-last-dce224ec"
   robocopy "$W\deploy\hf_space" $S Dockerfile README.md
   robocopy "$W\app" "$S\app" /E /XD _jobs __pycache__
   foreach ($d in "src", "drishtisr", "configs") { robocopy "$W\$d" "$S\$d" /E /XD __pycache__ }
   robocopy "$W\artifacts\onnx\$O" "$S\artifacts\onnx\$O" model_fp32.onnx export_meta.json
   robocopy "$W\reports\mvp" "$S\reports\mvp" trust_scales.json quant_gate.json projection_gate.json
   robocopy "D:\SIH\DrishtiSR\runs\day3\a2" "$S\ckpt\a2" last.pt
   ```

   Use a new, empty folder for `$S`. robocopy only copies, so leftovers from
   an earlier attempt would stay in the folder and be uploaded too.

   The checkpoint must keep the folder name `a2` and the file name `last.pt`.
   The server identifies the model as `a2-last-dce224ec` from that path plus
   the file's hash, and finds the ONNX graph under that id.
4. **(Optional) Test locally** if Docker Desktop is installed:
   `docker build -t drishtisr $S` then `docker run -p 7860:7860 drishtisr`, and
   open http://localhost:7860. The first build takes several minutes because it
   downloads CPU PyTorch and the LPIPS AlexNet weights.
5. **Upload.** Install the Hub client once (`python -m pip install -U huggingface_hub`)
   and upload the staging folder. Replace `<user>` with your account name:

   ```powershell
   python -m huggingface_hub.commands.huggingface_cli upload <user>/drishtisr $S . --repo-type space --token $env:HF_TOKEN
   ```

   On recent huggingface_hub versions, `hf upload <user>/drishtisr $S . --repo-type space`
   does the same. The client stores large files (the `.pt`, `.onnx` and `.npy`
   files) through the Hub's large-file storage automatically.
6. **Wait for the build.** The Space page shows "Building" and then "Running".
   Build logs are under the "Logs" button. The app answers on the Space URL
   with the same UI as the local demo.

Notes:
- The TEST sample tiles are for display only. They are never used for training
  or model selection.
- Free CPU Spaces sleep when idle. The first request after waking reloads the
  model (about 10–20 s).
- To change the thread count, set `DRISHTI_THREADS` under the Space's
  Settings → Variables.
