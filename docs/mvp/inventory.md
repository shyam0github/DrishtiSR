# MVP inventory (P0b, 2026-09-11)

Read-only survey of main tree `D:\SIH\DrishtiSR` at `1e7986e`. Line numbers are
from the worktree copy at the same commit. **Package root note:** the code lives
in `src/` (import `src.X`); `drishtisr.X` is an alias (`drishtisr/__init__.py:29`
sets `__path__` to `src/`). There is no `src/drishtisr/`. Wherever a prompt says
`src/drishtisr/<pkg>`, read `src/<pkg>`.

Abbreviations: `M` = `D:\SIH\DrishtiSR`, `W` = `D:\SIH\DrishtiSR-mvp`.

## 1. Model builder
- `src/models/edsr.py:88` `EDSR(scale=4, n_resblocks=16, n_feats=48, in_ch=4, out_ch=4, res_scale=1.0, uncertainty=False, var_feats=16, logvar_min=-10, logvar_max=10, logvar_init=-9)`. Head conv → 16 ResBlocks + trailing conv (`body.0..16`) → `tail.0` PixelShuffle upsampler + `tail.1` output conv; optional `var_head` (log-variance, clamped).
- `src/models/edsr.py:193` `build_model(name="edsr_baseline", **kw)` is the single entry point; `:200` `count_params`.
- `src/models/edsr.py:250` `model_from_checkpoint(path, device)`: rebuilds from `payload["args"]` (keys `ARCH_KEYS` at `:247` = scale, n_resblocks, n_feats, in_ch; plus uncertainty/var_feats/logvar_* when `args.uncertainty`), strict load, eval mode. `:209` `load_backbone_state` loads a no-head checkpoint into a head model.
- Duplicates of the same logic: `src/infer/tiled.py:239` `load_ckpt`, `scripts/eval_runA.py:255` `load_checkpoint_model` (used by eval and day3_ab_report).
- Architecture storage: checkpoints carry `args` = `vars(argparse)` of `src/train.py` (includes n_resblocks, n_feats, in_ch, scale, hf_cutoff, lambdas). No separate arch block.
- Checkpoint dict (`src/train.py:732` `save()`, `:759` torch.save): Run A keys `model, opt, scaler, it, best, args`; Day 3 keys additionally `config_hash, git_sha, git_dirty, lambdas, spectral_downsample, sharpness_reference, val_metrics, val_metrics_iter`. `it` is 0-based (`it+1` = iteration label used by eval_all_ckpts).
- MVP wrapper: `src/infer/predictor.py` `build_model_from_checkpoint` (args first, else `infer_architecture` from state-dict shapes).

## 2. Checkpoints (absolute paths, all present)
| run | file | it (1-based) | sha256[:8] | arch | note |
|---|---|---|---|---|---|
| Run A | `M\runs\runA\best.pt` | 8000 | 9267674f | 16/64, 1,518,724 params | over budget; † row + loader test only |
| Run A | `M\runs\runA\last.pt` | 40000 | be69c1e3 | 16/64 | |
| A2 | `M\runs\day3\a2\best.pt` | 10500 | 44d9295b | 16/48 | best.pt = training-loop PSNR pick |
| **A2** | **`M\runs\day3\a2\last.pt`** | **12000** | **dce224ec** | 16/48, 855,652 | **SELECTED it12000** (`reports/day3_ab_summary.json` `selection.a2.selected`); checkpoint_id `a2-last-dce224ec` |
| B1 | `M\runs\day3\b1\best.pt` / `last.pt` | 10500 / 12000 | 7d715c32 / fff6d2b1 | 16/48 | selected it12000 = last.pt |
| B2 | `M\runs\day3\b2\best.pt` / `last.pt` | 5500 / 12000 | 7e827d58 / d6d5cae7 | 16/48 | selected it12000 = last.pt |
- `M\runs\kaggle\day3\runs\{a2,b1,b2}\` are byte-identical raw downloads (a2/last.pt sha dce224ec confirmed); use `M\runs\day3\` paths.
- No intermediate `it*.pt` files exist locally; only best/last per run.

## 3. Normalisation
- None. Model consumes surface reflectance directly (`configs/frozen_day3.yaml:147-154` `normalization.kind: none`, `clip: false`).
- `reflectance = DN / 10000`: `configs/base.yaml:105` `dataset.reflectance_scale`; nominal range `:111-112` (assertion only); validation guard `reflectance_valid_max: 2.0` `:134`; nodata DN 65535 masked to 0.0 before division `:140-142`; conversion `src/data/base.py:361` `SRPairDataset.to_reflectance`.
- Channel order: `configs/base.yaml:96` `dataset.bands: [B04, B03, B02, B08]` = R, G, B, NIR (`src/data/sen2naip.py:69` `NATIVE_BANDS` same order). RGB = indices 0,1,2; FCC (NIR,R,G) = 3,0,1.
- On-disk dtype: cache npz members `lr` `(4,130,130)` / `hr` `(4,520,520)` **uint16 DN** (`configs/base.yaml:243`); SEN2NAIP DN are already BOA-offset-corrected.

## 4. Data API
- Frozen split IDs: `M\outputs\splits_sen2naipv2.csv` (columns `sample_id,split,group_id,scene_key,lon,lat`; 3809 train / 300 val / 300 test tiles). Located by `src/data/loader.py:106` `resolve_split_assignments` (`paths.manifest_dir` / `splits.output_name`, `configs/base.yaml:306,370`). **Not present in W** (`outputs/` is per tree): pass absolute path.
- Loader: `src/data/loader.py:648` `build_dataloaders(cfg, dataset=None, logger=None)` → `{"train","val","datasets","indices","split_source"}`; val = `PatchDataset` grid mode (`:277`), 64-px LR patches (`configs/base.yaml:313-316`), 1199 val patches. Requires the dataset catalog.
- Dataset: `src/data/sen2naip.py:82` `SEN2NAIPv2Dataset`; `:1007` `load_sample(idx)` → `{"lr","hr","meta"}` float32 reflectance full tile. **Catalog load `:176` hits the network (tacoreader)**. Offline load by ID: `_cache_paths(sample_id)` `:273` → `<cache>/sen2naipv2-crosssensor/<id>.npz` + `.json`, then `to_reflectance`; crops via `src/data/patches.py:303` `centre_crop_pair(lr, hr, lr_size, scale)`. `tests/mvp/test_predictor.py` does exactly this.
- Cache root: `M\outputs\cache\sen2naipv2-crosssensor\` (6523 files ≈ 3.2k pairs). Resolved by `src/utils/paths.py:262` `resolve_cache_dir` (override `paths.cache_dir=<abs>`).
- TACO source: `configs/base.yaml:184` `sen2naipv2.subset: sen2naipv2-crosssensor`, `min_correlation 0.9` `:210`, `num_samples 4409` `:203`; per-sample `.json` sidecar records `sample_id, crs, geotransform, correlation, profile`; `meta["subset"]` in load_sample. No .taco filename/version recorded beyond `tacofoundation:<subset>`.

## 5. tiled.py
- `src/infer/tiled.py:78` `sr_array(lr, model, scale=4, tile=256, overlap=32, device="cuda", amp=True)`; lr float32 `(C,H,W)` reflectance → `(C,4H,4W)` reflectance, never clipped. One tile per forward (**no batching**). Edge tiles are **reflect-padded to full `tile`** and cropped back. Blend: separable Hann ramp over the full overlap (`:34`, `:57`), ramps disabled on raster borders; weight-normalised accumulation.
- Model I/O: any `torch.nn.Module` returning a single tensor, reflectance in/out, output channels must equal input channels (output buffer sized from `lr.shape[0]`).
- `:141` `run_file(src, dst, model, ..., reflect_div=10000, out_dtype="uint16", bands=None, dn_offset=0.0)` — GeoTIFF DN→reflectance `(dn-dn_offset)/reflect_div`, writes uint16 (clip to [0,65535], logged) or float32; pixel size /scale. CLI `:272`, default device cuda-if-available, amp on.
- **Eval does NOT call tiled.py.** `scripts/eval_runA.py:367` `make_model_sr_fn` → `model(lr)` on batched reflectance patches; Evaluator (`src/metrics/aggregate.py:514`) and `scripts/eval_all_ckpts.py:283` `score_method` call it; no normalisation before/after. `TorchPredictor` uses `sr_array` with a single unpadded tile for inputs ≤ tile → bit-identical to `model(lr)` (verified, max diff 0.0).

## 6. Spectral consistency
- Operator: `src/losses/spectral.py:195` `down4(x, scale=4, mode="area")` — `F.interpolate(mode="area")` exact 4×4 block mean; antialiased modes only. Config `configs/base.yaml:538` `loss.spectral.downsample: area`, numerics `:545-561`; frozen `configs/frozen_day3.yaml:186`.
- `:335` `spectral_terms(sr, lr, scale=None, mode, eps, cos_clamp, min_norm, denorm=None, per_sample=False)` → `{"l1_spec" (reflectance), "sam" (radians), "sam_valid_frac"}`; `:501` `spectral_consistency` (weighted); `:568` `spectral_settings_from_cfg(cfg)` gives the kwargs.
- Floors (HR in place of SR, full VAL): l1_spec 0.005733, SAM 1.2748° (`configs/base.yaml:509-512`, `reports/day3_spectral_floor.md`).
- opensr-test consistency (`reflectance` L1 etc.): `src/eval/opensr_harness.py` (see §8).

## 7. Blur-hazard metrics
- `src/metrics/sharpness.py:129` `gradient_magnitude(image)` — mean Sobel magnitude, replicate padding, per band then averaged, `(B,)`.
- `:191` `hf_energy_ratio(image, cutoff=0.25)` — per-band mean-removed FFT power fraction with radial freq > cutoff×0.5 cyc/px, band-averaged; `DEFAULT_HF_CUTOFF=0.25` `:63`.
- `:264` `sharpness_terms`, `:294` `reference_sharpness(hr, lr, scale, cutoff)` (bicubic here is torch antialias **off**, differs from the eval bicubic).
- Val logging: `src/train.py:772` `validate()` → log.csv `val_sharpness`, `val_hf_energy`; run `sharpness_reference.json`. Blur index: `scripts/day3_ab_report.py:144` `sharpness_pass` (bicubic via `get_baseline`, antialias on) and `configs/base.yaml:1731-1733` (`blur_index_min 0.90`).

## 8. Metrics module
- `src/metrics/image_quality.py:278` `psnr(sr, hr, data_range=1.0)` → `BandMetric(per_band, mean)` (mean of per-band dB); `:342` `ssim(sr, hr, data_range, gaussian_weights, sigma, win_size)` (skimage); `:528` `lpips(sr, hr, rgb_indices, net="alex", data_range, clip_reflectance=True)` → `LPIPSResult(distance, clipped_fraction, caveat)` — RGB bands by name (`:448` `rgb_band_indices(cfg)`), NIR dropped, clip to [0,1] on a copy (only clip in module); `:658` `sam(sr, hr, zero_vector_policy)` → `SAMResult(mean_deg, map_deg, num_undefined)`; `:786` `ergas(sr, hr, scale, zero_mean_policy)`. Inputs `(C,H,W)` or `(B,C,H,W)`, ndarray or tensor.
- Data range fixed 1.0 reflectance (`configs/base.yaml:675`); settings `:664-727`. Harness `src/metrics/aggregate.py:343` `Evaluator(cfg)`, `.metrics_for_batch(sr, hr)` `:423`, `.run(loader, sr_fn, name, split)` `:514` → `EvaluationResult` (`per_sample` DataFrame, `settings`, `to_json/to_csv`).
- AlexNet LPIPS weights are cached locally (`~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`).
- opensr-test: `src/eval/opensr_harness.py:606` `score_arrays(lr, sr, hr, cfg, metrics=None, settings=None)` → `{metric: float}` over `OPENSR_METRICS` `:138` (reflectance, spectral, spatial, synthesis, ha_metric, om_metric, im_metric); `:210` `build_metrics(cfg)`; `:913` `run_opensr_test(dataloader, sr_fn, n_samples, cfg, ...)` → `OpenSRResult` (`to_frame`, `summary`, `to_json`). One sample at a time, `(C,H,W)` torch float, reflectance.

## 9. Bicubic baseline
- `src/eval/baselines.py:126` `bicubic_upsample(lr, scale, antialias=True)`; eval uses `get_baseline("bicubic", cfg.sr.scale)` (`:219`) → antialias=True, `align_corners=False`, unclipped (`configs/base.yaml:739` `baseline.antialias: true`).

## 10. Run outputs
- `run_metadata.json` (`M\runs\day3\a2\`): `config_hash, frozen_config, git{commit,commit_short,dirty,branch,dirty_files}, lambdas{spectral_lambda1,spectral_lambda2}, spectral_downsample, spectral_on, git_sha, git_dirty, seed, args{…train argparse…}, started_utc, torch, python, cuda`.
- `run_summary.json`: `config_hash, git_sha, git_dirty, lambdas, spectral_downsample, iters_requested, iters_completed, best_val_psnr, elapsed_min, final_val{psnr,l1_spec,sam_spec,sam_valid_frac,sharpness,hf_energy,ssim,lpips,sam_hr,ergas,n_val_batches,n_metric_batches}, sharpness_reference{…}, finished_utc`.
- `sharpness_reference.json`: `hr_sharpness, hr_hf_energy, bicubic_sharpness, bicubic_hf_energy, n_batches, hf_cutoff`. `args.json` = train args.
- Per-step val log `log.csv`: `iter,loss,lr,val_psnr,sec_per_100it,val_l1_spec,val_sam,val_sam_valid_frac,val_sharpness,val_hf_energy,val_ssim,val_lpips,val_sam_hr,val_ergas` (Run A log.csv has fewer columns). Val = 25-batch (400-patch) AMP subsample.
- Eval per-pair cache: `M\outputs\metrics\eval_all_ckpts\<run>\<itNNNNN>\{key.json,per_pair.csv,meta.json}` (runs: a2, b1, b2, runA, bicubic, hr_oracle); `scripts/eval_all_ckpts.py:243` `cached`. Report: `reports/day3_results.{md,json}`.
- Run A eval: `reports/day2_runA.json` / `.md` (gate block, per-method summaries), `reports/day2_curves.png`; per-sample CSVs `M\outputs\metrics\day2_runA_{method}.csv`.

## 11. Config
- `src/config.py:136` `load_frozen(path=configs/frozen_day3.yaml, verify=True)` recomputes SHA256 over canonical sorted-key JSON minus `config_hash` (`:78` `canonical_bytes`, `:121` `compute_config_hash`) and raises `FrozenConfigError` on mismatch; `python -m src.config --write` (`:244`) rewrites the hash. General loader: `src/utils/config.py:17` `load_config(path=configs/base.yaml, smoke=False, overrides=None)`, `:70` `add_standard_args`.
- `configs/frozen_day3.yaml` model values: name edsr_baseline, scale 4, n_resblocks 16, n_feats 48, in_ch 4, out_ch 4, res_scale 1.0, expected_parameters 855652; hash `e6082c69bba3086aaf71d0a413e64d621d860e5506da507ce2d76edb92f1fc0a`.

## 12. Post-hoc checkpoint selection
- Rule: `configs/base.yaml:1659-1673` `eval_all_ckpts.selection` — metric opensr-test `reflectance` (lowest), CI-tie via paired bootstrap (n_boot 10000, ci 0.95), tie-break `lpips` lower. **The consistency metric is opensr-test's, not the training `l1_spec`.**
- Implementation: `src/eval/ckpt_selection.py:285` `select_checkpoint(candidates, metric, tie_metric, tie_better, n_boot, ci, seed)`; driver `scripts/eval_all_ckpts.py` (`:172` `discover_checkpoints`, `:283` `score_method`).
- `scripts/day3_ab_report.py`: re-applies the rule to cached tables (`:285`), sharpness pass (`:144`), verdict/blur index, overfitting check (`:190`), figure (`:219`); config `configs/base.yaml:1724-1768`; outputs `reports/day3_ab_metrics.csv`, `reports/day3_ab_summary.json`, `reports/day3_val_curves.png`.

## 13. Data verification
- `scripts/verify_data_root.py` (`:513` `main`; checks cache `:213`, manifest `:252`, splits `:294`, catalog coverage `:334`, sample description `:413`).
- `reports/day3_data_validity.md:5`: **Verdict (b)** — real S2 L2A LR, NAIP HR radiometrically harmonised by the dataset authors; not a reporting bug (a), not synthetic degradation (c) (LR vs area-downsampled HR r ≈ 0.97, spatially offset). Machine-readable `reports/day3_data_validity.json`. Mandated wording: the sentence in RULES.md Day 3 addendum.

## 14. UI
- Stand-alone mock UI with drag-drop upload / uncertainty overlay: **NOT FOUND** (no HTML/py mock outside `frontend/`).
- Day 3 React + MapLibre skeleton `frontend/` (out of scope; reuse labels/colours only). Config block `configs/base.yaml:2011` (`frontend.map.center_lonlat [77.21, 28.61]`, zoom 11, OSM basemap; `tile.lr_px 256`, `overlap_lr_px 32`, `max_tiles 36`).
  - `frontend/src/App.tsx:87` title "DrishtiSR · Sentinel-2 10 m → 2.5 m"; sections "Area of interest", "Run super-resolution", "Uncertainty overlay", "Validation split, Day 3 results".
  - `frontend/src/components/AiBadge.tsx:1` text "AI-RECONSTRUCTED — NOT MEASURED DATA", `:14` Tailwind amber-300 on black ring.
  - `frontend/src/components/BackendBadge.tsx:18-20` checking slate-200/700, online emerald-600, offline slate-500.
  - `frontend/src/components/SwipeSlider.tsx:167` comparison slider ("Swipe between the 10 m input (left) and the AI reconstruction (right)"); `UncertaintyToggle.tsx`; `MetricsPanel.tsx` columns "PSNR dB", "SSIM", "LPIPS", "SAM°", "ERGAS", "L1_spec".
  - `frontend/src/map/layers.ts:78` AOI colours `#2563eb` (ok) / `#dc2626` (over budget). All network calls via `frontend/src/api/client.ts`; fixture `frontend/src/api/fixtures/metrics.json`.
  - Report palette (matplotlib): control `#2a78d6`, primary `#eb6834`, surface `#fcfcfb`, ink `#0b0b0b`, muted `#52514e`/`#898781` (`configs/base.yaml:1707-1767`).

## 15. Kaggle
- Kernel build folder `M\outputs\kaggle_kernels\day3\{day3.ipynb, kernel-metadata.json}` (generated, gitignored); downloads `M\outputs\kaggle\day3\` and `M\runs\kaggle\day3\`.
- `scripts/kaggle_run.py`: jobs from `configs/kaggle_jobs.yaml` (`day3` at `:277`, entry `scripts/day3_runs.py`); `:972` `generate_kernel`, `:795` `build_tokens`, `:867` `kernel_metadata` (dataset via `dataset_sources`), `:1465` `cmd_push` (refuses dirty/unpushed tree), `:1770` `cmd_fetch`. Code reaches Kaggle by **git clone of GitHub at a baked-in SHA** (`notebooks/templates/kaggle_job.py:47-48`); data via the attached dataset mount (`--data-root auto`).
- Metadata: `src/utils/gitmeta.py:74` `git_metadata(root=None)` → `{commit, commit_short, dirty, branch, dirty_files}`; `src/utils/kaggle_session.py`.

## 16. Main-tree snapshot (Step 1)
- Path `D:\SIH\DrishtiSR`, branch `main`, HEAD `1e7986eebcb06183646d64224044d81a5847b9cc`.
- `git status --porcelain`: empty (clean). Dirty import-surface files (src/{models,infer,losses,metrics,data}): **none** → no merge risk at snapshot time (P0a may have edited PROJECT_STATE.md / docs/PROGRESS.md / .gitignore since; `.gitignore` is also appended here — expect a trivial merge).

## 17. Python invocation
- `D:\SIH\DrishtiSR\.venv\Scripts\python.exe` (3.11.9) with cwd `W` and `PYTHONPATH=W` (root, not `W\src`). No editable install; `drishtisr.__file__` → `W\drishtisr\__init__.py`, `drishtisr.infer.tiled` → `W\src\infer\tiled.py`. `scripts/mvp/_py.py` not needed, not created. `PYTHONPATH=W\src` fails (`No module named 'drishtisr'`).
- Packages: torch 2.5.1+cpu, numpy 1.26.4, scipy 1.14.1, pillow 11.1.0, omegaconf 2.3.0, pytest 8.3.4, onnx 1.17.0, onnxruntime 1.20.1, rasterio 1.4.3, lpips 0.1.4, opensr-test 1.3.3, jsonschema 4.26.0. **MISSING: onnxsim, fastapi, uvicorn, python-multipart, psutil.**

## 18. Delhi Sentinel-2 imagery (Day 2)
- `M\data\delhi\delhi_A_20241030.tif` (4×1243×1294) and `M\data\delhi\delhi_B_20241030.tif` (4×1241×1295); uint16 raw L2A DN, band descriptions B04,B03,B02,B08, EPSG:32643, 10 m, nodata unset; item S2A_MSIL2A_20241030T052941_R105_T43RGM (baseline 05.11). Manifest `M\data\delhi\manifest.json` (`boa_add_offset_dn` −1000 per band, source "derived-from-processing-baseline").
- DN offset: raw, **not** offset-corrected → `reflectance = (dn − 1000) / 10000` (`configs/base.yaml:1534` `delhi.dn_offset: 1000.0`). Day 2 conversion: `src/infer/tiled.py:141` `run_file(..., dn_offset=1000)` / CLI `--dn-offset 1000`. Fetch: `scripts/fetch_delhi.py:242` `_band_offset`, `:302` `read_bbox_stack`. Prior SR outputs `M\outputs\delhi_A_sr.tif`, `delhi_A_sr_h1.tif`.
- AOIs `configs/base.yaml:1444-1450`: A central Delhi [77.15, 28.55, 77.28, 28.66], B Gurugram [77.05, 28.38, 77.18, 28.49].

## 19. Day 3 evidence
- `reports/day3_ab_metrics.csv` columns: `method,row,iteration,lambda1,lambda2,n_pairs,psnr_db,ssim,lpips,sam_deg,ergas,consistency_opensr_l1,consistency_l1_spec_area,sobel_ratio_vs_gt,hf_ratio_vs_gt,blur_index`; rows bicubic/baseline, {a2,b1,b2}×{selected,final} (n_pairs 1199). A2: PSNR 38.8708, SSIM 0.888901, LPIPS 0.332209, SAM 1.97205°, ERGAS 2.89657, opensr L1 0.00319858, l1_spec 0.00276109; bicubic: 38.4717 / 0.882467 / 0.403318 / 2.09246 / 3.05026 / 0.00217046 / 0.00121004.
- `reports/day3_ab_summary.json` keys: `seed, n_val, hf_cutoff, gt_sobel, gt_hf, bicubic_hf, metadata{a2,b1,b2}, metadata_flags, provenance_unanimous, verdicts{b1_selected,b2_selected,b1_final,b2_final}, overfitting, selection{run:{selected,final}}`.
- `reports/day3_val_curves.png` (A2 vs B1: PSNR, LPIPS, HF ratio vs GT from log.csv).
