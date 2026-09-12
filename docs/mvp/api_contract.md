# DrishtiSR API contract v1 (MVP)
Base http://127.0.0.1:8000. JSON. Reflectance units unless stated.

MODEL = {"backend":"onnx-int8"|"onnx-fp32"|"torch-fp32","checkpoint_id":str,"params":int,"model_bytes":int,"threads":int,"interim":bool,"has_scale_head":bool}

GET /api/health → 200 {"status":"ok","model":MODEL}

GET /api/samples → 200 {"samples":[{"id":str,"label":str,"thumb_url":str,"has_gt":bool,"lr_size":[int,int]}]}

POST /api/upscale (multipart/form-data)
  fields: file (GeoTIFF, 4 bands) XOR sample_id (str)
          tta: "0"|"4"|"8" (default "4")
          dn_mode: "auto"|"reflectance"|"dn10000"|"dn10000_offset1000" (default "auto")
  limits: LR height and width each within [32, 512], else 400
→ 200
{
  "job_id": str,
  "input": {"source":"upload"|"sample","sample_id":str|null,"lr_size":[h,w],"sr_size":[4h,4w],"georeferenced":bool,"dn_mode_applied":str},
  "model": MODEL,
  "uncertainty_method": "none"|"tta4"|"tta8"|"learned_laplace",
  "images": {
    "lr_rgb":url, "lr_fcc":url, "bicubic_rgb":url, "bicubic_fcc":url, "sr_rgb":url, "sr_fcc":url,
    "hr_rgb":url|null, "hr_fcc":url|null,
    "uncertainty":url|null,
    "consistency":url
  },
  "downloads": {"sr_tif":url, "uncertainty_tif":url|null},
  "metrics": {
    "reference_free": {
      "spec_l1":float, "spec_sam_deg":float,
      "spec_l1_bicubic":float, "spec_sam_bicubic_deg":float,
      "hf_ratio_vs_bicubic":float,
      "unc_mean":float|null, "unc_p95":float|null,
      "runtime_ms": {"sr":float, "uncertainty":float|null, "total":float}
    },
    "with_gt": null | {
      "sr":      {"lpips":float,"ssim":float,"psnr":float,"sam_deg":float,"ergas":float},
      "bicubic": {"lpips":float,"ssim":float,"psnr":float,"sam_deg":float,"ergas":float},
      "spec_l1_hr":float, "spec_sam_hr_deg":float,
      "hf_ratio_hr_vs_bicubic":float
    }
  },
  "refs": {"spec_l1_gt_floor":0.005733, "spec_sam_gt_floor_deg":1.2748, "unc_display_max":float, "cons_display_max":float, "sharpness_warn_below":1.05},
  "warnings": [str]
}
Errors → 400 / 422 / 500 with {"error": str}

GET /files/{job_id}/{name} → PNG or GeoTIFF

## Semantics
- spec_* = degrade(X) compared with LR, using the training degradation operator. Bicubic sits near 0; HR sits at the floor; low values alone can mean blur.
- hf_ratio_vs_bicubic = HF energy(SR) / HF energy(bicubic); bicubic = 1.0.
- SR shown and measured = single forward pass (no self-ensemble). TTA outputs are used only for uncertainty.
- Overlays: RGBA PNG at SR size, fixed scales 0..unc_display_max and 0..cons_display_max (never per-image min–max). Consistency map is the 10 m map nearest-upsampled ×4.
- Display stretch: per-band 2–98 percentile computed on the LR input, applied identically to LR, bicubic, SR and HR. RGB uses the inventory band order; FCC = NIR, R, G.
- UI ordering: with_gt present → LPIPS, spectral consistency, sharpness, then SSIM, SAM, ERGAS, PSNR last. Absent → spectral consistency, sharpness, uncertainty, runtime.
- Blur guard: hf_ratio_vs_bicubic < sharpness_warn_below → UI warns that low spectral error can come from blur.
