"""P9: FastAPI backend against docs/mvp/api_contract.md v1 (TestClient).

Needs the A2 checkpoint (main tree) and app/samples built by
scripts/mvp/build_samples.py; tests SKIP naming what is missing. Responses are
validated with a JSON Schema transcribed from the contract.
"""

from __future__ import annotations

import datetime as _dt
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from src.utils.gitmeta import git_metadata
from src.utils.paths import repo_root

jsonschema = pytest.importorskip("jsonschema")

WORKTREE = repo_root()
SAMPLES = WORKTREE / "app" / "samples"


def _main_tree() -> Path:
    common = subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "--path-format=absolute",
                             "--git-common-dir"], check=True, capture_output=True, text=True).stdout.strip()
    return Path(common).parent


A2_CKPT = _main_tree() / "runs" / "day3" / "a2" / "last.pt"

# ------------------------------------------------------------ contract schema
NUM = {"type": "number"}
NUM_N = {"type": ["number", "null"]}
URL = {"type": "string", "pattern": "^/"}
URL_N = {"type": ["string", "null"], "pattern": "^/"}
INT2 = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}


def obj(props, required=None):
    return {"type": "object", "properties": props, "required": list(required or props),
            "additionalProperties": False}


MODEL = obj({"backend": {"enum": ["onnx-int8", "onnx-fp32", "torch-fp32"]},
             "checkpoint_id": {"type": "string"}, "params": {"type": "integer"},
             "model_bytes": {"type": "integer"}, "threads": {"type": "integer"},
             "interim": {"type": "boolean"}, "has_scale_head": {"type": "boolean"}})
HEALTH = obj({"status": {"const": "ok"}, "model": MODEL})
SAMPLES_SCHEMA = obj({"samples": {"type": "array", "items": obj({
    "id": {"type": "string"}, "label": {"type": "string"}, "thumb_url": URL,
    "has_gt": {"type": "boolean"}, "lr_size": INT2})}})
GT_METRICS = obj({k: NUM for k in ("lpips", "ssim", "psnr", "sam_deg", "ergas")})
UPSCALE = obj({
    "job_id": {"type": "string"},
    "input": obj({"source": {"enum": ["upload", "sample"]}, "sample_id": {"type": ["string", "null"]},
                  "lr_size": INT2, "sr_size": INT2, "georeferenced": {"type": "boolean"},
                  "dn_mode_applied": {"type": "string"}}),
    "model": MODEL,
    "uncertainty_method": {"enum": ["none", "tta4", "tta8", "learned_laplace"]},
    "images": obj({**{k: URL for k in ("lr_rgb", "lr_fcc", "bicubic_rgb", "bicubic_fcc", "sr_rgb",
                                        "sr_fcc", "consistency", "sr_degraded_rgb",
                                        "consistency_sam")},
                   "hr_rgb": URL_N, "hr_fcc": URL_N, "uncertainty": URL_N}),
    "downloads": obj({"sr_tif": URL, "uncertainty_tif": URL_N}),
    "metrics": obj({
        "reference_free": obj({
            **{k: NUM for k in ("spec_l1", "spec_sam_deg", "spec_l1_bicubic", "spec_sam_bicubic_deg",
                                "hf_ratio_vs_bicubic")},
            "unc_mean": NUM_N, "unc_p95": NUM_N,
            "runtime_ms": obj({"sr": NUM, "uncertainty": NUM_N, "total": NUM})}),
        "with_gt": {"oneOf": [{"type": "null"}, obj({
            "sr": GT_METRICS, "bicubic": GT_METRICS, "spec_l1_hr": NUM, "spec_sam_hr_deg": NUM,
            "hf_ratio_hr_vs_bicubic": NUM})]}}),
    "refs": obj({"spec_l1_gt_floor": {"const": 0.005733}, "spec_sam_gt_floor_deg": {"const": 1.2748},
                 "unc_display_max": NUM, "cons_display_max": NUM,
                 "sharpness_warn_below": {"const": 1.05}}),
    "warnings": {"type": "array", "items": {"type": "string"}},
})
ERROR = obj({"error": {"type": "string"}})


def check(instance, schema):
    jsonschema.validate(instance, schema)


# ------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def client():
    if not A2_CKPT.is_file():
        pytest.skip(f"main-tree checkpoint absent: {A2_CKPT}")
    from fastapi.testclient import TestClient

    from app.server import create_app

    with TestClient(create_app()) as c:
        yield c


def _samples(client, has_gt=None):
    items = client.get("/api/samples").json()["samples"]
    return [s for s in items if has_gt is None or s["has_gt"] is has_gt]


def _geotiff(bands=4, size=128, dtype="float32", epsg=32643) -> bytes:
    rng = np.random.default_rng(0)
    base = rng.random((bands, size // 8, size // 8)).astype(np.float32)
    data = np.kron(base, np.ones((8, 8), np.float32)) * 0.3 + 0.05
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=size, height=size, count=bands, dtype=dtype,
                      crs=f"EPSG:{epsg}", transform=from_origin(710000.0, 3170000.0, 10.0, 10.0)) as dst:
            dst.write(data.astype(dtype))
        return mem.read()


def _post_file(client, payload: bytes, **fields):
    return client.post("/api/upscale", files={"file": ("in.tif", payload, "image/tiff")},
                       data={"tta": "0", **fields})


def _tif(client, url):
    r = client.get(url)
    assert r.status_code == 200, url
    with MemoryFile(r.content) as mem, mem.open() as src:
        return src.res, src.crs, src.width, src.height, src.count


def _assert_fetchable(client, body):
    for group in ("images", "downloads"):
        for key, url in body[group].items():
            if url is not None:
                assert client.get(url).status_code == 200, (key, url)


# ------------------------------------------------------------ tests
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    check(r.json(), HEALTH)
    assert r.json()["model"]["interim"] is False


def test_samples_list(client):
    r = client.get("/api/samples")
    assert r.status_code == 200
    check(r.json(), SAMPLES_SCHEMA)
    gt = _samples(client, True)
    if not gt:
        pytest.skip("app/samples not built (python scripts/mvp/build_samples.py)")
    assert len(gt) == 6
    for s in r.json()["samples"]:
        assert client.get(s["thumb_url"]).status_code == 200


@pytest.mark.parametrize("tta", ["0", "4"])
def test_upscale_sample(client, tta):
    gt = _samples(client, True)
    if not gt:
        pytest.skip("app/samples not built")
    r = client.post("/api/upscale", data={"sample_id": gt[0]["id"], "tta": tta})
    assert r.status_code == 200, r.text
    body = r.json()
    check(body, UPSCALE)
    assert body["input"]["source"] == "sample" and body["input"]["sample_id"] == gt[0]["id"]
    h, w = body["input"]["lr_size"]
    assert body["input"]["sr_size"] == [4 * h, 4 * w]
    assert body["metrics"]["with_gt"] is not None
    assert body["images"]["hr_rgb"] is not None
    if tta == "0":
        assert body["uncertainty_method"] == "none"
        assert body["images"]["uncertainty"] is None and body["downloads"]["uncertainty_tif"] is None
        assert body["metrics"]["reference_free"]["unc_mean"] is None
    else:
        assert body["uncertainty_method"] == "tta4"
        assert body["images"]["uncertainty"] is not None
    _assert_fetchable(client, body)


def test_upload_georeferenced(client):
    r = _post_file(client, _geotiff())
    assert r.status_code == 200, r.text
    body = r.json()
    check(body, UPSCALE)
    assert body["input"] == {"source": "upload", "sample_id": None, "lr_size": [128, 128],
                             "sr_size": [512, 512], "georeferenced": True,
                             "dn_mode_applied": "reflectance"}
    assert body["metrics"]["with_gt"] is None and body["images"]["hr_rgb"] is None
    assert any("band order" in w for w in body["warnings"])
    res, crs, w, h, count = _tif(client, body["downloads"]["sr_tif"])
    assert res == pytest.approx((2.5, 2.5)) and crs.to_epsg() == 32643
    assert (w, h, count) == (512, 512, 4)
    _assert_fetchable(client, body)


def test_upload_uint16_auto_warns_dn(client):
    payload = _geotiff(dtype="uint16")  # values < 1 truncate to 0: still exercises auto -> dn10000
    r = _post_file(client, payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input"]["dn_mode_applied"] == "dn10000"
    assert any("dn10000_offset1000" in w for w in body["warnings"])


def test_upload_three_bands_400(client):
    r = _post_file(client, _geotiff(bands=3))
    assert r.status_code == 400
    check(r.json(), ERROR)
    assert "bands" in r.json()["error"]


def test_upload_too_large_400(client):
    r = _post_file(client, _geotiff(size=600))
    assert r.status_code == 400
    check(r.json(), ERROR)
    assert "512" in r.json()["error"]


def test_bad_requests(client):
    r = client.post("/api/upscale", data={"tta": "4"})
    assert r.status_code == 400 and "exactly one" in r.json()["error"]
    r = client.post("/api/upscale", data={"sample_id": "nope", "tta": "4"})
    assert r.status_code == 400
    gt = _samples(client)
    if gt:
        r = client.post("/api/upscale", data={"sample_id": gt[0]["id"], "tta": "3"})
        assert r.status_code == 422
        check(r.json(), ERROR)


def test_files_path_traversal(client):
    for url in ("/files/..%2F..%2Fapp/server.py", "/files/x/..%5C..%5Cserver.py",
                "/files/../pipeline.py", "/files/nojob/sr.tif"):
        r = client.get(url)
        assert r.status_code == 404 or (r.status_code == 200 and "text/html" in r.headers["content-type"]), url


def test_static_and_fixtures(client):
    assert client.get("/").status_code == 200
    assert client.get("/fixtures/health.json").status_code == 200


def test_delhi_sample(client):
    delhi = [s for s in _samples(client, False) if s["id"].startswith("delhi")]
    if not delhi:
        pytest.skip("no Delhi samples built")
    r = client.post("/api/upscale", data={"sample_id": delhi[0]["id"], "tta": "0"})
    assert r.status_code == 200, r.text
    body = r.json()
    check(body, UPSCALE)
    assert body["metrics"]["with_gt"] is None and body["input"]["georeferenced"] is True
    res, crs, w, h, _ = _tif(client, body["downloads"]["sr_tif"])
    assert res == pytest.approx((2.5, 2.5)) and crs.to_epsg() == 32643
    lh, lw = body["input"]["lr_size"]
    assert (h, w) == (4 * lh, 4 * lw)


def _cpu_load():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out)
    except Exception:
        return None


def test_latency_report(client):
    """End-to-end request time for a 128^2 sample at tta 0/4/8 -> reports/mvp/api_latency.json."""
    gt = [s for s in _samples(client, True) if s["lr_size"] == [128, 128]]
    if not gt:
        pytest.skip("no 128x128 sample built")
    sid = gt[0]["id"]
    client.post("/api/upscale", data={"sample_id": sid, "tta": "0"})  # warm-up (LPIPS load)
    load = _cpu_load()
    result = {}
    for tta in ("0", "4", "8"):
        wall, rt = [], []
        for _ in range(3):
            t0 = time.perf_counter()
            r = client.post("/api/upscale", data={"sample_id": sid, "tta": tta})
            wall.append((time.perf_counter() - t0) * 1e3)
            assert r.status_code == 200
            rt.append(r.json()["metrics"]["reference_free"]["runtime_ms"])
        result[f"tta{tta}"] = {
            "request_ms_median": statistics.median(wall),
            "runtime_ms_sr_median": statistics.median(x["sr"] for x in rt),
            "runtime_ms_uncertainty_median": (statistics.median(x["uncertainty"] for x in rt)
                                              if rt[0]["uncertainty"] is not None else None),
            "runtime_ms_total_median": statistics.median(x["total"] for x in rt),
        }
    model = client.get("/api/health").json()["model"]
    git = git_metadata(WORKTREE)
    out = WORKTREE / "reports" / "mvp" / "api_latency.json"
    out.write_text(json.dumps({
        "what": "P9 API end-to-end POST /api/upscale time (TestClient, in-process) for a 128x128 TEST "
                "display sample, incl. renders, GeoTIFFs and GT metrics; median of 3 after warm-up",
        "checkpoint_id": model["checkpoint_id"], "backend": model["backend"], "threads": model["threads"],
        "sample_id": sid, "lr_size": [128, 128], "split": "test", "n_samples": 1, "repeats": 3,
        "cpu_load_percent_before": load, "provisional": load is None or load >= 20,
        "interim": model["interim"], "results": result,
        "git_sha": git.get("commit"), "git_dirty": git.get("dirty"),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
