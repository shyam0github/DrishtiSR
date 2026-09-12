"""P11 end-to-end: the real server (scripts/serve.py) in a subprocess over HTTP.

Replicates the UI flow (health -> samples -> upscale sample at tta 4 -> fetch every
image -> download SR GeoTIFF) and an upload flow, then compares live responses
with app/fixtures/*.json key by key (types and nullability) so the UI mock mode
and the live API cannot drift. Needs the A2 checkpoint (main tree) and app/samples
(scripts/mvp/build_samples.py); SKIPS naming what is missing.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator, List

import pytest

from src.utils.paths import repo_root

httpx = pytest.importorskip("httpx")
rasterio = pytest.importorskip("rasterio")
from rasterio.io import MemoryFile  # noqa: E402

ROOT = repo_root()
STATIC = ROOT / "app" / "static"
FIXTURES = ROOT / "app" / "fixtures"
SAMPLES = ROOT / "app" / "samples"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
STARTUP_S = 240

# Contract fields that may be null (docs/mvp/api_contract.md v1), as dotted paths.
NULLABLE = {
    "input.sample_id", "images.hr_rgb", "images.hr_fcc", "images.uncertainty",
    "downloads.uncertainty_tif", "metrics.reference_free.unc_mean",
    "metrics.reference_free.unc_p95", "metrics.reference_free.runtime_ms.uncertainty",
    "metrics.with_gt",
}


def _main_tree() -> Path:
    common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute",
                             "--git-common-dir"], check=True, capture_output=True, text=True).stdout.strip()
    return Path(common).parent


A2_CKPT = _main_tree() / "runs" / "day3" / "a2" / "last.pt"


# ------------------------------------------------------------ server
def _free_port(start: int = 8100) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port")


@pytest.fixture(scope="module")
def server() -> Iterator[httpx.Client]:
    if not A2_CKPT.is_file():
        pytest.skip(f"main-tree checkpoint absent: {A2_CKPT}")
    if not any(SAMPLES.glob("*/lr.npy")):
        pytest.skip("app/samples not built (python scripts/mvp/build_samples.py)")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
    env.pop("DRISHTI_UNC_CKPT", None)  # test the default engine
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "serve.py"), "--no-browser",
                             "--port", str(_free_port())], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines: deque = deque(maxlen=200)
    threading.Thread(target=lambda: [lines.append(l.rstrip()) for l in proc.stdout], daemon=True).start()
    base, deadline = None, time.monotonic() + STARTUP_S
    try:
        while time.monotonic() < deadline and proc.poll() is None:
            if base is None:
                m = next((re.search(r"(http://127\.0\.0\.1:\d+/)", l) for l in list(lines)
                          if l.startswith("DrishtiSR demo")), None)
                base = m.group(1) if m else None
            if base:
                try:
                    if httpx.get(base + "api/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
            time.sleep(0.5)
        else:
            pytest.fail("server did not become healthy:\n" + "\n".join(lines))
        with httpx.Client(base_url=base, timeout=300) as client:
            client.startup_log = list(lines)  # type: ignore[attr-defined]
            yield client
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


# ------------------------------------------------------------ helpers
def _kind(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "number"
    return type(v).__name__


def _drift(fixture: Any, live: Any, path: str = "") -> List[str]:
    """Differences in keys, types and nullability between fixture and live JSON."""
    fk, lk = _kind(fixture), _kind(live)
    if "null" in (fk, lk):
        return [] if fk == lk or path in NULLABLE else [f"{path}: {fk} vs {lk} (not nullable)"]
    if fk == "int" and lk == "number" or fk == "number" and lk == "int":
        return [] if fk == "number" else [f"{path}: fixture int, live float"]
    if fk != lk:
        return [f"{path}: fixture {fk}, live {lk}"]
    out: List[str] = []
    if fk == "dict":
        for k in sorted(set(fixture) ^ set(live)):
            out.append(f"{path}.{k}: only in {'fixture' if k in fixture else 'live'}".lstrip("."))
        for k in sorted(set(fixture) & set(live)):
            out += _drift(fixture[k], live[k], f"{path}.{k}".lstrip("."))
    elif fk == "list":
        for i, item in enumerate(live):
            if fixture:
                out += _drift(fixture[min(i, len(fixture) - 1)], item, f"{path}[]")
    return out


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _fetch_images(client, body) -> None:
    for key, url in body["images"].items():
        if url is None:
            continue
        r = client.get(url)
        assert r.status_code == 200, (key, url)
        assert r.headers["content-type"] == "image/png" and r.content[:8] == PNG_MAGIC, key


def _tif(client, url):
    r = client.get(url)
    assert r.status_code == 200, url
    assert r.headers["content-type"] == "image/tiff"
    with MemoryFile(r.content) as mem, mem.open() as src:
        return src.count, src.height, src.width, src.res, src.crs


def _pick_gt_sample(samples):
    gt = [s for s in samples if s["has_gt"]]
    if not gt:
        pytest.skip("no has_gt sample built")
    return next((s for s in gt if "high" in s["id"]), gt[0])  # high-texture first, like the demo


# ------------------------------------------------------------ tests
def test_startup_banner(server):
    log = "\n".join(server.startup_log)
    for field in ("backend", "checkpoint_id", "interim"):
        assert re.search(rf"^\s+{field}\s+\S", log, re.M), f"{field} missing from:\n{log}"


def test_health_matches_fixture(server):
    r = server.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert _drift(_fixture("health.json"), body) == []
    assert body["model"]["interim"] is False


def test_index_and_every_static_asset(server):
    r = server.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    refs = re.findall(r'(?:href|src)="([^"#:]+)"', r.text)
    assert refs, "index.html references no assets"
    for ref in refs:
        assert server.get("/" + ref.lstrip("/")).status_code == 200, ref
    for f in STATIC.rglob("*"):
        if f.is_file():
            rel = f.relative_to(STATIC).as_posix()
            resp = server.get("/" + rel)
            assert resp.status_code == 200 and resp.content == f.read_bytes(), rel


def test_samples_match_fixture(server):
    body = server.get("/api/samples").json()
    assert _drift(_fixture("samples.json"), body) == []
    for s in body["samples"]:
        r = server.get(s["thumb_url"])
        assert r.status_code == 200 and r.content[:8] == PNG_MAGIC, s["id"]


def test_ui_flow_sample_tta4(server):
    sample = _pick_gt_sample(server.get("/api/samples").json()["samples"])
    r = server.post("/api/upscale", data={"sample_id": sample["id"], "tta": "4", "dn_mode": "auto"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert _drift(_fixture("sample_response.json"), body) == []
    # the fixture is a has_gt tta4 response: nothing may be null here
    assert body["uncertainty_method"] == "tta4" and body["metrics"]["with_gt"] is not None
    assert all(v is not None for v in body["images"].values())
    assert body["input"]["lr_size"] == sample["lr_size"]
    _fetch_images(server, body)
    h, w = body["input"]["lr_size"]
    count, th, tw, _, _ = _tif(server, body["downloads"]["sr_tif"])
    assert (count, th, tw) == (4, 4 * h, 4 * w)
    count, th, tw, _, _ = _tif(server, body["downloads"]["uncertainty_tif"])
    assert (th, tw) == (4 * h, 4 * w)


def test_upload_flow(server):
    samples = server.get("/api/samples").json()["samples"]
    src = next((SAMPLES / s["id"] / "lr.tif" for s in samples
                if (SAMPLES / s["id"] / "lr.tif").is_file()), None)
    if src is None:
        pytest.skip("no sample lr.tif to upload")
    r = server.post("/api/upscale", files={"file": ("lr.tif", src.read_bytes(), "image/tiff")},
                    data={"tta": "0", "dn_mode": "auto"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert _drift(_fixture("sample_response.json"), body) == []
    assert body["input"]["source"] == "upload" and body["input"]["sample_id"] is None
    assert body["metrics"]["with_gt"] is None and body["images"]["uncertainty"] is None
    _fetch_images(server, body)
    with rasterio.open(src) as ds:
        in_res, in_crs = ds.res, ds.crs
    count, th, tw, res, crs = _tif(server, body["downloads"]["sr_tif"])
    h, w = body["input"]["lr_size"]
    assert (count, th, tw) == (4, 4 * h, 4 * w)
    if body["input"]["georeferenced"]:
        assert crs == in_crs and res == pytest.approx((in_res[0] / 4, in_res[1] / 4))


def test_error_shape(server):
    r = server.post("/api/upscale", data={"tta": "4"})
    assert r.status_code == 400 and set(r.json()) == {"error"}
