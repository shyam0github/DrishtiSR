"""Static checks for the demo web UI (app/static) and its mock fixtures (app/fixtures)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "app" / "static"
FIXTURES = ROOT / "app" / "fixtures"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def test_static_files_exist():
    for name in ("index.html", "app.js", "styles.css"):
        assert (STATIC / name).is_file(), name


def test_every_js_element_id_exists_in_html():
    js = _read("app.js")
    html_ids = set(re.findall(r'\bid="([^"]+)"', _read("index.html")))
    js_ids = set(re.findall(r'\$\(\s*["\']([^"\']+)["\']\s*\)', js))
    js_ids |= set(re.findall(r'getElementById\(\s*["\']([^"\']+)["\']\s*\)', js))
    js_ids |= set(re.findall(r'closest\(\s*["\']#([\w-]+)["\']\s*\)', js))
    assert len(js_ids) > 30, "id extraction found suspiciously few ids"
    missing = sorted(js_ids - html_ids)
    assert not missing, f"ids used in app.js but absent from index.html: {missing}"


def test_html_ids_unique():
    ids = re.findall(r'\bid="([^"]+)"', _read("index.html"))
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, dupes


@pytest.mark.parametrize("name", ["health.json", "samples.json", "sample_response.json"])
def test_fixture_urls_resolve(name):
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    urls = [s for s in _strings(data) if s.startswith("/")]
    for url in urls:
        if url.startswith("/files/"):
            # Server job outputs (downloads); the UI disables these links in mock mode.
            continue
        assert url.startswith("/fixtures/"), f"{name}: unexpected URL {url}"
        assert (FIXTURES / url.removeprefix("/fixtures/")).is_file(), f"{name}: missing {url}"


def test_fixture_overlays_are_rgba():
    from PIL import Image

    data = json.loads((FIXTURES / "sample_response.json").read_text(encoding="utf-8"))
    for key in ("uncertainty", "consistency"):
        with Image.open(FIXTURES / data["images"][key].removeprefix("/fixtures/")) as im:
            assert im.mode == "RGBA", key


def test_no_external_references():
    for p in STATIC.rglob("*"):
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            hits = re.findall(r"https?://\S+", text)
            assert not hits, f"{p.name} references external URLs: {hits[:3]}"


def test_html_references_local_assets():
    html = _read("index.html")
    assert 'src="app.js"' in html and 'type="module"' in html
    assert 'href="styles.css"' in html


def test_required_ui_text_present():
    html = _read("index.html")
    js = _read("app.js")
    assert "AI-reconstructed 2.5 m — not a measurement" in html
    assert "Turing Testers" in html
    assert "INTERIM checkpoint" in html
    assert "Uncertainty (TTA disagreement)" in js and "Uncertainty (learned σ)" in js
    assert "HR itself sits at the floor — near zero can mean blur" in js
    rules = (ROOT / "docs" / "mvp" / "RULES.md").read_text(encoding="utf-8")
    m = re.search(r'exactly this sentence[^"]*"([^"]+)"', rules)
    assert m, "data sentence not found in RULES.md"
    assert m.group(1) in html, "About-the-data sentence must match RULES.md verbatim"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_node_check_app_js():
    res = subprocess.run(["node", "--check", str(STATIC / "app.js")], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
