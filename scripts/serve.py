"""One-command DrishtiSR demo: load the model, serve the UI, open the browser.

Run (RULES.md Python mechanism; cwd and PYTHONPATH = worktree root):
    D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe scripts/serve.py [--port N] [--no-browser]

Threads: DRISHTI_THREADS (default 6) sets torch/onnxruntime threads;
OMP_NUM_THREADS / MKL_NUM_THREADS default to the same value unless already set.
The first free port from 8000 (or --port) upward is used. Other engine settings
(DRISHTI_CKPT, DRISHTI_BACKEND, DRISHTI_PROJECT, DRISHTI_UNC_CKPT) are read by
app.pipeline.Engine.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# Thread env must be set before torch / onnxruntime are imported.
THREADS = os.environ.setdefault("DRISHTI_THREADS", "6")
os.environ.setdefault("OMP_NUM_THREADS", THREADS)
os.environ.setdefault("MKL_NUM_THREADS", THREADS)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def free_port(host: str, start: int, tries: int = 100) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port in {start}..{start + tries - 1}")


def _open_when_ready(url: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "api/health", timeout=2) as r:
                if r.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(0.3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000, help="first port to try (default 8000)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    import uvicorn

    from app.pipeline import Engine
    from app.server import create_app

    port = free_port(args.host, args.port)
    engine = Engine()
    info = engine.model_info()
    url = f"http://{args.host}:{port}/"
    print(f"DrishtiSR demo  {url}", flush=True)
    print(f"  backend       {info['backend']}  ({engine.backend_reason})", flush=True)
    print(f"  checkpoint_id {info['checkpoint_id']}", flush=True)
    print(f"  interim       {str(info['interim']).lower()}", flush=True)
    print(f"  threads       {info['threads']} (OMP {os.environ['OMP_NUM_THREADS']}, "
          f"MKL {os.environ['MKL_NUM_THREADS']})", flush=True)
    for w in engine.startup_warnings:
        print(f"  WARNING       {w}", flush=True)
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    uvicorn.run(create_app(engine), host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
