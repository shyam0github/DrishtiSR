"""FastAPI server for DrishtiSR API contract v1 (docs/mvp/api_contract.md).

Run:  python -m app.server   (http://127.0.0.1:8000, cwd = repo root, PYTHONPATH = repo root)

Routes: /api/health, /api/samples, /api/samples/{id}/thumb.png, POST /api/upscale,
/files/{job_id}/{name} (job outputs), /fixtures (UI mock data), / (static UI,
mounted last). Every error is JSON ``{"error": str}``. No CORS middleware:
same-origin only, and a POST whose Origin differs from its Host is refused.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.io_geotiff import DN_MODES, MAX_SIDE, MIN_SIDE, InputError, read_input
from app.pipeline import APP_DIR, JOBS_DIR, SAFE_ID, Engine

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
TTA_VALUES = {"0": 0, "4": 4, "8": 8}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}\.(png|tif|json)$")
MEDIA = {"png": "image/png", "tif": "image/tiff", "json": "application/json"}


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def create_app(engine: Optional[Engine] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine if engine is not None else Engine()
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(title="DrishtiSR API", version="1", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    # ------------------------------------------------------------ errors
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return _error(exc.status, exc.message)

    @app.exception_handler(InputError)
    async def _input_error(_: Request, exc: InputError):
        return _error(exc.status, str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        parts = [f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg')}" for e in exc.errors()]
        return _error(422, "invalid request: " + "; ".join(parts))

    @app.exception_handler(Exception)
    async def _internal_error(_: Request, exc: Exception):
        return _error(500, f"internal error: {type(exc).__name__}: {exc}")

    @app.middleware("http")
    async def _same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and origin != "null":
            if urlparse(origin).netloc != request.headers.get("host", ""):
                return _error(400, "cross-origin requests are not accepted; use the same origin.")
        return await call_next(request)

    # ------------------------------------------------------------ API
    @app.get("/api/health")
    def health(request: Request):
        return {"status": "ok", "model": request.app.state.engine.model_info()}

    @app.get("/api/samples")
    def samples(request: Request):
        return {"samples": request.app.state.engine.list_samples()}

    @app.get("/api/samples/{sample_id}/thumb.png")
    def sample_thumb(sample_id: str, request: Request):
        d = request.app.state.engine.sample_dir(sample_id)
        if d is None or not (d / "thumb.png").is_file():
            raise ApiError(404, f"unknown sample {sample_id!r}.")
        return FileResponse(d / "thumb.png", media_type="image/png")

    @app.post("/api/upscale")
    def upscale(request: Request,
                file: Optional[UploadFile] = File(None),
                sample_id: Optional[str] = Form(None),
                tta: str = Form("4"),
                dn_mode: str = Form("auto")):
        engine: Engine = request.app.state.engine
        if file is not None and not file.filename and not file.size:
            file = None
        sample_id = sample_id or None
        if (file is None) == (sample_id is None):
            raise ApiError(400, "send exactly one of 'file' (a 4-band GeoTIFF) or 'sample_id'.")
        if tta not in TTA_VALUES:
            raise ApiError(422, f"tta must be one of {sorted(TTA_VALUES)}; got {tta!r}.")
        if dn_mode not in DN_MODES:
            raise ApiError(422, f"dn_mode must be one of {list(DN_MODES)}; got {dn_mode!r}.")

        if file is not None:
            data = file.file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise ApiError(400, f"upload larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
            lr, profile, applied, warns = read_input(data, dn_mode)
            return engine.run(lr, None, TTA_VALUES[tta], profile, "upload", None, applied, warns)

        loaded = engine.load_sample(sample_id)
        if loaded is None:
            raise ApiError(400, f"unknown sample_id {sample_id!r}; see /api/samples.")
        lr, hr, profile, manifest = loaded
        _, h, w = lr.shape
        if not (MIN_SIDE <= h <= MAX_SIDE and MIN_SIDE <= w <= MAX_SIDE):
            raise ApiError(400, f"sample LR size {h}x{w} outside [{MIN_SIDE}, {MAX_SIDE}].")
        warns = []
        if dn_mode not in ("auto", "reflectance"):
            warns.append(f"dn_mode={dn_mode} ignored: samples are stored as reflectance.")
        return engine.run(lr, hr, TTA_VALUES[tta], profile, "sample", sample_id, "reflectance", warns)

    @app.get("/files/{job_id}/{name}")
    def job_file(job_id: str, name: str):
        if not SAFE_ID.match(job_id) or not SAFE_NAME.match(name):
            raise ApiError(404, "file not found.")
        root = JOBS_DIR.resolve()
        path = (root / job_id / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ApiError(404, "file not found.")
        return FileResponse(path, media_type=MEDIA[path.suffix.lstrip(".")], filename=name)

    app.mount("/fixtures", StaticFiles(directory=APP_DIR / "fixtures"), name="fixtures")
    app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
