# DrishtiSR demo, Hugging Face Docker-SDK Space (CPU). Build context = a staging
# folder that mirrors the repo layout (see deploy/hf_space/README.md, step 3):
#   app/ src/ drishtisr/ configs/ artifacts/onnx/a2-last-dce224ec/
#   reports/mvp/{trust_scales,quant_gate,projection_gate}.json ckpt/a2/last.pt
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime deps only (measured import surface of app.server + one upscale), pinned
# to the project venv. lpips with --no-deps so it cannot resolve another torch.
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install numpy==1.26.4 scipy==1.14.1 pandas==2.2.3 scikit-image==0.25.2 pillow==11.1.0 \
        rasterio==1.4.3 omegaconf==2.3.0 pyyaml==6.0.2 onnx==1.17.0 onnxruntime==1.20.1 \
        tqdm==4.67.1 fastapi==0.141.1 uvicorn==0.52.4 python-multipart==0.0.32 \
 && pip install --no-deps lpips==0.1.4

# HF Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    TORCH_HOME=/home/user/.cache/torch \
    PYTHONPATH=/home/user/app \
    DRISHTI_CKPT=/home/user/app/ckpt/a2/last.pt \
    DRISHTI_BACKEND=auto \
    DRISHTI_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
WORKDIR /home/user/app

# LPIPS AlexNet backbone (~233 MB) fetched at build time, not on the first request.
RUN python -c "import lpips; lpips.LPIPS(net='alex', verbose=False)"

COPY --chown=user configs/ configs/
COPY --chown=user drishtisr/ drishtisr/
COPY --chown=user src/ src/
COPY --chown=user app/ app/
COPY --chown=user artifacts/onnx/a2-last-dce224ec/ artifacts/onnx/a2-last-dce224ec/
COPY --chown=user reports/mvp/trust_scales.json reports/mvp/quant_gate.json reports/mvp/projection_gate.json reports/mvp/
COPY --chown=user ckpt/a2/last.pt ckpt/a2/last.pt
RUN mkdir -p app/_jobs

EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn app.server:app --host 0.0.0.0 --port 7860 --workers 1"]
