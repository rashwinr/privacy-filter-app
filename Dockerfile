# syntax=docker/dockerfile:1.7
# ---- Stage 1: build deps ----
# NOTE: Model weights are NOT pre-baked into the image. The 2.6 GB safetensors
# file is downloaded on first request to /tmp/hf_cache (the only writable dir
# on Cloud Run). This keeps Cloud Build well under its 30-min limit and image
# size small. First request is slow (~60-90s); subsequent are fast.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# System libs needed by extractors (Tesseract, Poppler for pdf2image, libGL for PIL/cv backends).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---- Stage 2: runtime ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_cache \
    HUGGINGFACE_HUB_CACHE=/tmp/hf_cache \
    PORT=8080 \
    HOST=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for Cloud Run best practices
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY app /app/app
COPY frontend /app/frontend

# Cloud Run mounts /tmp; we use ./data locally but allow override.
ENV LOCAL_DATA_DIR=/tmp/data
RUN mkdir -p /tmp/data/uploads /tmp/data/redacted /tmp/hf_cache \
 && chown -R appuser:appuser /tmp/data /tmp/hf_cache

USER appuser
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8080} --workers 1"]
