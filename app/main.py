"""FastAPI app: privacy-filter PII detection + redaction service.

Endpoints
---------
GET  /                          → frontend (single-page HTML)
GET  /api/health                → liveness + model status
GET  /api/supported-types       → list of accepted file extensions
POST /api/redact                → multipart upload; returns RedactionResult
GET  /api/files/{kind}/{key}    → download originals or redacted outputs
                                  (kind ∈ {uploads, redacted})
"""
from __future__ import annotations

import gc
import logging
import os
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.model import PrivacyFilter
from app.redactor import get_handler, supported_extensions
from app.schemas import Entity, HealthResponse, RedactionResult
from app.storage import get_storage

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("privacy_filter")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load model so first request is fast.
    pf = PrivacyFilter.instance()
    try:
        pf.load()
    except Exception:
        # Don't crash startup — health endpoint will reflect failure.
        logger.exception("Model failed to load at startup")
    yield


app = FastAPI(
    title="Privacy Filter Test App",
    version="0.1.0",
    description="Upload a file → detect & redact personal information using openai/privacy-filter.",
    lifespan=lifespan,
)


# --- Static frontend (served from / ) ---
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return RedirectResponse("/docs")


@app.get("/api/health", response_model=HealthResponse)
async def health():
    pf = PrivacyFilter.instance()
    return HealthResponse(
        status="ok" if pf.loaded else "loading",
        model=pf.model_name,
        device=pf.device,
        model_loaded=pf.loaded,
    )


@app.get("/api/supported-types")
async def supported_types():
    return {"extensions": supported_extensions()}


@app.post("/api/redact", response_model=RedactionResult)
async def redact_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    try:
        handler = get_handler(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    storage = get_storage()
    job_id = uuid.uuid4().hex[:12]
    safe_name = Path(file.filename).name
    upload_key = f"{job_id}__{safe_name}"

    # Wrap the whole pipeline so we can guarantee a memory cleanup pass
    # at the end (Cloud Run's 4 GiB instances OOM-kill if buffers from a
    # previous request linger when a second large doc arrives, which the
    # client sees as HTTP 503).
    raw_bytes: bytes | None = None
    text: str | None = None
    entities_raw: list = []
    try:
        raw_bytes = await file.read()
        storage.save("uploads", upload_key, raw_bytes)
        upload_path = storage.local_path("uploads", upload_key)
        # Drop the in-memory copy as soon as it's on disk.
        raw_bytes = None

        # 1. Extract text
        try:
            text = handler.extract(upload_path)
        except Exception as e:
            logger.exception("Extraction failed")
            raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

        # 2. Run privacy-filter
        pf = PrivacyFilter.instance()
        try:
            entities_raw = pf.detect(text) if text else []
        except Exception as e:
            logger.exception("Model inference failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        entities = [Entity(**e) for e in entities_raw]
        counts = Counter(e.entity_group for e in entities)

        # 3. Produce redacted output (same format)
        redacted_key = f"{job_id}__redacted{handler.out_extension}"
        redacted_local = storage.local_path("redacted", redacted_key)
        redacted_local.parent.mkdir(parents=True, exist_ok=True)
        try:
            handler.redact(upload_path, entities_raw, redacted_local)
        except Exception as e:
            logger.exception("Redaction failed")
            raise HTTPException(status_code=500, detail=f"Redaction failed: {e}")

        # If using GCS, push the redacted bytes up.
        if os.getenv("STORAGE_BACKEND", "local").lower() == "gcs":
            with open(redacted_local, "rb") as f:
                storage.save("redacted", redacted_key, f.read())

        # 4. Build text previews (truncate)
        preview_orig = text[:2000] if text else None
        redacted_text_for_preview = None
        if handler.name in {"text"}:
            redacted_text_for_preview = redacted_local.read_text(encoding="utf-8", errors="replace")[:2000]
        elif handler.name in {"pdf", "docx", "image", "dicom"}:
            # Best-effort: re-extract from redacted output for preview
            try:
                redacted_text_for_preview = handler.extract(redacted_local)[:2000]
            except Exception:
                redacted_text_for_preview = None

        return RedactionResult(
            job_id=job_id,
            filename=safe_name,
            content_type=file.content_type or "application/octet-stream",
            entities=entities,
            entity_counts=dict(counts),
            original_url=storage.url("uploads", upload_key),
            redacted_url=storage.url("redacted", redacted_key),
            text_preview_original=preview_orig,
            text_preview_redacted=redacted_text_for_preview,
            notes=None,
        )
    finally:
        # Drop large transient buffers so the next request starts clean.
        raw_bytes = None
        text = None
        entities_raw = []
        gc.collect()


@app.get("/api/files/{kind}/{key}")
async def download_file(kind: str, key: str):
    if kind not in {"uploads", "redacted"}:
        raise HTTPException(status_code=404, detail="Unknown kind")
    storage = get_storage()
    if os.getenv("STORAGE_BACKEND", "local").lower() == "gcs":
        return RedirectResponse(storage.url(kind, key))
    p = storage.local_path(kind, key)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(p, filename=key.split("__", 1)[-1])
