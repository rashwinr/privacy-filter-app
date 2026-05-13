"""FastAPI app: privacy-filter PII detection + redaction service.

Endpoints
---------
GET  /                          → frontend (single-page HTML)
GET  /api/health                → liveness + model status
GET  /api/supported-types       → list of accepted file extensions
GET  /api/stats                 → live usage counters (visits, docs redacted)
POST /api/demo-token            → self-service JWT (name + email → signed token)
POST /api/redact                → multipart upload; returns RedactionResult  [auth required]
GET  /api/files/{kind}/{key}    → download originals or redacted outputs      [auth required]
                                  (kind ∈ {uploads, redacted})
"""
from __future__ import annotations

import gc
import json
import logging
import os
import tempfile
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from jose import jwt
from pydantic import BaseModel, EmailStr

from app.auth import require_bearer
from app.model import PrivacyFilter
from app.redactor import get_handler, supported_extensions
from app.schemas import Entity, HealthResponse, RedactionResult
from app.stats import get_stats, record_redaction, record_visit
from app.storage import get_storage, _guess_content_type

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
async def root(request: Request):
    # Record each page load; track unique visitors by client IP.
    client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() \
                or (request.client.host if request.client else None)
    try:
        record_visit(client_ip)
    except Exception:
        pass  # Never let stats tracking break the page load.
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


# ---------------------------------------------------------------------------
# Demo token — self-service JWT issuance
# ---------------------------------------------------------------------------

class DemoTokenRequest(BaseModel):
    name: str
    email: EmailStr


@app.post("/api/demo-token")
async def create_demo_token(body: DemoTokenRequest):
    """Issue a signed demo JWT for the given name + email.

    The token is valid for DEMO_TOKEN_EXPIRY_DAYS days (default: 7).
    Pass it as ``Authorization: Bearer <token>`` on protected endpoints.
    """
    secret = os.getenv("SECRET_KEY", "")
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="Demo tokens are not available: SECRET_KEY is not configured.",
        )

    expiry_days = int(os.getenv("DEMO_TOKEN_EXPIRY_DAYS", "7"))
    now = int(time.time())
    payload = {
        "sub": body.email,
        "name": body.name,
        "email": body.email,
        "type": "demo",
        "iat": now,
        "exp": now + expiry_days * 86_400,
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    logger.info("Demo token issued for %s (%s), expires in %dd", body.name, body.email, expiry_days)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_days": expiry_days,
        "name": body.name,
        "email": body.email,
    }


@app.post("/api/redact", response_model=RedactionResult)
async def redact_file(
    file: UploadFile = File(...),
    _claims: Dict[str, Any] = Depends(require_bearer),
):
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

        # Write to a local temp path first so the extractor can read it
        # without a round-trip GCS download after save().
        tmp_upload_dir = Path(tempfile.gettempdir()) / "pf_uploads"
        tmp_upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = tmp_upload_dir / upload_key
        upload_path.write_bytes(raw_bytes)

        # Push to configured storage (GCS or local ./data).
        storage.save("uploads", upload_key, raw_bytes)
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

        # Always write to a local temp path first — GCSStorage.local_path()
        # would attempt a GCS download for a file that doesn't exist yet.
        tmp_redact_dir = Path(tempfile.gettempdir()) / "pf_redacted"
        tmp_redact_dir.mkdir(parents=True, exist_ok=True)
        redacted_local = tmp_redact_dir / redacted_key

        try:
            handler.redact(upload_path, entities_raw, redacted_local)
        except Exception as e:
            logger.exception("Redaction failed")
            raise HTTPException(status_code=500, detail=f"Redaction failed: {e}")

        # Upload to GCS (or local storage keeps the file in place).
        with open(redacted_local, "rb") as fh:
            storage.save("redacted", redacted_key, fh.read())


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

        result = RedactionResult(
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
        # Count successful redactions for the dashboard.
        try:
            record_redaction()
        except Exception:
            pass
        return result
    finally:
        # Drop large transient buffers so the next request starts clean.
        raw_bytes = None
        text = None
        entities_raw = []
        gc.collect()


@app.get("/api/files/{kind}/{key}")
async def download_file(
    kind: str,
    key: str,
    _claims: Dict[str, Any] = Depends(require_bearer),
):
    if kind not in {"uploads", "redacted"}:
        raise HTTPException(status_code=404, detail="Unknown kind")
    storage = get_storage()
    if os.getenv("STORAGE_BACKEND", "local").lower() == "gcs":
        try:
            data = storage.open_read(kind, key)
        except Exception as e:
            logger.exception("GCS download failed")
            raise HTTPException(status_code=404, detail=f"File not found in GCS: {e}")
        filename = key.split("__", 1)[-1]
        content_type = _guess_content_type(filename)
        return StreamingResponse(
            data,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    p = storage.local_path(kind, key)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(p, filename=key.split("__", 1)[-1])


# ---------------------------------------------------------------------------
# Async submit — enqueue via Celery, return task_id immediately (202)
# ---------------------------------------------------------------------------

@app.post("/api/submit", status_code=202)
async def submit_redact(
    file: UploadFile = File(...),
    _claims: Dict[str, Any] = Depends(require_bearer),
):
    """Enqueue a redaction job and return a task_id immediately (202 Accepted).

    Poll ``GET /api/task-status/{task_id}`` to check progress and retrieve the
    full RedactionResult when the job finishes.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    try:
        from app.redactor import get_handler
        get_handler(file.filename)           # validate extension early
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    # Write the upload to a shared temp path the worker can read.
    job_id = uuid.uuid4().hex[:12]
    safe_name = Path(file.filename).name
    upload_key = f"{job_id}__{safe_name}"

    tmp_upload_dir = Path(tempfile.gettempdir()) / "pf_uploads"
    tmp_upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = tmp_upload_dir / upload_key
    upload_path.write_bytes(await file.read())

    from app.tasks import process_redaction_task
    result = process_redaction_task.apply_async(
        kwargs={
            "upload_path": str(upload_path),
            "upload_key": upload_key,
            "job_id": job_id,
            "original_filename": safe_name,
            "content_type": file.content_type or "application/octet-stream",
        }
    )
    task_id = result.id
    logger.info("Enqueued privacy-filter task %s for file %s", task_id, safe_name)
    return {"task_id": task_id, "job_id": job_id, "status": "queued"}


@app.get("/api/task-status/{task_id}")
async def task_status(task_id: str):
    """Poll the status of an async redaction job.

    Returns JSON with ``status`` ∈ {queued, running, completed, failed}.
    When ``status == completed`` the full RedactionResult is included.
    """
    import redis as _redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # 1. Check our own result cache first (set by the task on completion)
    try:
        r = _redis.from_url(redis_url, decode_responses=True)
        cached = r.get(f"result:{task_id}")
        if cached:
            data = json.loads(cached)
            return data
    except Exception:
        pass

    # 2. Fall back to Celery task state
    try:
        from app.tasks import process_redaction_task
        async_result = process_redaction_task.AsyncResult(task_id)
        state = async_result.state       # PENDING / PROGRESS / SUCCESS / FAILURE
        if state == "PENDING":
            return {"task_id": task_id, "status": "queued"}
        if state == "PROGRESS":
            meta = async_result.info or {}
            return {"task_id": task_id, "status": "running",
                    "step": meta.get("step"), "progress": meta.get("progress")}
        if state == "SUCCESS":
            result = async_result.result or {}
            result["status"] = "completed"
            return result
        if state == "FAILURE":
            return {"task_id": task_id, "status": "failed",
                    "error": str(async_result.result)}
        return {"task_id": task_id, "status": state.lower()}
    except Exception as e:
        logger.exception("task_status lookup failed")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/stats")
async def stats():
    """Return live usage counters for the dashboard."""
    try:
        return get_stats()
    except Exception as e:
        logger.exception("Stats fetch failed")
        return {"page_visits": 0, "unique_visitors": 0, "docs_redacted": 0}

