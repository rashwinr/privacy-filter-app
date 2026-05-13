"""
privacy_filter/app/tasks.py — Celery background task for PII redaction.

Flow:
  1. Read uploaded file from temp path (written by the API before enqueue)
  2. Extract text via the registered handler
  3. Run the PrivacyFilter model (detect entities)
  4. Produce redacted output file
  5. Upload both originals + redacted to GCS/local storage
  6. Cache full RedactionResult JSON in Redis for 24 h
  7. Fire-and-forget session log
"""

import gc
import json
import logging
import os
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path

# Allow `from app.*` imports when the worker runs outside the package root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the shared Celery app from NHCX common module.
# The worker container sets PYTHONPATH=/app so `common` is importable.
from common.celery_app import celery_app  # noqa: E402

logger = logging.getLogger(__name__)

RESULT_TTL = int(os.getenv("TASK_RESULT_TTL", 86400))   # 24 h
SESSION_LOGGER_URL = os.getenv("SESSION_LOGGER_URL", "http://session-logger:8002")


def _get_redis():
    import redis as _redis
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return _redis.from_url(url, decode_responses=True)


def _fire_log(payload: dict):
    """POST a session log to the logger service — never raises."""
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            client.post(f"{SESSION_LOGGER_URL}/log", json=payload)
    except Exception as exc:
        logger.warning(f"[session-logger] fire-and-forget failed: {exc}")


@celery_app.task(bind=True, name="privacy_filter.app.tasks.process_redaction_task",
                 time_limit=900, soft_time_limit=840)
def process_redaction_task(self, upload_path: str, upload_key: str, job_id: str,
                            original_filename: str, content_type: str):
    """
    Async Celery task for PII redaction.
    `upload_path`  — absolute path to the uploaded file written by the API handler.
    `upload_key`   — storage key (used for GCS / local-storage lookup).
    `job_id`       — hex ID shared with the API caller.
    """
    task_id = self.request.id
    log_payload = {"service": "privacy-filter", "ip_address": "unknown"}

    def update(step: str, pct: int):
        self.update_state(state="PROGRESS",
                          meta={"step": step, "progress": pct, "task_id": task_id})

    try:
        from app.model import PrivacyFilter
        from app.redactor import get_handler
        from app.storage import get_storage, _guess_content_type
        from app.schemas import Entity

        update("Initialising", 5)
        upload_path_obj = Path(upload_path)
        if not upload_path_obj.exists():
            raise FileNotFoundError(f"Upload not found at {upload_path}")

        # 1. Push to storage (may already be there if API wrote it — safe no-op for GCS)
        storage = get_storage()
        update("Saving upload", 10)
        storage.save("uploads", upload_key, upload_path_obj.read_bytes())

        # 2. Extract text
        update("Extracting text", 20)
        handler = get_handler(original_filename)
        try:
            text = handler.extract(upload_path_obj)
        except Exception as e:
            raise RuntimeError(f"Extraction failed: {e}") from e

        # 3. Run privacy-filter model
        update("Running PII detection", 45)
        pf = PrivacyFilter.instance()
        pf.load()   # idempotent — model is cached after first load
        entities_raw = pf.detect(text) if text else []

        from app.schemas import Entity
        entities = [Entity(**e) for e in entities_raw]
        counts = Counter(e.entity_group for e in entities)

        # 4. Produce redacted output
        update("Redacting", 70)
        redacted_key = f"{job_id}__redacted{handler.out_extension}"
        tmp_redact_dir = Path(tempfile.gettempdir()) / "pf_redacted"
        tmp_redact_dir.mkdir(parents=True, exist_ok=True)
        redacted_local = tmp_redact_dir / redacted_key

        try:
            handler.redact(upload_path_obj, entities_raw, redacted_local)
        except Exception as e:
            raise RuntimeError(f"Redaction failed: {e}") from e

        update("Uploading redacted", 85)
        with open(redacted_local, "rb") as fh:
            storage.save("redacted", redacted_key, fh.read())

        # 5. Text previews
        preview_orig = text[:2000] if text else None
        redacted_preview = None
        if handler.name in {"text"}:
            redacted_preview = redacted_local.read_text(encoding="utf-8", errors="replace")[:2000]
        elif handler.name in {"pdf", "docx", "image", "dicom"}:
            try:
                redacted_preview = handler.extract(redacted_local)[:2000]
            except Exception:
                pass

        # 6. Build result and cache in Redis
        update("Storing result", 95)
        result = {
            "status": "completed",
            "task_id": task_id,
            "job_id": job_id,
            "filename": original_filename,
            "content_type": content_type,
            "entities": [e.model_dump() for e in entities],
            "entity_counts": dict(counts),
            "original_url": storage.url("uploads", upload_key),
            "redacted_url": storage.url("redacted", redacted_key),
            "text_preview_original": preview_orig,
            "text_preview_redacted": redacted_preview,
            "notes": None,
        }

        r = _get_redis()
        r.setex(f"result:{task_id}", RESULT_TTL, json.dumps(result))

        log_payload["pdf_location"] = f"pf_uploads/{upload_key}"
        update("Completed", 100)
        logger.info(f"[{task_id}] Privacy-filter task completed — {len(entities)} entities")

        # Record stats
        try:
            from app.stats import record_redaction
            record_redaction()
        except Exception:
            pass

        return result

    except Exception as exc:
        logger.exception(f"[{task_id}] Privacy-filter task failed: {exc}")
        error_payload = {"status": "failed", "task_id": task_id, "job_id": job_id, "error": str(exc)}
        try:
            r = _get_redis()
            r.setex(f"result:{task_id}", RESULT_TTL, json.dumps(error_payload))
        except Exception:
            pass
        raise

    finally:
        gc.collect()
        _fire_log(log_payload)
