"""stats.py — lightweight usage counters backed by GCS (or in-memory fallback).

Tracks:
  page_visits      — total page loads (GET /)
  unique_visitors  — unique IPs (hashed for privacy, stored as a set in GCS)
  docs_redacted    — total successful /api/redact completions

Storage layout (GCS backend):
  gs://<GCS_BUCKET>/<GCS_PREFIX>/stats/counters.json
  gs://<GCS_BUCKET>/<GCS_PREFIX>/stats/visitor_hashes.json  (set of sha256 hex)

All GCS reads/writes are wrapped in try/except so a GCS hiccup never breaks
the main redact endpoint. Falls back gracefully to in-memory counters.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# ── In-memory state (always used, GCS is the persistence layer) ───────────────
_lock = Lock()
_counters: dict[str, int] = {
    "page_visits": 0,
    "docs_redacted": 0,
    "unique_visitors": 0,
}
_visitor_hashes: set[str] = set()
_initialized = False


# ── GCS helpers ───────────────────────────────────────────────────────────────

def _gcs_enabled() -> bool:
    return os.getenv("STORAGE_BACKEND", "local").lower() == "gcs"


def _get_bucket():
    """Return the GCS bucket object (reuses existing auth logic)."""
    from app.storage import GCSStorage
    # We borrow the same auth mechanism as GCSStorage without creating a full
    # Storage instance — just need the bucket reference.
    client = GCSStorage._get_gcs_client()
    return client.bucket(os.getenv("GCS_BUCKET", "tanuh-bcd-bucket"))


def _gcs_prefix() -> str:
    prefix = os.getenv("GCS_PREFIX", "privacy-app").rstrip("/")
    return f"{prefix}/stats"


def _read_json_blob(bucket, name: str, default: Any) -> Any:
    try:
        blob = bucket.blob(name)
        if blob.exists():
            return json.loads(blob.download_as_text())
    except Exception as exc:
        logger.warning("stats: GCS read failed for %s: %s", name, exc)
    return default


def _write_json_blob(bucket, name: str, data: Any) -> None:
    try:
        blob = bucket.blob(name)
        blob.upload_from_string(json.dumps(data), content_type="application/json")
    except Exception as exc:
        logger.warning("stats: GCS write failed for %s: %s", name, exc)


# ── Initialisation (load persisted counts from GCS) ───────────────────────────

def _ensure_initialized() -> None:
    global _initialized, _counters, _visitor_hashes
    if _initialized:
        return
    if not _gcs_enabled():
        sync_with_storage()
        _initialized = True
        return
    try:
        bucket = _get_bucket()
        prefix = _gcs_prefix()
        saved = _read_json_blob(bucket, f"{prefix}/counters.json", {})
        _counters["page_visits"] = int(saved.get("page_visits", 0))
        _counters["docs_redacted"] = int(saved.get("docs_redacted", 0))
        
        # We no longer auto-sync with storage on every startup. 
        # This allows manual overrides in counters.json to persist.
        # To force a re-scan, sync_with_storage() can be called manually or via a trigger.
        
        hashes = _read_json_blob(bucket, f"{prefix}/visitor_hashes.json", [])
        _visitor_hashes = set(hashes)
        _counters["unique_visitors"] = len(_visitor_hashes)
        logger.info(
            "stats: loaded from GCS — visits=%d docs=%d visitors=%d",
            _counters["page_visits"],
            _counters["docs_redacted"],
            _counters["unique_visitors"],
        )
    except Exception as exc:
        logger.warning("stats: GCS init failed (in-memory only): %s", exc)
    _initialized = True


def _persist() -> None:
    """Flush current counters to GCS (best-effort, called under lock)."""
    if not _gcs_enabled():
        return
    try:
        bucket = _get_bucket()
        prefix = _gcs_prefix()
        _write_json_blob(bucket, f"{prefix}/counters.json", {
            "page_visits": _counters["page_visits"],
            "docs_redacted": _counters["docs_redacted"],
            "unique_visitors": _counters["unique_visitors"],
        })
        _write_json_blob(bucket, f"{prefix}/visitor_hashes.json",
                         list(_visitor_hashes))
    except Exception as exc:
        logger.warning("stats: GCS persist failed: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def record_visit(ip: str | None) -> None:
    """Increment page_visits; also track unique visitors by hashed IP."""
    with _lock:
        _ensure_initialized()
        _counters["page_visits"] += 1
        if ip:
            h = hashlib.sha256(ip.encode()).hexdigest()
            is_new = h not in _visitor_hashes
            if is_new:
                _visitor_hashes.add(h)
                _counters["unique_visitors"] = len(_visitor_hashes)
        _persist()


def record_redaction() -> None:
    """Increment docs_redacted after a successful /api/redact."""
    with _lock:
        _ensure_initialized()
        _counters["docs_redacted"] += 1
        _persist()


def get_stats() -> dict[str, int]:
    """Return a snapshot of current counters."""
    with _lock:
        _ensure_initialized()
        return dict(_counters)


def sync_with_storage() -> None:
    """Re-scan the 'redacted' storage folder to update the count."""
    try:
        from app.storage import get_storage
        storage = get_storage()
        files = storage.list_all("redacted")
        with _lock:
            _counters["docs_redacted"] = len(files)
        logger.info("stats: synced docs_redacted with storage: %d files found", len(files))
    except Exception as e:
        logger.warning("stats: failed to sync with storage: %s", e)
