"""Storage abstraction: local filesystem or GCS.

Switch via STORAGE_BACKEND env var: "local" (default) or "gcs".

Local backend
─────────────
  Writes to ./data/{uploads,redacted}/{job_id}__{name}

GCS backend
───────────
  Bucket  : GCS_BUCKET   (default: tanuh-bcd-bucket)
  Prefix  : GCS_PREFIX   (default: privacy-app)
  Layout  :
    privacy-app/uploads/<job_id>__<filename>
    privacy-app/redacted/<job_id>__redacted.<ext>

Auth priority (matches NHCX pattern):
  1. GCS_CREDENTIALS_JSON env var  → dedicated GCS service account JSON path
  2. GOOGLE_APPLICATION_CREDENTIALS → shared SA or ADC path
  3. Plain ADC (GCP metadata server / gcloud login)

Failures are surfaced — callers should handle gracefully.
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Optional

logger = logging.getLogger(__name__)


class Storage(ABC):
    @abstractmethod
    def save(self, kind: str, key: str, data: bytes) -> str:
        """kind: 'uploads' | 'redacted'. Returns a stable path/URI."""

    @abstractmethod
    def open_read(self, kind: str, key: str) -> BinaryIO: ...

    @abstractmethod
    def local_path(self, kind: str, key: str) -> Path:
        """Return a *local* path. For GCS, downloads to a temp file first."""

    @abstractmethod
    def url(self, kind: str, key: str) -> str:
        """Return a URL the frontend can hit."""


class LocalStorage(Storage):
    def __init__(self, root: str = "./data") -> None:
        self.root = Path(root).resolve()
        (self.root / "uploads").mkdir(parents=True, exist_ok=True)
        (self.root / "redacted").mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, key: str) -> Path:
        assert kind in {"uploads", "redacted"}
        return self.root / kind / key

    def save(self, kind: str, key: str, data: bytes) -> str:
        p = self._path(kind, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            f.write(data)
        return str(p)

    def open_read(self, kind: str, key: str):
        return self._path(kind, key).open("rb")

    def local_path(self, kind: str, key: str) -> Path:
        return self._path(kind, key)

    def url(self, kind: str, key: str) -> str:
        # Served via FastAPI route /api/files/{kind}/{key}
        return f"/api/files/{kind}/{key}"


class GCSStorage(Storage):  # pragma: no cover - exercised in cloud
    """
    Stores files in gs://<GCS_BUCKET>/<GCS_PREFIX>/{uploads,redacted}/<key>

    Default target: gs://tanuh-bcd-bucket/privacy-app/uploads/...
                    gs://tanuh-bcd-bucket/privacy-app/redacted/...
    """

    def __init__(self, bucket: str, prefix: str = "privacy-app") -> None:
        from google.cloud import storage as gcs

        self.gcs_client = self._get_gcs_client()
        self.bucket = self.gcs_client.bucket(bucket)
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        logger.info(
            f"GCSStorage initialised — bucket={bucket}, prefix={self.prefix!r}"
        )

    # ── Auth ────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_gcs_client():
        """
        Return an authenticated GCS client.

        Priority:
          1. GCS_CREDENTIALS_JSON  → dedicated GCS service account (recommended)
          2. GOOGLE_APPLICATION_CREDENTIALS → shared SA
          3. ADC (metadata server / gcloud login)
        """
        from google.cloud import storage as gcs

        gcs_creds = os.getenv("GCS_CREDENTIALS_JSON", "")
        gcp_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

        if gcs_creds and os.path.isfile(gcs_creds):
            logger.info(f"GCS auth: dedicated GCS SA → {gcs_creds}")
            return gcs.Client.from_service_account_json(gcs_creds)
        if gcp_creds and os.path.isfile(gcp_creds):
            logger.info(f"GCS auth: GOOGLE_APPLICATION_CREDENTIALS → {gcp_creds}")
            return gcs.Client.from_service_account_json(gcp_creds)
        logger.info("GCS auth: Application Default Credentials (ADC)")
        return gcs.Client()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _blob_name(self, kind: str, key: str) -> str:
        """Constructs the full GCS object path."""
        return f"{self.prefix}{kind}/{key}"

    # ── Storage interface ─────────────────────────────────────────────────────

    def save(self, kind: str, key: str, data: bytes) -> str:
        blob_name = self._blob_name(kind, key)
        blob = self.bucket.blob(blob_name)
        # Infer a sensible content type from the key extension
        content_type = _guess_content_type(key)
        blob.upload_from_string(data, content_type=content_type)
        gcs_uri = f"gs://{self.bucket.name}/{blob_name}"
        logger.info(f"GCS upload: {gcs_uri}")
        return gcs_uri

    def open_read(self, kind: str, key: str) -> BinaryIO:
        blob = self.bucket.blob(self._blob_name(kind, key))
        return io.BytesIO(blob.download_as_bytes())

    def local_path(self, kind: str, key: str) -> Path:
        """Download GCS object to a local temp file and return its path."""
        blob = self.bucket.blob(self._blob_name(kind, key))
        tmp = Path(tempfile.gettempdir()) / "pf_cache" / kind
        tmp.mkdir(parents=True, exist_ok=True)
        target = tmp / key
        blob.download_to_filename(str(target))
        return target

    def url(self, kind: str, key: str) -> str:
        """Generate a v4 signed URL (15 min) so the browser can download directly."""
        from datetime import timedelta

        blob = self.bucket.blob(self._blob_name(kind, key))
        return blob.generate_signed_url(
            expiration=timedelta(minutes=15), version="v4"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

_CONTENT_TYPES: dict[str, str] = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt":  "text/plain",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".dcm":  "application/dicom",
    ".json": "application/json",
}


def _guess_content_type(key: str) -> str:
    ext = Path(key).suffix.lower()
    return _CONTENT_TYPES.get(ext, "application/octet-stream")


# ── Factory ───────────────────────────────────────────────────────────────────

def get_storage() -> Storage:
    """
    Instantiate the configured storage backend.

    ENV vars:
        STORAGE_BACKEND          local | gcs        (default: local)
        GCS_BUCKET               GCS bucket name    (default: tanuh-bcd-bucket)
        GCS_PREFIX               folder prefix      (default: privacy-app)
        GCS_CREDENTIALS_JSON     path to GCS SA JSON (priority 1)
        GOOGLE_APPLICATION_CREDENTIALS  path to SA JSON (priority 2)
        LOCAL_DATA_DIR           root for local backend (default: ./data)
    """
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend == "gcs":
        bucket = os.getenv("GCS_BUCKET", "tanuh-bcd-bucket")
        prefix = os.getenv("GCS_PREFIX", "privacy-app")
        logger.info(
            f"Storage backend: GCS  bucket={bucket}  prefix={prefix}"
        )
        return GCSStorage(bucket=bucket, prefix=prefix)

    logger.info("Storage backend: local")
    return LocalStorage(root=os.getenv("LOCAL_DATA_DIR", "./data"))
