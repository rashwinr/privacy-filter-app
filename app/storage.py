"""Storage abstraction: local filesystem now, GCS-ready.

Switch via STORAGE_BACKEND env var: "local" (default) or "gcs".
The local backend writes to ./data/{uploads,redacted}/{job_id}__{name}.
The GCS backend writes to gs://$GCS_BUCKET/$GCS_PREFIX{uploads,redacted}/...
"""
from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Optional


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
    def __init__(self, bucket: str, prefix: str = "") -> None:
        from google.cloud import storage as gcs

        self.client = gcs.Client()
        self.bucket = self.client.bucket(bucket)
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""

    def _key(self, kind: str, key: str) -> str:
        return f"{self.prefix}{kind}/{key}"

    def save(self, kind: str, key: str, data: bytes) -> str:
        blob = self.bucket.blob(self._key(kind, key))
        blob.upload_from_string(data)
        return f"gs://{self.bucket.name}/{self._key(kind, key)}"

    def open_read(self, kind: str, key: str):
        import io

        blob = self.bucket.blob(self._key(kind, key))
        return io.BytesIO(blob.download_as_bytes())

    def local_path(self, kind: str, key: str) -> Path:
        import tempfile

        blob = self.bucket.blob(self._key(kind, key))
        tmp = Path(tempfile.gettempdir()) / "pf_cache" / kind
        tmp.mkdir(parents=True, exist_ok=True)
        target = tmp / key
        blob.download_to_filename(str(target))
        return target

    def url(self, kind: str, key: str) -> str:
        # Generate a v4 signed URL (15 min) so the browser can download directly.
        from datetime import timedelta

        blob = self.bucket.blob(self._key(kind, key))
        return blob.generate_signed_url(expiration=timedelta(minutes=15), version="v4")


def get_storage() -> Storage:
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend == "gcs":
        bucket = os.getenv("GCS_BUCKET")
        if not bucket:
            raise RuntimeError("STORAGE_BACKEND=gcs but GCS_BUCKET is not set")
        return GCSStorage(bucket=bucket, prefix=os.getenv("GCS_PREFIX", ""))
    return LocalStorage(root=os.getenv("LOCAL_DATA_DIR", "./data"))
