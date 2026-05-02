"""Storage abstraction: LocalStorage works on disk; GCS path is mocked."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.storage import LocalStorage, get_storage


def test_local_storage_save_and_read(tmp_path: Path):
    s = LocalStorage(root=str(tmp_path))
    s.save("uploads", "abc__hello.txt", b"hello")
    assert (tmp_path / "uploads" / "abc__hello.txt").read_bytes() == b"hello"
    with s.open_read("uploads", "abc__hello.txt") as f:
        assert f.read() == b"hello"


def test_local_storage_url_format(tmp_path: Path):
    s = LocalStorage(root=str(tmp_path))
    assert s.url("redacted", "x.txt") == "/api/files/redacted/x.txt"


def test_local_storage_creates_dirs(tmp_path: Path):
    s = LocalStorage(root=str(tmp_path / "nested" / "deep"))
    assert (tmp_path / "nested" / "deep" / "uploads").is_dir()
    assert (tmp_path / "nested" / "deep" / "redacted").is_dir()


def test_local_storage_rejects_unknown_kind(tmp_path: Path):
    s = LocalStorage(root=str(tmp_path))
    with pytest.raises(AssertionError):
        s.save("trash", "x", b"")


def test_get_storage_defaults_to_local(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    s = get_storage()
    assert isinstance(s, LocalStorage)


def test_get_storage_gcs_requires_bucket(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "gcs")
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    with pytest.raises(RuntimeError, match="GCS_BUCKET"):
        get_storage()


def test_gcs_storage_save_uses_blob_upload(monkeypatch):
    """Mock google.cloud.storage so we don't need real GCP credentials."""
    fake_blob = MagicMock()
    fake_blob.generate_signed_url.return_value = "https://signed.example/blob"
    fake_bucket = MagicMock()
    fake_bucket.blob.return_value = fake_blob
    fake_bucket.name = "test-bucket"
    fake_client = MagicMock()
    fake_client.bucket.return_value = fake_bucket

    fake_storage_module = types.ModuleType("google.cloud.storage")
    fake_storage_module.Client = MagicMock(return_value=fake_client)

    # Build a fake `google.cloud` package tree.
    google_pkg = types.ModuleType("google")
    cloud_pkg = types.ModuleType("google.cloud")
    cloud_pkg.storage = fake_storage_module
    google_pkg.cloud = cloud_pkg

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_pkg)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake_storage_module)

    # Re-import storage so GCSStorage picks up the patched module.
    import app.storage as storage_mod
    importlib.reload(storage_mod)

    s = storage_mod.GCSStorage(bucket="test-bucket", prefix="pf/")
    uri = s.save("uploads", "abc.txt", b"data")
    assert uri == "gs://test-bucket/pf/uploads/abc.txt"
    fake_bucket.blob.assert_called_with("pf/uploads/abc.txt")
    fake_blob.upload_from_string.assert_called_once_with(b"data")

    url = s.url("redacted", "x.txt")
    assert url == "https://signed.example/blob"
