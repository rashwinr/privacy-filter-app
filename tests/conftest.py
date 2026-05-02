"""Shared fixtures for the privacy-filter test suite.

The most important thing this file does is install a *fake* PrivacyFilter
singleton so no test ever downloads the real 1.5B-parameter checkpoint.
Tests that explicitly need the real model can opt in via the
`requires_model` marker and skip themselves when weights are missing.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Dict, Any

import pytest

# Make the project root importable as `app` regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------

@dataclass
class FakePrivacyFilter:
    """Drop-in replacement for app.model.PrivacyFilter.

    Instead of running a transformer, it scans the input for a configurable
    set of (substring, label) pairs and returns char-offset spans the same
    shape as the real pipeline output.
    """
    model_name: str = "fake/privacy-filter"
    device: str = "cpu"
    aggregation: str = "simple"
    _loaded: bool = False
    rules: List[tuple] = field(
        default_factory=lambda: [
            # (substring, label, score)
            ("Alice Smith", "private_person", 0.999),
            ("Harry Potter", "private_person", 0.999),
            ("Bob Jones", "private_person", 0.998),
            ("alice@example.com", "private_email", 0.999),
            ("harry.potter@hogwarts.edu", "private_email", 0.999),
            ("555-1234", "private_phone", 0.97),
            ("555-867-5309", "private_phone", 0.98),
            ("221B Baker Street", "private_address", 0.95),
            ("https://secret.example.com", "private_url", 0.93),
            ("1990-01-15", "private_date", 0.94),
            ("ACCT-4242-4242", "account_number", 0.96),
            ("sk-test-DEADBEEF", "secret", 0.99),
        ]
    )

    def load(self) -> None:
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def detect(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        if not self._loaded:
            self.load()
        out: List[Dict[str, Any]] = []
        for needle, label, score in self.rules:
            start = 0
            while True:
                idx = text.find(needle, start)
                if idx == -1:
                    break
                out.append({
                    "entity_group": label,
                    "score": score,
                    "word": needle,
                    "start": idx,
                    "end": idx + len(needle),
                })
                start = idx + len(needle)
        # Sort by start so downstream code that depends on order is stable.
        out.sort(key=lambda e: e["start"])
        return out


@pytest.fixture
def fake_filter() -> FakePrivacyFilter:
    return FakePrivacyFilter()


@dataclass
class FakeNERModel:
    """Drop-in replacement for app.ner_model.NERModel.

    Recognises a configurable list of (substring, label, score) triples,
    same approach as FakePrivacyFilter but emitting org_name /
    address_location / private_person labels.
    """
    _loaded: bool = False
    # The real bert-base-NER pipeline returns CoNLL-2003 tags (PER/ORG/LOC/MISC)
    # in entity_group; ner_model.detect() then maps those to our internal
    # vocabulary. The fake mirrors that contract.
    rules: List[tuple] = field(
        default_factory=lambda: [
            ("Shanmuga Hospital Ltd", "ORG", 0.99),
            ("PUNJAB NATIONAL BANK", "ORG", 0.99),
            ("Punjab National Bank", "ORG", 0.99),
            ("Indian Institute of Science", "ORG", 0.98),
            ("TANUH", "ORG", 0.95),
            ("Acme Industries Ltd", "ORG", 0.97),
            ("Bangalore", "LOC", 0.97),
            ("Salem", "LOC", 0.97),
            ("Tamil Nadu", "LOC", 0.97),
            ("Karnataka", "LOC", 0.97),
            ("India", "LOC", 0.95),
        ]
    )

    def load(self) -> None:
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def detect(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        if not self._loaded:
            self.load()
        out: List[Dict[str, Any]] = []
        for needle, label, score in self.rules:
            start = 0
            while True:
                idx = text.find(needle, start)
                if idx == -1:
                    break
                out.append({
                    "entity_group": label,
                    "score": score,
                    "word": needle,
                    "start": idx,
                    "end": idx + len(needle),
                    "_source": "ner",
                })
                start = idx + len(needle)
        out.sort(key=lambda e: e["start"])
        return out


def _make_fake_pipe(rules):
    """Build a callable that mimics a HuggingFace token-classification pipeline.

    For every (substring, label, score) rule it finds the substring in the
    input and emits a span dict with char offsets and aggregation keys the
    real pipeline would produce under aggregation_strategy='simple'.
    """
    def _pipe(text):
        out = []
        for needle, label, score in rules:
            start = 0
            while True:
                idx = text.find(needle, start)
                if idx == -1:
                    break
                out.append({
                    "entity_group": label,
                    "score": score,
                    "word": needle,
                    "start": idx,
                    "end": idx + len(needle),
                })
                start = idx + len(needle)
        return out
    return _pipe


@pytest.fixture(autouse=True)
def _patch_model_singleton(monkeypatch, request):
    """Patch PrivacyFilter and NERModel internal pipelines with fakes.

    Patching ``_pipe`` (rather than replacing the whole singleton) keeps the
    real ``detect()`` orchestration logic in play -- so tests exercise the
    same merging / chunking / overlap-resolution code that runs in prod.

    Disable for tests marked ``requires_model`` (real-checkpoint tests).
    """
    if request.node.get_closest_marker("requires_model"):
        yield
        return

    from app import model as model_mod
    from app import ner_model as ner_mod

    pf = model_mod.PrivacyFilter.instance()
    pf._pipe = _make_fake_pipe(FakePrivacyFilter().rules)
    pf._loaded = True

    ner = ner_mod.NERModel.instance()
    ner._pipe = _make_fake_pipe(FakeNERModel().rules)
    ner._loaded = True
    ner._load_failed = False

    yield

    # Teardown: clear the singletons so the next test gets a fresh one.
    pf._pipe = None
    pf._loaded = False
    ner._pipe = None
    ner._loaded = False


# ---------------------------------------------------------------------------
# Filesystem fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch) -> Path:
    """Isolate ./data per test by pointing LOCAL_DATA_DIR at tmp_path."""
    data = tmp_path / "data"
    (data / "uploads").mkdir(parents=True)
    (data / "redacted").mkdir(parents=True)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(data))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    return data


@pytest.fixture
def sample_text() -> str:
    return (
        "Patient note:\n"
        "Hi, my name is Alice Smith and my email is alice@example.com.\n"
        "My phone is 555-1234 and I live at 221B Baker Street.\n"
        "Account ACCT-4242-4242, DOB 1990-01-15.\n"
        "Visit https://secret.example.com for the portal.\n"
        "API key: sk-test-DEADBEEF.\n"
    )


@pytest.fixture
def make_txt(tmp_path: Path) -> Callable[[str, str], Path]:
    def _make(name: str, content: str) -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p
    return _make


# ---------------------------------------------------------------------------
# Optional-dependency markers
# ---------------------------------------------------------------------------

def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def has_pymupdf() -> bool:
    return _module_available("fitz")


@pytest.fixture(scope="session")
def has_pydicom() -> bool:
    return _module_available("pydicom")


@pytest.fixture(scope="session")
def has_docx() -> bool:
    return _module_available("docx")


@pytest.fixture(scope="session")
def has_pil() -> bool:
    return _module_available("PIL")


@pytest.fixture(scope="session")
def has_tesseract() -> bool:
    return shutil.which("tesseract") is not None


@pytest.fixture(scope="session")
def has_fastapi() -> bool:
    return _module_available("fastapi")
