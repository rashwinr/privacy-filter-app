"""Generic Named-Entity-Recognition model for ORG / LOC / PER / MISC.

The ``openai/privacy-filter`` model only knows 8 personal-PII categories and
explicitly does **not** detect organizations, companies, cities, states,
countries, or other named entities. To redact those generically (without
hand-curated keyword lists), we run a second, lightweight NER model in
parallel.

Default model: ``dslim/bert-base-NER`` -- ~110M params, CoNLL-2003 trained,
CPU-friendly, recognises:

* ``PER``  -> mapped to ``private_person``
* ``ORG``  -> mapped to ``org_name``
* ``LOC``  -> mapped to ``address_location``
* ``MISC`` -> mapped to ``misc_entity`` (off by default)

The model is loaded lazily on first ``detect()`` call so unit tests that
don't exercise inference don't pay the download cost.

Override the model with ``NER_MODEL`` env var (e.g. set to a multilingual
model for non-English documents).

Disable entirely with ``ENABLE_NER_MODEL=0``.
"""
from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# Default mapping from CoNLL-2003 tags to our internal label vocabulary.
_LABEL_MAP = {
    "PER": "private_person",
    "ORG": "org_name",
    "LOC": "address_location",
    "MISC": "misc_entity",
}

# By default we surface PER / ORG / LOC. MISC is too noisy (catches
# "English", "Tax Invoice", etc.) so it's opt-in via env.
_DEFAULT_KEEP = {"PER", "ORG", "LOC"}


class NERModel:
    """Singleton wrapper around a HuggingFace NER pipeline."""

    _instance: "NERModel | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        self.model_name = os.getenv("NER_MODEL", "dslim/bert-base-NER")
        self.device = os.getenv("MODEL_DEVICE", "cpu")
        self.aggregation = os.getenv("NER_AGGREGATION", "simple")
        self._pipe = None
        self._loaded = False
        self._load_failed = False

    @classmethod
    def instance(cls) -> "NERModel":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Test helper: drop the singleton so env-var changes take effect."""
        with cls._lock:
            cls._instance = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded or self._load_failed:
            return
        from transformers import pipeline  # lazy import

        device_arg = -1 if self.device == "cpu" else 0
        try:
            logger.info("Loading NER model %s on %s", self.model_name, self.device)
            self._pipe = pipeline(
                task="token-classification",
                model=self.model_name,
                aggregation_strategy=self.aggregation,
                device=device_arg,
            )
            self._loaded = True
            logger.info("NER model loaded.")
        except Exception:
            # NER is auxiliary -- don't crash the app if it fails to load.
            logger.exception("Failed to load NER model; continuing without it")
            self._load_failed = True

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """Return entities in the same shape as the privacy-filter model."""
        if os.getenv("ENABLE_NER_MODEL", "1") != "1":
            return []
        if not text or not text.strip():
            return []
        if not self._loaded:
            self.load()
        if self._load_failed or self._pipe is None:
            return []

        keep_raw = os.getenv("NER_KEEP_LABELS", "")
        keep = (
            {x.strip().upper() for x in keep_raw.split(",") if x.strip()}
            if keep_raw
            else set(_DEFAULT_KEEP)
        )

        # bert-base-NER has a 512-token cap. Chunk by lines to stay well below
        # that and to keep per-page bookkeeping easy.
        MAX_CHARS = 1500
        out: List[Dict[str, Any]] = []
        offset = 0
        for chunk in _chunk_by_lines(text, MAX_CHARS):
            try:
                raw = self._pipe(chunk)
            except Exception:
                logger.exception("NER inference failed on a chunk; skipping")
                offset += len(chunk)
                continue
            for e in raw:
                tag = (e.get("entity_group") or e.get("entity") or "").upper()
                # Strip BIO prefix if present (e.g. 'B-ORG' -> 'ORG')
                if tag.startswith(("B-", "I-", "S-", "E-")):
                    tag = tag[2:]
                if tag not in keep:
                    continue
                label = _LABEL_MAP.get(tag, f"ner_{tag.lower()}")
                start = e.get("start")
                end = e.get("end")
                if start is None or end is None:
                    continue
                out.append({
                    "entity_group": label,
                    "score": float(e.get("score", 0.0)),
                    "word": e.get("word", "") or chunk[int(start):int(end)],
                    "start": int(start) + offset,
                    "end": int(end) + offset,
                    "_source": "ner",
                })
            offset += len(chunk)
        return out


def _chunk_by_lines(text: str, max_chars: int) -> List[str]:
    """Split text into <= max_chars chunks at line boundaries."""
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > max_chars and buf:
            chunks.append("".join(buf))
            buf, size = [line], len(line)
        else:
            buf.append(line)
            size += len(line)
    if buf:
        chunks.append("".join(buf))
    return chunks
