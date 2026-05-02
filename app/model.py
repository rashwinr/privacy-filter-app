"""Privacy-filter model singleton.

Loads `openai/privacy-filter` once at startup and exposes a `detect(text)`
method that returns a list of entity dicts.
"""
from __future__ import annotations

import logging
import os
from threading import Lock
from typing import List, Dict, Any

from app import rule_detectors
from app.ner_model import NERModel

logger = logging.getLogger(__name__)


class PrivacyFilter:
    _instance: "PrivacyFilter | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        self.model_name = os.getenv("MODEL_NAME", "openai/privacy-filter")
        self.device = os.getenv("MODEL_DEVICE", "cpu")
        self.aggregation = os.getenv("MODEL_AGGREGATION", "simple")
        self._pipe = None
        self._loaded = False

    @classmethod
    def instance(cls) -> "PrivacyFilter":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Load the privacy-filter model + auxiliary NER model.

        Both are loaded eagerly at startup so cold-start cost is paid once.
        The NER model is best-effort -- if it fails to load, the privacy
        filter still works.
        """
        # Best-effort: warm up the auxiliary NER model in parallel.
        try:
            NERModel.instance().load()
        except Exception:
            logger.exception("NER auxiliary model failed to load")
        if self._loaded:
            return
        # Lazy import keeps `import app.model` cheap for tests.
        from transformers import pipeline

        logger.info("Loading privacy-filter model: %s on %s", self.model_name, self.device)
        device_arg = -1 if self.device == "cpu" else 0
        # Some torch builds: device="mps"/"cuda" string also works via device_map.
        try:
            self._pipe = pipeline(
                task="token-classification",
                model=self.model_name,
                aggregation_strategy=self.aggregation,
                device=device_arg,
            )
        except Exception:
            # Fall back to CPU if requested device unavailable
            logger.exception("Falling back to CPU for model load")
            self._pipe = pipeline(
                task="token-classification",
                model=self.model_name,
                aggregation_strategy=self.aggregation,
                device=-1,
            )
            self.device = "cpu"
        self._loaded = True
        logger.info("Privacy-filter model loaded.")

    @property
    def loaded(self) -> bool:
        return self._loaded

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """Return aggregated PII spans for the given text.

        Each item: {entity_group, score, word, start, end}
        Token-classification pipeline already returns char offsets when
        `aggregation_strategy="simple"` is set.
        """
        if not self._loaded:
            self.load()
        if not text or not text.strip():
            return []
        # Model has 128k context; for very long inputs we still chunk
        # conservatively to avoid memory spikes on Cloud Run.
        MAX_CHARS = 60000
        if len(text) <= MAX_CHARS:
            raw = self._pipe(text)
            ml_entities = [self._normalize(e) for e in raw]
        else:
            # Chunk by paragraphs preserving offsets
            ml_entities = []
            offset = 0
            for chunk in _chunk_text(text, MAX_CHARS):
                raw = self._pipe(chunk)
                for e in raw:
                    e_norm = self._normalize(e)
                    if e_norm.get("start") is not None:
                        e_norm["start"] += offset
                        e_norm["end"] += offset
                    ml_entities.append(e_norm)
                offset += len(chunk)

        # Layer in the auxiliary generic-NER model (organisations, locations,
        # people) -- catches things the privacy-filter model wasn't trained
        # on, like company names, cities, states, countries.
        try:
            ner_entities = NERModel.instance().detect(text)
        except Exception:
            logger.exception("NER auxiliary detection failed; continuing")
            ner_entities = []

        # And deterministic regex detectors for structured identifiers
        # (tax IDs, IFSC, Aadhaar, SSN, EIN, IBAN, etc.).
        rule_entities = rule_detectors.detect(text)

        # Merge in priority order: rules > NER > privacy-filter on overlap.
        # Rules are deterministic, NER is generic semantic, and the
        # privacy-filter model is the most specialised.
        combined = _resolve_overlaps(
            _resolve_overlaps(ml_entities, ner_entities),
            rule_entities,
        )
        return _merge_adjacent(combined)

    @staticmethod
    def _normalize(e: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entity_group": e.get("entity_group") or e.get("entity"),
            "score": float(e.get("score", 0.0)),
            "word": e.get("word", ""),
            "start": int(e["start"]) if "start" in e and e["start"] is not None else None,
            "end": int(e["end"]) if "end" in e and e["end"] is not None else None,
        }


def _resolve_overlaps(
    lower: List[Dict[str, Any]],
    higher: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge two entity lists, dropping spans fully contained in the other.

    ``higher`` wins on overlap: a span in ``lower`` that is fully covered by
    one in ``higher`` is dropped. The reverse is also true -- if a span in
    ``higher`` is *strictly* covered by an equal-or-larger span in
    ``higher`` itself we'd keep both (different label = different reason to
    redact). Same-shape duplicates (same start/end/label) are deduped.
    """
    if not higher:
        return _dedupe(lower)
    higher_ranges = [
        (int(r["start"]), int(r["end"]))
        for r in higher
        if r.get("start") is not None and r.get("end") is not None
    ]
    out: List[Dict[str, Any]] = []
    for e in lower:
        s, en = e.get("start"), e.get("end")
        if s is None or en is None:
            out.append(e)
            continue
        # Drop only if there is a STRICTLY larger covering span in 'higher'.
        # (An identical span is handled by _dedupe below.)
        covered = any(
            rs <= int(s) and int(en) <= re_ and (rs, re_) != (int(s), int(en))
            for rs, re_ in higher_ranges
        )
        if not covered:
            out.append(e)
    out.extend(higher)
    out.sort(
        key=lambda e: (
            int(e["start"]) if e.get("start") is not None else 0,
            int(e["end"]) if e.get("end") is not None else 0,
        )
    )
    return _dedupe(out)


def _dedupe(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove identical (start, end, label) duplicates and shorter spans
    contained inside a longer same-label span.

    Without this, multiple regex detectors that fire on the same text
    (e.g. ``_LONG_ACCOUNT`` and ``_CC_LIKE``) would produce two spans for
    the same account number, and ``_merge_adjacent`` would concatenate
    them into garbage like ``"NNNNNN-NNNNNN"``.
    """
    seen: set[tuple[int, int, str]] = set()
    unique: List[Dict[str, Any]] = []
    for e in entities:
        s = e.get("start")
        en = e.get("end")
        lbl = e.get("entity_group")
        if s is None or en is None or lbl is None:
            unique.append(e)
            continue
        key = (int(s), int(en), str(lbl))
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)
    # Now drop shorter spans contained inside a longer span of the SAME label.
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for e in unique:
        if e.get("start") is None or e.get("end") is None:
            continue
        by_label.setdefault(str(e["entity_group"]), []).append(e)
    drop_ids: set[int] = set()
    for spans in by_label.values():
        spans.sort(key=lambda x: (int(x["end"]) - int(x["start"])), reverse=True)
        kept: List[Dict[str, Any]] = []
        for sp in spans:
            s_s, s_e = int(sp["start"]), int(sp["end"])
            if any(
                int(k["start"]) <= s_s and s_e <= int(k["end"]) and (k is not sp)
                for k in kept
            ):
                drop_ids.add(id(sp))
            else:
                kept.append(sp)
    return [e for e in unique if id(e) not in drop_ids]


def _merge_adjacent(
    entities: List[Dict[str, Any]],
    max_gap: int = 2,
) -> List[Dict[str, Any]]:
    """Merge consecutive same-label spans separated by <= max_gap chars.

    The token-classification pipeline can return subword fragments (e.g.
    ``John`` + ``Doe`` as two `private_person` spans, or ``1985-03-`` + ``15``
    as two `private_date` spans). Merging produces one clean span per real
    entity so downstream redaction inserts a single tag.
    """
    spans = [
        e for e in entities
        if e.get("start") is not None and e.get("end") is not None
    ]
    spans.sort(key=lambda e: (int(e["start"]), int(e["end"])))

    merged: List[Dict[str, Any]] = []
    for e in spans:
        if (
            merged
            and merged[-1]["entity_group"] == e["entity_group"]
            and int(e["start"]) - int(merged[-1]["end"]) <= max_gap
        ):
            prev = merged[-1]
            prev["end"] = max(int(prev["end"]), int(e["end"]))
            # word concatenation is best-effort; downstream code uses offsets.
            prev["word"] = (prev.get("word", "") + e.get("word", "")).strip()
            # Keep the lowest score so we don't overstate confidence.
            prev["score"] = min(float(prev["score"]), float(e["score"]))
        else:
            merged.append(dict(e))

    # Append entities without offsets (rare) at the end, unmerged.
    merged.extend(
        e for e in entities
        if e.get("start") is None or e.get("end") is None
    )
    return merged


def _chunk_text(text: str, max_chars: int) -> List[str]:
    """Split text into <= max_chars chunks at paragraph/line boundaries."""
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
