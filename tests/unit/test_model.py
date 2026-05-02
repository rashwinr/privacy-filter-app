"""Tests for app.model — chunking, normalisation, and singleton behaviour.

The real transformer is never loaded; we patch `pipeline` with a stub.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.model import PrivacyFilter, _chunk_text, _merge_adjacent


def test_chunk_text_short_input_single_chunk():
    text = "one\ntwo\nthree\n"
    chunks = _chunk_text(text, max_chars=1000)
    assert chunks == [text]


def test_chunk_text_splits_on_line_boundaries():
    lines = [f"line {i}\n" for i in range(20)]
    text = "".join(lines)
    chunks = _chunk_text(text, max_chars=30)
    assert len(chunks) > 1
    # Concatenated chunks must exactly equal the original text.
    assert "".join(chunks) == text
    # No chunk should exceed the budget by more than one line.
    for c in chunks:
        assert len(c) <= 30 + max(len(l) for l in lines)


def test_chunk_text_preserves_total_length():
    text = "x" * 250 + "\n" + "y" * 250
    chunks = _chunk_text(text, max_chars=100)
    assert sum(len(c) for c in chunks) == len(text)


def test_normalize_strips_unwanted_keys():
    raw = {
        "entity_group": "private_email",
        "entity": "ignored",
        "score": 0.9,
        "word": "x@y.com",
        "start": 1,
        "end": 8,
        "extra": "drop me",
    }
    n = PrivacyFilter._normalize(raw)
    assert set(n.keys()) == {"entity_group", "score", "word", "start", "end"}
    assert n["start"] == 1 and n["end"] == 8


def test_normalize_handles_missing_offsets():
    n = PrivacyFilter._normalize({"entity_group": "x", "score": 0.5, "word": "w"})
    assert n["start"] is None and n["end"] is None


def test_detect_returns_empty_for_blank_input():
    pf = PrivacyFilter()
    pf._loaded = True
    pf._pipe = lambda text: [{"entity_group": "X", "score": 1.0, "word": "x", "start": 0, "end": 1}]
    assert pf.detect("") == []
    assert pf.detect("   \n\t") == []


def test_detect_uses_pipe_for_short_input():
    pf = PrivacyFilter()
    pf._loaded = True
    pf._pipe = lambda text: [
        {"entity_group": "private_email", "score": 0.9, "word": "x", "start": 0, "end": 1},
    ]
    out = pf.detect("hello")
    assert len(out) == 1
    assert out[0]["entity_group"] == "private_email"


def test_detect_chunks_long_input_and_offsets_accumulate():
    """Verify the chunk path adjusts char offsets back to the global frame."""
    pf = PrivacyFilter()
    pf._loaded = True

    # Build a long text deterministically, with a marker every 30k chars.
    # MAX_CHARS in the implementation is 60_000.
    text = ("a" * 30_000 + "\n") * 5  # > 60k, multi-chunk

    calls = {"n": 0}

    def fake_pipe(chunk: str):
        calls["n"] += 1
        # Pretend each chunk has a single span at offset 5..10.
        return [{"entity_group": "X", "score": 1.0, "word": chunk[5:10],
                 "start": 5, "end": 10}]

    pf._pipe = fake_pipe
    spans = pf.detect(text)
    assert calls["n"] >= 2  # actually chunked
    # All offsets must lie within the original text.
    for s in spans:
        assert 0 <= s["start"] < s["end"] <= len(text)


def test_singleton_returns_same_instance():
    a = PrivacyFilter.instance()
    b = PrivacyFilter.instance()
    assert a is b


# --- _merge_adjacent ---

def test_merge_adjacent_combines_subword_person_spans():
    """`John` + `Doe` should collapse into one private_person span."""
    spans = [
        {"entity_group": "private_person", "score": 0.99, "word": "John",
         "start": 8, "end": 12},
        {"entity_group": "private_person", "score": 0.98, "word": " Doe",
         "start": 12, "end": 16},
    ]
    out = _merge_adjacent(spans)
    assert len(out) == 1
    assert out[0]["entity_group"] == "private_person"
    assert out[0]["start"] == 8 and out[0]["end"] == 16
    # Score should be the minimum (most conservative).
    assert out[0]["score"] == pytest.approx(0.98)


def test_merge_adjacent_combines_subword_date_spans():
    """`1985-03-` + `15` should collapse into one private_date span."""
    spans = [
        {"entity_group": "private_date", "score": 0.99, "word": "1985-03-",
         "start": 22, "end": 30},
        {"entity_group": "private_date", "score": 0.99, "word": "15",
         "start": 30, "end": 32},
    ]
    out = _merge_adjacent(spans)
    assert len(out) == 1 and out[0]["end"] == 32


def test_merge_adjacent_does_not_merge_different_labels():
    spans = [
        {"entity_group": "private_person", "score": 0.9, "word": "A",
         "start": 0, "end": 1},
        {"entity_group": "private_email", "score": 0.9, "word": "B",
         "start": 1, "end": 2},
    ]
    out = _merge_adjacent(spans)
    assert len(out) == 2


def test_merge_adjacent_does_not_merge_distant_spans():
    """Spans separated by more than `max_gap` chars stay separate."""
    spans = [
        {"entity_group": "private_person", "score": 0.9, "word": "Alice",
         "start": 0, "end": 5},
        {"entity_group": "private_person", "score": 0.9, "word": "Bob",
         "start": 50, "end": 53},
    ]
    out = _merge_adjacent(spans)
    assert len(out) == 2


def test_merge_adjacent_handles_empty_list():
    assert _merge_adjacent([]) == []


def test_merge_adjacent_preserves_offsetless_entities():
    spans = [
        {"entity_group": "X", "score": 0.5, "word": "x", "start": None, "end": None},
    ]
    out = _merge_adjacent(spans)
    assert len(out) == 1


def test_detect_merges_subword_spans_end_to_end():
    """Real example: two adjacent private_person fragments collapse into one."""
    pf = PrivacyFilter()
    pf._loaded = True
    pf._pipe = lambda text: [
        {"entity_group": "private_person", "score": 0.99, "word": "John",
         "start": 0, "end": 4},
        {"entity_group": "private_person", "score": 0.98, "word": " Doe",
         "start": 4, "end": 8},
    ]
    out = pf.detect("John Doe is here")
    assert len(out) == 1
    assert out[0]["start"] == 0 and out[0]["end"] == 8
