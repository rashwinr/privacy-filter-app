"""Plain-text extractor + redactor."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.extractors.txt import extract_text, redact


def test_extract_text_roundtrips_utf8(tmp_path: Path):
    p = tmp_path / "in.txt"
    body = "Café — naïve façade. 你好.\nLine 2."
    p.write_text(body, encoding="utf-8")
    assert extract_text(p) == body


def test_redact_replaces_spans_with_label(tmp_path: Path, sample_text: str):
    src = tmp_path / "src.txt"
    src.write_text(sample_text, encoding="utf-8")
    out = tmp_path / "out.txt"

    entities = [
        {"entity_group": "private_person", "word": "Alice Smith",
         "start": sample_text.index("Alice Smith"),
         "end":   sample_text.index("Alice Smith") + len("Alice Smith"),
         "score": 0.99},
        {"entity_group": "private_email", "word": "alice@example.com",
         "start": sample_text.index("alice@example.com"),
         "end":   sample_text.index("alice@example.com") + len("alice@example.com"),
         "score": 0.99},
    ]
    redact(src, entities, out)
    redacted = out.read_text(encoding="utf-8")
    assert "Alice Smith" not in redacted
    assert "alice@example.com" not in redacted
    assert "[REDACTED:PRIVATE_PERSON]" in redacted
    assert "[REDACTED:PRIVATE_EMAIL]" in redacted


def test_redact_handles_overlapping_in_reverse_order(tmp_path: Path):
    """Spans must be applied right-to-left so earlier offsets stay valid."""
    text = "AAA BBB CCC"
    src = tmp_path / "src.txt"
    src.write_text(text)
    out = tmp_path / "out.txt"

    entities = [
        {"entity_group": "X", "word": "AAA", "start": 0, "end": 3, "score": 1.0},
        {"entity_group": "Y", "word": "BBB", "start": 4, "end": 7, "score": 1.0},
        {"entity_group": "Z", "word": "CCC", "start": 8, "end": 11, "score": 1.0},
    ]
    redact(src, entities, out)
    result = out.read_text()
    assert result == "[REDACTED:X] [REDACTED:Y] [REDACTED:Z]"


def test_redact_with_empty_entities_is_passthrough(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("nothing to see here")
    out = tmp_path / "out.txt"
    redact(src, [], out)
    assert out.read_text() == "nothing to see here"


def test_redact_skips_entities_missing_offsets(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("hello world")
    out = tmp_path / "out.txt"
    redact(src, [{"entity_group": "X", "word": "hello", "score": 1.0}], out)
    # Without start/end the redactor must leave content unchanged.
    assert out.read_text() == "hello world"


def test_end_to_end_with_fake_filter(make_txt, fake_filter, sample_text):
    """Use the FakePrivacyFilter just like production code would."""
    src: Path = make_txt("note.txt", sample_text)
    out = src.with_name("note.redacted.txt")

    entities = fake_filter.detect(sample_text)
    assert any(e["entity_group"] == "private_person" for e in entities)
    redact(src, entities, out)

    redacted = out.read_text()
    # Every detected substring should be gone from the output.
    for needle, *_ in fake_filter.rules:
        if needle in sample_text:
            assert needle not in redacted, f"{needle!r} leaked through"
