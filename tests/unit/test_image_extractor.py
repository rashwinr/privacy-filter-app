"""Image extractor + redactor (Tesseract OCR)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
pytest.importorskip("pytesseract")

if shutil.which("tesseract") is None:
    pytest.skip("tesseract binary not available", allow_module_level=True)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app.extractors.image import extract_text, redact  # noqa: E402


def _make_image_with_text(path: Path, text: str, size=(800, 200)) -> Path:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 60), text, fill="black", font=font)
    img.save(path)
    return path


def test_ocr_extracts_text(tmp_path: Path):
    p = _make_image_with_text(tmp_path / "a.png", "Hello Alice")
    out = extract_text(p)
    # OCR is approximate — assert each word is present.
    assert "Hello" in out
    assert "Alice" in out


def test_redact_draws_black_box_over_pii(tmp_path: Path):
    """We can't easily assert the pixels, but the output file must exist
    and remain a valid image of the same size."""
    p = _make_image_with_text(tmp_path / "a.png", "Patient Alice Smith here")
    out = tmp_path / "redacted.png"

    text = extract_text(p)
    # Build a synthetic span over the OCR-recognised "Alice" word.
    if "Alice" not in text:
        pytest.skip("OCR did not recognise the test word; environment-dependent")
    start = text.index("Alice")
    entities = [{
        "entity_group": "private_person",
        "score": 0.99,
        "word": "Alice",
        "start": start,
        "end": start + len("Alice"),
    }]
    redact(p, entities, out)

    assert out.exists()
    redacted_img = Image.open(out)
    assert redacted_img.size == Image.open(p).size

    # After redaction, OCR should no longer recover "Alice" cleanly.
    after = extract_text(out)
    assert "Alice" not in after or "PRIVATE_PERSON" in after
