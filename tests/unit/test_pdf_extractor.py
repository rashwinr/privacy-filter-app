"""PDF extractor + redactor (PyMuPDF)."""
from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")  # PyMuPDF
PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app.extractors.pdf import (  # noqa: E402
    extract_text,
    has_text_layer,
    redact,
)

_HAS_TESSERACT = shutil.which("tesseract") is not None


def _make_pdf(path: Path, lines: list[str]) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=12)
        y += 18
    doc.save(str(path))
    doc.close()
    return path


def test_extract_text_reads_inserted_lines(tmp_path: Path):
    pdf = _make_pdf(tmp_path / "doc.pdf", [
        "Patient: Alice Smith",
        "Email: alice@example.com",
    ])
    text = extract_text(pdf)
    assert "Alice Smith" in text
    assert "alice@example.com" in text


def test_has_text_layer_true_for_native_pdf(tmp_path: Path):
    pdf = _make_pdf(tmp_path / "doc.pdf", ["hello world"])
    assert has_text_layer(pdf) is True


def test_redact_removes_pii_from_text_layer(tmp_path: Path, fake_filter):
    pdf = _make_pdf(tmp_path / "doc.pdf", [
        "Patient: Alice Smith",
        "Contact alice@example.com",
        "Notes: nothing sensitive here.",
    ])
    out = tmp_path / "redacted.pdf"

    text = extract_text(pdf)
    entities = fake_filter.detect(text)
    redact(pdf, entities, out)

    redacted_text = extract_text(out)
    assert "Alice Smith" not in redacted_text
    assert "alice@example.com" not in redacted_text
    # The label should appear as the replacement annotation text.
    assert "PRIVATE_PERSON" in redacted_text or "PRIVATE_EMAIL" in redacted_text
    # Untouched line still readable.
    assert "nothing sensitive here" in redacted_text


def test_redact_with_no_entities_is_passthrough(tmp_path: Path):
    pdf = _make_pdf(tmp_path / "doc.pdf", ["plain document"])
    out = tmp_path / "redacted.pdf"
    redact(pdf, [], out)
    assert "plain document" in extract_text(out)


# ---------------------------------------------------------------------------
# OCR fallback for scanned (image-only) PDFs
# ---------------------------------------------------------------------------

def _make_scanned_pdf(path: Path, lines: list[str], font_size: int = 32) -> Path:
    """Create an image-only PDF (no embedded text layer).

    We render the lines onto a PIL image, then embed that image as the only
    content of a new PDF page. PyMuPDF's get_text() returns "" for these
    pages, so this exercises the OCR fallback path.
    """
    # Render text to an image. Use the default PIL font so the test does not
    # depend on platform-specific TrueType fonts.
    width, height = 1200, 1600
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=font_size)  # PIL >= 10
    except TypeError:
        font = ImageFont.load_default()
    y = 80
    for line in lines:
        draw.text((80, y), line, fill="black", font=font)
        y += font_size + 24
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Create a PDF whose only content is that image.
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(str(path))
    doc.close()
    return path


@pytest.mark.skipif(not _HAS_TESSERACT, reason="tesseract binary not installed")
def test_extract_text_ocrs_scanned_pdf(tmp_path: Path):
    """A PDF whose only content is an image of text should still yield text."""
    pdf = _make_scanned_pdf(tmp_path / "scan.pdf", [
        "Patient Alice Smith",
        "Email alice@example.com",
    ])
    # Sanity check: the PDF really has no embedded text layer.
    with fitz.open(pdf) as doc:
        assert (doc[0].get_text("text") or "").strip() == ""

    text = extract_text(pdf)
    # OCR is fuzzy, so check for a few stable substrings rather than exact match.
    lowered = text.lower()
    assert "alice" in lowered
    assert "smith" in lowered
    assert "@example.com" in lowered or "example.com" in lowered


@pytest.mark.skipif(not _HAS_TESSERACT, reason="tesseract binary not installed")
def test_redact_blacks_out_pii_on_scanned_pdf(tmp_path: Path, fake_filter):
    pdf = _make_scanned_pdf(tmp_path / "scan.pdf", [
        "Patient Alice Smith",
        "Notes: nothing sensitive here.",
    ])
    out = tmp_path / "scan-redacted.pdf"

    text = extract_text(pdf)
    entities = fake_filter.detect(text)
    # Pre-condition: the fake filter actually found Alice Smith via OCR text.
    assert any(e["entity_group"] == "private_person" for e in entities)

    redact(pdf, entities, out)

    # Re-OCR the redacted PDF — the PII text must no longer be readable.
    redacted_text = extract_text(out).lower()
    # Tesseract may misread heavy black bars as random chars, but the original
    # "alice smith" should not survive intact.
    assert "alice smith" not in redacted_text


@pytest.mark.skipif(not _HAS_TESSERACT, reason="tesseract binary not installed")
def test_mixed_pdf_text_page_plus_scanned_page(tmp_path: Path, fake_filter):
    """A PDF with one text page and one scanned page should redact both."""
    # Build the text page first.
    pdf = tmp_path / "mixed.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Bob Jones writes to alice@example.com", fontsize=14)

    # Append a scanned-style page.
    img = Image.new("RGB", (1200, 1600), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=32)
    except TypeError:
        font = ImageFont.load_default()
    d.text((80, 80), "Patient Alice Smith", fill="black", font=font)
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    p2 = doc.new_page(width=1200, height=1600)
    p2.insert_image(p2.rect, stream=buf.getvalue())
    doc.save(str(pdf)); doc.close()

    text = extract_text(pdf)
    assert "Bob Jones" in text
    assert "alice" in text.lower() and "smith" in text.lower()

    out = tmp_path / "mixed-redacted.pdf"
    redact(pdf, fake_filter.detect(text), out)

    redacted = extract_text(out).lower()
    assert "bob jones" not in redacted
    assert "alice smith" not in redacted
