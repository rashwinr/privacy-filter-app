"""Image extractor + redactor (OCR-based).

Pipeline:
  1. Run Tesseract OCR with detailed output (per-word bounding boxes).
  2. Reconstruct the full text *along with* a map from char-offset -> bbox.
  3. After detection, resolve each span's char range back to one or more
     bboxes and draw black rectangles with the entity label burned on top.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple

from PIL import Image, ImageDraw, ImageFont
import pytesseract


def _ocr_with_boxes(image: Image.Image) -> Tuple[str, List[Tuple[int, int, Tuple[int, int, int, int]]]]:
    """Return (full_text, [(char_start, char_end, (x, y, w, h)), ...])."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    text_parts: List[str] = []
    spans: List[Tuple[int, int, Tuple[int, int, int, int]]] = []
    cursor = 0
    n = len(data["text"])
    for i in range(n):
        word = data["text"][i]
        if not word or not word.strip():
            continue
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(word)
        cursor += len(word)
        bbox = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        spans.append((start, cursor, bbox))
    return "".join(text_parts), spans


def extract_text(path: Path) -> str:
    img = Image.open(path)
    text, _ = _ocr_with_boxes(img)
    return text


def extract_text_and_boxes(path: Path):
    img = Image.open(path)
    text, spans = _ocr_with_boxes(img)
    return img, text, spans


def redact(path: Path, entities: List[Dict[str, Any]], out_path: Path) -> None:
    img, _text, ocr_spans = extract_text_and_boxes(path)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for e in entities:
        s, t = e.get("start"), e.get("end")
        if s is None or t is None:
            continue
        label = e.get("entity_group", "PII").upper()
        # Find every OCR word whose char-range overlaps the entity range.
        for ws, we, (x, y, w, h) in ocr_spans:
            if we <= s or ws >= t:
                continue
            draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0))
            if font is not None:
                draw.text((x + 2, y + 2), label, fill=(255, 255, 255), font=font)

    img.save(out_path)
