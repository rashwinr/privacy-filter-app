"""PDF extractor + redactor with OCR fallback for scanned pages.

Strategy
--------
1. **Per-page hybrid extraction.** For each page we first try the embedded text
   layer via PyMuPDF. If a page returns little or no text (the page is a scan
   or an image of text), we render the page to a high-resolution bitmap and
   OCR it with Tesseract. The full document text is the concatenation of
   per-page text, and we keep a *page index* that records, for every page,
   how it was extracted (``"text"`` vs ``"ocr"``), the character offset where
   that page's text starts in the global string, and -- for OCR pages -- the
   list of (char_start, char_end, bbox_in_pdf_points) tuples.

2. **Per-page hybrid redaction.** For text pages we use ``search_for`` +
   ``add_redact_annot`` so the underlying text is truly removed. For OCR
   pages we resolve each detected entity's char-range back to its OCR word
   bboxes, draw black rectangles directly on the page, and write the entity
   label on top. The original page content is then covered by the bitmap
   redaction (``add_redact_annot`` with ``fill`` covers existing pixels).

This makes the redactor robust to image-only PDFs, mixed PDFs, and PDFs that
have a *partial* OCR layer.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

# A page is treated as "needs OCR" if its embedded text layer has fewer than
# this many non-whitespace characters. Empirically this catches scanned pages
# that contain stray digital artefacts (page numbers, watermarks, etc).
_MIN_TEXT_CHARS_PER_PAGE = 20

# Render scanned pages at this DPI before handing to Tesseract. 200 DPI is a
# good balance between text recognition quality (especially for small fonts
# and email addresses in headers/footers) and memory use. With 8 GiB Cloud
# Run instances and concurrency=1 this is comfortably within budget.
# Override via ``PRIVACY_FILTER_OCR_DPI``.
import os as _os
_OCR_RENDER_DPI = int(_os.getenv("PRIVACY_FILTER_OCR_DPI", "200"))

# Tesseract page-segmentation mode and engine. PSM 3 (default) handles mixed
# layout pages with mixed columns and blocks reasonably well. Override via
# ``PRIVACY_FILTER_TESSERACT_CONFIG`` if a corpus benefits from a different
# configuration.
_TESS_CONFIG = _os.getenv("PRIVACY_FILTER_TESSERACT_CONFIG", "--oem 1 --psm 3")


@dataclass
class _OcrWord:
    char_start: int        # offset within the global document text
    char_end: int
    bbox_pdf: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points


@dataclass
class _PageInfo:
    index: int             # 0-based page number
    mode: str              # "text" or "ocr"
    char_start: int        # global offset of this page's text
    char_end: int
    ocr_words: List[_OcrWord] = field(default_factory=list)


def _render_page_to_pil(page: "fitz.Page", dpi: int = _OCR_RENDER_DPI) -> Tuple[Image.Image, float]:
    """Render a PDF page to a PIL Image. Returns (image, scale_factor).

    scale_factor converts pixel coordinates back to PDF points
    (1 PDF point = 1/72 inch).
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img, zoom


def _ocr_page(
    page: "fitz.Page",
    char_offset: int,
) -> Tuple[str, List[_OcrWord]]:
    """OCR a single page; return (text, [_OcrWord with global char offsets])."""
    img, zoom = _render_page_to_pil(page)
    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT,
        config=_TESS_CONFIG,
    )
    parts: List[str] = []
    words: List[_OcrWord] = []
    cursor = char_offset
    n = len(data["text"])
    for i in range(n):
        word = data["text"][i]
        if not word or not word.strip():
            continue
        if parts:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(word)
        cursor += len(word)
        # Convert pixel bbox -> PDF points by dividing by the zoom factor.
        x_px = float(data["left"][i])
        y_px = float(data["top"][i])
        w_px = float(data["width"][i])
        h_px = float(data["height"][i])
        bbox_pdf = (
            x_px / zoom,
            y_px / zoom,
            (x_px + w_px) / zoom,
            (y_px + h_px) / zoom,
        )
        words.append(_OcrWord(start, cursor, bbox_pdf))
    return "".join(parts), words


def _build_page_index(doc: "fitz.Document") -> Tuple[str, List[_PageInfo]]:
    """Return (full_text, [PageInfo, ...]) covering every page of the doc."""
    full_parts: List[str] = []
    pages: List[_PageInfo] = []
    cursor = 0
    for i, page in enumerate(doc):
        embedded = page.get_text("text") or ""
        if len(embedded.strip()) >= _MIN_TEXT_CHARS_PER_PAGE:
            text = embedded
            mode = "text"
            ocr_words: List[_OcrWord] = []
        else:
            logger.info("PDF page %d has little/no embedded text — running OCR", i + 1)
            try:
                text, ocr_words = _ocr_page(page, char_offset=cursor)
                mode = "ocr"
            except Exception:
                logger.exception("OCR failed for page %d; falling back to embedded text", i + 1)
                text = embedded
                mode = "text"
                ocr_words = []
        start = cursor
        full_parts.append(text)
        cursor += len(text)
        # Pages are joined with a newline so char offsets stay aligned.
        if i < len(doc) - 1:
            full_parts.append("\n")
            cursor += 1
        pages.append(_PageInfo(
            index=i,
            mode=mode,
            char_start=start,
            char_end=start + len(text),
            ocr_words=ocr_words,
        ))
    return "".join(full_parts), pages


# --- Public API used by the dispatcher ---

def extract_text(path: Path) -> str:
    """Concatenate text from every page, OCR-ing scanned pages as needed."""
    with fitz.open(path) as doc:
        full, _pages = _build_page_index(doc)
    return full


def has_text_layer(path: Path) -> bool:
    """True if at least one page has embedded text. (Used by tests / callers.)"""
    with fitz.open(path) as doc:
        for page in doc:
            if (page.get_text("text") or "").strip():
                return True
    return False


def redact(path: Path, entities: List[Dict[str, Any]], out_path: Path) -> None:
    """Redact a PDF, handling text pages and OCR'd (scanned) pages uniformly.

    For each page we decide based on the page index whether to:
      - run literal text-search redaction (text pages), or
      - cover OCR word bboxes with black rectangles (scanned pages).
    """
    with fitz.open(path) as doc:
        _full, pages = _build_page_index(doc)

        # Bucket entities by page using the global char offsets.
        per_page: Dict[int, List[Dict[str, Any]]] = {p.index: [] for p in pages}
        unbucketed: List[Dict[str, Any]] = []
        for e in entities:
            s = e.get("start"); t = e.get("end")
            if s is None or t is None:
                unbucketed.append(e)
                continue
            placed = False
            for p in pages:
                if s < p.char_end and t > p.char_start:
                    per_page[p.index].append(e)
                    placed = True
            if not placed:
                unbucketed.append(e)

        for page_info in pages:
            page = doc[page_info.index]
            page_entities = per_page.get(page_info.index, [])

            if page_info.mode == "text":
                # ---- Text page: precise, true redaction ----
                # Use literal text-search to find each entity word on the page.
                # We also accept the unbucketed entities here as a safety net,
                # since duplicates only result in extra redaction (no-op when
                # the word does not appear).
                seen: set = set()
                candidates = page_entities + unbucketed
                for e in candidates:
                    word = (e.get("word") or "").strip()
                    if not word:
                        continue
                    label = e.get("entity_group", "PII").upper()
                    key = (word, label)
                    if key in seen:
                        continue
                    seen.add(key)
                    for r in page.search_for(word, quads=False):
                        page.add_redact_annot(
                            r,
                            text=f"[{label}]",
                            fill=(0, 0, 0),
                            text_color=(1, 1, 1),
                        )
                page.apply_redactions()
            else:
                # ---- OCR page: redact by OCR word bbox ----
                for e in page_entities:
                    s = int(e["start"]); t = int(e["end"])
                    label = e.get("entity_group", "PII").upper()
                    for w in page_info.ocr_words:
                        # overlap test in global char space
                        if w.char_end <= s or w.char_start >= t:
                            continue
                        rect = fitz.Rect(*w.bbox_pdf)
                        page.add_redact_annot(
                            rect,
                            text=f"[{label}]",
                            fill=(0, 0, 0),
                            text_color=(1, 1, 1),
                        )
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)

        doc.save(out_path, garbage=4, deflate=True, clean=True)
