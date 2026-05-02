"""Format dispatcher: pick the right extractor/redactor by file type."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Dict, Any

from app.extractors import txt as txt_x
from app.extractors import pdf as pdf_x
from app.extractors import docx as docx_x
from app.extractors import image as image_x
from app.extractors import dicom as dicom_x


@dataclass
class FormatHandler:
    name: str
    extract: Callable[[Path], str]
    redact: Callable[[Path, List[Dict[str, Any]], Path], None]
    out_extension: str  # extension preserved for redacted output


# MIME / extension -> handler
_HANDLERS: dict[str, FormatHandler] = {
    ".txt": FormatHandler("text", txt_x.extract_text, txt_x.redact, ".txt"),
    ".md":  FormatHandler("text", txt_x.extract_text, txt_x.redact, ".md"),
    ".log": FormatHandler("text", txt_x.extract_text, txt_x.redact, ".log"),
    ".csv": FormatHandler("text", txt_x.extract_text, txt_x.redact, ".csv"),
    ".pdf": FormatHandler("pdf", pdf_x.extract_text, pdf_x.redact, ".pdf"),
    ".docx": FormatHandler("docx", docx_x.extract_text, docx_x.redact, ".docx"),
    ".png": FormatHandler("image", image_x.extract_text, image_x.redact, ".png"),
    ".jpg": FormatHandler("image", image_x.extract_text, image_x.redact, ".jpg"),
    ".jpeg": FormatHandler("image", image_x.extract_text, image_x.redact, ".jpeg"),
    ".tif": FormatHandler("image", image_x.extract_text, image_x.redact, ".tif"),
    ".tiff": FormatHandler("image", image_x.extract_text, image_x.redact, ".tiff"),
    ".dcm": FormatHandler("dicom", dicom_x.extract_text, dicom_x.redact, ".dcm"),
    ".dicom": FormatHandler("dicom", dicom_x.extract_text, dicom_x.redact, ".dcm"),
}


def get_handler(filename: str) -> FormatHandler:
    suffix = Path(filename).suffix.lower()
    if suffix not in _HANDLERS:
        raise ValueError(
            f"Unsupported file type: {suffix or '<none>'}. "
            f"Supported: {', '.join(sorted(_HANDLERS.keys()))}"
        )
    return _HANDLERS[suffix]


def supported_extensions() -> list[str]:
    return sorted(_HANDLERS.keys())
