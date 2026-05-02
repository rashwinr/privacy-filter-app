"""Format dispatcher: app.redactor.get_handler / supported_extensions."""
from __future__ import annotations

import pytest

# Importing the dispatcher pulls in every extractor module. If any optional
# dep (pydicom, fitz, ...) is missing, skip the whole file gracefully.
pytest.importorskip("pydicom")
pytest.importorskip("fitz")
pytest.importorskip("docx")
pytest.importorskip("PIL")

from app.redactor import get_handler, supported_extensions  # noqa: E402


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("memo.txt", "text"),
        ("notes.MD", "text"),
        ("table.csv", "text"),
        ("report.pdf", "pdf"),
        ("Letter.PDF", "pdf"),
        ("brief.docx", "docx"),
        ("scan.png", "image"),
        ("photo.JPG", "image"),
        ("xray.tiff", "image"),
        ("study.dcm", "dicom"),
        ("series.dicom", "dicom"),
    ],
)
def test_get_handler_recognises_known_types(filename, expected):
    h = get_handler(filename)
    assert h.name == expected


def test_get_handler_rejects_unknown_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_handler("archive.zip")


def test_get_handler_rejects_no_extension():
    with pytest.raises(ValueError):
        get_handler("README")


def test_supported_extensions_listed():
    exts = supported_extensions()
    for required in (".txt", ".pdf", ".docx", ".png", ".dcm"):
        assert required in exts


def test_handler_out_extension_normalises_dicom():
    # Both .dcm and .dicom should write out as .dcm
    assert get_handler("a.dcm").out_extension == ".dcm"
    assert get_handler("a.dicom").out_extension == ".dcm"
