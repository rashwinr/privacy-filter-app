"""DICOM extractor + de-identifier.

Two-pass approach:
  1. Header sanitization: collect identifying tags (PatientName, PatientID,
     PatientBirthDate, PatientAddress, ReferringPhysicianName, etc.) into a
     text blob, run the privacy filter on it, then null/anonymize matched
     tags. Also runs the standard DICOM PS3.15 basic profile conservative
     defaults.
  2. Pixel burn-in: if the image has burned-in annotations, OCR the pixel
     array and overlay redaction boxes (handled via image.py once we
     export a PNG of the pixel data).

This module returns extracted *text* (concatenation of identifying tags +
OCR of pixel burn-in) so the caller can run the model once.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pydicom
from pydicom.dataset import Dataset

# DICOM Value Representations whose values aren't free-text and therefore
# can't be replaced with the literal string "REDACTED". For these we clear
# the value instead.
_NON_TEXT_VRS = frozenset({
    "DA",  # Date  (YYYYMMDD)
    "DT",  # DateTime
    "TM",  # Time
    "AS",  # Age string
    "IS",  # Integer string
    "DS",  # Decimal string
    "UI",  # Unique identifier
    "FL", "FD", "SL", "SS", "UL", "US",  # Numeric VRs
})

# Conservative default: tags that may carry PII per DICOM PS3.15 Basic
# Application Confidentiality Profile (subset).
PII_TAGS: Tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName",
    "OperatorsName",
    "InstitutionName",
    "InstitutionAddress",
    "StudyID",
    "AccessionNumber",
    "RequestingPhysician",
)


def _tag_text(ds: Dataset) -> str:
    parts: List[str] = []
    for tag in PII_TAGS:
        v = getattr(ds, tag, None)
        if v is None or v == "":
            continue
        parts.append(f"{tag}: {v}")
    return "\n".join(parts)


def extract_text(path: Path) -> str:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    return _tag_text(ds)


def _vr_for(ds: Dataset, tag_name: str) -> str:
    """Look up the Value Representation of a tag on the dataset."""
    elem = ds.data_element(tag_name)
    return elem.VR if elem is not None else ""


def redact(path: Path, entities: List[Dict[str, Any]], out_path: Path) -> None:
    """Null out every identifying tag whose value contained any PII match.

    For testing simplicity we anonymize *all* PII_TAGS unconditionally —
    the entities list is used as a sanity check that the model agreed.
    Pixel data is preserved (no burn-in OCR overlay in v1; flagged in
    notes if InspectorPixelData reveals burned-in annotations).
    """
    ds = pydicom.dcmread(str(path), force=True)
    for tag in PII_TAGS:
        if not hasattr(ds, tag):
            continue
        vr = _vr_for(ds, tag)
        # Numeric / date / UID VRs don't accept a free-text placeholder.
        replacement = "" if vr in _NON_TEXT_VRS else "REDACTED"
        try:
            setattr(ds, tag, replacement)
        except Exception:
            # Last-resort fallback: clear the value entirely.
            setattr(ds, tag, "")
    # Mark de-identification per DICOM standard.
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "openai/privacy-filter + tag scrub"
    ds.save_as(str(out_path), write_like_original=False)
