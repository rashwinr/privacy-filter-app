"""DICOM extractor + de-identifier.

Two-pass approach
─────────────────
1. Header sanitization
   Collects identifying tags (PatientName, PatientID, PatientBirthDate, etc.)
   into a text blob, runs the privacy filter on that blob, then
   null/anonymizes matched tags per DICOM PS3.15 Basic Confidentiality
   Profile.  Also removes ALL private tags via pydicom's
   ``remove_private_tags()`` — vendor-specific elements that can carry PHI
   in proprietary fields (ported from deidentification.py).

2. Pixel burn-in redaction
   Blacks out the top ``DICOM_PIXEL_REDACT_ROWS`` rows of the stored pixel
   array to remove any text overlaid ("burned in") on the scan image.
   The row count defaults to 100 (matching the original deidentification.py
   script) and is controlled by the ``DICOM_PIXEL_REDACT_ROWS`` env var.
   If the DICOM object has no pixel data (SR, KO, etc.) this step is
   silently skipped.

This module returns extracted *text* (concatenation of identifying tags) so
the caller can run the NER model once and feed entities back to ``redact()``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of pixel rows to black out at the top of the DICOM image.
# Matches the original deidentification.py: redacted_pixel_array[0:100, :] = 0
_PIXEL_REDACT_ROWS: int = int(os.getenv("DICOM_PIXEL_REDACT_ROWS", "100"))

# ---------------------------------------------------------------------------
# DICOM Value Representations that can't hold a free-text placeholder.
# For these VRs we set the value to "" instead of "REDACTED".
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# PII tags — DICOM PS3.15 Basic Application Confidentiality Profile (subset)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tag_text(ds: Dataset) -> str:
    """Concatenate all non-empty PII tag values into a text blob for NER."""
    parts: List[str] = []
    for tag in PII_TAGS:
        v = getattr(ds, tag, None)
        if v is None or v == "":
            continue
        parts.append(f"{tag}: {v}")
    return "\n".join(parts)


def _vr_for(ds: Dataset, tag_name: str) -> str:
    """Look up the Value Representation of a named tag on a dataset."""
    elem = ds.data_element(tag_name)
    return elem.VR if elem is not None else ""


def _blackout_pixel_rows(ds: Dataset, n_rows: int) -> bool:
    """
    Black out the top ``n_rows`` rows of the pixel array in-place.

    Mirrors deidentification.py:
        redacted_pixel_array = ds.pixel_array.copy()
        redacted_pixel_array[0:100, :] = 0
        ds.PixelData = redacted_pixel_array.tobytes()

    Returns True if pixel data was modified, False if skipped (no pixels).
    """
    if n_rows <= 0:
        return False

    try:
        pixel_array = ds.pixel_array.copy()
    except (AttributeError, Exception) as exc:
        # SR, KO, PR objects have no pixel data — skip silently.
        logger.debug("DICOM pixel blackout skipped (no pixel data): %s", exc)
        return False

    if pixel_array.ndim < 2:
        logger.debug("DICOM pixel blackout skipped: unexpected array shape %s", pixel_array.shape)
        return False

    actual_rows = min(n_rows, pixel_array.shape[0])
    pixel_array[:actual_rows, ...] = 0  # works for 2-D (greyscale) and 3-D (RGB/multi-frame)

    # Write modified pixel data back into the dataset
    ds.PixelData = pixel_array.tobytes()
    logger.info("DICOM pixel burn-in redaction: blacked out top %d rows", actual_rows)
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(path: Path) -> str:
    """Return a text blob of PII-carrying header tags for NER processing."""
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    return _tag_text(ds)


def redact(path: Path, entities: List[Dict[str, Any]], out_path: Path) -> None:
    """
    Fully de-identify a DICOM file:

    1. Anonymize all PII_TAGS (NER-guided; we scrub unconditionally for safety).
    2. Remove all private/vendor tags via ``remove_private_tags()``
       — catches proprietary PHI fields not covered by the standard tag list.
       (Ported from deidentification.py line 34.)
    3. Black out the top ``DICOM_PIXEL_REDACT_ROWS`` rows of the pixel array
       to eliminate burned-in patient annotations.
       (Ported from deidentification.py lines 37–41.)
    4. Write DICOM compliance markers and save.

    Args:
        path:     Path to the original DICOM file.
        entities: NER entity list from PrivacyFilter.detect() (used for audit;
                  we always scrub all PII_TAGS regardless).
        out_path: Destination path for the de-identified DICOM.
    """
    ds = pydicom.dcmread(str(path), force=True)

    # ── Step 1: Anonymize known PII tags ─────────────────────────────────────
    for tag in PII_TAGS:
        if not hasattr(ds, tag):
            continue
        vr = _vr_for(ds, tag)
        replacement = "" if vr in _NON_TEXT_VRS else "REDACTED"
        try:
            setattr(ds, tag, replacement)
        except Exception:
            setattr(ds, tag, "")

    # ── Step 2: Remove ALL private/vendor tags ────────────────────────────────
    # Ported from deidentification.py: ds.remove_private_tags()
    try:
        ds.remove_private_tags()
        logger.info("DICOM private tags removed")
    except Exception as exc:
        logger.warning("DICOM remove_private_tags failed (non-fatal): %s", exc)

    # ── Step 3: Pixel burn-in redaction ───────────────────────────────────────
    # Ported from deidentification.py:
    #   redacted_pixel_array[0:100, :] = 0
    #   ds.PixelData = redacted_pixel_array.tobytes()
    _blackout_pixel_rows(ds, _PIXEL_REDACT_ROWS)

    # ── Step 4: DICOM compliance markers ─────────────────────────────────────
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = (
        "openai/privacy-filter NER + tag scrub + "
        "private tag removal + pixel burn-in redaction"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_path), write_like_original=False)
    logger.info("De-identified DICOM saved → %s", out_path)
