"""DICOM tag extractor + de-identifier."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydicom")

import pydicom  # noqa: E402
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage  # noqa: E402

from app.extractors.dicom import PII_TAGS, extract_text, redact  # noqa: E402


def _make_minimal_dicom(path: Path) -> Path:
    """Build a minimal valid DICOM file with PII tags populated."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.PatientName = "Smith^Alice"
    ds.PatientID = "PID-0001"
    ds.PatientBirthDate = "19900115"
    ds.PatientSex = "F"
    ds.ReferringPhysicianName = "Dr^House"
    ds.InstitutionName = "Acme Hospital"
    ds.AccessionNumber = "ACC-42"
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = "1.2.3.4.5"
    ds.StudyInstanceUID = "1.2.3.4"
    ds.SeriesInstanceUID = "1.2.3.4.6"
    ds.Modality = "OT"

    ds.save_as(str(path), write_like_original=False)
    return path


def test_extract_text_collects_pii_tags(tmp_path: Path):
    src = _make_minimal_dicom(tmp_path / "in.dcm")
    text = extract_text(src)
    assert "PatientName" in text
    assert "Alice" in text  # name component leaked into the text blob
    assert "PID-0001" in text
    assert "Acme Hospital" in text


def test_redact_anonymises_pii_tags(tmp_path: Path):
    src = _make_minimal_dicom(tmp_path / "in.dcm")
    out = tmp_path / "out.dcm"
    redact(src, [], out)  # entity list is informational; redaction is unconditional

    ds = pydicom.dcmread(str(out), force=True)
    # Every PII tag must no longer carry the original value.
    assert "Smith" not in str(getattr(ds, "PatientName", ""))
    assert getattr(ds, "PatientID", "") in {"REDACTED", ""}
    assert getattr(ds, "InstitutionName", "") in {"REDACTED", ""}
    # The de-identification flag must be set per DICOM standard.
    assert ds.PatientIdentityRemoved == "YES"
    assert "openai/privacy-filter" in str(ds.DeidentificationMethod)


def test_redact_preserves_modality_and_sop(tmp_path: Path):
    """Non-PII tags must NOT be touched."""
    src = _make_minimal_dicom(tmp_path / "in.dcm")
    out = tmp_path / "out.dcm"
    redact(src, [], out)
    ds = pydicom.dcmread(str(out), force=True)
    assert ds.Modality == "OT"
    assert ds.StudyInstanceUID == "1.2.3.4"
    assert ds.SOPInstanceUID == "1.2.3.4.5"


def test_redact_clears_date_and_numeric_tags_without_invalid_value(tmp_path: Path):
    """Date/numeric VRs must be cleared, not stuffed with 'REDACTED'."""
    src = _make_minimal_dicom(tmp_path / "in.dcm")
    out = tmp_path / "out.dcm"
    redact(src, [], out)

    ds = pydicom.dcmread(str(out), force=True)
    # PatientBirthDate has VR=DA — must end up empty, not the literal 'REDACTED'.
    pbd = str(getattr(ds, "PatientBirthDate", ""))
    assert pbd != "REDACTED"
    # Free-text fields can keep the placeholder.
    assert getattr(ds, "PatientName", None) is None or "REDACTED" in str(ds.PatientName)


def test_pii_tag_list_is_non_empty_and_unique():
    assert len(PII_TAGS) > 5
    assert len(PII_TAGS) == len(set(PII_TAGS))
