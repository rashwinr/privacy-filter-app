"""Plain-text extractor + redactor."""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any


def extract_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def redact(path: Path, entities: List[Dict[str, Any]], out_path: Path) -> None:
    """Replace each detected span with a [REDACTED:LABEL] tag.

    Uses character offsets returned by the model so we don't depend on
    string matching (which would fail on duplicates).
    """
    text = extract_text(path)
    # Sort spans by start desc so offsets stay valid as we slice.
    spans = sorted(
        [e for e in entities if e.get("start") is not None and e.get("end") is not None],
        key=lambda e: e["start"],
        reverse=True,
    )
    for e in spans:
        s, t = int(e["start"]), int(e["end"])
        label = e.get("entity_group", "PII").upper()
        text = text[:s] + f"[REDACTED:{label}]" + text[t:]
    out_path.write_text(text, encoding="utf-8")
