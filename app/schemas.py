"""Pydantic schemas for API contracts."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A single detected PII span."""
    entity_group: str = Field(..., description="Privacy label (e.g., private_person)")
    score: float
    word: str
    start: Optional[int] = None
    end: Optional[int] = None


class RedactionResult(BaseModel):
    job_id: str
    filename: str
    content_type: str
    entities: List[Entity]
    entity_counts: dict[str, int]
    original_url: str
    redacted_url: str
    text_preview_original: Optional[str] = None
    text_preview_redacted: Optional[str] = None
    notes: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    model_loaded: bool

    model_config = {
        "protected_namespaces": ()
    }
