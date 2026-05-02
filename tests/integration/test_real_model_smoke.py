"""Optional smoke test against the real openai/privacy-filter checkpoint.

Skipped by default. To run:

    pytest -m requires_model -k smoke

This downloads ~1.5 GB on first run.
"""
from __future__ import annotations

import os

import pytest


@pytest.mark.slow
@pytest.mark.requires_model
def test_real_model_detects_basic_pii():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    # Allow the user to opt out even within the marker.
    if os.getenv("RUN_REAL_MODEL_SMOKE", "0") != "1":
        pytest.skip("Set RUN_REAL_MODEL_SMOKE=1 to run the real-model smoke test.")

    from app.model import PrivacyFilter

    pf = PrivacyFilter()
    pf.load()
    spans = pf.detect("My name is Harry Potter and my email is harry.potter@hogwarts.edu.")
    labels = {s["entity_group"] for s in spans}
    assert "private_person" in labels
    assert "private_email" in labels
