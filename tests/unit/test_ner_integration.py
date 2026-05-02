"""Integration tests for the three-layer detection pipeline.

Verifies that the privacy-filter model + NER auxiliary model + regex rules
collectively cover the realistic invoice failure case (company names,
addresses, tax IDs, etc.) without any hand-curated keyword list.
"""
from __future__ import annotations

from app.model import PrivacyFilter


INVOICE_TEXT = """\
Invoice No. 202526013                                Dated 13-Feb-26
                    Shanmuga Hospital Ltd.,
                    51/24 SARADHA COLLEGE ROAD
                                SALEM
                    CIN : L85110TZ2020PLC033974
              UDYAM-TN-20-0015819 (Small/Services)
                       GSTIN/UIN: 33ABDCS8326A1ZP
                  State Name : Tamil Nadu, Code : 33

   Party : TANUH
           The Chairman
           Department of Electrical Engineering
           Indian Institute of Science,
           Bangalore 560 012
   GSTIN/UIN   : 29AAATI1501J2ZV

Company's PAN     :  ABDCS8326A
Bank Name        : PUNJAB NATIONAL BANK
A/c No.          : 9180002100039809
Branch & IFS Code: Salem Main Branch & PUNB0041600
"""


def _labels_in(entities):
    return {e.entity_group if hasattr(e, "entity_group") else e["entity_group"] for e in entities}


def _words_for(entities, label):
    out = set()
    for e in entities:
        eg = e.entity_group if hasattr(e, "entity_group") else e["entity_group"]
        if eg == label:
            out.add(e.word if hasattr(e, "word") else e["word"])
    return out


def test_pipeline_redacts_organisations():
    pf = PrivacyFilter.instance()
    ents = pf.detect(INVOICE_TEXT)
    orgs = _words_for(ents, "org_name")
    # NER fake catches all of these.
    assert any("Shanmuga Hospital" in o for o in orgs)
    assert any("PUNJAB NATIONAL BANK" in o for o in orgs)
    assert "Indian Institute of Science" in orgs
    assert "TANUH" in orgs


def test_pipeline_redacts_locations():
    pf = PrivacyFilter.instance()
    ents = pf.detect(INVOICE_TEXT)
    locs = _words_for(ents, "address_location")
    assert "Bangalore" in locs
    assert "Salem" in locs
    assert "Tamil Nadu" in locs


def test_pipeline_redacts_tax_ids():
    pf = PrivacyFilter.instance()
    ents = pf.detect(INVOICE_TEXT)
    assert "33ABDCS8326A1ZP" in _words_for(ents, "tax_id_gstin")
    assert "29AAATI1501J2ZV" in _words_for(ents, "tax_id_gstin")
    assert "L85110TZ2020PLC033974" in _words_for(ents, "tax_id_cin")
    assert "UDYAM-TN-20-0015819" in _words_for(ents, "tax_id_udyam")
    assert "ABDCS8326A" in _words_for(ents, "tax_id_pan")


def test_pipeline_redacts_bank_codes_and_accounts():
    pf = PrivacyFilter.instance()
    ents = pf.detect(INVOICE_TEXT)
    assert "PUNB0041600" in _words_for(ents, "bank_ifsc")
    assert "9180002100039809" in _words_for(ents, "account_number")


def test_no_overlapping_duplicates_for_org_inside_ner_span():
    """If NER says 'Shanmuga Hospital Ltd' and rule extra says 'Shanmuga',
    the rule span (covered by NER) should not produce a duplicate.

    This is the kind of double-redact problem _resolve_overlaps prevents.
    """
    import os
    os.environ["EXTRA_REDACTION_KEYWORDS"] = "Shanmuga"
    try:
        pf = PrivacyFilter.instance()
        ents = pf.detect(INVOICE_TEXT)
        # Find spans that cover the substring 'Shanmuga'.
        idx = INVOICE_TEXT.find("Shanmuga Hospital Ltd")
        # There should be at most one span starting in this region.
        starts = [
            (e["start"] if isinstance(e, dict) else e.start)
            for e in ents
            if (e["start"] if isinstance(e, dict) else e.start) is not None
            and idx <= (e["start"] if isinstance(e, dict) else e.start) <= idx + 2
        ]
        # Either one merged 'org_name' span or two from rule+NER -- but the
        # rule entry of just 'Shanmuga' should be subsumed by the NER span
        # of 'Shanmuga Hospital Ltd'. Concretely: no rule span starting at
        # exactly 'idx' with end == idx + len('Shanmuga').
        rule_only = [
            e for e in ents
            if (e.get("_source") if isinstance(e, dict) else getattr(e, "_source", None)) == "rule:extra"
        ]
        # The rule-extra span for 'Shanmuga' was contained in the NER
        # 'Shanmuga Hospital Ltd' span -> should have been dropped.
        assert all(
            (r["word"] if isinstance(r, dict) else r.word).lower() == "tanuh"
            or "shanmuga" not in (r["word"] if isinstance(r, dict) else r.word).lower()
            for r in rule_only
        ) or len(rule_only) == 0
    finally:
        os.environ.pop("EXTRA_REDACTION_KEYWORDS", None)


def test_pipeline_still_catches_personal_pii():
    """Sanity: the privacy-filter detections still come through alongside NER + rules."""
    text = "Patient Alice Smith, alice@example.com, born 1990-01-15."
    pf = PrivacyFilter.instance()
    ents = pf.detect(text)
    labels = _labels_in(ents)
    assert "private_person" in labels
    assert "private_email" in labels
    assert "private_date" in labels
