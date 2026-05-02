"""Tests for the structured-identifier rule detector module."""
from __future__ import annotations

from app import rule_detectors


# Sample text from a real Indian tax invoice.
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
   State Name  : Karnataka, Code : 29

Company's PAN     :  ABDCS8326A
A/c Holder's Name: Shanmuga Hospital Ltd
Bank Name        : PUNJAB NATIONAL BANK
A/c No.          : 9180002100039809
Branch & IFS Code: Salem Main Branch & PUNB0041600
Contact: priya.k@example.com  https://example.com/team
"""


def _words_for(entities, label):
    return {e["word"] for e in entities if e["entity_group"] == label}


def test_detects_gstin():
    ents = rule_detectors.detect(INVOICE_TEXT)
    assert "33ABDCS8326A1ZP" in _words_for(ents, "tax_id_gstin")
    assert "29AAATI1501J2ZV" in _words_for(ents, "tax_id_gstin")


def test_detects_cin():
    ents = rule_detectors.detect(INVOICE_TEXT)
    assert "L85110TZ2020PLC033974" in _words_for(ents, "tax_id_cin")


def test_detects_udyam():
    ents = rule_detectors.detect(INVOICE_TEXT)
    assert "UDYAM-TN-20-0015819" in _words_for(ents, "tax_id_udyam")


def test_detects_pan():
    ents = rule_detectors.detect(INVOICE_TEXT)
    assert "ABDCS8326A" in _words_for(ents, "tax_id_pan")


def test_detects_ifsc():
    ents = rule_detectors.detect(INVOICE_TEXT)
    assert "PUNB0041600" in _words_for(ents, "bank_ifsc")


def test_detects_long_account_number():
    ents = rule_detectors.detect(INVOICE_TEXT)
    accounts = _words_for(ents, "account_number")
    assert "9180002100039809" in accounts


def test_detects_email():
    ents = rule_detectors.detect(INVOICE_TEXT)
    assert "priya.k@example.com" in _words_for(ents, "private_email")


def test_detects_url():
    ents = rule_detectors.detect(INVOICE_TEXT)
    urls = _words_for(ents, "private_url")
    assert any("example.com/team" in u for u in urls)


def test_detects_aadhaar_with_spaces():
    text = "Aadhaar: 1234 5678 9012 issued in 2020."
    ents = rule_detectors.detect(text)
    assert any(e["entity_group"] == "aadhaar" for e in ents)


def test_detects_us_ssn():
    ents = rule_detectors.detect("SSN: 123-45-6789")
    assert "123-45-6789" in _words_for(ents, "us_ssn")


def test_detects_us_ein():
    ents = rule_detectors.detect("EIN: 12-3456789 for the company.")
    assert "12-3456789" in _words_for(ents, "us_ein")


def test_detects_iban():
    ents = rule_detectors.detect("IBAN DE89370400440532013000 is on file.")
    assert "DE89370400440532013000" in _words_for(ents, "bank_iban")


def test_detects_indian_phone():
    text = "Call me at +91 98765 43210 or 9876543210."
    ents = rule_detectors.detect(text)
    phones = _words_for(ents, "private_phone")
    assert any("9876543210" in p.replace(" ", "") for p in phones)


def test_extra_redaction_keywords_via_env(monkeypatch):
    monkeypatch.setenv("EXTRA_REDACTION_KEYWORDS", "TANUH,Shanmuga")
    ents = rule_detectors.detect(INVOICE_TEXT)
    extra = {e["word"].lower() for e in ents if e.get("_source") == "rule:extra"}
    assert "tanuh" in extra
    assert "shanmuga" in extra


def test_disabled_specific_detectors(monkeypatch):
    monkeypatch.setenv("DISABLED_RULE_DETECTORS", "pan,gstin")
    ents = rule_detectors.detect(INVOICE_TEXT)
    assert "tax_id_pan" not in {e["entity_group"] for e in ents}
    assert "tax_id_gstin" not in {e["entity_group"] for e in ents}


def test_disabled_globally_via_env(monkeypatch):
    monkeypatch.setenv("ENABLE_RULE_DETECTORS", "0")
    ents = rule_detectors.detect(INVOICE_TEXT)
    assert ents == []


def test_pan_does_not_match_inside_longer_word():
    text = "Random id 1ABCDE1234F2 here."
    ents = rule_detectors.detect(text)
    assert _words_for(ents, "tax_id_pan") == set()


def test_empty_text_returns_empty():
    assert rule_detectors.detect("") == []
    assert rule_detectors.detect("   \n\t ") == []


# --- Person-name detectors ---------------------------------------------------


def test_detects_honorific_prefixed_name():
    """MR./MRS./DR. + capitalised tokens should be tagged as a person."""
    text = "Insured Name : MR. ASHWIN RAJ KUMAR"
    ents = rule_detectors.detect(text)
    persons = _words_for(ents, "private_person")
    assert any("ASHWIN RAJ KUMAR" in p for p in persons)


def test_honorific_does_not_cross_line_break():
    """Honorific match must stop at end-of-line, not eat next-line label."""
    text = "Insured Name : MR. ASHWIN RAJ KUMAR\nEmail: x@y.com\n"
    ents = rule_detectors.detect(text)
    for e in ents:
        if e.get("_source") == "rule:honorific_name":
            assert "\n" not in e["word"]
            assert "Email" not in e["word"]


def test_detects_indian_honorifics():
    text = "Nominee SMT. LATHA M and proposer SHRI RAVI KUMAR"
    ents = rule_detectors.detect(text)
    persons = _words_for(ents, "private_person")
    assert any("LATHA M" in p for p in persons)
    assert any("RAVI KUMAR" in p for p in persons)


def test_detects_allcaps_multiword_name():
    text = "Account holder ASHWIN RAJ KUMAR has the policy."
    ents = rule_detectors.detect(text)
    persons = _words_for(ents, "private_person")
    assert "ASHWIN RAJ KUMAR" in persons


def test_allcaps_does_not_match_two_token_headers():
    """Avoid eating common 2-word headers like 'POLICY NUMBER'."""
    text = "POLICY NUMBER and PERIOD OF INSURANCE start below."
    ents = rule_detectors.detect(text)
    persons = _words_for(ents, "private_person")
    # 2-token sequences should NOT be tagged.
    assert "POLICY NUMBER" not in persons
    # 3-token sequences DO match (intentional, low score, ML/NER refines).
    # We just check we don't over-claim 2-token matches.


# --- Labeled-field value detector --------------------------------------------


def test_labeled_field_redacts_insured_name_value():
    text = "Insured Name : MR. ASHWIN RAJ KUMAR\nOther stuff\n"
    ents = rule_detectors.detect(text)
    sources = [(e["word"], e["entity_group"], e.get("_source")) for e in ents]
    assert any(
        e.get("_source") == "rule:labeled_field"
        and "ASHWIN RAJ KUMAR" in e["word"]
        for e in ents
    ), sources


def test_labeled_field_redacts_bank_name_as_org():
    text = "Bank Name : HDFC BANK LIMITED\n"
    ents = rule_detectors.detect(text)
    labeled = [e for e in ents if e.get("_source") == "rule:labeled_field"]
    assert any(
        e["entity_group"] == "org_name" and "HDFC BANK LIMITED" in e["word"]
        for e in labeled
    )


def test_labeled_field_redacts_email_label():
    text = "Email : nia.800000tata@newindia.co.in\n"
    ents = rule_detectors.detect(text)
    labeled = [e for e in ents if e.get("_source") == "rule:labeled_field"]
    assert any(
        e["entity_group"] == "private_email"
        and "nia.800000tata@newindia.co.in" in e["word"].lower()
        for e in labeled
    )


def test_labeled_field_handles_dash_separator():
    text = "Customer Name - JANE DOE\n"
    ents = rule_detectors.detect(text)
    labeled = [e for e in ents if e.get("_source") == "rule:labeled_field"]
    assert any("JANE DOE" in e["word"] for e in labeled)


def test_labeled_field_disabled_via_env(monkeypatch):
    monkeypatch.setenv("DISABLED_RULE_DETECTORS", "labeled_field")
    text = "Insured Name : MR. ASHWIN RAJ KUMAR"
    ents = rule_detectors.detect(text)
    assert not any(e.get("_source") == "rule:labeled_field" for e in ents)


# --- Loose email regex (OCR-tolerant) ----------------------------------------


def test_loose_email_with_space_around_at():
    text = "reach us at support @ tmibasl.com today"
    ents = rule_detectors.detect(text)
    emails = _words_for(ents, "private_email")
    assert any("@" in e and "tmibasl" in e for e in emails)


def test_loose_email_with_space_before_tld():
    text = "reach us at support@tmibasl. com today"
    ents = rule_detectors.detect(text)
    emails = _words_for(ents, "private_email")
    assert any("tmibasl" in e for e in emails)


def test_loose_email_with_linebreak_in_local_part():
    text = "contact:\nsupport\n@tmibasl.com here"
    ents = rule_detectors.detect(text)
    emails = _words_for(ents, "private_email")
    assert any("tmibasl" in e for e in emails)


def test_loose_email_does_not_swallow_prose():
    """Critical: must not match across English prose preceding the email."""
    text = "Hi, my name is Alice Smith and my email is alice@example.com."
    ents = rule_detectors.detect(text)
    emails = _words_for(ents, "private_email")
    # Should match alice@example.com -- and ONLY that.
    assert "alice@example.com" in emails
    # No match should include the word 'name' or 'email' (prose).
    assert not any("name" in e.lower() or "email is" in e.lower() for e in emails)
