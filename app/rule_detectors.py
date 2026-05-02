"""Rule-based detectors for structured identifiers the ML models can't see.

Two complementary models cover the *named-entity* space:

* ``openai/privacy-filter`` -- personal PII (names, dates, addresses,
  phones, emails, accounts, secrets).
* ``dslim/bert-base-NER`` -- generic ORG / LOC / PER / MISC.

Together those cover company names, cities, states, countries, people --
generically, in any document, with no hand-curated keyword list.

What ML still can't see well are *structured identifiers* with a fixed
shape: tax IDs, bank routing codes, government numbers. Those are easy
for regex and impossible to mistake -- if the shape matches, it's almost
certainly that thing. This module covers exactly those.

Each detector returns the same dict shape as the ML models:
``{entity_group, score, word, start, end}``.

Configuration via environment variables:

* ``ENABLE_RULE_DETECTORS`` -- master switch (default: 1)
* ``DISABLED_RULE_DETECTORS`` -- comma-separated detector names to skip
* ``EXTRA_REDACTION_KEYWORDS`` -- comma-separated literal phrases to
  redact (escape hatch for organisation aliases the NER model misses,
  e.g. brand acronyms)
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Pattern, Tuple

logger = logging.getLogger(__name__)


# --- Structured-identifier patterns -------------------------------------------
# Each entry: (regex, label, score, name)

# Indian GSTIN: 15 chars, e.g. 33ABDCS8326A1ZP
_GSTIN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b")
# Indian PAN: AAAAA9999A
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
# Indian CIN: 21 chars, e.g. L85110TZ2020PLC033974
_CIN = re.compile(r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b")
# UDYAM number, e.g. UDYAM-TN-20-0015819
_UDYAM = re.compile(r"\bUDYAM-[A-Z]{2}-\d{2}-\d{7}\b")
# Indian TAN: 4 letters + 5 digits + 1 letter
_TAN = re.compile(r"\b[A-Z]{4}\d{5}[A-Z]\b")
# Indian IFSC: 4 letters + '0' + 6 alphanum
_IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
# Aadhaar 12-digit, optionally space/hyphen separated
_AADHAAR = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
# Indian mobile: optional +91/0, then [6-9] + 9 digits
_INDIAN_PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{9}(?!\d)")
# Generic 9-18 digit account-like numbers in non-digit context.
_LONG_ACCOUNT = re.compile(r"(?<!\d)\d{9,18}(?!\d)")
# US SSN
_US_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# US EIN (Employer Identification Number): NN-NNNNNNN
_US_EIN = re.compile(r"\b\d{2}-\d{7}\b")
# Generic credit-card-ish 13-19 digit (with optional space/hyphen separators).
_CC_LIKE = re.compile(
    r"(?<!\d)(?:\d{4}[\s-]?){3,4}\d{1,4}(?!\d)"
)
# IBAN: country code + 2 check + up to 30 alphanumeric, e.g. DE89370400440532013000
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
# Common email pattern (the ML model usually catches these too -- belt+braces).
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Loose email pattern that tolerates a SMALL amount of whitespace around the
# ``@`` and the trailing dot, and a single line-break splitting the local
# part or domain. Matches things like:
#   ``support@tmibasl. om``
#   ``support @ tmibasl.com``
#   ``support@\ntmibasl.com``
# We deliberately avoid allowing free-form whitespace inside the local part
# (which would let ``my name is Alice@example.com`` match the whole prefix).
# Each whitespace run is bounded to <= 3 chars and the local + domain bodies
# can each contain at most one inline whitespace run.
_EMAIL_LOOSE = re.compile(
    r"\b[A-Za-z0-9._%+\-]{1,40}(?:[ \t]?\n[ \t]*[A-Za-z0-9._%+\-]{1,40})?"
    r"[ \t\n]{0,3}@[ \t\n]{0,3}"
    r"[A-Za-z0-9._\-]{2,40}(?:[ \t]?\n[ \t]*[A-Za-z0-9._\-]{1,40})?"
    r"[ \t]{0,2}\.[ \t]{0,2}[A-Za-z]{2,6}\b"
)
# URL
_URL = re.compile(r"\bhttps?://[^\s<>]+", re.IGNORECASE)

# --- Person-name heuristics --------------------------------------------------
# Honorific-prefixed names (English + common Indian honorifics).
# Catches ``MR. ASHWIN RAJ KUMAR``, ``Dr. Jane Q. Public``, ``Smt. Latha M.``.
# Restricted to a single line (``[ \t]+``, not ``\s+``) so it doesn't eat
# downstream label keywords on the next line.
_HONORIFIC_NAME = re.compile(
    r"\b(?:MR|MRS|MS|MISS|DR|PROF|SHRI|SMT|SRI|MX|SIR|MADAM|REV|FR|SR)\.?[ \t]+"
    # 1st token requires 2+ chars (avoid matching just an article like 'A').
    r"[A-Z][A-Za-z'\-]+\.?"
    # Subsequent tokens may be a single capital initial (e.g. ``LATHA M``).
    r"(?:[ \t]+[A-Z][A-Za-z'\-]*\.?){0,4}",
)
# All-caps multi-word phrases of 3+ tokens that look like a person/proper-noun
# block (e.g. "ASHWIN RAJ KUMAR"). We restrict each token to 2-20 letters and
# require 3-6 tokens to avoid eating common 2-word headers like "POLICY
# NUMBER". False-positive risk is further reduced because the rule fires at
# a low score and ML/NER detectors win on overlap when they recognise the
# span as ORG/LOC instead of PER.
_ALLCAPS_NAME = re.compile(
    r"\b(?:[A-Z]{2,20}\s+){2,5}[A-Z]{2,20}\b",
)
# Labeled-field values: lines that start with a sensitive label, then a colon,
# then the value to redact. Captured group 1 is the value.
# Examples handled (case-insensitive):
#   ``Insured Name : MR. ASHWIN RAJ KUMAR``
#   ``Name of Nominee: SMT LATHA M``
#   ``Email :  nia.800000tata@newindia.co.in``
#   ``Bank Name : HDFC BANK LIMITED``
_LABELED_FIELDS = re.compile(
    r"(?im)^[ \t]*(?:"
    r"insured\s*name|name\s*of\s*(?:insured|nominee|proposer|policy\s*holder|account\s*holder)"
    r"|customer\s*name|holder\s*name|proposer\s*name|policy\s*holder|nominee\s*name|nominee"
    r"|registered\s*owner|owner\s*name|driver\s*name|patient\s*name"
    r"|e[\-\s]?mail(?:\s*id)?|email\s*address"
    r"|bank\s*name|branch\s*name|company\s*name|employer\s*name|firm\s*name|organi[sz]ation\s*name"
    r")[ \t]*[:\-][ \t]*([^\n\r]{2,200})$",
)
# Map labeled-field label keywords to a redaction tag.
_LABEL_KEYWORD_TAGS: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"e[\-\s]?mail|email", re.IGNORECASE), "private_email"),
    (re.compile(r"bank\s*name|branch\s*name", re.IGNORECASE), "org_name"),
    (re.compile(r"company|employer|firm|organi[sz]ation", re.IGNORECASE), "org_name"),
    # Default fallback: person name.
]


_DEFAULT_DETECTORS: List[Tuple[Pattern[str], str, float, str]] = [
    (_GSTIN, "tax_id_gstin", 0.99, "gstin"),
    (_CIN, "tax_id_cin", 0.99, "cin"),
    (_UDYAM, "tax_id_udyam", 0.99, "udyam"),
    # PAN check after CIN (CIN starts with letter; PAN is 10 chars total).
    (_PAN, "tax_id_pan", 0.97, "pan"),
    (_TAN, "tax_id_tan", 0.95, "tan"),
    (_IFSC, "bank_ifsc", 0.99, "ifsc"),
    (_AADHAAR, "aadhaar", 0.95, "aadhaar"),
    (_US_SSN, "us_ssn", 0.99, "us_ssn"),
    (_US_EIN, "us_ein", 0.95, "us_ein"),
    (_IBAN, "bank_iban", 0.95, "iban"),
    (_INDIAN_PHONE, "private_phone", 0.93, "indian_phone"),
    (_CC_LIKE, "account_number", 0.80, "cc_like"),
    (_LONG_ACCOUNT, "account_number", 0.85, "long_account"),
    (_EMAIL, "private_email", 0.95, "email"),
    (_EMAIL_LOOSE, "private_email", 0.85, "email_loose"),
    (_URL, "private_url", 0.90, "url"),
    (_HONORIFIC_NAME, "private_person", 0.90, "honorific_name"),
    (_ALLCAPS_NAME, "private_person", 0.70, "allcaps_name"),
]


def _disabled_set() -> set[str]:
    raw = os.getenv("DISABLED_RULE_DETECTORS", "").strip()
    if not raw:
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _custom_keywords_pattern() -> Pattern[str] | None:
    """Optional escape hatch: literal phrases to always redact.

    Useful for brand acronyms or codenames the NER model can't infer
    from context (e.g. internal project names). Comma-separated,
    case-insensitive, whole-word matched.
    """
    raw = os.getenv("EXTRA_REDACTION_KEYWORDS", "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    parts.sort(key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(p) for p in parts) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def detect(text: str) -> List[Dict[str, object]]:
    """Run all enabled rule-based detectors over ``text``."""
    if not text:
        return []
    if os.getenv("ENABLE_RULE_DETECTORS", "1") != "1":
        return []

    disabled = _disabled_set()
    out: List[Dict[str, object]] = []

    for regex, label, score, name in _DEFAULT_DETECTORS:
        if name in disabled or label in disabled:
            continue
        for m in regex.finditer(text):
            out.append({
                "entity_group": label,
                "score": score,
                "word": m.group(0),
                "start": m.start(),
                "end": m.end(),
                "_source": f"rule:{name}",
            })

    extra = _custom_keywords_pattern()
    if extra is not None and "extra" not in disabled:
        for m in extra.finditer(text):
            out.append({
                "entity_group": "org_name",
                "score": 0.99,
                "word": m.group(0),
                "start": m.start(),
                "end": m.end(),
                "_source": "rule:extra",
            })

    # Labeled-field detector: redact whatever value follows a sensitive label.
    # Picks up things ML/NER miss when OCR garbles the value (e.g. ``Bank Name
    # : HDFC BANK LIMITED`` where the value is just letters but the LABEL
    # tells us it's sensitive).
    if "labeled_field" not in disabled:
        for m in _LABELED_FIELDS.finditer(text):
            value = m.group(1).strip()
            if not value:
                continue
            # Compute the actual char offsets of the *value* group.
            v_start = m.start(1)
            v_end = v_start + len(m.group(1))
            # Trim trailing whitespace from the matched value range.
            stripped = m.group(1).rstrip()
            v_end = v_start + len(stripped)
            # Trim leading whitespace too.
            ls = len(m.group(1)) - len(m.group(1).lstrip())
            v_start += ls
            label_text = m.group(0)[: m.start(1) - m.start(0)]
            tag = "private_person"  # default
            for pat, t in _LABEL_KEYWORD_TAGS:
                if pat.search(label_text):
                    tag = t
                    break
            out.append({
                "entity_group": tag,
                "score": 0.92,
                "word": stripped,
                "start": v_start,
                "end": v_end,
                "_source": "rule:labeled_field",
            })

    out.sort(key=lambda e: (int(e["start"]), int(e["end"])))
    return out
