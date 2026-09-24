"""
normalize.py

Normalization for business_name and business_address fields. This is the
foundation both blocking and feature engineering sit on -- most of the
"noise" described in the challenge PDF (legal suffix variants, abbreviation
variants, punctuation) is cleaned up here once rather than re-handled in
every downstream feature.

Kept deliberately language/script agnostic (no hardcoded country branching)
since the test set introduces France, which never appears in training --
any hardcoded {US, India} logic would silently fail on it.
"""

from __future__ import annotations

import re
import unicodedata

# Common legal-entity suffixes across the US / India / France business
# register naming conventions seen in the challenge description. This list
# is intentionally broad and suffix-based (regex, not exact match) so novel
# variants (e.g. "Pvt. Ltd", "P Ltd", "SARL", "SAS") still collapse.
_LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b("
    r"incorporated|inc|corporation|corp|company|co|"
    r"limited|ltd|llc|llp|lp|"
    r"private|pvt|"
    r"sarl|sas|sa|eurl|"          # French entity types
    r"pty|gmbh|plc"
    r")\.?\b",
    flags=re.IGNORECASE,
)

_ADDR_ABBREV_MAP = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bste\b": "suite",
    r"\bno\b": "number",
    r"\bnr\b": "near",
}

_PUNCT_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_PATTERN = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    """
    Fold accented characters to ASCII equivalents (e.g. Renee vs Renée,
    Muller vs Müller). Helps with transliteration variants across sources
    and is script-agnostic enough to be safe for French entries too.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_name(name: str) -> str:
    """
    Normalize a business_name for comparison:
      1. lowercase + accent-fold
      2. '&' -> 'and'
      3. strip legal suffixes (Corp/Ltd/Pvt/SARL/...)
      4. strip punctuation
      5. collapse whitespace

    Returns the cleaned string. Callers needing the *original* name for
    display/output should keep the raw column separately -- this function
    is for comparison only.
    """
    if not name:
        return ""
    text = _strip_accents(name).lower()
    text = text.replace("&", " and ")
    text = _LEGAL_SUFFIX_PATTERN.sub(" ", text)
    text = _PUNCT_PATTERN.sub(" ", text)
    text = _WS_PATTERN.sub(" ", text).strip()
    return text


def normalize_address(address: str) -> str:
    """
    Normalize a business_address for comparison:
      1. lowercase + accent-fold
      2. expand common abbreviations (Rd -> road, St -> street, ...)
      3. strip punctuation
      4. collapse whitespace

    Landmark phrases ("near sbi atm") are intentionally left intact rather
    than stripped -- they're weak but nonzero signal, handled downstream as
    a soft feature rather than filtered here.
    """
    if not address:
        return ""
    text = _strip_accents(address).lower()
    text = _PUNCT_PATTERN.sub(" ", text)
    for pattern, repl in _ADDR_ABBREV_MAP.items():
        text = re.sub(pattern, repl, text)
    text = _WS_PATTERN.sub(" ", text).strip()
    return text


def extract_postal_code(address: str) -> str:
    """
    Heuristically pull a trailing numeric postal/PIN code out of an address
    string, if present. Pure regex on the provided text -- no geocoding
    API, no external lookup (both are prohibited by the challenge rules).
    Returns "" when nothing plausible is found.
    """
    if not address:
        return ""
    # 5-6 digit runs, optionally with a space/hyphen in the middle (covers
    # US ZIP, US ZIP+4, and Indian 6-digit PIN codes).
    match = re.search(r"\b(\d{5,6}(?:[-\s]\d{3,4})?)\b", address)
    return match.group(1) if match else ""


def char_ngrams(text: str, n: int = 3) -> set:
    """Return the set of character n-grams for a normalized string."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}
