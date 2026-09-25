"""
normalize.py

Text normalization, rewritten against EDA findings (see EDA_REPORT.md
Section C, G4):

  - Legal-suffix stripping must anchor to the END of the string only.
    "co", "sa", "associates", "group" are common mid-string words in real
    business names (the EDA found "associates"/"group" appearing almost
    as often mid-string as trailing in the US sample) -- stripping them
    anywhere would corrupt real names, not just clean noise.
  - Suffix stripping alone converts 18.9%-20.8% of true pairs into exact
    matches (EDA C15) -- worth getting right.
  - India: landmark prefixes ("near", "opp", "behind") precede the useful
    part of ~7-9% of true-pair addresses (EDA C12/C13) and should be
    stripped as a prefix, not left to dilute similarity scores.
  - France: "St" is ambiguous -- "Saint" in French, "Street" in English.
    Address abbreviation expansion must be country-aware, not global.
  - Postal code extraction must skip India entirely (0.00% of Indian S1
    addresses contain any PIN code -- EDA C14) and must anchor to the END
    of the string for US (9.3% of US addresses start with a 5-digit
    street number that a naive regex misreads as a ZIP -- EDA C14).
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# Legal suffix stripping -- anchored to end-of-string only
# --------------------------------------------------------------------------- #

# Suffix inventory extended per EDA C15 / F21: US, India, and France (new in
# test, unseen in train -- SARL/SAS/EURL/SA/SCI cover 63.3% of French names).
_LEGAL_SUFFIXES = [
    # US
    "incorporated", "inc", "corporation", "corp", "company", "co",
    "limited", "ltd", "llc", "llp", "lp",
    # India
    "private", "pvt", "pvt ltd", "pvt. ltd", "p ltd",
    # France
    "sarl", "sas", "eurl", "sa", "sci",
    # generic / other regions seen in noisy sources
    "pty", "gmbh", "plc",
]
# Longest-first so "pvt ltd" matches before the bare "ltd" fallback would.
_LEGAL_SUFFIXES.sort(key=len, reverse=True)

# Anchored with $ (end of string, since we match against an already-stripped
# string) -- this is the fix: the old pattern used \b...\b with no end
# anchor, so "co" or "sa" appearing mid-name (e.g. "Coastal Sailing Co-op")
# would get silently mangled. Only a suffix that is the LAST token(s) is
# stripped.
_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _LEGAL_SUFFIXES) + r")\.?\s*$",
    flags=re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Address landmark prefixes (India-heavy pattern, EDA C12/C13)
# --------------------------------------------------------------------------- #

_LANDMARK_PREFIX_PATTERN = re.compile(
    r"^\s*(near|opp\.?|opposite|behind|beside|next to)\s+",
    flags=re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Address abbreviation expansion -- country-aware (France's "St" ambiguity)
# --------------------------------------------------------------------------- #

# Safe everywhere: unambiguous abbreviations.
_ADDR_ABBREV_MAP_COMMON = {
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bste\b": "suite",
    r"\bnr\b": "near",
}

# English-only: "St" = "Street" is wrong in French ("St-Honore" = "Saint
# Honore"). Only applied when country is NOT France.
_ADDR_ABBREV_MAP_ENGLISH_ONLY = {
    r"\bst\b": "street",
    r"\bno\b": "number",
}

_PUNCT_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_PATTERN = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    """Fold accented characters to ASCII (Café -> Cafe, Müller -> Muller)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_name(name: str) -> str:
    """
    Normalize a business_name for comparison:
      1. lowercase + accent-fold
      2. '&' -> 'and'
      3. strip punctuation
      4. strip a trailing legal suffix (end-anchored only -- see module
         docstring; this must run AFTER punctuation stripping so "Pvt.
         Ltd." and "Pvt Ltd" normalize identically)
      5. collapse whitespace
    """
    if not name:
        return ""
    text = _strip_accents(name).lower()
    text = text.replace("&", " and ")
    text = _PUNCT_PATTERN.sub(" ", text)
    text = _WS_PATTERN.sub(" ", text).strip()
    text = _SUFFIX_PATTERN.sub("", text).strip()
    return text


def normalize_address(address: str, country: str = "") -> str:
    """
    Normalize a business_address for comparison:
      1. lowercase + accent-fold
      2. strip a leading landmark phrase ("Near SBI ATM, ..." -> "sbi atm, ...")
      3. expand common (country-agnostic) abbreviations
      4. expand English-only abbreviations (Rd/St/No) UNLESS country is
         France, where "St" means Saint, not Street
      5. strip punctuation
      6. collapse whitespace

    `country` is optional but should be passed whenever known -- omitting
    it defaults to the (safe for US/India) English abbreviation behavior,
    which would be WRONG for French addresses.
    """
    if not address:
        return ""
    text = _strip_accents(address).lower()
    text = _LANDMARK_PREFIX_PATTERN.sub("", text)

    for pattern, repl in _ADDR_ABBREV_MAP_COMMON.items():
        text = re.sub(pattern, repl, text)

    is_france = country.strip().lower() == "france"
    if not is_france:
        for pattern, repl in _ADDR_ABBREV_MAP_ENGLISH_ONLY.items():
            text = re.sub(pattern, repl, text)

    text = _PUNCT_PATTERN.sub(" ", text)
    text = _WS_PATTERN.sub(" ", text).strip()
    return text


def extract_postal_code(address: str, country: str = "") -> str:
    """
    Heuristically pull a trailing postal code out of an address, if
    present. Regex only -- no geocoding API (prohibited by the challenge).

    Per EDA C14:
      - India: PIN codes are absent from 100% of S1 addresses. Return ""
        unconditionally for India rather than risk a spurious match on
        some other 6-digit number in the string.
      - US: anchor to the END of the string. 9.3% of US addresses START
        with a 5-digit street number ("12345 Elm St") which a naive
        unanchored regex would misread as a ZIP.
      - France: postal code (5 digits) typically precedes the city name,
        but is missing in 99.6% of test addresses -- treat as a weak
        signal only, this rarely fires.
    """
    if not address:
        return ""
    if country.strip().lower() == "india":
        return ""

    # End-anchored: 5-digit US ZIP (optionally +4), allowing trailing
    # punctuation/whitespace after it.
    match = re.search(r"\b(\d{5}(?:-\d{4})?)\s*$", address.strip())
    if match:
        return match.group(1)

    # France only: postal code (5 digits) typically precedes the city name
    # (e.g. "75008 Paris"), rather than trailing the string like a US ZIP.
    # Restricted to France specifically -- applying this pattern to US
    # addresses would reintroduce the exact street-number-as-ZIP confusion
    # (EDA C14) this function exists to avoid, since a US street number is
    # also "5 digits followed by a capitalized word" (e.g. "12345 Elm").
    if country.strip().lower() == "france":
        match = re.search(r"\b(\d{5})\b(?=\s+[A-ZÀ-Ý])", address)
        if match:
            return match.group(1)

    return ""


def char_ngrams(text: str, n: int = 3) -> set:
    """Return the set of character n-grams for a normalized string."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}
