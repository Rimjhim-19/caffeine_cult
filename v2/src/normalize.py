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
  - India: landmark markers ("near", "opp", "behind") sit next to the
    useful part of ~7-9% of true-pair addresses (EDA C12/C13) and should
    be dropped rather than left to dilute similarity scores.
  - France: "St" is ambiguous -- "Saint" in French, "Street" in English.
    Address abbreviation expansion must be country-aware, not global.
  - Postal code extraction must skip India entirely (0.00% of Indian S1
    addresses contain any PIN code -- EDA C14) and must anchor to the END
    of the string for US (9.3% of US addresses start with a 5-digit
    street number that a naive regex misreads as a ZIP -- EDA C14).

CHANGES FROM THE PREVIOUS VERSION -- each measured on 11,560 sampled true
training pairs. Exact-name-match rate went 40.92% -> 48.61%.

  1. Suffix stripping is now ITERATIVE. One pass strips only the last
     token, so "Pioneer Industries Private Limited" became "pioneer
     industries private" and never collided with "Pioneer Industries".
     Measured: 859 of 11,560 pairs carried a stranded suffix. Adding
     "private limited" to the list would not have fixed this -- the
     pattern is $-anchored, so only one match per call is possible
     regardless of what the list contains.
  2. Stripping NEVER returns empty. "Ltd", "Limited" and "SA" previously
     normalized to "". An empty name scores fuzz.ratio == 100 and Jaccard
     == 1.0 against every other empty name, so those records merge with
     each other. Under F_0.5 a false merge costs roughly double a miss.
  3. despace() added, worth +3.3 points on its own. 5.37% of true-match
     partner names are concatenated domains: "energyvrtextile.com" for
     "Energy Vr Textile Corporation". TLDs are now stripped BEFORE
     punctuation removal (a TLD is only recognisable while the dot
     survives), and the despaced form makes the concatenation comparable.
  4. Junk prefixes ("M/s", "Mr", leading dashes) stripped -- common in
     Source 2/3.
  5. Landmark markers dropped ANYWHERE, not just at string start.
     Measured position: 371 mid-string vs 5 at prefix. Only the marker
     word goes; the landmark text stays, since both sides of a true pair
     usually carry the same landmark and it is real shared signal.

Also note: extract_postal_code fires on NOTHING in this dataset -- 0
extractions across 11,560 sampled records (0 of 7,053 US, 0 of 4,507
India), because US addresses here end with a state rather than a ZIP. It
is kept for France and for safety, but do not build features that assume
it returns anything. extract_digit_runs is the numeric signal that works:
79.46% of true pairs share a digit run.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Set

# --------------------------------------------------------------------------- #
# Legal suffix stripping -- anchored to end-of-string, applied repeatedly
# --------------------------------------------------------------------------- #

# Single tokens only. Compound suffixes ("pvt ltd", "private limited") fall
# out of the iterative loop below and need no separate entries.
_LEGAL_SUFFIXES = frozenset({
    # US
    "incorporated", "inc", "corporation", "corp", "company", "co",
    "limited", "ltd", "llc", "llp", "lp",
    # India
    "private", "pvt",
    # France (unseen in train; SARL/SAS/EURL/SA/SCI cover 63.3% of French names)
    "sarl", "sas", "eurl", "sa", "sci", "sasu", "snc",
    # generic / other regions seen in noisy sources
    "pty", "gmbh", "plc", "bv", "nv", "ag",
})

# "Foo Pvt Ltd Co" is already pathological; this only guards a runaway loop.
_MAX_SUFFIX_STRIPS = 4

# Junk prefixes observed in S2/S3: leading dashes, honorifics, "M/s".
_JUNK_PREFIX_PATTERN = re.compile(
    r"^\s*(?:[-\u2013\u2014]+\s*|m/s\.?\s+|mr\.?\s+|mrs\.?\s+|ms\.?\s+|dr\.?\s+)+",
    flags=re.IGNORECASE,
)

# Stripped BEFORE punctuation removal, otherwise ".com" survives as a bare
# "com" token that the suffix list does not cover.
_TLD_PATTERN = re.compile(
    r"\.(?:com|net|org|info|biz|io|co\.in|co\.uk|in|us|fr|edu|gov)\b",
    flags=re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Address landmark markers -- dropped anywhere, marker word only
# --------------------------------------------------------------------------- #

_LANDMARK_MARKER_PATTERN = re.compile(
    r"\b(?:near|nr|opp\.?|opposite|behind|beside|next\s+to|adjacent\s+to)\b",
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
    r"\bhwy\b": "highway",
    r"\bln\b": "lane",
}

# English-only: "St" = "Street" is wrong in French ("St-Honore" = "Saint
# Honore"). Only applied when country is NOT France.
_ADDR_ABBREV_MAP_ENGLISH_ONLY = {
    r"\bst\b": "street",
    r"\bno\b": "number",
}

# In French, "St" means "Saint", not "Street". Kept separate from the
# English expansion so the two country-specific meanings cannot leak.
# NOTE: re.VERBOSE was removed here -- under VERBOSE the regex engine
# ignores the literal space in the pattern, so it was not matching what the
# source reads as matching. It happened to still work via the lookahead,
# but only by accident.
_FRANCE_ST_PATTERN = re.compile(r"\bst\.?(?=\W|$)", flags=re.IGNORECASE)

_PUNCT_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_PATTERN = re.compile(r"\s+")
_DIGIT_RUN_PATTERN = re.compile(r"\d+")


def _strip_accents(text: str) -> str:
    """Fold accented characters to ASCII (Cafe -> Cafe, Muller -> Muller)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _strip_trailing_suffixes(text: str) -> str:
    """Remove trailing legal-suffix tokens, repeatedly.

    Returns the input unchanged if stripping would empty it: a name that is
    nothing but suffixes ("Ltd", "Limited", "SA") must not normalize to ""
    -- see change 2 in the module docstring.
    """
    tokens = text.split()
    for _ in range(_MAX_SUFFIX_STRIPS):
        if len(tokens) <= 1:
            break
        if tokens[-1] in _LEGAL_SUFFIXES:
            tokens = tokens[:-1]
        else:
            break
    return " ".join(tokens) if tokens else text


def normalize_name(name: str, country: str = "") -> str:
    """
    Normalize a business_name for comparison:
      1. strip junk prefixes ("M/s", "Mr", leading dashes)
      2. strip domain TLDs (while the dot still survives)
      3. lowercase + accent-fold
      4. '&' -> 'and'
      5. for France only, expand standalone "St" / "St." to "Saint"
      6. strip punctuation
      7. strip trailing legal suffixes, iteratively, never to empty
      8. collapse whitespace
    """
    if not name:
        return ""
    text = _JUNK_PREFIX_PATTERN.sub(" ", name)
    text = _TLD_PATTERN.sub(" ", text)
    text = _strip_accents(text).lower()
    text = text.replace("&", " and ")
    if country.strip().lower() == "france":
        text = _FRANCE_ST_PATTERN.sub("saint", text)
    text = _PUNCT_PATTERN.sub(" ", text)
    text = _WS_PATTERN.sub(" ", text).strip()
    return _strip_trailing_suffixes(text)


def despace(text: str) -> str:
    """Whitespace-free form, for comparing concatenated names to spaced ones.

    "energyvrtextile" (from energyvrtextile.com) and "energy vr textile"
    are the same business; despacing both makes them string-identical.
    """
    return text.replace(" ", "")


def normalize_address(address: str, country: str = "") -> str:
    """
    Normalize a business_address for comparison:
      1. lowercase + accent-fold
      2. drop landmark markers anywhere ("Near SBI ATM" -> "sbi atm")
      3. expand common (country-agnostic) abbreviations
      4. expand English-only abbreviations (St/No) UNLESS country is
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
    text = _LANDMARK_MARKER_PATTERN.sub(" ", text)

    for pattern, repl in _ADDR_ABBREV_MAP_COMMON.items():
        text = re.sub(pattern, repl, text)

    if country.strip().lower() == "france":
        text = _FRANCE_ST_PATTERN.sub("saint", text)
    else:
        for pattern, repl in _ADDR_ABBREV_MAP_ENGLISH_ONLY.items():
            text = re.sub(pattern, repl, text)

    text = _PUNCT_PATTERN.sub(" ", text)
    return _WS_PATTERN.sub(" ", text).strip()


def extract_digit_runs(address: str) -> Set[str]:
    """All maximal digit runs in an address.

    This is the numeric signal that actually exists in this data. Postal
    codes do not (0 extractions across 11,560 sampled records), but
    building, plot and door numbers are common and agree on 79.46% of true
    pairs. Returned as a set so callers can take overlap or Jaccard.
    """
    if not address:
        return set()
    return set(_DIGIT_RUN_PATTERN.findall(address))


def extract_postal_code(address: str, country: str = "") -> str:
    """
    Heuristically pull a trailing postal code out of an address, if
    present. Regex only -- no geocoding API (prohibited by the challenge).

    WARNING: this fires on essentially nothing in this dataset -- it
    returned "" for 100% of sampled records. Prefer extract_digit_runs.

    Per EDA C14:
      - India: PIN codes are absent from 100% of S1 addresses. Return ""
        unconditionally rather than risk a spurious match on some other
        6-digit number in the string.
      - US: anchor to the END of the string. 9.3% of US addresses START
        with a 5-digit street number ("12345 Elm St") which a naive
        unanchored regex would misread as a ZIP.
      - France: postal code (5 digits) typically precedes the city name.
    """
    if not address:
        return ""
    if country.strip().lower() == "india":
        return ""

    match = re.search(r"\b(\d{5}(?:-\d{4})?)\s*$", address.strip())
    if match:
        return match.group(1)

    if country.strip().lower() == "france":
        match = re.search(r"\b(\d{5})\b(?=\s+[A-Za-z])", address)
        if match:
            return match.group(1)

    return ""


def char_ngrams(text: str, n: int = 3) -> set:
    """Return the set of character n-grams for a normalized string."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}