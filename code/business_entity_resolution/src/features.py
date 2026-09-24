"""
features.py

Pairwise feature engineering for (Source1 record, candidate record) pairs.
Every feature here is computed purely from the two text fields provided in
the dataset -- no external lookups, no geocoding APIs (both prohibited by
the challenge rules). Features are grouped into: name similarity, address
similarity, postal/component agreement, and structural/meta features.

Output is a flat dict of floats per pair, which train.py / infer.py stack
into a DataFrame for LightGBM.
"""

from __future__ import annotations

from typing import Dict

import rapidfuzz.fuzz as fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from normalize import normalize_address, normalize_name, extract_postal_code, char_ngrams


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _token_jaccard(a: str, b: str) -> float:
    return _jaccard(set(a.split()), set(b.split()))


def _tfidf_cosine(a: str, b: str) -> float:
    """
    Single-pair TF-IDF cosine on word tokens. Cheap since it's just 2 docs.
    Falls back to 0.0 for degenerate inputs (empty strings, or strings with
    no tokens sklearn's default word-boundary pattern picks up, e.g. pure
    single characters) rather than raising -- those are legitimately
    "no similarity signal" pairs.
    """
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit([a, b])
        mat = vec.transform([a, b])
        return float(cosine_similarity(mat[0], mat[1])[0][0])
    except ValueError:
        return 0.0


def compute_pair_features(
    s1_name: str, s1_address: str, s1_country: str,
    other_name: str, other_address: str, other_country: str,
    other_source: str,  # "S2" or "S3" -- from entity_id prefix
) -> Dict[str, float]:
    """
    Compute the full feature vector for one (Source1, candidate) pair.

    Args are the RAW (un-normalized) fields; normalization happens inside
    so callers can just pass the source columns directly.
    """
    n1, n2 = normalize_name(s1_name), normalize_name(other_name)
    a1, a2 = normalize_address(s1_address), normalize_address(other_address)

    feats: Dict[str, float] = {}

    # --- Name similarity ---------------------------------------------------
    feats["name_exact_match"] = float(n1 == n2 and n1 != "")
    feats["name_levenshtein_ratio"] = fuzz.ratio(n1, n2) / 100.0
    feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(n1, n2) / 100.0
    feats["name_token_set_ratio"] = fuzz.token_set_ratio(n1, n2) / 100.0
    feats["name_partial_ratio"] = fuzz.partial_ratio(n1, n2) / 100.0
    feats["name_token_jaccard"] = _token_jaccard(n1, n2)
    feats["name_char3gram_jaccard"] = _jaccard(char_ngrams(n1, 3), char_ngrams(n2, 3))
    feats["name_tfidf_cosine"] = _tfidf_cosine(n1, n2)
    feats["name_len_ratio"] = (
        min(len(n1), len(n2)) / max(len(n1), len(n2)) if n1 and n2 else 0.0
    )

    # --- Address similarity -------------------------------------------------
    feats["addr_levenshtein_ratio"] = fuzz.ratio(a1, a2) / 100.0
    feats["addr_token_sort_ratio"] = fuzz.token_sort_ratio(a1, a2) / 100.0
    feats["addr_token_set_ratio"] = fuzz.token_set_ratio(a1, a2) / 100.0
    feats["addr_token_jaccard"] = _token_jaccard(a1, a2)
    feats["addr_tfidf_cosine"] = _tfidf_cosine(a1, a2)

    # --- Postal / component agreement (heuristic regex extraction only) ----
    p1, p2 = extract_postal_code(s1_address or ""), extract_postal_code(other_address or "")
    feats["postal_both_present"] = float(bool(p1) and bool(p2))
    feats["postal_match"] = float(bool(p1) and p1 == p2)

    # --- Cross / structural --------------------------------------------------
    feats["name_substring_in_address"] = float(bool(n1) and n1 in a2)
    feats["country_match"] = float(
        bool(s1_country) and s1_country.strip().lower() == (other_country or "").strip().lower()
    )
    feats["country_either_missing"] = float(not s1_country or not other_country)
    feats["is_source3"] = float(other_source == "S3")  # S2 vs S3 noise profiles may differ

    return feats


FEATURE_NAMES = list(
    compute_pair_features(
        "sample name", "sample address", "sample country",
        "sample name", "sample address", "sample country", "S2",
    ).keys()
)
