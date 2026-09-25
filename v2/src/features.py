"""
features.py

Pairwise feature engineering, rewritten against EDA findings.

Why this rewrite:

  - SPEED: the original version refit a fresh TfidfVectorizer for every
    single pair (_tfidf_cosine). At real scale (up to ~112 candidates per
    S1 entity x ~2.2M S1 entities, EDA E20) that's on the order of 100M+
    pair evaluations -- refitting a vectorizer that many times is
    infeasible. blocking.py already computes name/address/combo cosine
    similarity once per pair as a byproduct of candidate generation
    (`channel_scores`); this version reuses those instead of recomputing.

  - ADDRESS IS THE STRONGEST SIGNAL: EDA D16 found only 0.09% of random
    negative pairs have addr_token_sort_ratio >= 0.60, vs. meaningful
    overlap on name-based features for hard negatives in India (9.59% of
    negatives have name_token_set >= 0.60). Address similarity is kept as
    a first-class, heavily-weighted feature set, not an afterthought.

  - COUNTRY-AWARE NORMALIZATION: normalize_address and extract_postal_code
    both now take `country` (see normalize.py) -- features.py passes it
    through so France's "St"-is-not-"Street" and India's absent-PIN-code
    quirks are respected.

  - POSTAL FEATURES ARE NOW LOW-WEIGHT BY CONSTRUCTION: postal codes are
    valid in only 10.93% of US S1 addresses and 0.00% of India (EDA C14),
    so postal_match will be zero/uninformative for the large majority of
    pairs. It's kept as a weak corroborating feature, never the decisive
    one -- LightGBM will naturally down-weight it given how sparse it is,
    but we also don't let it gate anything upstream.
"""

from __future__ import annotations

from typing import Dict, Optional

import rapidfuzz.fuzz as fuzz

from normalize import normalize_address, normalize_name, extract_postal_code, char_ngrams


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _token_jaccard(a: str, b: str) -> float:
    return _jaccard(set(a.split()), set(b.split()))


def compute_pair_features(
    s1_name: str, s1_address: str, s1_country: str,
    other_name: str, other_address: str, other_country: str,
    other_source: str,  # "S2" or "S3"
    channel_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute the feature vector for one (Source1, candidate) pair.

    `channel_scores` should be the {"name":..,"address":..,"combo":..}
    dict blocking.py already computed for this exact pair during
    candidate generation -- pass it through rather than recomputing
    TF-IDF cosine here. If None (e.g. ad-hoc use outside the normal
    pipeline), those three features default to 0.0.

    Country is passed through to normalize_address/extract_postal_code
    since both are country-aware (France's "St" ambiguity, India's absent
    postal codes -- see normalize.py).
    """
    n1, n2 = normalize_name(s1_name), normalize_name(other_name)
    a1 = normalize_address(s1_address, s1_country)
    a2 = normalize_address(other_address, other_country)

    feats: Dict[str, float] = {}

    # --- Reused blocking-stage channel similarities (cheap, already computed) ---
    cs = channel_scores or {}
    feats["channel_name_cosine"] = cs.get("name", 0.0)
    feats["channel_address_cosine"] = cs.get("address", 0.0)
    feats["channel_combo_cosine"] = cs.get("combo", 0.0)

    # --- Name similarity -----------------------------------------------------
    feats["name_exact_match"] = float(n1 == n2 and n1 != "")
    feats["name_levenshtein_ratio"] = fuzz.ratio(n1, n2) / 100.0
    feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(n1, n2) / 100.0
    feats["name_token_set_ratio"] = fuzz.token_set_ratio(n1, n2) / 100.0
    feats["name_partial_ratio"] = fuzz.partial_ratio(n1, n2) / 100.0
    feats["name_token_jaccard"] = _token_jaccard(n1, n2)
    feats["name_char3gram_jaccard"] = _jaccard(char_ngrams(n1, 3), char_ngrams(n2, 3))
    feats["name_len_ratio"] = (
        min(len(n1), len(n2)) / max(len(n1), len(n2)) if n1 and n2 else 0.0
    )

    # --- Address similarity ---------------------------------------------------
    # Strongest discriminative signal per EDA D16 -- kept as a full feature
    # set, not a single summary number.
    feats["addr_levenshtein_ratio"] = fuzz.ratio(a1, a2) / 100.0
    feats["addr_token_sort_ratio"] = fuzz.token_sort_ratio(a1, a2) / 100.0
    feats["addr_token_set_ratio"] = fuzz.token_set_ratio(a1, a2) / 100.0
    feats["addr_token_jaccard"] = _token_jaccard(a1, a2)
    feats["addr_both_present"] = float(bool(a1) and bool(a2))

    # --- Postal (weak, sparse -- see module docstring) ------------------------
    p1 = extract_postal_code(s1_address or "", s1_country)
    p2 = extract_postal_code(other_address or "", other_country)
    feats["postal_both_present"] = float(bool(p1) and bool(p2))
    feats["postal_match"] = float(bool(p1) and p1 == p2)

    # --- Cross / structural -----------------------------------------------------
    feats["name_substring_in_address"] = float(bool(n1) and n1 in a2)
    feats["country_match"] = float(
        bool(s1_country) and s1_country.strip().lower() == (other_country or "").strip().lower()
    )
    feats["is_source3"] = float(other_source == "S3")

    return feats


FEATURE_NAMES = list(
    compute_pair_features(
        "sample name", "sample address", "sample country",
        "sample name", "sample address", "sample country", "S2",
        channel_scores={"name": 0.5, "address": 0.5, "combo": 0.5},
    ).keys()
)
