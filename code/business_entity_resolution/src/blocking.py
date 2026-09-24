"""
blocking.py

Candidate generation stage. This determines the recall ceiling of the
whole pipeline, so it deliberately combines two independent signals rather
than relying on a single blocking key:

  1. Char n-gram TF-IDF cosine similarity (via nearest-neighbor search) --
     robust to typos, abbreviation differences, and transliteration/word
     transpositions, and script-agnostic enough to generalize to the
     unseen France test records with no special-casing.
  2. Sorted-neighborhood on the normalized name prefix -- a cheap recall
     backstop that catches near-duplicates the vectorizer might rank just
     outside top-K.

country is deliberately NOT used as a hard filter (see challenge rules:
treat it as an open-set label, and a noisy/missing country field should
not silently drop a true match). It's left available as a soft feature
for the matching stage instead.

Output of this stage (per Source-1 entity, a list of Source2/Source3
candidate ids) is exactly what gets written to candidate_pairs.tsv.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from normalize import normalize_name


def _char_ngram_analyzer(text: str, n_lo: int = 3, n_hi: int = 5):
    """TF-IDF analyzer emitting char n-grams of sizes n_lo..n_hi."""
    grams = []
    for n in range(n_lo, n_hi + 1):
        if len(text) < n:
            continue
        grams.extend(text[i:i + n] for i in range(len(text) - n + 1))
    return grams


def _build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=lambda t: _char_ngram_analyzer(t, 3, 5),
        min_df=1,
    )


def _top_k_neighbors(query_matrix, pool_matrix, k: int) -> np.ndarray:
    """
    Return the indices (into the pool) of the top-k nearest neighbors for
    every row in query_matrix, using cosine distance via sklearn's
    NearestNeighbors (brute force w/ cosine metric -- swap for FAISS if the
    pool size makes this too slow on the real dataset).
    """
    k = min(k, pool_matrix.shape[0])
    if k == 0:
        return np.empty((query_matrix.shape[0], 0), dtype=int)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(pool_matrix)
    _, indices = nn.kneighbors(query_matrix)
    return indices


def _sorted_neighborhood(s1_names: List[str], pool_names: List[str],
                          window: int = 5) -> List[List[int]]:
    """
    Cheap recall backstop: sort both S1 names and the candidate pool by
    normalized-name prefix, then for each S1 entity take pool entries
    within a small window of its sorted position. Catches near-duplicates
    that fall just outside the TF-IDF top-K.

    Returns, per S1 entity (in original order), a list of pool indices.
    """
    pool_order = sorted(range(len(pool_names)), key=lambda i: pool_names[i])
    pool_sorted_names = [pool_names[i] for i in pool_order]

    import bisect
    results = []
    for name in s1_names:
        pos = bisect.bisect_left(pool_sorted_names, name)
        lo = max(0, pos - window)
        hi = min(len(pool_sorted_names), pos + window)
        results.append([pool_order[j] for j in range(lo, hi)])
    return results


def generate_candidates(
    source1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    top_k: int = 20,
    sorted_neighborhood_window: int = 5,
) -> Dict[str, List[str]]:
    """
    Generate candidate matches from `other_df` (either the Source 2 or
    Source 3 pool) for every entity in `source1_df`.

    Returns {source1_entity_id: [candidate_entity_id, ...]} -- union of the
    TF-IDF nearest-neighbor set and the sorted-neighborhood set, deduped.

    Call this once per (Source1, Source2) and once per (Source1, Source3)
    and merge the two result dicts (see merge_candidate_dicts below) to get
    the full candidate set required by candidate_pairs.tsv.
    """
    if len(other_df) == 0 or len(source1_df) == 0:
        return {row["entity_id"]: [] for _, row in source1_df.iterrows()}

    s1_norm = source1_df["business_name"].map(normalize_name).tolist()
    other_norm = other_df["business_name"].map(normalize_name).tolist()

    vectorizer = _build_vectorizer()
    # Fit jointly so both matrices live in the same feature space.
    combined = vectorizer.fit_transform(s1_norm + other_norm)
    s1_matrix = combined[: len(s1_norm)]
    other_matrix = combined[len(s1_norm):]

    nn_indices = _top_k_neighbors(s1_matrix, other_matrix, top_k)
    sn_indices = _sorted_neighborhood(s1_norm, other_norm,
                                       sorted_neighborhood_window)

    other_ids = other_df["entity_id"].tolist()
    s1_ids = source1_df["entity_id"].tolist()

    out: Dict[str, List[str]] = {}
    for i, s1_id in enumerate(s1_ids):
        idx_set = set(nn_indices[i].tolist()) | set(sn_indices[i])
        out[s1_id] = [other_ids[j] for j in sorted(idx_set)]
    return out


def merge_candidate_dicts(*dicts: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Union candidate lists (e.g. Source2 candidates + Source3 candidates) per S1 id."""
    merged: Dict[str, List[str]] = {}
    for d in dicts:
        for s1_id, cand_ids in d.items():
            merged.setdefault(s1_id, [])
            existing = set(merged[s1_id])
            merged[s1_id].extend(c for c in cand_ids if c not in existing)
    return merged


def measure_recall_ceiling(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> float:
    """
    Diagnostic: of all true matches in ground_truth, what fraction actually
    appear in the blocking candidate set? This is the hard ceiling on
    achievable recall for the matching stage -- run this on your
    validation split BEFORE tuning the matcher. If it's capping out low,
    raise top_k / widen the sorted-neighborhood window rather than trying
    to fix it with a better classifier.
    """
    total_true = 0
    total_found = 0
    for s1_id, true_ids in ground_truth.items():
        cand_set = set(candidates.get(s1_id, []))
        for t in true_ids:
            total_true += 1
            if t in cand_set:
                total_found += 1
    return total_found / total_true if total_true else 1.0
