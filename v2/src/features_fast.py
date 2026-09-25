"""
features_fast.py

Pairwise features, computed in batch.

The existing features.py calls rapidfuzz once per pair inside a Python
loop. At ~100M candidate pairs that does not finish. This module computes
each metric across every pair at once with rapidfuzz.process.cpdist, which
runs in parallel C++: measured at 7 metrics over 3.0M pairs in 18 seconds.

It also reuses the similarity and rank columns blocking already produced,
rather than recomputing any TF-IDF cosine here.

Input is the long-form candidate table blocking cached to disk, so no
re-blocking happens at train or inference time.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from normalize import (normalize_address, normalize_name, despace,
                       extract_digit_runs)

# Columns blocking already computed. Listed here so the model sees them
# alongside the string metrics; rank and margin are usually stronger than
# any individual similarity.
BLOCKING_FEATURES = [
    "sim_name", "sim_address", "rank_name", "rank_address",
    "sim_best", "cand_count", "sim_max_for_s1", "sim_margin",
]

STRING_FEATURES = [
    "n_ratio", "n_token_sort", "n_token_set", "n_partial", "n_despace",
    "a_ratio", "a_token_sort", "a_token_set",
    "n_jaccard", "a_jaccard", "digit_jaccard", "digit_any",
    "n_exact", "a_both_present", "n_len_ratio",
    "nonlatin_name", "is_source3",
]

FEATURE_NAMES = BLOCKING_FEATURES + STRING_FEATURES


def _jaccard_sets(left: List[set], right: List[set]) -> np.ndarray:
    return np.array(
        [len(a & b) / len(a | b) if (a or b) else 0.0
         for a, b in zip(left, right)],
        dtype=np.float32,
    )


def _is_latin(text: str) -> bool:
    """Cheap script check. Source 1 is always Latin; ~7% of Source 2/3 is
    Devanagari, Gurmukhi or Tamil, and for those pairs every name-based
    metric is structurally meaningless -- the model needs to know that so
    it can lean on the address instead of reading 0.0 as evidence of a
    mismatch."""
    return all(ord(ch) < 0x0900 for ch in text if ch.isalpha())


def build_features(cand_df: pd.DataFrame, s1_df: pd.DataFrame,
                   pool_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add feature columns to the cached candidate table.

    cand_df: the long-form table from blocking (candidates_*.pkl)
    s1_df:   Source 1 records
    pool_df: Source 2 and Source 3 concatenated
    """
    df = cand_df.copy()

    def _prep(frame: pd.DataFrame) -> dict:
        names = [normalize_name(n, c) for n, c in
                 zip(frame["business_name"], frame["country"])]
        addrs = [normalize_address(a, c) for a, c in
                 zip(frame["business_address"], frame["country"])]
        return dict(zip(frame["entity_id"],
                        zip(names, addrs, frame["business_address"],
                            frame["business_name"])))

    s1_map = _prep(s1_df)
    pool_map = _prep(pool_df)
    if verbose:
        print(f"[features] normalized {len(s1_map):,} S1 + {len(pool_map):,} pool")

    left = [s1_map[i] for i in df["source1_entity_id"]]
    right = [pool_map[i] for i in df["candidate_entity_id"]]

    ln = [x[0] for x in left]
    rn = [x[0] for x in right]
    la = [x[1] for x in left]
    ra = [x[1] for x in right]

    # Batched across all pairs at once -- this is the whole point of the
    # module. workers=-1 uses every core.
    for col, scorer, a, b in [
        ("n_ratio", fuzz.ratio, ln, rn),
        ("n_token_sort", fuzz.token_sort_ratio, ln, rn),
        ("n_token_set", fuzz.token_set_ratio, ln, rn),
        ("n_partial", fuzz.partial_ratio, ln, rn),
        ("a_ratio", fuzz.ratio, la, ra),
        ("a_token_sort", fuzz.token_sort_ratio, la, ra),
        ("a_token_set", fuzz.token_set_ratio, la, ra),
    ]:
        df[col] = process.cpdist(a, b, scorer=scorer, workers=-1) / 100.0

    # Concatenated-name form: 5.37% of true-match partner names are domains
    # ("energyvrtextile.com" for "Energy Vr Textile Corporation"), which no
    # token-based metric can see.
    df["n_despace"] = process.cpdist(
        [despace(x) for x in ln], [despace(x) for x in rn],
        scorer=fuzz.ratio, workers=-1) / 100.0
    if verbose:
        print(f"[features] string metrics done ({len(df):,} pairs)")

    df["n_jaccard"] = _jaccard_sets([set(x.split()) for x in ln],
                                    [set(x.split()) for x in rn])
    df["a_jaccard"] = _jaccard_sets([set(x.split()) for x in la],
                                    [set(x.split()) for x in ra])

    # Digit runs replace postal codes: postal extraction fires on nothing in
    # this dataset, while 79.46% of true pairs share a digit run.
    d_left = [extract_digit_runs(x[2]) for x in left]
    d_right = [extract_digit_runs(x[2]) for x in right]
    df["digit_jaccard"] = _jaccard_sets(d_left, d_right)
    df["digit_any"] = np.array([1.0 if (a & b) else 0.0
                                for a, b in zip(d_left, d_right)],
                               dtype=np.float32)

    df["n_exact"] = np.array([1.0 if (a == b and a) else 0.0
                              for a, b in zip(ln, rn)], dtype=np.float32)
    df["a_both_present"] = np.array([1.0 if (a and b) else 0.0
                                     for a, b in zip(la, ra)], dtype=np.float32)
    df["n_len_ratio"] = np.array(
        [min(len(a), len(b)) / max(len(a), len(b)) if (a and b) else 0.0
         for a, b in zip(ln, rn)], dtype=np.float32)
    df["nonlatin_name"] = np.array([0.0 if _is_latin(x[3]) else 1.0
                                    for x in right], dtype=np.float32)
    df["is_source3"] = df["candidate_entity_id"].str.startswith("S3").astype(np.float32)

    if verbose:
        print(f"[features] {len(FEATURE_NAMES)} features on {len(df):,} pairs")
    return df


def add_labels(df: pd.DataFrame, ground_truth: dict) -> pd.DataFrame:
    """label = 1 when the candidate is a true match for that S1 entity."""
    truth = {k: set(v) for k, v in ground_truth.items()}
    df["label"] = [1 if c in truth.get(s, ()) else 0
                   for s, c in zip(df["source1_entity_id"],
                                   df["candidate_entity_id"])]
    return df