"""
blocking.py

Candidate generation, rewritten against EDA findings.

Why this rewrite (see EDA_REPORT.md Sections E and G):

  - SCALE: train alone is ~2.2M S1 x ~5M S2 x ~5M S3 records. The original
    version used sklearn's NearestNeighbors(algorithm="brute"), which
    materializes dense distance computations -- infeasible at this size.
    This version uses sparse TF-IDF matrix multiplication (sparse x sparse
    stays sparse, since two records only "overlap" on shared n-grams) plus
    a partial top-K selection per row, which is what the EDA's own
    blocking benchmark (Section E17) used and validated.

  - COUNTRY PARTITIONING: EDA B9 found ZERO cross-country true matches in
    7.64M ground-truth pairs (100.000000% country consistency). Blocking
    now partitions by country FIRST and only searches within the same
    country -- this is both a correctness-safe recall improvement (no risk
    of dropping a true match, since none exist across countries) and a
    ~2.5x reduction in search space per partition.

  - MULTI-CHANNEL: EDA E18 found name-only TF-IDF blocking recovers only
    61.9%-74.8% of true India matches even at K=50 (India has heavy
    DBA/trade-name divergence from the registered name -- EDA C12/C13,
    D16). The union of name-only + address-only + combined(name+address)
    channels reaches 97.84%-100% recall at K=10. This version always
    builds and unions all three channels.

Candidates returned here go straight into candidate_pairs.tsv, and the
per-channel similarity scores are also returned so features.py can reuse
them as features instead of recomputing TF-IDF per pair (too slow at this
scale -- see features.py docstring).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from normalize import normalize_address, normalize_name

CHANNELS = ("name", "address", "combo")


def _char_ngram_analyzer(text: str, n_lo: int = 3, n_hi: int = 5):
    grams = []
    for n in range(n_lo, n_hi + 1):
        if len(text) < n:
            continue
        grams.extend(text[i:i + n] for i in range(len(text) - n + 1))
    return grams


def _build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer=lambda t: _char_ngram_analyzer(t, 3, 5), min_df=1)


def _prepare_channel_texts(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build the three normalized text channels for a source dataframe.
    Assumes df has business_name, business_address, country columns.
    """
    names = [
        normalize_name(n) for n in df["business_name"].tolist()
    ]
    addrs = [
        normalize_address(a, c)
        for a, c in zip(df["business_address"].tolist(), df["country"].tolist())
    ]
    combos = [f"{n} {a}".strip() for n, a in zip(names, addrs)]
    return {"name": names, "address": addrs, "combo": combos}


def _sparse_top_k_per_row(sim: sp.csr_matrix, k: int) -> List[np.ndarray]:
    """
    For each row of a sparse similarity matrix, return the column indices
    of its top-k highest values (unsorted within the top-k, which is fine
    -- we only need set membership for candidate generation).

    Only touches each row's actual nonzero entries (the whole point of
    keeping this sparse) rather than materializing a dense n1 x n2 array.
    """
    sim = sim.tocsr()
    out = []
    indptr, indices, data = sim.indptr, sim.indices, sim.data
    for row in range(sim.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        row_indices = indices[start:end]
        row_data = data[start:end]
        if len(row_data) <= k:
            out.append(row_indices)
            continue
        top_k_local = np.argpartition(row_data, -k)[-k:]
        out.append(row_indices[top_k_local])
    return out


def generate_candidates_for_country(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    top_k: int = 20,
    row_batch_size: int = 2000,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], Dict[str, float]]]:
    """
    Generate candidates from `other_df` for every S1 entity in `s1_df`,
    ASSUMING both dataframes have already been filtered to the same
    country (country partitioning happens one level up, in
    generate_all_candidates).

    Returns:
      candidates: {s1_entity_id: [candidate_entity_id, ...]}  (union of
        the three channels' top-K, deduped)
      channel_scores: {(s1_entity_id, candidate_entity_id): {"name": sim,
        "address": sim, "combo": sim}} -- reused downstream as features
        instead of recomputing TF-IDF per pair.

    Processes S1 in batches (row_batch_size) to bound memory: each batch's
    sparse similarity block against the full `other_df` pool is computed,
    top-K extracted, then discarded before the next batch.
    """
    s1_ids = s1_df["entity_id"].tolist()
    other_ids = other_df["entity_id"].tolist()

    if len(other_df) == 0 or len(s1_df) == 0:
        return {sid: [] for sid in s1_ids}, {}

    s1_channels = _prepare_channel_texts(s1_df)
    other_channels = _prepare_channel_texts(other_df)

    # Fit one vectorizer per channel, jointly over S1 + other, so both
    # sides share the same feature space for that channel.
    vectorizers = {}
    s1_matrices = {}
    other_matrices = {}
    for ch in CHANNELS:
        vec = _build_vectorizer()
        combined = vec.fit_transform(s1_channels[ch] + other_channels[ch])
        s1_matrices[ch] = combined[: len(s1_channels[ch])]
        other_matrices[ch] = combined[len(s1_channels[ch]):]
        vectorizers[ch] = vec

    candidates: Dict[str, List[str]] = {}
    channel_scores: Dict[Tuple[str, str], Dict[str, float]] = {}

    n_rows = len(s1_ids)
    for batch_start in range(0, n_rows, row_batch_size):
        batch_end = min(batch_start + row_batch_size, n_rows)
        batch_s1_ids = s1_ids[batch_start:batch_end]

        # per-channel top-k indices for this batch
        per_channel_topk: Dict[str, List[np.ndarray]] = {}
        per_channel_sim: Dict[str, sp.csr_matrix] = {}
        for ch in CHANNELS:
            batch_matrix = s1_matrices[ch][batch_start:batch_end]
            # sparse x sparse -> sparse: only nonzero n-gram overlaps
            # produce nonzero similarity entries.
            sim = batch_matrix @ other_matrices[ch].T
            per_channel_sim[ch] = sim.tocsr()
            per_channel_topk[ch] = _sparse_top_k_per_row(sim, top_k)

        for local_i, s1_id in enumerate(batch_s1_ids):
            union_idx = set()
            for ch in CHANNELS:
                union_idx.update(per_channel_topk[ch][local_i].tolist())
            cand_ids = [other_ids[j] for j in sorted(union_idx)]
            candidates[s1_id] = cand_ids

            for j in sorted(union_idx):
                cand_id = other_ids[j]
                scores = {}
                for ch in CHANNELS:
                    scores[ch] = float(per_channel_sim[ch][local_i, j])
                channel_scores[(s1_id, cand_id)] = scores

    return candidates, channel_scores


def merge_candidate_dicts(*dicts: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Union candidate lists (e.g. Source2 candidates + Source3 candidates) per S1 id."""
    merged: Dict[str, List[str]] = {}
    for d in dicts:
        for s1_id, cand_ids in d.items():
            merged.setdefault(s1_id, [])
            existing = set(merged[s1_id])
            merged[s1_id].extend(c for c in cand_ids if c not in existing)
    return merged


def generate_all_candidates(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
    top_k: int = 20, row_batch_size: int = 2000,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], Dict[str, float]]]:
    """
    Full blocking stage: partitions all three sources by country (EDA B9:
    100% of true matches are same-country, so this is free recall-safe
    reduction), then runs multi-channel blocking against S2 and S3
    separately within each country partition, and unions the results.

    This is the top-level function train.py / infer.py should call.
    """
    all_candidates: Dict[str, List[str]] = {sid: [] for sid in s1_df["entity_id"]}
    all_channel_scores: Dict[Tuple[str, str], Dict[str, float]] = {}

    countries = s1_df["country"].unique().tolist()
    for country in countries:
        s1_part = s1_df[s1_df["country"] == country].reset_index(drop=True)
        s2_part = s2_df[s2_df["country"] == country].reset_index(drop=True)
        s3_part = s3_df[s3_df["country"] == country].reset_index(drop=True)

        cand_s2, scores_s2 = generate_candidates_for_country(
            s1_part, s2_part, top_k=top_k, row_batch_size=row_batch_size)
        cand_s3, scores_s3 = generate_candidates_for_country(
            s1_part, s3_part, top_k=top_k, row_batch_size=row_batch_size)

        merged = merge_candidate_dicts(cand_s2, cand_s3)
        all_candidates.update(merged)
        all_channel_scores.update(scores_s2)
        all_channel_scores.update(scores_s3)

    return all_candidates, all_channel_scores


def measure_recall_ceiling(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> float:
    """
    Diagnostic: of all true matches in ground_truth, what fraction survived
    into the blocking candidate set? This is the hard ceiling on achievable
    recall -- per EDA E18/E20, the union-channel config should land in the
    97-100% range. If your measured number is meaningfully below that,
    raise top_k before touching the matcher.
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
