"""
assignment.py

Global deduplication / assignment stage.

Why this exists (EDA B7): across all 7,638,365 ground-truth pairs, ZERO
S2 or S3 records map to more than one S1 entity -- every noisy record
belongs to at most one real business. This is a hard structural property
of the correct answer, not a coincidence of this particular dataset slice.

The per-S1 independent thresholding in train.py/infer.py (score every
candidate, keep those above threshold) does NOT enforce this on its own:
it's entirely possible for the same S2 record to score above threshold
for two different S1 entities if their names/addresses are close to each
other. Left unfixed, that produces predictions that are structurally
guaranteed wrong for at least one of the two S1 entities -- a correctness
bug, not just a tuning issue, and a direct precision hit under the
precision-heavy F_0.5 metric.

This module resolves that: given all (s1_id, candidate_id, score) triples
above threshold, it greedily keeps each candidate_id for only the highest-
scoring S1 entity it appears under, dropping the rest. Simple greedy
(rather than exact bipartite/Hungarian assignment) is sufficient here
since the constraint is one-sided -- only candidate_id needs to be unique;
an S1 entity keeping multiple candidates is expected and correct (EDA B6:
median match count per S1 is 3, up to 10).
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def deduplicate_assignment(
    scored_pairs: List[Tuple[str, str, float]],
) -> Dict[str, List[str]]:
    """
    Args:
        scored_pairs: list of (s1_entity_id, candidate_entity_id, score)
            for every pair that passed the matching threshold.

    Returns:
        {s1_entity_id: [candidate_entity_id, ...]} where every
        candidate_entity_id across the whole result appears under exactly
        one s1_entity_id -- whichever S1 it scored highest against.

    S1 entities with zero surviving candidates are NOT included in the
    returned dict; callers should fill in empty lists for any S1 id not
    present (as infer.py already does for singletons).
    """
    # Highest score first, so the first time we see a given candidate_id
    # is its best-scoring assignment.
    scored_pairs_sorted = sorted(scored_pairs, key=lambda x: x[2], reverse=True)

    claimed_candidates: set = set()
    result: Dict[str, List[str]] = {}

    for s1_id, cand_id, _score in scored_pairs_sorted:
        if cand_id in claimed_candidates:
            continue
        claimed_candidates.add(cand_id)
        result.setdefault(s1_id, []).append(cand_id)

    return result
