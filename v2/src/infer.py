"""
infer.py

Runs the full pipeline (country-partitioned multi-channel blocking ->
feature engineering -> trained matcher -> global dedup assignment) over
the test set and writes the two required output files:

    output/candidate_pairs.tsv    (blocking stage output -- pre-threshold)
    output/matching_results.tsv   (final matches -- scored on leaderboard)

Every S1 test entity gets exactly one row in both files (empty list if
blocking/matching found nothing). matching_results is guaranteed to be a
subset of candidate_pairs by construction: the dedup assignment
(assignment.py) only ever REMOVES pairs that scored below threshold or
lost a tie to a higher-scoring S1 entity -- it never introduces an id that
wasn't already a candidate.

Usage:
    python3 src/infer.py \
        --test-dir dataset/test \
        --model model/matcher.txt \
        --threshold-config model/threshold.json \
        --output-dir output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from assignment import deduplicate_assignment
from blocking import generate_all_candidates
from features import FEATURE_NAMES
from io_utils import read_source, write_candidate_pairs, write_matching_results
from train import build_feature_table  # reuse the single implementation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--model", default="model/matcher.txt")
    parser.add_argument("--threshold-config", default="model/threshold.json")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--top-k", type=int, default=None,
                         help="overrides the top_k stored in threshold config if given")
    parser.add_argument("--row-batch-size", type=int, default=None,
                         help="overrides the row_batch_size stored in threshold config if given")
    args = parser.parse_args()

    s1 = read_source(os.path.join(args.test_dir, "test_source1.tsv"))
    s2 = read_source(os.path.join(args.test_dir, "test_source2.tsv"))
    s3 = read_source(os.path.join(args.test_dir, "test_source3.tsv"))

    with open(args.threshold_config) as f:
        cfg = json.load(f)
    threshold = cfg["threshold"]
    top_k = args.top_k if args.top_k is not None else cfg.get("top_k", 20)
    row_batch_size = args.row_batch_size if args.row_batch_size is not None else cfg.get("row_batch_size", 2000)

    print(f"[infer] {len(s1)} S1 test entities | threshold={threshold} "
          f"top_k={top_k} row_batch_size={row_batch_size}")
    print(f"[infer] test countries: {sorted(s1['country'].unique().tolist())}")

    candidates, channel_scores = generate_all_candidates(
        s1, s2, s3, top_k=top_k, row_batch_size=row_batch_size)

    # Every S1 test id must appear even if blocking found nothing.
    all_s1_ids = s1["entity_id"].tolist()
    for s1_id in all_s1_ids:
        candidates.setdefault(s1_id, [])

    booster = lgb.Booster(model_file=args.model)
    feature_df = build_feature_table(s1, s2, s3, candidates, channel_scores, ground_truth=None)

    matches: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}
    if not feature_df.empty:
        scores = booster.predict(feature_df[FEATURE_NAMES])
        triples = list(zip(
            feature_df["source1_entity_id"].tolist(),
            feature_df["candidate_entity_id"].tolist(),
            scores.tolist(),
        ))
        above_threshold = [(s1_id, c, sc) for s1_id, c, sc in triples if sc >= threshold]
        # Global dedup: each candidate id goes to only the highest-scoring
        # S1 entity that claimed it (EDA B7: this is always true of the
        # real answer -- 0 exceptions across 7.64M ground-truth pairs).
        matches.update(deduplicate_assignment(above_threshold))

    os.makedirs(args.output_dir, exist_ok=True)
    write_candidate_pairs(os.path.join(args.output_dir, "candidate_pairs.tsv"), candidates)
    write_matching_results(os.path.join(args.output_dir, "matching_results.tsv"), matches)

    n_matched = sum(1 for v in matches.values() if v)
    print(f"[done] {n_matched}/{len(all_s1_ids)} S1 entities got >=1 match")
    print(f"[done] wrote {args.output_dir}/candidate_pairs.tsv and matching_results.tsv")


if __name__ == "__main__":
    main()
