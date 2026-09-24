"""
infer.py

Runs the full pipeline (blocking -> feature engineering -> trained matcher)
over the test set and writes the two required output files:

    output/candidate_pairs.tsv    (blocking stage output -- pre-threshold)
    output/matching_results.tsv   (final matches -- scored on leaderboard)

Every Source1 test entity gets exactly one row in both files (empty list
if blocking/matching found nothing), matching_results is guaranteed to be
a subset of candidate_pairs by construction (we threshold within the same
candidate set, never introduce new ids).

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
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from blocking import generate_candidates, merge_candidate_dicts
from features import compute_pair_features, FEATURE_NAMES
from io_utils import read_source, write_candidate_pairs, write_matching_results


def entity_prefix(entity_id: str) -> str:
    return entity_id.split("-")[0]


def build_unlabeled_feature_table(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
    candidates: Dict[str, List[str]],
) -> pd.DataFrame:
    """Same as train.build_labeled_feature_table but without ground truth / labels."""
    s1_lookup = s1_df.set_index("entity_id").to_dict("index")
    s2_lookup = s2_df.set_index("entity_id").to_dict("index")
    s3_lookup = s3_df.set_index("entity_id").to_dict("index")

    rows = []
    for s1_id, cand_ids in candidates.items():
        if s1_id not in s1_lookup:
            continue
        s1_rec = s1_lookup[s1_id]
        for cand_id in cand_ids:
            prefix = entity_prefix(cand_id)
            lookup = s2_lookup if prefix == "S2" else s3_lookup
            if cand_id not in lookup:
                continue
            cand_rec = lookup[cand_id]
            feats = compute_pair_features(
                s1_rec["business_name"], s1_rec["business_address"], s1_rec["country"],
                cand_rec["business_name"], cand_rec["business_address"], cand_rec["country"],
                prefix,
            )
            feats["source1_entity_id"] = s1_id
            feats["candidate_entity_id"] = cand_id
            rows.append(feats)
    return pd.DataFrame(rows)


def score_and_threshold(
    booster: lgb.Booster, feature_df: pd.DataFrame, threshold: float,
    all_s1_ids: List[str],
) -> Dict[str, List[str]]:
    """
    Apply the trained model to every candidate pair, keep those scoring
    >= threshold, group by S1 id. Every S1 id (including those with zero
    candidates) gets an entry -- required so every test entity appears in
    the output even as an empty row.
    """
    out: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}
    if feature_df.empty:
        return out
    scores = booster.predict(feature_df[FEATURE_NAMES])
    feature_df = feature_df.assign(score=scores)
    for s1_id, group in feature_df.groupby("source1_entity_id"):
        kept = group.loc[group["score"] >= threshold, "candidate_entity_id"].tolist()
        out[s1_id] = kept
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--model", default="model/matcher.txt")
    parser.add_argument("--threshold-config", default="model/threshold.json")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--top-k", type=int, default=None,
                         help="overrides the top_k stored in threshold config if given")
    args = parser.parse_args()

    s1 = read_source(os.path.join(args.test_dir, "test_source1.tsv"))
    s2 = read_source(os.path.join(args.test_dir, "test_source2.tsv"))
    s3 = read_source(os.path.join(args.test_dir, "test_source3.tsv"))

    with open(args.threshold_config) as f:
        cfg = json.load(f)
    threshold = cfg["threshold"]
    top_k = args.top_k if args.top_k is not None else cfg.get("top_k", 20)

    print(f"[infer] {len(s1)} S1 test entities | threshold={threshold} top_k={top_k}")

    cand_s2 = generate_candidates(s1, s2, top_k=top_k)
    cand_s3 = generate_candidates(s1, s3, top_k=top_k)
    candidates = merge_candidate_dicts(cand_s2, cand_s3)

    # every S1 test id must appear even if blocking found nothing
    all_s1_ids = s1["entity_id"].tolist()
    for s1_id in all_s1_ids:
        candidates.setdefault(s1_id, [])

    booster = lgb.Booster(model_file=args.model)
    feature_df = build_unlabeled_feature_table(s1, s2, s3, candidates)
    matches = score_and_threshold(booster, feature_df, threshold, all_s1_ids)

    os.makedirs(args.output_dir, exist_ok=True)
    write_candidate_pairs(os.path.join(args.output_dir, "candidate_pairs.tsv"), candidates)
    write_matching_results(os.path.join(args.output_dir, "matching_results.tsv"), matches)

    n_matched = sum(1 for v in matches.values() if v)
    print(f"[done] {n_matched}/{len(all_s1_ids)} S1 entities got >=1 match")
    print(f"[done] wrote {args.output_dir}/candidate_pairs.tsv and matching_results.tsv")


if __name__ == "__main__":
    main()
