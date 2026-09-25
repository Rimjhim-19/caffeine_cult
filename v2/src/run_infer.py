"""
run_infer.py

Scores the cached test candidates with the trained matcher and writes the
two submission files.

    python src/run_infer.py \
        --candidates ../cache/candidates_test.pkl \
        --data-dir ../dataset/test \
        --model ../model/matcher.txt \
        --threshold-config ../model/threshold.json \
        --output-dir ../output

Writes output/matching_results.tsv (the file scored on the leaderboard)
and output/candidate_pairs.tsv (blocking's candidate set, audited but not
scored). Every Source 1 test entity gets exactly one row in both, empty
where nothing was matched -- a missing entity is an outright rejection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assignment import deduplicate_assignment
from features_fast import FEATURE_NAMES, build_features
from io_utils import read_source, write_candidate_pairs, write_matching_results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="candidates_test.pkl from run_blocking --split test")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="model/matcher.txt")
    ap.add_argument("--threshold-config", default="model/threshold.json")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the tuned threshold")
    args = ap.parse_args()

    t0 = time.time()
    cands = pd.read_pickle(args.candidates)
    s1 = read_source(os.path.join(args.data_dir, "test_source1.tsv"))
    pool = pd.concat([
        read_source(os.path.join(args.data_dir, "test_source2.tsv")),
        read_source(os.path.join(args.data_dir, "test_source3.tsv")),
    ], ignore_index=True)
    print(f"[load] {len(s1):,} S1 test entities | {len(cands):,} candidate "
          f"pairs ({time.time() - t0:.0f}s)")
    print(f"[load] countries: {sorted(s1['country'].unique().tolist())}")

    with open(args.threshold_config) as f:
        cfg = json.load(f)
    threshold = args.threshold if args.threshold is not None else cfg["threshold"]
    print(f"[infer] threshold {threshold:.3f} (val F_0.5 {cfg.get('val_f0.5', 0):.4f})")

    pool_in_scope = pool[pool["entity_id"].isin(set(cands["candidate_entity_id"]))]
    df = build_features(cands, s1, pool_in_scope)

    booster = lgb.Booster(model_file=args.model)
    scores = np.asarray(booster.predict(df[FEATURE_NAMES]))
    print(f"[infer] scored {len(scores):,} pairs")

    keep = scores >= threshold
    triples = list(zip(df["source1_entity_id"].values[keep],
                       df["candidate_entity_id"].values[keep],
                       scores[keep]))
    print(f"[infer] {len(triples):,} pairs above threshold")

    # Each Source 2/3 record belongs to at most one Source 1 entity -- true
    # of every one of the 7,638,365 training ground-truth pairs. Thresholding
    # alone does not enforce it.
    matches = deduplicate_assignment(triples)
    dropped = len(triples) - sum(len(v) for v in matches.values())
    print(f"[infer] assignment dropped {dropped:,} double-claimed candidates")

    # Every test entity needs a row, matched or not.
    all_s1 = s1["entity_id"].tolist()
    final = {s1_id: matches.get(s1_id, []) for s1_id in all_s1}
    candidate_lists = {s1_id: [] for s1_id in all_s1}
    for s1_id, cand_id in zip(cands["source1_entity_id"],
                              cands["candidate_entity_id"]):
        candidate_lists[s1_id].append(cand_id)

    os.makedirs(args.output_dir, exist_ok=True)
    write_matching_results(
        os.path.join(args.output_dir, "matching_results.tsv"), final)
    write_candidate_pairs(
        os.path.join(args.output_dir, "candidate_pairs.tsv"), candidate_lists)

    n_matched = sum(1 for v in final.values() if v)
    total = sum(len(v) for v in final.values())
    print(f"[done] {n_matched:,}/{len(all_s1):,} entities matched "
          f"({n_matched / len(all_s1):.1%}), {total:,} total matches, "
          f"{total / max(1, n_matched):.2f} per matched entity")
    print(f"[done] wrote {args.output_dir}/matching_results.tsv + "
          f"candidate_pairs.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())