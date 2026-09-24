"""
score_local.py

Standalone scorer implementing the challenge's exact macro-averaged F_0.5
metric, so you can score any predictions file against a held-out ground
truth split without touching the leaderboard. Distinct from the
challenge-provided utils/validate_submission.py, which only checks output
*format* -- this checks actual match quality.

Usage:
    python3 src/score_local.py \
        --predictions output/matching_results.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import read_ground_truth, ground_truth_to_dict
from train import f_beta_macro  # reuse the single implementation of the metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                         help="a matching_results.tsv-shaped file")
    parser.add_argument("--ground-truth", required=True,
                         help="ground truth tsv (source1_entity_id, matched_entity_ids)")
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()

    # predictions file shares the same 2-column shape as ground truth
    # (source1_entity_id, matched_entity_ids), so we parse it the same way.
    import pandas as pd
    pred_raw = pd.read_csv(args.predictions, sep="\t", dtype=str, keep_default_na=False)
    predictions = {}
    for _, row in pred_raw.iterrows():
        raw = row["matched_entity_ids"].strip()
        predictions[row["source1_entity_id"]] = [x for x in raw.split(",") if x] if raw else []

    gt = ground_truth_to_dict(read_ground_truth(args.ground_truth))
    all_s1_ids = list(gt.keys())

    missing = [s1_id for s1_id in all_s1_ids if s1_id not in predictions]
    if missing:
        print(f"[warn] {len(missing)} ground-truth entities missing from predictions "
              f"(scored as empty predictions): e.g. {missing[:5]}")

    score = f_beta_macro(predictions, gt, all_s1_ids, beta=args.beta)
    print(f"Macro F_{args.beta}: {score:.4f}  (n={len(all_s1_ids)} entities)")


if __name__ == "__main__":
    main()
