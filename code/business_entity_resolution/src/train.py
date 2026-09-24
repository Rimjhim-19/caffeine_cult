"""
train.py

End-to-end training:
  1. Load train sources + ground truth.
  2. Split Source1 entities into train/validation folds (entity-level split
     so no leakage of a single S1 entity's pairs across folds).
  3. Run blocking on the train fold to get candidate pairs.
  4. Label each candidate pair (positive = in ground truth, negative =
     candidate but not a true match -- these are HARD negatives, produced
     by the same blocking stage used at inference, which is exactly what
     makes them useful).
  5. Compute pairwise features, train a LightGBM binary classifier.
  6. Run the full pipeline (blocking + trained model) on the validation
     fold, sweep the decision threshold, and report the macro-averaged
     F_0.5 the challenge actually scores on -- pick the threshold that
     maximizes it.
  7. Save the model + chosen threshold for infer.py.

Usage:
    python3 src/train.py \
        --train-dir dataset/train \
        --model-out model/matcher.txt \
        --threshold-out model/threshold.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from blocking import generate_candidates, merge_candidate_dicts, measure_recall_ceiling
from features import compute_pair_features, FEATURE_NAMES
from io_utils import read_source, read_ground_truth, ground_truth_to_dict


def entity_prefix(entity_id: str) -> str:
    """'S2-00047' -> 'S2'. Source is derived from the ID prefix, per spec."""
    return entity_id.split("-")[0]


def split_source1(source1_df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Entity-level train/val split on Source1 ids (no pair leakage across folds)."""
    ids = source1_df["entity_id"].tolist()
    rng = np.random.RandomState(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    val_ids = set(shuffled[:n_val])
    train_mask = ~source1_df["entity_id"].isin(val_ids)
    return source1_df[train_mask].reset_index(drop=True), source1_df[~train_mask].reset_index(drop=True)


def build_candidate_pairs(s1_df, s2_df, s3_df, top_k: int) -> Dict[str, List[str]]:
    """Blocking stage: union of Source2 and Source3 candidates per S1 entity."""
    cand_s2 = generate_candidates(s1_df, s2_df, top_k=top_k)
    cand_s3 = generate_candidates(s1_df, s3_df, top_k=top_k)
    return merge_candidate_dicts(cand_s2, cand_s3)


def build_labeled_feature_table(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
    candidates: Dict[str, List[str]], ground_truth: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Turn candidate pairs into a labeled feature table for training.
    label = 1 if the candidate id is in that S1 entity's true match list,
    else 0 (a hard negative, since it passed blocking but isn't a match).
    """
    s1_lookup = s1_df.set_index("entity_id").to_dict("index")
    s2_lookup = s2_df.set_index("entity_id").to_dict("index")
    s3_lookup = s3_df.set_index("entity_id").to_dict("index")

    rows = []
    for s1_id, cand_ids in candidates.items():
        if s1_id not in s1_lookup:
            continue
        s1_rec = s1_lookup[s1_id]
        true_set = set(ground_truth.get(s1_id, []))
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
            feats["label"] = int(cand_id in true_set)
            rows.append(feats)
    return pd.DataFrame(rows)


def train_matcher(feature_df: pd.DataFrame) -> lgb.Booster:
    """Train a LightGBM binary classifier on the labeled pair feature table."""
    X = feature_df[FEATURE_NAMES]
    y = feature_df["label"]
    train_set = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbose": -1,
    }
    # NOTE: on real data, replace num_boost_round with early stopping against
    # a held-out eval_set for a properly tuned model.
    booster = lgb.train(params, train_set, num_boost_round=300)
    return booster


def predict_matches(
    booster: lgb.Booster, feature_df: pd.DataFrame, threshold: float
) -> Dict[str, List[str]]:
    """Score candidate pairs and keep those above threshold, grouped by S1 id."""
    if feature_df.empty:
        return {}
    scores = booster.predict(feature_df[FEATURE_NAMES])
    feature_df = feature_df.assign(score=scores)
    out: Dict[str, List[str]] = {}
    for s1_id, group in feature_df.groupby("source1_entity_id"):
        kept = group.loc[group["score"] >= threshold, "candidate_entity_id"].tolist()
        out[s1_id] = kept
    return out


def f_beta_macro(
    predictions: Dict[str, List[str]], ground_truth: Dict[str, List[str]],
    all_s1_ids: List[str], beta: float = 0.5,
) -> float:
    """
    Macro-averaged F_beta exactly as specified: computed per Source1
    entity (singletons included, scoring 1.0 when correctly predicted
    empty and 0.0 on any false merge), then averaged across all entities.
    """
    scores = []
    for s1_id in all_s1_ids:
        pred = set(predictions.get(s1_id, []))
        true = set(ground_truth.get(s1_id, []))
        if not true and not pred:
            scores.append(1.0)
            continue
        if not pred:
            scores.append(0.0)
            continue
        tp = len(pred & true)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(true) if true else 0.0
        if precision == 0 and recall == 0:
            scores.append(0.0)
            continue
        beta2 = beta ** 2
        denom = (beta2 * precision) + recall
        f = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
        scores.append(f)
    return float(np.mean(scores)) if scores else 0.0


def tune_threshold(
    booster: lgb.Booster, val_feature_df: pd.DataFrame,
    ground_truth: Dict[str, List[str]], all_s1_ids: List[str],
) -> Tuple[float, float]:
    """Sweep thresholds, return (best_threshold, best_f0.5) on the val fold."""
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        preds = predict_matches(booster, val_feature_df, float(t))
        f = f_beta_macro(preds, ground_truth, all_s1_ids, beta=0.5)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--model-out", default="model/matcher.txt")
    parser.add_argument("--threshold-out", default="model/threshold.json")
    parser.add_argument("--top-k", type=int, default=20,
                         help="blocking top-K per source, per S1 entity")
    args = parser.parse_args()

    s1 = read_source(os.path.join(args.train_dir, "train_source1.tsv"))
    s2 = read_source(os.path.join(args.train_dir, "train_source2.tsv"))
    s3 = read_source(os.path.join(args.train_dir, "train_source3.tsv"))
    gt = ground_truth_to_dict(read_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv")))

    s1_train, s1_val = split_source1(s1)

    print(f"[train] {len(s1_train)} S1 train entities, {len(s1_val)} S1 val entities")

    train_candidates = build_candidate_pairs(s1_train, s2, s3, args.top_k)
    recall_ceiling = measure_recall_ceiling(train_candidates, gt)
    print(f"[blocking] train recall ceiling: {recall_ceiling:.4f} "
          f"(this is the max recall the matcher can possibly achieve)")

    train_feats = build_labeled_feature_table(s1_train, s2, s3, train_candidates, gt)
    print(f"[features] {len(train_feats)} labeled training pairs "
          f"({train_feats['label'].sum()} positive)")

    booster = train_matcher(train_feats)

    val_candidates = build_candidate_pairs(s1_val, s2, s3, args.top_k)
    val_feats = build_labeled_feature_table(s1_val, s2, s3, val_candidates, gt)
    best_t, best_f = tune_threshold(booster, val_feats, gt, s1_val["entity_id"].tolist())
    print(f"[val] best threshold={best_t:.2f} -> macro F_0.5={best_f:.4f}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    booster.save_model(args.model_out)
    with open(args.threshold_out, "w") as f:
        json.dump({"threshold": best_t, "val_f0.5": best_f,
                    "top_k": args.top_k}, f, indent=2)
    print(f"[done] model -> {args.model_out}, threshold config -> {args.threshold_out}")


if __name__ == "__main__":
    main()
