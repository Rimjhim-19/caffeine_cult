"""
train_matcher.py

Trains the LightGBM matcher from the candidate table blocking already
cached, and picks the decision threshold that maximises macro F_0.5.

Deliberately does NOT re-run blocking. Blocking is the expensive stage and
its output is on disk; re-deriving it on every training run would cost an
hour per experiment for no new information.

    python src/train_matcher.py \
        --candidates ../cache_combo/candidates_train.pkl \
        --data-dir ../dataset/train \
        --model-out ../model/matcher.txt

Outputs model/matcher.txt and model/threshold.json for run_infer.py.
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
from features_fast import FEATURE_NAMES, add_labels, build_features
from io_utils import read_ground_truth, read_source, ground_truth_to_dict


def f_beta_macro(predictions, ground_truth, all_s1_ids, beta: float = 0.5) -> float:
    """Macro-averaged F_beta, exactly as the challenge specifies: computed
    per Source 1 entity, then averaged. A singleton scores 1.0 when
    correctly predicted empty and 0.0 on any false merge."""
    beta2 = beta ** 2
    scores = []
    for s1_id in all_s1_ids:
        pred = set(predictions.get(s1_id, []))
        true = set(ground_truth.get(s1_id, []))
        if not true and not pred:
            scores.append(1.0)
            continue
        if not pred or not true:
            scores.append(0.0)
            continue
        tp = len(pred & true)
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / len(pred)
        recall = tp / len(true)
        scores.append((1 + beta2) * precision * recall
                      / (beta2 * precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def predict_with_assignment(df: pd.DataFrame, scores: np.ndarray,
                            threshold: float) -> dict:
    """Threshold, then enforce the one-S1-per-candidate constraint.

    Verified on all 7,638,365 training ground-truth pairs: no Source 2 or 3
    record is ever matched to more than one Source 1 entity. Independent
    per-entity thresholding does not respect that, so the same record can
    be claimed twice -- guaranteeing at least one of them is wrong. The
    greedy pass keeps each candidate only for its highest-scoring S1.
    """
    keep = scores >= threshold
    if not keep.any():
        return {}
    triples = list(zip(df["source1_entity_id"].values[keep],
                       df["candidate_entity_id"].values[keep],
                       scores[keep]))
    return deduplicate_assignment(triples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="candidates_train.pkl from run_blocking")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-out", default="model/matcher.txt")
    ap.add_argument("--threshold-out", default="model/threshold.json")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-ratio", type=float, default=0.0,
                    help="keep this many negatives per positive (0 = all). "
                         "Only ever subsample the TRAINING fold.")
    args = ap.parse_args()

    t0 = time.time()
    cands = pd.read_pickle(args.candidates)
    s1 = read_source(os.path.join(args.data_dir, "train_source1.tsv"))
    pool = pd.concat([
        read_source(os.path.join(args.data_dir, "train_source2.tsv")),
        read_source(os.path.join(args.data_dir, "train_source3.tsv")),
    ], ignore_index=True)
    gt = ground_truth_to_dict(read_ground_truth(
        os.path.join(args.data_dir, "train_ground_truth.tsv")))
    print(f"[load] {len(cands):,} candidate pairs ({time.time() - t0:.0f}s)")

    # Only the entities blocking actually ran on, and only their rows.
    s1 = s1[s1["entity_id"].isin(set(cands["source1_entity_id"]))]
    pool = pool[pool["entity_id"].isin(set(cands["candidate_entity_id"]))]
    print(f"[load] {len(s1):,} S1 entities in scope")

    df = build_features(cands, s1, pool)
    df = add_labels(df, gt)
    print(f"[features] positive rate {df['label'].mean():.4f}")

    # Split by ENTITY, never by pair: an entity's positives and negatives
    # must land on the same side, or validation scores are inflated.
    ids = s1["entity_id"].tolist()
    rng = np.random.RandomState(args.seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_frac))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    print(f"[split] {len(train_ids):,} train / {len(val_ids):,} val entities")

    train_df = df[df["source1_entity_id"].isin(train_ids)]
    val_df = df[df["source1_entity_id"].isin(val_ids)]

    if args.neg_ratio > 0:
        pos = train_df[train_df["label"] == 1]
        neg = train_df[train_df["label"] == 0]
        n_neg = min(len(neg), int(len(pos) * args.neg_ratio))
        train_df = pd.concat([pos, neg.sample(n=n_neg, random_state=args.seed)])
        print(f"[split] subsampled to {len(train_df):,} training pairs")

    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
        },
        lgb.Dataset(train_df[FEATURE_NAMES], label=train_df["label"]),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(val_df[FEATURE_NAMES], label=val_df["label"])],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(200)],
    )
    print(f"[train] stopped at iteration {booster.best_iteration}")

    val_scores = booster.predict(val_df[FEATURE_NAMES],
                                 num_iteration=booster.best_iteration)
    val_scores = np.asarray(val_scores)
    val_list = sorted(val_ids)

    # F_0.5 weights precision twice as heavily as recall, so the optimum
    # sits well above 0.5. Swept rather than assumed.
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.20, 0.96, 0.025):
        preds = predict_with_assignment(val_df, val_scores, float(t))
        f = f_beta_macro(preds, gt, val_list, beta=0.5)
        if f > best_f:
            best_f, best_t = f, float(t)
        print(f"  threshold {t:.3f} -> F_0.5 {f:.4f}")

    print(f"\n[val] best threshold {best_t:.3f} -> macro F_0.5 {best_f:.4f}")

    imp = sorted(zip(FEATURE_NAMES,
                     booster.feature_importance("gain")),
                 key=lambda x: -x[1])
    print("[val] top features: " + ", ".join(f"{n}({int(g)})" for n, g in imp[:8]))

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    booster.save_model(args.model_out, num_iteration=booster.best_iteration)
    with open(args.threshold_out, "w") as f:
        json.dump({"threshold": best_t, "val_f0.5": best_f,
                   "best_iteration": booster.best_iteration,
                   "features": FEATURE_NAMES}, f, indent=2)
    print(f"[done] {args.model_out} + {args.threshold_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())