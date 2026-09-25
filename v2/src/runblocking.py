"""
run_blocking.py

Standalone entry point for the blocking stage. Run this on its own before
wiring blocking into train.py -- it is the slowest part of the pipeline and
you want its output cached on disk, not recomputed every time someone
retunes a threshold.

Typical first run (subsample S1 so you get a result in minutes, but always
block against the FULL pool -- shrinking the pool makes every number you
measure optimistic):

    python src/run_blocking.py --data-dir ../dataset/train --split train \
        --sample-s1 40000 --out-dir ../cache

Full test-set run (slow -- see the note on --n-jobs):

    python src/run_blocking.py --data-dir ../dataset/test --split test \
        --out-dir ../cache

Outputs, into --out-dir:
    candidates_<split>.pkl   long-form pair table (feature stage reads this)
    candidate_pairs.tsv      submission-shaped file (test split only)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blocking import BlockingConfig, generate_all_candidates, measure_recall_ceiling
from io_utils import read_source, read_ground_truth, ground_truth_to_dict, write_candidate_pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="folder holding <split>_source1/2/3.tsv")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out-dir", default="cache")
    ap.add_argument("--sample-s1", type=int, default=0,
                    help="block only this many S1 entities (0 = all). "
                         "Sampling S1 is safe; sampling the pool is not.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-sim", type=float, default=0.05)
    ap.add_argument("--rel-floor", type=float, default=0.25)
    ap.add_argument("--max-df", type=float, default=0.05)
    ap.add_argument("--cache-dir", default=None,
                    help="reuse fitted vectorizers + pool matrices across "
                         "runs; ~25 min of fixed cost paid once")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="partitions in parallel. Memory scales with this.")
    ap.add_argument("--batch-size", type=int, default=500,
                    help="S1 rows per similarity block. Lower this first if "
                         "you hit a MemoryError.")
    ap.add_argument("--channels", default="name,address",
                    help="comma-separated: name,address,combo")
    args = ap.parse_args()

    s = args.split
    t0 = time.time()
    s1 = read_source(os.path.join(args.data_dir, f"{s}_source1.tsv"))
    s2 = read_source(os.path.join(args.data_dir, f"{s}_source2.tsv"))
    s3 = read_source(os.path.join(args.data_dir, f"{s}_source3.tsv"))
    print(f"[load] {len(s1):,} S1 | {len(s2):,} S2 | {len(s3):,} S3 "
          f"({time.time() - t0:.0f}s)")
    print(f"[load] countries: {sorted(s1['country'].unique().tolist())}")

    if args.sample_s1 and args.sample_s1 < len(s1):
        s1 = s1.sample(n=args.sample_s1, random_state=args.seed)
        print(f"[load] sampled down to {len(s1):,} S1 entities "
              f"(pool left at full size on purpose)")

    cfg = BlockingConfig(
            top_k=args.top_k,
            min_sim=args.min_sim,
            rel_floor=args.rel_floor,
            max_df=args.max_df,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            n_jobs=args.n_jobs,
            channels=tuple(args.channels.split(",")),
    )

    t0 = time.time()
    cands = generate_all_candidates(s1, s2, s3, cfg, verbose=True)
    elapsed = time.time() - t0
    print(f"[blocking] {len(cands.s1_ids):,} pairs in {elapsed / 60:.1f} min")

    os.makedirs(args.out_dir, exist_ok=True)
    frame = cands.to_frame()
    out_pkl = os.path.join(args.out_dir, f"candidates_{s}.pkl")
    frame.to_pickle(out_pkl)
    print(f"[write] {out_pkl}  {frame.shape}")

    as_dict = cands.to_dict()

    # Training split: we have labels, so report the number that actually
    # matters -- the ceiling this candidate set puts on final recall.
    gt_path = os.path.join(args.data_dir, f"{s}_ground_truth.tsv")
    if os.path.isfile(gt_path):
        gt = ground_truth_to_dict(read_ground_truth(gt_path))
        stats = measure_recall_ceiling(as_dict, gt,
                                       s1_ids=s1["entity_id"].tolist())
        print("[recall] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                      else f"{k}={v:,}"
                                      for k, v in stats.items()))
        if stats["recall_ceiling"] < 0.95:
            print("[recall] below 0.95 -- raise --top-k or lower --rel-floor "
                  "before touching the matcher; no classifier can recover a "
                  "true match that blocking never proposed.")

    # Test split: emit the submission-shaped file. Every S1 entity needs a
    # row even when blocking found nothing for it.
    if s == "test":
        for s1_id in s1["entity_id"]:
            as_dict.setdefault(s1_id, [])
        out_tsv = os.path.join(args.out_dir, "candidate_pairs.tsv")
        write_candidate_pairs(out_tsv, as_dict)
        print(f"[write] {out_tsv}  ({len(as_dict):,} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())