"""
Section D: Similarity Separability Analysis
Compares true pairs vs 5x random non-matching pairs drawn from the same country.
Computes name similarity (ratio, token_sort_ratio, token_set_ratio) and address similarity.
Generates histograms, separability metrics, and hard-case identification (name_sim < 0.6).
Outputs saved to eda/output/
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "dataset"
OUTPUT_DIR = "eda/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)

def main():
    print("Loading S1 records for Section D...")
    s1_records = {} # s1_id -> (name, addr, country)
    with open(os.path.join(DATA_DIR, "train", "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for idx, line in enumerate(fp):
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4:
                s1_records[p[0]] = (p[1], p[2], p[3])
            if idx >= 300000: # sample 300k S1 entities
                break
    print(f"Loaded {len(s1_records):,} candidate S1 records.")

    print("Extracting true pairs from ground truth...")
    true_pairs_us = []
    true_pairs_in = []
    s1_matched_all = set()

    with open(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            s1_id = p[0]
            if len(p) < 2 or not p[1].strip():
                continue
            matches = p[1].split(",")
            if s1_id not in s1_records:
                continue
            ctry = s1_records[s1_id][2]
            s1_matched_all.add(s1_id)
            for m in matches:
                if ctry == "US" and len(true_pairs_us) < 10000:
                    true_pairs_us.append((s1_id, m))
                elif ctry == "India" and len(true_pairs_in) < 10000:
                    true_pairs_in.append((s1_id, m))
            if len(true_pairs_us) >= 10000 and len(true_pairs_in) >= 10000:
                break

    print(f"Sampled {len(true_pairs_us):,} US true pairs and {len(true_pairs_in):,} India true pairs.")
    all_true_pairs = true_pairs_us + true_pairs_in

    needed_s2 = set()
    needed_s3 = set()
    for _, m in all_true_pairs:
        if m.startswith("S2-"):
            needed_s2.add(m)
        else:
            needed_s3.add(m)

    # Also collect random candidate pool from S2 and S3 for negative pair generation
    s2_pool_us = []
    s2_pool_in = []
    s3_pool_us = []
    s3_pool_in = []
    
    noisy_records = {} # m_id -> (name, addr, ctry)

    print("Reading S2 for needed true matches and random pool...")
    with open(os.path.join(DATA_DIR, "train", "train_source2.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for idx, line in enumerate(fp):
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 4:
                continue
            e_id, nm, ad, ctry = p[0], p[1], p[2], p[3]
            if e_id in needed_s2:
                noisy_records[e_id] = (nm, ad, ctry)
            # Pool for negative sampling
            if ctry == "US" and len(s2_pool_us) < 30000:
                s2_pool_us.append((e_id, nm, ad))
            elif ctry == "India" and len(s2_pool_in) < 30000:
                s2_pool_in.append((e_id, nm, ad))

    print("Reading S3 for needed true matches and random pool...")
    with open(os.path.join(DATA_DIR, "train", "train_source3.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for idx, line in enumerate(fp):
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 4:
                continue
            e_id, nm, ad, ctry = p[0], p[1], p[2], p[3]
            if e_id in needed_s3:
                noisy_records[e_id] = (nm, ad, ctry)
            if ctry == "US" and len(s3_pool_us) < 30000:
                s3_pool_us.append((e_id, nm, ad))
            elif ctry == "India" and len(s3_pool_in) < 30000:
                s3_pool_in.append((e_id, nm, ad))

    print(f"Retrieved {len(noisy_records):,} true match records.")

    # Compute similarities for TRUE pairs
    print("Computing similarities for TRUE pairs...")
    true_results = []
    for s1_id, m_id in all_true_pairs:
        if m_id not in noisy_records:
            continue
        s1_name, s1_addr, ctry = s1_records[s1_id]
        m_name, m_addr, _ = noisy_records[m_id]

        true_results.append({
            "pair_type": "TRUE",
            "country": ctry,
            "name_ratio": fuzz.ratio(s1_name.lower(), m_name.lower()) / 100.0,
            "name_token_sort": fuzz.token_sort_ratio(s1_name.lower(), m_name.lower()) / 100.0,
            "name_token_set": fuzz.token_set_ratio(s1_name.lower(), m_name.lower()) / 100.0,
            "addr_ratio": (fuzz.ratio(s1_addr.lower(), m_addr.lower()) / 100.0) if s1_addr and m_addr else 0.0,
            "addr_token_sort": (fuzz.token_sort_ratio(s1_addr.lower(), m_addr.lower()) / 100.0) if s1_addr and m_addr else 0.0,
        })
    df_true = pd.DataFrame(true_results)

    # Generate 5x NON-MATCHING pairs within same country
    print("Generating 5x non-matching pairs from same country...")
    neg_results = []
    s1_us_list = [k for k, v in s1_records.items() if v[2] == "US"]
    s1_in_list = [k for k, v in s1_records.items() if v[2] == "India"]
    
    noisy_pool_us = s2_pool_us + s3_pool_us
    noisy_pool_in = s2_pool_in + s3_pool_in

    num_neg_us = len(df_true[df_true["country"] == "US"]) * 5
    num_neg_in = len(df_true[df_true["country"] == "India"]) * 5

    for _ in range(num_neg_us):
        s1_id = random.choice(s1_us_list)
        m_id, m_name, m_addr = random.choice(noisy_pool_us)
        s1_name, s1_addr, _ = s1_records[s1_id]
        neg_results.append({
            "pair_type": "NON-MATCH",
            "country": "US",
            "name_ratio": fuzz.ratio(s1_name.lower(), m_name.lower()) / 100.0,
            "name_token_sort": fuzz.token_sort_ratio(s1_name.lower(), m_name.lower()) / 100.0,
            "name_token_set": fuzz.token_set_ratio(s1_name.lower(), m_name.lower()) / 100.0,
            "addr_ratio": (fuzz.ratio(s1_addr.lower(), m_addr.lower()) / 100.0) if s1_addr and m_addr else 0.0,
            "addr_token_sort": (fuzz.token_sort_ratio(s1_addr.lower(), m_addr.lower()) / 100.0) if s1_addr and m_addr else 0.0,
        })

    for _ in range(num_neg_in):
        s1_id = random.choice(s1_in_list)
        m_id, m_name, m_addr = random.choice(noisy_pool_in)
        s1_name, s1_addr, _ = s1_records[s1_id]
        neg_results.append({
            "pair_type": "NON-MATCH",
            "country": "India",
            "name_ratio": fuzz.ratio(s1_name.lower(), m_name.lower()) / 100.0,
            "name_token_sort": fuzz.token_sort_ratio(s1_name.lower(), m_name.lower()) / 100.0,
            "name_token_set": fuzz.token_set_ratio(s1_name.lower(), m_name.lower()) / 100.0,
            "addr_ratio": (fuzz.ratio(s1_addr.lower(), m_addr.lower()) / 100.0) if s1_addr and m_addr else 0.0,
            "addr_token_sort": (fuzz.token_sort_ratio(s1_addr.lower(), m_addr.lower()) / 100.0) if s1_addr and m_addr else 0.0,
        })
    df_neg = pd.DataFrame(neg_results)

    df_all = pd.concat([df_true, df_neg], ignore_index=True)

    # 16. Statistics and Separability Table
    print("\n--- 16. Similarity Distribution & Hard Case Analysis ---")
    summary_stats = []
    for ctry in ["ALL", "US", "India"]:
        sub_true = df_true if ctry == "ALL" else df_true[df_true["country"] == ctry]
        sub_neg = df_neg if ctry == "ALL" else df_neg[df_neg["country"] == ctry]
        
        # Hard cases where name_ratio < 0.6
        hard_name_raw = (sub_true["name_ratio"] < 0.6).mean() * 100
        hard_name_sort = (sub_true["name_token_sort"] < 0.6).mean() * 100
        hard_name_set = (sub_true["name_token_set"] < 0.6).mean() * 100
        
        # Hard address cases
        hard_addr_sort = (sub_true["addr_token_sort"] < 0.6).mean() * 100
        
        # Cases where both name AND address are < 0.6
        hard_both = ((sub_true["name_token_set"] < 0.6) & (sub_true["addr_token_sort"] < 0.6)).mean() * 100
        
        # False alarm rate in negatives at 0.6
        fp_name_set = (sub_neg["name_token_set"] >= 0.6).mean() * 100
        fp_addr_sort = (sub_neg["addr_token_sort"] >= 0.6).mean() * 100
        
        summary_stats.append({
            "country": ctry,
            "true_sample_size": len(sub_true),
            "neg_sample_size": len(sub_neg),
            "mean_true_name_ratio": round(sub_true["name_ratio"].mean(), 3),
            "mean_neg_name_ratio": round(sub_neg["name_ratio"].mean(), 3),
            "mean_true_name_token_set": round(sub_true["name_token_set"].mean(), 3),
            "mean_neg_name_token_set": round(sub_neg["name_token_set"].mean(), 3),
            "mean_true_addr_token_sort": round(sub_true["addr_token_sort"].mean(), 3),
            "mean_neg_addr_token_sort": round(sub_neg["addr_token_sort"].mean(), 3),
            "true_name_ratio_lt_0.6_pct": round(hard_name_raw, 2),
            "true_name_token_sort_lt_0.6_pct": round(hard_name_sort, 2),
            "true_name_token_set_lt_0.6_pct": round(hard_name_set, 2),
            "true_addr_token_sort_lt_0.6_pct": round(hard_addr_sort, 2),
            "true_both_lt_0.6_pct": round(hard_both, 2),
            "neg_name_token_set_ge_0.6_pct": round(fp_name_set, 2),
            "neg_addr_token_sort_ge_0.6_pct": round(fp_addr_sort, 2),
        })

    df_stats = pd.DataFrame(summary_stats)
    df_stats.to_csv(os.path.join(OUTPUT_DIR, "table_D16_similarity_separability.csv"), index=False)
    print(df_stats.to_string())

    # Plot Histograms of True vs Non-True
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Name Raw Ratio
    axes[0, 0].hist(df_true["name_ratio"], bins=50, alpha=0.6, label="True Pairs", color="blue", density=True)
    axes[0, 0].hist(df_neg["name_ratio"], bins=50, alpha=0.6, label="Random Non-Matches", color="red", density=True)
    axes[0, 0].set_title("Name Levenshtein Ratio (True vs Non-Match)")
    axes[0, 0].set_xlabel("Similarity [0 - 1.0]")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. Name Token Set Ratio
    axes[0, 1].hist(df_true["name_token_set"], bins=50, alpha=0.6, label="True Pairs", color="blue", density=True)
    axes[0, 1].hist(df_neg["name_token_set"], bins=50, alpha=0.6, label="Random Non-Matches", color="red", density=True)
    axes[0, 1].set_title("Name Token Set Ratio (True vs Non-Match)")
    axes[0, 1].set_xlabel("Similarity [0 - 1.0]")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. Address Token Sort Ratio
    axes[1, 0].hist(df_true["addr_token_sort"], bins=50, alpha=0.6, label="True Pairs", color="green", density=True)
    axes[1, 0].hist(df_neg["addr_token_sort"], bins=50, alpha=0.6, label="Random Non-Matches", color="orange", density=True)
    axes[1, 0].set_title("Address Token Sort Ratio (True vs Non-Match)")
    axes[1, 0].set_xlabel("Similarity [0 - 1.0]")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 4. Joint Scatter / 2D Density of True Pairs (Name vs Address)
    axes[1, 1].scatter(df_true["name_token_set"][:1500], df_true["addr_token_sort"][:1500], alpha=0.3, color="purple", s=10)
    axes[1, 1].axvline(0.6, color="red", linestyle="--", label="Name threshold 0.6")
    axes[1, 1].axhline(0.6, color="blue", linestyle="--", label="Addr threshold 0.6")
    axes[1, 1].set_title("True Pairs: Name Token Set vs Address Token Sort")
    axes[1, 1].set_xlabel("Name Token Set Ratio")
    axes[1, 1].set_ylabel("Address Token Sort Ratio")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "plot_D_similarity_separability.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nHistogram plot saved to {plot_path}")

if __name__ == "__main__":
    main()
