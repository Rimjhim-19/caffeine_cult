"""
Section B: Ground-Truth Structure Analysis
Analyzes singletons, match counts, cardinality, orphan rates, country consistency,
and intra-source duplicates.
Outputs saved to eda/output/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "dataset"
OUTPUT_DIR = "eda/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading S1 entity -> country mapping...")
    s1_country = {}
    with open(os.path.join(DATA_DIR, "train", "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline() # header
        for line in fp:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                s1_country[parts[0]] = parts[3]
    print(f"Loaded {len(s1_country):,} S1 entities.")

    print("\nParsing Ground Truth...")
    gt_total_s1 = 0
    singletons_overall = 0
    singletons_by_country = Counter()
    total_by_country = Counter()

    # Match counts per S1
    match_count_dist = Counter() # total matches -> count of S1
    s2_match_count_dist = Counter()
    s3_match_count_dist = Counter()

    overlap_dist = Counter() # 's2_only', 's3_only', 'both', 'none'

    # Cardinality check
    s2_to_s1 = {}
    s3_to_s1 = {}
    s2_mult_mapped = Counter()
    s3_mult_mapped = Counter()

    # Keep all pairs for country consistency and intra-source duplicate check
    matched_s2_set = set()
    matched_s3_set = set()
    
    # Store sample pairs for inspection
    # Also track s1 -> list of (s2_ids, s3_ids)
    s1_to_matches = {}

    with open(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline() # header
        for line in fp:
            gt_total_s1 += 1
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 and parts[1].strip() else ""
            
            ctry = s1_country.get(s1_id, "Unknown")
            total_by_country[ctry] += 1

            if not matched_str:
                singletons_overall += 1
                singletons_by_country[ctry] += 1
                overlap_dist["singleton"] += 1
                match_count_dist[0] += 1
                s2_match_count_dist[0] += 1
                s3_match_count_dist[0] += 1
                continue

            matches = matched_str.split(",")
            s2_matches = [m for m in matches if m.startswith("S2-")]
            s3_matches = [m for m in matches if m.startswith("S3-")]
            
            num_total = len(matches)
            num_s2 = len(s2_matches)
            num_s3 = len(s3_matches)

            match_count_dist[num_total] += 1
            s2_match_count_dist[num_s2] += 1
            s3_match_count_dist[num_s3] += 1

            if num_s2 > 0 and num_s3 > 0:
                overlap_dist["both"] += 1
            elif num_s2 > 0:
                overlap_dist["s2_only"] += 1
            elif num_s3 > 0:
                overlap_dist["s3_only"] += 1

            for m in s2_matches:
                matched_s2_set.add(m)
                if m in s2_to_s1:
                    s2_mult_mapped[m] += 1
                else:
                    s2_to_s1[m] = s1_id

            for m in s3_matches:
                matched_s3_set.add(m)
                if m in s3_to_s1:
                    s3_mult_mapped[m] += 1
                else:
                    s3_to_s1[m] = s1_id

            # Save a subset of multi-match S1s for duplicate checks
            if len(s1_to_matches) < 20000 and (num_s2 >= 2 or num_s3 >= 2):
                s1_to_matches[s1_id] = (s2_matches, s3_matches)

    # 5. Singleton rates
    singleton_rows = [{
        "country": "OVERALL",
        "total_s1": gt_total_s1,
        "singletons": singletons_overall,
        "singleton_rate_pct": (singletons_overall / gt_total_s1) * 100
    }]
    for ctry in sorted(total_by_country.keys()):
        tot = total_by_country[ctry]
        sin = singletons_by_country[ctry]
        singleton_rows.append({
            "country": ctry,
            "total_s1": tot,
            "singletons": sin,
            "singleton_rate_pct": (sin / tot) * 100
        })
    df_singletons = pd.DataFrame(singleton_rows)
    df_singletons.to_csv(os.path.join(OUTPUT_DIR, "table_B5_singletons.csv"), index=False)
    print("\n--- 5. Singleton Rates ---")
    print(df_singletons.to_string())

    # 6. Distribution of matches
    non_singleton_total = gt_total_s1 - singletons_overall
    overlap_rows = [
        {"category": "Singleton (0 matches)", "count": overlap_dist["singleton"], "pct_of_all_s1": (overlap_dist["singleton"] / gt_total_s1) * 100, "pct_of_matched_s1": 0.0},
        {"category": "Both S2 and S3", "count": overlap_dist["both"], "pct_of_all_s1": (overlap_dist["both"] / gt_total_s1) * 100, "pct_of_matched_s1": (overlap_dist["both"] / non_singleton_total) * 100},
        {"category": "S2 Only", "count": overlap_dist["s2_only"], "pct_of_all_s1": (overlap_dist["s2_only"] / gt_total_s1) * 100, "pct_of_matched_s1": (overlap_dist["s2_only"] / non_singleton_total) * 100},
        {"category": "S3 Only", "count": overlap_dist["s3_only"], "pct_of_all_s1": (overlap_dist["s3_only"] / gt_total_s1) * 100, "pct_of_matched_s1": (overlap_dist["s3_only"] / non_singleton_total) * 100},
    ]
    df_overlap = pd.DataFrame(overlap_rows)
    df_overlap.to_csv(os.path.join(OUTPUT_DIR, "table_B6_source_overlap.csv"), index=False)
    print("\n--- 6. Source Overlap ---")
    print(df_overlap.to_string())

    # Distribution of match counts (table B6b)
    match_dist_rows = []
    for k in sorted(set(list(match_count_dist.keys()) + list(s2_match_count_dist.keys()) + list(s3_match_count_dist.keys()))):
        if k > 10 and match_count_dist[k] == 0:
            continue
        match_dist_rows.append({
            "num_matches": k,
            "total_matches_count": match_count_dist[k],
            "s2_matches_count": s2_match_count_dist[k],
            "s3_matches_count": s3_match_count_dist[k],
        })
    df_match_dist = pd.DataFrame(match_dist_rows)
    df_match_dist.to_csv(os.path.join(OUTPUT_DIR, "table_B6_match_count_dist.csv"), index=False)
    print("\n--- 6b. Match Count Distribution (up to 10) ---")
    print(df_match_dist.head(11).to_string())

    # 7. Cardinality check
    print("\n--- 7. Cardinality Check ---")
    print(f"S2 entities mapped to >1 S1: {len(s2_mult_mapped)}")
    print(f"S3 entities mapped to >1 S1: {len(s3_mult_mapped)}")
    cardinality_safe = (len(s2_mult_mapped) == 0 and len(s3_mult_mapped) == 0)
    print(f"Is 1-to-1 assignment on S2/S3 side strictly safe? {cardinality_safe}")

    # 8. Orphan / Unmatched rate per source and country & 9. Country Consistency
    print("\nReading S2 and S3 for Orphan Rates and Country Consistency...")
    s2_country_counts = Counter()
    s2_matched_country_counts = Counter()
    s3_country_counts = Counter()
    s3_matched_country_counts = Counter()

    country_mismatches = []
    total_evaluated_pairs = 0
    consistent_pairs = 0

    # Read S2
    s2_records_sample = {}
    with open(os.path.join(DATA_DIR, "train", "train_source2.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for idx, line in enumerate(fp):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            e_id, b_name, b_addr, ctry = parts[0], parts[1], parts[2], parts[3]
            s2_country_counts[ctry] += 1
            if e_id in matched_s2_set:
                s2_matched_country_counts[ctry] += 1
                total_evaluated_pairs += 1
                s1_id = s2_to_s1[e_id]
                s1_ctry = s1_country.get(s1_id, "Unknown")
                if s1_ctry == ctry:
                    consistent_pairs += 1
                else:
                    country_mismatches.append({
                        "s1_id": s1_id, "s1_country": s1_ctry,
                        "matched_id": e_id, "matched_country": ctry,
                        "source": "S2"
                    })
            if idx < 50000 and e_id in matched_s2_set:
                s2_records_sample[e_id] = (b_name, b_addr)

    # Read S3
    s3_records_sample = {}
    with open(os.path.join(DATA_DIR, "train", "train_source3.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for idx, line in enumerate(fp):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            e_id, b_name, b_addr, ctry = parts[0], parts[1], parts[2], parts[3]
            s3_country_counts[ctry] += 1
            if e_id in matched_s3_set:
                s3_matched_country_counts[ctry] += 1
                total_evaluated_pairs += 1
                s1_id = s3_to_s1[e_id]
                s1_ctry = s1_country.get(s1_id, "Unknown")
                if s1_ctry == ctry:
                    consistent_pairs += 1
                else:
                    country_mismatches.append({
                        "s1_id": s1_id, "s1_country": s1_ctry,
                        "matched_id": e_id, "matched_country": ctry,
                        "source": "S3"
                    })
            if idx < 50000 and e_id in matched_s3_set:
                s3_records_sample[e_id] = (b_name, b_addr)

    # Table B8: Orphan rates
    orphan_rows = []
    for ctry in sorted(set(list(s2_country_counts.keys()) + list(s3_country_counts.keys()))):
        # S2
        tot2 = s2_country_counts[ctry]
        mat2 = s2_matched_country_counts[ctry]
        orph2 = tot2 - mat2
        orphan_rows.append({
            "source": "S2",
            "country": ctry,
            "total_records": tot2,
            "matched_records": mat2,
            "orphan_records": orph2,
            "orphan_rate_pct": (orph2 / tot2) * 100 if tot2 > 0 else 0
        })
        # S3
        tot3 = s3_country_counts[ctry]
        mat3 = s3_matched_country_counts[ctry]
        orph3 = tot3 - mat3
        orphan_rows.append({
            "source": "S3",
            "country": ctry,
            "total_records": tot3,
            "matched_records": mat3,
            "orphan_records": orph3,
            "orphan_rate_pct": (orph3 / tot3) * 100 if tot3 > 0 else 0
        })
    df_orphans = pd.DataFrame(orphan_rows)
    df_orphans.to_csv(os.path.join(OUTPUT_DIR, "table_B8_orphan_rates.csv"), index=False)
    print("\n--- 8. Orphan Rates ---")
    print(df_orphans.to_string())

    # 9. Country Consistency
    consistency_pct = (consistent_pairs / total_evaluated_pairs) * 100 if total_evaluated_pairs > 0 else 0
    print("\n--- 9. Country Consistency ---")
    print(f"Total evaluated pairs: {total_evaluated_pairs:,}")
    print(f"Consistent pairs: {consistent_pairs:,} ({consistency_pct:.6f}%)")
    print(f"Mismatches: {len(country_mismatches)}")
    df_mismatches = pd.DataFrame(country_mismatches)
    df_mismatches.to_csv(os.path.join(OUTPUT_DIR, "table_B9_country_mismatches.csv"), index=False)
    if not df_mismatches.empty:
        print("Sample mismatches:")
        print(df_mismatches.head(10).to_string())
    else:
        print("100.0% Country Consistency! Zero mismatches found!")

    # 10. Intra-source near-duplicate check
    print("\n--- 10. Intra-source Near-Duplicate Analysis ---")
    # For S1 entities with multiple S2 or multiple S3 matches, check name/addr similarity between the co-matched items
    co_s2_sims = []
    co_s3_sims = []
    
    for s1_id, (s2_list, s3_list) in s1_to_matches.items():
        if len(s2_list) >= 2:
            # Check pairwise similarities
            for i in range(len(s2_list)):
                for j in range(i + 1, min(i + 4, len(s2_list))):
                    id_a, id_b = s2_list[i], s2_list[j]
                    if id_a in s2_records_sample and id_b in s2_records_sample:
                        na, aa = s2_records_sample[id_a]
                        nb, ab = s2_records_sample[id_b]
                        name_sim = fuzz.ratio(na.lower(), nb.lower())
                        addr_sim = fuzz.token_sort_ratio(aa.lower(), ab.lower()) if aa and ab else 0
                        co_s2_sims.append((name_sim, addr_sim))
        if len(s3_list) >= 2:
            for i in range(len(s3_list)):
                for j in range(i + 1, min(i + 4, len(s3_list))):
                    id_a, id_b = s3_list[i], s3_list[j]
                    if id_a in s3_records_sample and id_b in s3_records_sample:
                        na, aa = s3_records_sample[id_a]
                        nb, ab = s3_records_sample[id_b]
                        name_sim = fuzz.ratio(na.lower(), nb.lower())
                        addr_sim = fuzz.token_sort_ratio(aa.lower(), ab.lower()) if aa and ab else 0
                        co_s3_sims.append((name_sim, addr_sim))

    s2_high_sim = sum(1 for n, a in co_s2_sims if n >= 80)
    s3_high_sim = sum(1 for n, a in co_s3_sims if n >= 80)
    print(f"Sampled co-matched S2 pairs: {len(co_s2_sims)} (high name sim >= 80: {s2_high_sim} ({s2_high_sim/len(co_s2_sims)*100:.1f}%))" if co_s2_sims else "S2 sample empty")
    print(f"Sampled co-matched S3 pairs: {len(co_s3_sims)} (high name sim >= 80: {s3_high_sim} ({s3_high_sim/len(co_s3_sims)*100:.1f}%))" if co_s3_sims else "S3 sample empty")
    
    # Save Section B Summary Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Orphan rate by source and country
    df_orphans.pivot(index="country", columns="source", values="orphan_rate_pct").plot(kind="bar", ax=axes[0], color=["#3498db", "#e74c3c"])
    axes[0].set_title("Orphan (Distractor) Rate (%) by Country & Source")
    axes[0].set_ylabel("Orphan %")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 2: Match count distribution
    df_match_dist.head(6).set_index("num_matches")[["s2_matches_count", "s3_matches_count"]].plot(kind="bar", ax=axes[1], color=["#2ecc71", "#9b59b6"])
    axes[1].set_title("Match Count per S1 Entity (0 to 5)")
    axes[1].set_xlabel("Number of Matches")
    axes[1].set_ylabel("Count of S1 Entities")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "plot_B_ground_truth_structure.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    main()
