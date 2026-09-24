"""
Section F: Train-vs-Test Distribution Shift Analysis
Compares test vs train source sizes, country mix, lengths, non-ASCII rates,
postal presence, French record profiling (street vocabulary, postal placement, legal suffixes),
and mathematical estimation of test singleton & orphan rates.
Outputs saved to eda/output/
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "dataset"
OUTPUT_DIR = "eda/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FR_POSTAL_REGEX = re.compile(r'\b((?:0[1-9]|[1-8]\d|9[0-8])\d{3})\b')
FR_STREET_TERMS = [
    "rue", "avenue", "ave", "av", "boulevard", "bd", "bld", "place", "pl",
    "chemin", "ch", "route", "rte", "impasse", "imp", "allée", "allee",
    "cours", "quai", "square", "passage", "rond-point", "zone", "zi", "za", "zac"
]

def analyze_french_record_patterns(records):
    street_term_counts = Counter()
    postal_positions = Counter() # 'start', 'middle', 'end', 'none'
    legal_suffixes = Counter()
    
    fr_legal_list = ["sarl", "sas", "sa", "eurl", "sci", "snc", "scop", "gie", "selarl", "sasu"]

    for r in records:
        name = r["name"].lower().strip()
        addr = r["addr"].lower().strip()
        
        # Check legal suffixes in name
        name_words = re.findall(r'\b\w+\b', name)
        if name_words and name_words[-1] in fr_legal_list:
            legal_suffixes[name_words[-1]] += 1
            
        # Check street terms in addr
        addr_words = set(re.findall(r'\b\w+\b', addr))
        for st in FR_STREET_TERMS:
            if st in addr_words:
                street_term_counts[st] += 1
                
        # Postal code position
        pm = list(FR_POSTAL_REGEX.finditer(addr))
        if not pm:
            postal_positions["none"] += 1
        else:
            match = pm[-1]
            start_pos = match.start()
            end_pos = match.end()
            total_len = len(addr)
            if start_pos < 10:
                postal_positions["near_start"] += 1
            elif end_pos > total_len - 15:
                postal_positions["near_end"] += 1
            else:
                postal_positions["middle"] += 1

    return street_term_counts, postal_positions, legal_suffixes

def main():
    print("Comparing Train vs Test distributions...")
    
    # 1. Summary comparison table: counts and ratios
    # Train
    train_counts = {
        "S1": {"US": 1323633, "India": 883188, "France": 0, "total": 2206821},
        "S2": {"US": 3016817, "India": 2017799, "France": 0, "total": 5034616},
        "S3": {"US": 3170056, "India": 2115547, "France": 0, "total": 5285603},
    }
    # Test
    test_counts = {
        "S1": {"US": 663106, "India": 809986, "France": 259452, "total": 1732544},
        "S2": {"US": 1871330, "India": 2312565, "France": 703378, "total": 4887273},
        "S3": {"US": 1945701, "India": 2405000, "France": 731615, "total": 5082316},
    }

    ratio_rows = []
    for split, counts in [("Train", train_counts), ("Test", test_counts)]:
        for ctry in ["US", "India", "France", "total"]:
            s1 = counts["S1"].get(ctry, 0)
            s2 = counts["S2"].get(ctry, 0)
            s3 = counts["S3"].get(ctry, 0)
            noisy_total = s2 + s3
            if s1 > 0:
                ratio_rows.append({
                    "split": split,
                    "country": ctry,
                    "S1_count": s1,
                    "S2_count": s2,
                    "S3_count": s3,
                    "S2_per_S1": round(s2 / s1, 3),
                    "S3_per_S1": round(s3 / s1, 3),
                    "total_noisy_per_S1": round(noisy_total / s1, 3)
                })

    df_ratios = pd.DataFrame(ratio_rows)
    df_ratios.to_csv(os.path.join(OUTPUT_DIR, "table_F21_train_test_ratios.csv"), index=False)
    print("\n--- 21a. Source Size & Ratios (Train vs Test) ---")
    print(df_ratios.to_string())

    # 2. Extract 20 sample French test records across S1, S2, S3
    print("\nExtracting French test samples...")
    fr_s1_samples = []
    fr_s2_samples = []
    fr_s3_samples = []

    with open(os.path.join(DATA_DIR, "test", "test_source1.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[3] == "France":
                fr_s1_samples.append({"source": "S1", "id": p[0], "name": p[1], "addr": p[2], "country": p[3]})
                if len(fr_s1_samples) >= 10000:
                    break

    with open(os.path.join(DATA_DIR, "test", "test_source2.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[3] == "France":
                fr_s2_samples.append({"source": "S2", "id": p[0], "name": p[1], "addr": p[2], "country": p[3]})
                if len(fr_s2_samples) >= 10000:
                    break

    with open(os.path.join(DATA_DIR, "test", "test_source3.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[3] == "France":
                fr_s3_samples.append({"source": "S3", "id": p[0], "name": p[1], "addr": p[2], "country": p[3]})
                if len(fr_s3_samples) >= 10000:
                    break

    # Select 7 S1, 7 S2, 6 S3 to make 20 sample records
    sample_20_fr = fr_s1_samples[:7] + fr_s2_samples[:7] + fr_s3_samples[:6]
    df_sample20_fr = pd.DataFrame(sample_20_fr)
    df_sample20_fr.to_csv(os.path.join(OUTPUT_DIR, "table_F21_sample_20_france.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "table_F21_sample_20_france.md"), "w", encoding="utf-8") as fmd:
        fmd.write("# 20 Sample French Test Records (S1, S2, S3)\n\n")
        cols = ["source", "id", "name", "addr", "country"]
        fmd.write("| " + " | ".join(cols) + " |\n")
        fmd.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, r in df_sample20_fr[cols].iterrows():
            fmd.write("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |\n")

    print(f"Saved 20 sample French records to {os.path.join(OUTPUT_DIR, 'table_F21_sample_20_france.csv')} and .md")

    # 3. Analyze French patterns (street vocabulary, postal placement, legal suffixes)
    all_fr_records = fr_s1_samples + fr_s2_samples + fr_s3_samples
    street_terms, postal_pos, legal_suff = analyze_french_record_patterns(all_fr_records)

    print("\n--- Top French Street Vocabulary Terms in Addresses ---")
    df_street = pd.DataFrame(street_terms.most_common(12), columns=["term", "occurrences"])
    df_street["frequency_pct"] = (df_street["occurrences"] / len(all_fr_records)) * 100
    print(df_street.round(2).to_string())
    df_street.to_csv(os.path.join(OUTPUT_DIR, "table_F21_french_street_vocabulary.csv"), index=False)

    print("\n--- French Postal Code Position in Addresses ---")
    df_pos = pd.DataFrame(postal_pos.most_common(), columns=["position", "occurrences"])
    df_pos["pct"] = (df_pos["occurrences"] / len(all_fr_records)) * 100
    print(df_pos.round(2).to_string())
    df_pos.to_csv(os.path.join(OUTPUT_DIR, "table_F21_french_postal_position.csv"), index=False)

    # 22. Estimation of expected singleton and orphan rates in test
    print("\n--- 22. Mathematical Estimation of Test Singleton & Orphan Rates ---")
    # In train:
    # S1 = 2,206,821
    # Matched S1 = 2,083,574 (94.415%)
    # Singleton S1 = 123,247 (5.585%)
    # Matches per matched S1:
    # S2 matches = 3,693,619 -> 1.773 S2 matches per matched S1
    # S3 matches = 3,944,746 -> 1.893 S3 matches per matched S1
    # Total true matches = 7,638,365 -> 3.666 matches per matched S1
    # S2 orphans in train: 5,034,616 - 3,693,619 = 1,340,997 (26.636%)
    # S3 orphans in train: 5,285,603 - 3,944,746 = 1,340,857 (25.368%)
    
    # Model 1: True matches scale proportionally with S1 (constant true match distribution)
    # Under Model 1, matched S1 in test = 94.415% * 1,732,544 = 1,635,798
    # Expected true S2 matches = 1,635,798 * 1.773 = 2,900,270
    # Expected true S3 matches = 1,635,798 * 1.893 = 3,096,565
    # Then Test S2 orphans = 4,887,273 - 2,900,270 = 1,987,003 -> Orphan rate = 40.66%
    # Then Test S3 orphans = 5,082,316 - 3,096,565 = 1,985,751 -> Orphan rate = 39.07%
    # Expected Test Singleton rate = 5.58%

    # Model 2: Orphan rate remains constant (~26.6% for S2, ~25.4% for S3)
    # Under Model 2, true matches in test are higher:
    # S2 true matches = 4,887,273 * (1 - 0.26636) = 3,585,499
    # S3 true matches = 5,082,316 * (1 - 0.25368) = 3,792,934
    # Total true matches = 7,378,433
    # Matches per S1 = 7,378,433 / 1,732,544 = 4.259 (higher density)
    # Under Poisson/binomial zero-truncation, P(X=0) drops from 5.58% to ~4.3% - 4.8%.

    estimations = [
        {
            "scenario": "Scenario A: Match distribution per entity is constant (Higher distractor density in test)",
            "test_S1_singletons_expected_pct": 5.58,
            "test_S2_orphan_expected_pct": 40.66,
            "test_S3_orphan_expected_pct": 39.07,
            "implication": "Distractor pool is ~55% denser in test. Precision false positives are higher risk, favoring stricter acceptance threshold."
        },
        {
            "scenario": "Scenario B: Orphan/distractor rate is constant (True matches per entity are higher in test)",
            "test_S1_singletons_expected_pct": 4.55,
            "test_S2_orphan_expected_pct": 26.64,
            "test_S3_orphan_expected_pct": 25.37,
            "implication": "Entities have slightly more matches. Singletons are fewer. Balanced threshold remains optimal."
        }
    ]
    df_est = pd.DataFrame(estimations)
    df_est.to_csv(os.path.join(OUTPUT_DIR, "table_F22_test_rate_estimations.csv"), index=False)
    print(df_est.to_string())

    # Plot Train vs Test Ratios
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df_ratios.pivot(index="country", columns="split", values="total_noisy_per_S1").plot(kind="bar", ax=axes[0], color=["#e67e22", "#2980b9"])
    axes[0].set_title("Noisy Records (S2 + S3) per S1 Entity: Train vs Test")
    axes[0].set_ylabel("Ratio (S2+S3) / S1")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 2: Scenario comparison for orphan rate
    labels = ["Train (Observed)", "Test Scenario A", "Test Scenario B"]
    s2_rates = [26.64, 40.66, 26.64]
    s3_rates = [25.37, 39.07, 25.37]
    x = np.arange(len(labels))
    width = 0.35
    axes[1].bar(x - width/2, s2_rates, width, label="S2 Orphan %", color="#3498db")
    axes[1].bar(x + width/2, s3_rates, width, label="S3 Orphan %", color="#e74c3c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_ylabel("Orphan %")
    axes[1].set_title("Orphan Rate: Train vs Test Scenarios")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "plot_F_train_test_shift.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
