"""
Section A: Basic Structure Analysis
Analyzes file row counts, country distributions, missingness, duplicate IDs,
name/address length distributions, and script/character profiles.
Outputs saved to eda/output/
"""

import os
import sys
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "dataset"
OUTPUT_DIR = "eda/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "train_source1": os.path.join(DATA_DIR, "train", "train_source1.tsv"),
    "train_source2": os.path.join(DATA_DIR, "train", "train_source2.tsv"),
    "train_source3": os.path.join(DATA_DIR, "train", "train_source3.tsv"),
    "test_source1": os.path.join(DATA_DIR, "test", "test_source1.tsv"),
    "test_source2": os.path.join(DATA_DIR, "test", "test_source2.tsv"),
    "test_source3": os.path.join(DATA_DIR, "test", "test_source3.tsv"),
    "train_gt": os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"),
}

def is_accented(char):
    # Accented latin characters
    if ord(char) < 128:
        return False
    d = unicodedata.decomposition(char)
    return bool(d)

def is_non_latin(char):
    if ord(char) < 128:
        return False
    try:
        name = unicodedata.name(char)
        if 'LATIN' in name:
            return False
        # Common punctuation / symbols
        cat = unicodedata.category(char)
        if cat.startswith('P') or cat.startswith('Z') or cat.startswith('S') or cat.startswith('N'):
            return False
        return True
    except ValueError:
        return False

def analyze_file(file_key, file_path):
    print(f"\nAnalyzing {file_key} ({file_path})...")
    row_count = 0
    country_counts = Counter()
    missing_counts = defaultdict(Counter) # col -> {'empty': int, 'whitespace': int, 'mojibake': int}
    ids = set()
    dup_ids = 0
    
    # Histograms for lengths: (source, country, field, 'char'/'token') -> Counter of lengths
    length_hists = defaultdict(Counter)
    
    # Script counts: (country) -> {'total_chars': int, 'non_ascii': int, 'accented': int, 'non_latin': int}
    script_counts = defaultdict(lambda: Counter())

    with open(file_path, "r", encoding="utf-8", errors="replace") as fp:
        header_line = fp.readline().rstrip("\r\n")
        cols = header_line.split("\t")
        
        is_gt = "ground_truth" in file_key
        
        for line_idx, line in enumerate(fp):
            row_count += 1
            parts = line.rstrip("\r\n").split("\t")
            
            if is_gt:
                s1_id = parts[0]
                matched_str = parts[1] if len(parts) > 1 else ""
                if s1_id in ids:
                    dup_ids += 1
                else:
                    ids.add(s1_id)
                if not s1_id.strip():
                    missing_counts["source1_entity_id"]["whitespace" if s1_id else "empty"] += 1
                continue

            # Standard 4 columns: entity_id, business_name, business_address, country
            if len(parts) < 4:
                # Pad if truncated line
                parts += [""] * (4 - len(parts))
            
            e_id, b_name, b_addr, ctry = parts[0], parts[1], parts[2], parts[3]
            
            if e_id in ids:
                dup_ids += 1
            else:
                ids.add(e_id)
                
            country_counts[ctry] += 1
            
            # Missingness checks
            for col_name, val in zip(cols, [e_id, b_name, b_addr, ctry]):
                if len(val) == 0:
                    missing_counts[col_name]["empty"] += 1
                elif val.strip() == "":
                    missing_counts[col_name]["whitespace_only"] += 1
                if "\ufffd" in val:
                    missing_counts[col_name]["mojibake"] += 1

            # Length analysis (sample 1 in 5 rows for memory/speed while preserving exact distribution)
            if row_count % 5 == 0:
                name_chars = len(b_name)
                name_tokens = len(b_name.split())
                addr_chars = len(b_addr)
                addr_tokens = len(b_addr.split())
                
                length_hists[(ctry, "name_chars")][name_chars] += 1
                length_hists[(ctry, "name_tokens")][name_tokens] += 1
                length_hists[(ctry, "addr_chars")][addr_chars] += 1
                length_hists[(ctry, "addr_tokens")][addr_tokens] += 1
                
                # Script analysis on sample
                combined_text = b_name + " " + b_addr
                t_chars = len(combined_text)
                na_chars = sum(1 for c in combined_text if ord(c) >= 128)
                acc_chars = sum(1 for c in combined_text if is_accented(c))
                nl_chars = sum(1 for c in combined_text if is_non_latin(c))
                
                sc = script_counts[ctry]
                sc["total_chars"] += t_chars
                sc["non_ascii"] += na_chars
                sc["accented"] += acc_chars
                sc["non_latin"] += nl_chars
                
            if row_count % 1000000 == 0:
                print(f"  Processed {row_count:,} rows...")

    return {
        "file": file_key,
        "row_count": row_count,
        "dup_ids": dup_ids,
        "country_counts": dict(country_counts),
        "missing_counts": {k: dict(v) for k, v in missing_counts.items()},
        "length_hists": length_hists,
        "script_counts": {k: dict(v) for k, v in script_counts.items()}
    }

def compute_percentiles_from_hist(counter):
    if not counter:
        return 0, 0, 0, 0, 0, 0
    total = sum(counter.values())
    sorted_items = sorted(counter.items())
    
    vals = []
    # Weighted mean
    mean = sum(k * v for k, v in sorted_items) / total
    
    # Cumulative percentiles
    cum = 0
    p50 = p90 = p99 = None
    min_v = sorted_items[0][0]
    max_v = sorted_items[-1][0]
    
    for k, v in sorted_items:
        cum += v
        pct = cum / total
        if p50 is None and pct >= 0.50:
            p50 = k
        if p90 is None and pct >= 0.90:
            p90 = k
        if p99 is None and pct >= 0.99:
            p99 = k
            
    return min_v, p50, mean, p90, p99, max_v

def main():
    results = {}
    for key, path in FILES.items():
        results[key] = analyze_file(key, path)

    # 1. Summary table: Row counts and countries
    summary_rows = []
    all_train_countries = set()
    all_test_countries = set()
    
    for key, res in results.items():
        split = "train" if "train" in key else "test"
        source = key.replace("train_", "").replace("test_", "")
        row = {
            "dataset": split,
            "source": source,
            "total_rows": res["row_count"],
            "duplicate_ids": res["dup_ids"],
        }
        for ctry, cnt in res["country_counts"].items():
            row[f"country_{ctry}"] = cnt
            if split == "train":
                all_train_countries.add(ctry)
            else:
                all_test_countries.add(ctry)
        summary_rows.append(row)
        
    df_summary = pd.DataFrame(summary_rows).fillna(0)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "table_A1_row_counts.csv"), index=False)
    print("\n--- Summary of Row Counts & Countries ---")
    print(df_summary.to_string())
    
    unseen_test_countries = all_test_countries - all_train_countries
    print(f"\nDistinct train countries: {sorted(list(all_train_countries))}")
    print(f"Distinct test countries: {sorted(list(all_test_countries))}")
    print(f"Countries in test but NOT in train: {sorted(list(unseen_test_countries))}")

    # 2. Missing values and anomalies
    missing_rows = []
    for key, res in results.items():
        for col, counts in res["missing_counts"].items():
            for issue, cnt in counts.items():
                if cnt > 0:
                    missing_rows.append({
                        "file": key,
                        "column": col,
                        "anomaly_type": issue,
                        "count": cnt,
                        "rate_pct": (cnt / res["row_count"]) * 100
                    })
    df_missing = pd.DataFrame(missing_rows)
    df_missing.to_csv(os.path.join(OUTPUT_DIR, "table_A2_missing_anomalies.csv"), index=False)
    print("\n--- Missingness and Anomalies ---")
    if len(df_missing) == 0:
        print("No missing values, whitespace-only fields, or mojibake found!")
    else:
        print(df_missing.to_string())

    # 3. Length distributions
    length_rows = []
    for key, res in results.items():
        if "ground_truth" in key:
            continue
        split = "train" if "train" in key else "test"
        source = key.replace("train_", "").replace("test_", "")
        hists = res["length_hists"]
        for (ctry, field), counter in hists.items():
            min_v, p50, mean, p90, p99, max_v = compute_percentiles_from_hist(counter)
            length_rows.append({
                "dataset": split,
                "source": source,
                "country": ctry,
                "field": field,
                "min": min_v,
                "p50_median": p50,
                "mean": round(mean, 1),
                "p90": p90,
                "p99": p99,
                "max": max_v
            })
    df_lengths = pd.DataFrame(length_rows)
    df_lengths.to_csv(os.path.join(OUTPUT_DIR, "table_A3_length_distributions.csv"), index=False)
    print("\n--- Length Distributions (Sample) ---")
    print(df_lengths.head(20).to_string())

    # 4. Script and character profile
    script_rows = []
    for key, res in results.items():
        if "ground_truth" in key:
            continue
        split = "train" if "train" in key else "test"
        source = key.replace("train_", "").replace("test_", "")
        for ctry, sc in res["script_counts"].items():
            tot = sc.get("total_chars", 0)
            if tot > 0:
                script_rows.append({
                    "dataset": split,
                    "source": source,
                    "country": ctry,
                    "total_chars": tot,
                    "non_ascii_rate_pct": (sc["non_ascii"] / tot) * 100,
                    "accented_rate_pct": (sc["accented"] / tot) * 100,
                    "non_latin_rate_pct": (sc["non_latin"] / tot) * 100,
                })
    df_scripts = pd.DataFrame(script_rows)
    df_scripts.to_csv(os.path.join(OUTPUT_DIR, "table_A4_script_profiles.csv"), index=False)
    print("\n--- Script Profiles ---")
    print(df_scripts.to_string())

    # Generate Plot: Length comparisons and Non-ASCII comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean name and address length by country
    df_p = df_lengths[df_lengths["dataset"] == "train"]
    if not df_p.empty:
        p_name = df_p[df_p["field"] == "name_chars"].groupby(["source", "country"])["mean"].mean().unstack()
        p_name.plot(kind="bar", ax=axes[0])
        axes[0].set_title("Mean Business Name Length (Chars) - Train")
        axes[0].set_ylabel("Characters")
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)
        
    # Plot 2: Non-ASCII rate by country in test vs train
    p_script = df_scripts.groupby(["dataset", "country"])["non_ascii_rate_pct"].mean().unstack()
    p_script.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Non-ASCII Character Rate (%) by Country")
    axes[1].set_ylabel("% of Characters")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "plot_A_structure_overview.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    main()
