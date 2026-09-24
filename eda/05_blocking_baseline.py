"""
Section E: Quick Blocking Baseline (Candidate Recall & Tradeoff)
Implements chunked sparse TF-IDF (char 3-5 grams) + argpartition top-K blocker.
Evaluates candidate recall at K = 5, 10, 20, 50, 100 across 5 channels:
(a) name only
(b) name + address
(c) address only
(d) same postal code
(e) union of channels
Reports candidate list sizes, tradeoffs, and 30 missed false negatives at K=50.
Outputs saved to eda/output/
"""

import os
import sys
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "dataset"
OUTPUT_DIR = "eda/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

US_ZIP_REGEX = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
IN_PIN_REGEX = re.compile(r'\b([1-9]\d{5})\b')

def extract_postal(addr, country):
    if not addr:
        return ""
    if country == "US":
        m = US_ZIP_REGEX.findall(addr)
        return m[-1] if m else ""
    elif country == "India":
        m = IN_PIN_REGEX.findall(addr)
        return m[-1] if m else ""
    return ""

def get_top_k_indices(query_mat, cand_mat, k=100, chunk_size=200):
    """
    Computes cosine similarity matrix via sparse matrix multiplication in chunks
    and retrieves top-K candidate indices using np.argpartition.
    """
    n_queries = query_mat.shape[0]
    n_cands = cand_mat.shape[0]
    effective_k = min(k, n_cands)
    
    top_indices = []
    
    for start_idx in range(0, n_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, n_queries)
        q_chunk = query_mat[start_idx:end_idx]
        
        # Dense similarity scores for this chunk: (chunk_size, n_cands)
        scores = q_chunk.dot(cand_mat.T).toarray()
        
        chunk_top = []
        for i in range(scores.shape[0]):
            row = scores[i]
            if effective_k < len(row):
                part = np.argpartition(-row, effective_k - 1)[:effective_k]
                part = part[np.argsort(-row[part])]
            else:
                part = np.argsort(-row)
            chunk_top.append(part)
        top_indices.extend(chunk_top)
        
    return top_indices

def run_blocking_for_quadrant(country, source, s1_queries, candidates_pool, gt_mapping):
    print(f"\n--- Running Blocker for {country} - {source} ---")
    print(f"Number of S1 queries: {len(s1_queries):,}")
    print(f"Number of candidates in search pool: {len(candidates_pool):,}")
    
    cand_ids = [c["id"] for c in candidates_pool]
    cand_id_to_idx = {cid: idx for idx, cid in enumerate(cand_ids)}
    idx_to_cand = candidates_pool
    
    # Pre-extract texts
    q_names = [q["name"] for q in s1_queries]
    q_addrs = [q["addr"] for q in s1_queries]
    q_combos = [q["name"] + " " + q["addr"] for q in s1_queries]
    q_postals = [extract_postal(q["addr"], country) for q in s1_queries]

    c_names = [c["name"] for c in candidates_pool]
    c_addrs = [c["addr"] for c in candidates_pool]
    c_combos = [c["name"] + " " + c["addr"] for c in candidates_pool]
    c_postals = [extract_postal(c["addr"], country) for c in candidates_pool]

    # Pre-build postal inverted index
    postal_to_cands = defaultdict(list)
    for idx, p in enumerate(c_postals):
        if p:
            postal_to_cands[p].append(idx)

    # 1. Fit TF-IDF models (char 3-5 grams)
    print("Fitting TF-IDF models (char 3-5 grams)...")
    v_name = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=40000)
    v_combo = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=40000)
    v_addr = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=40000)

    print("Transforming candidates...")
    c_mat_name = v_name.fit_transform(c_names)
    c_mat_combo = v_combo.fit_transform(c_combos)
    c_mat_addr = v_addr.fit_transform(c_addrs)

    print("Transforming queries...")
    q_mat_name = v_name.transform(q_names)
    q_mat_combo = v_combo.transform(q_combos)
    q_mat_addr = v_addr.transform(q_addrs)

    # 2. Retrieve top-100 candidates for each channel
    print("Retrieving top-100 candidates for name channel...")
    top100_name = get_top_k_indices(q_mat_name, c_mat_name, k=100)

    print("Retrieving top-100 candidates for combo channel...")
    top100_combo = get_top_k_indices(q_mat_combo, c_mat_combo, k=100)

    print("Retrieving top-100 candidates for addr channel...")
    top100_addr = get_top_k_indices(q_mat_addr, c_mat_addr, k=100)

    # Postal channel candidate retrieval
    top100_postal = []
    for p in q_postals:
        if p and p in postal_to_cands:
            top100_postal.append(postal_to_cands[p][:100])
        else:
            top100_postal.append([])

    # 3. Evaluate recall at K in [5, 10, 20, 50, 100]
    total_true_matches = sum(len(gt_mapping.get(q["id"], [])) for q in s1_queries)
    print(f"Total ground-truth matches to find: {total_true_matches}")

    k_values = [5, 10, 20, 50, 100]
    results_quadrant = []
    missed_at_50 = []

    for k in k_values:
        channels = {
            "name_only": top100_name,
            "combo_name_addr": top100_combo,
            "addr_only": top100_addr,
            "postal_code": top100_postal
        }

        for ch_name, cand_lists in channels.items():
            recovered = 0
            cand_sizes = []
            for q_idx, q in enumerate(s1_queries):
                true_m = gt_mapping.get(q["id"], set())
                if not true_m:
                    continue
                retrieved_idxs = cand_lists[q_idx][:k]
                cand_sizes.append(len(retrieved_idxs))
                retrieved_ids = {cand_ids[idx] for idx in retrieved_idxs}
                recovered += len(true_m.intersection(retrieved_ids))

            recall = (recovered / total_true_matches) * 100 if total_true_matches > 0 else 0
            avg_size = np.mean(cand_sizes) if cand_sizes else 0
            results_quadrant.append({
                "country": country,
                "source": source,
                "channel": ch_name,
                "K": k,
                "recovered_matches": recovered,
                "total_true_matches": total_true_matches,
                "recall_pct": round(recall, 2),
                "avg_cand_size": round(avg_size, 1)
            })

        # Evaluate Union of all 4 channels at this K
        recovered_union = 0
        union_sizes = []
        for q_idx, q in enumerate(s1_queries):
            true_m = gt_mapping.get(q["id"], set())
            if not true_m:
                continue
            u_idxs = set(top100_name[q_idx][:k]) | set(top100_combo[q_idx][:k]) | set(top100_addr[q_idx][:k]) | set(top100_postal[q_idx][:k])
            union_sizes.append(len(u_idxs))
            retrieved_ids = {cand_ids[idx] for idx in u_idxs}
            recovered_union += len(true_m.intersection(retrieved_ids))
            
            if k == 50:
                missed = true_m - retrieved_ids
                for m_id in missed:
                    if m_id in cand_id_to_idx:
                        c_rec = idx_to_cand[cand_id_to_idx[m_id]]
                        missed_at_50.append({
                            "country": country,
                            "source": source,
                            "s1_id": q["id"],
                            "s1_name": q["name"],
                            "s1_addr": q["addr"],
                            "missed_id": m_id,
                            "missed_name": c_rec["name"],
                            "missed_addr": c_rec["addr"],
                        })

        recall_u = (recovered_union / total_true_matches) * 100 if total_true_matches > 0 else 0
        avg_u_size = np.mean(union_sizes) if union_sizes else 0
        results_quadrant.append({
            "country": country,
            "source": source,
            "channel": "union_all",
            "K": k,
            "recovered_matches": recovered_union,
            "total_true_matches": total_true_matches,
            "recall_pct": round(recall_u, 2),
            "avg_cand_size": round(avg_u_size, 1)
        })

    return results_quadrant, missed_at_50

def main():
    print("Loading Ground Truth and sampling S1 benchmark entities...")
    # Load 500 S1 per country
    s1_all = {}
    with open(os.path.join(DATA_DIR, "train", "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for idx, line in enumerate(fp):
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4:
                s1_all[p[0]] = {"id": p[0], "name": p[1], "addr": p[2], "country": p[3]}
            if idx >= 100000:
                break

    gt_s2 = defaultdict(set)
    gt_s3 = defaultdict(set)
    s1_us_selected = []
    s1_in_selected = []

    with open(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            s1_id = p[0]
            if len(p) < 2 or not p[1].strip() or s1_id not in s1_all:
                continue
            matches = p[1].split(",")
            s2_m = {m for m in matches if m.startswith("S2-")}
            s3_m = {m for m in matches if m.startswith("S3-")}
            if not s2_m and not s3_m:
                continue
                
            ctry = s1_all[s1_id]["country"]
            if ctry == "US" and len(s1_us_selected) < 500:
                s1_us_selected.append(s1_all[s1_id])
                gt_s2[s1_id] = s2_m
                gt_s3[s1_id] = s3_m
            elif ctry == "India" and len(s1_in_selected) < 500:
                s1_in_selected.append(s1_all[s1_id])
                gt_s2[s1_id] = s2_m
                gt_s3[s1_id] = s3_m
            if len(s1_us_selected) >= 500 and len(s1_in_selected) >= 500:
                break

    print(f"Selected {len(s1_us_selected)} US S1 queries and {len(s1_in_selected)} India S1 queries.")

    needed_s2 = set()
    needed_s3 = set()
    for q in s1_us_selected + s1_in_selected:
        needed_s2.update(gt_s2[q["id"]])
        needed_s3.update(gt_s3[q["id"]])

    print(f"Total needed true S2 records: {len(needed_s2):,}; S3 records: {len(needed_s3):,}")

    # Build candidate pools by streaming through ALL of S2 and S3:
    # 1. Guarantee EVERY needed record is included
    # 2. Add up to 20,000 distractors per country
    cand_s2_us = []
    cand_s2_in = []
    cand_s3_us = []
    cand_s3_in = []

    print("Scanning train_source2.tsv to collect 100% of needed records + 20,000 distractors/country...")
    with open(os.path.join(DATA_DIR, "train", "train_source2.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 4:
                continue
            rec = {"id": p[0], "name": p[1], "addr": p[2], "country": p[3]}
            if rec["id"] in needed_s2:
                if rec["country"] == "US":
                    cand_s2_us.append(rec)
                else:
                    cand_s2_in.append(rec)
            else:
                if rec["country"] == "US" and len(cand_s2_us) < 20000:
                    cand_s2_us.append(rec)
                elif rec["country"] == "India" and len(cand_s2_in) < 20000:
                    cand_s2_in.append(rec)

    print("Scanning train_source3.tsv to collect 100% of needed records + 20,000 distractors/country...")
    with open(os.path.join(DATA_DIR, "train", "train_source3.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 4:
                continue
            rec = {"id": p[0], "name": p[1], "addr": p[2], "country": p[3]}
            if rec["id"] in needed_s3:
                if rec["country"] == "US":
                    cand_s3_us.append(rec)
                else:
                    cand_s3_in.append(rec)
            else:
                if rec["country"] == "US" and len(cand_s3_us) < 20000:
                    cand_s3_us.append(rec)
                elif rec["country"] == "India" and len(cand_s3_in) < 20000:
                    cand_s3_in.append(rec)

    print(f"Pool sizes: S2 US: {len(cand_s2_us):,}, S2 India: {len(cand_s2_in):,}, S3 US: {len(cand_s3_us):,}, S3 India: {len(cand_s3_in):,}")

    random.shuffle(cand_s2_us)
    random.shuffle(cand_s2_in)
    random.shuffle(cand_s3_us)
    random.shuffle(cand_s3_in)

    all_blocking_results = []
    all_missed_examples = []

    quadrants = [
        ("US", "S2", s1_us_selected, cand_s2_us, gt_s2),
        ("US", "S3", s1_us_selected, cand_s3_us, gt_s3),
        ("India", "S2", s1_in_selected, cand_s2_in, gt_s2),
        ("India", "S3", s1_in_selected, cand_s3_in, gt_s3),
    ]

    for ctry, src, q_list, cand_pool, gt_map in quadrants:
        res, missed = run_blocking_for_quadrant(ctry, src, q_list, cand_pool, gt_map)
        all_blocking_results.extend(res)
        all_missed_examples.extend(missed)

    # 18 & 20. Save results table
    df_block = pd.DataFrame(all_blocking_results)
    df_block.to_csv(os.path.join(OUTPUT_DIR, "table_E18_blocking_recall.csv"), index=False)
    print("\n--- 18 & 20. Candidate Recall & Candidate Size Summary ---")
    print(df_block[df_block["K"] == 50].to_string())

    # 19. Save 30 Missed Examples at K=50
    df_missed = pd.DataFrame(all_missed_examples)
    if not df_missed.empty:
        df_missed_30 = df_missed.head(30)
        df_missed_30.to_csv(os.path.join(OUTPUT_DIR, "table_E19_missed_at_K50.csv"), index=False)
        
        with open(os.path.join(OUTPUT_DIR, "table_E19_missed_at_K50.md"), "w", encoding="utf-8") as fmd:
            fmd.write("# 30 True Pairs Missed by All Channels at K=50\n\n")
            cols = ["country", "source", "s1_id", "s1_name", "s1_addr", "missed_id", "missed_name", "missed_addr"]
            fmd.write("| " + " | ".join(cols) + " |\n")
            fmd.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            for _, r in df_missed_30[cols].iterrows():
                fmd.write("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |\n")
        print(f"\nSaved {len(df_missed_30)} missed examples to {os.path.join(OUTPUT_DIR, 'table_E19_missed_at_K50.csv')} and .md")
    else:
        print("Zero true matches were missed at K=50 across all quadrants!")

    # Plot Recall vs K for Union channel across quadrants
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df_u = df_block[df_block["channel"] == "union_all"]
    for (ctry, src), group in df_u.groupby(["country", "source"]):
        axes[0].plot(group["K"], group["recall_pct"], marker="o", label=f"{ctry} - {src}")
    axes[0].set_title("Union Channel: Candidate Recall vs K")
    axes[0].set_xlabel("Top-K per Channel")
    axes[0].set_ylabel("Candidate Recall (%)")
    axes[0].set_ylim(80, 101)
    axes[0].axhline(97.0, color="red", linestyle="--", label="97% Target Recall")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Channel comparison at K=50
    df_k50 = df_block[df_block["K"] == 50]
    p_k50 = df_k50.pivot(index="channel", columns=["country", "source"], values="recall_pct")
    p_k50.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Channel Comparison: Recall (%) at K=50")
    axes[1].set_ylabel("Recall (%)")
    axes[1].set_ylim(0, 105)
    axes[1].axhline(97.0, color="red", linestyle="--", label="97% Target")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "plot_E_blocking_recall.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
