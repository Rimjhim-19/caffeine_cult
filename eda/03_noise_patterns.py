"""
Section C: Noise-Pattern Study on True Pairs & Format Profiling
Analyzes 40 sample pairs, noise categorizations, address noise,
postal code patterns (US, India, France), and legal suffix inventory.
Outputs saved to eda/output/
"""

import os
import sys
import re
import random
import unicodedata
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "dataset"
OUTPUT_DIR = "eda/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)

COMMON_LEGAL_SUFFIXES = [
    "inc", "incorporated", "llc", "l.l.c.", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "pvt ltd", "private limited", "co", "company", "llp", "plc",
    "gmbh", "sarl", "sas", "sa", "s.a.", "s.a.s.", "s.a.r.l.", "eurl", "sci",
    "associates", "group", "services", "solutions", "enterprises", "holdings"
]

STREET_ABBR_MAP = {
    "st": "street", "ave": "avenue", "rd": "road", "dr": "drive", "blvd": "boulevard",
    "ln": "lane", "ct": "court", "pl": "place", "pkwy": "parkway", "cir": "circle",
    "hwy": "highway", "fwy": "freeway", "sq": "square", "ste": "suite", "apt": "apartment",
    "fl": "floor", "bldg": "building", "dept": "department"
}

INDIAN_LANDMARKS = ["near", "opp", "opposite", "behind", "beside", "adjacent", "front of", "cross", "main", "phase", "sector", "nagar", "colony", "bazaar", "road", "gali"]

US_ZIP_REGEX = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
IN_PIN_REGEX = re.compile(r'\b([1-9]\d{5})\b')
FR_POSTAL_REGEX = re.compile(r'\b((?:0[1-9]|[1-8]\d|9[0-8])\d{3})\b')

def extract_postal(addr, country):
    if not addr:
        return None
    if country == "US":
        m = US_ZIP_REGEX.findall(addr)
        return m[-1] if m else None
    elif country == "India":
        m = IN_PIN_REGEX.findall(addr)
        return m[-1] if m else None
    elif country == "France":
        m = FR_POSTAL_REGEX.findall(addr)
        return m[-1] if m else None
    return None

def strip_legal_suffix(name):
    lower = name.lower().strip()
    words = re.findall(r'\b\w+\b', lower)
    if not words:
        return lower
    # check 2-word then 1-word suffix
    if len(words) >= 2 and f"{words[-2]} {words[-1]}" in COMMON_LEGAL_SUFFIXES:
        return " ".join(words[:-2])
    if len(words) >= 1 and words[-1] in COMMON_LEGAL_SUFFIXES:
        return " ".join(words[:-1])
    return lower

def normalize_abbr(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return " ".join(STREET_ABBR_MAP.get(w, w) for w in words)

def analyze_true_pair_noise(s1_name, s1_addr, s1_ctry, m_name, m_addr, m_src):
    res = {
        "source": m_src,
        "country": s1_ctry,
    }
    
    # 1. Name noise categories
    s1_nl = s1_name.lower().strip()
    m_nl = m_name.lower().strip()
    
    res["name_exact"] = (s1_nl == m_nl)
    
    s1_nosuff = strip_legal_suffix(s1_name)
    m_nosuff = strip_legal_suffix(m_name)
    res["name_match_strip_suffix"] = (s1_nosuff == m_nosuff and not res["name_exact"])
    
    token_sort = fuzz.token_sort_ratio(s1_nl, m_nl)
    raw_ratio = fuzz.ratio(s1_nl, m_nl)
    token_set = fuzz.token_set_ratio(s1_nl, m_nl)
    
    res["name_transposition"] = (token_sort >= 98 and raw_ratio < 95)
    res["name_typo"] = (raw_ratio >= 80 and raw_ratio < 100 and not res["name_transposition"])
    res["name_diff_dba"] = (raw_ratio < 50 and token_set < 60)
    
    # Subset check
    s1_w = set(re.findall(r'\b\w+\b', s1_nl))
    m_w = set(re.findall(r'\b\w+\b', m_nl))
    res["name_subset"] = (s1_w.issubset(m_w) or m_w.issubset(s1_w)) and (s1_w != m_w)
    
    # 2. Address noise categories
    p_s1 = extract_postal(s1_addr, s1_ctry)
    p_m = extract_postal(m_addr, s1_ctry)
    
    res["addr_missing_in_matched"] = (not m_addr.strip())
    res["addr_has_postal_both"] = bool(p_s1 and p_m)
    res["addr_postal_same"] = bool(p_s1 and p_m and p_s1 == p_m)
    res["addr_postal_diff"] = bool(p_s1 and p_m and p_s1 != p_m)
    res["addr_postal_missing_either"] = bool(not p_s1 or not p_m)
    
    # Landmark text (especially for India)
    m_addr_lower = m_addr.lower()
    has_landmark = any(re.search(rf'\b{lm}\b', m_addr_lower) for lm in ["near", "opp", "opposite", "behind", "beside", "adjacent"])
    res["addr_landmark_text"] = has_landmark
    
    # Address reordering
    addr_ts = fuzz.token_sort_ratio(s1_addr.lower(), m_addr.lower()) if s1_addr and m_addr else 0
    addr_raw = fuzz.ratio(s1_addr.lower(), m_addr.lower()) if s1_addr and m_addr else 0
    res["addr_reordered"] = (addr_ts >= 80 and addr_raw < 60)
    
    # Address abbreviation differences
    s1_addr_norm = normalize_abbr(s1_addr)
    m_addr_norm = normalize_abbr(m_addr)
    res["addr_abbr_difference"] = (s1_addr_norm == m_addr_norm and addr_raw < 95)
    
    return res

def main():
    print("Loading Ground Truth and sampling true pairs...")
    # Stratified collection of true pairs: (country, source)
    pairs_by_quadrant = defaultdict(list)
    
    # First get S1 metadata
    s1_data = {}
    with open(os.path.join(DATA_DIR, "train", "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4:
                s1_data[p[0]] = (p[1], p[2], p[3]) # name, addr, country

    # Sample pairs from ground truth
    with open(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            s1_id = p[0]
            if len(p) < 2 or not p[1].strip():
                continue
            matches = p[1].split(",")
            if s1_id not in s1_data:
                continue
            ctry = s1_data[s1_id][2]
            
            for m in matches:
                src = "S2" if m.startswith("S2-") else "S3"
                quadrant = (ctry, src)
                if len(pairs_by_quadrant[quadrant]) < 12500: # 12.5k per quadrant = 50k total pairs
                    pairs_by_quadrant[quadrant].append((s1_id, m))
                    
    print("Sampled target pairs per quadrant:")
    for q, l in pairs_by_quadrant.items():
        print(f"  {q}: {len(l):,} pairs")

    # Map matched_id -> list of s1_ids
    needed_s2 = {}
    needed_s3 = {}
    for (ctry, src), pair_list in pairs_by_quadrant.items():
        for s1_id, m_id in pair_list:
            if src == "S2":
                needed_s2[m_id] = s1_id
            else:
                needed_s3[m_id] = s1_id

    # Retrieve S2 records
    matched_records = {}
    print(f"\nRetrieving {len(needed_s2):,} needed S2 records...")
    with open(os.path.join(DATA_DIR, "train", "train_source2.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[0] in needed_s2:
                matched_records[p[0]] = (p[1], p[2], p[3])

    print(f"Retrieving {len(needed_s3):,} needed S3 records...")
    with open(os.path.join(DATA_DIR, "train", "train_source3.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[0] in needed_s3:
                matched_records[p[0]] = (p[1], p[2], p[3])

    print(f"Total matched records retrieved: {len(matched_records):,}")

    # 11. Extract 40 representative pairs (10 per quadrant) for Item 11 table
    sample_40_rows = []
    for quadrant in [("US", "S2"), ("US", "S3"), ("India", "S2"), ("India", "S3")]:
        pairs = [p for p in pairs_by_quadrant[quadrant] if p[1] in matched_records][:10]
        for s1_id, m_id in pairs:
            s1_name, s1_addr, s1_ctry = s1_data[s1_id]
            m_name, m_addr, m_ctry = matched_records[m_id]
            sample_40_rows.append({
                "country": s1_ctry,
                "source": quadrant[1],
                "s1_id": s1_id,
                "s1_business_name": s1_name,
                "s1_business_address": s1_addr,
                "matched_id": m_id,
                "matched_business_name": m_name,
                "matched_business_address": m_addr,
                "name_ratio": round(fuzz.ratio(s1_name, m_name), 1),
                "token_sort_ratio": round(fuzz.token_sort_ratio(s1_name, m_name), 1),
                "addr_token_sort": round(fuzz.token_sort_ratio(s1_addr, m_addr), 1) if s1_addr and m_addr else 0
            })
    df_sample40 = pd.DataFrame(sample_40_rows)
    df_sample40.to_csv(os.path.join(OUTPUT_DIR, "table_C11_sample_40_true_pairs.csv"), index=False)
    
    # Save a readable markdown table of 40 pairs
    with open(os.path.join(OUTPUT_DIR, "table_C11_sample_40_true_pairs.md"), "w", encoding="utf-8") as fmd:
        fmd.write("# Sample 40 True Pairs Across Quadrants\n\n")
        cols_to_print = ["country", "source", "s1_id", "s1_business_name", "matched_id", "matched_business_name", "s1_business_address", "matched_business_address"]
        fmd.write("| " + " | ".join(cols_to_print) + " |\n")
        fmd.write("| " + " | ".join(["---"] * len(cols_to_print)) + " |\n")
        for _, r in df_sample40[cols_to_print].iterrows():
            fmd.write("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols_to_print) + " |\n")
    print(f"\nSaved 40 sample pairs to {os.path.join(OUTPUT_DIR, 'table_C11_sample_40_true_pairs.csv')} and .md")

    # 12 & 13. Analyze noise categories on the full 50k true pairs
    print("\nCategorizing noise on 50,000 true pairs...")
    noise_results = []
    for quadrant, pairs in pairs_by_quadrant.items():
        ctry, src = quadrant
        for s1_id, m_id in pairs:
            if m_id not in matched_records:
                continue
            s1_name, s1_addr, _ = s1_data[s1_id]
            m_name, m_addr, _ = matched_records[m_id]
            res = analyze_true_pair_noise(s1_name, s1_addr, ctry, m_name, m_addr, src)
            noise_results.append(res)

    df_noise = pd.DataFrame(noise_results)
    
    # Group by (country, source)
    metrics = [
        "name_exact", "name_match_strip_suffix", "name_transposition",
        "name_typo", "name_subset", "name_diff_dba",
        "addr_missing_in_matched", "addr_has_postal_both", "addr_postal_same",
        "addr_postal_diff", "addr_postal_missing_either", "addr_landmark_text",
        "addr_reordered", "addr_abbr_difference"
    ]
    summary_noise = df_noise.groupby(["country", "source"])[metrics].mean() * 100
    summary_noise = summary_noise.round(2)
    summary_noise.to_csv(os.path.join(OUTPUT_DIR, "table_C12_13_noise_categories.csv"))
    print("\n--- 12 & 13. Noise Frequency (%) by Country & Source ---")
    print(summary_noise.to_string())

    # 14. Postal code formats & false positive profiling
    print("\n--- 14. Postal Code Format Profiling & Street Number Confusion ---")
    # Check street number confusion: addresses with a 5 or 6 digit number that appears at the start of an address (street number) vs postal code
    postal_profile = []
    
    # Check US train
    us_addresses = [s1_data[k][1] for k in list(s1_data.keys())[:100000] if s1_data[k][2] == "US"]
    us_has_zip = sum(1 for a in us_addresses if US_ZIP_REGEX.search(a))
    # Number at start (street number) that is 5 digits
    us_start_5digit = sum(1 for a in us_addresses if re.match(r'^\d{5}\b', a.strip()))
    postal_profile.append({
        "country": "US (Train S1)",
        "sample_size": len(us_addresses),
        "has_expected_postal_pct": (us_has_zip / len(us_addresses)) * 100,
        "starts_with_5_or_6_digits_pct": (us_start_5digit / len(us_addresses)) * 100
    })

    # Check India train
    in_addresses = [s1_data[k][1] for k in list(s1_data.keys())[:100000] if s1_data[k][2] == "India"]
    in_has_pin = sum(1 for a in in_addresses if IN_PIN_REGEX.search(a))
    in_start_digits = sum(1 for a in in_addresses if re.match(r'^\d{5,6}\b', a.strip()))
    postal_profile.append({
        "country": "India (Train S1)",
        "sample_size": len(in_addresses),
        "has_expected_postal_pct": (in_has_pin / len(in_addresses)) * 100,
        "starts_with_5_or_6_digits_pct": (in_start_digits / len(in_addresses)) * 100
    })

    # Check France in test
    fr_addresses = []
    with open(os.path.join(DATA_DIR, "test", "test_source1.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[3] == "France":
                fr_addresses.append(p[2])
                if len(fr_addresses) >= 100000:
                    break
    fr_has_postal = sum(1 for a in fr_addresses if FR_POSTAL_REGEX.search(a))
    fr_start_digits = sum(1 for a in fr_addresses if re.match(r'^\d{5}\b', a.strip()))
    postal_profile.append({
        "country": "France (Test S1)",
        "sample_size": len(fr_addresses),
        "has_expected_postal_pct": (fr_has_postal / len(fr_addresses)) * 100,
        "starts_with_5_or_6_digits_pct": (fr_start_digits / len(fr_addresses)) * 100
    })
    
    df_postal = pd.DataFrame(postal_profile)
    df_postal.to_csv(os.path.join(OUTPUT_DIR, "table_C14_postal_profiling.csv"), index=False)
    print(df_postal.to_string())

    # 15. Legal suffix inventory & position analysis
    print("\n--- 15. Legal Suffix Inventory & Position Analysis ---")
    suffix_candidates = [
        "inc", "llc", "corp", "corporation", "ltd", "limited", "pvt", "llp", "plc",
        "sarl", "sas", "sa", "eurl", "sci", "co", "associates", "group"
    ]
    
    # Count trailing vs middle occurrences
    suffix_stats = []
    
    # Evaluate across samples of US train, India train, and France test
    def evaluate_suffixes(name_list, ctry_label):
        for suf in suffix_candidates:
            trailing_cnt = 0
            middle_cnt = 0
            for nm in name_list:
                tokens = [w.lower() for w in re.findall(r'\b\w+\b', nm)]
                if not tokens:
                    continue
                if tokens[-1] == suf or (len(tokens) >= 2 and f"{tokens[-2]} {tokens[-1]}" == suf):
                    trailing_cnt += 1
                elif suf in tokens[:-1]:
                    middle_cnt += 1
            if trailing_cnt > 0 or middle_cnt > 0:
                suffix_stats.append({
                    "country": ctry_label,
                    "suffix": suf,
                    "trailing_count": trailing_cnt,
                    "trailing_pct": (trailing_cnt / len(name_list)) * 100,
                    "middle_count": middle_cnt,
                    "middle_pct": (middle_cnt / len(name_list)) * 100,
                    "middle_to_trailing_ratio": round(middle_cnt / trailing_cnt, 3) if trailing_cnt > 0 else 0
                })

    us_names = [s1_data[k][0] for k in list(s1_data.keys())[:50000] if s1_data[k][2] == "US"]
    in_names = [s1_data[k][0] for k in list(s1_data.keys())[:50000] if s1_data[k][2] == "India"]
    
    fr_names = []
    with open(os.path.join(DATA_DIR, "test", "test_source1.tsv"), "r", encoding="utf-8", errors="replace") as fp:
        fp.readline()
        for line in fp:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4 and p[3] == "France":
                fr_names.append(p[1])
                if len(fr_names) >= 50000:
                    break

    evaluate_suffixes(us_names, "US (Train S1)")
    evaluate_suffixes(in_names, "India (Train S1)")
    evaluate_suffixes(fr_names, "France (Test S1)")
    
    df_suffixes = pd.DataFrame(suffix_stats)
    df_suffixes.to_csv(os.path.join(OUTPUT_DIR, "table_C15_legal_suffixes.csv"), index=False)
    print("Top trailing & middle legal suffixes:")
    print(df_suffixes.sort_values(by=["country", "trailing_count"], ascending=[True, False]).groupby("country").head(5).to_string())

if __name__ == "__main__":
    main()
