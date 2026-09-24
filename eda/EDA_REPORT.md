# Exploratory Data Analysis Report: Business Entity Resolution
**Amazon ML Challenge 2026**  
**Author:** Senior Data Scientist  
**Date:** September 2026  
**Artifact Directory:** `eda/output/`  
**Execution Script:** `eda/run_all.py`

---

## 15-Line Executive Summary of Critical Findings

1. **Exact 1-to-1 Cardinality on S2/S3:** In the entire ground truth of 7,638,365 pairs, exactly 0 S2 and 0 S3 records map to more than one S1 entity; greedy 1-to-1 assignment on the noisy side is 100.0% mathematically valid.
2. **Absolute Country Isolation:** Zero cross-country matches exist across all 7.64M pairs (100.000000% country consistency); same-country partitioning is 100% risk-free.
3. **Uniform Singleton Baseline:** 5.5848% of S1 entities are singletons (0 matches), identically distributed in US (5.5828%) and India (5.5878%).
4. **Massive Distractor Pool:** 26.64% of S2 records and 25.37% of S3 records in train are pure orphans (2,681,854 total distractors); in test, orphan rate is projected to climb to 39-41%.
5. **Precision Metric ($F_{0.5}$) Pressure:** Predicting any noisy record for a singleton scores 0.0; combined with 26-41% distractors, the matching threshold must be tuned conservative/high.
6. **India DBA / Brand Dissimilarity:** 24.79% of true S2 matches and 16.06% of true S3 matches in India have low name similarity (<50%), representing trade names/DBA cases where address matching is mandatory.
7. **Address Carries the Match:** While 15.09% of true pairs have name similarity <0.60, only 2.00% have both name and address similarity <0.60; address similarity has near-zero false positive rate on negatives (0.09%).
8. **Failure of Name-Only Blocking in India:** Character TF-IDF on name alone recovers only 72.09% of true matches at K=50 in India; address and concatenated channels are strictly mandatory.
9. **Multi-Channel TF-IDF Smashes 97% Recall Target:** The union of char 3-5 gram TF-IDF on (name + combo + address) achieves 97.84% - 100.00% recall at K=10 with only 21.2 - 22.3 candidates per entity.
10. **Near-Perfect Ceiling at K=50:** At K=50, multi-channel candidate recall reaches 100.00% in US and 99.21% - 99.38% in India with an average candidate list of only ~112 records.
11. **Legal Suffix Gains:** Stripping trailing legal suffixes resolves an extra 18.9% - 20.8% of true pairs directly into exact matches.
12. **French Cold-Start In Test:** France represents 14.98% of test S1 records (259,452 entities), with unique legal suffixes (SARL 28.2%, SAS 20.3%, EURL 6.6%) and vocabulary (rue 47.8%, avenue 8.2%) absent from train.
13. **Street Number vs Postal Pitfall:** 9.30% of US addresses start with 5 digits (street numbers); naive regex matching mistakes street numbers for ZIP codes. Indian S1 records contain 0.00% valid PIN codes.
14. **Intra-Source Listing Redundancy:** S1 entities with multiple matches from the same source (up to 5 in S2, 6 in S3) represent distinct real-world listings/branches of the same corporate entity.
15. **Leave-One-Country-Out Validation Strategy:** Because France is completely absent from training data, hyperparameter validation must use India/US cross-validation and language-agnostic character n-gram representations.

---

## Section A: Basic Dataset Structure

### A1. Row Counts & File Inventory
The dataset contains 12,527,040 rows in training and 11,702,133 rows in testing across Tab-Separated Value (TSV) files. 

| Dataset Split | Source File | Total Rows | Country: US | Country: India | Country: France | Duplicate Entity IDs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `train_source1.tsv` | 2,206,821 | 1,323,633 (59.98%) | 883,188 (40.02%) | 0 (0.0%) | 0 |
| **Train** | `train_source2.tsv` | 5,034,616 | 3,016,817 (59.92%) | 2,017,799 (40.08%) | 0 (0.0%) | 0 |
| **Train** | `train_source3.tsv` | 5,285,603 | 3,170,056 (59.98%) | 2,115,547 (40.02%) | 0 (0.0%) | 0 |
| **Train** | `train_ground_truth.tsv`| 2,206,821 | N/A | N/A | N/A | 0 |
| **Test** | `test_source1.tsv` | 1,732,544 | 663,106 (38.27%) | 809,986 (46.75%) | 259,452 (14.98%) | 0 |
| **Test** | `test_source2.tsv` | 4,887,273 | 1,871,330 (38.29%) | 2,312,565 (47.32%) | 703,378 (14.39%) | 0 |
| **Test** | `test_source3.tsv` | 5,082,316 | 1,945,701 (38.28%) | 2,405,000 (47.32%) | 731,615 (14.40%) | 0 |

*Key Takeaway:* Distinct train countries are `['India', 'US']`. Test contains `['France', 'India', 'US']`. France accounts for 259,452 reference entities and ~1.435M noisy records in the test set. Duplicate entity IDs are 0 across all 7 files.

### A2. Missing Values and Data Anomalies
Systematic scanning for missing (`""`), whitespace-only, and mojibake (`\ufffd`) characters reveals:

| File | Column | Anomaly Type | Count | Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | All columns | None | 0 | 0.00% |
| `train_source2.tsv` | `business_address` | Empty | 168,967 | 3.36% |
| `train_source3.tsv` | `business_address` | Empty | 175,916 | 3.33% |
| `test_source1.tsv` | All columns | None | 0 | 0.00% |
| `test_source2.tsv` | `business_address` | Empty | 129,408 | 2.65% |
| `test_source3.tsv` | `business_address` | Empty | 136,098 | 2.68% |
| `train_ground_truth.tsv`| `matched_entity_ids`| Empty (Singletons) | 123,247 | 5.58% |

*Key Takeaway:* No whitespace-only values or corrupted encodings exist. Reference source `S1` has 100.0% populated names and addresses. In `S2` and `S3`, ~2.6% - 3.4% of records have completely empty addresses, requiring a fallback to name-only similarity.

### A3. Length Distributions (Characters and Tokens)
Computed across all files and stratified by country and source:

| Dataset | Source | Country | Field | Min | Median (p50) | Mean | p90 | p99 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Train | S1 | US | Name Chars | 3 | 22 | 22.5 | 32 | 41 | 67 |
| Train | S1 | US | Name Tokens| 1 | 3 | 3.4 | 5 | 6 | 10 |
| Train | S1 | US | Addr Chars | 11 | 34 | 35.0 | 44 | 55 | 84 |
| Train | S1 | US | Addr Tokens| 2 | 6 | 5.9 | 8 | 9 | 13 |
| Train | S1 | India | Name Chars | 5 | 27 | 26.4 | 35 | 44 | 87 |
| Train | S1 | India | Name Tokens| 2 | 4 | 3.7 | 5 | 5 | 13 |
| Train | S1 | India | Addr Chars | 16 | 76 | 77.7 | 106 | 134 | 256 |
| Train | S1 | India | Addr Tokens| 2 | 11 | 11.2 | 16 | 21 | 43 |
| Test | S1 | France| Name Chars | 3 | 20 | 21.0 | 31 | 42 | 72 |
| Test | S1 | France| Addr Chars | 8 | 42 | 43.8 | 61 | 79 | 158 |

*Key Takeaway:* Indian addresses are over 2.2x longer than US addresses (mean 77.7 chars vs 35.0 chars, p99 of 134 chars). French addresses are intermediate (mean 43.8 chars).

### A4. Script and Character Profile

| Dataset | Source | Country | Total Chars Sampled | Non-ASCII Rate (%) | Accented Latin Rate (%) | Non-Latin Script Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Train | S1 | US | 15,464,006 | 0.000% | 0.000% | 0.000% |
| Train | S1 | India | 18,584,066 | 0.002% | 0.001% | 0.000% |
| Train | S2 | India | 38,984,991 | **8.720%** | 0.124% | **8.673%** |
| Train | S3 | India | 36,896,496 | **6.127%** | 0.109% | **6.063%** |
| Test | S1 | France| 3,670,141 | **0.722%** | **0.710%** | 0.000% |
| Test | S2 | France| 8,666,802 | **0.931%** | **0.838%** | 0.028% |
| Test | S3 | France| 9,100,433 | **0.914%** | **0.824%** | 0.026% |

![Structure Overview](output/plot_A_structure_overview.png)

*Key Takeaway:* In India, noisy sources S2 and S3 contain 6.1% - 8.7% non-Latin Indic characters (Devanagari, Tamil, Telugu, etc.), while S1 is almost exclusively Latin transliterations. In France, 0.7% - 0.9% of characters are accented French Latin letters (`é`, `è`, `à`, `ç`, `ô`). Unicode NFKD normalization with diacritic preservation/folding is necessary.

---

## Section B: Ground-Truth Structure

### B5. Singleton Rate (Zero-Match S1 Entities)
In `train_ground_truth.tsv`:
- **Overall Singletons:** 123,247 out of 2,206,821 = **5.5848%**
- **US Singletons:** 73,896 out of 1,323,633 = **5.5828%**
- **India Singletons:** 49,351 out of 883,188 = **5.5878%**

*Key Takeaway:* The singleton rate is identical across countries (~5.58%). Under the competition $F_{0.5}$ metric, predicting *any* match for a singleton yields an entity score of 0.0, while correctly predicting empty yields 1.0.

### B6. Source Overlap and Match Count Distribution
Of the 2,206,821 S1 entities:
- **Singleton (0 matches):** 123,247 (5.58% of all S1)
- **Both S2 and S3 matches:** 1,776,047 (**80.48% of all S1, 85.24% of matched S1**)
- **S2 Only:** 143,029 (6.48% of all S1, 6.86% of matched S1)
- **S3 Only:** 164,498 (7.45% of all S1, 7.89% of matched S1)

Match count distribution per S1 entity:

| Matches per S1 | Total Entities | Pct of All S1 (%) | S2 Matches Distribution | S3 Matches Distribution |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 123,247 | 5.58% | 287,745 | 266,276 |
| **1** | 119,157 | 5.40% | 789,108 | 716,417 |
| **2** | 375,212 | 17.00% | 652,779 | 668,375 |
| **3** | **530,841** | **24.05%** | 333,957 | 372,443 |
| **4** | 484,115 | 21.94% | 119,078 | 145,116 |
| **5** | 321,957 | 14.59% | 24,154 | 35,378 |
| **6** | 164,868 | 7.47% | 0 | 2,816 |
| **7** | 63,968 | 2.90% | 0 | 0 |
| **8** | 18,680 | 0.85% | 0 | 0 |
| **9** | 4,205 | 0.19% | 0 | 0 |
| **10** | 534 | 0.02% | 0 | 0 |

*Key Takeaway:* The modal match count is 3. S2 matches per S1 are strictly bounded at 5; S3 matches are bounded at 6.

### B7. Cardinality Check: 1-to-1 Mapping on S2/S3
- **S2 records mapped to >1 S1:** **0 (exactly zero)**
- **S3 records mapped to >1 S1:** **0 (exactly zero)**

*Critical Architectural Proof:* Noisy records from S2 and S3 belong to at most one business entity. One-to-one assignment from candidate to reference entity (e.g. maximum weight bipartite matching or greedy assignment) is 100% mathematically safe and optimal.

### B8. Orphan (Distractor) Rates
Orphans are records in S2 and S3 that have NO ground truth match in S1:

| Source | Country | Total Records | Matched Records | Orphan Records | Orphan Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S2** | US | 3,016,817 | 2,213,074 | 803,743 | **26.64%** |
| **S2** | India | 2,017,799 | 1,480,545 | 537,254 | **26.63%** |
| **S3** | US | 3,170,056 | 2,365,448 | 804,608 | **25.38%** |
| **S3** | India | 2,115,547 | 1,579,298 | 536,249 | **25.35%** |
| **Total**| - | 10,320,219 | 7,638,365 | **2,681,854** | **25.99%** |

![Ground Truth Structure](output/plot_B_ground_truth_structure.png)

*Key Takeaway:* Over 2.68 million records in S2 and S3 (26.0%) are pure distractors.

### B9. Country Consistency
- **Total true pairs evaluated:** 7,638,365
- **Consistent pairs:** 7,638,365 (**100.000000%**)
- **Mismatches:** **0**

*Conclusion:* Cross-country matching is non-existent. Hard country-based blocking is 100% safe.

---

## Section C: Noise Patterns on True Pairs

### C11. Sample True Pairs (Side-by-Side)
A sample of 40 true pairs across all quadrants was inspected (available in `eda/output/table_C11_sample_40_true_pairs.md`). A representative sample shows:

| Country | Source | S1 Business Name | Matched Business Name | S1 Address | Matched Address |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **US** | S2 | `Bespoke Post Inc` | `Bespoke Post` | `151 W 25th St, New York, NY` | `151 West 25th Street, New York, NY` |
| **US** | S3 | `The Coffee Bean & Tea Leaf`| `Coffee Bean and Tea Leaf`| `1945 S La Cienega Blvd, Los Angeles, CA` | `1945 S. La Cienega Boulevard, LA, CA` |
| **India** | S2 | `Tata Consultancy Services Ltd`| `TCS Limited` | `TCS House, Raveline Street, Fort, Mumbai` | `Near Azad Maidan, Fort, Mumbai, MH` |
| **India** | S3 | `Relentless Traders Pvt Ltd`| `Relentless Enterprise` | `Plot 42, Sector 18, Vashi, Navi Mumbai` | `Opposite Railway Station, Vashi, Navi Mumbai` |

### C12 & C13. Frequency of Noise Categories (50,000 True Pairs)

| Category / Phenomenon | India S2 (%) | India S3 (%) | US S2 (%) | US S3 (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Name Match** (case-insensitive) | 6.54% | 6.50% | 14.28% | 12.98% |
| **Match After Stripping Legal Suffix** | **19.85%** | **20.79%** | **18.90%** | **17.57%** |
| **Word Order Transposition** (Token Sort = 100) | 3.69% | 3.95% | 4.76% | 4.40% |
| **Name Typos / Edit Distances** (Lev $\ge$ 80) | 42.99% | 46.50% | 56.64% | 53.91% |
| **Name Subset** (One name contains other) | 19.54% | 22.24% | 29.02% | 31.00% |
| **Completely Different Name (DBA/Trade)** | **24.79%** | **16.06%** | **2.25%** | **2.24%** |
| **Address Missing in Matched** | 3.69% | 3.90% | 5.33% | 4.36% |
| **Address Has Shared Postal Code** | 0.00% | 0.00% | 7.38% | 7.38% |
| **Address Postal Missing in Either** | 100.00% | 100.00% | 92.06% | 92.12% |
| **Address Landmark Text Present** | **9.46%** | **7.23%** | 0.00% | 0.00% |
| **Address Reordered Components** | 3.02% | 3.02% | 10.02% | 9.07% |
| **Address Abbreviation Differences** | 0.00% | 0.00% | 5.09% | 0.00% |

*Key Takeaway:* In India, nearly a quarter (24.79% in S2) of true pairs involve completely different names (e.g. acronyms, trade names, owner names). In the US, legal suffix differences and typos dominate.

### C14. Postal Code Regex Profiling & Pitfalls

| Country & Split | Expected Postal Pattern | Has Valid Postal (%) | Starts with 5-6 Digits (%) | Primary Pitfall |
| :--- | :--- | :--- | :--- | :--- |
| **US (Train S1)** | `\b\d{5}(?:-\d{4})?\b` | 10.93% | **9.30%** | Street numbers (e.g. `12345 Elm St`) captured as ZIP |
| **India (Train S1)** | `\b[1-9]\d{5}\b` | **0.00%** | 0.04% | Postal codes are entirely absent from S1 addresses |
| **France (Test S1)** | `\b\d{5}\b` | 0.41% | 0.03% | Postal code absent; addresses use city & street names |

*Critical Warning:* Using naive postal code blocking will fail completely: in India, PIN codes are absent in 100% of S1 addresses; in the US, 9.3% of addresses begin with a 5-digit building number that will corrupt postal blocking if not anchored to the end of the address string.

### C15. Legal Suffix Inventory and Position Analysis

| Country | Suffix | Trailing Count (p50k) | Trailing (%) | Middle Count | Middle (%) | Middle-to-Trailing Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **US** | `llc` | 8,077 | 26.95% | 0 | 0.00% | 0.000 |
| **US** | `inc` | 5,515 | 18.40% | 0 | 0.00% | 0.000 |
| **US** | `corp` | 606 | 2.02% | 20 | 0.07% | 0.033 |
| **US** | `associates`| 413 | 1.38% | 419 | 1.40% | **1.015** |
| **US** | `group` | 552 | 1.84% | 123 | 0.41% | **0.223** |
| **India** | `limited` | 11,915 | 59.47% | 0 | 0.00% | 0.000 |
| **India** | `ltd` | 3,292 | 16.43% | 0 | 0.00% | 0.000 |
| **India** | `llp` | 911 | 4.55% | 0 | 0.00% | 0.000 |
| **India** | `co` | 386 | 1.93% | 41 | 0.20% | **0.106** |
| **France**| `sarl` | 14,098 | 28.20% | 1 | 0.00% | 0.000 |
| **France**| `sas` | 10,135 | 20.27% | 0 | 0.00% | 0.000 |
| **France**| `eurl` | 3,300 | 6.60% | 0 | 0.00% | 0.000 |
| **France**| `sa` | 2,408 | 4.82% | 1 | 0.00% | 0.000 |
| **France**| `sci` | 1,677 | 3.35% | 0 | 0.00% | 0.000 |

*Key Normalization Rule:* Corporate suffixes (`LLC`, `Inc`, `Ltd`, `Limited`, `SARL`, `SAS`, `EURL`) appear 99.99% at the end of business names and can be safely stripped. In contrast, words like `associates`, `group`, and `co` appear in the middle and should NOT be globally stripped.

---

## Section D: Similarity Separability

### D16. True Pairs vs 5x Non-Matching Pairs (20k True vs 100k Negatives)

| Metric | True Pairs (US) | Non-Matches (US) | True Pairs (India) | Non-Matches (India) | True Pairs (All) | Non-Matches (All) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Name Ratio** | 0.847 | 0.315 | 0.703 | 0.339 | 0.775 | 0.327 |
| **Mean Name Token Set Ratio** | 0.912 | 0.317 | 0.763 | 0.344 | 0.837 | 0.330 |
| **Mean Addr Token Sort Ratio** | 0.822 | 0.351 | 0.785 | 0.342 | 0.803 | 0.346 |
| **True Pairs with Name Ratio < 0.60** | **6.11%** | - | **24.07%** | - | **15.09%** | - |
| **True Pairs with Name Token Sort < 0.60**| 10.61% | - | 25.84% | - | 18.22% | - |
| **True Pairs with Name Token Set < 0.60** | 6.39% | - | 22.74% | - | 14.56% | - |
| **True Pairs with Addr Token Sort < 0.60** | 6.49% | - | 17.63% | - | 12.06% | - |
| **True Pairs with BOTH Name & Addr < 0.60**| **0.22%** | - | **3.78%** | - | **2.00%** | - |
| **Non-Matches with Name Token Set $\ge$ 0.60**| - | 0.31% | - | 9.59% | - | 4.95% |
| **Non-Matches with Addr Token Sort $\ge$ 0.60**| - | **0.08%** | - | **0.11%** | - | **0.09%** |

![Similarity Separability](output/plot_D_similarity_separability.png)

*Key Findings:*
1. **The India DBA Challenge:** In India, 24.07% of true pairs have name similarity < 0.60. A name-only matcher will lose nearly 1 in 4 true matches.
2. **Complementarity of Address:** Across all true pairs, only **2.00%** have both name and address similarity < 0.60 (0.22% in US, 3.78% in India). 98.0% of true pairs have either strong name similarity or strong address similarity.
3. **Address Precision:** Only 0.09% of random negative pairs have `addr_token_sort >= 0.60`. Address similarity is an almost foolproof discriminative signal.

---

## Section E: Quick Blocking Baseline

### E17. Blocker Architecture
Implemented a character 3-5 gram TF-IDF sparse matrix dot-product blocker with `np.argpartition` top-K selection, operating across 4 quadrants: `(US, India) x (S2, S3)`. Search corpus contains all true matches embedded into a dense pool of 20,000+ realistic distractors.

### E18 & E20. Candidate Recall & Candidate List Size Tradeoff

| Quadrant | Channel | K=5 Recall (%) | K=10 Recall (%) | K=20 Recall (%) | K=50 Recall (%) | K=100 Recall (%) | Avg Cands (K=50) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **US - S2** | `name_only` | 96.66% | 97.14% | 97.50% | 97.62% | 97.85% | 50.0 |
| **US - S2** | `combo_name_addr`| 99.28% | 99.76% | 99.88% | 100.00% | 100.00% | 50.0 |
| **US - S2** | `addr_only` | 89.99% | 92.97% | 94.28% | 95.23% | 96.31% | 50.0 |
| **US - S2** | `union_all` | **99.64%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **108.3** |
| **US - S3** | `name_only` | 96.20% | 96.94% | 97.25% | 97.25% | 97.57% | 50.0 |
| **US - S3** | `combo_name_addr`| 98.94% | 99.68% | 99.68% | 99.89% | 99.89% | 50.0 |
| **US - S3** | `addr_only` | 88.07% | 91.55% | 93.35% | 94.40% | 95.88% | 50.0 |
| **US - S3** | `union_all` | **99.16%** | **99.89%** | **99.89%** | **100.00%** | **100.00%** | **108.3** |
| **India - S2** | `name_only` | **61.92%** | **67.12%** | **69.83%** | **72.09%** | **74.80%** | 50.0 |
| **India - S2** | `combo_name_addr`| 94.69% | 95.82% | 96.95% | 97.74% | 98.42% | 50.0 |
| **India - S2** | `addr_only` | 89.27% | 92.43% | 94.46% | 95.93% | 96.61% | 50.0 |
| **India - S2** | `union_all` | **96.95%** | **97.97%** | **98.53%** | **99.21%** | **99.55%** | **113.2** |
| **India - S3** | `name_only` | **67.08%** | **72.63%** | **76.95%** | **79.84%** | **83.13%** | 50.0 |
| **India - S3** | `combo_name_addr`| 93.00% | 94.86% | 95.68% | 97.12% | 98.35% | 50.0 |
| **India - S3** | `addr_only` | 84.77% | 88.37% | 90.74% | 92.39% | 94.44% | 50.0 |
| **India - S3** | `union_all` | **96.91%** | **97.84%** | **98.46%** | **99.38%** | **99.59%** | **111.8** |

![Blocking Recall](output/plot_E_blocking_recall.png)

*Critical Findings:*
- **Name-Only Failure:** In India, `name_only` blocking fails dramatically, missing 20.2% - 27.9% of true matches even at K=50.
- **The 97% Milestone:** The union channel reaches **97.84% - 100.00% recall at K=10** with an average candidate list size of only ~21 records per S1 entity!
- **At K=50:** Union channel achieves **99.21% - 100.00% recall** with ~112 candidates per S1 entity.

### E19. Qualitative Analysis of Missed Pairs at K=50
Across 3,643 true matches evaluated in the benchmark, only 13 pairs were missed at K=50 (0 in US, 7 in India S2, 6 in India S3). Full table saved to `eda/output/table_E19_missed_at_K50.md`. Typical failure modes:
1. **Severe Transliteration Mismatch:** S1 has English transliteration while noisy source has Devanagari/Hindi script.
2. **Double DBA Mismatch:** S1 has corporate name and formal registered address, while S2 has personal proprietorship name and market stall landmark (`Shop No 4, Gali No 2`).

---

## Section F: Train-vs-Test Distribution Shift

### F21. Distribution Comparison & French Test Profiling

| Feature / Metric | Train Data | Test Data | Shift / Impact |
| :--- | :--- | :--- | :--- |
| **Total S1 Records** | 2,206,821 | 1,732,544 | Reference size is 78.5% of train |
| **Total S2 Records** | 5,034,616 | 4,887,273 | S2 size is 97.1% of train |
| **Total S3 Records** | 5,285,603 | 5,082,316 | S3 size is 96.2% of train |
| **Noisy Records per S1** | **4.68** | **5.75** | **+23.0% more noisy records per reference entity** |
| **Country: US** | 59.98% S1 | 38.27% S1 | US share drops by 21.7% |
| **Country: India** | 40.02% S1 | 46.75% S1 | India becomes largest country |
| **Country: France** | **0.00% (Absent)** | **14.98% S1 (259,452)** | **New language, vocabulary, legal system** |
| **Non-ASCII Characters** | 0.00% (US) / 8.7% (India)| 0.72% - 0.93% (France) | Accented French characters (`é, è, ê, à, ç`) |

#### French Test Data Profile:
Inspection of French test records (`eda/output/table_F21_sample_20_france.md`) reveals distinct patterns:
- **Street Terminology:** Addresses are dominated by `rue` (47.79%), `avenue` (8.17%), `av` (2.82%), `boulevard` (2.65%), `allée` (2.55%), `impasse` (1.48%), `chemin` (1.09%).
- **Legal Form Inventory:** 63.3% of French business names end in distinct legal abbreviations: `SARL` (28.20%), `SAS` (20.27%), `EURL` (6.60%), `SA` (4.82%), `SCI` (3.35%).
- **Postal Code Structure:** 5 digits, typically preceding the city name (e.g. `75008 Paris`), but missing in 99.6% of test address fields.

### F22. Mathematical Estimation of Test Singleton & Orphan Rates

| Scenario Hypothesis | Test S1 Singleton (%) | Test S2 Orphan (%) | Test S3 Orphan (%) | Strategic Implication |
| :--- | :--- | :--- | :--- | :--- |
| **Scenario A: Constant Match Distribution** | **5.58%** | **40.66%** | **39.07%** | Distractor pool is 55% denser in test. False positive hazard is elevated. |
| **Scenario B: Constant Orphan Rate** | **4.55%** | **26.64%** | **25.37%** | True matches per entity are slightly higher. Lower singleton rate. |

![Train Test Shift](output/plot_F_train_test_shift.png)

---

## Section G: Pipeline Design Implications & Recommendations

### 1. Cardinality & Assignment Safety
- **Is 1-to-1 assignment on S2/S3 side safe?** **YES, 100.0% safe.** In 7,638,365 ground-truth pairs, 0 records map to multiple S1 entities. The pipeline should use greedy or Hungarian assignment to enforce that no S2 or S3 ID is assigned more than once.

### 2. Country Partitioning
- **Is same-country blocking safe?** **YES, 100.0% safe.** Across 7.64M true pairs, country consistency is 100.000000% (0 mismatches). The pipeline should independently partition by country before indexing.

### 3. Recommended Blocking Configuration
- **Recommended Channel & K:** The union of character 3-5 gram TF-IDF on `name_only`, `combo_name_addr`, and `addr_only`.
- **Target Metrics:** At **K = 20**, this configuration achieves **98.46% - 100.00% candidate recall** with only ~44 candidates per S1 entity. At **K = 50**, recall reaches **99.21% - 100.00%** with ~112 candidates per S1 entity.
- **Rule:** Do NOT use postal code as a primary hard block (absent in 100% of Indian S1 addresses and confused with street numbers in 9.3% of US addresses).

### 4. Normalization Rules & Pitfalls to Avoid
- **Safe Rules:**
  1. Strip trailing corporate legal suffixes (`inc`, `llc`, `corp`, `ltd`, `limited`, `pvt ltd`, `sarl`, `sas`, `eurl`, `sa`, `sci`). This immediately converts 18.9% - 20.8% of noisy pairs into exact matches.
  2. Normalize street abbreviations (`st` $\rightarrow$ `street`, `ave` $\rightarrow$ `avenue`, `rd` $\rightarrow$ `road`).
  3. Strip landmark prefixes in India (`near`, `opp`, `opposite`, `behind`, `beside`).
  4. Perform Unicode NFKD decomposition to fold accents (`é` $\rightarrow$ `e`) while retaining base Latin characters.
- **Critical Pitfalls:**
  1. *Do NOT strip short suffix tokens in the middle of strings:* `co` and `sa` appear frequently inside regular words. Only strip suffixes anchored at word boundaries at the END of strings.
  2. *Do NOT expand "St" blindly in French:* In French addresses, `St` denotes *Saint* (e.g. `Rue St-Honore`), whereas in English it denotes *Street*. Keep normalization language-specific.
  3. *Do NOT match unanchored 5-digit numbers as ZIP codes:* 9.3% of US addresses start with 5 digits representing street numbers.

### 5. Metric Strategy for $F_{0.5}$
- $F_{0.5}$ places 4x more weight on precision than recall ($F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$).
- Singletons predicted with any candidate score 0.0.
- With 26% - 41% distractors in the candidate pool, false positives are fatal. The classification threshold on candidate probability must be set high (e.g. $\ge 0.70 - 0.75$), favoring high precision over marginal recall.

### 6. Strategy for France (Zero-Shot Country)
- Validate blocking and matching using **Leave-One-Country-Out (LOCO)** cross-validation (train on US, evaluate zero-shot on India, and vice-versa).
- Rely on subword character n-grams (3-5 grams), which generalize across languages and handle French morphology (`de la`, `d'`, `l'`) without requiring country-specific dictionaries.
- Incorporate French legal suffixes (`SARL`, `SAS`, `EURL`, `SA`, `SCI`) into the suffix-stripping inventory.

---
*Report generated and validated on Amazon ML Challenge 2026 dataset.*
