# Business Entity Resolution — Pipeline

Two-stage entity resolution, rewritten against a full EDA of the real
dataset (2.2M S1 / 5M S2 / 5M S3 records in train). See "Design notes"
below for what changed and why.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Expected data layout (not included in this package):

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## Run: end-to-end

**1. Train** — country-partitioned, multi-channel blocking on an internal
train/val split of Source1, labels pairs from ground truth, trains the
matcher, sweeps the decision threshold, and applies the global dedup
assignment before reporting validation macro F_0.5.

```bash
python3 src/train.py \
    --train-dir dataset/train \
    --model-out model/matcher.txt \
    --threshold-out model/threshold.json \
    --top-k 20
```

Watch the printed **blocking recall ceiling** first — this is the hard
cap on achievable F_0.5 regardless of matcher quality. Raise `--top-k`
if it's meaningfully below ~0.97.

**2. Infer** — same pipeline over the full test set, writes both output files.

```bash
python3 src/infer.py \
    --test-dir dataset/test \
    --model model/matcher.txt \
    --threshold-config model/threshold.json \
    --output-dir output
```

**3. Validate format** (challenge-provided script):

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**4. Score locally** against a held-out ground truth split:

```bash
python3 src/score_local.py \
    --predictions output/matching_results.tsv \
    --ground-truth dataset/train/train_ground_truth.tsv
```

## Pipeline structure

```
src/
├── io_utils.py       # TSV read/write matching the exact challenge format
├── normalize.py       # country-aware name/address normalization, postal extraction
├── blocking.py         # country-partitioned, multi-channel candidate generation
├── features.py           # pairwise similarity features (reuses blocking channel scores)
├── assignment.py           # global dedup: each S2/S3 id claimed by at most one S1
├── train.py                  # train/val split, labeling, LightGBM training, threshold tuning
├── infer.py                    # test-set inference, dedup, writes both output TSVs
└── score_local.py                # standalone macro F_0.5 scorer
```

## Design notes (EDA-driven)

The pipeline was rewritten after an EDA of the real training/test data
(`EDA_REPORT.md`, 2.2M S1 / 5M S2 / 5M S3 rows). Key findings and the
resulting design decisions:

- **Scale.** Brute-force nearest-neighbor search (the original approach)
  doesn't run at millions of rows. `blocking.py` now uses sparse TF-IDF
  matrix multiplication (sparse x sparse stays sparse — only n-gram
  overlaps produce nonzero entries) with batched, partial top-K
  extraction, processed per country partition.
- **Country partitioning is a hard filter, not a soft feature.** EDA
  found 0 cross-country true matches across 7.64M ground-truth pairs —
  100.000000% country consistency. Blocking now searches only within the
  same country, which is both recall-safe and a ~2.5x search-space cut.
- **Multi-channel blocking is mandatory, not name-only.** Name-only
  TF-IDF blocking misses 20-28% of true India matches even at K=50 (heavy
  DBA/trade-name divergence from the registered name). The union of
  name-only + address-only + combined(name+address) channels reaches
  97.84-100% recall at K=10. `blocking.py` always builds and unions all
  three.
- **Legal-suffix stripping is end-anchored only.** Words like "co", "sa",
  "associates" are common *inside* real business names, not just as
  trailing legal suffixes — stripping them anywhere (the original bug)
  would corrupt legitimate names. `normalize.py` now only strips a suffix
  that is the last token(s) of the string.
- **Address normalization is country-aware.** "St" means "Street" in
  English addresses but "Saint" in French ones (e.g. "Rue St-Honoré").
  English-only abbreviation expansion is skipped for France.
- **Postal code extraction is country-gated.** India has 0% valid PIN
  codes in S1 addresses (never even attempted there); US extraction is
  end-anchored so it isn't fooled by the 9.3% of US addresses that start
  with a 5-digit street number; France uses a separate, France-only
  "digits before a capitalized city name" heuristic.
- **Per-pair TF-IDF cosine (the original `features.py`) was cut.** Too
  slow at real scale — refitting a vectorizer per pair over 100M+
  candidate pairs isn't feasible. `features.py` now reuses the channel
  similarity scores `blocking.py` already computed as a byproduct of
  candidate generation.
- **Global dedup assignment.** EDA found 0 S2/S3 records mapping to more
  than one S1 entity across 7.64M ground-truth pairs — every noisy record
  belongs to at most one real business. Per-S1-independent thresholding
  does not enforce this on its own (the same S2 record can score above
  threshold for two different S1 entities). `assignment.py` greedily
  keeps each candidate for only the highest-scoring S1 entity that
  claimed it, applied as the last step before writing output.
- **Threshold sweep is unbiased but expected to land high.** F_0.5
  weights precision 2x over recall, and the candidate pool is 26-41%
  distractors — the sweep in `train.py` still checks the full 0.30-0.95
  range rather than assuming a value, but ~0.70-0.75 is the EDA's
  expectation.
- **Model**: LightGBM (MIT license), far under the 8B parameter cap.
- **No external data or APIs** anywhere in the pipeline (no geocoding, no
  business registry lookups), per the challenge's fair-play rules.

## Known limitations / next steps on real data

- `train.py` uses a fixed `num_boost_round` — switch to early stopping
  against a held-out LightGBM eval set once training on the full dataset.
- `row_batch_size` (default 2000) trades memory for speed in blocking —
  tune upward if memory allows, downward if a batch's sparse similarity
  block is too large for available RAM at the real S2/S3 pool sizes.
- `assignment.py` uses greedy (highest-score-wins) rather than exact
  bipartite/Hungarian assignment. This is sufficient here because the
  uniqueness constraint is one-sided (only candidate ids need to be
  unique; an S1 entity keeping multiple candidates is expected — EDA
  found a median of 3 matches per S1, up to 10), but note it as a
  simplification.
- The EDA's French legal-suffix and street-vocabulary findings (SARL,
  SAS, EURL, SA, SCI; rue/avenue/boulevard) are incorporated into
  `normalize.py`'s suffix list, but France is entirely absent from
  training data — validate blocking/matching behavior on France
  specifically via a leave-one-country-out check (train on US, evaluate
  zero-shot on India, and vice versa) before trusting test-time French
  performance, per the EDA's recommended validation strategy.
