# Business Entity Resolution — Pipeline

Two-stage entity resolution: TF-IDF/char-n-gram blocking for candidate
generation, then a LightGBM binary classifier scoring each (Source1,
candidate) pair, thresholded to maximize macro F_0.5.

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

**1. Train** — builds blocking candidates on an internal train/val split of
Source1, labels pairs from ground truth, trains the matcher, and sweeps the
decision threshold to maximize macro F_0.5 on the validation fold.

```bash
python3 src/train.py \
    --train-dir dataset/train \
    --model-out model/matcher.txt \
    --threshold-out model/threshold.json \
    --top-k 20
```

Prints the blocking recall ceiling (max achievable recall given the
candidate set) and the tuned val F_0.5 — sanity-check both before
proceeding.

**2. Infer** — runs blocking + trained model over the full test set and
writes both required output files.

```bash
python3 src/infer.py \
    --test-dir dataset/test \
    --model model/matcher.txt \
    --threshold-config model/threshold.json \
    --output-dir output
```

Produces `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

**3. Validate format** (challenge-provided script, not included here):

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**4. Score locally** against a held-out ground truth split (independent of
the leaderboard, implements the exact macro F_0.5 formula from the
challenge spec):

```bash
python3 src/score_local.py \
    --predictions output/matching_results.tsv \
    --ground-truth dataset/train/train_ground_truth.tsv
```

## Pipeline structure

```
src/
├── io_utils.py       # TSV read/write matching the exact challenge format
├── normalize.py       # name/address normalization, postal extraction
├── blocking.py         # candidate generation (TF-IDF NN + sorted-neighborhood)
├── features.py          # pairwise similarity feature engineering
├── train.py               # train/val split, labeling, LightGBM training, threshold tuning
├── infer.py                # test-set inference, writes both output TSVs
└── score_local.py           # standalone macro F_0.5 scorer
```

## Design notes

- **Blocking** unions two signals (TF-IDF char n-gram nearest-neighbor +
  sorted-neighborhood) rather than relying on one, since blocking recall
  is the hard ceiling on final score. `top_k` and the sorted-neighborhood
  window are the main levers to raise recall at the cost of more pairs to
  score.
- **country** is used only as a soft feature (`country_match`), never a
  hard filter — the test set introduces France, unseen in training, so any
  hardcoded `{US, India}` branching would silently break on it.
- **No external data or APIs** are used anywhere in the pipeline (no
  geocoding, no business registry lookups), per the challenge's fair-play
  rules — postal codes are extracted with a local regex heuristic only.
- **Threshold tuning** directly optimizes the challenge's macro F_0.5
  metric (not accuracy or plain F1) on a held-out validation fold, since
  F_0.5 weights precision 2x over recall and the right operating point is
  usually well above 0.5.
- **Model**: LightGBM (MIT license), far under the 8B parameter cap.
- Negatives for training are **hard negatives** — non-matching candidates
  that passed the same blocking stage used at inference — rather than
  random negatives, since those are what the matcher actually has to
  discriminate against at inference time.

## Known limitations / next steps on real data

- `NearestNeighbors(algorithm="brute")` in `blocking.py` is fine for
  moderate pool sizes; swap for FAISS/ANN if Source2/Source3 are large
  enough to make brute-force cosine search slow.
- `train.py` uses a fixed `num_boost_round` — replace with early stopping
  against a proper LightGBM validation set once real data is available.
- Per-pair TF-IDF cosine in `features.py` (`_tfidf_cosine`) refits a
  vectorizer per pair for simplicity; vectorize once over the full
  candidate set for a meaningful speedup at scale.

