"""
blocking.py

Candidate generation. This stage sets the hard recall ceiling for the whole
pipeline: a true match that never appears here can never be recovered by
any matcher, however good.

Design decisions, each tied to a measurement on the real data:

  WORD-LEVEL TOKENS, NOT CHAR N-GRAMS.
    Char 3-5 gram TF-IDF produces a similarity matrix with measured density
    0.21 (name channel) and 0.76 (name+address channel) -- because common
    trigrams appear in nearly every record, "sparse x sparse stays sparse"
    does not hold. At a 2,000-row batch against a 3M pool that is 15 GB and
    55 GB respectively. Word-level tokens measure 0.024 and 0.18, a 10-30x
    reduction, and lose nothing: sharing at least one name-or-address token
    covers 99.99% of true pairs (1 miss in 11,560 sampled).

  COUNTRY IS A HARD PARTITION.
    Verified across all 7,638,365 training ground-truth pairs: zero
    cross-country matches. Partitioning is therefore recall-free and cuts
    the search space ~2.5x. Countries are read from the data, never
    hardcoded -- France appears only at test time and must partition like
    any other label.

  TWO CHANNELS, UNIONED.
    Name-token overlap alone covers 85.66% of true pairs; address-token
    overlap covers 95.57%; the union covers 99.99%. Address carries more
    signal than name here, largely because ~7% of Source 2/3 names are in
    Devanagari, Gurmukhi or Tamil while Source 1 is always Latin -- for
    those records name similarity is structurally zero, and 99.88% of them
    are still reachable through the address.

  SCORE FLOOR, NOT FIXED TOP-K.
    A plain top-K hands every S1 entity exactly K candidates even when the
    best of them has cosine 0.02. That inflates candidate_pairs.tsv and
    wrecks the reduction ratio the organisers audit. Candidates must clear
    both an absolute floor and a floor relative to that row's best match.

  COLUMNAR OUTPUT.
    Scores come back as parallel numpy arrays, not a
    {(s1_id, cand_id): {...}} dict. At ~1.7M x ~60 candidates that dict is
    ~100M tuple-keyed entries -- hundreds of GB of Python objects. The
    arrays go straight to parquet and are read back by the feature stage.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:  # joblib ships with scikit-learn, but don't hard-fail
    _HAS_JOBLIB = False

from normalize import normalize_address, normalize_name

CHANNELS: Tuple[str, ...] = ("name", "address")


@dataclass
class BlockingConfig:
    """Tunable knobs. Defaults are the measured-sane starting point.

    top_k            candidates kept per channel, per source, per S1 entity
    min_sim          absolute cosine floor; below this a candidate is noise
    rel_floor        fraction of the row's best score a candidate must reach
    batch_size       S1 rows per similarity block (memory/speed tradeoff)
    max_df           drop tokens appearing in more than this fraction of the
                     pool -- trims the giant posting lists ("road", "limited")
                     that dominate the sparse product's nonzeros
    min_df           drop hapax tokens; 1 keeps everything (typos are signal)
    """

    top_k: int = 20
    min_sim: float = 0.05
    rel_floor: float = 0.25
    batch_size: int = 500
    max_df: float = 0.05
    min_df: int = 1
    channels: Tuple[str, ...] = CHANNELS

    # Cache of normalized pool text + fitted vectorizer + pool matrix, keyed
    # by (tag, country, source, channel). Building these is ~25 min of fixed
    # cost per run at full scale, independent of how many S1 entities you
    # query -- so it dominates short runs and is pure waste on repeats.
    # Set to None to disable.
    cache_dir: Optional[str] = None

    # Partitions are fully independent (no shared state, no cross-country
    # matches), so they parallelise cleanly. Each worker holds its own pool
    # matrix, so memory scales with n_jobs -- start at 2 if RAM is tight.
    n_jobs: int = 1


@dataclass
class CandidateSet:
    """Long-form candidate table: one row per (S1 entity, candidate) pair.

    Parallel arrays rather than a dict of dicts -- see module docstring.
    `to_frame()` gives the parquet-ready DataFrame the feature stage reads.
    """

    s1_ids: List[str] = field(default_factory=list)
    cand_ids: List[str] = field(default_factory=list)
    sim_name: List[float] = field(default_factory=list)
    sim_address: List[float] = field(default_factory=list)
    rank_name: List[int] = field(default_factory=list)
    rank_address: List[int] = field(default_factory=list)

    def extend(self, other: "CandidateSet") -> None:
        self.s1_ids.extend(other.s1_ids)
        self.cand_ids.extend(other.cand_ids)
        self.sim_name.extend(other.sim_name)
        self.sim_address.extend(other.sim_address)
        self.rank_name.extend(other.rank_name)
        self.rank_address.extend(other.rank_address)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "source1_entity_id": self.s1_ids,
            "candidate_entity_id": self.cand_ids,
            "sim_name": np.asarray(self.sim_name, dtype=np.float32),
            "sim_address": np.asarray(self.sim_address, dtype=np.float32),
            "rank_name": np.asarray(self.rank_name, dtype=np.int16),
            "rank_address": np.asarray(self.rank_address, dtype=np.int16),
        })
        # Listwise features: how this candidate stands relative to the rest
        # of its own S1 entity's shortlist. These are consistently among the
        # strongest inputs to the matcher -- "best of a good set" and "best
        # of a bad set" look identical without them.
        df["sim_best"] = np.maximum(df["sim_name"], df["sim_address"])
        grp = df.groupby("source1_entity_id")["sim_best"]
        df["cand_count"] = grp.transform("size").astype(np.int16)
        df["sim_max_for_s1"] = grp.transform("max").astype(np.float32)
        df["sim_margin"] = (df["sim_max_for_s1"] - df["sim_best"]).astype(np.float32)
        return df

    def to_dict(self) -> Dict[str, List[str]]:
        """{s1_id: [cand_id, ...]} -- the shape candidate_pairs.tsv needs."""
        out: Dict[str, List[str]] = {}
        for s1_id, cand_id in zip(self.s1_ids, self.cand_ids):
            out.setdefault(s1_id, []).append(cand_id)
        return out


def _channel_text(df: pd.DataFrame, channel: str) -> List[str]:
    """Normalized text for one channel. Address normalization is
    country-aware (French 'St' is 'Saint', not 'Street')."""
    if channel == "name":
        return [normalize_name(n) for n in df["business_name"].tolist()]
    if channel == "address":
        return [
            normalize_address(a, c)
            for a, c in zip(df["business_address"].tolist(),
                            df["country"].tolist())
        ]
    if channel == "combo":
        names = [normalize_name(n) for n in df["business_name"].tolist()]
        addrs = [
            normalize_address(a, c)
            for a, c in zip(df["business_address"].tolist(),
                            df["country"].tolist())
        ]
        return [f"{n} {a}".strip() for n, a in zip(names, addrs)]
    raise ValueError(f"unknown channel: {channel}")


def _cache_path(cfg: BlockingConfig, tag: str, channel: str) -> Optional[str]:
    if not cfg.cache_dir:
        return None
    os.makedirs(cfg.cache_dir, exist_ok=True)
    # Vocabulary depends on these, so they belong in the key -- otherwise a
    # stale cache silently reuses a matrix fitted under different settings.
    key = f"{tag}__{channel}__mdf{cfg.max_df}__ndf{cfg.min_df}.pkl"
    return os.path.join(cfg.cache_dir, key)


def _fit_pool_channel(pool_df: pd.DataFrame, channel: str, tag: str,
                      cfg: BlockingConfig):
    """Fitted vectorizer + pool matrix for one channel, cached to disk.

    This is the expensive half of blocking: normalizing millions of pool
    records and fitting TF-IDF over them. It depends only on the pool, never
    on which S1 entities you happen to be querying, so it is computed once
    per (partition, channel) and reused by every later run.
    """
    path = _cache_path(cfg, tag, channel)
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\S+",
        max_df=cfg.max_df,
        min_df=cfg.min_df,
        dtype=np.float32,
    )
    try:
        pool_matrix = vec.fit_transform(_channel_text(pool_df, channel)).tocsr()
    except ValueError:
        return None

    payload = (vec, pool_matrix)
    if path:
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=4)
    return payload


def _fit_channel(pool_texts: Sequence[str], s1_texts: Sequence[str],
                 cfg: BlockingConfig) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Fit TF-IDF on the POOL and transform both sides.

    Fitting on the pool alone (not pool + S1) is deliberate: IDF should
    describe the corpus being searched. It also halves peak memory, and it
    means the same fitted vectorizer can be reused across S1 batches.

    TfidfVectorizer L2-normalizes rows by default, so the dot product of
    two rows IS their cosine similarity -- no separate normalization step,
    and no distance-to-similarity conversion.
    """
    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\S+",     # text is pre-normalized; split on whitespace
        max_df=cfg.max_df,
        min_df=cfg.min_df,
        dtype=np.float32,
    )
    try:
        pool_matrix = vec.fit_transform(pool_texts)
    except ValueError:
        # Every token pruned (tiny or degenerate partition) -- no signal.
        return None, None
    s1_matrix = vec.transform(s1_texts)
    return s1_matrix.tocsr(), pool_matrix.tocsr()


def _row_top_k(data: np.ndarray, indices: np.ndarray,
               cfg: BlockingConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Top-K of one sparse row, after applying both floors.

    Returns (column_indices, scores) sorted best-first so rank is just the
    position. Touches only the row's nonzeros -- never densifies.
    """
    if data.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    keep = data >= cfg.min_sim
    if not keep.any():
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    data, indices = data[keep], indices[keep]

    # Relative floor: a candidate far below this row's best is noise even
    # if it clears the absolute floor.
    keep = data >= (data.max() * cfg.rel_floor)
    data, indices = data[keep], indices[keep]

    if data.size > cfg.top_k:
        sel = np.argpartition(data, -cfg.top_k)[-cfg.top_k:]
        data, indices = data[sel], indices[sel]

    order = np.argsort(-data)
    return indices[order], data[order]


def generate_candidates(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    cfg: Optional[BlockingConfig] = None,
    tag: str = "",
) -> CandidateSet:
    """Candidates from one pool (S2 or S3) for one country partition.

    Both frames must already be filtered to the same country; partitioning
    happens in generate_all_candidates.
    """
    cfg = cfg or BlockingConfig()
    result = CandidateSet()
    if len(s1_df) == 0 or len(pool_df) == 0:
        return result

    s1_ids = s1_df["entity_id"].tolist()
    pool_ids = pool_df["entity_id"].tolist()

    matrices: Dict[str, Tuple[sp.csr_matrix, sp.csr_matrix]] = {}
    for ch in cfg.channels:
        fitted = _fit_pool_channel(pool_df, ch, tag, cfg)
        if fitted is None:
            continue
        vec, pool_m = fitted
        # S1 is the cheap side: transform only, using the pool's vocabulary
        # and IDF. Never refit here -- that would put the two sides in
        # different feature spaces and silently zero out every similarity.
        s1_m = vec.transform(_channel_text(s1_df, ch)).tocsr()
        matrices[ch] = (s1_m, pool_m)
    if not matrices:
        return result

    for start in range(0, len(s1_ids), cfg.batch_size):
        end = min(start + cfg.batch_size, len(s1_ids))

        # Per channel: the batch's sparse similarity block, kept sparse.
        blocks: Dict[str, sp.csr_matrix] = {}
        for ch, (s1_m, pool_m) in matrices.items():
            blocks[ch] = (s1_m[start:end] @ pool_m.T).tocsr()

        for local_i in range(end - start):
            s1_id = s1_ids[start + local_i]

            picked: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            row_lookup: Dict[str, Dict[int, float]] = {}
            for ch, block in blocks.items():
                lo, hi = block.indptr[local_i], block.indptr[local_i + 1]
                idx, sco = _row_top_k(block.data[lo:hi],
                                      block.indices[lo:hi], cfg)
                picked[ch] = (idx, sco)
                # Full row map, so a candidate found by one channel still
                # gets its true similarity under the other channel rather
                # than a fabricated zero.
                row_lookup[ch] = dict(zip(block.indices[lo:hi].tolist(),
                                          block.data[lo:hi].tolist()))

            union: Dict[int, None] = {}
            for ch in picked:
                for j in picked[ch][0].tolist():
                    union[j] = None
            if not union:
                continue

            rank_of = {
                ch: {int(j): r for r, j in enumerate(picked[ch][0].tolist())}
                for ch in picked
            }
            name_lookup = row_lookup.get("name", {})
            addr_lookup = row_lookup.get("address", {})
            name_rank = rank_of.get("name", {})
            addr_rank = rank_of.get("address", {})

            for j in union:
                result.s1_ids.append(s1_id)
                result.cand_ids.append(pool_ids[j])
                result.sim_name.append(float(name_lookup.get(j, 0.0)))
                result.sim_address.append(float(addr_lookup.get(j, 0.0)))
                # 999 = "this channel did not shortlist it", which the tree
                # model can split on cleanly.
                result.rank_name.append(int(name_rank.get(j, 999)))
                result.rank_address.append(int(addr_rank.get(j, 999)))

    return result


def generate_all_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    cfg: Optional[BlockingConfig] = None,
    verbose: bool = True,
) -> CandidateSet:
    """Full blocking stage: partition by country, block against S2 and S3.

    Countries are taken from the data. France is not special-cased and does
    not need to be -- it partitions like US and India, and nothing in this
    module branches on the label's value.
    """
    cfg = cfg or BlockingConfig()

    # Build the partition list first so it can be farmed out. Countries come
    # from the data: at test time France appears here alongside US and India
    # with no code change, because nothing branches on the label's value.
    jobs = []
    for country in s1_df["country"].unique().tolist():
        s1_part = s1_df[s1_df["country"] == country]
        for pool_df, label in ((s2_df, "S2"), (s3_df, "S3")):
            pool_part = pool_df[pool_df["country"] == country]
            jobs.append((f"{country}_{label}", country, label,
                         s1_part, pool_part))

    def _run(tag, country, label, s1_part, pool_part):
        part = generate_candidates(s1_part, pool_part, cfg, tag=tag)
        if verbose:
            print(f"[blocking] {country:>8s} x {label}: "
                  f"{len(s1_part):>8,} S1 x {len(pool_part):>9,} pool "
                  f"-> {len(part.s1_ids):>10,} pairs", flush=True)
        return part

    if cfg.n_jobs > 1 and _HAS_JOBLIB and len(jobs) > 1:
        # Each worker holds its own pool matrix, so peak memory is roughly
        # n_jobs x the single-partition footprint. Halve batch_size if this
        # pushes the machine into swap -- swapping is far slower than running
        # the partitions sequentially would have been.
        parts = Parallel(n_jobs=min(cfg.n_jobs, len(jobs)), backend="loky")(
            delayed(_run)(*job) for job in jobs
        )
    else:
        parts = [_run(*job) for job in jobs]

    result = CandidateSet()
    for part in parts:
        result.extend(part)
    return result


def measure_recall_ceiling(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    s1_ids: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Blocking quality diagnostics.

    IMPORTANT: pass `s1_ids` when scoring a fold. Ground truth covers every
    S1 entity, so measuring train-fold candidates against the full ground
    truth counts the validation fold's true matches as blocking misses and
    understates recall by roughly the validation fraction. That bug made
    the previous pipeline report ~0.78 where the real figure was ~0.97.

    Returns recall (fraction of true matches present in the candidate set),
    mean candidates per entity, and the fraction of entities blocking found
    nothing for.
    """
    if s1_ids is not None:
        keys = set(s1_ids)
        ground_truth = {k: v for k, v in ground_truth.items() if k in keys}

    total_true = found = 0
    total_cands = 0
    empty = 0
    for s1_id, true_ids in ground_truth.items():
        cand_set = set(candidates.get(s1_id, []))
        total_cands += len(cand_set)
        if not cand_set:
            empty += 1
        total_true += len(true_ids)
        found += sum(1 for t in true_ids if t in cand_set)

    n = max(1, len(ground_truth))
    return {
        "recall_ceiling": found / total_true if total_true else 1.0,
        "mean_candidates": total_cands / n,
        "empty_fraction": empty / n,
        "n_entities": len(ground_truth),
        "n_true_matches": total_true,
    }