"""
io_utils.py

Shared I/O helpers for reading the challenge's TSV inputs and writing the
two required output files (matching_results.tsv, candidate_pairs.tsv) in
the exact format the validator / leaderboard expects.

All challenge files are tab-separated. ID lists inside a cell are comma
separated with NO quoting, so we never let pandas' default csv quoting
touch these columns.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import pandas as pd


# --------------------------------------------------------------------------- #
# Reading source files
# --------------------------------------------------------------------------- #

def read_source(path: str) -> pd.DataFrame:
    """
    Read one source file (train_source1.tsv / test_source2.tsv / etc).

    Expected columns: entity_id, business_name, business_address, country.
    Always pass sep="\\t" explicitly -- the challenge PDF calls this out
    because addresses/ID lists contain commas, so a naive read_csv without
    an explicit tab separator silently collapses everything into one column.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    expected = {"entity_id", "business_name", "business_address", "country"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def read_ground_truth(path: str) -> pd.DataFrame:
    """
    Read train_ground_truth.tsv.

    Columns: source1_entity_id, matched_entity_ids (comma-separated, may be
    empty for singletons).
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    expected = {"source1_entity_id", "matched_entity_ids"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def ground_truth_to_dict(gt_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map source1_entity_id -> list of matched entity_ids (possibly empty)."""
    out: Dict[str, List[str]] = {}
    for _, row in gt_df.iterrows():
        raw = row["matched_entity_ids"].strip()
        ids = [x for x in raw.split(",") if x] if raw else []
        out[row["source1_entity_id"]] = ids
    return out


# --------------------------------------------------------------------------- #
# Writing output files
# --------------------------------------------------------------------------- #

def _write_id_list_tsv(path: str, id_col: str, list_col: str,
                        rows: Dict[str, List[str]]) -> None:
    """
    Write a two-column TSV: <id_col>\t<list_col>, where list_col values are
    comma-joined with no quoting. Used for both matching_results.tsv and
    candidate_pairs.tsv since they share the same shape.

    Enforces: one row per key, no duplicate IDs within a list, empty string
    (not "nan"/"None") when the list is empty.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE,
                             escapechar="\\", lineterminator="\n")
        writer.writerow([id_col, list_col])
        for source1_id, matched_ids in rows.items():
            # de-dup while preserving order
            seen = set()
            deduped = []
            for m in matched_ids:
                if m not in seen:
                    seen.add(m)
                    deduped.append(m)
            writer.writerow([source1_id, ",".join(deduped)])


def write_matching_results(path: str, matches: Dict[str, List[str]]) -> None:
    """Write output/matching_results.tsv (the file scored on the leaderboard)."""
    _write_id_list_tsv(path, "source1_entity_id", "matched_entity_ids", matches)


def write_candidate_pairs(path: str, candidates: Dict[str, List[str]]) -> None:
    """Write output/candidate_pairs.tsv (blocking-stage output, unscored)."""
    _write_id_list_tsv(path, "source1_entity_id", "candidate_entity_ids", candidates)
