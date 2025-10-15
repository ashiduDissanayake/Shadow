#!/usr/bin/env python3
"""
Summarize feature quality over training subjects (exclude holdouts) and write a single CSV.

Default holdouts: S10, S11 (configurable via HOLDOUT_SUBJECTS).
We compute for each feature across all rows from the remaining subjects:
- all_nan: whether the column is entirely NaN on training
- na_count, na_frac: total NaNs and fraction
- unique_count: number of unique non-NaN values
- var: variance ignoring NaNs (nanvar)
- most_freq_ratio: dominant value frequency ratio (ignoring NaNs)
- is_constant: unique_count == 1 (ignoring NaNs)
- is_near_constant: most_freq_ratio >= FQ_CONST_FREQ or var < FQ_MIN_VAR
- good_na: na_frac <= FQ_MAX_NA_GOOD

Env vars:
- HOLDOUT_SUBJECTS="S10,S11"  Comma-separated subjects to exclude from training.
- FQ_CONST_FREQ=0.99          Threshold for near-constant by dominant ratio.
- FQ_MIN_VAR=1e-12            Threshold for near-constant by tiny variance.
- FQ_MAX_NA_GOOD=0.5          Threshold for good NaN fraction.
- FQ_PREFIXES="EDA.,ACC."     Optional: restrict features by these prefixes.
- FQ_OUT="feature_quality_training.csv"  Output filename under pipeline/features.
- FQ_VERBOSE=1                Print extra info.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = PIPELINE_DIR / "features"
ALL_PATH = FEATURES_DIR / "ALL_features.parquet"

META_COLS = ["subject", "start", "label"]


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip() in ("1", "true", "True", "yes", "y")


def _parse_subject_list(val: str) -> List[str]:
    if not val:
        return []
    return [s.strip() for s in val.split(',') if s.strip()]


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def _most_freq_ratio(s: pd.Series) -> float:
    vc = s.dropna().value_counts(dropna=True)
    if vc.empty:
        return 1.0
    return float(vc.iloc[0] / max(1, vc.sum()))


def main():
    verbose = _env_bool("FQ_VERBOSE", False)
    if not ALL_PATH.exists():
        raise FileNotFoundError(f"Not found: {ALL_PATH}")

    df = pd.read_parquet(ALL_PATH)
    all_subjects = sorted(df["subject"].unique())

    holdouts_env = os.getenv("HOLDOUT_SUBJECTS", "S10,S11")
    holdouts = set(_parse_subject_list(holdouts_env))
    train_df = df.loc[~df["subject"].isin(holdouts)].reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Training set is empty after excluding holdouts.")

    prefixes_raw = os.getenv("FQ_PREFIXES", "").strip()
    feat_all = _feature_cols(df)
    if prefixes_raw:
        prefs = [p.strip() for p in prefixes_raw.split(',') if p.strip()]
        feat_cols = [c for c in feat_all if any(c.startswith(p) for p in prefs)]
    else:
        feat_cols = feat_all

    const_freq_th = float(os.getenv("FQ_CONST_FREQ", "0.99"))
    min_var = float(os.getenv("FQ_MIN_VAR", "1e-12"))
    max_na_good = float(os.getenv("FQ_MAX_NA_GOOD", "0.5"))
    out_name = os.getenv("FQ_OUT", "feature_quality_training.csv")

    if verbose:
        print(f"Subjects total={len(all_subjects)}; holdouts={sorted(holdouts)}; train_subjects={sorted(set(train_df['subject']))}")
        print(f"Train shape={train_df.shape}; feature columns considered={len(feat_cols)}")
        print(f"Thresholds: const_freq>={const_freq_th}, min_var<{min_var}, good_na<= {max_na_good}")

    rows = []
    n = len(train_df)
    for c in feat_cols:
        s = train_df[c]
        na_count = int(s.isna().sum())
        na_frac = float(na_count / n) if n else 0.0
        unique_count = int(s.nunique(dropna=True))
        var = float(np.nanvar(s.values)) if n else float("nan")
        mfr = _most_freq_ratio(s)
        is_constant = unique_count == 1
        is_near_constant = (mfr >= const_freq_th) or (np.isfinite(var) and var < min_var)
        good_na = na_frac <= max_na_good
        all_nan = na_count == n

        rows.append({
            "feature": c,
            "all_nan": all_nan,
            "na_count": na_count,
            "na_frac": na_frac,
            "unique_count": unique_count,
            "var": var,
            "most_freq_ratio": mfr,
            "is_constant": is_constant,
            "is_near_constant": is_near_constant,
            "good_na": good_na,
        })

    out_df = pd.DataFrame(rows).sort_values(["all_nan", "is_constant", "is_near_constant", "na_frac", "feature"])  # coarse ordering
    out_path = FEATURES_DIR / out_name
    out_df.to_csv(out_path, index=False)

    if verbose:
        print(f"Wrote {len(out_df)} feature rows -> {out_path}")
        # Quick counts
        print("Counts:")
        print("  all_nan:", int(out_df["all_nan"].sum()))
        print("  is_constant:", int(out_df["is_constant"].sum()))
        print("  is_near_constant:", int(out_df["is_near_constant"].sum()))
        print("  good_na:", int(out_df["good_na"].sum()))


if __name__ == "__main__":
    main()
