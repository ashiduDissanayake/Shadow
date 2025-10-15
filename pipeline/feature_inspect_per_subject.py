#!/usr/bin/env python3
"""
Inspect features per subject without removing anything.

This script loads ALL_features.parquet and, for each subject, saves:
- A per-subject file with the selected feature columns (optionally filtered by prefixes) and meta columns
- A per-subject summary JSON with simple stats per feature (NaN fraction, mean, std, min, max, most-frequent ratio, unique count)

Env vars:
- INSPECT_SUBJECTS="S2,S3"     Restrict to a subset of subjects (default: all subjects in data)
- INSPECT_PREFIXES="EDA.,ACC."  Comma-separated feature prefixes to include (default: include all features)
- INSPECT_LIMIT_ROWS=0           If >0, limit number of rows per subject written (for quick review)
- INSPECT_EXPORT_FORMAT=parquet  parquet|csv (default parquet)
- INSPECT_INCLUDE_META=1         Include meta columns [subject,start,label] in per-subject export (default 1)
- INSPECT_VERBOSE=1              Print extra info

Outputs under pipeline/features/inspection/:
- rows_per_subject/{subject}.{parquet|csv}
- summaries_per_subject/{subject}_summary.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = PIPELINE_DIR / "features"
ALL_PATH = FEATURES_DIR / "ALL_features.parquet"
OUT_BASE = FEATURES_DIR / "inspection"
OUT_ROWS = OUT_BASE / "rows_per_subject"
OUT_SUMM = OUT_BASE / "summaries_per_subject"

META_COLS = ["subject", "start", "label"]


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip() in ("1", "true", "True", "yes", "y")


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def _most_freq_ratio(s: pd.Series) -> float:
    vc = s.dropna().value_counts(dropna=True)
    if vc.empty:
        return 1.0
    return float(vc.iloc[0] / max(1, vc.sum()))


def _summarize_subject(df_subj: pd.DataFrame, feat_cols: List[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for c in feat_cols:
        s = df_subj[c]
        summary[c] = {
            "na_frac": float(s.isna().mean()),
            "mean": float(np.nanmean(s.values)) if s.size else float("nan"),
            "std": float(np.nanstd(s.values)) if s.size else float("nan"),
            "min": float(np.nanmin(s.values)) if s.size else float("nan"),
            "max": float(np.nanmax(s.values)) if s.size else float("nan"),
            "most_freq_ratio": _most_freq_ratio(s),
            "unique_count": int(s.nunique(dropna=True)),
        }
    return summary


def main():
    verbose = _env_bool("INSPECT_VERBOSE", False)
    if not ALL_PATH.exists():
        raise FileNotFoundError(f"Not found: {ALL_PATH}")

    OUT_ROWS.mkdir(parents=True, exist_ok=True)
    OUT_SUMM.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ALL_PATH)
    all_subjects = sorted(df["subject"].unique())

    # Subject filter
    only_raw = os.getenv("INSPECT_SUBJECTS", "").strip()
    if only_raw:
        only_set = {s.strip() for s in only_raw.split(',') if s.strip()}
        subjects = [s for s in all_subjects if s in only_set]
    else:
        subjects = all_subjects

    # Feature prefix filter
    prefixes_raw = os.getenv("INSPECT_PREFIXES", "").strip()
    feat_all = _feature_cols(df)
    if prefixes_raw:
        prefs = [p.strip() for p in prefixes_raw.split(',') if p.strip()]
        feat_cols = [c for c in feat_all if any(c.startswith(p) for p in prefs)]
    else:
        feat_cols = feat_all

    limit_rows = int(os.getenv("INSPECT_LIMIT_ROWS", "0") or 0)
    fmt = os.getenv("INSPECT_EXPORT_FORMAT", "parquet").lower()
    include_meta = _env_bool("INSPECT_INCLUDE_META", True)

    if verbose:
        print(f"Subjects: {len(subjects)} of {len(all_subjects)} total")
        print(f"Feature columns selected: {len(feat_cols)} of {len(feat_all)} total")

    for subj in subjects:
        df_subj = df.loc[df["subject"] == subj].reset_index(drop=True)
        cols = (META_COLS + feat_cols) if include_meta else feat_cols
        df_out = df_subj[cols]
        if limit_rows > 0:
            df_out = df_out.head(limit_rows)

        # Save per-subject rows
        out_path = OUT_ROWS / f"{subj}.{('csv' if fmt == 'csv' else 'parquet')}"
        if fmt == 'csv':
            df_out.to_csv(out_path, index=False)
        else:
            df_out.to_parquet(out_path, index=False, compression="zstd")

        # Summary JSON
        summary = _summarize_subject(df_subj, feat_cols)
        (OUT_SUMM / f"{subj}_summary.json").write_text(json.dumps(summary, indent=2))

        if verbose:
            print(f"Saved {subj}: rows={len(df_out)} features={len(feat_cols)} -> {out_path}")

    print(f"Done. Outputs in {OUT_BASE}")


if __name__ == "__main__":
    main()
