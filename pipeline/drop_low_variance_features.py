#!/usr/bin/env python3
"""
Drop constant and near-constant features across all subjects, based on the training summary.

Inputs:
- pipeline/features/ALL_features.parquet
- pipeline/features/feature_quality_training.csv (default; configurable via FEATURE_QUALITY_CSV)

Logic:
- Read the quality CSV (computed on training subjects only).
- Build the union of features where is_constant==True OR is_near_constant==True.
- Drop those columns from ALL_features for ALL subjects (including holdouts).

Env vars:
- FEATURE_QUALITY_CSV: path to quality CSV (default pipeline/features/feature_quality_training.csv)
- PRUNE_OUT: output filename under pipeline/features (default ALL_features_pruned.parquet)
- PRUNE_VERBOSE=1: print details

Outputs:
- pipeline/features/ALL_features_pruned.parquet
- pipeline/features/dropped_low_variance_features.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = PIPELINE_DIR / "features"
ALL_PATH = FEATURES_DIR / "ALL_features.parquet"


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip() in ("1", "true", "True", "yes", "y")


def main():
    verbose = _env_bool("PRUNE_VERBOSE", True)
    quality_csv = os.getenv("FEATURE_QUALITY_CSV", str(FEATURES_DIR / "feature_quality_training.csv"))
    out_name = os.getenv("PRUNE_OUT", "ALL_features_pruned.parquet")

    if not ALL_PATH.exists():
        raise FileNotFoundError(f"Not found: {ALL_PATH}")
    if not Path(quality_csv).exists():
        raise FileNotFoundError(f"Not found: {quality_csv}")

    # Load inputs
    df_all = pd.read_parquet(ALL_PATH)
    qdf = pd.read_csv(quality_csv)

    # Determine features to drop (union constant or near-constant)
    if not {"feature", "is_constant", "is_near_constant"}.issubset(qdf.columns):
        raise ValueError("Quality CSV missing required columns: feature, is_constant, is_near_constant")

    drop_list: List[str] = sorted(set(qdf.loc[(qdf["is_constant"]) | (qdf["is_near_constant"]), "feature"].tolist()))

    # Drop from ALL_features if present
    present_drop = [c for c in drop_list if c in df_all.columns]
    missing = [c for c in drop_list if c not in df_all.columns]

    pruned = df_all.drop(columns=present_drop)

    out_path = FEATURES_DIR / out_name
    pruned.to_parquet(out_path, index=False, compression="zstd")

    # Save dropped list for reference
    (FEATURES_DIR / "dropped_low_variance_features.json").write_text(json.dumps({
        "dropped": present_drop,
        "missing_in_all": missing,
    }, indent=2))

    if verbose:
        print(f"Dropped {len(present_drop)} features (union of constant/near-constant). Missing in ALL: {len(missing)}")
        if present_drop:
            print("Dropped:")
            for c in present_drop:
                print(f"  - {c}")
        if missing:
            print("Missing in ALL (skipped):")
            for c in missing:
                print(f"  - {c}")
        print(f"Input ALL_features: {df_all.shape} -> Pruned: {pruned.shape}")
        print(f"Saved pruned -> {out_path}")


if __name__ == "__main__":
    main()
