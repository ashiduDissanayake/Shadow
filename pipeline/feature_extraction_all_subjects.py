#!/usr/bin/env python3
"""
Run strict feature extraction over all WESAD subjects (S2..S17, excluding S12) using the existing S2 processor.

Inputs (default; can override via FEAT_DATA_DIR):
- pipeline/wesad_windows_parquet_per_subject/S[2..17].parquet

Outputs (in a separate subfolder under pipeline/features):
- {OUT_ROOT}/subjects/SX_features.parquet  (per-subject)
- {OUT_ROOT}/{COMBINED_NAME}               (combined across processed subjects)
- {OUT_ROOT}/run_manifest.json             (what ran, when, env flags, subjects, failures)

Environment flags (forwarded to the S2 extractor where applicable):
- FEAT_DATA_DIR=path         override input dataset directory (abs or relative to pipeline/)
- FEAT_LIMIT                 int, limit rows per subject for smoke tests
- FEAT_SKIP_HRV=1            skip HRV features entirely (keeps BVP stats)
- FEAT_SUPPRESS_WARNINGS=1   suppress numerical RuntimeWarnings from stats
- FEAT_OVERWRITE=1           force recompute even if per-subject output exists
- FEAT_COMBINED_ONLY=1       skip per-subject files, only produce combined
- FEAT_SUBJECTS="S2,S3"      restrict to subset (by subject name)
- FEAT_EXCLUDE="S10,S11"     exclude list (by subject name)
- FEAT_PARQUET_EXT=zstd      parquet compression: zstd|snappy (default zstd)
- FEAT_COMBINED_NAME=ALL_features.parquet   name for combined output (default)
- FEAT_OUT_SUBDIR=strict_run                subfolder under pipeline/features for all outputs (default strict_run)
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from feature_extraction_s2 import process_subject_s2, DATA_DIR, FEATURES_DIR


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip() in ("1", "true", "True", "yes", "y")


def _list_subject_files(data_dir: Path) -> List[Path]:
    files = sorted([p for p in data_dir.glob("S*.parquet") if p.is_file()])
    # Keep only subjects with numeric suffix (e.g., S2..S17)
    files = [p for p in files if p.stem[1:].isdigit()]
    # Optional include/exclude lists
    only = os.getenv("FEAT_SUBJECTS", "").strip()
    exclude = os.getenv("FEAT_EXCLUDE", "").strip()
    if only:
        allowed = {s.strip() for s in only.split(',') if s.strip()}
        files = [p for p in files if p.stem in allowed]
    if exclude:
        banned = {s.strip() for s in exclude.split(',') if s.strip()}
        files = [p for p in files if p.stem not in banned]
    return files


def _prepare_out_dirs() -> Dict[str, Path]:
    # Place all outputs under FEATURES_DIR / FEAT_OUT_SUBDIR
    subdir = os.getenv("FEAT_OUT_SUBDIR", "strict_run")
    out_root = FEATURES_DIR / subdir
    out_subjects = out_root / "subjects"
    out_root.mkdir(parents=True, exist_ok=True)
    out_subjects.mkdir(parents=True, exist_ok=True)
    return {"root": out_root, "subjects": out_subjects}


def main():
    # Locate inputs
    subjects = _list_subject_files(DATA_DIR)
    if not subjects:
        raise FileNotFoundError(f"No subject parquet files found in {DATA_DIR}")
    print(f"Found {len(subjects)} subjects: {[p.stem for p in subjects]}")

    # Output structure
    out_dirs = _prepare_out_dirs()
    OUT_ROOT = out_dirs["root"]
    OUT_SUBJ = out_dirs["subjects"]

    # Run options
    overwrite = _env_bool("FEAT_OVERWRITE", False)
    combined_only = _env_bool("FEAT_COMBINED_ONLY", False)
    compression = os.getenv("FEAT_PARQUET_EXT", "zstd")
    combined_name = os.getenv("FEAT_COMBINED_NAME", "ALL_features.parquet")

    # Inform about FEAT_LIMIT smoke tests
    feat_limit = os.getenv("FEAT_LIMIT")
    if feat_limit:
        print(f"FEAT_LIMIT={feat_limit} set; limiting rows per subject for smoke testing.")

    all_parts: List[pd.DataFrame] = []
    failures: List[str] = []
    processed: List[str] = []

    for spath in subjects:
        subj = spath.stem  # e.g., "S2"
        print("\n" + "=" * 80)
        print(f"[{subj}] Processing from {spath}")

        out_path = OUT_SUBJ / f"{subj}_features.parquet"
        if out_path.exists() and not overwrite:
            print(f"[{subj}] Skipping (exists). Set FEAT_OVERWRITE=1 to recompute.")
            try:
                df_existing = pd.read_parquet(out_path)
                all_parts.append(df_existing)
                processed.append(subj)
            except Exception as e:
                print(f"[{subj}] Failed to read existing {out_path}: {e}. Recomputing...")
            else:
                continue

        # Run strict per-window extraction
        try:
            df = process_subject_s2(spath)
        except Exception as e:
            print(f"[{subj}] ERROR: {e}")
            failures.append(subj)
            continue

        if not combined_only:
            try:
                df.to_parquet(out_path, index=False, compression=compression)
            except Exception:
                df.to_parquet(out_path, index=False, compression="snappy")
            print(f"[{subj}] Saved -> {out_path} (rows={len(df)}, cols={len(df.columns)})")

        all_parts.append(df)
        processed.append(subj)

    if not all_parts:
        raise RuntimeError("No subjects processed successfully; aborting.")

    # Combined output
    combined = pd.concat(all_parts, ignore_index=True)
    combined_out = OUT_ROOT / combined_name
    try:
        combined.to_parquet(combined_out, index=False, compression=compression)
    except Exception:
        combined.to_parquet(combined_out, index=False, compression="snappy")
    print("\n" + "-" * 80)
    print(f"Saved combined -> {combined_out} (rows={len(combined)}, cols={len(combined.columns)})")

    # Run manifest for traceability
    manifest: Dict[str, Any] = {
        "started_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_dir": str(DATA_DIR),
        "out_root": str(OUT_ROOT),
        "out_subjects_dir": str(OUT_SUBJ),
        "combined_out": str(combined_out),
        "env": {
            "FEAT_DATA_DIR": os.getenv("FEAT_DATA_DIR", ""),
            "FEAT_LIMIT": os.getenv("FEAT_LIMIT", ""),
            "FEAT_SKIP_HRV": os.getenv("FEAT_SKIP_HRV", ""),
            "FEAT_SUPPRESS_WARNINGS": os.getenv("FEAT_SUPPRESS_WARNINGS", ""),
            "FEAT_OVERWRITE": os.getenv("FEAT_OVERWRITE", ""),
            "FEAT_COMBINED_ONLY": os.getenv("FEAT_COMBINED_ONLY", ""),
            "FEAT_SUBJECTS": os.getenv("FEAT_SUBJECTS", ""),
            "FEAT_EXCLUDE": os.getenv("FEAT_EXCLUDE", ""),
            "FEAT_PARQUET_EXT": os.getenv("FEAT_PARQUET_EXT", ""),
            "FEAT_COMBINED_NAME": os.getenv("FEAT_COMBINED_NAME", ""),
            "FEAT_OUT_SUBDIR": os.getenv("FEAT_OUT_SUBDIR", ""),
        },
        "processed_subjects": processed,
        "failed_subjects": failures,
        "combined_rows": int(len(combined)),
        "combined_cols": int(len(combined.columns)),
    }
    (OUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    if failures:
        print(f"Subjects failed: {failures}")
    else:
        print("All subjects processed successfully.")


if __name__ == "__main__":
    main()
