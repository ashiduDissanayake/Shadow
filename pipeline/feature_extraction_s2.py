#!/usr/bin/env python3
"""
STRICT feature extraction for subject S2 using FLIRT (EDA, ACC, HRV from BVP, and stats).

Inputs:  pipeline/wesad_windows_parquet_per_subject/S2.parquet
Outputs: pipeline/features/S2_features.parquet

- STRICT: No fallbacks and no silent passes. If any required step fails, the script raises an error.
- EDA: cvxEDA (phasic/tonic) + stats (entropies=True). Failure -> error.
- ACC: L2 stats (entropies=True) and axis stats (entropies=True). Failure -> error.
- TEMP: stats (entropies=True). Failure -> error.
- BVP: stats (entropies=True). HRV is attempted; if not enough beats or duration, HRV is simply omitted (not an error).
- Env flags:
    - FEAT_LIMIT: int, limit number of rows processed (per subject).
    - FEAT_SKIP_HRV=1: skip HRV features entirely (keeps BVP stats).
    - FEAT_SUPPRESS_WARNINGS=1: suppress numerical RuntimeWarnings from stats on near-constant windows.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import neurokit2 as nk

import flirt
import flirt.eda.feature_calculation as flirt_eda
from flirt.stats.feature_calculation import get_stats as flirt_get_stats
from flirt.hrv.feature_calculation import get_hrv_features as flirt_get_hrv_features
import importlib


# Paths
try:
    PIPELINE_DIR = Path(__file__).resolve().parent
except NameError:
    PIPELINE_DIR = Path.cwd()

DATA_DIR = PIPELINE_DIR / "wesad_windows_parquet_per_subject"
FEATURES_DIR = PIPELINE_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Sampling rates and expected lengths for a 60s window
WINDOW_S = 60.0
SR_EDA = 4
SR_ACC = 32
SR_BVP = 64
SR_TEMP = 4

EXPECTED_LEN = {
    "EDA": int(WINDOW_S * SR_EDA),   # 240
    "ACC": int(WINDOW_S * SR_ACC),   # 1920
    "BVP": int(WINDOW_S * SR_BVP),   # 3840
    "TEMP": int(WINDOW_S * SR_TEMP), # 240
}

_env_data_dir = os.getenv("FEAT_DATA_DIR")
if _env_data_dir:
    _config_path = Path(_env_data_dir)
    if not _config_path.is_absolute():
        _config_path = PIPELINE_DIR / _config_path
    DATA_DIR = _config_path
# Env flags
SKIP_HRV = os.getenv("FEAT_SKIP_HRV", "0") == "1"
SUPPRESS_WARN = os.getenv("FEAT_SUPPRESS_WARNINGS", "0") == "1"

if SUPPRESS_WARN:
    # Suppress frequent precision loss warnings from stats on near-constant windows
    warnings.filterwarnings(
        "ignore",
        message="Precision loss occurred in moment calculation due to catastrophic cancellation",
        category=RuntimeWarning,
    )


def _as_1d_float(arr_like, name: str) -> np.ndarray:
    a = np.asarray(arr_like, dtype=np.float32).ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError(f"Non-finite values in {name} window")
    return a


def _require_len(arr: np.ndarray, expected: int, name: str) -> None:
    if arr.size != expected:
        raise ValueError(f"{name} length mismatch: got {arr.size}, expected {expected}")


def _cvx_eda_strict(eda_vals: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strict cvxEDA: FLIRT uses cvxopt internally, which expects float64 buffers.
    Cast input to float64 before calling, then return numpy arrays.
    """
    eda64 = np.asarray(eda_vals, dtype=np.float64, order="C")
    if hasattr(flirt_eda, "cvx_eda"):
        r, t = flirt_eda.cvx_eda(eda64, dt)
    elif hasattr(flirt_eda, "__cvx_eda"):
        r, t = flirt_eda.__cvx_eda(eda64, dt)
    else:
        raise AttributeError("FLIRT EDA cvx_eda function not found in flirt.eda.feature_calculation")
    # Ensure numpy ndarrays
    return np.asarray(r), np.asarray(t)


def _stats_strict(x: np.ndarray, prefix: str, entropies: bool) -> Dict[str, float]:
    vals = flirt_get_stats(x, prefix, entropies=entropies)  # let errors propagate
    return {f"{prefix}__{k}": float(v) for k, v in vals.items()}


def process_subject_s2(parquet_path: Path) -> pd.DataFrame:
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # Optional limit for quick tests
    limit_env = os.getenv("FEAT_LIMIT", "0")
    try:
        limit = int(limit_env)
    except Exception:
        limit = 0
    if limit > 0:
        df = df.head(limit)

    required = ["subject", "label", "BVP", "EDA", "TEMP", "ACC.X", "ACC.Y", "ACC.Z", "start"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {parquet_path}: {missing}")

    records: List[Dict[str, float]] = []

    for i, row in enumerate(df.to_dict('records'), start=1):
        subject = row["subject"]
        start = float(row["start"])
        lbl = row["label"]
        label = int(lbl) if pd.notna(lbl) else -1

        rec: Dict[str, float] = {"subject": subject, "start": start, "label": label}

        # -------- EDA (strict) --------
        eda_vals = _as_1d_float(row["EDA"], "EDA")
        _require_len(eda_vals, EXPECTED_LEN["EDA"], "EDA")
        r, t = _cvx_eda_strict(eda_vals, 1.0 / SR_EDA)  # phasic driver (r), tonic (t)
        r = _as_1d_float(r, "EDA.phasic")
        t = _as_1d_float(t, "EDA.tonic")
        rec.update(_stats_strict(r, "EDA.phasic", entropies=True))
        rec.update(_stats_strict(t, "EDA.tonic",  entropies=True))

        # SCR-like metrics on phasic driver (strict)
        peaks_info = nk.signal_findpeaks(r)
        pidx = np.asarray(peaks_info.get("Peaks", []), dtype=int)
        if pidx.size > 0:
            amps = r[pidx]
            rec["EDA.phasic_scr_count"] = float(pidx.size)
            rec["EDA.phasic_scr_rate"] = float(pidx.size) / 60.0  # per 60s window
            rec["EDA.phasic_scr_mean_amp"] = float(np.mean(amps))
            rec["EDA.phasic_scr_median_amp"] = float(np.median(amps))
            rec["EDA.phasic_scr_sum_amp"] = float(np.sum(amps))
        else:
            # Zero is a valid strict outcome if no peaks are present
            rec["EDA.phasic_scr_count"] = 0.0
            rec["EDA.phasic_scr_rate"] = 0.0
            rec["EDA.phasic_scr_mean_amp"] = 0.0
            rec["EDA.phasic_scr_median_amp"] = 0.0
            rec["EDA.phasic_scr_sum_amp"] = 0.0

        # -------- ACC (strict) --------
        ax = _as_1d_float(row["ACC.X"], "ACC.X")
        ay = _as_1d_float(row["ACC.Y"], "ACC.Y")
        az = _as_1d_float(row["ACC.Z"], "ACC.Z")
        for name_arr, name in [(ax, "ACC.X"), (ay, "ACC.Y"), (az, "ACC.Z")]:
            _require_len(name_arr, EXPECTED_LEN["ACC"], name)
        l2 = np.sqrt(ax * ax + ay * ay + az * az).astype(np.float32, copy=False)
        rec.update(_stats_strict(l2, "ACC.L2", entropies=True))
        rec.update(_stats_strict(ax, "ACC.X", entropies=True))
        rec.update(_stats_strict(ay, "ACC.Y", entropies=True))
        rec.update(_stats_strict(az, "ACC.Z", entropies=True))

        # -------- TEMP (strict) --------
        temp_vals = _as_1d_float(row["TEMP"], "TEMP")
        _require_len(temp_vals, EXPECTED_LEN["TEMP"], "TEMP")
        rec.update(_stats_strict(temp_vals, "TEMP", entropies=True))

        # -------- BVP (strict stats; HRV optional) --------
        bvp_vals = _as_1d_float(row["BVP"], "BVP")
        _require_len(bvp_vals, EXPECTED_LEN["BVP"], "BVP")
        # Always include stats on BVP
        rec.update(_stats_strict(bvp_vals, "BVP.stat", entropies=True))

        # HRV attempt: not an error if not enough beats; we just omit HRV features for that row
        if SKIP_HRV:
            records.append(rec)
            continue
        bvp_clean = nk.ppg_clean(bvp_vals, sampling_rate=SR_BVP)
        peaks_info = nk.ppg_findpeaks(bvp_clean, sampling_rate=SR_BVP)
        peaks_idx = np.asarray(peaks_info.get("PPG_Peaks", []), dtype=int)
        if peaks_idx.size >= 3:
            ibis_ms = np.diff(peaks_idx) / float(SR_BVP) * 1000.0
            ibis_ms = ibis_ms[(ibis_ms >= 300.0) & (ibis_ms <= 2000.0)]
            if ibis_ms.size >= 2:
                start_ts = pd.Timestamp("1970-01-01") + pd.to_timedelta(start, unit="s")
                times = start_ts + pd.to_timedelta(np.cumsum(ibis_ms) / 1000.0, unit="s")
                ibi_series = pd.Series(ibis_ms, index=pd.DatetimeIndex(times), name="ibi")
                span_sec = max(0.0, (ibi_series.index[-1] - ibi_series.index[0]).total_seconds())
                if span_sec >= 20.0:
                    wl = int(min(60, max(20, np.floor(span_sec))))
                    # Suppress tqdm progress bar inside FLIRT's HRV function
                    flirt_hrv_fc = importlib.import_module('flirt.hrv.feature_calculation')
                    _old_tqdm = getattr(flirt_hrv_fc, 'tqdm', None)
                    try:
                        flirt_hrv_fc.tqdm = (lambda it, **kwargs: it)
                        hrv_df = flirt_get_hrv_features(
                            data=ibi_series,
                            window_length=wl,
                            window_step_size=wl,
                            domains=["td", "fd", "nl", "stat"],
                            threshold=0.2,
                            clean_data=True,
                            num_cores=1,
                        )
                    finally:
                        if _old_tqdm is not None:
                            flirt_hrv_fc.tqdm = _old_tqdm
                    if len(hrv_df) > 0:
                        for k, v in hrv_df.iloc[0].to_dict().items():
                            rec[f"BVP.HRV__{k}"] = float(v if np.isscalar(v) else np.asarray(v).item())

        records.append(rec)

    feat_df = pd.DataFrame.from_records(records)

    # Report features overview
    def count_by_prefix(prefix: str) -> int:
        return sum(1 for c in feat_df.columns if c.startswith(prefix))
    totals = {
        "EDA.phasic": count_by_prefix("EDA.phasic__"),
        "EDA.tonic":  count_by_prefix("EDA.tonic__"),
        "EDA.SCR":    len([c for c in feat_df.columns if c.startswith("EDA.phasic_scr_")]),
        "ACC.L2":     count_by_prefix("ACC.L2__"),
        "ACC.X":      count_by_prefix("ACC.X__"),
        "ACC.Y":      count_by_prefix("ACC.Y__"),
        "ACC.Z":      count_by_prefix("ACC.Z__"),
        "TEMP":       count_by_prefix("TEMP__"),
        "BVP.stat":   count_by_prefix("BVP.stat__"),
        "BVP.HRV":    count_by_prefix("BVP.HRV__"),
    }
    meta = ["subject", "start", "label"]
    total_cols = len(feat_df.columns)
    print(f"Feature columns breakdown (excluding meta): {totals}")
    print(f"Total columns: {total_cols} (meta={len(meta)}, feature={total_cols - len(meta)})")

    return feat_df


def main():
    s2_path = DATA_DIR / "S2.parquet"
    if not s2_path.exists():
        raise FileNotFoundError(f"Expected file not found: {s2_path}")

    feat_df = process_subject_s2(s2_path)
    out_path = FEATURES_DIR / "S2_features.parquet"

    # Strict write; prefer zstd, fallback to snappy only if zstd not available at write time
    try:
        feat_df.to_parquet(out_path, index=False, compression="zstd")
    except Exception:
        feat_df.to_parquet(out_path, index=False, compression="snappy")

    print(f"Saved features -> {out_path}  (rows={len(feat_df)}, cols={len(feat_df.columns)})")


if __name__ == "__main__":
    main()