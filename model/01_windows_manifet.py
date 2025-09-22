# 01_windows_manifest.py
# Build a global window manifest with tri-state labels (stress, nonstress, junk),
# purity, and time boundaries per subject. Windows are computed from the shared
# minimum duration across modalities & labels to guarantee alignment.

import os
from glob import glob
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
processed_dir = "data/working/wesad_processed"
out_manifest = os.path.join(processed_dir, "windows_manifest_W30_S10_tau75.parquet")

# Sampling rates (Hz) - verify for your data
ACC_FS  = 32
BVP_FS  = 64
EDA_FS  = 4
TEMP_FS = 4
LABEL_FS = 700  # WESAD label stream is typically 700 Hz

# Windowing and purity
W_SEC  = 30
S_SEC  = 10
TAU    = 0.75

# Label mapping fallback (only used if labels aren't already 0/1/2)
LABEL_MAP = {0:0, 5:0, 6:0, 7:0, 1:1, 3:1, 4:1, 2:2}
# ----------------------------------------


def find_subjects(base_dir: str) -> list:
    return sorted({os.path.basename(p).split("_")[0] for p in glob(os.path.join(base_dir, "S*_labels.parquet"))})

def load_series(path: str) -> pd.Series:
    df = pd.read_parquet(path)
    if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
        return df.iloc[:, 0]
    raise ValueError(f"Expected a single-column labels parquet: {path}")

def min_duration_seconds(subject: str) -> float:
    paths = {
        "ACC":  os.path.join(processed_dir, f"{subject}_ACC.parquet"),
        "BVP":  os.path.join(processed_dir, f"{subject}_BVP.parquet"),
        "EDA":  os.path.join(processed_dir, f"{subject}_EDA.parquet"),
        "TEMP": os.path.join(processed_dir, f"{subject}_TEMP.parquet"),
        "LBL":  os.path.join(processed_dir, f"{subject}_labels.parquet"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {k} for {subject}: {p}")

    acc_len  = len(pd.read_parquet(paths["ACC"]))
    bvp_len  = len(pd.read_parquet(paths["BVP"]))
    eda_len  = len(pd.read_parquet(paths["EDA"]))
    temp_len = len(pd.read_parquet(paths["TEMP"]))
    lbl_len  = len(load_series(paths["LBL"]))

    durations = [
        acc_len  / float(ACC_FS),
        bvp_len  / float(BVP_FS),
        eda_len  / float(EDA_FS),
        temp_len / float(TEMP_FS),
        lbl_len  / float(LABEL_FS),
    ]
    return max(0.0, min(durations))

def compute_windows(min_dur_sec: float, W_sec: float, S_sec: float) -> list[tuple[float,float]]:
    if min_dur_sec < W_sec:
        return []
    n = int(np.floor((min_dur_sec - W_sec) / S_sec)) + 1
    starts = [i * S_sec for i in range(n)]
    return [(t, t + W_sec) for t in starts]

def label_props_in_window(labels: pd.Series, start_s: float, end_s: float, fs: int) -> tuple[float,float,float]:
    i0 = int(np.floor(start_s * fs))
    i1 = int(np.floor(end_s   * fs))
    i1 = min(i1, len(labels))
    if i0 >= i1:
        return (0.0, 0.0, 1.0)
    seg = labels.iloc[i0:i1]
    total = len(seg)
    if total == 0:
        return (0.0, 0.0, 1.0)
    p2 = float((seg == 2).sum())/total
    p1 = float((seg == 1).sum())/total
    p0 = float((seg == 0).sum())/total
    return (p2, p1, p0)

def tri_state_from_props(p_stress: float, p_non: float, tau: float) -> tuple[int,str,float]:
    if p_stress >= tau:
        return (1, "stress", max(p_stress, p_non))
    if p_non   >= tau:
        return (0, "nonstress", max(p_stress, p_non))
    return (-1, "junk", max(p_stress, p_non))

def main():
    subs = find_subjects(processed_dir)
    print("Subjects:", subs)

    rows = []
    for s in subs:
        lbl_path = os.path.join(processed_dir, f"{s}_labels.parquet")
        labels = load_series(lbl_path)
        if not set(pd.unique(labels)).issubset({0,1,2}):
            labels = labels.map(LABEL_MAP).astype(int)

        min_dur = min_duration_seconds(s)
        windows = compute_windows(min_dur, W_SEC, S_SEC)
        if not windows:
            print(f"[WARN] {s}: duration {min_dur:.2f}s < W={W_SEC}s → no windows")
            continue

        for idx, (t0, t1) in enumerate(windows):
            p2, p1, p0 = label_props_in_window(labels, t0, t1, LABEL_FS)
            bin_lbl, cat, purity = tri_state_from_props(p2, p1, TAU)
            rows.append({
                "subject": s,
                "window_idx": idx,
                "W_sec": float(W_SEC),
                "S_sec": float(S_SEC),
                "start_time_s": float(t0),
                "end_time_s": float(t1),
                "prop_stress": float(p2),
                "prop_nonstress": float(p1),
                "prop_junk": float(p0),
                "purity": float(purity),
                "bin_label": int(bin_lbl),  # 1 stress, 0 nonstress, -1 junk
                "category": cat
            })

    manifest = pd.DataFrame(rows)
    print("Total windows:", len(manifest))
    if len(manifest):
        print(manifest["category"].value_counts())
    manifest.to_parquet(out_manifest, index=False)
    print("Saved manifest to:", out_manifest)

if __name__ == "__main__":
    main()