# 02_extract_features_flirt_acc_all.py
# Apply FLIRT's ACC feature extractor to ALL modalities (ACC, BVP, EDA, TEMP).
# Joins with the manifest (W=30, S=10, tau=0.75), drops junk, and saves final features.

import os
import numpy as np
import pandas as pd
import flirt  # FLIRT >= 0.5.0

# ---------------- CONFIG ----------------
processed_dir = "data/output"
input_dir = "data/working/wesad_processed"
manifest_path = os.path.join(input_dir, "windows_manifest_W30_S10_tau75.parquet")
out_features  = os.path.join(processed_dir, "final_features_ACCextractor_allmods_W30_S10_tau75.parquet")

# Sampling rates (Hz) — verify for your data
ACC_FS  = 32
BVP_FS  = 64
EDA_FS  = 4
TEMP_FS = 4

# Windowing
W_SEC = 30
S_SEC = 10

# Parallelism (None = FLIRT default/all cores)
NUM_CORES = None  # e.g., set to 4 if you want to limit cores
# ----------------------------------------


def ns_freq_str(fs: float) -> str:
    # Use 'ns' (nanoseconds). This fixes the pandas deprecation for 'N'.
    ns = int(round(1e9 / fs))
    return f"{ns}ns"

def build_time_index(n: int, fs: float) -> pd.DatetimeIndex:
    return pd.date_range(start=pd.Timestamp(0), periods=n, freq=ns_freq_str(fs))

def min_duration_seconds(subject: str) -> float:
    paths = {
        "ACC":  os.path.join(input_dir, f"{subject}_ACC.parquet"),
        "BVP":  os.path.join(input_dir, f"{subject}_BVP.parquet"),
        "EDA":  os.path.join(input_dir, f"{subject}_EDA.parquet"),
        "TEMP": os.path.join(input_dir, f"{subject}_TEMP.parquet"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {k} for {subject}: {p}")
    acc_len  = len(pd.read_parquet(paths["ACC"]))
    bvp_len  = len(pd.read_parquet(paths["BVP"]))
    eda_len  = len(pd.read_parquet(paths["EDA"]))
    temp_len = len(pd.read_parquet(paths["TEMP"]))
    durations = [
        acc_len/ACC_FS,
        bvp_len/BVP_FS,
        eda_len/EDA_FS,
        temp_len/TEMP_FS,
    ]
    return max(0.0, min(durations))

def prep_ts(df: pd.DataFrame, cols: list[str], fs: float) -> pd.DataFrame:
    # Attach a correct DateTimeIndex per modality and drop NaNs
    idx = build_time_index(len(df), fs)
    return df.set_index(idx)[cols].dropna()

def extract_with_acc(df: pd.DataFrame, cols: list[str], fs: float,
                     W_sec: float, S_sec: float, prefix: str,
                     num_cores: int | None = None) -> pd.DataFrame:
    """
    Apply FLIRT's ACC feature extractor to the given columns of df.
    Prefix the resulting columns to tag the modality.
    """
    X = prep_ts(df, cols, fs)
    kwargs = dict(window_length=W_sec, window_step_size=S_sec, data_frequency=fs)
    if num_cores is not None:
        kwargs["num_cores"] = num_cores
    F = flirt.get_acc_features(X, **kwargs)
    return F.add_prefix(prefix).reset_index(drop=True)

def main():
    manifest = pd.read_parquet(manifest_path)
    manifest = manifest[(manifest["W_sec"] == float(W_SEC)) & (manifest["S_sec"] == float(S_SEC))].reset_index(drop=True)
    subjects = sorted(manifest["subject"].unique().tolist())
    print("Subjects:", subjects)

    kept_all = []

    for s in subjects:
        print("Extracting features (ACC-extractor for all) for", s)
        acc = pd.read_parquet(os.path.join(input_dir, f"{s}_ACC.parquet"))
        bvp = pd.read_parquet(os.path.join(input_dir, f"{s}_BVP.parquet"))
        eda = pd.read_parquet(os.path.join(input_dir, f"{s}_EDA.parquet"))
        tmp = pd.read_parquet(os.path.join(input_dir, f"{s}_TEMP.parquet"))

        # Trim to shared min duration across modalities so window counts align
        min_dur = min_duration_seconds(s)
        acc = acc.iloc[:int(np.floor(min_dur * ACC_FS))].reset_index(drop=True)
        bvp = bvp.iloc[:int(np.floor(min_dur * BVP_FS))].reset_index(drop=True)
        eda = eda.iloc[:int(np.floor(min_dur * EDA_FS))].reset_index(drop=True)
        tmp = tmp.iloc[:int(np.floor(min_dur * TEMP_FS))].reset_index(drop=True)

        # Apply ACC extractor to each modality (prefix by modality)
        accF  = extract_with_acc(acc, ['x', 'y', 'z'], ACC_FS,  W_SEC, S_SEC, prefix="acc_",  num_cores=NUM_CORES)
        bvpF  = extract_with_acc(bvp, ['BVP'],         BVP_FS,  W_SEC, S_SEC, prefix="bvp_", num_cores=NUM_CORES)
        edaF  = extract_with_acc(eda, ['EDA'],         EDA_FS,  W_SEC, S_SEC, prefix="eda_", num_cores=NUM_CORES)
        tempF = extract_with_acc(tmp, ['TEMP'],        TEMP_FS, W_SEC, S_SEC, prefix="temp_",num_cores=NUM_CORES)

        # Truncate to min across modalities for alignment
        L = min(len(accF), len(bvpF), len(edaF), len(tempF))
        if len({len(accF), len(bvpF), len(edaF), len(tempF)}) != 1:
            print(f"[WARN] {s}: modality window mismatch acc/bvp/eda/temp = {len(accF)}/{len(bvpF)}/{len(edaF)}/{len(tempF)} → trunc {L}")
        feat = pd.concat([
            bvpF.iloc[:L].reset_index(drop=True),
            accF.iloc[:L].reset_index(drop=True),
            edaF.iloc[:L].reset_index(drop=True),
            tempF.iloc[:L].reset_index(drop=True)
        ], axis=1)

        # Align with subject manifest rows (by row order), then drop junk
        mani_sub = manifest[manifest["subject"] == s].reset_index(drop=True)
        if len(mani_sub) != len(feat):
            LL = min(len(mani_sub), len(feat))
            print(f"[WARN] {s}: manifest/features windows = {len(mani_sub)}/{len(feat)} → trunc {LL}")
            mani_sub = mani_sub.iloc[:LL].reset_index(drop=True)
            feat = feat.iloc[:LL].reset_index(drop=True)

        joined = pd.concat([mani_sub, feat], axis=1)
        kept = joined[joined["category"] != "junk"].reset_index(drop=True)
        kept_all.append(kept)

    if kept_all:
        final_df = pd.concat(kept_all, ignore_index=True)
        final_df = final_df.rename(columns={"bin_label": "label"})
        final_df = final_df[final_df["label"].isin([0,1])].reset_index(drop=True)

        # Report feature count (excluding manifest/meta)
        feature_cols = [c for c in final_df.columns if any(c.startswith(p) for p in ["acc_","bvp_","eda_","temp_"])]
        print("Total feature columns:", len(feature_cols))

        final_df.to_parquet(out_features, index=False)
        print("Saved final features to:", out_features)
        print("Final shape:", final_df.shape)
        print("Subjects:", sorted(final_df["subject"].unique().tolist()))
    else:
        print("No features produced. Check earlier steps and data availability.")

if __name__ == "__main__":
    main()