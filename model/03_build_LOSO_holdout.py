# 03_build_loso_and_holdout.py
# Build LOSO folds (train subjects vs one test subject per fold), optionally
# reserve holdout subjects (never used in LOSO). Check for degenerate folds.

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
processed_dir = "data/output"
features_path = os.path.join(processed_dir, "final_features_ACCextractor_allmods_W30_S10_tau75.parquet")
folds_dir = os.path.join(processed_dir, "folds_W30_S10_tau75")
os.makedirs(folds_dir, exist_ok=True)

holdout_subjects = ["S16", "S4"]
HOLDOUT_FRAC = 0.10
SEED = 42
# ----------------------------------------


def main():
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing features parquet: {features_path}")
    df = pd.read_parquet(features_path)

    subjects_all = sorted(df["subject"].unique().tolist())

    if not holdout_subjects:
        n_hold = max(1, int(round(HOLDOUT_FRAC * len(subjects_all))))
        _, hold = train_test_split(subjects_all, test_size=n_hold, random_state=SEED)
        holdout = sorted(hold)
    else:
        holdout = sorted(holdout_subjects)

    trainval_subjects = [s for s in subjects_all if s not in holdout]

    with open(os.path.join(folds_dir, "holdout_subjects.txt"), "w") as f:
        for s in holdout:
            f.write(s + "\n")
    print("Holdout subjects:", holdout)

    folds = []
    for test_subj in trainval_subjects:
        train_subjs = [s for s in trainval_subjects if s != test_subj]
        folds.append({
            "fold_id": len(folds),
            "test_subject": test_subj,
            "train_subjects": train_subjs
        })

    degenerate = []
    for fold in folds:
        tr = df[df["subject"].isin(fold["train_subjects"])]
        te = df[df["subject"] == fold["test_subject"]]
        tr_labels = set(tr["label"].unique().tolist())
        if not ({0,1}.issubset(tr_labels)):
            degenerate.append(fold["test_subject"])
        if len(te) == 0:
            degenerate.append(fold["test_subject"])

    payload = {
        "validation_strategy": "LeaveOneSubjectOut",
        "n_folds": len(folds),
        "total_subjects": len(trainval_subjects),
        "folds": folds,
        "holdout_subjects": holdout,
        "degenerate_folds": sorted(list(set(degenerate)))
    }
    with open(os.path.join(folds_dir, "loso_folds.json"), "w") as f:
        json.dump(payload, f, indent=2)

    summary = df.groupby("subject").agg(
        n_windows=("label", "count"),
        n_stress=("label", lambda x: int((x==1).sum())),
        n_nonstress=("label", lambda x: int((x==0).sum()))
    ).reset_index()
    summary.to_csv(os.path.join(folds_dir, "per_subject_counts.csv"), index=False)

    print(f"Saved LOSO folds and summaries in {folds_dir}")
    if degenerate:
        print("[WARN] Degenerate folds (train missing a class or empty test):", sorted(list(set(degenerate))))

if __name__ == "__main__":
    main()