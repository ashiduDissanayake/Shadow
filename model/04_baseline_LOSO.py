# 04_baseline_loso.py
# Baseline LOSO with train-side threshold tuning (no leakage).
# Reads folds from 03_build_loso_and_holdout.py outputs.

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, KFold

# ---------------- CONFIG ----------------
processed_dir = "data/output"
features_path = os.path.join(processed_dir, "final_features_ACCextractor_allmods_W30_S10_tau75.parquet")
folds_dir = os.path.join(processed_dir, "folds_W30_S10_tau75")
out_dir = os.path.join(processed_dir, "04_baseline_loso")
os.makedirs(out_dir, exist_ok=True)

SEED = 42
TUNE_THRESHOLD = True
THRESH_GRID = np.linspace(0.1, 0.9, 81)  # try lowering threshold → more recall
POS_CLASS_WEIGHT = 4.53  # e.g., 1.5 to favor recall; None → use "balanced"
# ----------------------------------------

META = {"subject", "label", "session", "category", "bin_label", "W_sec", "S_sec", "start_idx", "end_idx", "window_id"}

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    pref = [c for c in df.columns if c.startswith(("acc_", "bvp_", "eda_", "temp_"))]
    if pref:
        return pref
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num if c not in META and c != "label"]

def impute_median(train: pd.DataFrame, test: pd.DataFrame):
    med = train.median(numeric_only=True)
    return train.fillna(med), test.fillna(med)

def inner_oof_threshold(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> float:
    # Group-aware OOF probabilities on the train portion; pick threshold maximizing F1
    uniq = np.unique(groups)
    if len(uniq) >= 3:
        splits = GroupKFold(n_splits=min(5, len(uniq))).split(X, y, groups)
    else:
        splits = KFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y)
    oof = np.zeros_like(y, dtype=float)
    seen = np.zeros_like(y, dtype=bool)
    for tr, va in splits:
        clf = ExtraTreesClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight=("balanced" if POS_CLASS_WEIGHT is None else {0:1.0, 1:float(POS_CLASS_WEIGHT)}),
            n_jobs=-1, random_state=SEED
        )
        clf.fit(X.iloc[tr], y[tr])
        oof[va] = clf.predict_proba(X.iloc[va])[:, 1]
        seen[va] = True
    if not seen.all():
        oof[~seen] = np.mean(y)
    best_f1, best_t = -1.0, 0.5
    for t in THRESH_GRID:
        pred = (oof >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t

def main():
    if not os.path.exists(features_path):
        raise FileNotFoundError(features_path)
    folds_path = os.path.join(folds_dir, "loso_folds.json")
    if not os.path.exists(folds_path):
        raise FileNotFoundError(folds_path)

    df = pd.read_parquet(features_path)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    feats = get_feature_cols(df)

    with open(folds_path, "r") as f:
        folds = json.load(f)["folds"]

    rows = []
    for fold in folds:
        test_subject = fold["test_subject"]
        train_subjects = set(fold["train_subjects"])

        train = df[df["subject"].isin(train_subjects)].copy()
        test = df[df["subject"] == test_subject].copy()

        X_tr = train[feats].replace([np.inf, -np.inf], np.nan)
        X_te = test[feats].replace([np.inf, -np.inf], np.nan)

        keep = X_tr.columns[X_tr.notna().any(axis=0)].tolist()
        X_tr = X_tr[keep]
        X_te = X_te[keep]
        X_tr, X_te = impute_median(X_tr, X_te)

        y_tr = train["label"].astype(int).to_numpy()
        y_te = test["label"].astype(int).to_numpy()
        groups_tr = train["subject"].to_numpy()

        # Train model on full train
        clf = ExtraTreesClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight=("balanced" if POS_CLASS_WEIGHT is None else {0:1.0, 1:float(POS_CLASS_WEIGHT)}),
            n_jobs=-1, random_state=SEED
        )
        clf.fit(X_tr, y_tr)

        # Threshold: tune on train (OOF) if enabled; else 0.5
        thr = 0.5
        if TUNE_THRESHOLD:
            thr = inner_oof_threshold(X_tr, y_tr, groups_tr)

        p = clf.predict_proba(X_te)[:, 1]
        y_pred = (p >= thr).astype(int)

        f1 = f1_score(y_te, y_pred, zero_division=0)
        bal = balanced_accuracy_score(y_te, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()

        rows.append({
            "test_subject": test_subject,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "threshold": float(thr),
            "f1": float(f1),
            "balanced_accuracy": float(bal),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "fold_metrics.csv"), index=False)
    res.describe().to_csv(os.path.join(out_dir, "aggregate_summary.csv"))
    print(f"Saved baseline LOSO (tuned) results to {out_dir}")

if __name__ == "__main__":
    main()