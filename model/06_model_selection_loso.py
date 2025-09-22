# 06_model_selection_loso.py
# LOSO model comparison with train-side threshold tuning, using selected features from 05.
# Models: ExtraTrees, RandomForest, LogisticRegression, LinearSVC, MLP (simple 64x64 ReLU), XGBoost (if installed)
#
# Outputs under data/output/06_model_selection_loso:
#   - all_models_all_folds.csv
#   - model_ranking.csv
#   - <ModelName>_fold_metrics.csv
#   - fold_<subject>/fold_results.csv

import os
import json
import numpy as np
import pandas as pd
import warnings

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, KFold
from sklearn.exceptions import ConvergenceWarning

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False

# ---------------- CONFIG ----------------
processed_dir = "data/output"
features_path = os.path.join(processed_dir, "final_features_ACCextractor_allmods_W30_S10_tau75.parquet")
folds_dir = os.path.join(processed_dir, "folds_W30_S10_tau75")

selection_dir = os.path.join(processed_dir, "05_selection_loso")
selection_summary_path = os.path.join(selection_dir, "selection_summary.json")
feature_list_csv = os.path.join(selection_dir, "final_feature_order.csv")

out_dir = os.path.join(processed_dir, "06_model_selection_loso")
os.makedirs(out_dir, exist_ok=True)

SEED = 42
# Align with step 05 configuration
POS_CLASS_WEIGHT = 4.53  # for tree models; XGB uses scale_pos_weight
THRESH_GRID = np.linspace(0.1, 0.9, 81)
VAR_THRESH = 1e-6  # drop near-constant (train-based), to mirror step 05

# Simple MLP hyperparams (as per your best runs)
MLP_HIDDEN = (64, 64)
MLP_ACTIVATION = "relu"
MLP_ALPHA = 0.0003
MLP_MAX_ITER = 1000
MLP_EARLY_STOP = True
MLP_N_NO_CHANGE = 20

# Reduce console noise
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# ----------------------------------------

META = {"subject", "label", "session", "category", "bin_label", "W_sec", "S_sec", "start_idx", "end_idx", "window_id"}

def load_feature_list(path_csv: str, selection_summary_json: str) -> list[str]:
    if not os.path.exists(selection_summary_json):
        raise FileNotFoundError(selection_summary_json)
    if not os.path.exists(path_csv):
        raise FileNotFoundError(path_csv)
    with open(selection_summary_json, "r") as f:
        ss = json.load(f)
    global_N = int(ss.get("global_N", 32))
    df = pd.read_csv(path_csv)
    feats = df["feature"].tolist()
    feats = feats[:global_N] if global_N > 0 else feats
    return feats

def impute_median(train: pd.DataFrame, test: pd.DataFrame):
    med = train.median(numeric_only=True)
    return train.fillna(med), test.fillna(med)

def drop_near_constant(train: pd.DataFrame, test: pd.DataFrame, thr: float):
    var = train.var(axis=0, ddof=0)
    keep = var.index[var >= thr].tolist()
    return train[keep], test[keep]

def get_models(seed: int):
    tree_weight = {0: 1.0, 1: float(POS_CLASS_WEIGHT)}
    models = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,           # align with step 05
            max_depth=None,            # align with step 05
            min_samples_leaf=2,
            class_weight=tree_weight,
            n_jobs=-1,
            random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=tree_weight,
            n_jobs=-1,
            random_state=seed
        ),
        "LogisticRegression": Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=seed)),
        ]),
        "LinearSVC": Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=seed)),
        ]),
        "MLP": Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=MLP_HIDDEN,
                activation=MLP_ACTIVATION,
                alpha=MLP_ALPHA,
                learning_rate="adaptive",
                max_iter=MLP_MAX_ITER,
                early_stopping=MLP_EARLY_STOP,
                n_iter_no_change=MLP_N_NO_CHANGE,
                random_state=seed
            )),
        ]),
    }
    if _XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            scale_pos_weight=float(POS_CLASS_WEIGHT),
        )
    return models

def predict_scores(model, X: pd.DataFrame) -> np.ndarray:
    # Returns a score in [0,1] per sample
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if isinstance(p, list):
            p = p[0]
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1]
        if p.ndim == 1:
            return p
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = np.asarray(s, dtype=float)
        s_min, s_max = s.min(), s.max()
        denom = (s_max - s_min) + 1e-12
        return (s - s_min) / denom
    # Fallback: constant prior
    return np.full(shape=(len(X),), fill_value=0.5, dtype=float)

def inner_oof_threshold(make_model_fn, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> float:
    uniq = np.unique(groups)
    if len(uniq) >= 3:
        splits = GroupKFold(n_splits=min(5, len(uniq))).split(X, y, groups)
    else:
        splits = KFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y)
    oof = np.zeros_like(y, dtype=float)
    seen = np.zeros_like(y, dtype=bool)
    for tr, va in splits:
        m = make_model_fn()
        m.fit(X.iloc[tr], y[tr])
        oof[va] = predict_scores(m, X.iloc[va])
        seen[va] = True
    if not seen.all():
        oof[~seen] = np.mean(y)
    best_f1, best_t = -1.0, 0.5
    for t in THRESH_GRID:
        pred = (oof >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t

def main():
    # Check inputs
    if not os.path.exists(features_path):
        raise FileNotFoundError(features_path)
    folds_path = os.path.join(folds_dir, "loso_folds.json")
    if not os.path.exists(folds_path):
        raise FileNotFoundError(folds_path)
    feature_cols = load_feature_list(feature_list_csv, selection_summary_path)

    # Load data
    df = pd.read_parquet(features_path)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    # Keep only desired features that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Load folds
    with open(folds_path, "r") as f:
        folds = json.load(f)["folds"]

    # Prepare models
    base_models = get_models(SEED)

    all_rows = []

    for fold in folds:
        test_subject = fold["test_subject"]
        train_subjects = set(fold["train_subjects"])
        fold_dir = os.path.join(out_dir, f"fold_{test_subject}")
        os.makedirs(fold_dir, exist_ok=True)

        # Split
        train = df[df["subject"].isin(train_subjects)].copy()
        test = df[df["subject"] == test_subject].copy()

        X_tr = train[feature_cols].replace([np.inf, -np.inf], np.nan)
        X_te = test[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Drop all-NaN (train-based)
        keep_nonan = X_tr.columns[X_tr.notna().any(axis=0)].tolist()
        X_tr = X_tr[keep_nonan]
        X_te = X_te[keep_nonan]

        # Drop near-constant (train-based), aligned with step 05
        X_tr, X_te = drop_near_constant(X_tr, X_te, VAR_THRESH)

        # Impute
        X_tr, X_te = impute_median(X_tr, X_te)
        y_tr = train["label"].astype(int).to_numpy()
        y_te = test["label"].astype(int).to_numpy()
        groups_tr = train["subject"].to_numpy()

        fold_rows = []

        for name in base_models.keys():
            # Factory for a fresh model instance
            def make_model():
                return get_models(SEED)[name]

            # Tune threshold on train via OOF
            thr = inner_oof_threshold(make_model, X_tr, y_tr, groups_tr)

            # Fit full model
            m = make_model()
            m.fit(X_tr, y_tr)
            scores = predict_scores(m, X_te)
            y_pred = (scores >= thr).astype(int)

            f1 = f1_score(y_te, y_pred, zero_division=0)
            bal = balanced_accuracy_score(y_te, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()

            rec = {
                "model": name,
                "test_subject": test_subject,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_features": int(X_tr.shape[1]),
                "threshold": float(thr),
                "f1": float(f1),
                "balanced_accuracy": float(bal),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            }
            fold_rows.append(rec)
            all_rows.append(rec)

        # Save per-fold all-model results
        pd.DataFrame(fold_rows).to_csv(os.path.join(fold_dir, "fold_results.csv"), index=False)

    # Save all folds
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(os.path.join(out_dir, "all_models_all_folds.csv"), index=False)

    # Per-model files and ranking
    rank_rows = []
    for name, g in all_df.groupby("model"):
        g.to_csv(os.path.join(out_dir, f"{name}_fold_metrics.csv"), index=False)
        rank_rows.append({
            "model": name,
            "mean_f1": float(g["f1"].mean()),
            "std_f1": float(g["f1"].std()),
            "mean_bal_acc": float(g["balanced_accuracy"].mean()),
        })
    rank_df = pd.DataFrame(rank_rows).sort_values(["mean_f1", "mean_bal_acc"], ascending=False)
    rank_df.to_csv(os.path.join(out_dir, "model_ranking.csv"), index=False)

if __name__ == "__main__":
    main()