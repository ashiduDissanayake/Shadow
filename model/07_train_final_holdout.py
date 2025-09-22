# 07_train_final_holdout.py
# Train the final model on non-holdout subjects and evaluate on holdout subjects (e.g., S9, S10).
#
# Inputs (matching step 06 paths/config):
#   - data/output/final_features_ACCextractor_allmods_W30_S10_tau75.parquet
#   - data/output/folds_W30_S10_tau75/holdout_subjects.txt
#   - data/output/05_selection_loso/selection_summary.json
#   - data/output/05_selection_loso/final_feature_order.csv
#
# Outputs under data/output/07_holdout_evaluation/:
#   - holdout_metrics.csv                    (per-holdout subject metrics)
#   - aggregate_summary.csv                  (aggregate metrics across all holdouts)
#   - class_distribution_train.csv           (per-subject class counts - train)
#   - class_distribution_holdout.csv         (per-subject class counts - holdout)
#   - subject_predictions/subject_<ID>.csv   (per-row predictions with scores and labels)
#   - threshold_tuning_oof.csv               (OOF scores used to select decision threshold on train)
#   - threshold_curve_train_oof.csv          (F1/precision/recall across thresholds on OOF scores)
#   - threshold_curve_holdout_all.csv        (F1/precision/recall across thresholds on holdout - for diagnostics only)
#   - threshold_selected.txt                 (best threshold selected on train OOF)
#   - features_used.csv                      (the final feature list actually used)
#   - trained_model.joblib                   (fitted Pipeline: StandardScaler -> MLPClassifier)
#   - calibrator.joblib                      (optional Platt scaling calibrator fitted on train OOF)
#   - meta.json                              (basic run metadata)
#
# Notes:
#   - Uses the selected top-N features from step 05 (global_N in selection_summary.json).
#   - Tunes the decision threshold on the training set via subject-wise OOF to maximize F1.
#   - Optionally calibrates scores using logistic regression (Platt scaling) trained ONLY on train OOF.
#   - Applies train-based near-constant feature pruning (var < 1e-6) to mirror step 05.
#   - Robust to missing columns and inf/NaN values (train-based all-NaN drop + median imputation).
#
# Run:
#   python model/07_train_final_holdout.py
#
# Optional:
#   - Override base dir with env var PROCESSED_DIR (default: data/output).

import os
import json
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.class_weight import compute_sample_weight

# ---------------- CONFIG ----------------
SEED = 42
THRESH_GRID = np.linspace(0.1, 0.9, 81)

# Enable Platt scaling (logistic regression) on train OOF scores
CALIBRATE = True

# Near-constant variance threshold (mirror step 05)
VAR_THRESH = 1e-6

PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "data/output")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "final_features_ACCextractor_allmods_W30_S10_tau75.parquet")
FOLDS_DIR = os.path.join(PROCESSED_DIR, "folds_W30_S10_tau75")
HOLDOUT_LIST_PATH = os.path.join(FOLDS_DIR, "holdout_subjects.txt")

SELECTION_DIR = os.path.join(PROCESSED_DIR, "05_selection_loso")
SELECTION_SUMMARY_PATH = os.path.join(SELECTION_DIR, "selection_summary.json")
FEATURE_ORDER_CSV = os.path.join(SELECTION_DIR, "final_feature_order.csv")

OUT_DIR = os.path.join(PROCESSED_DIR, "07_holdout_evaluation")
PRED_DIR = os.path.join(OUT_DIR, "subject_predictions")
os.makedirs(PRED_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

META_COLS_CANDIDATES = [
    "subject", "session", "window_id",
    "start_idx", "end_idx", "W_sec", "S_sec",
    "category",
]
# ----------------------------------------


def load_feature_list(order_csv: str, selection_summary_json: str) -> List[str]:
    if not os.path.exists(selection_summary_json):
        raise FileNotFoundError(selection_summary_json)
    if not os.path.exists(order_csv):
        raise FileNotFoundError(order_csv)
    with open(selection_summary_json, "r") as f:
        ss = json.load(f)
    # Use exactly what's in selection_summary.json; fall back to 16 if missing
    global_N = int(ss.get("global_N", 16))
    df = pd.read_csv(order_csv)
    feats = df["feature"].tolist()
    return feats[:global_N] if global_N > 0 else feats


def impute_median(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    med = train.median(numeric_only=True)
    return train.fillna(med), test.fillna(med)


def prune_near_constant(X_tr: pd.DataFrame, X_te: pd.DataFrame, var_thresh: float) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # Drop all-NaN already done; now prune near-constant using train variance
    variances = X_tr.var(axis=0, ddof=0)
    keep_cols = variances[variances >= var_thresh].index.tolist()
    return X_tr[keep_cols], X_te[keep_cols], keep_cols


def get_final_model(seed: int) -> Pipeline:
    # Final chosen model (from step 06): MLP with StandardScaler
    return Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            alpha=0.0003,
            learning_rate="adaptive",
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        )),
    ])


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


def apply_calibrator(scores: np.ndarray, calibrator: LogisticRegression | None) -> np.ndarray:
    if calibrator is None:
        return scores
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    return calibrator.predict_proba(scores)[:, 1]


def threshold_curve(y_true: np.ndarray, scores: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in grid:
        pred = (scores >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        rows.append({"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f1)})
    return pd.DataFrame(rows)


def inner_oof_scores(model_factory, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, sample_weight: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subject-wise OOF predictions for threshold tuning. Returns:
      - oof_scores: shape (n_samples,)
      - fold_ids:   shape (n_samples,)
    """
    uniq = np.unique(groups)
    if len(uniq) >= 3:
        splitter = GroupKFold(n_splits=min(5, len(uniq))).split(X, y, groups)
    else:
        splitter = KFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y)

    oof_scores = np.zeros_like(y, dtype=float)
    seen = np.zeros_like(y, dtype=bool)
    fold_ids = np.full_like(y, fill_value=-1, dtype=int)

    for fold_idx, (tr, va) in enumerate(splitter):
        m = model_factory()
        fit_kwargs = {}
        if sample_weight is not None:
            # pass sample weights to the classifier step
            fit_kwargs["clf__sample_weight"] = sample_weight[tr]
        m.fit(X.iloc[tr], y[tr], **fit_kwargs)
        oof_scores[va] = predict_scores(m, X.iloc[va])
        seen[va] = True
        fold_ids[va] = fold_idx

    if not seen.all():
        # Unlikely, but just in case KFold drops something
        oof_scores[~seen] = np.mean(y)
        fold_ids[~seen] = -1

    return oof_scores, fold_ids


def select_threshold_from_scores(y: np.ndarray, scores: np.ndarray, grid: np.ndarray) -> float:
    best_f1, best_t = -1.0, 0.5
    for t in grid:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t


def read_holdouts(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        raw = f.read().strip()
    # Support comma or newline separated lists
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    holds = [p for p in parts if p and not p.startswith("#")]
    return holds


def safe_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Robust metric computation (classes might be single-valued per subject)
    out = {}
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["average_precision"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["average_precision"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})
    return out


def per_subject_class_counts(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    g = df.groupby("subject")[label_col].value_counts().unstack(fill_value=0)
    g = g.rename(columns={0: "neg", 1: "pos"}).reset_index()
    if "pos" not in g.columns:
        g["pos"] = 0
    if "neg" not in g.columns:
        g["neg"] = 0
    g["n"] = g["pos"] + g["neg"]
    g["pos_rate"] = g["pos"] / g["n"].replace(0, np.nan)
    return g.sort_values("subject")


def main():
    # Load selected features
    feature_cols = load_feature_list(FEATURE_ORDER_CSV, SELECTION_SUMMARY_PATH)

    # Load dataset
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH)

    # Binary labels only
    if "label" not in df.columns:
        raise KeyError("Column 'label' not found in features table.")
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    # Load holdouts
    holdouts = read_holdouts(HOLDOUT_LIST_PATH)
    holdout_set = set(holdouts)

    # Validate subject id formats exist in df
    unique_subjects = set(map(str, df["subject"].unique()))
    missing = [h for h in holdouts if h not in unique_subjects]
    if missing:
        print(f"[WARN] Holdout subjects not found in data: {missing}. "
              f"Available example subjects: {sorted(list(unique_subjects))[:10]}")

    # Split into train vs holdout by subject
    train_df = df[~df["subject"].isin(holdout_set)].copy()
    test_df = df[df["subject"].isin(holdout_set)].copy()

    if len(test_df) == 0:
        raise RuntimeError("Holdout test set is empty. Check subject IDs format in holdout_subjects.txt vs df['subject'].")

    # Keep only desired feature columns that exist
    feature_cols_real = [c for c in feature_cols if c in df.columns]
    if len(feature_cols_real) == 0:
        raise RuntimeError("None of the selected features exist in the features table. Check final_feature_order.csv and selection_summary.json.")

    # Optional: export class distributions for visibility
    per_subject_class_counts(train_df).to_csv(os.path.join(OUT_DIR, "class_distribution_train.csv"), index=False)
    per_subject_class_counts(test_df).to_csv(os.path.join(OUT_DIR, "class_distribution_holdout.csv"), index=False)

    # Train/test frames with cleaning
    X_tr = train_df[feature_cols_real].replace([np.inf, -np.inf], np.nan)
    X_te = test_df[feature_cols_real].replace([np.inf, -np.inf], np.nan)

    # Drop all-NaN columns based on train only
    keep = X_tr.columns[X_tr.notna().any(axis=0)].tolist()
    X_tr = X_tr[keep]
    X_te = X_te[keep]

    # Impute medians
    X_tr, X_te = impute_median(X_tr, X_te)

    # Prune near-constant features based on train variance (mirror step 05)
    X_tr, X_te, keep_var = prune_near_constant(X_tr, X_te, VAR_THRESH)

    # Prepare y and groups
    y_tr = train_df["label"].astype(int).to_numpy()
    y_te = test_df["label"].astype(int).to_numpy()
    groups_tr = train_df["subject"].to_numpy()

    # Compute sample weights for class balance (helps recall on rare positives)
    sw_tr = compute_sample_weight(class_weight="balanced", y=y_tr)

    # Train-side OOF scores for threshold tuning
    def model_factory():
        return get_final_model(SEED)

    oof_scores, fold_ids = inner_oof_scores(model_factory, X_tr, y_tr, groups_tr, sample_weight=sw_tr)

    # Optional Platt scaling calibrator fit on OOF scores (TRAIN ONLY)
    calibrator = None
    oof_scores_used = oof_scores.copy()
    if CALIBRATE:
        calibrator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        calibrator.fit(oof_scores.reshape(-1, 1), y_tr)
        oof_scores_used = apply_calibrator(oof_scores, calibrator)

    # Save OOF and threshold curve
    oof_df = pd.DataFrame({
        "score_raw": oof_scores,
        "score": oof_scores_used,
        "y": y_tr,
        "subject": groups_tr,
        "fold_idx": fold_ids,
    })
    oof_df.to_csv(os.path.join(OUT_DIR, "threshold_tuning_oof.csv"), index=False)

    curve_oof = threshold_curve(y_tr, oof_scores_used, THRESH_GRID)
    curve_oof.to_csv(os.path.join(OUT_DIR, "threshold_curve_train_oof.csv"), index=False)

    # Select threshold on OOF (calibrated if enabled)
    best_thr = select_threshold_from_scores(y_tr, oof_scores_used, THRESH_GRID)
    with open(os.path.join(OUT_DIR, "threshold_selected.txt"), "w") as f:
        f.write(f"{best_thr:.4f}\n")

    # Fit final model on full training data (with sample weights)
    final_model = model_factory()
    final_model.fit(X_tr, y_tr, clf__sample_weight=sw_tr)

    # Save calibrator if used
    if calibrator is not None:
        dump(calibrator, os.path.join(OUT_DIR, "calibrator.joblib"))

    # Evaluate per holdout subject
    per_subject_rows = []
    meta_cols = [c for c in META_COLS_CANDIDATES if c in test_df.columns]
    for subj, g in test_df.groupby("subject"):
        X_sub = X_te.loc[g.index]
        y_sub = g["label"].astype(int).to_numpy()

        scores_raw = predict_scores(final_model, X_sub)
        scores = apply_calibrator(scores_raw, calibrator)
        y_pred = (scores >= best_thr).astype(int)

        m = safe_metrics(y_sub, scores, y_pred)
        per_subject_rows.append({
            "subject": subj,
            "n": int(len(g)),
            "threshold": float(best_thr),
            **m,
        })

        # Save subject predictions
        pred_df = pd.DataFrame({
            "score_raw": scores_raw,
            "score": scores,
            "y_true": y_sub,
            "y_pred": y_pred,
        })
        if meta_cols:
            pred_df = pd.concat([g[meta_cols].reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
        pred_df.to_csv(os.path.join(PRED_DIR, f"subject_{subj}.csv"), index=False)

    # Save per-subject metrics
    holdout_metrics_df = pd.DataFrame(per_subject_rows)
    holdout_metrics_df.to_csv(os.path.join(OUT_DIR, "holdout_metrics.csv"), index=False)

    # Aggregate overall metrics across all holdouts
    scores_raw_all = predict_scores(final_model, X_te)
    scores_all = apply_calibrator(scores_raw_all, calibrator)
    y_pred_all = (scores_all >= best_thr).astype(int)

    # Diagnostics: threshold curve on the holdout set (for analysis only)
    curve_te = threshold_curve(y_te, scores_all, THRESH_GRID)
    curve_te.to_csv(os.path.join(OUT_DIR, "threshold_curve_holdout_all.csv"), index=False)

    # Confusion matrix aggregate
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred_all, labels=[0, 1]).ravel()
    agg = {
        "subjects": ",".join(holdouts),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(X_tr.shape[1]),
        "threshold": float(best_thr),
        "f1": float(f1_score(y_te, y_pred_all, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred_all)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "calibrated": bool(CALIBRATE),
    }
    try:
        agg["roc_auc"] = float(roc_auc_score(y_te, scores_all))
    except Exception:
        agg["roc_auc"] = float("nan")
    try:
        agg["average_precision"] = float(average_precision_score(y_te, scores_all))
    except Exception:
        agg["average_precision"] = float("nan")

    pd.DataFrame([agg]).to_csv(os.path.join(OUT_DIR, "aggregate_summary.csv"), index=False)

    # Save features actually used
    pd.DataFrame({"feature": X_tr.columns}).to_csv(os.path.join(OUT_DIR, "features_used.csv"), index=False)

    # Persist trained model
    dump(final_model, os.path.join(OUT_DIR, "trained_model.joblib"))

    # Save meta
    meta = {
        "seed": SEED,
        "processed_dir": PROCESSED_DIR,
        "features_path": FEATURES_PATH,
        "folds_dir": FOLDS_DIR,
        "holdout_subjects": holdouts,
        "selection_dir": SELECTION_DIR,
        "feature_order_csv": FEATURE_ORDER_CSV,
        "n_features_used": int(X_tr.shape[1]),
        "var_thresh": VAR_THRESH,
        "calibrated": CALIBRATE,
        "model": "Pipeline(StandardScaler -> MLPClassifier(64,64,relu,alpha=0.0003,adaptive,max_iter=1000,early_stopping=20))",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Holdout evaluation complete. Results at: {OUT_DIR}")
    print(f"Aggregate F1={agg['f1']:.4f}, BAcc={agg['balanced_accuracy']:.4f}, ROC-AUC={agg['roc_auc']}, AP={agg['average_precision']}")


if __name__ == "__main__":
    main()