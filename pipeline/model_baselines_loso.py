#!/usr/bin/env python3
"""
Baseline model comparison with leakage-safe preprocessing.

Data:
- Reads pipeline/features/ALL_features_pruned.parquet (strict; errors if missing)
- Uses 'label' as target; groups by 'subject'.

Protocol:
- Holdouts: S10,S11 (configurable). These are NOT used in training or CV; they're used only for final evaluation.
- On the remaining subjects, run GroupKFold LOSO (subject-wise) to estimate generalization.
- Preprocessing is fit on each train fold then applied to val fold (no leakage).
- Models compared: Logistic Regression (L2/L1), Linear SVM (L2/L1), RBF SVM (optional), RandomForest, ExtraTrees, HistGradientBoosting, small MLP.

Outputs under pipeline/results/baselines/:
- per_fold_metrics.csv      Metrics per model per LOSO fold (subject)
- summary_metrics.csv       Aggregated metrics per model (mean, std across folds)
- holdout_metrics.csv       Metrics per model on S10,S11 (and combined)
- run_config.json           Config used (models, params, subjects)

Env flags:
- HOLDOUT_SUBJECTS="S10,S11"
- FAST=1                   Default ON: subsample training rows to ~6000 and skip RBF SVM
- N_JOBS=auto              n_jobs for tree/XGB; auto -> use -1
- USE_XGB=0                If 1, include XGBoost; errors if xgboost missing (no fallback)
- VERBOSE=1                extra logging
Model filtering:
- DISABLE_MODELS="logreg_l1"   Comma-separated model names to skip entirely
- ONLY_MODEL=                  If set, run only this model (e.g., "linear_svm_l1")
Preprocessing/cleanup:
- ALLOW_INF_TO_NAN=0       If 1, replace +/-inf with NaN (then impute); else error on inf
- INF_DROP_FRAC=0          Drop feature columns with inf-rate > this fraction (pre-scan on train)
- DROP_NA_FRAC=0           Drop columns whose NaN fraction exceeds this (fit per train fold)
- WINSOR=0                 If 1, clip features to quantile bounds from train fold
- WINSOR_Q="0.005,0.995"   Quantiles used by winsorization when WINSOR=1
Feature selection/regularization:
- SELECT_K=0               If >0, SelectKBest(mutual_info) with k features (leakage-safe)
- L1_SELECT=0              If 1, add L1-based selector (LogReg saga) to scaled pipelines
- L1_C=0.1                 Inverse regularization strength for L1 selector
- L1_K=0                   If >0, keep top-k by |coef|; else keep all non-zero weights
- L1_MAX_ITER=5000         Max iter for L1 selector
Model knobs:
- LOGREG_C=1.0             L2 LogisticRegression C
- LOGREG_L1_C=0.5          L1 LogisticRegression C
- LINSVM_C=1.0             LinearSVC C
- LINSVM_L1_C=0.5          LinearSVC with L1 penalty C
- RF_TREES=300 RF_DEPTH=12 RF_LEAF=1
- ET_TREES=400 ET_DEPTH=12 ET_LEAF=1
- HGB_DEPTH=6 HGB_LR=0.1 HGB_ITERS=300
- MLP_HIDDEN="64,32" MLP_ALPHA=1e-4
Class imbalance handling:
- CLASS_WEIGHT_MODE=none   Options: none | balanced
- STRESS_LABELS=           Comma-separated label values to upweight (e.g., "2" for WESAD stress)
- STRESS_W_MULT=1.0        Multiplier applied to listed labels' sample weights (e.g., 1.5 or 2.0)
Decision threshold optimization (binary only):
- THRESH_OPT=0             If 1, optimize decision threshold on training fold via inner GroupKFold
- THRESH_BETA=2.0          Beta for F-beta optimization (default aligns with F2)
- THRESH_GRID="0.1,0.2,...,0.9"  Comma-separated thresholds to search
- THRESH_INNER_SPLITS=5    Number of inner GroupKFold splits (capped by subjects in train fold)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.class_weight import compute_class_weight


PIPELINE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = PIPELINE_DIR / "features"
RESULTS_DIR = PIPELINE_DIR / "results" / os.getenv("RESULTS_SUBDIR", "baselines")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip() in ("1", "true", "True", "yes", "y")


def _get_data_path_strict() -> Path:
    # Allow explicit override first
    override = os.getenv("FEATURES_COMBINED_PATH")
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = FEATURES_DIR / p
        if not p.exists():
            raise FileNotFoundError(f"FEATURES_COMBINED_PATH set but file not found: {p}")
        return p
    # Strict: require pruned file; do not fallback silently.
    pruned = FEATURES_DIR / "ALL_features_pruned.parquet"
    if not pruned.exists():
        raise FileNotFoundError(f"Required file missing: {pruned}. Run pruning first or set FEATURES_COMBINED_PATH to the dataset file.")
    return pruned


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("subject", "start", "label")]


def _scan_inf(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    # Return per-column count of inf values
    isinf = np.isinf(df[cols].to_numpy())
    counts = isinf.sum(axis=0)
    return pd.Series(counts, index=cols)


def _handle_inf_strict(df: pd.DataFrame, cols: List[str], allow_fix: bool) -> pd.DataFrame:
    counts = _scan_inf(df, cols)
    total_inf = int(counts.sum())
    if total_inf == 0:
        return df
    if not allow_fix:
        offenders = counts[counts > 0].sort_values(ascending=False)
        top = offenders.head(15)
        details = "\n".join([f"  - {k}: {int(v)}" for k, v in top.items()])
        raise ValueError(
            f"Found {total_inf} infinite values in features. Set ALLOW_INF_TO_NAN=1 to replace inf with NaN and proceed. Top offending columns:\n{details}"
        )
    # Replace inf with NaN explicitly
    df2 = df.copy()
    df2[cols] = df2[cols].replace([np.inf, -np.inf], np.nan)
    return df2


class NaNColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns with NaN fraction > max_na_frac (fit on train only)."""
    def __init__(self, max_na_frac: float = 0.5):
        self.max_na_frac = float(max_na_frac)
        self.keep_cols: List[str] = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NaNColumnDropper expects a pandas DataFrame input")
        na_frac = X.isna().mean()
        self.keep_cols = [c for c in X.columns if na_frac[c] <= self.max_na_frac]
        if not self.keep_cols:
            raise ValueError("NaNColumnDropper dropped all columns; relax max_na_frac or inspect data")
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NaNColumnDropper expects a pandas DataFrame input")
        missing = [c for c in self.keep_cols if c not in X.columns]
        if missing:
            raise ValueError(f"NaNColumnDropper missing expected columns at transform: {missing}")
        return X[self.keep_cols]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip values to quantile bounds learned on train (ignoring NaNs)."""
    def __init__(self, q_low: float = 0.005, q_high: float = 0.995):
        assert 0.0 <= q_low < q_high <= 1.0
        self.q_low = q_low
        self.q_high = q_high
        self.bounds: Dict[str, Tuple[float, float]] = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Winsorizer expects a pandas DataFrame input")
        lows = X.quantile(self.q_low, numeric_only=True)
        highs = X.quantile(self.q_high, numeric_only=True)
        self.bounds = {}
        for c in X.columns:
            lo = lows.get(c, np.nan)
            hi = highs.get(c, np.nan)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                self.bounds[c] = (float(lo), float(hi))
        if not self.bounds:
            raise ValueError("Winsorizer found no valid bounds; check data or quantiles")
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Winsorizer expects a pandas DataFrame input")
        Xc = X.copy()
        for c, (lo, hi) in self.bounds.items():
            if c in Xc.columns:
                Xc[c] = Xc[c].clip(lower=lo, upper=hi)
        return Xc


class L1Selector(BaseEstimator, TransformerMixin):
    """Select features via L1-penalized Logistic Regression coefficients.
    If k>0, keep top-k by |coef|; else keep all non-zero-weight features.
    """
    def __init__(self, C: float = 0.1, max_iter: int = 5000, k: int = 0):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.k = int(k)
        self.keep_idx_: np.ndarray | None = None

    def fit(self, X, y):
        from sklearn.utils.validation import check_is_fitted
        # Work with numpy arrays; assume any scaling done earlier in pipeline
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
        clf = LogisticRegression(penalty="l1", solver="saga", C=self.C, class_weight="balanced", max_iter=self.max_iter)
        clf.fit(X_arr, y)
        coefs = np.abs(clf.coef_)
        # For multiclass, aggregate importance across classes
        importance = coefs.max(axis=0)
        if self.k > 0:
            # Select top-k indices
            if self.k > importance.size:
                raise ValueError(f"L1Selector k={self.k} > n_features={importance.size}")
            self.keep_idx_ = np.argpartition(-importance, self.k - 1)[: self.k]
        else:
            self.keep_idx_ = np.flatnonzero(importance > 0)
        if self.keep_idx_.size == 0:
            raise ValueError("L1Selector kept zero features; increase C or reduce regularization")
        return self

    def transform(self, X):
        if self.keep_idx_ is None:
            raise RuntimeError("L1Selector not fitted")
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.keep_idx_]
        X_arr = np.asarray(X)
        return X_arr[:, self.keep_idx_]


def _build_models(skip_rbf: bool, n_jobs: int, use_xgb: bool) -> Dict[str, Pipeline]:
    # Pipelines per model ensuring leakage-safe preprocessing
    models: Dict[str, Pipeline] = {}
    # Shared preprocessors with optional NaN drop and winsorization (fit on train only)
    drop_na_frac = float(os.getenv("DROP_NA_FRAC", "0"))
    use_winsor = _env_bool("WINSOR", False)
    q_low, q_high = (float(x) for x in os.getenv("WINSOR_Q", "0.005,0.995").split(",")) if use_winsor else (0.005, 0.995)

    pre_base: List[Tuple[str, BaseEstimator]] = []
    if drop_na_frac > 0.0:
        pre_base.append(("dropna_cols", NaNColumnDropper(max_na_frac=drop_na_frac)))
    if use_winsor:
        pre_base.append(("winsor", Winsorizer(q_low=q_low, q_high=q_high)))

    pre_scale = Pipeline(pre_base + [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ])
    pre_noscale = Pipeline(pre_base + [
        ("impute", SimpleImputer(strategy="median")),
    ])

    # Logistic Regression (L2)
    logreg_c = float(os.getenv("LOGREG_C", "1.0"))
    models["logreg_l2"] = Pipeline([
        ("prep", pre_scale),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", C=logreg_c)),
    ])

    # Logistic Regression (L1)
    logreg_l1_c = float(os.getenv("LOGREG_L1_C", "0.5"))
    models["logreg_l1"] = Pipeline([
        ("prep", pre_scale),
        ("clf", LogisticRegression(max_iter=3000, solver="saga", penalty="l1", class_weight="balanced", C=logreg_l1_c)),
    ])

    # Linear SVM
    lsvm_c = float(os.getenv("LINSVM_C", "1.0"))
    models["linear_svm"] = Pipeline([
        ("prep", pre_scale),
        ("clf", LinearSVC(C=lsvm_c, class_weight="balanced", max_iter=8000)),
    ])

    # Linear SVM with L1 (requires dual=False)
    lsvm_l1_c = float(os.getenv("LINSVM_L1_C", "0.5"))
    models["linear_svm_l1"] = Pipeline([
        ("prep", pre_scale),
        ("clf", LinearSVC(C=lsvm_l1_c, penalty="l1", dual=False, class_weight="balanced", max_iter=8000)),
    ])

    # RBF SVM intentionally skipped when skip_rbf=True (FAST default)
    if not skip_rbf:
        models["svm_rbf"] = Pipeline([
            ("prep", pre_scale),
            ("clf", SVC(C=1.0, gamma="scale", class_weight="balanced")),
        ])

    # Tree ensembles (no scaling) with env-tunable capacity
    rf_trees = int(os.getenv("RF_TREES", "300"))
    rf_depth = int(os.getenv("RF_DEPTH", "12"))
    rf_leaf = int(os.getenv("RF_LEAF", "1"))
    et_trees = int(os.getenv("ET_TREES", "400"))
    et_depth = int(os.getenv("ET_DEPTH", "12"))
    et_leaf = int(os.getenv("ET_LEAF", "1"))
    models["rf"] = Pipeline([
        ("prep", pre_noscale),
        ("clf", RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth, min_samples_leaf=rf_leaf, n_jobs=n_jobs, class_weight="balanced_subsample", random_state=42)),
    ])
    models["extratrees"] = Pipeline([
        ("prep", pre_noscale),
        ("clf", ExtraTreesClassifier(n_estimators=et_trees, max_depth=et_depth, min_samples_leaf=et_leaf, n_jobs=n_jobs, class_weight="balanced_subsample", random_state=42)),
    ])

    # Histogram Gradient Boosting (handles missing internally but we impute anyway)
    hgb_depth = int(os.getenv("HGB_DEPTH", "6"))
    hgb_lr = float(os.getenv("HGB_LR", "0.1"))
    hgb_iters = int(os.getenv("HGB_ITERS", "300"))
    models["hgb"] = Pipeline([
        ("prep", pre_noscale),
        ("clf", HistGradientBoostingClassifier(max_depth=hgb_depth, learning_rate=hgb_lr, max_iter=hgb_iters, random_state=42)),
    ])

    # Small MLP with env-tunable hidden sizes
    hidden_env = os.getenv("MLP_HIDDEN", "64,32")
    hidden = tuple(int(x.strip()) for x in hidden_env.split(',') if x.strip()) if hidden_env else (64, 32)
    mlp_alpha = float(os.getenv("MLP_ALPHA", "1e-4"))
    models["mlp_small"] = Pipeline([
        ("prep", pre_scale),
        ("clf", MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=mlp_alpha, batch_size=256,
                               learning_rate_init=1e-3, max_iter=100, early_stopping=True, random_state=42)),
    ])

    # Optional XGBoost (strict: error if requested but not installed)
    if use_xgb:
        try:
            from xgboost import XGBClassifier  # type: ignore
        except Exception as e:
            raise ImportError("USE_XGB=1 set but xgboost is not available. Please install xgboost.") from e
        models["xgb"] = Pipeline([
            ("prep", pre_noscale),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=n_jobs,
                random_state=42,
                tree_method="hist",
                importance_type="gain",
            )),
        ])

    return models


def _metrics(y_true, y_pred) -> Dict[str, float]:
    # Primary: F2 (macro). Also report F1 (macro), precision/recall (macro), accuracy, and balanced accuracy.
    return {
        "f2_macro": float(fbeta_score(y_true, y_pred, beta=2.0, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _compute_sample_weights(y: pd.Series, mode: str, stress_labels: List[str], stress_mult: float) -> np.ndarray:
    # Base weights
    w = np.ones(len(y), dtype=float)
    classes = np.unique(y)
    if mode.lower() == "balanced":
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        cw_map = {cls: cw[i] for i, cls in enumerate(classes)}
        w = np.array([cw_map[val] for val in y], dtype=float)
    # Optional extra boost for specified labels
    if stress_labels and stress_mult and stress_mult != 1.0:
        boost_set = set(stress_labels)
        for i, val in enumerate(y):
            if str(val) in boost_set:
                w[i] *= stress_mult
    return w


def _get_scores(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    # Prefer predict_proba for class 1; fallback to decision_function; else None
    clf = pipe.named_steps.get("clf")
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X)
        # decision_function can be 1D or (n_samples, n_classes)
        if scores.ndim == 1:
            # scale to [0,1] via sigmoid-like mapping for thresholding consistency
            s = np.asarray(scores, dtype=float)
            # normalize to [0,1]
            s = (s - s.min()) / (s.max() - s.min() + 1e-9)
            return s
        elif scores.ndim == 2 and scores.shape[1] == 2:
            s = scores[:, 1]
            s = (s - s.min()) / (s.max() - s.min() + 1e-9)
            return s
    raise AttributeError("Pipeline does not support probability or decision scores for thresholding")


def _optimize_threshold(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                        fit_params: dict, beta: float, grid: List[float], inner_splits: int, verbose: bool) -> float:
    # Inner GroupKFold over training subjects
    uniq_subj = sorted(groups.unique())
    n_splits = min(len(uniq_subj), max(2, inner_splits))
    gkf_inner = GroupKFold(n_splits=n_splits)
    thr_scores = {t: [] for t in grid}
    for tr_idx, va_idx in gkf_inner.split(X, y, groups=groups):
        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        Xva, yva = X.iloc[va_idx], y.iloc[va_idx]
        model = clone(pipe)
        try:
            model.fit(Xtr, ytr, **fit_params)
        except TypeError:
            model.fit(Xtr, ytr)
        try:
            s = _get_scores(model, Xva)
        except Exception:
            # If scores not available, fall back to labels and break
            return 0.5
        for t in grid:
            yhat = (s >= t).astype(int)
            thr_scores[t].append(fbeta_score(yva, yhat, beta=beta, average="binary", zero_division=0))
    # Pick threshold with best mean F-beta
    best_t = max(thr_scores.keys(), key=lambda t: float(np.mean(thr_scores[t])) if thr_scores[t] else -1.0)
    if verbose:
        means = {t: float(np.mean(v)) if v else 0.0 for t, v in thr_scores.items()}
        top = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print(f"THRESH_OPT candidates (top): {top}")
    return float(best_t)


def main():
    verbose = _env_bool("VERBOSE", False)
    fast = _env_bool("FAST", True)  # default to FAST
    n_jobs_env = os.getenv("N_JOBS", "auto")
    n_jobs = -1 if n_jobs_env == "auto" else int(n_jobs_env)
    holdouts_env = os.getenv("HOLDOUT_SUBJECTS", "S10,S11")
    holdouts = {s.strip() for s in holdouts_env.split(',') if s.strip()}
    use_xgb = _env_bool("USE_XGB", False)

    used_path = _get_data_path_strict()
    df = pd.read_parquet(used_path)

    # Split train/holdout by subject
    mask_hold = df["subject"].isin(list(holdouts))
    df_train = df.loc[~mask_hold].reset_index(drop=True)
    df_hold = df.loc[mask_hold].reset_index(drop=True)
    if verbose:
        print(f"Data: {used_path.name} | Train subjects={sorted(df_train['subject'].unique())} | Holdouts={sorted(df_hold['subject'].unique())}")
        print(f"Shapes: train={df_train.shape}, holdout={df_hold.shape}")

    # Optionally subsample for speed
    if fast and len(df_train) > 6000:
        df_train = df_train.sample(n=6000, random_state=42)
        if verbose:
            print(f"FAST=1 -> subsampled train to {len(df_train)} rows")

    X_cols = _feature_cols(df_train)
    # Optional: restrict to a fixed feature list file (JSON array of names)
    feat_list_file = os.getenv("FEATURE_LIST_FILE")
    if feat_list_file:
        try:
            with open(feat_list_file, "r") as f:
                desired = json.load(f)
            if not isinstance(desired, list):
                raise ValueError("FEATURE_LIST_FILE must contain a JSON list of feature names")
            desired_set = set(map(str, desired))
            keep = [c for c in X_cols if c in desired_set]
            if not keep:
                raise ValueError(f"FEATURE_LIST_FILE yielded 0 intersecting features. File={feat_list_file}")
            dropped = [c for c in X_cols if c not in desired_set]
            if verbose:
                print(f"Restricting to FEATURE_LIST_FILE: keeping {len(keep)} features, dropping {len(dropped)}")
            X_cols = keep
            df_train = df_train[["subject", "start", "label"] + X_cols]
            df_hold = df_hold[["subject", "start", "label"] + X_cols]
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to apply FEATURE_LIST_FILE={feat_list_file}: {e}")
    # Optional: drop features with high INF rate before fixing
    inf_drop_frac_env = os.getenv("INF_DROP_FRAC", "0")
    try:
        inf_drop_frac = float(inf_drop_frac_env)
    except Exception:
        inf_drop_frac = 0.0
    if inf_drop_frac > 0.0:
        counts_train = _scan_inf(df_train, X_cols)
        nrows = max(1, len(df_train))
        drop_inf_cols = [c for c, cnt in counts_train.items() if (cnt / nrows) > inf_drop_frac]
        if drop_inf_cols:
            if verbose:
                print(f"Dropping {len(drop_inf_cols)} features for INF rate > {inf_drop_frac}")
            keep = [c for c in X_cols if c not in drop_inf_cols]
            X_cols = keep
            df_train = df_train[["subject", "start", "label"] + X_cols]
            df_hold = df_hold[["subject", "start", "label"] + X_cols]

    # Strict infinity handling
    allow_fix_inf = _env_bool("ALLOW_INF_TO_NAN", False)
    df_train = _handle_inf_strict(df_train, X_cols, allow_fix=allow_fix_inf)
    df_hold = _handle_inf_strict(df_hold, X_cols, allow_fix=allow_fix_inf)

    X_train = df_train[X_cols]
    y_train = df_train["label"]
    groups = df_train["subject"]

    # Include RBF SVM only when FAST=0 (skip when fast)
    models = _build_models(skip_rbf=fast, n_jobs=n_jobs, use_xgb=use_xgb)

    # Optional: leakage-safe univariate MI feature selection (SelectKBest) inside each pipeline
    select_k_env = os.getenv("SELECT_K", "0")
    try:
        select_k = int(select_k_env)
    except Exception:
        select_k = 0
    if select_k > 0:
        if verbose:
            print(f"Enabling SelectKBest(mutual_info) with k={select_k}")
        # Wrap each model pipeline: insert 'select' step before classifier
        new_models: Dict[str, Pipeline] = {}
        for name, pipe in models.items():
            steps = pipe.steps
            # Find position of classifier (last step)
            clf_name, clf_step = steps[-1]
            pre_steps = steps[:-1]
            # Insert select step after preprocessing
            pre_steps.append(("select_mi", SelectKBest(score_func=mutual_info_classif, k=select_k)))
            new_pipe = Pipeline(pre_steps + [(clf_name, clf_step)])
            new_models[name] = new_pipe
        models = new_models

    # Optional: L1-based selector (LogisticRegression L1 saga) for scaled pipelines
    if _env_bool("L1_SELECT", False):
        l1_c = float(os.getenv("L1_C", "0.1"))
        l1_k = int(os.getenv("L1_K", "0"))
        l1_max_iter = int(os.getenv("L1_MAX_ITER", "5000"))
        scaled_model_names = {"logreg_l2", "logreg_l1", "linear_svm", "linear_svm_l1", "mlp_small"}
        if verbose:
            print(f"Enabling L1 selector for scaled models: C={l1_c}, k={l1_k}, max_iter={l1_max_iter}")
        new_models: Dict[str, Pipeline] = {}
        for name, pipe in models.items():
            steps = pipe.steps
            clf_name, clf_step = steps[-1]
            pre_steps = steps[:-1]
            if name in scaled_model_names:
                pre_steps.append(("select_l1", L1Selector(C=l1_c, max_iter=l1_max_iter, k=l1_k)))
            new_models[name] = Pipeline(pre_steps + [(clf_name, clf_step)])
        models = new_models

    # Optionally disable some models entirely
    disable_env = os.getenv("DISABLE_MODELS", "logreg_l1")
    disabled = {s.strip() for s in disable_env.split(',') if s.strip()}
    if disabled:
        models = {k: v for k, v in models.items() if k not in disabled}
        if verbose:
            present = ", ".join(sorted(disabled))
            print(f"Disabled models: {present}")

    # Optionally keep only a single model
    only_model = os.getenv("ONLY_MODEL")
    if only_model:
        if only_model not in models:
            raise ValueError(f"ONLY_MODEL={only_model} not among available: {list(models.keys())}")
        models = {only_model: models[only_model]}
        if verbose:
            print(f"Running ONLY_MODEL: {only_model}")

    # LOSO across training subjects
    gkf = GroupKFold(n_splits=len(df_train["subject"].unique()))
    per_fold_rows = []

    # Class imbalance config
    cw_mode = os.getenv("CLASS_WEIGHT_MODE", "none")
    stress_labels_env = os.getenv("STRESS_LABELS", "")
    stress_labels = [s.strip() for s in stress_labels_env.split(',') if s.strip()]
    stress_mult = float(os.getenv("STRESS_W_MULT", "1.0"))

    # Threshold optimization config
    thresh_opt = _env_bool("THRESH_OPT", False)
    thresh_beta = float(os.getenv("THRESH_BETA", "2.0"))
    thresh_grid_env = os.getenv("THRESH_GRID", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    thresh_grid = [float(x.strip()) for x in thresh_grid_env.split(',') if x.strip()]
    thresh_inner_splits = int(os.getenv("THRESH_INNER_SPLITS", "5"))

    for model_name, pipe in models.items():
        fold_idx = 0
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            fold_idx += 1
            Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
            ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]
            val_subj = groups.iloc[val_idx].iloc[0]
            # Build sample weights if requested
            fit_params = {}
            if cw_mode.lower() != "none" or stress_labels:
                sw = _compute_sample_weights(ytr, cw_mode, stress_labels, stress_mult)
                fit_params = {"clf__sample_weight": sw}

            # Fit, passing sample_weight when supported
            try:
                pipe.fit(Xtr, ytr, **fit_params)
            except TypeError:
                pipe.fit(Xtr, ytr)
            # Train score (overfitting diagnostic) with optional thresholding
            yhat_tr = None
            if thresh_opt:
                # Optimize threshold on the training subset using inner GroupKFold
                try:
                    t_star = _optimize_threshold(pipe, Xtr, ytr, groups.iloc[train_idx], fit_params, thresh_beta, thresh_grid, thresh_inner_splits, verbose)
                    try:
                        s_tr = _get_scores(pipe, Xtr)
                        yhat_tr = (s_tr >= t_star).astype(int)
                    except Exception:
                        yhat_tr = pipe.predict(Xtr)
                except Exception:
                    yhat_tr = pipe.predict(Xtr)
            else:
                yhat_tr = pipe.predict(Xtr)
            m_tr = _metrics(ytr, yhat_tr)
            m_tr.update({"model": model_name, "fold": fold_idx, "val_subject": val_subj, "split": "train"})
            per_fold_rows.append(m_tr)

            # Validation score using same threshold if available
            if thresh_opt:
                try:
                    s_val = _get_scores(pipe, Xval)
                    yhat = (s_val >= t_star).astype(int)
                except Exception:
                    yhat = pipe.predict(Xval)
            else:
                yhat = pipe.predict(Xval)
            m = _metrics(yval, yhat)
            m.update({"model": model_name, "fold": fold_idx, "val_subject": val_subj, "split": "val"})
            per_fold_rows.append(m)
            if verbose:
                print(f"{model_name} | fold {fold_idx:02d} | subj={val_subj} | VAL acc={m['acc']:.3f} bal_acc={m['bal_acc']:.3f} f1={m['f1_macro']:.3f} F2={m['f2_macro']:.3f}")

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(RESULTS_DIR / "per_fold_metrics.csv", index=False)

    # Aggregate summary across folds per model and split
    metric_cols = ["f2_macro", "f1_macro", "precision_macro", "recall_macro", "acc", "bal_acc"]
    summ = per_fold_df.groupby(["model", "split"])[metric_cols].agg(["mean", "std"]).reset_index()
    # Flatten columns
    summ.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in summ.columns.values]
    summ.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

    # Overfit/underfit report: gap between train and val (mean per model)
    g = per_fold_df.pivot_table(index=["model", "fold"], columns="split", values="f2_macro").reset_index()
    if {"train", "val"}.issubset(set(g.columns)):
        g["f2_gap_train_minus_val"] = g["train"] - g["val"]
        gap = g.groupby("model")["f2_gap_train_minus_val"].agg(["mean", "std", "min", "max"]).reset_index()
        gap.to_csv(RESULTS_DIR / "overfit_gap_f2.csv", index=False)

    # Fit on all training; evaluate on holdout subjects separately and combined
    hold_rows = []
    X_hold = df_hold[X_cols]
    y_hold = df_hold["label"]
    # Per holdout subject and combined
    for model_name, pipe in models.items():
        # Fit with sample weights on full train if configured
        fit_params = {}
        if cw_mode.lower() != "none" or stress_labels:
            sw_full = _compute_sample_weights(y_train, cw_mode, stress_labels, stress_mult)
            fit_params = {"clf__sample_weight": sw_full}
        try:
            pipe.fit(X_train, y_train, **fit_params)
        except TypeError:
            pipe.fit(X_train, y_train)
        # Optional threshold optimization on full train
        if thresh_opt:
            try:
                t_full = _optimize_threshold(pipe, X_train, y_train, groups, fit_params, thresh_beta, thresh_grid, thresh_inner_splits, verbose)
                try:
                    s_all = _get_scores(pipe, X_hold)
                    yhat_all = (s_all >= t_full).astype(int)
                except Exception:
                    yhat_all = pipe.predict(X_hold)
            except Exception:
                yhat_all = pipe.predict(X_hold)
        else:
            # Combined
            yhat_all = pipe.predict(X_hold)
        m_all = _metrics(y_hold, yhat_all)
        m_all.update({"model": model_name, "split": "holdout_all"})
        hold_rows.append(m_all)
        # Per subject
        for subj in sorted(df_hold["subject"].unique()):
            mask = df_hold["subject"] == subj
            ytrue = y_hold[mask]
            yhat = yhat_all[mask]
            m = _metrics(ytrue, yhat)
            m.update({"model": model_name, "split": f"holdout_{subj}"})
            hold_rows.append(m)

    hold_df = pd.DataFrame(hold_rows)
    hold_df.to_csv(RESULTS_DIR / "holdout_metrics.csv", index=False)

    # Save run config
    config = {
        "data_path": str(used_path),
        "n_models": len(models),
        "models": list(models.keys()),
        "disabled_models": sorted(list(disabled)),
        "holdouts": sorted(list(holdouts)),
        "fast": fast,
        "n_jobs": n_jobs,
        "n_train_subjects": int(len(df_train["subject"].unique())),
    }
    (RESULTS_DIR / "run_config.json").write_text(json.dumps(config, indent=2))

    print(f"Saved per-fold metrics -> {RESULTS_DIR / 'per_fold_metrics.csv'}")
    print(f"Saved summary metrics  -> {RESULTS_DIR / 'summary_metrics.csv'}")
    print(f"Saved holdout metrics  -> {RESULTS_DIR / 'holdout_metrics.csv'}")
    print(f"Saved run config       -> {RESULTS_DIR / 'run_config.json'}")


if __name__ == "__main__":
    main()
