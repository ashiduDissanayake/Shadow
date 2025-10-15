#!/usr/bin/env python3
"""
Leakage-safe stability feature selection across LOSO folds.

- Input: pipeline/features/ALL_features_pruned.parquet (strict)
- Excludes holdouts (HOLDOUT_SUBJECTS) for selection.
- For each LOSO fold over training subjects:
  * Pre-clean: optional INF drop, inf->NaN, high-NaN drop (per train fold)
  * Preprocess: impute median, standardize
  * Compute rankings via:
      - mutual_info_classif (univariate)
      - LogisticRegression L1 (saga) coefficient magnitudes
      - ExtraTrees feature_importances_
  * Combine per-fold ranks into a simple average combined rank
  * Record per-fold combined ranks per feature

Outputs (under pipeline/results/feature_selection/ or RESULTS_SUBDIR_FS):
  - consensus_ranking.csv: feature, ranks per method, mean ranks, stability@K
  - top_k_lists/top_{K}.json: ordered feature list of top K by consensus

Env:
  - HOLDOUT_SUBJECTS="S10,S11"
  - FAST=1 (sample up to 6000 train rows)
  - ALLOW_INF_TO_NAN=0 | INF_DROP_FRAC=0 | DROP_NA_FRAC=0.5
  - L1SEL_C=0.1 | L1SEL_MAX_ITER=5000
  - ET_TREES=300 | ET_DEPTH=12 | N_JOBS=auto
  - K_GRID="32,64,96,128"
  - RESULTS_SUBDIR_FS (default: feature_selection)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier


PIPELINE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = PIPELINE_DIR / "features"
RESULTS_DIR = PIPELINE_DIR / "results" / os.getenv("RESULTS_SUBDIR_FS", "feature_selection")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip() in ("1", "true", "True", "yes", "y")


def _get_data_path_strict() -> Path:
    pruned = FEATURES_DIR / "ALL_features_pruned.parquet"
    if not pruned.exists():
        raise FileNotFoundError(f"Required file missing: {pruned}. Run pruning first.")
    return pruned


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("subject", "start", "label")]


def _scan_inf(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    isinf = np.isinf(df[cols].to_numpy())
    counts = isinf.sum(axis=0)
    return pd.Series(counts, index=cols)


def _handle_inf_strict(df: pd.DataFrame, cols: List[str], allow_fix: bool) -> pd.DataFrame:
    isinf = np.isinf(df[cols].to_numpy())
    total_inf = int(isinf.sum())
    if total_inf == 0:
        return df
    if not allow_fix:
        offenders = _scan_inf(df, cols)
        offenders = offenders[offenders > 0].sort_values(ascending=False)
        top = offenders.head(15)
        details = "\n".join([f"  - {k}: {int(v)}" for k, v in top.items()])
        raise ValueError(
            f"Found {total_inf} infinite values in features. Set ALLOW_INF_TO_NAN=1 to replace inf with NaN and proceed. Top offending columns:\n{details}"
        )
    df2 = df.copy()
    df2[cols] = df2[cols].replace([np.inf, -np.inf], np.nan)
    return df2


def main():
    verbose = _env_bool("VERBOSE", False)
    fast = _env_bool("FAST", True)
    n_jobs_env = os.getenv("N_JOBS", "auto")
    n_jobs = -1 if n_jobs_env == "auto" else int(n_jobs_env)
    holdouts_env = os.getenv("HOLDOUT_SUBJECTS", "S10,S11")
    holdouts = {s.strip() for s in holdouts_env.split(',') if s.strip()}

    used_path = _get_data_path_strict()
    df = pd.read_parquet(used_path)
    mask_hold = df["subject"].isin(list(holdouts))
    df_train = df.loc[~mask_hold].reset_index(drop=True)
    if fast and len(df_train) > 6000:
        df_train = df_train.sample(n=6000, random_state=42)
        if verbose:
            print(f"FAST=1 -> subsampled train to {len(df_train)} rows")

    X_cols = _feature_cols(df_train)

    # Pre-scan INF: optionally drop high-inf features
    inf_drop_frac = float(os.getenv("INF_DROP_FRAC", "0"))
    if inf_drop_frac > 0.0:
        counts_train = _scan_inf(df_train, X_cols)
        nrows = max(1, len(df_train))
        drop_inf_cols = [c for c, cnt in counts_train.items() if (cnt / nrows) > inf_drop_frac]
        if drop_inf_cols and verbose:
            print(f"Dropping {len(drop_inf_cols)} features for INF rate > {inf_drop_frac}")
        X_cols = [c for c in X_cols if c not in drop_inf_cols]
        df_train = df_train[["subject", "start", "label"] + X_cols]

    allow_fix_inf = _env_bool("ALLOW_INF_TO_NAN", False)
    df_train = _handle_inf_strict(df_train, X_cols, allow_fix=allow_fix_inf)

    drop_na_frac = float(os.getenv("DROP_NA_FRAC", "0"))

    # Prepare structures to accumulate per-fold ranks
    subjects = sorted(df_train["subject"].unique())
    gkf = GroupKFold(n_splits=len(subjects))
    per_fold_ranks: Dict[str, List[float]] = {c: [] for c in X_cols}
    per_method_ranks: Dict[str, Dict[str, List[float]]] = {
        "mi": {c: [] for c in X_cols},
        "l1": {c: [] for c in X_cols},
        "et": {c: [] for c in X_cols},
    }

    X_all = df_train[X_cols]
    y_all = df_train["label"]
    groups = df_train["subject"]

    # Helper: compute rank (1..n) with ties handled by average rank; missing -> worst
    def to_rank(vec: pd.Series) -> pd.Series:
        # Larger is better; ranks ascending with 1 best
        # Replace NaNs with -inf so they get worst rank later
        s = vec.copy()
        s = s.fillna(-np.inf)
        order = s.rank(method="average", ascending=False)
        # Any -inf become worst rank
        order[s == -np.inf] = len(s)
        return order

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), start=1):
        Xtr = X_all.iloc[tr_idx]
        ytr = y_all.iloc[tr_idx]
        Xva = X_all.iloc[va_idx]  # used only for potential permutation later

        # Per-fold high-NaN drop
        local_cols = list(X_cols)
        if drop_na_frac > 0.0:
            na_frac = Xtr.isna().mean()
            local_cols = [c for c in local_cols if na_frac[c] <= drop_na_frac]
            Xtr = Xtr[local_cols]
            Xva = Xva[local_cols]

        # Preprocess: impute + scale
        pre = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        Xtr_s = pre.fit_transform(Xtr)

        # MI (univariate) computed on original scale after imputation
        imp_only = SimpleImputer(strategy="median")
        Xtr_imp = imp_only.fit_transform(Xtr)
        try:
            mi_scores = mutual_info_classif(Xtr_imp, ytr, discrete_features=False, random_state=42)
            mi = pd.Series(mi_scores, index=local_cols)
        except Exception:
            mi = pd.Series(np.nan, index=local_cols)
        mi_rank = to_rank(mi)

        # L1 logistic on scaled
        l1_c = float(os.getenv("L1SEL_C", "0.1"))
        l1_it = int(os.getenv("L1SEL_MAX_ITER", "5000"))
        try:
            clf_l1 = LogisticRegression(penalty="l1", solver="saga", C=l1_c, class_weight="balanced", max_iter=l1_it)
            clf_l1.fit(Xtr_s, ytr)
            coefs = np.abs(clf_l1.coef_)
            l1_import = pd.Series(coefs.max(axis=0), index=local_cols)
        except Exception:
            l1_import = pd.Series(np.nan, index=local_cols)
        l1_rank = to_rank(l1_import)

        # ExtraTrees importance
        et_trees = int(os.getenv("ET_TREES", "300"))
        et_depth = int(os.getenv("ET_DEPTH", "12"))
        et = ExtraTreesClassifier(n_estimators=et_trees, max_depth=et_depth, n_jobs=n_jobs, random_state=42, class_weight="balanced_subsample")
        try:
            et.fit(Xtr.fillna(Xtr.median()), ytr)
            et_import = pd.Series(et.feature_importances_, index=local_cols)
        except Exception:
            et_import = pd.Series(np.nan, index=local_cols)
        et_rank = to_rank(et_import)

        # Combined rank: mean of available ranks (drop NaNs)
        combined = pd.concat([mi_rank, l1_rank, et_rank], axis=1)
        combined.columns = ["mi", "l1", "et"]
        comb_rank = combined.mean(axis=1)

        # Store per-method and combined ranks back aligned to full X_cols (fill with worst)
        worst = len(local_cols)
        for c in X_cols:
            if c in comb_rank.index:
                per_fold_ranks[c].append(float(comb_rank[c]))
                per_method_ranks["mi"][c].append(float(mi_rank[c]))
                per_method_ranks["l1"][c].append(float(l1_rank[c]))
                per_method_ranks["et"][c].append(float(et_rank[c]))
            else:
                per_fold_ranks[c].append(float(worst))
                per_method_ranks["mi"][c].append(float(worst))
                per_method_ranks["l1"][c].append(float(worst))
                per_method_ranks["et"][c].append(float(worst))

        if verbose:
            subj_val = groups.iloc[va_idx].iloc[0]
            print(f"Fold {fold_idx:02d} done | val subject={subj_val} | features={len(local_cols)}")

    # Aggregate to consensus
    rows = []
    for c in X_cols:
        comb_r = np.array(per_fold_ranks[c], dtype=float)
        mi_r = np.array(per_method_ranks["mi"][c], dtype=float)
        l1_r = np.array(per_method_ranks["l1"][c], dtype=float)
        et_r = np.array(per_method_ranks["et"][c], dtype=float)
        rows.append({
            "feature": c,
            "rank_comb_mean": float(np.mean(comb_r)),
            "rank_comb_std": float(np.std(comb_r)),
            "rank_mi_mean": float(np.mean(mi_r)),
            "rank_l1_mean": float(np.mean(l1_r)),
            "rank_et_mean": float(np.mean(et_r)),
        })
    rank_df = pd.DataFrame(rows)
    rank_df = rank_df.sort_values("rank_comb_mean", ascending=True).reset_index(drop=True)

    # Stability@K: fraction of folds rank <= K
    k_grid_env = os.getenv("K_GRID", "32,64,96,128")
    k_grid = [int(x.strip()) for x in k_grid_env.split(',') if x.strip()]
    # Prepare per-feature per-K stability
    stab_cols: List[str] = []
    for K in k_grid:
        col = f"stability_top_{K}"
        stab_cols.append(col)
        vals = []
        for c in X_cols:
            comb_r = np.array(per_fold_ranks[c], dtype=float)
            vals.append(float(np.mean(comb_r <= K)))
        rank_df[col] = vals

    # Save ranking
    out_csv = RESULTS_DIR / "consensus_ranking.csv"
    rank_df.to_csv(out_csv, index=False)

    # Emit top-K lists
    lists_dir = RESULTS_DIR / "top_k_lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    for K in k_grid:
        topK = rank_df.head(K)["feature"].tolist()
        (lists_dir / f"top_{K}.json").write_text(json.dumps(topK, indent=2))

    print(f"Wrote consensus ranking -> {out_csv}")
    print(f"Wrote top-K lists -> {lists_dir}")


if __name__ == "__main__":
    main()
