# 05_feature_selection_loso.py
# Simplified LOSO feature selection WITH DEBUG LOGS:
# - Clean: replace inf, drop all-NaN, drop near-constant
# - Rank by F-statistic (ANOVA) only
# - Correlation clustering; pick reps per cluster as a percentage of cluster size
# - Evaluate candidate N via train-side OOF F1 with threshold tuning
# - Choose smallest N within DELTA of best inner F1; evaluate on test subject
# - Aggregate across folds to pick a global N and produce final_feature_order.csv
# - Added debug prints and CSVs for feature counts at each stage

import os
import json
import numpy as np
import pandas as pd

from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, KFold

# ---------------- CONFIG ----------------
processed_dir = "data/output"
features_path = os.path.join(processed_dir, "final_features_ACCextractor_allmods_W30_S10_tau75.parquet")
folds_dir = os.path.join(processed_dir, "folds_W30_S10_tau75")
out_dir = os.path.join(processed_dir, "05_selection_loso")
os.makedirs(out_dir, exist_ok=True)

# Pre-filters
MIN_VAR = 1e-6  # drop near-constant

# Correlation clustering
CORR_THR = 0.60  # edge if |corr| >= CORR_THR
CLUSTER_REP_PCT = 0.25  # reps per cluster = ceil(pct * cluster_size), clamped to [1, CLUSTER_REP_MAX]
CLUSTER_REP_MAX = 7

# Candidate feature counts to test (top-N from F-score-ranked cluster reps)
CANDIDATE_N = [16, 24, 32, 48, 64]
DELTA = 0.01  # pick smallest N within DELTA of best inner F1

# Model / threshold tuning
SEED = 42
N_ESTIMATORS = 300
POS_CLASS_WEIGHT = 4.53  # your sweet spot
THRESH_GRID = np.linspace(0.1, 0.9, 81)

# Debug verbosity
VERBOSE = True
# ----------------------------------------

META = {"subject", "label", "session", "category", "bin_label", "W_sec", "S_sec", "start_idx", "end_idx", "window_id"}

def log(msg: str):
    if VERBOSE:
        print(msg, flush=True)

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    # Prefer sensor prefixes; else numeric minus meta/label
    pref = [c for c in df.columns if c.startswith(("acc_", "bvp_", "eda_", "temp_"))]
    if pref:
        return pref
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num if c not in META and c != "label"]

def impute_median(train: pd.DataFrame, test: pd.DataFrame):
    med = train.median(numeric_only=True)
    return train.fillna(med), test.fillna(med)

def drop_low_variance(X: pd.DataFrame, thr: float):
    var = X.var(axis=0, ddof=0)
    keep = var.index[var > thr].tolist()
    return X[keep], keep

def score_f_anova(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    F, _ = f_classif(X.values, y)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(F, index=X.columns).sort_values(ascending=False)

def correlation_clusters(X: pd.DataFrame, thr: float) -> dict[int, list[str]]:
    cols = X.columns.tolist()
    n = len(cols)
    if n == 0:
        return {}
    corr = X.corr().fillna(0.0).to_numpy()
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= thr:
                adj[i].append(j)
                adj[j].append(i)
    visited = [False] * n
    clusters = {}
    cid = 0
    for i in range(n):
        if not visited[i]:
            stack = [i]
            comp_idx = []
            visited[i] = True
            while stack:
                u = stack.pop()
                comp_idx.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            clusters[cid] = [cols[k] for k in comp_idx]
            cid += 1
    return clusters

def select_cluster_reps(clusters: dict[int, list[str]], scores: pd.Series, pct: float, rep_max: int) -> list[str]:
    reps = []
    for _, members in clusters.items():
        m = len(members)
        k = max(1, min(rep_max, int(np.ceil(pct * m))))
        ranked = sorted(members, key=lambda f: scores.get(f, 0.0), reverse=True)
        reps.extend(ranked[:k])
    # Deduplicate while preserving order
    seen, out = set(), []
    for f in reps:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

def make_tree():
    cw = {0: 1.0, 1: float(POS_CLASS_WEIGHT)} if POS_CLASS_WEIGHT is not None else "balanced"
    return ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=cw,
        n_jobs=-1,
        random_state=SEED
    )

def inner_oof_best_threshold(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
    # Get OOF probabilities on train; pick threshold maximizing F1 (no leakage)
    uniq = np.unique(groups)
    if len(uniq) >= 3:
        splits = GroupKFold(n_splits=min(5, len(uniq))).split(X, y, groups)
    else:
        splits = KFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y)
    oof = np.zeros_like(y, dtype=float)
    seen = np.zeros_like(y, dtype=bool)
    for tr, va in splits:
        clf = make_tree()
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        seen[va] = True
    if not seen.all():
        oof[~seen] = np.mean(y)
    best_f1, best_t = -1.0, 0.5
    for t in THRESH_GRID:
        pred = (oof >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_f1, best_t

def main():
    # Inputs
    if not os.path.exists(features_path):
        raise FileNotFoundError(features_path)
    folds_path = os.path.join(folds_dir, "loso_folds.json")
    if not os.path.exists(folds_path):
        raise FileNotFoundError(folds_path)

    df = pd.read_parquet(features_path)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    feats_all = get_feature_cols(df)
    log(f"[INFO] Total starting features detected: {len(feats_all)}")

    with open(folds_path, "r") as f:
        folds = json.load(f)["folds"]

    per_fold_rows = []
    inner_scores_by_N = {N: [] for N in CANDIDATE_N}
    per_fold_reps = {}  # test_subject -> reps_sorted (by F-score)
    debug_rows = []     # aggregate fold debugging

    for fold in folds:
        test_subject = fold["test_subject"]
        train_subjects = set(fold["train_subjects"])
        fold_dir = os.path.join(out_dir, f"fold_{test_subject}")
        os.makedirs(fold_dir, exist_ok=True)

        log(f"\n[fold {test_subject}] ------------------------")

        # Split
        train = df[df["subject"].isin(train_subjects)].copy()
        test = df[df["subject"] == test_subject].copy()

        # Stage: initial
        n_start = len(feats_all)

        X_tr = train[feats_all].replace([np.inf, -np.inf], np.nan)
        X_te = test[feats_all].replace([np.inf, -np.inf], np.nan)

        # Stage: drop all-NaN (train-based)
        keep_nonan = X_tr.columns[X_tr.notna().any(axis=0)].tolist()
        X_tr = X_tr[keep_nonan]
        X_te = X_te[keep_nonan]
        n_after_nonan = len(keep_nonan)

        # Stage: drop near-constant
        X_tr, kept = drop_low_variance(X_tr, MIN_VAR)
        X_te = X_te[kept]
        n_after_var = len(kept)

        # Impute
        X_tr, X_te = impute_median(X_tr, X_te)
        y_tr = train["label"].astype(int).to_numpy()
        y_te = test["label"].astype(int).to_numpy()
        groups_tr = train["subject"].to_numpy()

        # F-score ranking
        f_scores = score_f_anova(X_tr, y_tr)  # sorted desc
        pd.DataFrame({"feature": f_scores.index, "F": f_scores.values}).to_csv(
            os.path.join(fold_dir, "ranking_fscore.csv"), index=False
        )

        # Correlation clustering
        clusters = correlation_clusters(X_tr, CORR_THR)
        with open(os.path.join(fold_dir, "clusters.json"), "w") as f:
            json.dump(clusters, f, indent=2)

        # Cluster stats
        cluster_sizes = [(cid, len(members)) for cid, members in clusters.items()]
        if cluster_sizes:
            sizes_only = [s for _, s in cluster_sizes]
            size_min = int(np.min(sizes_only))
            size_med = float(np.median(sizes_only))
            size_max = int(np.max(sizes_only))
            n_clusters = len(cluster_sizes)
        else:
            size_min = size_med = size_max = 0
            n_clusters = 0

        pd.DataFrame(cluster_sizes, columns=["cluster_id", "size"]).to_csv(
            os.path.join(fold_dir, "cluster_sizes.csv"), index=False
        )

        # Representatives per cluster
        reps = select_cluster_reps(clusters, f_scores, CLUSTER_REP_PCT, CLUSTER_REP_MAX)
        reps_sorted = sorted(reps, key=lambda f: f_scores.get(f, 0.0), reverse=True)
        per_fold_reps[test_subject] = reps_sorted
        pd.DataFrame({"feature": reps_sorted}).to_csv(
            os.path.join(fold_dir, "cluster_representatives_ordered.csv"), index=False
        )

        # reps by cluster (debug)
        reps_by_cluster_rows = []
        reps_set = set(reps_sorted)
        for cid, members in clusters.items():
            sel_cnt = len([m for m in members if m in reps_set])
            reps_by_cluster_rows.append({"cluster_id": cid, "size": len(members), "reps_selected": sel_cnt})
        if reps_by_cluster_rows:
            pd.DataFrame(reps_by_cluster_rows).to_csv(
                os.path.join(fold_dir, "reps_by_cluster.csv"), index=False
            )

        # Inner evaluation over candidate Ns (limited by reps count)
        Ns = [N for N in CANDIDATE_N if N <= len(reps_sorted)]
        if not Ns:
            Ns = [min(16, max(1, len(reps_sorted)))]
        inner_rows = []
        best_inner_f1 = -1.0
        for N in Ns:
            featsN = reps_sorted[:N]
            f1_inner, thr = inner_oof_best_threshold(X_tr[featsN], y_tr, groups_tr)
            inner_rows.append({"N": int(N), "f1": float(f1_inner), "threshold": float(thr)})
            inner_scores_by_N.setdefault(N, []).append(f1_inner)
            best_inner_f1 = max(best_inner_f1, f1_inner)

        pd.DataFrame(inner_rows).to_csv(os.path.join(fold_dir, "inner_cv_results.csv"), index=False)

        # Choose smallest N within DELTA of best
        best = max(inner_rows, key=lambda r: r["f1"])
        candidates = [r for r in inner_rows if (best["f1"] - r["f1"]) <= DELTA]
        chosen = sorted(candidates, key=lambda r: (r["N"], -r["f1"]))[0]
        N_chosen = chosen["N"]
        thr_chosen = chosen["threshold"]
        feats_final = reps_sorted[:N_chosen]

        # Save selected features for this fold
        with open(os.path.join(fold_dir, "selected_features.txt"), "w") as f:
            for name in feats_final:
                f.write(name + "\n")

        # Train final model on full train; evaluate on test
        clf = make_tree()
        clf.fit(X_tr[feats_final], y_tr)
        p = clf.predict_proba(X_te[feats_final])[:, 1]
        y_pred = (p >= thr_chosen).astype(int)

        f1 = f1_score(y_te, y_pred, zero_division=0)
        bal = balanced_accuracy_score(y_te, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()

        per_fold_rows.append({
            "test_subject": test_subject,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "N_chosen": int(N_chosen),
            "threshold": float(thr_chosen),
            "f1": float(f1),
            "balanced_accuracy": float(bal),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })

        # Debug summary row
        debug_rows.append({
            "test_subject": test_subject,
            "n_feat_start": int(n_start),
            "n_feat_after_nonan": int(n_after_nonan),
            "n_feat_after_variance": int(n_after_var),
            "n_clusters": int(n_clusters),
            "cluster_size_min": int(size_min),
            "cluster_size_median": float(size_med),
            "cluster_size_max": int(size_max),
            "n_reps_total": int(len(reps_sorted)),
            "Ns_tried": ";".join(str(x) for x in Ns),
            "best_inner_f1": float(best_inner_f1),
            "chosen_N": int(N_chosen),
            "chosen_threshold": float(thr_chosen),
        })

        # Console debug print
        log(
            f"[fold {test_subject}] start={n_start}, nonan={n_after_nonan}, var={n_after_var}, "
            f"clusters={n_clusters} (min/med/max={size_min}/{size_med:.1f}/{size_max}), "
            f"reps={len(reps_sorted)}, Ns={Ns}, best_inner_f1={best_inner_f1:.3f}, "
            f"chosen_N={N_chosen}, thr={thr_chosen:.2f}, "
            f"test_f1={f1:.3f}, bal_acc={bal:.3f}"
        )

    # Save per-fold metrics
    df_folds = pd.DataFrame(per_fold_rows)
    df_folds.to_csv(os.path.join(out_dir, "fold_metrics.csv"), index=False)
    df_folds.describe().to_csv(os.path.join(out_dir, "aggregate_summary.csv"))

    # Save per-fold debug summary
    if debug_rows:
        pd.DataFrame(debug_rows).to_csv(os.path.join(out_dir, "per_fold_debug.csv"), index=False)

    # Aggregate candidate N performance across folds
    agg_rows = []
    for N, vals in inner_scores_by_N.items():
        if len(vals) == 0:
            continue
        vals = np.array(vals, dtype=float)
        agg_rows.append({
            "N": int(N),
            "mean_inner_f1": float(np.mean(vals)),
            "std_inner_f1": float(np.std(vals, ddof=0)),
            "n_folds": int(len(vals)),
        })
    df_aggN = pd.DataFrame(agg_rows).sort_values("N")
    df_aggN.to_csv(os.path.join(out_dir, "candidate_N_summary.csv"), index=False)

    # Choose global N (smallest within DELTA of best mean inner F1)
    if not df_aggN.empty:
        best_row = df_aggN.iloc[df_aggN["mean_inner_f1"].argmax()]
        best_mean = best_row["mean_inner_f1"]
        within = df_aggN[df_aggN["mean_inner_f1"] >= (best_mean - DELTA)]
        global_N = int(within.sort_values(["N", "mean_inner_f1"], ascending=[True, False]).iloc[0]["N"])
    else:
        global_N = CANDIDATE_N[0]

    # Build final global feature order by frequency of top global_N per fold
    freq = {}
    for subj, reps_sorted in per_fold_reps.items():
        topN = reps_sorted[:global_N]
        for ftr in topN:
            freq[ftr] = freq.get(ftr, 0) + 1
    freq_df = pd.DataFrame(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])), columns=["feature", "frequency"])
    freq_df.to_csv(os.path.join(out_dir, "aggregated_feature_frequencies.csv"), index=False)
    freq_df.to_csv(os.path.join(out_dir, "final_feature_order.csv"), index=False)

    # Save selection summary
    summary = {
        "CANDIDATE_N": CANDIDATE_N,
        "DELTA": DELTA,
        "CORR_THR": CORR_THR,
        "CLUSTER_REP_PCT": CLUSTER_REP_PCT,
        "CLUSTER_REP_MAX": CLUSTER_REP_MAX,
        "MIN_VAR": MIN_VAR,
        "POS_CLASS_WEIGHT": POS_CLASS_WEIGHT,
        "global_N": int(global_N),
        "notes": "Simplified: F-stat only; clustering for redundancy; proportional reps; OOF threshold tuning; with debug counts.",
    }
    with open(os.path.join(out_dir, "selection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n[INFO] Saved selection outputs to: {out_dir}")
    log(f"[INFO] Recommended global N: {global_N}")

if __name__ == "__main__":
    main()