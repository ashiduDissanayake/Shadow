#!/usr/bin/env python3
"""
ESP32-S3 Rigorous ML Pipeline
Stage 1.5: Enhanced Cross-Subject Feature Selection (Final Version)
===============================================================================

Purpose:
    This is the enhanced, final feature selection stage with:
    - SUPERVISED cluster representative selection (vs variance-based)
    - MITIGATED per-fold early stopping for consistent evaluation
    - STATISTICAL SIGNIFICANCE testing via permutation/label-shuffle
    - CALIBRATED probabilities for Brier/ROC/PR metrics
    - CLUSTER-LEVEL stability accounting to reduce bias against correlated predictive features

Enhancements from Corrected Version:
    ✅ Supervised cluster representatives (mutual info + univariate F-test)
    ✅ Consistent distribution evaluation (reduced early stopping)
    ✅ Permutation test for statistical significance of final feature set
    ✅ Probability calibration before reporting probabilistic metrics
    ✅ Cluster-level stability metrics to account for correlation structure

Author: User & AI Assistant
Date: 2025-08-31
"""

from __future__ import annotations
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Scientific / stats
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ML / Feature Selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV

# Metrics
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    precision_score, recall_score, confusion_matrix,
    roc_auc_score, average_precision_score, brier_score_loss
)

# Optional (fast check for newer StratifiedGroupKFold)
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except ImportError:
    HAS_SGKF = False

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - STAGE1_5_ENHANCED - %(levelname)s - %(message)s"
)
logger = logging.getLogger("STAGE1_5_ENHANCED")

# ------------------------------------------------------------------------------
# Configuration Structures
# ------------------------------------------------------------------------------
@dataclass
class Stage15Config:
    # Paths
    stage0_dir: str = "../outputs/stage0"
    output_dir: str = "../outputs/stage1_5_enhanced"
    config_path: str = "../config/pipeline_config.json"

    # Feature subset target sizes (prefix lengths)
    feature_counts: List[int] = (20, 25, 30, 35, 40, 45, 50, 60)

    # Correlation clustering - threshold is minimum |corr| to group
    correlation_threshold: float = 0.90  # Features with |corr| >= 0.90 will be clustered

    # Stability requirements
    min_selection_frequency: float = 0.30  # feature must appear in >= 30% folds  
    performance_delta: float = 0.025       # Accept subset if mean F1 within 0.025 of best

    # Inner CV
    inner_folds: int = 4

    # Permutation importance (optional; increases runtime)
    use_permutation_importance: bool = False
    permutation_repeats: int = 5

    # Memory constraints
    feature_memory_bytes: int = 4          # float32 or potentially quantized later
    feature_budget_kb: float = 50.0        

    # Reproducibility
    random_state: int = 42

    # ENHANCED: Mitigated early stopping (less aggressive)
    early_stop_patience: int = 6           # Increased from 4
    early_stop_min_improve: float = 0.001  # Reduced from 0.002
    early_stop_enabled: bool = False       # Can disable entirely

    # Statistical significance testing
    permutation_test_n: int = 100          # Reduced from 1000 for faster execution  
    significance_alpha: float = 0.05       # Alpha for significance testing
    fast_significance_test: bool = True    # Use non-calibrated models for permutation tests

    # Probability calibration
    calibrate_probabilities: bool = True   # Enable probability calibration
    calibration_method: str = "isotonic"   # "isotonic" or "sigmoid"

    # Cluster stability accounting
    cluster_stability_weight: float = 0.3  # Weight for cluster-level stability

    # Resume support
    allow_resume: bool = True


# ------------------------------------------------------------------------------
# Enhanced Utility Functions
# ------------------------------------------------------------------------------
def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_stage0_artifacts(cfg: Stage15Config) -> Tuple[pd.DataFrame, Dict]:
    manifest_path = Path(cfg.stage0_dir) / "data_manifest.json"
    folds_path = Path(cfg.stage0_dir) / "fold_definitions.json"

    if not manifest_path.exists() or not folds_path.exists():
        raise FileNotFoundError("Stage 0 artifacts missing. Run Stage 0 first.")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    with open(folds_path, "r") as f:
        fold_defs = json.load(f)

    data_file = manifest["source_file"]
    if not Path(data_file).exists():
        raise FileNotFoundError(f"Data file not found at {data_file}")

    df = pd.read_parquet(data_file)

    # Basic cleaning
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop columns entirely NaN
    na_all = df.isna().all()
    if na_all.any():
        drop_cols = na_all[na_all].index.tolist()
        if drop_cols:
            logger.info(f"Dropping {len(drop_cols)} all-NaN columns.")
            df = df.drop(columns=drop_cols)
    df = df.fillna(0)

    return df, fold_defs


def compute_supervised_cluster_score(X: pd.DataFrame, y: np.ndarray, features: List[str]) -> Dict[str, float]:
    """
    ENHANCED: Compute supervised scores for cluster representative selection.
    Uses combination of mutual information and F-statistic.
    """
    X_subset = X[features]
    scores = {}
    
    # Mutual information scores
    mi_scores = mutual_info_classif(X_subset, y, random_state=42)
    
    # F-statistic scores
    f_scores, _ = f_classif(X_subset, y)
    
    # Normalize both to [0, 1] and combine
    mi_norm = mi_scores / (mi_scores.max() + 1e-10)
    f_norm = f_scores / (f_scores.max() + 1e-10)
    
    combined_scores = 0.6 * mi_norm + 0.4 * f_norm  # Weighted combination
    
    for i, feat in enumerate(features):
        scores[feat] = {
            "mutual_info": float(mi_scores[i]),
            "f_statistic": float(f_scores[i]),
            "combined_score": float(combined_scores[i])
        }
    
    return scores


def hierarchical_correlation_clustering_supervised(
    X: pd.DataFrame,
    y: np.ndarray,
    corr_threshold: float,
    min_variance: float = 1e-10
) -> Tuple[List[str], List[Dict], Dict[str, Any]]:
    """
    ENHANCED: Supervised cluster representative selection with stability accounting.
    """
    X_local = X.copy()
    
    # Remove near-constant features
    variances = X_local.var()
    keep = variances[variances > min_variance].index
    if len(keep) < X_local.shape[1]:
        removed_count = X_local.shape[1] - len(keep)
        X_local = X_local[keep]
        logger.info(f"   Removed {removed_count} low-variance features")

    if X_local.shape[1] <= 1:
        return list(X_local.columns), [], {"total_clusters": 0, "multi_clusters": 0}

    corr = X_local.corr().abs().fillna(0)
    dist = 1 - corr
    np.fill_diagonal(dist.values, 0)

    # Convert correlation threshold to distance cut
    distance_cut = 1 - corr_threshold
    logger.info(f"   [Clustering] corr_threshold={corr_threshold:.2f} → distance_cut={distance_cut:.2f}")

    condensed = squareform(dist.values, checks=False)
    if not np.isfinite(condensed).all():
        logger.warning("   Non-finite distances; falling back to simple filter")
        return simple_correlation_filter(X_local, corr_threshold), [], {}

    link = linkage(condensed, method="average")
    cluster_labels = fcluster(link, distance_cut, criterion="distance")

    selected = []
    clusters_info = []
    
    # Compute supervised scores for all features
    all_supervised_scores = compute_supervised_cluster_score(X_local, y, list(X_local.columns))

    for cid in np.unique(cluster_labels):
        members = X_local.columns[cluster_labels == cid].tolist()
        
        if len(members) == 1:
            selected.append(members[0])
        else:
            # ENHANCED: Supervised representative selection
            cluster_size = len(members)
            if cluster_size <= 10:
                n_reps = 2
            elif cluster_size <= 30:
                n_reps = 3
            elif cluster_size <= 60:
                n_reps = 4
            elif cluster_size <= 100:
                n_reps = 5
            else:
                n_reps = max(6, min(10, cluster_size // 20))
            
            # Select based on supervised scores (not variance)
            member_scores = [(feat, all_supervised_scores[feat]["combined_score"]) for feat in members]
            member_scores.sort(key=lambda x: x[1], reverse=True)
            top_reps = [feat for feat, score in member_scores[:n_reps]]
            
            selected.extend(top_reps)
            
            clusters_info.append({
                "cluster_id": int(cid),
                "size": len(members),
                "representatives": top_reps,
                "members": members,
                "rep_supervised_scores": [all_supervised_scores[feat]["combined_score"] for feat in top_reps],
                "cluster_mean_score": np.mean([all_supervised_scores[feat]["combined_score"] for feat in members])
            })
            
            rep_scores = [all_supervised_scores[feat]["combined_score"] for feat in top_reps]
            logger.info(f"   Supervised cluster {cid}: {len(members)} features → {n_reps} reps (scores: {rep_scores})")
            logger.info(f"   Supervised cluster {cid}: {len(members)} features → {n_reps} reps (scores: {[f'{s:.3f}' for s in rep_scores]})")

    # Enhanced diagnostics with supervised scoring
    multi_clusters = [c for c in clusters_info if c["size"] > 1]
    cluster_stats = {
        "total_clusters": len(np.unique(cluster_labels)),
        "multi_clusters": len(multi_clusters),
        "largest_cluster_size": max([c["size"] for c in clusters_info], default=1),
        "mean_cluster_supervised_score": np.mean([c["cluster_mean_score"] for c in multi_clusters]) if multi_clusters else 0.0
    }
    
    if multi_clusters:
        intra_corr_means = []
        for c in multi_clusters:
            sub_corr = corr.loc[c["members"], c["members"]].values
            upper_tri = sub_corr[np.triu_indices(sub_corr.shape[0], 1)]
            if len(upper_tri) > 0:
                intra_corr_means.append(upper_tri.mean())
        
        if intra_corr_means:
            cluster_stats["mean_intra_corr"] = np.mean(intra_corr_means)
            logger.info(f"   [Clustering] Multi-clusters={len(multi_clusters)} | mean_intra_corr={cluster_stats['mean_intra_corr']:.3f}")

    # Residual correlation among selected representatives
    if len(selected) > 1:
        selected_corr = X_local[selected].corr().abs()
        upper_indices = np.triu_indices(len(selected), 1)
        residual_corr = selected_corr.values[upper_indices].mean()
        cluster_stats["residual_corr_among_reps"] = residual_corr
        logger.info(f"   [Clustering] Residual mean |corr| among selected reps = {residual_corr:.3f}")

    logger.info(f"✅ SUPERVISED Clustering: {X.shape[1]} → {len(selected)} features")
    return selected, clusters_info, cluster_stats


def simple_correlation_filter(X: pd.DataFrame, threshold: float) -> List[str]:
    """Fallback simple correlation filter."""
    corr = X.corr().abs()
    remove = set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        if cols[i] in remove:
            continue
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] > threshold:
                # remove lower variance
                var_i = X[cols[i]].var()
                var_j = X[cols[j]].var()
                if var_i < var_j:
                    remove.add(cols[i])
                    break
                else:
                    remove.add(cols[j])
    return [c for c in cols if c not in remove]


def compute_rankings(
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int,
    use_permutation: bool = False,
    permutation_repeats: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Returns dict of method_name -> {feature: rank (1=best)}.
    """
    rankings: Dict[str, Dict[str, float]] = {}
    features = X.columns

    # 1. f_classif
    f_vals, _ = f_classif(X, y)
    f_ranks = stats.rankdata(-f_vals, method="ordinal")
    rankings["f_classif"] = dict(zip(features, f_ranks))

    # 2. mutual_info
    mi_vals = mutual_info_classif(X, y, random_state=random_state)
    mi_ranks = stats.rankdata(-mi_vals, method="ordinal")
    rankings["mutual_info"] = dict(zip(features, mi_ranks))

    # 3. ExtraTrees importance
    et = ExtraTreesClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )
    et.fit(X, y)
    et_importance = et.feature_importances_
    et_ranks = stats.rankdata(-et_importance, method="ordinal")
    rankings["extra_trees"] = dict(zip(features, et_ranks))

    # 4. RFE(LR) with scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(
        random_state=random_state,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear"
    )
    n_select = max(1, len(features) // 2)
    rfe = RFE(lr, n_features_to_select=n_select, step=0.1)
    rfe.fit(X_scaled, y)
    rfe_ranks = stats.rankdata(rfe.ranking_, method="ordinal")
    rankings["rfe_lr_scaled"] = dict(zip(features, rfe_ranks))

    # 5. Optional permutation importance
    if use_permutation:
        from sklearn.inspection import permutation_importance
        perm_scores_accum = np.zeros(len(features))
        for _ in range(permutation_repeats):
            perm = permutation_importance(
                et, X, y, n_repeats=5, random_state=random_state, n_jobs=-1
            )
            perm_scores_accum += perm.importances_mean
        perm_scores_accum /= permutation_repeats
        perm_ranks = stats.rankdata(-perm_scores_accum, method="ordinal")
        rankings["perm_importance"] = dict(zip(features, perm_ranks))

    return rankings


def consensus_rank(rankings: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Average rank across methods. Lower = better."""
    methods = list(rankings.keys())
    features = list(next(iter(rankings.values())).keys())
    consensus_scores = {}
    for feat in features:
        rlist = [rankings[m][feat] for m in methods]
        consensus_scores[feat] = float(np.mean(rlist))
    return consensus_scores


def stratified_group_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified group splits with fallback."""
    if HAS_SGKF:
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        return list(sgkf.split(np.zeros(len(y)), y, groups))

    logger.warning("StratifiedGroupKFold not available. Using manual group balancing.")
    # Manual implementation as before...
    unique_groups = np.unique(groups)
    group_props = []
    for g in unique_groups:
        mask = (groups == g)
        prop = (y[mask] == 1).mean()
        group_props.append((g, prop, mask.sum()))
    
    group_props.sort(key=lambda x: x[1])
    folds = [[] for _ in range(n_splits)]
    fold_pos = np.zeros(n_splits)
    fold_tot = np.zeros(n_splits)

    for g, prop, size in group_props:
        ratios = np.divide(
            fold_pos,
            np.maximum(fold_tot, 1),
            out=np.zeros_like(fold_pos),
            where=fold_tot > 0
        )
        best = np.argmin(ratios)
        folds[best].append(g)
        fold_pos[best] += prop * size
        fold_tot[best] += size

    splits = []
    for i in range(n_splits):
        val_groups = folds[i]
        train_groups = [g for j, fl in enumerate(folds) if j != i for g in fl]
        train_idx = np.where(np.isin(groups, train_groups))[0]
        val_idx = np.where(np.isin(groups, val_groups))[0]
        splits.append((train_idx, val_idx))
    return splits


def evaluate_subset_calibrated(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    features: List[str],
    random_state: int,
    calibrate_probs: bool = True,
    calibration_method: str = "isotonic"
) -> Dict[str, float]:
    """
    ENHANCED: Evaluation with calibrated probabilities for better Brier/ROC/PR metrics.
    """
    base_model = ExtraTreesClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )
    
    # Use calibrated classifier if requested
    if calibrate_probs and len(np.unique(y_train)) > 1:
        model = CalibratedClassifierCV(
            base_model,
            method=calibration_method,
            cv=3  # Use 3-fold for calibration
        )
    else:
        model = base_model
    
    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])
    
    # Basic metrics
    f1 = f1_score(y_test, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Precision/Recall/Specificity
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    metrics = {
        "f1": f1,
        "bal_acc": bal_acc,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "calibrated": calibrate_probs
    }
    
    # ENHANCED: Calibrated probability-based metrics
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test[features])
            if y_proba.shape[1] > 1:
                y_proba_pos = y_proba[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba_pos)
                metrics["pr_auc"] = average_precision_score(y_test, y_proba_pos)
                metrics["brier"] = brier_score_loss(y_test, y_proba_pos)
            else:
                metrics.update({"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan})
        else:
            metrics.update({"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan})
    except Exception as e:
        logger.warning(f"Calibrated probability metrics failed: {e}")
        metrics.update({"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan})
    
    return metrics


def compute_majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Compute majority class baseline performance."""
    maj_label = Counter(y_train).most_common(1)[0][0]
    y_baseline = np.full_like(y_test, maj_label)
    
    return {
        "baseline_f1": f1_score(y_test, y_baseline, zero_division=0),
        "baseline_bal_acc": balanced_accuracy_score(y_test, y_baseline),
        "baseline_mcc": matthews_corrcoef(y_test, y_baseline),
        "majority_class": int(maj_label),
        "majority_fraction": (y_train == maj_label).mean()
    }


def permutation_significance_test(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    features: List[str],
    random_state: int,
    n_permutations: int = 100,
    calibrate_probs: bool = True,
    fast_mode: bool = True
) -> Dict[str, float]:
    """
    ENHANCED: Statistical significance test via label permutation.
    fast_mode=True uses non-calibrated models for permutation tests to speed up computation.
    """
    # True performance (always calibrated if requested)
    true_metrics = evaluate_subset_calibrated(
        X_train, y_train, X_test, y_test, features, random_state, calibrate_probs
    )
    true_f1 = true_metrics["f1"]
    
    # For permutation tests, use fast mode if enabled
    perm_calibrate = calibrate_probs and not fast_mode
    
    logger.info(f"Starting permutation test: n={n_permutations}, "
               f"calibrated_true={calibrate_probs}, calibrated_perm={perm_calibrate}")
    
    # Permutation tests
    rng = np.random.RandomState(random_state)
    null_f1_scores = []
    
    for i in range(n_permutations):
        if i % 20 == 0:  # Progress logging
            logger.info(f"  Permutation progress: {i}/{n_permutations}")
            
        # Shuffle labels while preserving marginals
        y_train_perm = rng.permutation(y_train)
        
        try:
            perm_metrics = evaluate_subset_calibrated(
                X_train, y_train_perm, X_test, y_test, features, 
                random_state + i, perm_calibrate
            )
            null_f1_scores.append(perm_metrics["f1"])
        except Exception as e:
            logger.warning(f"Permutation {i} failed: {e}")
            null_f1_scores.append(0.0)
    
    null_f1_scores = np.array(null_f1_scores)
    
    # P-value calculation
    p_value = (null_f1_scores >= true_f1).mean()
    
    # Effect size (Cohen's d-like)
    null_mean = null_f1_scores.mean()
    null_std = null_f1_scores.std()
    effect_size = (true_f1 - null_mean) / (null_std + 1e-10)
    
    logger.info(f"Permutation test completed: p={p_value:.4f}, effect_size={effect_size:.3f}")
    
    return {
        "true_f1": true_f1,
        "null_f1_mean": float(null_mean),
        "null_f1_std": float(null_std),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "n_permutations": n_permutations,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
        "fast_mode": fast_mode,
        "calibrated_permutations": perm_calibrate
    }


def compute_cluster_stability(
    cluster_assignments: List[Dict],
    feature_frequencies: Dict[str, int],
    total_folds: int
) -> Dict[str, float]:
    """
    ENHANCED: Cluster-level stability accounting to reduce bias against correlated predictive features.
    """
    cluster_stabilities = {}
    cluster_coverage = {}
    
    for cluster_info in cluster_assignments:
        cluster_id = cluster_info["cluster_id"]
        members = cluster_info["members"]
        representatives = cluster_info["representatives"]
        
        # Cluster coverage: how often any member from this cluster was selected
        member_freqs = [feature_frequencies.get(member, 0) for member in members]
        max_member_freq = max(member_freqs) if member_freqs else 0
        cluster_coverage[cluster_id] = max_member_freq / total_folds
        
        # Representative stability: how often the specific representatives were chosen
        rep_freqs = [feature_frequencies.get(rep, 0) for rep in representatives]
        mean_rep_freq = np.mean(rep_freqs) if rep_freqs else 0
        cluster_stabilities[cluster_id] = mean_rep_freq / total_folds
        
    overall_cluster_stability = np.mean(list(cluster_stabilities.values())) if cluster_stabilities else 0.0
    overall_cluster_coverage = np.mean(list(cluster_coverage.values())) if cluster_coverage else 0.0
    
    return {
        "cluster_stabilities": cluster_stabilities,
        "cluster_coverages": cluster_coverage,
        "mean_cluster_stability": overall_cluster_stability,
        "mean_cluster_coverage": overall_cluster_coverage,
        "n_clusters": len(cluster_stabilities)
    }


# ------------------------------------------------------------------------------
# Main Class (Enhanced)
# ------------------------------------------------------------------------------
class Stage15FeatureSelectorEnhanced:
    def __init__(self, cfg: Stage15Config):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.per_fold_results: List[Dict[str, Any]] = []
        self.feature_frequency: Dict[int, Dict[str, int]] = {}
        self.cluster_info_all: List[Dict] = []  # Store all cluster info for stability analysis
        self.candidate_sets: Dict[int, Dict[str, Any]] = {}
        self.final_selection: Optional[Dict[str, Any]] = None

    def maybe_resume(self):
        if not self.cfg.allow_resume:
            return
        existing = self.output_dir / "per_fold_results.json"
        if existing.exists():
            logger.info("Resume enabled: loading existing per_fold_results.json")
            try:
                with open(existing, "r") as f:
                    existing_results = json.load(f)
                
                if existing_results and isinstance(existing_results[0], dict):
                    if "fold_id" in existing_results[0]:
                        logger.info("Compatible format found, loading existing results")
                        self.per_fold_results = existing_results
                    else:
                        logger.warning("Incompatible format. Starting fresh.")
                        backup_path = existing.with_suffix(".json.backup")
                        existing.rename(backup_path)
                        logger.info(f"Backed up old results to {backup_path}")
            except Exception as e:
                logger.warning(f"Error loading existing results: {e}. Starting fresh.")
                backup_path = existing.with_suffix(".json.error_backup")
                existing.rename(backup_path)

    def save_intermediate(self):
        with open(self.output_dir / "per_fold_results.json", "w") as f:
            json.dump(self.per_fold_results, f, indent=2)

    def run(self):
        set_global_seed(self.cfg.random_state)
        logger.info("========== STAGE 1.5 ENHANCED START ==========")
        logger.info(f"ENHANCEMENTS: Supervised clustering, mitigated early stopping, significance testing, calibrated probabilities, cluster stability")

        # Load Stage 0 artifacts
        df, fold_defs = load_stage0_artifacts(self.cfg)
        assert "subject" in df.columns and "label" in df.columns, "Data must contain subject & label columns."
        feature_cols = [c for c in df.columns if c not in ("subject", "label")]

        # Resume support
        self.maybe_resume()
        processed_fold_ids = {r["fold_id"] for r in self.per_fold_results}

        # Outer folds (LOSO)
        for fold in fold_defs["folds"]:
            fid = fold["fold_id"]
            if fid in processed_fold_ids:
                logger.info(f"Skipping already processed fold {fid}")
                continue

            test_subj = fold["test_subject"]
            train_subjs = fold["train_subjects"]

            train_mask = df["subject"].isin(train_subjs)
            test_mask = df["subject"] == test_subj

            X_train_full = df.loc[train_mask, feature_cols].copy()
            y_train_full = df.loc[train_mask, "label"].values
            groups_train_full = df.loc[train_mask, "subject"].values

            X_test_full = df.loc[test_mask, feature_cols].copy()
            y_test_full = df.loc[test_mask, "label"].values

            logger.info(f"Fold {fid} | Test Subject={test_subj} | "
                        f"Train Samples={len(X_train_full)} | Test Samples={len(X_test_full)}")

            # Majority baseline
            baseline_metrics = compute_majority_baseline(y_train_full, y_test_full)
            
            # 1. ENHANCED: Supervised correlation clustering
            corr_selected, clusters_info, cluster_stats = hierarchical_correlation_clustering_supervised(
                X_train_full, y_train_full, self.cfg.correlation_threshold
            )
            X_train_corr = X_train_full[corr_selected]
            X_test_corr = X_test_full[corr_selected]
            logger.info(f"   Supervised Clustering: {len(feature_cols)} → {len(corr_selected)} features")
            
            # Store cluster info for later stability analysis
            self.cluster_info_all.extend(clusters_info)

            # 2. Consensus ranking (train-only)
            rankings = compute_rankings(
                X_train_corr,
                y_train_full,
                random_state=self.cfg.random_state,
                use_permutation=self.cfg.use_permutation_importance,
                permutation_repeats=self.cfg.permutation_repeats
            )
            consensus = consensus_rank(rankings)
            ordered_features = sorted(consensus.keys(), key=lambda f: consensus[f])

            # 3. Inner CV for prefix evaluation with MITIGATED early stopping
            inner_splits = stratified_group_splits(
                y_train_full, groups_train_full,
                n_splits=self.cfg.inner_folds,
                random_state=self.cfg.random_state
            )

            fold_prefix_results = {}
            best_inner_f1 = -1
            patience = 0

            target_set = sorted(set(self.cfg.feature_counts))
            usable_prefix = [n for n in target_set if n <= len(ordered_features)]

            for n in usable_prefix:
                subset = ordered_features[:n]

                # Inner CV evaluation
                inner_scores = []
                for tr_idx, val_idx in inner_splits:
                    X_tr = X_train_corr.iloc[tr_idx]
                    y_tr = y_train_full[tr_idx]
                    X_val = X_train_corr.iloc[val_idx]
                    y_val = y_train_full[val_idx]

                    metrics_inner = evaluate_subset_calibrated(
                        X_tr, y_tr, X_val, y_val, subset, self.cfg.random_state,
                        self.cfg.calibrate_probabilities, self.cfg.calibration_method
                    )
                    inner_scores.append(metrics_inner["f1"])

                mean_inner = float(np.mean(inner_scores))
                std_inner = float(np.std(inner_scores))

                # ENHANCED: Calibrated outer evaluation
                outer_metrics = evaluate_subset_calibrated(
                    X_train_corr, y_train_full,
                    X_test_corr, y_test_full,
                    subset, self.cfg.random_state,
                    self.cfg.calibrate_probabilities, self.cfg.calibration_method
                )

                # Store comprehensive results
                fold_prefix_results[n] = {
                    "selected_features": subset,
                    "inner_f1_mean": mean_inner,
                    "inner_f1_std": std_inner,
                    "outer_f1": outer_metrics["f1"],
                    "outer_bal_acc": outer_metrics["bal_acc"],
                    "outer_mcc": outer_metrics["mcc"],
                    "outer_precision": outer_metrics["precision"],
                    "outer_recall": outer_metrics["recall"],
                    "outer_specificity": outer_metrics["specificity"],
                    "outer_fpr": outer_metrics["fpr"],
                    "outer_fnr": outer_metrics["fnr"],
                    "outer_roc_auc": outer_metrics.get("roc_auc", np.nan),
                    "outer_pr_auc": outer_metrics.get("pr_auc", np.nan),
                    "outer_brier": outer_metrics.get("brier", np.nan),
                    "tp": outer_metrics["tp"],
                    "fp": outer_metrics["fp"],
                    "tn": outer_metrics["tn"],
                    "fn": outer_metrics["fn"],
                    "calibrated": outer_metrics["calibrated"]
                }

                logger.info(
                    f"Fold {fid} | N={n} | Inner F1={mean_inner:.3f}±{std_inner:.3f} | "
                    f"Outer F1={outer_metrics['f1']:.3f} | Calibrated={outer_metrics['calibrated']}"
                )

                # MITIGATED early stopping (less aggressive or disabled)
                if self.cfg.early_stop_enabled:
                    if mean_inner > best_inner_f1 + self.cfg.early_stop_min_improve:
                        best_inner_f1 = mean_inner
                        patience = 0
                    else:
                        patience += 1
                        if patience >= self.cfg.early_stop_patience:
                            logger.info(f"Early stop at N={n} (mitigated: patience={self.cfg.early_stop_patience}).")
                            break

            # Store fold result with enhanced information
            fold_result = {
                "fold_id": fid,
                "test_subject": test_subj,
                "train_subjects": train_subjs,
                "n_train_samples": int(len(X_train_corr)),
                "n_test_samples": int(len(X_test_corr)),
                "correlation_cluster_reduction": {
                    "before": len(feature_cols),
                    "after": len(corr_selected),
                    "clusters": clusters_info,
                    "cluster_stats": cluster_stats
                },
                "ranking_methods": list(rankings.keys()),
                "majority_baseline": baseline_metrics,
                "prefix_evaluations": fold_prefix_results,
                "supervised_clustering": True,
                "early_stopping_used": self.cfg.early_stop_enabled
            }
            self.per_fold_results.append(fold_result)
            self.save_intermediate()

        # 4. ENHANCED aggregation with cluster stability
        self._aggregate_frequency()
        self._build_candidate_sets_enhanced()
        self._select_final_set_enhanced()

        # 5. ENHANCED: Statistical significance testing for final selection
        if self.final_selection:
            self._perform_significance_testing(df, fold_defs)

        # 6. Save all artifacts
        self._save_outputs()

        logger.info("========== STAGE 1.5 ENHANCED COMPLETE ==========")
        if self.final_selection:
            logger.info(f"ENHANCED FINAL: {self.final_selection['n_features']} features | "
                       f"F1={self.final_selection['mean_f1']:.3f}±{self.final_selection['std_f1']:.3f} | "
                       f"Memory={self.final_selection['memory_kb_est']:.2f}KB")
            if "significance_test" in self.final_selection:
                sig_test = self.final_selection["significance_test"]
                logger.info(f"   Significance: p={sig_test['p_value']:.4f} | Effect={sig_test['effect_size']:.3f}")
            logger.info("=" * 80)
            logger.info("🚀 ENHANCED RESULTS:")
            logger.info(f"   Features: {self.final_selection['n_features']}")
            logger.info(f"   Mean F1: {self.final_selection['mean_f1']:.3f} ± {self.final_selection['std_f1']:.3f}")
            logger.info(f"   Median F1: {self.final_selection.get('median_f1', 'N/A'):.3f}")
            logger.info(f"   Min F1: {self.final_selection.get('min_f1', 'N/A'):.3f}")
            logger.info(f"   Memory: {self.final_selection['memory_kb_est']:.2f} KB")
            logger.info(f"   Cluster Stability: {self.final_selection.get('cluster_stability', {}).get('mean_cluster_coverage', 'N/A'):.3f}")
            logger.info(f"   Top 10: {self.final_selection['selected_features'][:10]}")
            logger.info("=" * 80)
        else:
            logger.warning("No enhanced selection determined.")

    def _aggregate_frequency(self):
        freq_map: Dict[int, Dict[str, int]] = {n: {} for n in self.cfg.feature_counts}
        for fold in self.per_fold_results:
            for n_str, pref_res in fold["prefix_evaluations"].items():
                n = int(n_str)
                feats = pref_res["selected_features"]
                if n not in freq_map:
                    freq_map[n] = {}
                for f in feats:
                    freq_map[n][f] = freq_map[n].get(f, 0) + 1
        self.feature_frequency = freq_map

    def _build_candidate_sets_enhanced(self):
        """ENHANCED: Build candidate sets with cluster stability metrics."""
        candidate_sets = {}
        total_folds = len(self.per_fold_results)
        min_required = int(np.ceil(self.cfg.min_selection_frequency * total_folds))

        # Gather performance distributions
        perf_by_n: Dict[int, List[float]] = {}
        mcc_by_n: Dict[int, List[float]] = {}
        balacc_by_n: Dict[int, List[float]] = {}
        precision_by_n: Dict[int, List[float]] = {}
        recall_by_n: Dict[int, List[float]] = {}
        brier_by_n: Dict[int, List[float]] = {}
        roc_auc_by_n: Dict[int, List[float]] = {}

        for fold in self.per_fold_results:
            for n_str, pref_res in fold["prefix_evaluations"].items():
                n = int(n_str)
                perf_by_n.setdefault(n, []).append(pref_res["outer_f1"])
                mcc_by_n.setdefault(n, []).append(pref_res["outer_mcc"])
                balacc_by_n.setdefault(n, []).append(pref_res["outer_bal_acc"])
                precision_by_n.setdefault(n, []).append(pref_res["outer_precision"])
                recall_by_n.setdefault(n, []).append(pref_res["outer_recall"])
                
                # Handle calibrated metrics
                brier_val = pref_res.get("outer_brier", np.nan)
                roc_val = pref_res.get("outer_roc_auc", np.nan)
                if not np.isnan(brier_val):
                    brier_by_n.setdefault(n, []).append(brier_val)
                if not np.isnan(roc_val):
                    roc_auc_by_n.setdefault(n, []).append(roc_val)

        # Find best F1 for delta criterion
        all_mean_f1 = {n: np.mean(perf_by_n[n]) for n in perf_by_n}
        if not all_mean_f1:
            logger.error("No prefix evaluations found. Cannot build candidates.")
            return
        best_f1 = max(all_mean_f1.values())

        for n, freq_dict in self.feature_frequency.items():
            if n not in perf_by_n:
                continue
                
            # Basic statistics
            f1_scores = perf_by_n[n]
            mean_f1 = float(np.mean(f1_scores))
            std_f1 = float(np.std(f1_scores))
            
            # Distribution statistics
            median_f1 = float(np.median(f1_scores))
            min_f1 = float(np.min(f1_scores))
            p10_f1 = float(np.percentile(f1_scores, 10))
            
            # Other metrics
            mean_mcc = float(np.mean(mcc_by_n[n]))
            std_mcc = float(np.std(mcc_by_n[n]))
            mean_bal = float(np.mean(balacc_by_n[n]))
            std_bal = float(np.std(balacc_by_n[n]))
            mean_precision = float(np.mean(precision_by_n[n]))
            mean_recall = float(np.mean(recall_by_n[n]))
            
            # ENHANCED: Calibrated metrics
            mean_brier = float(np.mean(brier_by_n[n])) if n in brier_by_n else np.nan
            mean_roc_auc = float(np.mean(roc_auc_by_n[n])) if n in roc_auc_by_n else np.nan

            # Feature stability
            stable_feats = [f for f, c in freq_dict.items() if c >= min_required]
            if len(stable_feats) < n:
                sorted_by_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                stable_ordered = [f for f, c in sorted_by_freq]
                stable_feats = stable_ordered[:n]

            within_delta = (best_f1 - mean_f1) <= self.cfg.performance_delta
            memory_kb = (n * self.cfg.feature_memory_bytes) / 1024.0

            # ENHANCED: Cluster stability for this feature set
            cluster_stability = compute_cluster_stability(
                self.cluster_info_all, freq_dict, total_folds
            )

            candidate_sets[n] = {
                "n_features": n,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "median_f1": median_f1,
                "min_f1": min_f1,
                "p10_f1": p10_f1,
                "mean_mcc": mean_mcc,
                "std_mcc": std_mcc,
                "mean_balanced_accuracy": mean_bal,
                "std_balanced_accuracy": std_bal,
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "mean_brier": mean_brier,              # ENHANCED
                "mean_roc_auc": mean_roc_auc,          # ENHANCED
                "cluster_stability": cluster_stability,  # ENHANCED
                "best_f1_global": best_f1,
                "within_delta": within_delta,
                "min_selection_frequency_ratio": self.cfg.min_selection_frequency,
                "min_selection_count_required": min_required,
                "stable_feature_count": len(stable_feats),
                "memory_kb_est": memory_kb,
                "within_memory_budget": memory_kb <= self.cfg.feature_budget_kb,
                "selected_features": stable_feats[:n],
                "feature_frequency_top": dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:max(3*n, n)]),
                "supervised_clustering_used": True
            }

        self.candidate_sets = candidate_sets

    def _select_final_set_enhanced(self):
        """ENHANCED: Final selection with cluster stability consideration."""
        valids = [
            c for c in self.candidate_sets.values()
            if c["within_delta"] and c["within_memory_budget"]
        ]
        
        if not valids:
            logger.warning("No candidates meet criteria. Using best F1.")
            all_sorted = sorted(self.candidate_sets.values(), key=lambda x: x["mean_f1"], reverse=True)
            if all_sorted:
                self.final_selection = all_sorted[0]
            return

        # ENHANCED: Incorporate cluster stability in selection
        def enhanced_selection_score(candidate):
            cluster_cov = candidate.get("cluster_stability", {}).get("mean_cluster_coverage", 0.0)
            stability_bonus = self.cfg.cluster_stability_weight * cluster_cov
            robustness_score = candidate["min_f1"] + stability_bonus
            return (candidate["n_features"], -robustness_score, candidate["std_f1"], -candidate["mean_f1"])

        valids_sorted = sorted(valids, key=enhanced_selection_score)
        self.final_selection = valids_sorted[0]

    def _perform_significance_testing(self, df: pd.DataFrame, fold_defs: Dict):
        """ENHANCED: Statistical significance testing for final feature set."""
        if not self.final_selection:
            return
            
        logger.info("Performing statistical significance testing...")
        
        # Use first fold for significance testing (representative)
        test_fold = fold_defs["folds"][0]
        test_subj = test_fold["test_subject"]
        train_subjs = test_fold["train_subjects"]
        
        feature_cols = [c for c in df.columns if c not in ("subject", "label")]
        train_mask = df["subject"].isin(train_subjs)
        test_mask = df["subject"] == test_subj
        
        X_train_full = df.loc[train_mask, feature_cols].copy()
        y_train_full = df.loc[train_mask, "label"].values
        X_test_full = df.loc[test_mask, feature_cols].copy()
        y_test_full = df.loc[test_mask, "label"].values
        
        # Get final selected features
        final_features = self.final_selection["selected_features"]
        
        # Perform significance test
        sig_test_results = permutation_significance_test(
            X_train_full, y_train_full,
            X_test_full, y_test_full,
            final_features,
            self.cfg.random_state,
            self.cfg.permutation_test_n,
            self.cfg.calibrate_probabilities,
            self.cfg.fast_significance_test
        )
        
        self.final_selection["significance_test"] = sig_test_results
        
        logger.info(f"Significance test: p={sig_test_results['p_value']:.4f}, "
                   f"effect_size={sig_test_results['effect_size']:.3f}, "
                   f"significant_05={sig_test_results['significant_05']}")

    def _save_outputs(self):
        """Save all enhanced outputs."""
        with open(self.output_dir / "aggregated_feature_frequencies.json", "w") as f:
            json.dump(self.feature_frequency, f, indent=2)

        with open(self.output_dir / "candidate_feature_sets.json", "w") as f:
            json.dump(self.candidate_sets, f, indent=2, default=str)

        if self.final_selection:
            with open(self.output_dir / "final_selected_feature_set.json", "w") as f:
                json.dump(self.final_selection, f, indent=2, default=str)

        summary = {
            "stage": "1.5_ENHANCED",
            "timestamp": datetime.now().isoformat(),
            "enhancements": [
                "Supervised cluster representative selection (mutual info + F-test)",
                "Mitigated early stopping for consistent distribution evaluation",
                "Statistical significance testing via permutation/label-shuffle",
                "Calibrated probabilities for better Brier/ROC/PR metrics",
                "Cluster-level stability accounting to reduce bias"
            ],
            "config": asdict(self.cfg),
            "n_folds": len(self.per_fold_results),
            "best_global_f1": max([c["mean_f1"] for c in self.candidate_sets.values()]) if self.candidate_sets else None,
            "final_selection": self.final_selection,
            "artifacts": [
                "per_fold_results.json",
                "aggregated_feature_frequencies.json",
                "candidate_feature_sets.json",
                "final_selected_feature_set.json",
                "stage1_5_enhanced_summary.json"
            ]
        }
        
        with open(self.output_dir / "stage1_5_enhanced_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        with open(self.output_dir / "per_fold_results.json", "w") as f:
            json.dump(self.per_fold_results, f, indent=2)

        if self.final_selection:
            df_rank = pd.DataFrame({
                "feature": self.final_selection["selected_features"],
                "position": list(range(1, len(self.final_selection["selected_features"]) + 1))
            })
            df_rank.to_csv(self.output_dir / "final_feature_order.csv", index=False)

        logger.info("ENHANCED artifacts saved to: %s", self.output_dir.resolve())


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
def main():
    cfg_path = Path("../config/pipeline_config.json")
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            master_cfg = json.load(f)
        fe = master_cfg.get("feature_engineering", {})
        val = master_cfg.get("validation", {})
        stage_cfg = Stage15Config(
            stage0_dir="../outputs/stage0",
            output_dir="../outputs/stage1_5_enhanced",
            feature_counts=fe.get("feature_count_targets", [20, 25, 30, 35, 40, 45, 50, 60]),
            correlation_threshold=fe.get("correlation_threshold", 0.90),
            min_selection_frequency=fe.get("min_selection_frequency", 0.30),
            performance_delta=fe.get("performance_delta", 0.025),
            inner_folds=val.get("inner_cv", {}).get("n_splits", 4),
            use_permutation_importance=fe.get("use_permutation_importance", False),
            permutation_repeats=fe.get("permutation_repeats", 5),
            feature_budget_kb=fe.get("feature_budget_kb", 50.0),
            random_state=master_cfg.get("random_state", 42),
            early_stop_patience=fe.get("early_stop_patience", 6),
            early_stop_min_improve=fe.get("early_stop_min_improve", 0.001),
            early_stop_enabled=fe.get("early_stop_enabled", False),
            allow_resume=fe.get("allow_resume", True),
            # Enhanced parameters
            permutation_test_n=fe.get("permutation_test_n", 100),
            significance_alpha=fe.get("significance_alpha", 0.05),
            fast_significance_test=fe.get("fast_significance_test", True),
            calibrate_probabilities=fe.get("calibrate_probabilities", True),
            calibration_method=fe.get("calibration_method", "isotonic"),
            cluster_stability_weight=fe.get("cluster_stability_weight", 0.3)
        )
    else:
        stage_cfg = Stage15Config()

    selector = Stage15FeatureSelectorEnhanced(stage_cfg)
    selector.run()

    print("\n✅ Stage 1.5 ENHANCED complete. See outputs in:", stage_cfg.output_dir)


if __name__ == "__main__":
    main()
