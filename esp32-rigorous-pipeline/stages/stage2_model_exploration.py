#!/usr/bin/env python3
"""
ESP32-S3 Rigorous ML Pipeline
Stage 2 (Refactored): Model Family Exploration & Selection

Improvements:
- Reduced redundancy and duplicated logic
- Group-aware calibration performed ONLY after model family selection
- Unified threshold optimization logic
- Dataclasses for fold and aggregate results
- Cleaner separation of concerns

Author: User & AI Assistant (Refactored)
Date: 2025-08-31
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import json
import logging
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    precision_score, recall_score, confusion_matrix,
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, GroupKFold

# Optional models
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - STAGE2 - %(levelname)s - %(message)s"
)
logger = logging.getLogger("STAGE2")

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
SUBJECT_COL = "subject"
TARGET_COL = "label"

THRESHOLD_GRID = np.linspace(0.05, 0.95, 37)  # for direct threshold search
ECE_BINS = 15
MIN_VALID_FOLDS = 3
MIN_POS_FOR_PLATT = 20
MIN_POS_FOR_ISO = 50
MIN_NEG_FOR_PLATT = 20
MIN_NEG_FOR_ISO = 50
MIN_CLASS_FOR_FOLD = 1  # require both classes present
RANDOM_STATE_DEFAULT = 42

# ------------------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------------------
@dataclass
class Stage2Config:
    # Paths
    stage0_dir: str = "../outputs/stage0"
    stage1_5_dir: str = "../outputs/stage1_5_enhanced"
    output_dir: str = "../outputs/stage2_model_exploration"
    config_path: str = "../config/pipeline_config.json"

    # Models to test
    test_extra_trees: bool = True
    test_random_forest: bool = True
    test_lightgbm: bool = True
    test_xgboost: bool = True
    test_logistic_regression: bool = True
    test_mlp: bool = True
    test_cnn: bool = False  # disabled placeholder

    # Hyperparameters
    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1

    # Selection metrics
    primary_metric: str = "f1"
    secondary_metrics: List[str] = field(default_factory=lambda: ["balanced_accuracy", "mcc", "precision", "recall"])

    # Objective trade-offs (penalties)
    size_weight: float = 0.3
    latency_weight: float = 0.2
    variance_weight: float = 0.0  # optional penalty for std of primary metric

    # Threshold optimization
    optimize_threshold: bool = True
    threshold_metric: str = "f1"
    threshold_cv_folds: int = 3
    threshold_strategy: str = "inner_cv"  # "inner_cv" or "direct_grid"

    # Final calibration
    final_calibration: bool = True
    calibration_preference: str = "isotonic"  # "isotonic" or "platt"

    # Constraints
    max_model_size_mb: float = 10.0
    max_inference_time_ms: float = 50.0

    # Reproducibility
    random_state: int = RANDOM_STATE_DEFAULT

    # Misc
    save_environment: bool = False  # optionally dump pip freeze
    allow_resume: bool = True


@dataclass
class FoldResult:
    fold_id: Any
    test_subject: Any
    model_type: str
    metrics: Dict[str, Any]
    predictions: Dict[str, Any]
    optimal_threshold: float
    fold_status: Dict[str, Any]


@dataclass
class AggregateMetrics:
    model_type: str
    mean_primary: float
    std_primary: float
    composite_score: float
    meets_constraints: bool
    n_valid_folds: int
    n_degenerate_folds: int
    fold_completion_rate: float
    raw: Dict[str, Any]


@dataclass
class FinalArtifacts:
    model_type: str
    model: Any
    scaler: Optional[StandardScaler]
    calibrator: Optional[Any]
    calibrator_type: str
    optimal_threshold: float
    features: List[str]
    is_calibrated: bool
    training_subjects: List[Any]
    training_samples: int
    class_distribution: Dict[str, int]
    oof_metrics: Dict[str, Any]


# ------------------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------------------
def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)


def safe_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    # Force both labels for stability
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_BINS) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.any():
            ece += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def load_artifacts(cfg: Stage2Config) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    stage0 = Path(cfg.stage0_dir)
    manifest = json.loads(Path(stage0 / "data_manifest.json").read_text())
    folds = json.loads(Path(stage0 / "fold_definitions.json").read_text())
    features_json = json.loads(Path(cfg.stage1_5_dir, "final_selected_feature_set.json").read_text())

    df = pd.read_parquet(manifest["source_file"])
    df = df.replace([np.inf, -np.inf], np.nan)
    drop_cols = df.columns[df.isna().all()].tolist()
    if drop_cols:
        logger.info(f"Dropping all-NaN columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    df = df.fillna(0)

    return df, folds, features_json["selected_features"]


def create_model(model_type: str, cfg: Stage2Config):
    rs = cfg.random_state
    if model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split, min_samples_leaf=cfg.min_samples_leaf,
            random_state=rs, n_jobs=-1, class_weight="balanced"
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split, min_samples_leaf=cfg.min_samples_leaf,
            random_state=rs, n_jobs=-1, class_weight="balanced"
        )
    if model_type == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        return lgb.LGBMClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth if cfg.max_depth else -1,
            class_weight="balanced", random_state=rs, n_jobs=-1, verbose=-1
        )
    if model_type == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        return xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth if cfg.max_depth else 6,
            random_state=rs, n_jobs=-1, eval_metric="logloss"
        )
    if model_type == "logistic_regression":
        return LogisticRegression(
            random_state=rs, max_iter=2000, class_weight="balanced", solver="liblinear"
        )
    if model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500, random_state=rs,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=20
        )
    if model_type == "cnn":
        raise NotImplementedError("CNN disabled in refactored version")
    raise ValueError(f"Unknown model_type: {model_type}")


def get_probabilities(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()
    # Fallback: decision_function
    raw = model.decision_function(X)
    return 1 / (1 + np.exp(-raw))


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = safe_confusion(y_true, y_pred)
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def compute_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    out = {}
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception:
        out["pr_auc"] = np.nan
    try:
        out["brier_score"] = brier_score_loss(y_true, y_prob)
    except Exception:
        out["brier_score"] = np.nan
    return out


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str) -> Tuple[float, float]:
    """
    Direct threshold selection on provided probabilities.
    Returns (threshold, metric_score).
    """
    if metric == "f1":
        precision, recall, thr = precision_recall_curve(y_true, y_prob)
        f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
        idx = np.nanargmax(f1s)
        return (float(thr[idx]) if idx < len(thr) else 0.5, float(f1s[idx]))
    elif metric == "balanced_accuracy":
        best_t, best_score = 0.5, -1
        for t in THRESHOLD_GRID:
            pred = (y_prob >= t).astype(int)
            score = balanced_accuracy_score(y_true, pred)
            if score > best_score:
                best_t, best_score = t, score
        return float(best_t), float(best_score)
    else:
        return 0.5, 0.0


def inner_cv_threshold(model, X: np.ndarray, y: np.ndarray,
                       metric: str, folds: int, random_state: int) -> float:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof_prob = np.zeros_like(y, dtype=float)
    for tr, va in skf.split(X, y):
        m = type(model)(**model.get_params())
        m.fit(X[tr], y[tr])
        oof_prob[va] = get_probabilities(m, X[va])
    t, _ = select_threshold(y, oof_prob, metric)
    return t


def compute_composite(primary: float, std_primary: float,
                      size_mb: float, time_ms: float,
                      cfg: Stage2Config) -> float:
    size_pen = min(size_mb / cfg.max_model_size_mb, 1.0)
    time_pen = min(time_ms / cfg.max_inference_time_ms, 1.0)
    variance_pen = std_primary if cfg.variance_weight > 0 else 0.0
    return primary - cfg.size_weight * size_pen - cfg.latency_weight * time_pen - cfg.variance_weight * variance_pen


def serialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        # Convert numpy int64 keys to regular int
        return {str(k) if hasattr(k, 'item') else k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


# ------------------------------------------------------------------------------
# Calibration Manager
# ------------------------------------------------------------------------------
class CalibrationManager:
    def __init__(self, preference: str):
        self.preference = preference

    def generate_group_oof(self, model, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           n_splits: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        uniq = np.unique(groups)
        n_splits = min(n_splits, len(uniq))
        if n_splits < 2:
            raise ValueError("Insufficient groups for OOF calibration")
        gkf = GroupKFold(n_splits=n_splits)
        oof = np.zeros_like(y, dtype=float)
        mask = np.zeros_like(y, dtype=bool)
        for tr, va in gkf.split(X, y, groups):
            m = type(model)(**model.get_params())
            m.fit(X[tr], y[tr])
            oof[va] = get_probabilities(m, X[va])
            mask[va] = True
        return oof[mask], y[mask]

    def fit_calibrator(self, oof_prob: np.ndarray, y: np.ndarray) -> Tuple[Optional[Any], str, Dict[str, Any]]:
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        meta = {"positives": pos, "negatives": neg, "total": len(y)}
        if pos < MIN_POS_FOR_PLATT or neg < MIN_NEG_FOR_PLATT:
            meta["reason"] = "insufficient_class_counts"
            return None, "none", meta
        # Isotonic?
        if (self.preference == "isotonic" and
                pos >= MIN_POS_FOR_ISO and neg >= MIN_NEG_FOR_ISO):
            try:
                from sklearn.isotonic import IsotonicRegression
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(oof_prob, y)
                meta["method"] = "isotonic"
                return iso, "isotonic", meta
            except Exception as e:
                logger.warning(f"Isotonic failed: {e}; fallback to Platt")
        # Platt
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=1000)
            lr.fit(oof_prob.reshape(-1, 1), y)
            meta["method"] = "platt"
            return lr, "platt", meta
        except Exception as e:
            logger.error(f"Platt calibration failed: {e}")
            meta["reason"] = "calibration_failure"
            return None, "none", meta

    @staticmethod
    def apply(calibrator, cal_type: str, raw_prob: np.ndarray) -> np.ndarray:
        if calibrator is None or cal_type == "none":
            return raw_prob
        if cal_type == "isotonic":
            return calibrator.transform(raw_prob)
        if cal_type == "platt":
            return calibrator.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
        return raw_prob


# ------------------------------------------------------------------------------
# Evaluation / Aggregation
# ------------------------------------------------------------------------------
def is_degenerate_fold(y_test: np.ndarray) -> bool:
    classes = np.unique(y_test)
    return len(classes) < 2


def evaluate_fold(model_type: str,
                  cfg: Stage2Config,
                  X_train_df: pd.DataFrame,
                  y_train: np.ndarray,
                  X_test_df: pd.DataFrame,
                  y_test: np.ndarray,
                  features: List[str]) -> FoldResult:
    if is_degenerate_fold(y_test):
        return FoldResult(
            fold_id=None,
            test_subject=None,
            model_type=model_type,
            metrics={"fold_degenerate": True},
            predictions={"y_true": y_test.tolist(), "y_pred": [], "y_prob": []},
            optimal_threshold=float("nan"),
            fold_status={"degenerate": True}
        )

    # Scaling logic
    scaler = None
    needs_scaler = model_type in ["logistic_regression", "mlp", "cnn"]
    if needs_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df[features])
        X_test = scaler.transform(X_test_df[features])
    else:
        X_train = X_train_df[features].values
        X_test = X_test_df[features].values

    model = create_model(model_type, cfg)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_prob = get_probabilities(model, X_test)

    # Threshold
    if cfg.optimize_threshold:
        if cfg.threshold_strategy == "inner_cv":
            opt_t = inner_cv_threshold(
                model, X_train, y_train,
                metric=cfg.threshold_metric,
                folds=cfg.threshold_cv_folds,
                random_state=cfg.random_state
            )
        else:
            opt_t, _ = select_threshold(y_train, get_probabilities(model, X_train), cfg.threshold_metric)
    else:
        opt_t = 0.5

    y_pred = (y_prob >= opt_t).astype(int)

    cls_metrics = compute_classification_metrics(y_test, y_pred)
    prob_metrics = compute_probability_metrics(y_test, y_prob)

    # Add supplementary
    cls_metrics.update(prob_metrics)
    cls_metrics["optimal_threshold"] = opt_t
    cls_metrics["training_time_sec"] = train_time
    cls_metrics["calibrated"] = False
    cls_metrics["fold_degenerate"] = False
    cls_metrics["class_report"] = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    # Micro-benchmark inference
    try:
        # batch inference (10 samples or all)
        sample_batch = X_test[: min(10, len(X_test))]
        batch_start = time.perf_counter()
        _ = model.predict(sample_batch)
        batch_time = (time.perf_counter() - batch_start) * 1000 / sample_batch.shape[0]
    except Exception:
        batch_time = float("nan")
    cls_metrics["inference_time_ms"] = batch_time

    # Approx model size
    try:
        import pickle
        cls_metrics["model_size_mb"] = len(pickle.dumps(model)) / (1024 * 1024)
    except Exception:
        cls_metrics["model_size_mb"] = float("nan")

    predictions = {
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist()
    }

    return FoldResult(
        fold_id=None,
        test_subject=None,
        model_type=model_type,
        metrics=cls_metrics,
        predictions=predictions,
        optimal_threshold=opt_t,
        fold_status={"degenerate": False}
    )


def aggregate_model_results(model_type: str,
                            fold_results: List[FoldResult],
                            cfg: Stage2Config) -> AggregateMetrics:
    valid = [fr for fr in fold_results if not fr.metrics.get("fold_degenerate", False)]
    degenerate = [fr for fr in fold_results if fr.metrics.get("fold_degenerate", False)]

    agg: Dict[str, List[float]] = {}
    for fr in valid:
        for k, v in fr.metrics.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                agg.setdefault(k, []).append(v)

    summary: Dict[str, Any] = {}
    for k, vals in agg.items():
        summary[f"mean_{k}"] = float(np.mean(vals))
        summary[f"std_{k}"] = float(np.std(vals))
        summary[f"min_{k}"] = float(np.min(vals))
        summary[f"max_{k}"] = float(np.max(vals))
        summary[f"median_{k}"] = float(np.median(vals))

    n_valid = len(valid)
    n_total = len(fold_results)
    summary["n_valid_folds"] = n_valid
    summary["n_degenerate_folds"] = len(degenerate)
    summary["fold_completion_rate"] = (n_valid / n_total) if n_total else 0.0

    # Composite
    if all(x in summary for x in ["mean_f1", "mean_model_size_mb", "mean_inference_time_ms"]):
        summary["composite_score"] = compute_composite(
            summary["mean_f1"],
            summary.get("std_f1", 0.0),
            summary["mean_model_size_mb"],
            summary["mean_inference_time_ms"],
            cfg
        )

    meets_constraints = (
        summary.get("mean_model_size_mb", float("inf")) <= cfg.max_model_size_mb and
        summary.get("mean_inference_time_ms", float("inf")) <= cfg.max_inference_time_ms
    )

    return AggregateMetrics(
        model_type=model_type,
        mean_primary=summary.get(f"mean_{cfg.primary_metric}", float("nan")),
        std_primary=summary.get(f"std_{cfg.primary_metric}", float("nan")),
        composite_score=summary.get("composite_score", float("nan")),
        meets_constraints=meets_constraints,
        n_valid_folds=n_valid,
        n_degenerate_folds=summary["n_degenerate_folds"],
        fold_completion_rate=summary["fold_completion_rate"],
        raw=summary
    )


# ------------------------------------------------------------------------------
# Main Orchestrator
# ------------------------------------------------------------------------------
class Stage2ModelExplorer:
    def __init__(self, cfg: Stage2Config):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fold_defs: Dict[str, Any] = {}
        self.selected_features: List[str] = []
        self.data: pd.DataFrame = pd.DataFrame()
        self.model_fold_results: Dict[str, List[FoldResult]] = {}
        self.aggregates: Dict[str, AggregateMetrics] = {}
        self.ranking: List[str] = []
        self.final_artifacts: Optional[FinalArtifacts] = None

    def run(self):
        set_global_seed(self.cfg.random_state)
        self._load_inputs()
        model_types = self._collect_model_types()
        logger.info(f"Evaluating model families: {model_types}")

        for mt in model_types:
            self._evaluate_model_type_loso(mt)

        self._rank_models()
        best = self.ranking[0] if self.ranking else None
        if best:
            logger.info(f"Selected best model family: {best}")
            self._train_finalize(best)
        else:
            logger.error("No valid model families found.")
        self._save_all()

    # ---------------- Load & Setup ---------------- #
    def _load_inputs(self):
        df, fold_defs, feats = load_artifacts(self.cfg)
        self.data = df
        self.fold_defs = fold_defs
        self.selected_features = feats
        logger.info(f"Loaded data shape={df.shape} features={len(feats)} folds={len(fold_defs['folds'])}")

    def _collect_model_types(self) -> List[str]:
        types = []
        if self.cfg.test_extra_trees: types.append("extra_trees")
        if self.cfg.test_random_forest: types.append("random_forest")
        if self.cfg.test_lightgbm and HAS_LIGHTGBM: types.append("lightgbm")
        if self.cfg.test_xgboost and HAS_XGBOOST: types.append("xgboost")
        if self.cfg.test_logistic_regression: types.append("logistic_regression")
        if self.cfg.test_mlp: types.append("mlp")
        if self.cfg.test_cnn: types.append("cnn")  # placeholder
        return types

    # ---------------- Model Evaluation ---------------- #
    def _evaluate_model_type_loso(self, model_type: str):
        fold_results: List[FoldResult] = []
        for fold in self.fold_defs["folds"]:
            fid = fold["fold_id"]
            test_subj = fold["test_subject"]
            train_subjs = fold["train_subjects"]

            train_mask = self.data[SUBJECT_COL].isin(train_subjs)
            test_mask = self.data[SUBJECT_COL] == test_subj

            X_train_df = self.data.loc[train_mask, self.selected_features]
            y_train = self.data.loc[train_mask, TARGET_COL].values
            X_test_df = self.data.loc[test_mask, self.selected_features]
            y_test = self.data.loc[test_mask, TARGET_COL].values

            logger.info(f"[{model_type}] Fold={fid} TestSubject={test_subj} TrainN={len(X_train_df)} TestN={len(X_test_df)}")

            fr = evaluate_fold(
                model_type=model_type,
                cfg=self.cfg,
                X_train_df=X_train_df,
                y_train=y_train,
                X_test_df=X_test_df,
                y_test=y_test,
                features=self.selected_features
            )
            fr.fold_id = fid
            fr.test_subject = test_subj
            fold_results.append(fr)

            if not fr.metrics.get("fold_degenerate", False):
                logger.info(f"  -> F1={fr.metrics['f1']:.3f} BalAcc={fr.metrics['balanced_accuracy']:.3f} "
                            f"MCC={fr.metrics['mcc']:.3f} Thr={fr.metrics['optimal_threshold']:.3f}")
            else:
                logger.warning("  -> Degenerate fold (skipped)")

        self.model_fold_results[model_type] = fold_results
        agg = aggregate_model_results(model_type, fold_results, self.cfg)
        self.aggregates[model_type] = agg

        logger.info(f"[{model_type}] Aggregated: mean_{self.cfg.primary_metric}={agg.mean_primary:.3f} "
                    f"std={agg.std_primary:.3f} composite={agg.composite_score:.3f} "
                    f"valid_folds={agg.n_valid_folds}/{len(fold_results)}")

    # ---------------- Ranking ---------------- #
    def _rank_models(self):
        # Filter valid by # folds
        candidates = [
            (mt, agg) for mt, agg in self.aggregates.items()
            if agg.n_valid_folds >= MIN_VALID_FOLDS
        ]
        # Sort by composite then primary
        candidates.sort(key=lambda x: (
            np.nan_to_num(x[1].composite_score, nan=-1),
            np.nan_to_num(x[1].mean_primary, nan=-1)
        ), reverse=True)
        
        # Embedded-friendly model selection override
        # If MLP is within 0.01 of the best composite score, choose MLP for ESP32 deployment
        if len(candidates) >= 2:
            best_score = candidates[0][1].composite_score
            mlp_candidate = next(((mt, agg) for mt, agg in candidates if mt == "mlp"), None)
            
            if mlp_candidate and not np.isnan(best_score) and not np.isnan(mlp_candidate[1].composite_score):
                score_diff = best_score - mlp_candidate[1].composite_score
                if 0 <= score_diff <= 0.01:  # MLP is within 0.01 of the best
                    # Move MLP to the front for embedded deployment advantages
                    candidates = [mlp_candidate] + [c for c in candidates if c[0] != "mlp"]
                    logger.info(f"🎯 EMBEDDED OVERRIDE: Selected MLP (composite={mlp_candidate[1].composite_score:.3f}) "
                               f"over {candidates[1][0]} (composite={best_score:.3f}) for ESP32 deployment benefits")
        
        self.ranking = [mt for mt, _ in candidates]

        logger.info("\n--- MODEL FAMILY RANKING ---")
        for i, (mt, agg) in enumerate(candidates, start=1):
            logger.info(f"{i}. {mt}: composite={agg.composite_score:.3f} "
                        f"{self.cfg.primary_metric}={agg.mean_primary:.3f} "
                        f"std={agg.std_primary:.3f} valid_folds={agg.n_valid_folds}")

    # ---------------- Final Training & Calibration ---------------- #
    def _train_finalize(self, best_model_type: str):
        logger.info(f"\nFinalizing model: {best_model_type}")
        X_df = self.data[self.selected_features]
        y = self.data[TARGET_COL].values
        groups = self.data[SUBJECT_COL].values

        needs_scaler = best_model_type in ["logistic_regression", "mlp", "cnn"]
        scaler = StandardScaler().fit(X_df) if needs_scaler else None
        X_all = scaler.transform(X_df) if scaler is not None else X_df.values

        base_model = create_model(best_model_type, self.cfg)
        base_model.fit(X_all, y)

        # Group OOF for calibration & threshold
        cal_mgr = CalibrationManager(preference=self.cfg.calibration_preference)
        try:
            oof_proba, oof_y = cal_mgr.generate_group_oof(base_model, X_all, y, groups, n_splits=5)
        except Exception as e:
            logger.warning(f"Group OOF generation failed: {e}")
            oof_proba, oof_y = None, None

        calibrator = None
        cal_type = "none"
        cal_meta: Dict[str, Any] = {}
        if self.cfg.final_calibration and oof_proba is not None:
            calibrator, cal_type, cal_meta = cal_mgr.fit_calibrator(oof_proba, oof_y)

        if oof_proba is not None:
            calibrated_oof = cal_mgr.apply(calibrator, cal_type, oof_proba)
            # Threshold on calibrated OOF probabilities
            t_final, metric_val = select_threshold(oof_y, calibrated_oof, self.cfg.threshold_metric)
            oof_ece = expected_calibration_error(oof_y, calibrated_oof)
            try:
                oof_brier = brier_score_loss(oof_y, calibrated_oof)
            except Exception:
                oof_brier = np.nan
        else:
            t_final = 0.5
            metric_val = np.nan
            oof_ece = np.nan
            oof_brier = np.nan

        self.final_artifacts = FinalArtifacts(
            model_type=best_model_type,
            model=base_model,
            scaler=scaler,
            calibrator=calibrator,
            calibrator_type=cal_type,
            optimal_threshold=t_final,
            features=self.selected_features,
            is_calibrated=(cal_type != "none"),
            training_subjects=list(np.unique(groups)),
            training_samples=len(y),
            class_distribution=dict(zip(*np.unique(y, return_counts=True))),
            oof_metrics={
                "threshold_metric": self.cfg.threshold_metric,
                "threshold_score": metric_val,
                "oof_ece": oof_ece,
                "oof_brier": oof_brier,
                "calibration_meta": cal_meta
            }
        )

        logger.info(f"Final model ready. Calibrator={cal_type} Thr={t_final:.3f} "
                    f"OOF_ECE={oof_ece} OOF_Brier={oof_brier}")

        # Persist
        self._save_final_model()

    def _save_final_model(self):
        if not self.final_artifacts:
            return
        out = self.output_dir
        joblib.dump(self.final_artifacts.model, out / "final_model.joblib")
        if self.final_artifacts.scaler:
            joblib.dump(self.final_artifacts.scaler, out / "final_scaler.joblib")
        if self.final_artifacts.calibrator:
            joblib.dump(self.final_artifacts.calibrator, out / "final_calibrator.joblib")

        meta = asdict(self.final_artifacts)
        # remove non-serializable objects
        meta.pop("model")
        if meta.get("scaler") is not None:
            meta["scaler"] = "saved_scaler.joblib"
        if meta.get("calibrator") is not None:
            meta["calibrator"] = f"calibrator_{self.final_artifacts.calibrator_type}.joblib"
        Path(out / "final_model_artifacts.json").write_text(json.dumps(serialize(meta), indent=2))

    # ---------------- Persistence ---------------- #
    def _save_all(self):
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # Aggregates
        aggregates_json = {
            mt: serialize(agg.raw) for mt, agg in self.aggregates.items()
        }
        Path(out / "aggregated_metrics.json").write_text(json.dumps(aggregates_json, indent=2))

        # Ranking
        ranking_payload = {
            "ranking": self.ranking,
            "primary_metric": self.cfg.primary_metric
        }
        Path(out / "model_ranking.json").write_text(json.dumps(ranking_payload, indent=2))

        # Fold results
        folds_payload: Dict[str, Any] = {}
        for mt, fr_list in self.model_fold_results.items():
            folds_payload[mt] = [serialize({
                "fold_id": fr.fold_id,
                "test_subject": fr.test_subject,
                "metrics": fr.metrics,
                "optimal_threshold": fr.optimal_threshold
            }) for fr in fr_list]
        Path(out / "fold_results.json").write_text(json.dumps(folds_payload, indent=2))

        # Config & summary
        summary = {
            "config": asdict(self.cfg),
            "selected_features_count": len(self.selected_features),
            "best_model": self.ranking[0] if self.ranking else None,
            "final_model_saved": self.final_artifacts is not None
        }
        Path(out / "stage2_summary.json").write_text(json.dumps(summary, indent=2))

        logger.info(f"Artifacts saved in {out.resolve()}")

    # ---------------- Inference API ---------------- #
    def predict(self, X_new: pd.DataFrame) -> Dict[str, Any]:
        if not self.final_artifacts:
            raise RuntimeError("Final model not trained.")
        fa = self.final_artifacts
        X = X_new[fa.features].values
        if fa.scaler:
            X = fa.scaler.transform(X)
        raw_prob = get_probabilities(fa.model, X)
        prob = CalibrationManager.apply(fa.calibrator, fa.calibrator_type, raw_prob)
        pred = (prob >= fa.optimal_threshold).astype(int)
        return {
            "predictions": pred.tolist(),
            "probabilities": prob.tolist(),
            "raw_probabilities": raw_prob.tolist(),
            "threshold": fa.optimal_threshold,
            "calibrator_type": fa.calibrator_type
        }


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
def main():
    cfg_path = Path("../config/pipeline_config.json")
    if cfg_path.exists():
        master = json.loads(cfg_path.read_text())
        mc = master.get("model_exploration", {})
        cfg = Stage2Config(
            stage0_dir="../outputs/stage0",
            stage1_5_dir="../outputs/stage1_5_enhanced",
            output_dir="../outputs/stage2_model_exploration",
            test_extra_trees=mc.get("test_extra_trees", True),
            test_random_forest=mc.get("test_random_forest", True),
            test_lightgbm=mc.get("test_lightgbm", True),
            test_xgboost=mc.get("test_xgboost", True),
            test_logistic_regression=mc.get("test_logistic_regression", True),
            test_mlp=mc.get("test_mlp", True),
            test_cnn=mc.get("test_cnn", False),
            n_estimators=mc.get("n_estimators", 200),
            primary_metric=mc.get("primary_metric", "f1"),
            max_model_size_mb=mc.get("max_model_size_mb", 10.0),
            max_inference_time_ms=mc.get("max_inference_time_ms", 50.0),
            size_weight=mc.get("size_weight", 0.3),
            latency_weight=mc.get("latency_weight", 0.2),
            variance_weight=mc.get("variance_weight", 0.0),
            optimize_threshold=mc.get("optimize_threshold", True),
            threshold_metric=mc.get("threshold_metric", "f1"),
            threshold_cv_folds=mc.get("threshold_cv_folds", 3),
            threshold_strategy=mc.get("threshold_strategy", "inner_cv"),
            final_calibration=mc.get("final_calibration", True),
            calibration_preference=mc.get("calibration_preference", "isotonic"),
            random_state=master.get("random_state", RANDOM_STATE_DEFAULT),
            save_environment=mc.get("save_environment", False),
        )
    else:
        cfg = Stage2Config()

    explorer = Stage2ModelExplorer(cfg)
    explorer.run()
    print(f"\n✅ Stage 2 complete. Outputs: {cfg.output_dir}")


if __name__ == "__main__":
    main()