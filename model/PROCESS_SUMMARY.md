# Model pipeline: exact steps and outputs

This documents the six steps we executed in `model/` to prepare and select the final model (chosen: MLP).

## 1) Build window manifest (tri-state labels)
- Script: `model/01_windows_manifet.py` (header name: 01_windows_manifest.py)
- What it does: Creates a global window manifest per subject with tri-state labels (stress=1, nonstress=0, junk=-1), window purity, and time bounds.
- Key parameters:
  - Window length W = 30 s, step S = 10 s, purity threshold τ = 0.75
  - Label stream fs = 700 Hz; modality fs: ACC=32 Hz, BVP=64 Hz, EDA=4 Hz, TEMP=4 Hz
  - Windows computed against the shared minimum duration across modalities/labels to keep alignment
- Output: `data/working/wesad_processed/windows_manifest_W30_S10_tau75.parquet`

## 2) Extract time-series features (FLIRT ACC extractor for all modalities)
- Script: `model/02_extract_features_flirt.py`
- What it does: Applies FLIRT's ACC feature extractor to ACC, BVP, EDA, and TEMP separately, prefixes features by modality, aligns with the window manifest, drops `category == junk`, and writes the final feature table.
- Notes:
  - Each modality is trimmed to the shared min duration so window counts match
  - Uses W = 30 s, S = 10 s; optional parallelism via FLIRT
- Output: `model/data/output/final_features_ACCextractor_allmods_W30_S10_tau75.parquet`

## 3) Build LOSO folds (+ holdout)
- Script: `model/03_build_LOSO_holdout.py`
- What it does: Constructs Leave-One-Subject-Out folds over the available subjects and reserves holdouts.
- Configuration:
  - Holdout subjects: S9, S10
  - 13 LOSO folds across remaining subjects; no degenerate folds
- Outputs (under `model/data/output/folds_W30_S10_tau75/`):
  - `loso_folds.json` (n_folds=13)
  - `holdout_subjects.txt` (S9, S10)
  - `per_subject_counts.csv`

## 4) Baseline LOSO (ExtraTrees + train-side threshold tuning)
- Script: `model/04_baseline_LOSO.py`
- What it does: Baseline ExtraTreesClassifier on the full feature set, per-fold train/test split by subject.
- Details:
  - Train-side OOF threshold tuning over grid [0.1, 0.9] step 0.01 to maximize F1 (no leakage)
  - Class weight for positive class ≈ 1.65; median imputation; drop all-NaN features per train
- Outputs: `model/data/output/04_baseline_loso/`
  - `fold_metrics.csv` (per-fold metrics)
  - `aggregate_summary.csv`

## 5) Feature selection with LOSO (simple F-stat + correlation clustering)
- Script: `model/05_feature_selection_LOSO.py`
- What it does: Per LOSO fold, cleans features, ranks by ANOVA F-stat, prunes redundancy with correlation clustering, and chooses a small feature subset that performs well.
- Steps/parameters:
  - Clean: replace inf, drop all-NaN (train-based), drop near-constant (var < 1e-6)
  - Rank by F-statistic (ANOVA)
  - Correlation clustering with |corr| ≥ 0.80; select representatives per cluster: ceil(30% of size), max 5 per cluster
  - Inner OOF ExtraTrees with class weight 1.65 to tune decision threshold; evaluate candidate N in [16, 24, 32, 48, 64]
  - Pick the smallest N within Δ=0.01 of the best inner F1; evaluate on the held-out subject
  - Aggregate across folds to decide a global N
- Results:
  - Selected global_N = 16 (see `selection_summary.json`)
- Outputs: `model/data/output/05_selection_loso/`
  - `selection_summary.json`, `final_feature_order.csv`, `candidate_N_summary.csv`,
    `aggregated_feature_frequencies.csv`, per-fold diagnostics and metrics

## 6) Model selection with LOSO (selected features only)
- Script: `model/06_model_selection_loso.py`
- What it does: Compares models using only the selected features from step 5 (top global_N=16), with train-side OOF threshold tuning per model and LOSO evaluation.
- Models evaluated:
  - ExtraTrees, RandomForest, LogisticRegression, LinearSVC, MLP (64×64 ReLU, StandardScaler), XGBoost (if installed)
- MLP config (as used here):
  - Pipeline: StandardScaler → MLPClassifier(hidden_layer_sizes=(64,64), activation=relu,
    alpha=0.0003, learning_rate=adaptive, max_iter=1000, early_stopping=True, n_iter_no_change=20)
- Key outputs: `model/data/output/06_model_selection_loso/`
  - `all_models_all_folds.csv` (all folds, all models)
  - `<Model>_fold_metrics.csv` for each model
  - `model_ranking.csv` (sorted by mean F1 and mean balanced accuracy)
- Observed ranking (from `model_ranking.csv`):
  - XGBoost mean F1 ≈ 0.8344; MLP mean F1 ≈ 0.8113; others lower on average
- Final choice: MLP selected as the production model.

---

If you’d like, we can add a short rationale for choosing MLP (e.g., on-device feasibility, simpler dependency footprint, stability) and export a finalized MLP model artifact next.