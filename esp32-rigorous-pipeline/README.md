# ESP32-S3 Rigorous ML Pipeline

## 🎯 Project Goal
Create a **rigorous, leak-free machine learning pipeline** for ESP32-S3 stress detection that addresses all critical validation issues:

### ✅ **Problems Solved:**
- **Data Leakage**: Proper subject-level train/val/test splits
- **Temporal Leakage**: Address overlapping window issues  
- **Threshold Overfitting**: Proper nested cross-validation
- **Feature Selection Bias**: Unbiased feature selection with LOSO
- **Model Generalization**: True cross-subject validation
- **ESP32 Deployment**: Realistic memory and latency constraints

---

## 📋 **Pipeline Stages**

### **Stage 0: Data Integrity & Splitting** 🔍
**Objectives:**
- Guarantee zero subject leakage
- Provide stable train/val/test OR outer CV folds  
- Compute per-subject baseline stats for adaptive normalization

**Tasks:**
- Load full WESAD parquet with schema validation
- Subject statistics and partition strategy (LOSO)
- Save partition manifest and baseline stats
- Address temporal leakage from overlapping windows

**Deliverables:**
- `data_manifest.json`
- `per_subject_stats.json` 
- `fold_definitions.json`

---

### **Stage 1: Feature Engineering & Selection (Nested CV)** 🔧
**Objectives:**
- Identify stable, minimal feature subset
- Preserve median LOSO F1 within Δ ≤ 0.02 of full set
- Avoid leakage via nested CV

**Method:**
- Outer folds: LOSO (each subject test)
- Inner folds: GroupKFold for feature ranking
- Sequential evaluation with proper threshold tuning

**Deliverables:**
- `feature_selection_results.json`
- `selected_feature_list.txt`

---

### **Stage 2: Model Family Exploration** 🤖
**Objectives:**
- Test multiple model types on selected features
- Evaluate LOSO metrics: F1, balanced accuracy, MCC
- Calibrate probabilities within train folds

**Models to Test:**
- ExtraTrees / RandomForest
- Gradient Boosting (LightGBM/XGBoost)
- Logistic Regression (baseline)
- Small MLP / 1D CNN (optional)

**Deliverables:**
- `model_comparison.json`
- `chosen_model.joblib`

---

### **Stage 3: Compression & Optimization** ⚡
**Objectives:**
- Prune trees OR distill to tiny student network
- Evaluate retained performance via LOSO
- Choose final threshold as median of per-fold validated thresholds

**Options:**
- **A) Tree route**: Prune estimators + quantize
- **B) Distill route**: Train student with teacher probs + quantize

**Deliverables:**
- `compressed_model_metadata.json`
- `compressed_model.joblib`

---

### **Stage 4: Export & Embedded Implementation** 📱
**Objectives:**
- Convert to deployable artifact (C arrays or TFLite)
- Add unit tests comparing Python vs C inference
- Validate ESP32-S3 memory and latency constraints

**Deliverables:**
- `firmware/model_inference.c/.h`
- `test_vectors.bin`
- `size_report.txt`

---

### **Stage 5: Personalization & Adaptation** 🎯
**Objectives:**
- On-device baseline capture
- Adaptive threshold refinement
- Optional incremental calibration

**Deliverables:**
- `personalization_module.c`
- `adaptation_protocol.md`

---

### **Stage 6: Monitoring & Drift** 📊
**Objectives:**
- Provide hooks to log feature stats
- Flag baseline deviation beyond thresholds

**Deliverables:**
- `drift_rules.json`
- `monitoring_metrics_spec.md`

---

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.11+
- WESAD dataset access
- ESP32-S3 development environment

### **Quick Start**
```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run Stage 0 (Data Integrity)
python stages/stage0_data_integrity.py

# 3. Run Stage 1 (Feature Selection)
python stages/stage1_feature_selection.py

# ... continue with subsequent stages
```

---

## 📁 **Project Structure**
```
esp32-rigorous-pipeline/
├── README.md
├── requirements.txt
├── config/
│   ├── pipeline_config.json
│   └── esp32_constraints.json
├── stages/
│   ├── stage0_data_integrity.py
│   ├── stage1_feature_selection.py
│   ├── stage2_model_exploration.py
│   ├── stage3_compression.py
│   ├── stage4_embedded_export.py
│   ├── stage5_personalization.py
│   └── stage6_monitoring.py
├── outputs/
│   ├── stage0/
│   ├── stage1/
│   ├── stage2/
│   ├── stage3/
│   ├── stage4/
│   ├── stage5/
│   └── stage6/
├── tests/
├── firmware/
└── docs/
```

---

## ✅ **Success Criteria**

### **Stage 0**: Zero data leakage confirmed
### **Stage 1**: Feature reduction ≥40% with F1 drop ≤0.02
### **Stage 2**: Model selected with best LOSO performance
### **Stage 3**: Memory <150KB, F1 retention ≥95%
### **Stage 4**: ESP32 latency <10ms, 99% inference match
### **Stage 5**: False positive rate stable after adaptation
### **Stage 6**: Drift detection operational

---

## 🔬 **Key Innovations**
- **Leave-One-Subject-Out (LOSO)** validation throughout
- **Nested cross-validation** for unbiased feature selection
- **Proper temporal validation** addressing window overlap issues
- **ESP32-specific constraints** integrated from the start
- **Comprehensive drift detection** and personalization support

---

## 📚 **References**
- WESAD Dataset: Wearable Stress and Affect Detection
- ESP32-S3 Technical Reference Manual
- TensorFlow Lite Micro Documentation
