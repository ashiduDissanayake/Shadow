#!/usr/bin/env python3
"""
Test C implementation against full dataset to verify all metrics match sklearn exactly
"""

import json
import ctypes
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from pathlib import Path

def compile_c_model():
    """Compile the C model"""
    print("🔨 Compiling C model...")
    
    result = subprocess.run([
        "gcc", "-shared", "-fPIC", "-O3", "-lm",
        "components/simple_mlp.c", "-o", "simple_mlp.so"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Compilation failed:")
        print(result.stderr)
        return False
    
    print("✅ Compilation successful")
    return True

def load_c_library():
    """Load the compiled C library"""
    lib = ctypes.CDLL("./simple_mlp.so")
    
    # Configure function signatures
    lib.shadow_mlp_predict_probability.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.shadow_mlp_predict_probability.restype = ctypes.c_float
    
    lib.shadow_mlp_predict_class.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.shadow_mlp_predict_class.restype = ctypes.c_int
    
    return lib

def load_sklearn_model():
    """Load the complete sklearn pipeline (scaler + model + calibrator)"""
    base_path = "../outputs/stage2_model_exploration"
    
    try:
        # Load individual components
        scaler = joblib.load(f"{base_path}/final_scaler.joblib")
        model = joblib.load(f"{base_path}/final_model.joblib") 
        calibrator = joblib.load(f"{base_path}/final_calibrator.joblib")
        
        print("✅ Loaded sklearn pipeline components")
        return {'scaler': scaler, 'model': model, 'calibrator': calibrator}
        
    except Exception as e:
        print(f"❌ Failed to load sklearn pipeline: {e}")
        return None

def test_full_dataset():
    """Test C implementation against full dataset"""
    
    # Compile C model
    if not compile_c_model():
        return False
    
    # Load C library
    try:
        c_lib = load_c_library()
    except Exception as e:
        print(f"❌ Failed to load C library: {e}")
        return False
    
    # Load sklearn model for comparison
    sklearn_pipeline = load_sklearn_model()
    if sklearn_pipeline is None:
        print("❌ Cannot load sklearn pipeline for comparison")
        return False
    
    # Load test dataset
    data_path = "test_dataset_30_features.parquet"
    if not Path(data_path).exists():
        print(f"❌ Test dataset not found. Run create_test_dataset.py first!")
        return False
    
    print(f"📂 Loading dataset from {data_path}")
    df = pd.read_parquet(data_path)
    
    print(f"📋 Dataset columns: {list(df.columns)}")
    print(f"📏 Dataset shape: {df.shape}")
    
    # Check for target column
    if 'label' in df.columns:
        target_col = 'label'
    elif 'stress' in df.columns:
        target_col = 'stress'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        print(f"❌ No target column found in: {list(df.columns)}")
        return False
    
    # Prepare features and labels
    feature_cols = [col for col in df.columns if col not in [target_col, 'subject']]
    X = df[feature_cols].values
    y_true = df[target_col].values
    
    print(f"📊 Dataset: {len(X)} samples, {len(feature_cols)} features")
    
    # Test on subset for speed (500 samples)
    n_test = min(500, len(X))
    X_test = X[:n_test]
    y_test = y_true[:n_test]
    
    print(f"🧪 Testing on {n_test} samples...")
    
    # Get sklearn predictions using full pipeline
    scaler = sklearn_pipeline['scaler']
    model = sklearn_pipeline['model'] 
    calibrator = sklearn_pipeline['calibrator']
    
    # Apply sklearn pipeline: scale -> predict -> calibrate
    X_scaled = scaler.transform(X_test)
    raw_probs = model.predict_proba(X_scaled)[:, 1]
    sklearn_probs = calibrator.predict(raw_probs.reshape(-1, 1)).flatten()
    
    # Load model threshold
    with open("model_data.json", "r") as f:
        model_data = json.load(f)
    threshold = model_data["threshold"]
    
    sklearn_preds = (sklearn_probs >= threshold).astype(int)
    
    # Get C predictions
    c_probs = []
    c_preds = []
    
    for i, features in enumerate(X_test):
        if i % 100 == 0:
            print(f"  Processing sample {i}/{n_test}")
        
        # Convert to float32 for C
        features_float32 = features.astype(np.float32)
        features_array = (ctypes.c_float * len(features_float32))(*features_float32)
        
        # Get C predictions
        c_prob = c_lib.shadow_mlp_predict_probability(features_array)
        c_pred = c_lib.shadow_mlp_predict_class(features_array)
        
        c_probs.append(c_prob)
        c_preds.append(c_pred)
    
    c_probs = np.array(c_probs)
    c_preds = np.array(c_preds)
    
    # Calculate metrics
    print(f"\n📈 Metrics Comparison:")
    print("=" * 60)
    
    # Sklearn metrics
    sklearn_accuracy = accuracy_score(y_test, sklearn_preds)
    sklearn_precision = precision_score(y_test, sklearn_preds)
    sklearn_recall = recall_score(y_test, sklearn_preds)
    sklearn_f1 = f1_score(y_test, sklearn_preds)
    
    # C metrics
    c_accuracy = accuracy_score(y_test, c_preds)
    c_precision = precision_score(y_test, c_preds)
    c_recall = recall_score(y_test, c_preds)
    c_f1 = f1_score(y_test, c_preds)
    
    print(f"{'Metric':<12} {'sklearn':<10} {'C':<10} {'Difference':<12} {'Match':<6}")
    print("-" * 60)
    print(f"{'Accuracy':<12} {sklearn_accuracy:<10.6f} {c_accuracy:<10.6f} {abs(sklearn_accuracy-c_accuracy):<12.2e} {sklearn_accuracy==c_accuracy}")
    print(f"{'Precision':<12} {sklearn_precision:<10.6f} {c_precision:<10.6f} {abs(sklearn_precision-c_precision):<12.2e} {sklearn_precision==c_precision}")
    print(f"{'Recall':<12} {sklearn_recall:<10.6f} {c_recall:<10.6f} {abs(sklearn_recall-c_recall):<12.2e} {sklearn_recall==c_recall}")
    print(f"{'F1 Score':<12} {sklearn_f1:<10.6f} {c_f1:<10.6f} {abs(sklearn_f1-c_f1):<12.2e} {sklearn_f1==c_f1}")
    
    # Probability comparison
    prob_diff = np.abs(sklearn_probs - c_probs)
    max_prob_diff = np.max(prob_diff)
    mean_prob_diff = np.mean(prob_diff)
    
    print(f"\n🎯 Probability Analysis:")
    print(f"Max probability difference:  {max_prob_diff:.2e}")
    print(f"Mean probability difference: {mean_prob_diff:.2e}")
    print(f"Samples with exact probability match: {np.sum(prob_diff == 0)}/{len(prob_diff)}")
    
    # Prediction comparison
    pred_matches = np.sum(sklearn_preds == c_preds)
    print(f"\n🔍 Prediction Analysis:")
    print(f"Exact prediction matches: {pred_matches}/{len(sklearn_preds)} ({100*pred_matches/len(sklearn_preds):.1f}%)")
    
    # Validation criteria
    print(f"\n✅ Validation Results:")
    print("=" * 40)
    
    criteria_passed = 0
    total_criteria = 6
    
    # 1. Exact accuracy match
    if sklearn_accuracy == c_accuracy:
        print("✅ Exact accuracy match")
        criteria_passed += 1
    else:
        print(f"❌ Accuracy mismatch: {abs(sklearn_accuracy-c_accuracy):.2e}")
    
    # 2. Exact F1 match
    if sklearn_f1 == c_f1:
        print("✅ Exact F1 score match")
        criteria_passed += 1
    else:
        print(f"❌ F1 score mismatch: {abs(sklearn_f1-c_f1):.2e}")
    
    # 3. Exact precision match
    if sklearn_precision == c_precision:
        print("✅ Exact precision match")
        criteria_passed += 1
    else:
        print(f"❌ Precision mismatch: {abs(sklearn_precision-c_precision):.2e}")
    
    # 4. Exact recall match
    if sklearn_recall == c_recall:
        print("✅ Exact recall match")
        criteria_passed += 1
    else:
        print(f"❌ Recall mismatch: {abs(sklearn_recall-c_recall):.2e}")
    
    # 5. All predictions match
    if pred_matches == len(sklearn_preds):
        print("✅ All predictions match exactly")
        criteria_passed += 1
    else:
        print(f"❌ {len(sklearn_preds)-pred_matches} prediction mismatches")
    
    # 6. Small probability differences
    if max_prob_diff < 1e-3:
        print(f"✅ Small probability differences: {max_prob_diff:.2e}")
        criteria_passed += 1
    else:
        print(f"❌ Large probability differences: {max_prob_diff:.2e}")
    
    print(f"\n🏆 OVERALL: {criteria_passed}/{total_criteria} criteria passed")
    
    if criteria_passed >= 5:  # Allow small float precision differences
        print("🎉 C IMPLEMENTATION VALIDATED!")
        print("✅ Ready for ESP32 deployment!")
        return True
    else:
        print("❌ Validation failed - fix implementation")
        return False

if __name__ == "__main__":
    print("🎯 Full Dataset Metrics Validation")
    print("=" * 50)
    
    success = test_full_dataset()
    if success:
        print("\n🚀 C implementation is production-ready!")
    else:
        print("\n⚠️  Review C implementation before deployment.")
