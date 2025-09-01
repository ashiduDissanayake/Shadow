#!/usr/bin/env python3
"""
Extract exact weights, biases, scaler, and calibration data for C implementation
"""

import json
import joblib
import numpy as np
from pathlib import Path

def extract_model_data():
    """Extract all model components with exact precision"""
    stage2_dir = Path("../outputs/stage2_model_exploration")
    
    # Load components
    model = joblib.load(stage2_dir / "final_model.joblib")
    scaler = joblib.load(stage2_dir / "final_scaler.joblib")
    calibrator = joblib.load(stage2_dir / "final_calibrator.joblib")
    
    with open(stage2_dir / "final_model_artifacts.json") as f:
        metadata = json.load(f)
    
    # Extract weights and biases (exact sklearn layout)
    weights = []
    biases = []
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        print(f"Layer {i}: {W.shape} -> {b.shape}")
        weights.append(W.astype(np.float32))
        biases.append(b.astype(np.float32))
    
    # Extract scaler parameters
    feature_means = scaler.mean_.astype(np.float32)
    feature_scales = scaler.scale_.astype(np.float32)
    
    # Extract calibration knots (isotonic regression)
    calib_x = calibrator.X_thresholds_.astype(np.float32)
    
    # Get corresponding y values by evaluating calibrator
    calib_y = []
    for x in calib_x:
        y = calibrator.transform(np.array([x]))[0]
        calib_y.append(float(y))
    calib_y = np.array(calib_y, dtype=np.float32)
    
    print(f"Calibration points: {len(calib_x)}")
    print(f"X range: {calib_x.min():.6f} to {calib_x.max():.6f}")
    print(f"Y range: {calib_y.min():.6f} to {calib_y.max():.6f}")
    
    # Create C data structure
    data = {
        'features': metadata['features'],
        'threshold': float(metadata['optimal_threshold']),
        'layer_sizes': [30, 64, 32, 1],
        'weights': [w.tolist() for w in weights],
        'biases': [b.tolist() for b in biases],
        'scaler_means': feature_means.tolist(),
        'scaler_scales': feature_scales.tolist(),
        'calibration_x': calib_x.tolist(),
        'calibration_y': calib_y.tolist()
    }
    
    # Save for C code generation
    with open("model_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved model data to model_data.json")
    return data

if __name__ == "__main__":
    extract_model_data()
