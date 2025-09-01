#!/usr/bin/env python3
"""
Stage 4 - Step 1: Extract Model Data
Simple script to extract and understand our trained model components.
"""

import json
import joblib
import numpy as np
from pathlib import Path

def main():
    print("🔍 Step 1: Extracting Model Data")
    print("=" * 50)
    
    # Load our trained model
    stage2_dir = Path("../outputs/stage2_model_exploration")
    
    print("📂 Loading model artifacts...")
    model = joblib.load(stage2_dir / "final_model.joblib")
    scaler = joblib.load(stage2_dir / "final_scaler.joblib")
    calibrator = joblib.load(stage2_dir / "final_calibrator.joblib")
    
    with open(stage2_dir / "final_model_artifacts.json") as f:
        metadata = json.load(f)
    
    print(f"✅ Model type: {metadata['model_type']}")
    print(f"✅ Features: {len(metadata['features'])}")
    print(f"✅ Threshold: {metadata['optimal_threshold']:.4f}")
    
    # Examine MLP structure
    print("\n🧠 MLP Architecture:")
    print(f"   Hidden layers: {model.hidden_layer_sizes}")
    print(f"   Total layers: {model.n_layers_}")
    
    print("\n📊 Layer Details:")
    for i, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
        print(f"   Layer {i}: {weights.shape[0]} → {weights.shape[1]} (params: {weights.size + biases.size})")
    
    total_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    print(f"   Total parameters: {total_params:,}")
    
    # Examine scaler
    print(f"\n⚖️  Scaler: StandardScaler")
    print(f"   Features: {len(scaler.mean_)}")
    print(f"   Mean range: {scaler.mean_.min():.3f} to {scaler.mean_.max():.3f}")
    print(f"   Scale range: {scaler.scale_.min():.6f} to {scaler.scale_.max():.3f}")
    
    # Examine calibrator
    print(f"\n📈 Calibrator: {type(calibrator).__name__}")
    if hasattr(calibrator, 'X_thresholds_'):
        print(f"   Calibration points: {len(calibrator.X_thresholds_)}")
        print(f"   Input range: {calibrator.X_thresholds_.min():.3f} to {calibrator.X_thresholds_.max():.3f}")
    
    # Save readable model summary
    summary = {
        "model_info": {
            "type": "MLPClassifier",
            "architecture": list(model.hidden_layer_sizes),
            "n_layers": model.n_layers_,
            "total_parameters": total_params
        },
        "layer_shapes": [
            {"layer": i, "input_size": w.shape[0], "output_size": w.shape[1], "weights": w.size, "biases": b.size}
            for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_))
        ],
        "scaler_info": {
            "n_features": len(scaler.mean_),
            "mean_stats": {"min": float(scaler.mean_.min()), "max": float(scaler.mean_.max())},
            "scale_stats": {"min": float(scaler.scale_.min()), "max": float(scaler.scale_.max())}
        },
        "calibrator_info": {
            "type": type(calibrator).__name__,
            "n_points": len(calibrator.X_thresholds_) if hasattr(calibrator, 'X_thresholds_') else 0
        },
        "features": metadata['features'],
        "threshold": metadata['optimal_threshold']
    }
    
    with open("model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Saved model summary to: model_summary.json")
    print(f"📁 Ready for next step!")

if __name__ == "__main__":
    main()
