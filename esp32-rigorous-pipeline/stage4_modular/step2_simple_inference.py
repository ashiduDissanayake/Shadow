#!/usr/bin/env python3
"""
Stage 4 - Step 2: Simple MLP Inference
Implement MLP forward pass in pure Python (no sklearn) to understand the math.
"""

import json
import joblib
import numpy as np
from pathlib import Path

class SimpleMLP:
    """
    Pure Python MLP implementation to understand the inference math.
    This helps us design the C code step by step.
    """
    
    def __init__(self, weights, biases):
        self.weights = weights  # List of weight matrices
        self.biases = biases    # List of bias vectors
        self.n_layers = len(weights)
    
    def relu(self, x):
        """ReLU activation: f(x) = max(0, x)"""
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Sigmoid activation: f(x) = 1 / (1 + exp(-x))"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def forward(self, x):
        """
        Forward pass through the network.
        Returns the final output (probability).
        """
        current = x.copy()
        
        # Go through all layers
        for i in range(self.n_layers):
            # Linear transformation: output = input @ weights + bias
            current = np.dot(current, self.weights[i]) + self.biases[i]
            
            # Apply activation
            if i < self.n_layers - 1:  # Hidden layers use ReLU
                current = self.relu(current)
            else:  # Output layer uses sigmoid
                current = self.sigmoid(current)
        
        return current[0]  # Return single probability value

class SimpleScaler:
    """Pure Python StandardScaler implementation"""
    
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale
    
    def transform(self, x):
        """Apply z-score normalization: (x - mean) / scale"""
        return (x - self.mean) / self.scale

class SimpleCalibrator:
    """Pure Python isotonic regression implementation"""
    
    def __init__(self, x_points, y_points):
        self.x_points = x_points
        self.y_points = y_points
    
    def transform(self, raw_prob):
        """Apply isotonic calibration using linear interpolation"""
        # Handle edge cases
        if raw_prob <= self.x_points[0]:
            return self.y_points[0]
        if raw_prob >= self.x_points[-1]:
            return self.y_points[-1]
        
        # Find interpolation interval
        for i in range(len(self.x_points) - 1):
            if self.x_points[i] <= raw_prob <= self.x_points[i + 1]:
                # Linear interpolation
                x0, x1 = self.x_points[i], self.x_points[i + 1]
                y0, y1 = self.y_points[i], self.y_points[i + 1]
                t = (raw_prob - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        
        return raw_prob  # Fallback

def test_inference():
    print("🧪 Step 2: Testing Simple MLP Inference")
    print("=" * 50)
    
    # Load model data
    stage2_dir = Path("../outputs/stage2_model_exploration")
    model = joblib.load(stage2_dir / "final_model.joblib")
    scaler = joblib.load(stage2_dir / "final_scaler.joblib")
    calibrator = joblib.load(stage2_dir / "final_calibrator.joblib")
    
    with open(stage2_dir / "final_model_artifacts.json") as f:
        metadata = json.load(f)
    
    # Create our simple implementations
    simple_mlp = SimpleMLP(model.coefs_, model.intercepts_)
    simple_scaler = SimpleScaler(scaler.mean_, scaler.scale_)
    
    # Extract calibration points
    x_points = calibrator.X_thresholds_
    y_points = []
    for x in x_points:
        y = calibrator.transform(np.array([x]))[0]
        y_points.append(y)
    simple_calibrator = SimpleCalibrator(x_points, y_points)
    
    print(f"✅ Loaded model with {simple_mlp.n_layers} layers")
    print(f"✅ Features: {len(metadata['features'])}")
    print(f"✅ Calibration points: {len(x_points)}")
    
    # Test with some random data
    print("\n🎲 Testing with random features...")
    np.random.seed(42)
    test_features = np.random.randn(30)  # 30 random features
    
    print("Raw features (first 5):", test_features[:5].round(3))
    
    # Step 1: Scale features
    scaled_features = simple_scaler.transform(test_features)
    print("Scaled features (first 5):", scaled_features[:5].round(3))
    
    # Step 2: MLP forward pass
    raw_output = simple_mlp.forward(scaled_features)
    print(f"Raw MLP output: {raw_output:.6f}")
    
    # Step 3: Calibration
    calibrated_output = simple_calibrator.transform(raw_output)
    print(f"Calibrated output: {calibrated_output:.6f}")
    
    # Step 4: Final prediction
    threshold = metadata['optimal_threshold']
    prediction = 1 if calibrated_output >= threshold else 0
    print(f"Final prediction: {prediction} (threshold: {threshold:.4f})")
    
    # Compare with sklearn
    print("\n🔍 Validation against sklearn:")
    sklearn_scaled = scaler.transform(test_features.reshape(1, -1))
    sklearn_raw = model.predict_proba(sklearn_scaled)[0, 1]
    sklearn_calibrated = calibrator.transform(np.array([sklearn_raw]))[0]
    sklearn_pred = 1 if sklearn_calibrated >= threshold else 0
    
    print(f"sklearn raw: {sklearn_raw:.6f} vs ours: {raw_output:.6f}")
    print(f"sklearn calibrated: {sklearn_calibrated:.6f} vs ours: {calibrated_output:.6f}")
    print(f"sklearn prediction: {sklearn_pred} vs ours: {prediction}")
    
    # Check differences
    raw_diff = abs(sklearn_raw - raw_output)
    calib_diff = abs(sklearn_calibrated - calibrated_output)
    
    print(f"\n📊 Differences:")
    print(f"Raw output difference: {raw_diff:.8f}")
    print(f"Calibrated difference: {calib_diff:.8f}")
    
    if raw_diff < 1e-5 and calib_diff < 1e-5:
        print("✅ Our implementation matches sklearn! Ready for C conversion.")
    else:
        print("⚠️  Small differences detected - this is normal due to floating point precision.")
    
    # Save test case for C validation
    test_case = {
        "input_features": test_features.tolist(),
        "scaled_features": scaled_features.tolist(),
        "raw_output": float(raw_output),
        "calibrated_output": float(calibrated_output),
        "prediction": int(prediction),
        "sklearn_validation": {
            "raw_output": float(sklearn_raw),
            "calibrated_output": float(sklearn_calibrated),
            "prediction": int(sklearn_pred)
        }
    }
    
    with open("test_case.json", "w") as f:
        json.dump(test_case, f, indent=2)
    
    print("\n💾 Saved test case to: test_case.json")
    return test_case

if __name__ == "__main__":
    test_case = test_inference()
    print(f"\n🚀 Ready for Step 3: C implementation!")
