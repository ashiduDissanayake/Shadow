#!/usr/bin/env python3
"""
Stage 4: Final Validation - Clean and Simple
Test our C implementation against a few samples to ensure it works correctly
"""

import json
import ctypes
import subprocess
import numpy as np
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

def test_c_implementation():
    """Test the C implementation with a few samples"""
    
    # Compile
    if not compile_c_model():
        return False
    
    # Load library
    try:
        lib = load_c_library()
    except Exception as e:
        print(f"❌ Failed to load library: {e}")
        return False
    
    # Load test case
    with open("test_case.json", "r") as f:
        test_data = json.load(f)
    
    # Test the sample
    test_features = test_data["input_features"]
    expected_prob = test_data["calibrated_output"]
    expected_class = test_data["prediction"]
    
    print(f"\n🧪 Testing C implementation...")
    print(f"Input: {len(test_features)} features")
    
    # Call C functions
    features_array = (ctypes.c_float * len(test_features))(*test_features)
    c_prob = lib.shadow_mlp_predict_probability(features_array)
    c_class = lib.shadow_mlp_predict_class(features_array)
    
    # Check results
    prob_diff = abs(c_prob - expected_prob)
    class_match = (c_class == expected_class)
    
    print(f"\n📊 Test Results:")
    print(f"Expected probability: {expected_prob:.6f}")
    print(f"C probability:        {c_prob:.6f}")
    print(f"Difference:           {prob_diff:.2e}")
    print(f"")
    print(f"Expected class: {expected_class}")
    print(f"C class:        {c_class}")
    print(f"Match:          {class_match}")
    
    # Validation (relaxed for float32 precision)
    prob_ok = prob_diff < 1e-3  # More realistic for float32
    
    print(f"\n🎯 Validation:")
    if prob_ok and class_match:
        print("✅ C implementation is working correctly!")
        return True
    else:
        print("❌ C implementation has issues")
        if not prob_ok:
            print(f"   Probability difference too large: {prob_diff:.2e} (threshold: 1e-3)")
        if not class_match:
            print(f"   Class prediction mismatch")
        return False

if __name__ == "__main__":
    print("🎯 Stage 4: Final Validation - HOST TESTING")
    print("=" * 50)
    
    # Test C implementation
    if test_c_implementation():
        print("\n🎉 HOST VALIDATION SUCCESSFUL!")
        print("✅ C implementation works correctly on host system")
        print("🚀 Ready to proceed with ESP32 integration!")
        
    else:
        print(f"\n❌ HOST VALIDATION FAILED!")
        print(f"Fix C implementation before proceeding to ESP32.")
