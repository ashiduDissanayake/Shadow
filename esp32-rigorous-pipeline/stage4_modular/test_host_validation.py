#!/usr/bin/env python3
"""
Simple host validation test for C implementation
Just test that our C code works correctly on the host machine
"""

import json
import ctypes
import subprocess
import os

def compile_c_model():
    """Compile the C model for host testing"""
    print("🔨 Compiling C model for host...")
    
    result = subprocess.run([
        "gcc", "-shared", "-fPIC", "-O2", "-lm",
        "components/simple_mlp.c", "-o", "simple_mlp.so"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Compilation failed:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    print("✅ Compilation successful")
    return True

def load_c_library():
    """Load the compiled C library"""
    lib_path = "./simple_mlp.so"
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found: {lib_path}")
    
    lib = ctypes.CDLL(lib_path)
    
    # Set up function signatures
    lib.shadow_mlp_predict_probability.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.shadow_mlp_predict_probability.restype = ctypes.c_float
    
    lib.shadow_mlp_predict_class.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.shadow_mlp_predict_class.restype = ctypes.c_int
    
    return lib

def test_basic_functionality():
    """Test basic C implementation functionality"""
    
    # Step 1: Compile
    if not compile_c_model():
        return False
    
    # Step 2: Load library
    try:
        c_lib = load_c_library()
        print("✅ C library loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load C library: {e}")
        return False
    
    # Step 3: Load test case
    try:
        with open("test_case.json", "r") as f:
            test_case = json.load(f)
        print("✅ Test case loaded")
    except Exception as e:
        print(f"❌ Failed to load test case: {e}")
        return False
    
    # Step 4: Test predictions
    features = test_case["features"]
    expected_prob = test_case["expected_probability"]
    expected_class = test_case["expected_class"]
    
    print(f"\n🧪 Running test...")
    print(f"Features: {len(features)} values")
    print(f"Expected: prob={expected_prob:.6f}, class={expected_class}")
    
    # Make predictions
    features_array = (ctypes.c_float * len(features))(*features)
    
    c_prob = c_lib.shadow_mlp_predict_probability(features_array)
    c_class = c_lib.shadow_mlp_predict_class(features_array)
    
    print(f"C Result: prob={c_prob:.6f}, class={c_class}")
    
    # Check results
    prob_diff = abs(c_prob - expected_prob)
    class_match = (c_class == expected_class)
    
    print(f"\n📊 Validation:")
    print(f"Probability difference: {prob_diff:.2e}")
    print(f"Class prediction correct: {class_match}")
    
    # Simple pass/fail criteria
    prob_ok = prob_diff < 1e-3  # Allow some float precision difference
    
    if prob_ok and class_match:
        print("🎉 HOST VALIDATION PASSED!")
        print("✅ C implementation works correctly on host machine")
        return True
    else:
        print("❌ HOST VALIDATION FAILED!")
        if not prob_ok:
            print(f"  - Probability difference too large: {prob_diff:.2e}")
        if not class_match:
            print(f"  - Class mismatch: got {c_class}, expected {expected_class}")
        return False

if __name__ == "__main__":
    print("🚀 Testing C implementation on host machine...")
    print("=" * 50)
    
    success = test_basic_functionality()
    
    if success:
        print("\n✅ READY FOR NEXT STEP!")
        print("The C implementation works correctly on the host.")
        print("Now we can think about ESP32 adaptation.")
    else:
        print("\n❌ FIX NEEDED!")
        print("C implementation has issues on host machine.")
        print("Must fix before considering ESP32.")
