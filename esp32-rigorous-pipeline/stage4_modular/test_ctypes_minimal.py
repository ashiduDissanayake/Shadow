#!/usr/bin/env python3
"""
Minimal test to debug ctypes interface
"""

import ctypes
import numpy as np
import subprocess

def test_ctypes_interface():
    """Test ctypes interface step by step"""
    print("🔧 Testing ctypes interface...")
    
    # Compile as shared library 
    cmd = ["gcc", "-shared", "-fPIC", "-O0", "-g",  # Add debug info
           "-o", "components/libsimple_mlp.so",
           "components/simple_mlp.c", "-lm"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Shared library compilation failed: {result.stderr}")
        return False
    
    print("✅ Shared library compiled")
    
    # Load library
    try:
        lib = ctypes.CDLL("./components/libsimple_mlp.so")
        print("✅ Library loaded")
    except Exception as e:
        print(f"❌ Failed to load library: {e}")
        return False
    
    # Define simple function first
    try:
        lib.sigmoid.argtypes = [ctypes.c_float]
        lib.sigmoid.restype = ctypes.c_float
        
        # Test sigmoid
        result = lib.sigmoid(0.0)
        print(f"✅ sigmoid(0.0) = {result}")
        
        result = lib.sigmoid(1.0)
        print(f"✅ sigmoid(1.0) = {result}")
        
    except Exception as e:
        print(f"❌ Sigmoid test failed: {e}")
        return False
    
    # Test prediction function with minimal input
    try:
        lib.shadow_mlp_predict_probability.argtypes = [ctypes.POINTER(ctypes.c_float)]
        lib.shadow_mlp_predict_probability.restype = ctypes.c_float
        
        # Create minimal test input
        features = np.ones(30, dtype=np.float32)
        c_features = (ctypes.c_float * 30)(*features)
        
        print("Calling prediction function...")
        result = lib.shadow_mlp_predict_probability(c_features)
        print(f"✅ Prediction result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

if __name__ == "__main__":
    test_ctypes_interface()
