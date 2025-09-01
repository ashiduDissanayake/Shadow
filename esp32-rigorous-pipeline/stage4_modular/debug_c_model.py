#!/usr/bin/env python3
"""
Simple test to debug the C implementation
"""

import numpy as np
import ctypes
import subprocess
import os

def compile_simple_test():
    """Create a minimal test program"""
    
    # Create a simple test C file
    test_c = """
#include <stdio.h>
#include <math.h>

// Simple test functions
float test_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float test_add(float a, float b) {
    return a + b;
}

int main() {
    printf("Testing basic functions...\\n");
    printf("sigmoid(0) = %f\\n", test_sigmoid(0.0f));
    printf("sigmoid(1) = %f\\n", test_sigmoid(1.0f));
    printf("add(2, 3) = %f\\n", test_add(2.0f, 3.0f));
    return 0;
}
"""
    
    with open("test_simple.c", "w") as f:
        f.write(test_c)
    
    # Compile as executable
    cmd = ["gcc", "-o", "test_simple", "test_simple.c", "-lm"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Compilation failed: {result.stderr}")
        return False
    
    # Run the test
    result = subprocess.run(["./test_simple"], capture_output=True, text=True)
    print("Basic test output:")
    print(result.stdout)
    
    return True

def debug_c_model():
    """Debug our C model step by step"""
    print("🔍 Debugging C Model Implementation")
    print("=" * 50)
    
    # Test 1: Basic compilation
    print("Step 1: Testing basic C compilation...")
    if not compile_simple_test():
        return False
    
    # Test 2: Check if our model compiles at all
    print("\nStep 2: Testing model compilation...")
    cmd = ["gcc", "-c", "components/simple_mlp.c", "-o", "simple_mlp.o"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Model compilation failed:")
        print(result.stderr)
        return False
    else:
        print("✅ Model compiles successfully")
    
    # Test 3: Check size of compiled object
    if os.path.exists("simple_mlp.o"):
        size = os.path.getsize("simple_mlp.o")
        print(f"Compiled object size: {size:,} bytes ({size/1024:.1f} KB)")
        
        if size > 1024 * 1024:  # > 1MB
            print("⚠️  Object file is very large - may cause stack issues")
    
    # Test 4: Try to create a minimal wrapper
    wrapper_c = '''
#include "components/simple_mlp.h"
#include <stdio.h>

int main() {
    printf("Testing minimal wrapper...\\n");
    
    // Test with simple input
    float features[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) {
        features[i] = 1.0f;  // Simple test input
    }
    
    printf("Calling shadow_mlp_predict_probability...\\n");
    float prob = shadow_mlp_predict_probability(features);
    printf("Result: %f\\n", prob);
    
    return 0;
}
'''
    
    with open("test_wrapper.c", "w") as f:
        f.write(wrapper_c)
    
    print("\nStep 3: Testing minimal wrapper...")
    cmd = ["gcc", "-o", "test_wrapper", "test_wrapper.c", "simple_mlp.o", "-lm"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Wrapper compilation failed:")
        print(result.stderr)
        return False
    
    # Run the wrapper
    print("Running wrapper test...")
    result = subprocess.run(["./test_wrapper"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Wrapper execution failed:")
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return False
    else:
        print("✅ Wrapper executed successfully:")
        print(result.stdout)
        return True

if __name__ == "__main__":
    debug_c_model()
