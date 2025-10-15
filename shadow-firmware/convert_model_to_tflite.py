#!/usr/bin/env python3
"""
PyTorch to TFLite Model Converter
==================================

Converts the trained PyTorch stress detection model (best.pth) to TensorFlow Lite
format for deployment on ESP32-S3 microcontroller.

Pipeline:
    1. Load PyTorch model from best.pth
    2. Export to ONNX format
    3. Convert ONNX to TensorFlow SavedModel
    4. Convert TensorFlow to TFLite with INT8 quantization
    5. Generate C header file for embedding in firmware

Model Architecture:
    Input: (batch, 4, 240) - [ACC_MAG, BVP, EDA, TEMP] normalized signals
    Output: (batch, 1) - Stress probability [0.0-1.0]

Requirements:
    - torch
    - onnx
    - onnx2tf
    - tensorflow
    - numpy

Usage:
    python3 convert_model_to_tflite.py
"""

import os
import sys
import torch
import numpy as np
import onnx
import tensorflow as tf
from pathlib import Path

# Add parent directory to path to import model architecture
sys.path.append(str(Path(__file__).parent.parent / "model"))

# Configuration
MODEL_PATH = Path(__file__).parent / "best.pth"
OUTPUT_DIR = Path(__file__).parent / "model_output"
ONNX_PATH = OUTPUT_DIR / "stress_model.onnx"
TF_MODEL_PATH = OUTPUT_DIR / "tf_model"
TFLITE_PATH = OUTPUT_DIR / "stress_model.tflite"
TFLITE_QUANT_PATH = OUTPUT_DIR / "stress_model_quant.tflite"
C_HEADER_PATH = Path(__file__).parent / "components/cnn_inference/include/stress_model_data.h"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Model input shape
INPUT_SHAPE = (1, 4, 240)  # (batch, channels, samples)


def load_pytorch_model():
    """Load the trained PyTorch model."""
    print("=" * 80)
    print("Step 1: Loading PyTorch Model")
    print("=" * 80)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    print(f"Loading model from: {MODEL_PATH}")
    
    # Load with weights_only=False to handle numpy arrays
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Try to determine model architecture
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"\nModel layers:")
    for key in state_dict.keys():
        print(f"  {key}: {state_dict[key].shape}")
    
    # Dynamically reconstruct model architecture from state dict
    model = reconstruct_model_from_state_dict(state_dict)
    model.eval()
    
    print(f"\n✅ Model loaded successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model


def reconstruct_model_from_state_dict(state_dict):
    """Reconstruct the CNN model architecture from state dict."""
    import torch.nn as nn
    
    class StressDetectionCNN(nn.Module):
        def __init__(self):
            super(StressDetectionCNN, self).__init__()
            
            # Shared convolutional layers
            # Based on state dict: kernel_size=10, BatchNorm1d, 2 conv blocks
            self.shared_conv = nn.Sequential(
                nn.Conv1d(in_channels=4, out_channels=64, kernel_size=10, padding=4),  # 0
                nn.BatchNorm1d(64),                                                      # 1
                nn.ReLU(),                                                               # 2
                nn.Dropout(0.5),                                                         # 3
                nn.MaxPool1d(kernel_size=2),                                             # 4
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, padding=4), # 5
                nn.BatchNorm1d(128),                                                     # 6
                nn.ReLU(),                                                               # 7
                nn.Dropout(0.5),                                                         # 8
                nn.MaxPool1d(kernel_size=2),                                             # 9
            )
            
            # Global Average Pooling - reduces (batch, 128, time) to (batch, 128)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            
            # Shared fully connected layers
            # Input is 128 (from global pooling)
            self.shared_fc = nn.Sequential(
                nn.Dropout(0.5),                    # 0
                nn.Linear(128, 128),                # 1
                nn.ReLU(),                          # 2
            )
            
            # Universal-private layer
            self.universal_private = nn.Sequential(
                nn.Linear(128, 64),      # 0
                nn.ReLU(),                # 1
                nn.Dropout(0.5),          # 2
                nn.Linear(64, 1),         # 3
                nn.Sigmoid()              # 4
            )
        
        def forward(self, x):
            # x shape: (batch, channels, samples) = (batch, 4, 240)
            x = self.shared_conv(x)            # (batch, 128, time)
            x = self.global_pool(x)            # (batch, 128, 1)
            x = x.view(x.size(0), -1)          # (batch, 128)
            x = self.shared_fc(x)              # (batch, 128)
            x = self.universal_private(x)      # (batch, 1)
            return x
    
    # Create model and load state dict
    model = StressDetectionCNN()
    model.load_state_dict(state_dict)
    return model


def export_to_onnx(model):
    """Export PyTorch model to ONNX format."""
    print("\n" + "=" * 80)
    print("Step 2: Exporting to ONNX")
    print("=" * 80)
    
    # Create dummy input
    dummy_input = torch.randn(INPUT_SHAPE)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output path: {ONNX_PATH}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    
    print(f"\n✅ ONNX export successful")
    print(f"   File size: {ONNX_PATH.stat().st_size / 1024:.2f} KB")


def convert_onnx_to_tensorflow():
    """Convert ONNX model to TensorFlow SavedModel."""
    print("\n" + "=" * 80)
    print("Step 3: Converting ONNX to TensorFlow")
    print("=" * 80)
    
    import onnx2tf
    
    print(f"Input: {ONNX_PATH}")
    print(f"Output: {TF_MODEL_PATH}")
    
    # Convert using onnx2tf
    onnx2tf.convert(
        input_onnx_file_path=str(ONNX_PATH),
        output_folder_path=str(TF_MODEL_PATH),
        copy_onnx_input_output_names_to_tflite=True,
    )
    
    print(f"\n✅ TensorFlow conversion successful")


def convert_tensorflow_to_tflite():
    """Convert TensorFlow SavedModel to TFLite (float32 and int8 quantized)."""
    print("\n" + "=" * 80)
    print("Step 4: Converting TensorFlow to TFLite")
    print("=" * 80)
    
    # Load TensorFlow model
    converter = tf.lite.TFLiteConverter.from_saved_model(str(TF_MODEL_PATH))
    
    # Convert to float32 TFLite (baseline)
    print("\n4a. Converting to Float32 TFLite...")
    tflite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Float32 TFLite saved: {TFLITE_PATH}")
    print(f"   File size: {TFLITE_PATH.stat().st_size / 1024:.2f} KB")
    
    # Convert to INT8 quantized TFLite
    print("\n4b. Converting to INT8 Quantized TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(TF_MODEL_PATH))
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for quantization calibration
    def representative_dataset():
        for _ in range(100):
            # Generate random normalized signals
            data = np.random.randn(1, 4, 240).astype(np.float32)
            yield [data]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # FULL INT8 quantization (required for TFLite Micro)
    converter.inference_input_type = tf.int8  # INT8 input for embedded deployment
    converter.inference_output_type = tf.int8  # INT8 output for embedded deployment
    
    tflite_quant_model = converter.convert()
    with open(TFLITE_QUANT_PATH, 'wb') as f:
        f.write(tflite_quant_model)
    
    print(f"✅ INT8 Quantized TFLite saved: {TFLITE_QUANT_PATH}")
    print(f"   File size: {TFLITE_QUANT_PATH.stat().st_size / 1024:.2f} KB")
    
    # Calculate compression ratio
    float_size = TFLITE_PATH.stat().st_size
    quant_size = TFLITE_QUANT_PATH.stat().st_size
    compression = (1 - quant_size / float_size) * 100
    print(f"\n   Compression: {compression:.1f}% smaller")


def generate_c_header():
    """Generate C header file with embedded TFLite model."""
    print("\n" + "=" * 80)
    print("Step 5: Generating C Header File")
    print("=" * 80)
    
    # Read quantized TFLite model
    with open(TFLITE_QUANT_PATH, 'rb') as f:
        model_data = f.read()
    
    # Create C header directory
    C_HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate header content
    header_content = f"""/**
 * @file stress_model_data.h
 * @brief Embedded TensorFlow Lite model for stress detection
 * 
 * Auto-generated from: {MODEL_PATH.name}
 * Model size: {len(model_data)} bytes ({len(model_data) / 1024:.2f} KB)
 * 
 * Input shape: (1, 4, 240) - [ACC_MAG, BVP, EDA, TEMP]
 * Output shape: (1, 1) - Stress probability [0.0-1.0]
 * 
 * Quantization: INT8 (weights and activations)
 * Input/Output type: FLOAT32
 * 
 * Generated by: convert_model_to_tflite.py
 */

#ifndef STRESS_MODEL_DATA_H
#define STRESS_MODEL_DATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

// Model metadata
#define STRESS_MODEL_SIZE {len(model_data)}
#define STRESS_MODEL_INPUT_CHANNELS 4
#define STRESS_MODEL_INPUT_SAMPLES 240
#define STRESS_MODEL_OUTPUT_SIZE 1

// Embedded model data
extern const unsigned char g_stress_model_data[];
extern const unsigned int g_stress_model_data_len;

#ifdef __cplusplus
}}
#endif

#endif // STRESS_MODEL_DATA_H
"""
    
    # Write header file
    with open(C_HEADER_PATH, 'w') as f:
        f.write(header_content)
    
    # Generate C source file with model data
    c_source_path = C_HEADER_PATH.parent.parent / "stress_model_data.c"
    
    with open(c_source_path, 'w') as f:
        f.write('#include "stress_model_data.h"\n\n')
        f.write('// Embedded TensorFlow Lite model data\n')
        f.write('const unsigned char g_stress_model_data[] = {\n')
        
        # Write model bytes (16 per line)
        for i in range(0, len(model_data), 16):
            chunk = model_data[i:i+16]
            hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
            f.write(f'  {hex_values},\n')
        
        f.write('};\n\n')
        f.write(f'const unsigned int g_stress_model_data_len = {len(model_data)};\n')
    
    print(f"✅ C header generated: {C_HEADER_PATH}")
    print(f"✅ C source generated: {c_source_path}")
    print(f"   Model size: {len(model_data)} bytes ({len(model_data) / 1024:.2f} KB)")


def validate_conversion(pytorch_model):
    """Validate TFLite model against PyTorch model."""
    print("\n" + "=" * 80)
    print("Step 6: Validating Conversion")
    print("=" * 80)
    
    # Create test input
    test_input = np.random.randn(1, 4, 240).astype(np.float32)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(torch.from_numpy(test_input)).numpy()
    
    # TFLite inference (float32)
    interpreter_float = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter_float.allocate_tensors()
    
    input_details = interpreter_float.get_input_details()
    output_details = interpreter_float.get_output_details()
    
    interpreter_float.set_tensor(input_details[0]['index'], test_input)
    interpreter_float.invoke()
    tflite_float_output = interpreter_float.get_tensor(output_details[0]['index'])
    
    # TFLite inference (int8 quantized)
    interpreter_quant = tf.lite.Interpreter(model_path=str(TFLITE_QUANT_PATH))
    interpreter_quant.allocate_tensors()
    
    input_details = interpreter_quant.get_input_details()
    output_details = interpreter_quant.get_output_details()
    
    interpreter_quant.set_tensor(input_details[0]['index'], test_input)
    interpreter_quant.invoke()
    tflite_quant_output = interpreter_quant.get_tensor(output_details[0]['index'])
    
    # Calculate errors
    float_error = np.abs(pytorch_output - tflite_float_output).mean()
    quant_error = np.abs(pytorch_output - tflite_quant_output).mean()
    
    print(f"\nValidation Results:")
    print(f"  PyTorch output:      {pytorch_output[0, 0]:.6f}")
    print(f"  TFLite Float output: {tflite_float_output[0, 0]:.6f}")
    print(f"  TFLite Quant output: {tflite_quant_output[0, 0]:.6f}")
    print(f"\n  Float32 error: {float_error:.6f} ({float_error * 100:.3f}%)")
    print(f"  INT8 error:    {quant_error:.6f} ({quant_error * 100:.3f}%)")
    
    # Validate acceptable error threshold
    if float_error < 0.01 and quant_error < 0.05:
        print(f"\n✅ Validation PASSED - Errors within acceptable range")
    else:
        print(f"\n⚠️  WARNING - Errors may be too high")
        print(f"   Expected: Float < 1%, Quant < 5%")


def main():
    """Main conversion pipeline."""
    print("=" * 80)
    print("PyTorch to TFLite Model Converter")
    print("=" * 80)
    print(f"\nInput model:  {MODEL_PATH}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print(f"C header:     {C_HEADER_PATH}")
    
    try:
        # Step 1: Load PyTorch model
        pytorch_model = load_pytorch_model()
        
        # Step 2: Export to ONNX
        export_to_onnx(pytorch_model)
        
        # Step 3: Convert ONNX to TensorFlow
        convert_onnx_to_tensorflow()
        
        # Step 4: Convert TensorFlow to TFLite
        convert_tensorflow_to_tflite()
        
        # Step 5: Generate C header
        generate_c_header()
        
        # Step 6: Validate conversion
        validate_conversion(pytorch_model)
        
        print("\n" + "=" * 80)
        print("✅ CONVERSION COMPLETE")
        print("=" * 80)
        print(f"\nGenerated files:")
        print(f"  ONNX:         {ONNX_PATH}")
        print(f"  TensorFlow:   {TF_MODEL_PATH}")
        print(f"  TFLite:       {TFLITE_PATH}")
        print(f"  TFLite Quant: {TFLITE_QUANT_PATH}")
        print(f"  C Header:     {C_HEADER_PATH}")
        print(f"\nNext steps:")
        print(f"  1. Add cnn_inference component to ESP-IDF project")
        print(f"  2. Implement TFLite interpreter in C")
        print(f"  3. Integrate with signal preprocessor")
        print(f"  4. Test on ESP32-S3 hardware")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
