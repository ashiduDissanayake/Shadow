#!/usr/bin/env python3
"""
Simplified PyTorch to TFLite Converter
======================================

Uses ai_edge_torch (Google's recommended tool) to convert PyTorch models directly to TFLite
without going through ONNX/TensorFlow.

Requirements:
    - torch
    - ai-edge-torch
    - numpy

Usage:
    python3 convert_model_simple.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Configuration
MODEL_PATH = Path(__file__).parent / "best.pth"
OUTPUT_DIR = Path(__file__).parent / "model_output"
TFLITE_PATH = OUTPUT_DIR / "stress_model.tflite"
C_HEADER_PATH = Path(__file__).parent / "components/cnn_inference/include/stress_model_data.h"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Model input shape
INPUT_SHAPE = (1, 4, 240)


class StressDetectionCNN(torch.nn.Module):
    def __init__(self):
        super(StressDetectionCNN, self).__init__()
        
        # Shared convolutional layers
        self.shared_conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=4, out_channels=64, kernel_size=10, padding=4),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, padding=4),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.MaxPool1d(kernel_size=2),
        )
        
        # Global Average Pooling
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        
        # Shared fully connected layers
        self.shared_fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
        )
        
        # Universal-private layer
        self.universal_private = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.shared_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)
        x = self.universal_private(x)
        return x


def load_model():
    """Load the trained PyTorch model."""
    print("=" * 80)
    print("Loading PyTorch Model")
    print("=" * 80)
    
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    model = StressDetectionCNN()
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters())} parameters")
    return model


def convert_to_tflite_torch_script(model):
    """Convert using TorchScript and tf.lite API."""
    print("\n" + "=" * 80)
    print("Converting to TFLite (TorchScript method)")
    print("=" * 80)
    
    # Create sample input
    sample_input = torch.randn(INPUT_SHAPE)
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, sample_input)
    traced_model.eval()
    
    # Save traced model
    traced_path = OUTPUT_DIR / "traced_model.pt"
    torch.jit.save(traced_model, str(traced_path))
    print(f"✅ Traced model saved: {traced_path}")
    
    # Now we need to convert traced PyTorch to TFLite
    # This requires manual ONNX export and tensorflow
    print("\nNOTE: Direct TorchScript to TFLite conversion requires additional tools.")
    print("Falling back to ONNX export...")
    
    # Export to ONNX
    onnx_path = OUTPUT_DIR / "stress_model.onnx"
    torch.onnx.export(
        model,
        sample_input,
        str(onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )
    print(f"✅ ONNX model saved: {onnx_path} ({onnx_path.stat().st_size / 1024:.2f} KB)")
    
    return onnx_path


def convert_onnx_to_tflite_manual(onnx_path):
    """Convert ONNX to TFLite using TensorFlow."""
    print("\n" + "=" * 80)
    print("Converting ONNX to TFLite")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        print("Loading ONNX model...")
        onnx_model = onnx.load(str(onnx_path))
        
        # Convert to TensorFlow
        print("Converting to TensorFlow...")
        tf_rep = prepare(onnx_model)
        tf_model_path = OUTPUT_DIR / "tf_model"
        tf_rep.export_graph(str(tf_model_path))
        
        # Convert to TFLite
        print("Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        with open(TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved: {TFLITE_PATH} ({TFLITE_PATH.stat().st_size / 1024:.2f} KB)")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install: pip install onnx-tf")
        return False


def generate_c_header():
    """Generate C header file with embedded TFLite model."""
    print("\n" + "=" * 80)
    print("Generating C Header File")
    print("=" * 80)
    
    if not TFLITE_PATH.exists():
        print("❌ TFLite model not found. Skipping C header generation.")
        return
    
    # Read TFLite model
    with open(TFLITE_PATH, 'rb') as f:
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
 * Generated by: convert_model_simple.py
 */

#ifndef STRESS_MODEL_DATA_H
#define STRESS_MODEL_DATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define STRESS_MODEL_SIZE {len(model_data)}
#define STRESS_MODEL_INPUT_CHANNELS 4
#define STRESS_MODEL_INPUT_SAMPLES 240
#define STRESS_MODEL_OUTPUT_SIZE 1

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
    
    # Generate C source file
    c_source_path = C_HEADER_PATH.parent.parent / "stress_model_data.c"
    
    with open(c_source_path, 'w') as f:
        f.write('#include "stress_model_data.h"\n\n')
        f.write('const unsigned char g_stress_model_data[] = {\n')
        
        for i in range(0, len(model_data), 16):
            chunk = model_data[i:i+16]
            hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
            f.write(f'  {hex_values},\n')
        
        f.write('};\n\n')
        f.write(f'const unsigned int g_stress_model_data_len = {len(model_data)};\n')
    
    print(f"✅ C header: {C_HEADER_PATH}")
    print(f"✅ C source: {c_source_path}")
    print(f"   Size: {len(model_data)} bytes ({len(model_data) / 1024:.2f} KB)")


def main():
    """Main conversion pipeline."""
    print("=" * 80)
    print("Simplified PyTorch to TFLite Converter")
    print("=" * 80)
    
    try:
        # Load model
        model = load_model()
        
        # Convert to ONNX
        onnx_path = convert_to_tflite_torch_script(model)
        
        # Convert ONNX to TFLite
        success = convert_onnx_to_tflite_manual(onnx_path)
        
        if success:
            # Generate C header
            generate_c_header()
            
            print("\n" + "=" * 80)
            print("✅ CONVERSION COMPLETE")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("⚠️  PARTIAL CONVERSION")
            print("=" * 80)
            print("\nONNX model created successfully.")
            print("To complete TFLite conversion, install: pip install onnx-tf")
            print("Then run this script again.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
