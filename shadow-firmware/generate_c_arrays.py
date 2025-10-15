#!/usr/bin/env python3
"""
Generate C header and source files from TFLite model for ESP32 deployment.

This script converts stress_model_quant.tflite into:
  - stress_model_data.h: Header with declarations and macros
  - stress_model_data.c: Source with model bytes as const array

Usage:
    python3 generate_c_arrays.py

The generated files will be placed in:
    components/cnn_inference/include/stress_model_data.h
    components/cnn_inference/stress_model_data.c
"""

import os
import sys
from pathlib import Path

# File paths
SCRIPT_DIR = Path(__file__).parent
TFLITE_MODEL_PATH = SCRIPT_DIR / 'model_output' / 'stress_model_quant.tflite'
OUTPUT_DIR = SCRIPT_DIR / 'components' / 'cnn_inference'
HEADER_PATH = OUTPUT_DIR / 'include' / 'stress_model_data.h'
SOURCE_PATH = OUTPUT_DIR / 'stress_model_data.c'

def generate_c_arrays():
    """Generate C header and source files from TFLite model."""
    
    # Validate input file exists
    if not TFLITE_MODEL_PATH.exists():
        print(f"❌ Error: TFLite model not found at {TFLITE_MODEL_PATH}")
        print(f"   Please place stress_model_quant.tflite in model_output/ directory")
        return 1
    
    # Read model bytes
    with open(TFLITE_MODEL_PATH, 'rb') as f:
        model_data = f.read()
    
    model_size = len(model_data)
    print(f"📦 Read TFLite model: {model_size / 1024:.2f} KB ({model_size} bytes)")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR / 'include', exist_ok=True)
    
    # ========================================================================
    # Generate Header File
    # ========================================================================
    header_content = f'''/**
 * @file stress_model_data.h
 * @brief TFLite model data for stress detection CNN
 * 
 * This file contains the embedded TFLite model as a constant array.
 * Generated automatically from stress_model_quant.tflite
 * 
 * Model specifications:
 *   Input:  (1, 4, 240) float32 - [ACC_MAG, BVP, EDA, TEMP]
 *   Output: (1, 1) float32 - Stress probability [0.0-1.0]
 *   Size:   {model_size / 1024:.2f} KB
 *   Quantization: Dynamic range (INT8 weights, FLOAT32 activations)
 */

#ifndef STRESS_MODEL_DATA_H
#define STRESS_MODEL_DATA_H

#ifdef __cplusplus
extern "C" {{
#endif

#include <stdint.h>

/**
 * @brief Total size of the TFLite model in bytes
 */
#define STRESS_MODEL_SIZE {model_size}

/**
 * @brief Input tensor shape: [batch, channels, timesteps]
 */
#define STRESS_MODEL_INPUT_BATCH 1
#define STRESS_MODEL_INPUT_CHANNELS 4
#define STRESS_MODEL_INPUT_TIMESTEPS 240
#define STRESS_MODEL_INPUT_SIZE (STRESS_MODEL_INPUT_BATCH * STRESS_MODEL_INPUT_CHANNELS * STRESS_MODEL_INPUT_TIMESTEPS)

/**
 * @brief Output tensor shape: [batch, outputs]
 */
#define STRESS_MODEL_OUTPUT_BATCH 1
#define STRESS_MODEL_OUTPUT_SIZE 1

/**
 * @brief TFLite model data (aligned to 16 bytes for optimal performance)
 * 
 * This array contains the complete TFLite flatbuffer model.
 * Use this with TFLite Micro interpreter.
 */
extern const unsigned char g_stress_model_data[] __attribute__((aligned(16)));

/**
 * @brief Length of the model data array
 */
extern const unsigned int g_stress_model_data_len;

#ifdef __cplusplus
}}
#endif

#endif // STRESS_MODEL_DATA_H
'''
    
    # Write header file
    with open(HEADER_PATH, 'w') as f:
        f.write(header_content)
    
    print(f"✅ Generated header: {HEADER_PATH}")
    print(f"   Size: {len(header_content)} bytes")
    
    # ========================================================================
    # Generate Source File
    # ========================================================================
    source_lines = [
        '/**',
        ' * @file stress_model_data.c',
        ' * @brief TFLite model data implementation',
        ' * ',
        f' * Model size: {model_size / 1024:.2f} KB',
        ' * Generated automatically - DO NOT EDIT',
        ' */',
        '',
        '#include "stress_model_data.h"',
        '',
        '/**',
        ' * @brief Embedded TFLite model data',
        ' * ',
        ' * This array contains the complete quantized TFLite model.',
        ' * Aligned to 16 bytes for optimal memory access on ESP32.',
        ' */',
        'const unsigned char g_stress_model_data[] __attribute__((aligned(16))) = {',
    ]
    
    # Convert model bytes to hex format (16 bytes per line)
    bytes_per_line = 16
    for i in range(0, model_size, bytes_per_line):
        chunk = model_data[i:i+bytes_per_line]
        hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
        
        # Add comma except for last line
        if i + bytes_per_line < model_size:
            source_lines.append(f'    {hex_values},')
        else:
            source_lines.append(f'    {hex_values}')
    
    source_lines.extend([
        '};',
        '',
        '/**',
        ' * @brief Length of the model data array',
        ' */',
        f'const unsigned int g_stress_model_data_len = {model_size};',
        ''
    ])
    
    source_content = '\n'.join(source_lines)
    
    # Write source file
    with open(SOURCE_PATH, 'w') as f:
        f.write(source_content)
    
    print(f"✅ Generated source: {SOURCE_PATH}")
    print(f"   Size: {len(source_content) / 1024:.2f} KB")
    print(f"   Lines: {len(source_lines)}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ C ARRAY GENERATION COMPLETE!")
    print("=" * 80)
    
    print(f"""
Generated files:
  📄 {HEADER_PATH.relative_to(SCRIPT_DIR)}
  📄 {SOURCE_PATH.relative_to(SCRIPT_DIR)}

Model specifications:
  • Size:          {model_size / 1024:.2f} KB ({model_size} bytes)
  • Input shape:   (1, 4, 240) float32
  • Output shape:  (1, 1) float32
  • Alignment:     16 bytes (optimized for ESP32)
  • Quantization:  Dynamic range (INT8 weights, FLOAT32 activations)

Next steps:
  1. Create cnn_inference component (Task 5)
     - Add TFLite Micro library to ESP-IDF project
     - Implement cnn_inference.c with model loading
     - Create cnn_predict() function
  
  2. Integrate with main firmware (Task 6)
     - Replace feature extraction + MLP with CNN inference
     - Connect signal_preprocessor → CNN → BLE output
     - Remove FSM logic

  3. Build and test on ESP32-S3
     - Check memory usage (~250 KB for model + arena)
     - Validate inference latency (<100ms target)
     - Test accuracy against Python model

📝 See PHASE3_CNN_INTEGRATION.md (will be created) for detailed instructions
""")
    
    return 0

if __name__ == '__main__':
    try:
        exit_code = generate_c_arrays()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
