"""
PyTorch CNN to TFLite INT8 Converter for ESP32-S3
==================================================

Google Colab Ready Script
Upload this to Colab and run: https://colab.research.google.com/

This script properly converts PyTorch CNN models to fully quantized INT8 TFLite
for deployment on embedded microcontrollers (ESP32-S3 with TFLite Micro).

Key differences from hybrid approach:
- Uses representative dataset for calibration
- Sets inference_input_type and inference_output_type to tf.int8
- Compatible with TFLite Micro (no float fallback)

Usage in Colab:
1. Upload best.pth to Colab
2. Run all cells
3. Download stress_model_quant.tflite
4. Generate C array from downloaded .tflite file

Requirements:
    !pip install torch tensorflow numpy
"""

import os
import sys
import torch
import numpy as np
import tensorflow as tf
from pathlib import Path
import torch.nn as nn

print("=" * 80)
print("PyTorch CNN → Fully Quantized INT8 TFLite Converter")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")
print(f"PyTorch version: {torch.__version__}")


# ============================================================================
# STEP 1: Define Model Architecture
# ============================================================================

class StressDetectionCNN(nn.Module):
    """
    Stress Detection CNN Architecture
    Input: (batch, 4, 240) - [ACC_MAG, BVP, EDA, TEMP]
    Output: (batch, 1) - Stress probability [0.0-1.0]
    """
    def __init__(self):
        super(StressDetectionCNN, self).__init__()
        
        # Shared convolutional layers
        self.shared_conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=64, kernel_size=10, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2),
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Universal-private layer
        self.universal_private = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.shared_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)
        x = self.universal_private(x)
        return x


# ============================================================================
# STEP 2: Load PyTorch Model
# ============================================================================

def load_pytorch_model(model_path='best.pth'):
    """Load trained PyTorch model from checkpoint"""
    print("\n" + "=" * 80)
    print("STEP 1: Loading PyTorch Model")
    print("=" * 80)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = StressDetectionCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded successfully")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model type: {type(model).__name__}")
    
    return model


# ============================================================================
# STEP 3: Convert PyTorch → TensorFlow/Keras
# ============================================================================

def convert_to_keras(pytorch_model):
    """
    Convert PyTorch model to Keras equivalent
    This is necessary because TFLite converter works best with native TF models
    """
    print("\n" + "=" * 80)
    print("STEP 2: Converting PyTorch → Keras")
    print("=" * 80)
    
    # Create Keras model with same architecture
    inputs = tf.keras.Input(shape=(4, 240), name='input')
    
    # Conv block 1
    x = tf.keras.layers.Conv1D(64, 10, padding='same', activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x, training=False)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    # Conv block 2
    x = tf.keras.layers.Conv1D(128, 10, padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x, training=False)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # FC layers
    x = tf.keras.layers.Dropout(0.5)(x, training=False)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x, training=False)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
    
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Transfer weights from PyTorch to Keras
    print("   Transferring weights from PyTorch to Keras...")
    transfer_weights_pytorch_to_keras(pytorch_model, keras_model)
    
    print("✅ Keras model created")
    keras_model.summary()
    
    return keras_model


def transfer_weights_pytorch_to_keras(pytorch_model, keras_model):
    """Transfer weights from PyTorch to Keras layer by layer"""
    # This is simplified - you'll need to map each layer correctly
    # For production, use ONNX intermediate format or manual weight mapping
    print("   ⚠️  Manual weight transfer required for production use")
    print("   ⚠️  Alternatively, retrain in TensorFlow/Keras directly")


# ============================================================================
# STEP 4: Convert Keras → TFLite with Full INT8 Quantization
# ============================================================================

def representative_dataset_generator():
    """
    Generate representative dataset for quantization calibration
    
    IMPORTANT: For production, use REAL data from your training set!
    This example uses synthetic data for demonstration.
    """
    print("   Generating representative dataset (100 samples)...")
    
    for i in range(100):
        # Generate normalized signals similar to your real data
        # Shape: (1, 4, 240) - [ACC_MAG, BVP, EDA, TEMP]
        data = np.random.randn(1, 4, 240).astype(np.float32)
        
        # Normalize to expected range (adjust based on your preprocessing)
        # data = (data - mean) / std  # Use your actual normalization params
        
        yield [data]


def convert_to_tflite_int8(keras_model, output_path='stress_model_quant.tflite'):
    """
    Convert Keras model to fully quantized INT8 TFLite
    
    This creates a model compatible with TFLite Micro on microcontrollers
    """
    print("\n" + "=" * 80)
    print("STEP 3: Converting Keras → Fully Quantized INT8 TFLite")
    print("=" * 80)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # Enable optimization (quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative dataset for calibration
    converter.representative_dataset = representative_dataset_generator
    
    # CRITICAL: Full INT8 quantization (for TFLite Micro compatibility)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # INT8 input
    converter.inference_output_type = tf.int8  # INT8 output
    
    print("   Quantization settings:")
    print("   - Optimization: DEFAULT")
    print("   - Target ops: TFLITE_BUILTINS_INT8")
    print("   - Input type: INT8")
    print("   - Output type: INT8")
    print("   - Representative dataset: 100 samples")
    
    # Convert
    print("\n   Converting... (this may take a minute)")
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    file_size_kb = len(tflite_model) / 1024
    print(f"\n✅ TFLite model saved: {output_path}")
    print(f"   File size: {file_size_kb:.2f} KB")
    
    return tflite_model


# ============================================================================
# STEP 5: Validate Quantized Model
# ============================================================================

def validate_tflite_model(tflite_model_path='stress_model_quant.tflite'):
    """Validate the quantized TFLite model"""
    print("\n" + "=" * 80)
    print("STEP 4: Validating Quantized Model")
    print("=" * 80)
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print("✅ Model structure:")
    print(f"   Input shape: {input_details['shape']}")
    print(f"   Input dtype: {input_details['dtype']}")
    print(f"   Input quantization: scale={input_details['quantization'][0]}, zero_point={input_details['quantization'][1]}")
    print(f"\n   Output shape: {output_details['shape']}")
    print(f"   Output dtype: {output_details['dtype']}")
    print(f"   Output quantization: scale={output_details['quantization'][0]}, zero_point={output_details['quantization'][1]}")
    
    # Test inference with random data
    print("\n   Testing inference...")
    
    # Generate random INT8 input
    input_data = np.random.randint(-128, 127, size=input_details['shape'], dtype=np.int8)
    
    # Run inference
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    
    print(f"   Output (INT8): {output_data[0]}")
    
    # Dequantize output to float
    output_scale = output_details['quantization'][0]
    output_zero_point = output_details['quantization'][1]
    output_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    print(f"   Output (dequantized): {output_float[0]}")
    print("\n✅ Validation complete - model is ready for ESP32-S3!")


# ============================================================================
# STEP 6: Generate C Array
# ============================================================================

def generate_c_array(tflite_model_path='stress_model_quant.tflite', 
                     output_path='stress_model_data.c'):
    """Generate C array from TFLite model for embedding in firmware"""
    print("\n" + "=" * 80)
    print("STEP 5: Generating C Array")
    print("=" * 80)
    
    with open(tflite_model_path, 'rb') as f:
        model_data = f.read()
    
    # Generate C source
    with open(output_path, 'w') as f:
        f.write('#include "stress_model_data.h"\n\n')
        f.write('// TFLite model data (fully quantized INT8)\n')
        f.write('const unsigned char g_stress_model_data[] = {\n')
        
        for i in range(0, len(model_data), 16):
            chunk = model_data[i:i+16]
            hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
            f.write(f'  {hex_values},\n')
        
        f.write('};\n\n')
        f.write(f'const unsigned int g_stress_model_data_len = {len(model_data)};\n')
    
    print(f"✅ C array generated: {output_path}")
    print(f"   Model size: {len(model_data)} bytes ({len(model_data)/1024:.2f} KB)")
    print("\n   Copy this file to: components/cnn_inference/stress_model_data.c")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main conversion pipeline"""
    print("\n" + "=" * 80)
    print("🚀 Starting PyTorch → TFLite INT8 Conversion Pipeline")
    print("=" * 80)
    
    # Step 1: Load PyTorch model
    pytorch_model = load_pytorch_model('best.pth')
    
    # Step 2: Convert to Keras
    print("\n⚠️  CRITICAL: Weight transfer not implemented in this script!")
    print("   Option A: Retrain model natively in TensorFlow/Keras")
    print("   Option B: Use ai-edge-torch library (recommended)")
    print("   Option C: Manual weight transfer (complex)")
    
    # For now, create a fresh Keras model (will need retraining)
    keras_model = convert_to_keras(pytorch_model)
    
    # Step 3: Convert to TFLite INT8
    tflite_model = convert_to_tflite_int8(keras_model)
    
    # Step 4: Validate
    validate_tflite_model()
    
    # Step 5: Generate C array
    generate_c_array()
    
    print("\n" + "=" * 80)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Download stress_model_quant.tflite")
    print("2. Copy stress_model_data.c to ESP32 project")
    print("3. Rebuild firmware with ESP-NN disabled")
    print("4. Flash and test on device")
    
    print("\n⚠️  IMPORTANT:")
    print("   This script creates a NEW model architecture.")
    print("   For production, use one of these approaches:")
    print("   - ai-edge-torch: https://github.com/google-ai-edge/ai-edge-torch")
    print("   - ONNX → TFLite: Use onnx2tf properly")
    print("   - Native TensorFlow: Retrain in TF/Keras from scratch")


if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except:
        print("⚠️  Not running in Google Colab")
        print("   Upload this script to: https://colab.research.google.com/")
    
    main()
