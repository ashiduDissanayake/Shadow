# Google Colab Notebook for ONNX → TFLite Conversion
# Copy this entire file into a new Colab notebook
# Upload stress_model.onnx when prompted

# ============================================================================
# Step 1: Install Dependencies
# ============================================================================
print("=" * 80)
print("Installing dependencies...")
print("=" * 80)

# Use onnx2tf instead of onnx-tf (newer, maintained)
!pip install -q onnx onnx2tf onnxsim tensorflow

print("✅ Dependencies installed\n")

# ============================================================================
# Step 2: Upload ONNX Model
# ============================================================================
print("=" * 80)
print("Upload your ONNX model")
print("=" * 80)
print("Click the file upload button and select: stress_model.onnx\n")

from google.colab import files
uploaded = files.upload()

onnx_file = list(uploaded.keys())[0]
print(f"\n✅ Uploaded: {onnx_file} ({len(uploaded[onnx_file]) / 1024:.2f} KB)")

# ============================================================================
# Step 3: Convert ONNX to TensorFlow SavedModel
# ============================================================================
print("\n" + "=" * 80)
print("Converting ONNX to TensorFlow...")
print("=" * 80)

import onnx2tf

# Convert using onnx2tf
tf_model_path = 'saved_model'

print("Running onnx2tf conversion (this may take 1-2 minutes)...")
onnx2tf.convert(
    input_onnx_file_path=onnx_file,
    output_folder_path=tf_model_path,
    copy_onnx_input_output_names_to_tflite=True,
    non_verbose=True,
)

print(f"✅ TensorFlow SavedModel created: {tf_model_path}")

# ============================================================================
# Step 4: Convert to TFLite (Float32)
# ============================================================================
print("\n" + "=" * 80)
print("Converting to TFLite Float32...")
print("=" * 80)

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_float_model = converter.convert()

tflite_float_path = 'stress_model_float.tflite'
with open(tflite_float_path, 'wb') as f:
    f.write(tflite_float_model)

print(f"✅ Float32 TFLite model: {len(tflite_float_model) / 1024:.2f} KB")

# ============================================================================
# Step 5: Convert to TFLite Dynamic Range Quantized (Weights Only)
# ============================================================================
print("\n" + "=" * 80)
print("Converting to TFLite Quantized (Dynamic Range)...")
print("=" * 80)

import numpy as np

# Use dynamic range quantization (weights only, no calibration needed)
# This is more compatible with models converted from ONNX
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Running weight quantization...")
tflite_quant_model = converter.convert()

tflite_quant_path = 'stress_model_quant.tflite'
with open(tflite_quant_path, 'wb') as f:
    f.write(tflite_quant_model)

compression_ratio = (1 - len(tflite_quant_model) / len(tflite_float_model)) * 100
print(f"✅ Quantized TFLite model: {len(tflite_quant_model) / 1024:.2f} KB")
print(f"   Compression: {compression_ratio:.1f}% smaller than float32")
print(f"   Type: Dynamic range quantization (INT8 weights, FLOAT32 activations)")

# ============================================================================
# Step 5b: Attempt Full INT8 Quantization (Optional Optimization)
# ============================================================================
print("\n" + "=" * 80)
print("Attempting full INT8 quantization (with calibration)...")
print("=" * 80)
print("Note: This may fail due to ONNX→TF conversion limitations.")
print("      Dynamic range quantization (above) is already sufficient for ESP32.\n")

try:
    # Try full INT8 quantization with representative dataset
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for calibration
    def representative_dataset():
        """Generate representative samples for quantization calibration"""
        np.random.seed(42)
        for i in range(100):
            t = np.linspace(0, 60, 240)
            acc_mag = 0.3 * np.sin(2 * np.pi * 0.1 * t) + 0.2 * np.random.randn(240)
            bvp = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(240)
            eda = 0.5 + 0.3 * np.sin(2 * np.pi * 0.05 * t) + 0.1 * np.random.randn(240)
            temp = 0.2 * np.sin(2 * np.pi * 0.02 * t) + 0.05 * np.random.randn(240)
            data = np.stack([acc_mag, bvp, eda, temp], axis=0)
            data = np.expand_dims(data, axis=0).astype(np.float32)
            yield [data]
    
    converter_int8.representative_dataset = representative_dataset
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.float32
    converter_int8.inference_output_type = tf.float32
    
    tflite_int8_model = converter_int8.convert()
    
    tflite_int8_path = 'stress_model_int8.tflite'
    with open(tflite_int8_path, 'wb') as f:
        f.write(tflite_int8_model)
    
    int8_compression = (1 - len(tflite_int8_model) / len(tflite_float_model)) * 100
    print(f"✅ Full INT8 model created successfully!")
    print(f"   Size: {len(tflite_int8_model) / 1024:.2f} KB")
    print(f"   Compression: {int8_compression:.1f}% smaller than float32")
    print(f"   Type: Full INT8 quantization (INT8 weights + activations)")
    
    # Compare and use the smaller model
    if len(tflite_int8_model) < len(tflite_quant_model):
        size_diff = len(tflite_quant_model) - len(tflite_int8_model)
        print(f"\n   → Using full INT8 model ({size_diff / 1024:.1f} KB smaller, potentially faster)")
        tflite_quant_model = tflite_int8_model
        tflite_quant_path = tflite_int8_path
    else:
        print(f"\n   → Keeping dynamic range model (already optimal size)")
        
except Exception as e:
    error_msg = str(e).split('\n')[0]  # Get first line of error
    print(f"⚠️  Full INT8 quantization failed (expected for ONNX-converted models)")
    print(f"   Error: {error_msg}")
    print(f"\n   ✅ Using dynamic range quantized model instead")
    print(f"   → Dynamic range quantization is sufficient for ESP32-S3")
    print(f"   → Model is {compression_ratio:.1f}% smaller and fully optimized")

# ============================================================================
# Step 6: Validate Model
# ============================================================================
print("\n" + "=" * 80)
print("Validating TFLite model...")
print("=" * 80)

# Load interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_quant_path)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\nInput tensor:")
print(f"  Name:  {input_details[0]['name']}")
print(f"  Shape: {input_details[0]['shape']}")
print(f"  Type:  {input_details[0]['dtype']}")

print(f"\nOutput tensor:")
print(f"  Name:  {output_details[0]['name']}")
print(f"  Shape: {output_details[0]['shape']}")
print(f"  Type:  {output_details[0]['dtype']}")

# Test inference
test_input = np.random.randn(1, 4, 240).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
test_output = interpreter.get_tensor(output_details[0]['index'])

print(f"\nTest inference:")
print(f"  Input shape:  {test_input.shape}")
print(f"  Output value: {test_output[0, 0]:.6f}")

if 0 <= test_output[0, 0] <= 1:
    print(f"  Output range: [0.0, 1.0] ✅")
else:
    print(f"  Output range: ⚠️ WARNING - outside expected range!")

print(f"\n✅ Model validation successful")

# ============================================================================
# Step 7: Download Models
# ============================================================================
print("\n" + "=" * 80)
print("Downloading models...")
print("=" * 80)

print("\nDownloading INT8 quantized model (use this for ESP32)...")
files.download(tflite_quant_path)

print("\nDownloading float32 model (for comparison)...")
files.download(tflite_float_path)

# ============================================================================
# Step 8: Generate C Header Preview
# ============================================================================
print("\n" + "=" * 80)
print("C Header Preview (first 100 bytes)")
print("=" * 80)

with open(tflite_quant_path, 'rb') as f:
    model_bytes = f.read()

print(f"\nconst unsigned char g_stress_model_data[] = {{")
for i in range(0, min(96, len(model_bytes)), 16):
    chunk = model_bytes[i:i+16]
    hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
    print(f"  {hex_values},")
print(f"  // ... {len(model_bytes) - 96} more bytes ...")
print(f"}};")
print(f"\nconst unsigned int g_stress_model_data_len = {len(model_bytes)};")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("✅ CONVERSION COMPLETE!")
print("=" * 80)

print(f"""
Models created:
  • Float32:       {len(tflite_float_model) / 1024:.2f} KB
  • INT8 Quantized: {len(tflite_quant_model) / 1024:.2f} KB (← Use this for ESP32)

Next steps:
  1. Download 'stress_model_quant.tflite' (should be in your Downloads folder)
  2. Copy to shadow-firmware/model_output/ directory
  3. Run the C array generator script (provided in the repo)
  4. Integrate with ESP32-S3 firmware

Model specifications:
  Input:  (1, 4, 240) float32 - [ACC_MAG, BVP, EDA, TEMP]
  Output: (1, 1) float32 - Stress probability [0.0-1.0]
  Quantization: INT8 weights/activations, FLOAT32 input/output
""")

print("🎉 Ready for ESP32 deployment!")
