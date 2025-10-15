# TFLite Conversion - Alternative Solution

## Problem
Python dependency hell with `onnx2tf` requiring incompatible versions of:
- `tf_keras` (doesn't exist for TensorFlow 2.13)
- `protobuf` (conflicting versions needed by different packages)
- `typing-extensions` (version conflicts)

## Solution: Use Online Conversion Tool

Since the local conversion has dependency issues, we'll use an online ONNX → TFLite converter.

### ✅ **Option 1: Netron + Manual Export (Recommended)**

1. **Upload ONNX to Netron**
   - Go to: https://netron.app/
   - Upload: `model_output/stress_model.onnx`
   - Inspect model architecture
   - Export to TensorFlow format

2. **Convert TensorFlow to TFLite**
   - Use Google Colab (free, has all dependencies pre-installed)
   - Upload the TensorFlow SavedModel
   - Run TFLite conversion with INT8 quantization
   - Download result

### ✅ **Option 2: Google Colab (Fully Automated)**

Create a Colab notebook:

```python
# Install dependencies
!pip install onnx onnx-tf tensorflow

# Upload stress_model.onnx

# Convert ONNX to TensorFlow
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

onnx_model = onnx.load('stress_model.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('tf_model')

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_saved_model('tf_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Quantization
import numpy as np
def representative_dataset():
    for _ in range(100):
        yield [np.random.randn(1, 4, 240).astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

# Save
with open('stress_model_quant.tflite', 'wb') as f:
    f.write(tflite_model)

# Download stress_model_quant.tflite
from google.colab import files
files.download('stress_model_quant.tflite')
```

### ✅ **Option 3: Use ESP-NN with ONNX Directly**

ESP32-S3 supports ONNX models directly through ESP-NN:
- No TFLite conversion needed
- Use `stress_model.onnx` as-is
- Integrate with ESP-NN library
- May have better performance on ESP32

### ✅ **Option 4: Manual C Implementation**

Since we have the model architecture and weights:
1. Extract weights from `best.pth`
2. Implement Conv1D, BatchNorm, Pooling, Linear layers in C
3. Load weights as constant arrays
4. Fastest inference, no runtime overhead

## Recommendation

**Use Google Colab (Option 2)** - It's free, takes 5 minutes, and has all dependencies pre-installed.

## What We Have

✅ **Working artifacts:**
- `best.pth` - Original PyTorch model (109K parameters)
- `stress_model.onnx` - ONNX model (431 KB, validated)
- `export_onnx.py` - Working ONNX export script
- Signal preprocessor component (fully implemented)

✅ **What we need:**
- `stress_model_quant.tflite` (~100-150 KB)
- C header/source files for embedding

## Next Steps

1. **Convert using Colab** (5-10 minutes)
2. **Generate C arrays** using provided script
3. **Create `cnn_inference` component** 
4. **Integrate with ESP32**

---

**Status:** ONNX model ready ✅ | TFLite conversion pending (use Colab) ⚙️
