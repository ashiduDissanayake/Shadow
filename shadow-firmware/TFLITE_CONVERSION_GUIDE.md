# TFLite Conversion Quick Start Guide

## 🎯 Goal
Convert `stress_model.onnx` (431 KB) to `stress_model.tflite` (~100-150 KB with INT8 quantization).

---

## ⚡ Method 1: Docker (Recommended - No Dependency Hell!)

### Step 1: Pull TensorFlow Docker Image
```bash
docker pull tensorflow/tensorflow:latest
```

### Step 2: Run Conversion in Container
```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware

docker run -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  tensorflow/tensorflow:latest \
  bash -c "pip install onnx onnx2tf tf-keras onnx-graphsurgeon psutil flatbuffers simple-onnx-processing-tools --quiet && python3 convert_model_to_tflite.py"
```

### Expected Output
```
================================================================================
PyTorch to TFLite Model Converter
================================================================================
✅ Model loaded successfully
✅ ONNX export successful
✅ TensorFlow conversion successful
✅ Float32 TFLite saved: model_output/stress_model.tflite
✅ INT8 Quantized TFLite saved: model_output/stress_model_quant.tflite
✅ C header generated: components/cnn_inference/include/stress_model_data.h
✅ C source generated: components/cnn_inference/stress_model_data.c

CONVERSION COMPLETE
```

### What You Get
- `model_output/stress_model.tflite` - Float32 model (~430 KB)
- `model_output/stress_model_quant.tflite` - INT8 quantized (~100-150 KB) **← Use this!**
- `components/cnn_inference/include/stress_model_data.h` - C header
- `components/cnn_inference/stress_model_data.c` - C source with embedded model

---

## ⚡ Method 2: Google Colab (If Docker Unavailable)

### Step 1: Open Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create new notebook

### Step 2: Upload Files
```python
# Upload these files to Colab:
# - best.pth
# - convert_model_to_tflite.py
```

### Step 3: Install Dependencies
```python
!pip install torch onnx onnx2tf tf-keras onnx-graphsurgeon psutil
```

### Step 4: Run Conversion
```python
!python3 convert_model_to_tflite.py
```

### Step 5: Download Results
```python
from google.colab import files

# Download TFLite models
files.download('model_output/stress_model.tflite')
files.download('model_output/stress_model_quant.tflite')

# Download C files
files.download('components/cnn_inference/include/stress_model_data.h')
files.download('components/cnn_inference/stress_model_data.c')
```

---

## ⚡ Method 3: Local (If You're Feeling Brave)

### Step 1: Create Fresh Virtual Environment
```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware

# Use Python 3.10 or 3.11 for best compatibility
python3.11 -m venv .venv_tflite
source .venv_tflite/bin/activate
```

### Step 2: Install Dependencies (In Order!)
```bash
pip install --upgrade pip

# Core dependencies
pip install numpy==1.24.0
pip install protobuf==3.20.3

# TensorFlow
pip install tensorflow==2.13.0

# ONNX tools
pip install onnx==1.14.0
pip install tf-keras==2.13.0

# Conversion tools
pip install onnx2tf==1.17.0
pip install onnx-graphsurgeon
pip install psutil
pip install flatbuffers
pip install simple-onnx-processing-tools
```

### Step 3: Run Conversion
```bash
python3 convert_model_to_tflite.py
```

### Common Issues & Fixes

**Issue 1: Protobuf version conflict**
```bash
pip install protobuf==3.20.3 --force-reinstall
```

**Issue 2: TensorFlow import error**
```bash
pip uninstall tensorflow tensorflow-estimator
pip install tensorflow==2.13.0
```

**Issue 3: ONNX conversion fails**
```bash
# Try older version
pip install onnx2tf==1.15.0
```

---

## 🔍 Verification Steps

### After Conversion, Check:

1. **File exists and size is reasonable**
   ```bash
   ls -lh model_output/stress_model_quant.tflite
   # Should be ~100-150 KB
   ```

2. **C files generated**
   ```bash
   ls -la components/cnn_inference/include/stress_model_data.h
   ls -la components/cnn_inference/stress_model_data.c
   # Both should exist
   ```

3. **Model loads in TFLite**
   ```bash
   python3 -c "import tensorflow as tf; interpreter = tf.lite.Interpreter('model_output/stress_model_quant.tflite'); interpreter.allocate_tensors(); print('✅ Model loads successfully')"
   ```

4. **Test inference**
   ```bash
   python3 -c "
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter('model_output/stress_model_quant.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test input
test_input = np.random.randn(1, 4, 240).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f'✅ Inference successful: {output[0, 0]:.6f}')
"
   ```

---

## 📊 Expected Results

### Model Specifications
```
Input Tensor:
  - Name: "input"
  - Shape: [1, 4, 240]
  - Type: FLOAT32
  - Quantization: None (input stays float)

Output Tensor:
  - Name: "output"
  - Shape: [1, 1]
  - Type: FLOAT32
  - Range: 0.0 to 1.0 (sigmoid)

Model Size:
  - Float32: ~430 KB
  - INT8 Quantized: ~100-150 KB (target for ESP32)

Accuracy:
  - Float32 vs PyTorch: <1% error
  - INT8 vs PyTorch: <5% error
```

### C Header Preview
```c
// stress_model_data.h
#define STRESS_MODEL_SIZE 125432
#define STRESS_MODEL_INPUT_CHANNELS 4
#define STRESS_MODEL_INPUT_SAMPLES 240
#define STRESS_MODEL_OUTPUT_SIZE 1

extern const unsigned char g_stress_model_data[];
extern const unsigned int g_stress_model_data_len;
```

### C Source Preview
```c
// stress_model_data.c
#include "stress_model_data.h"

const unsigned char g_stress_model_data[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33,
  // ... thousands of bytes ...
};

const unsigned int g_stress_model_data_len = 125432;
```

---

## 🚀 After Conversion is Complete

### Next Steps (Phase 3)

1. **Verify files are generated:**
   ```bash
   ls -la components/cnn_inference/include/stress_model_data.h
   ls -la components/cnn_inference/stress_model_data.c
   ```

2. **Create CNN inference component:**
   ```bash
   cd components/cnn_inference
   # Create cnn_inference.h and cnn_inference.c
   # Integrate TFLite Micro runtime
   ```

3. **Update CMakeLists.txt:**
   ```cmake
   idf_component_register(
       SRCS "cnn_inference.c" "stress_model_data.c"
       INCLUDE_DIRS "include"
       REQUIRES signal_preprocessor tflite-micro
   )
   ```

4. **Test on ESP32:**
   ```bash
   cd shadow-firmware
   idf.py build
   idf.py flash monitor
   ```

---

## 🐛 Troubleshooting

### Problem: Docker command fails
**Solution:** Make sure Docker Desktop is running
```bash
docker ps  # Should list containers, not error
```

### Problem: Protobuf version conflict
**Solution:** Use specific version
```bash
pip install protobuf==3.20.3 --force-reinstall --no-deps
```

### Problem: Out of memory during conversion
**Solution:** Close other applications, or use smaller batch size in representative_dataset()

### Problem: Conversion succeeds but model doesn't work on ESP32
**Solution:** Verify quantization parameters match TFLite Micro requirements

---

## 📞 Success Criteria

✅ `stress_model_quant.tflite` exists and is ~100-150 KB  
✅ `stress_model_data.h` and `.c` generated successfully  
✅ Model loads without errors in TFLite interpreter  
✅ Test inference produces output in range [0.0, 1.0]  
✅ Quantization error is < 5% compared to PyTorch  

**When all criteria are met → Proceed to Phase 3: CNN Integration** 🎯

---

## 💡 Pro Tips

1. **Use INT8 quantized model** - Much smaller, faster on ESP32
2. **Verify model before embedding** - Test inference in Python first
3. **Check memory limits** - ESP32-S3 has 512KB SRAM, model + arena should fit
4. **Profile on hardware** - Inference time should be < 100ms
5. **Keep float inputs/outputs** - Easier integration with preprocessing

---

## 📚 Resources

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [TFLite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [ONNX to TFLite](https://github.com/onnx/onnx-tensorflow)
- [ESP-NN Library](https://github.com/espressif/esp-nn)

---

**Estimated Time:** 15-30 minutes (Docker) | 1-2 hours (Local)  
**Difficulty:** Easy (Docker) | Medium (Local)  
**Blocker Risk:** Low (Docker) | Medium (Local)

**Recommendation: Use Docker!** 🐳
