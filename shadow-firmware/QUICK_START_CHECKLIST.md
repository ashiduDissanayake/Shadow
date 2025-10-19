# 🚀 Quick Start: Model Conversion Checklist

**Date:** October 17, 2025  
**Status:** Ready to convert hybrid → full INT8 model

---

## 📋 Pre-Conversion Checklist

- [ ] **Locate PyTorch model file**
  - Path: `/Users/ashidudissanayake/Dev/Shadow/model-development/best.pth`
  - Verify it exists: `ls -lh /Users/ashidudissanayake/Dev/Shadow/model-development/best.pth`

- [ ] **Locate conversion script**
  - Path: `/Users/ashidudissanayake/Dev/Shadow/shadow-firmware/convert_pytorch_aiedge.py`
  - Verify it exists: ✅ (you're viewing it now!)

- [ ] **Read the complete guide**
  - Open: `MODEL_CONVERSION_COMPLETE_GUIDE.md`
  - Understand: Why we need full INT8 quantization

---

## 🎯 Option A: Google Colab (Recommended - 30 min)

### ✅ **Step 1: Open Google Colab** (2 min)
```
1. Go to: https://colab.research.google.com/
2. Sign in with Google account
3. Click: File → New notebook
```

### ✅ **Step 2: Upload Files** (3 min)
```python
# Click the folder icon (📁) on the left sidebar
# Click "Upload" button
# Select these 2 files:
#   - best.pth (from model-development folder)
#   - convert_pytorch_aiedge.py (from shadow-firmware folder)
```

### ✅ **Step 3: Install Dependencies** (5 min)
```python
# Copy-paste into Colab cell and run (Shift+Enter):
!pip install torch torchvision ai-edge-torch tensorflow numpy
```
⏱️ Wait for installation to complete (~5 min)

### ✅ **Step 4: Run Conversion** (10 min)
```python
# Copy-paste into new cell and run:
!python convert_pytorch_aiedge.py
```

**Expected Output:**
```
================================================================================
PyTorch → TFLite INT8 using AI Edge Torch
================================================================================
PyTorch: 2.x.x
TensorFlow: 2.x.x
AI Edge Torch: x.x.x

================================================================================
Loading PyTorch Model
================================================================================
✅ Model loaded: 123,456 parameters

================================================================================
Converting PyTorch → TFLite INT8
================================================================================
   Input shape: (1, 4, 240)
   Converting...
✅ Conversion complete (FLOAT32)

   Applying INT8 quantization...
✅ Quantized model saved: stress_model_quant.tflite
   Size: 120.34 KB

================================================================================
Validating TFLite Model
================================================================================
   Input: [1, 4, 240] <class 'numpy.int8'>
   Output: [1, 1] <class 'numpy.int8'>
✅ Model is ready for ESP32-S3!

================================================================================
✅ SUCCESS!
================================================================================

Next: Download stress_model_quant.tflite
Then: xxd -i stress_model_quant.tflite > stress_model_data.c
```

### ✅ **Step 5: Download Model** (1 min)
```python
# Copy-paste into new cell and run:
from google.colab import files
files.download('stress_model_quant.tflite')
```

**Save to:** `/Users/ashidudissanayake/Dev/Shadow/shadow-firmware/`

---

## 🎯 Option B: Local Conversion (Alternative - 30 min)

### ✅ **Step 1: Create Virtual Environment** (2 min)
```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware

# Create environment
python3 -m venv .venv_int8
source .venv_int8/bin/activate
```

### ✅ **Step 2: Install Dependencies** (5 min)
```bash
pip install torch torchvision ai-edge-torch tensorflow numpy
```

### ✅ **Step 3: Copy Model File** (1 min)
```bash
cp ../model-development/best.pth .
```

### ✅ **Step 4: Run Conversion** (10 min)
```bash
python convert_pytorch_aiedge.py
```

### ✅ **Step 5: Verify Output** (1 min)
```bash
ls -lh stress_model_quant.tflite
# Should show ~120 KB file
```

---

## 🔧 Post-Conversion: ESP32 Integration (15 min)

### ✅ **Step 1: Convert to C Array** (2 min)
```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware

# Convert TFLite model to C array
xxd -i stress_model_quant.tflite > stress_model_data_temp.c
```

### ✅ **Step 2: Edit C File** (3 min)
```bash
# Open stress_model_data_temp.c
# 1. Rename arrays to match expected names:
#    stress_model_quant_tflite[] → g_stress_model_data[]
#    stress_model_quant_tflite_len → (add new line below)
#
# 2. Add at the end:
#    const unsigned int g_stress_model_data_len = <LENGTH>;
#
# 3. Save as: components/cnn_inference/stress_model_data.c
```

**Example edit:**
```c
// Before:
unsigned char stress_model_quant_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, ...
};
unsigned int stress_model_quant_tflite_len = 124176;

// After:
const unsigned char g_stress_model_data[] = {
  0x1c, 0x00, 0x00, 0x00, ...
};
const unsigned int g_stress_model_data_len = 124176;
```

### ✅ **Step 3: Re-enable ESP-NN** (2 min)

**Option 1: Delete managed_components (Easiest)**
```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware
rm -rf managed_components
```

**Option 2: Revert CMakeLists.txt**
```bash
cd managed_components/espressif__esp-tflite-micro
git checkout CMakeLists.txt
```

### ✅ **Step 4: Clean Build** (5 min)
```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware

# Source ESP-IDF
. ~/Dev/esp/esp-idf/export.sh

# Clean previous build
idf.py fullclean

# Build fresh
idf.py build
```

### ✅ **Step 5: Flash & Test** (3 min)
```bash
idf.py flash monitor
```

**Look for:**
```
I (1166) cnn_inference: Initializing CNN with TFLite Micro...
I (1166) cnn_inference: Allocated 200 KB tensor arena in PSRAM
I (1176) cnn_inference: Model loaded: 124176 bytes
I (1186) cnn_inference: Operations registered: 34 ops...
I (1196) cnn_inference: Tensor arena: 187654 / 204800 bytes (91.6% used)
I (1206) cnn_inference: CNN initialized successfully ✅
I (1216) ShadowRealTime: ✅ CNN initialized successfully
```

**After 60 seconds:**
```
I (61216) Consumer: CNN inference: stress_prob=0.42 time=187ms
```

---

## ✅ Success Criteria

- [ ] **No "Hybrid models not supported" error**
- [ ] **"CNN initialized successfully ✅" appears in logs**
- [ ] **Inference runs after 60 seconds**
- [ ] **Inference time: 150-250ms** (with ESP-NN optimizations)
- [ ] **Memory usage: ~187 KB / 200 KB**

---

## 🚨 Troubleshooting

### ❌ **Still getting "Hybrid models not supported"**
```
Problem: You're using the old hybrid model
Solution: 
  1. Verify new model is converted (check file date)
  2. Make sure you replaced stress_model_data.c
  3. Run: idf.py fullclean && idf.py build
```

### ❌ **"AllocateTensors() failed"**
```
Problem: Missing TFLite operations or incorrect quantization
Solution:
  1. Check for "Didn't find op for builtin opcode" in logs
  2. Verify model is full INT8 (run validation script)
  3. Increase tensor arena size if needed
```

### ❌ **Conversion script fails**
```
Problem: Missing dependencies or wrong PyTorch model format
Solution:
  1. Check all packages installed: pip list | grep -E "torch|tensorflow|ai-edge"
  2. Verify best.pth is not corrupted: ls -lh best.pth
  3. Try running in Google Colab instead
```

### ❌ **"Input tensor type mismatch"**
```
Problem: Firmware expects different input type
Solution:
  1. Check model input type: netron stress_model_quant.tflite
  2. Update cnn_inference.cpp to match INT8 input
```

---

## 📊 Performance Comparison

| Metric | Hybrid Model (❌ Failed) | Full INT8 Model (✅ Expected) |
|--------|-------------------------|-------------------------------|
| **Boot Status** | ❌ "Hybrid not supported" | ✅ "CNN initialized successfully" |
| **Inference Time** | N/A (doesn't run) | ~180ms with ESP-NN |
| **Memory Usage** | N/A | ~187 KB / 200 KB |
| **Model Size** | 121 KB | ~120 KB |
| **Input Type** | FLOAT32 | INT8 |
| **Output Type** | FLOAT32 | INT8 |
| **Accuracy** | N/A | ~99% (same as PyTorch) |

---

## 📝 Notes

### **Why This Works:**
- ✅ Full INT8 quantization (weights + activations)
- ✅ Compatible with ESP-NN optimized kernels
- ✅ TFLite Micro supports fully quantized models
- ✅ 75% memory savings vs FLOAT32

### **Why Previous Attempts Failed:**
- ❌ Hybrid quantization (INT8 weights + FLOAT32 activations)
- ❌ TFLite Micro doesn't support hybrid models
- ❌ Both ESP-NN and standard kernels reject hybrid models
- ❌ No workaround available without reconversion

### **Representative Dataset Importance:**
The calibration data used during quantization determines accuracy:
- **Bad:** Random uniform data → 60-70% accuracy ❌
- **Acceptable:** Random Gaussian data → 95-98% accuracy ⚠️
- **Best:** Real WESAD samples → 99%+ accuracy ✅

---

## 🎯 Next Session Goals

After successful conversion:

1. **Task 7 Complete:** ✅ CNN inference working on device
2. **Task 8:** Implement device pairing with BLE characteristics
3. **Task 9:** Build macOS monitoring application
4. **Task 10:** Test multi-device scenarios
5. **Task 11:** Production validation

---

## 📞 Need Help?

If you're stuck:

1. **Check the logs carefully** - Error messages are specific
2. **Read MODEL_CONVERSION_COMPLETE_GUIDE.md** - Has detailed explanations
3. **Verify each step** - Don't skip the validation checks
4. **Test conversion in Colab first** - Easier to debug

---

**Ready to start?** Begin with the checklist above! 🚀

---

**Created:** October 17, 2025  
**Status:** Ready for execution  
**Estimated Time:** 45 minutes total
