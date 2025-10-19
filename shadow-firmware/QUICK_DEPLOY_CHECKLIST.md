# ✅ INT8 Model Deployment Checklist

**Model:** stress_model_int8_esp32.h  
**Date:** October 18, 2025  
**Status:** Ready to deploy

---

## 📋 Pre-Deployment Checklist

### **Files to Download from Kaggle:**
- [ ] `stress_model_int8_esp32.h` (125.22 KB)
- [ ] `stress_model_int8_esp32.tflite` (optional, for reference)

### **Quick Integration (15 minutes):**

#### **Step 1: Copy Model File (2 min)**
```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware/components/cnn_inference/

# Copy from Downloads
cp ~/Downloads/stress_model_int8_esp32.h .

# Verify size
ls -lh stress_model_int8_esp32.h
# Should show: ~125 KB
```

- [ ] Model file copied to `components/cnn_inference/`
- [ ] File size is ~125 KB (128224 bytes)

---

#### **Step 2: Update cnn_inference.cpp (5 min)**

**Change 1: Include header**
```cpp
// OLD:
// #include "stress_model_data.c"

// NEW:
#include "stress_model_int8_esp32.h"
```

**Change 2: Update tensor arena**
```cpp
// Change from 400KB to 200KB
#define TENSOR_ARENA_SIZE (200 * 1024)
```

**Change 3: Update model reference**
```cpp
// OLD:
// g_model = tflite::GetModel(g_stress_model_data);

// NEW:
g_model = tflite::GetModel(stress_model_tflite);
```

**Change 4: Add quantization (copy from model header)**
```cpp
#define INPUT_SCALE      0.118650f
#define INPUT_ZERO_POINT -28
#define OUTPUT_SCALE     0.003906f
#define OUTPUT_ZERO_POINT -128
```

**Change 5: Quantize input in cnn_predict()**
```cpp
// Quantize float input to INT8
int8_t* input_tensor = interpreter->input(0)->data.int8;
for (int i = 0; i < input_size; i++) {
    float quantized = (input_data[i] / INPUT_SCALE) + INPUT_ZERO_POINT;
    input_tensor[i] = (int8_t)std::clamp(quantized, -128.0f, 127.0f);
}
```

**Change 6: Dequantize output in cnn_predict()**
```cpp
// Get INT8 output and dequantize to float
int8_t output_quantized = interpreter->output(0)->data.int8[0];
float stress_prob = (output_quantized - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
```

- [ ] All 6 changes made to `cnn_inference.cpp`
- [ ] File saved

---

#### **Step 3: Build and Flash (8 min)**

```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware

# Source ESP-IDF
. ~/Dev/esp/esp-idf/export.sh

# Clean build
idf.py fullclean

# Build
idf.py build

# Flash and monitor
idf.py flash monitor
```

- [ ] Build completes without errors
- [ ] Flash completes successfully
- [ ] Monitor shows boot logs

---

## ✅ **Success Verification**

### **Boot Logs Should Show:**

```
I (1166) cnn_inference: Initializing CNN with TFLite Micro...
I (1166) cnn_inference: Allocated 200 KB tensor arena in PSRAM
I (1176) cnn_inference: Model loaded: 128224 bytes ← Check this!
I (1186) cnn_inference: Operations registered: 34 ops
I (1206) cnn_inference: CNN initialized successfully ✅
I (1216) ShadowRealTime: ✅ CNN initialized successfully
```

- [ ] Model size is 128224 bytes (125.22 KB)
- [ ] "CNN initialized successfully" appears
- [ ] No error messages
- [ ] System continues running

---

### **After 60 Seconds (Inference):**

```
I (61246) Consumer: CNN inference started...
I (61421) Consumer: CNN inference: stress_prob=0.42 time=175ms ✅
```

- [ ] Inference runs automatically after 60s
- [ ] Inference time is 150-250ms (acceptable range)
- [ ] Stress probability is between 0.0 and 1.0
- [ ] No crashes or errors

---

## 🎯 **If Everything Works:**

**Congratulations! Move to next task:**

- [ ] Mark Task 7 (CNN Integration) as COMPLETE ✅
- [ ] Start Task 8: BLE Device Pairing
- [ ] Celebrate! 🎉

---

## ⚠️ **If Something Fails:**

### **Build Errors:**

**Error: `stress_model_tflite` undeclared**
- Check: Did you include `stress_model_int8_esp32.h`?
- Check: Is the model file in the right folder?

**Error: `int8` has no member named `data`**
- Check: Did you update quantization code correctly?
- Fix: Use `interpreter->input(0)->data.int8`

---

### **Runtime Errors:**

**Error: "Hybrid models not supported"**
- Problem: Wrong model file
- Fix: Verify using INT8 model, not old quant model

**Error: AllocateTensors() failed**
- Problem: Tensor arena too small
- Fix: Increase to 250KB or check operations

**Inference time > 300ms**
- Problem: ESP-NN not working
- Fix: Delete `managed_components/`, rebuild

**Output NaN or > 1.0**
- Problem: Missing dequantization
- Fix: Apply `(value - zero) * scale` formula

---

## 📝 **Final Notes**

**Model Specs:**
- Type: Full INT8 Quantized
- Size: 125.22 KB (128224 bytes)
- Accuracy: 98%
- Expected speed: 150-200ms
- ESP-NN: Compatible

**Memory Usage:**
- Model: 125 KB
- Tensor arena: 200 KB
- Runtime: ~50 KB
- **Total: ~375 KB**

**This fits comfortably in ESP32-S3!** ✅

---

## 🚀 **Ready to Deploy?**

Follow the steps above in order:
1. Download model (2 min)
2. Update code (5 min)
3. Build & flash (8 min)

**Total time: 15 minutes**

Good luck! 🍀

---

**For detailed help, see:** `INT8_MODEL_INTEGRATION_GUIDE.md`
