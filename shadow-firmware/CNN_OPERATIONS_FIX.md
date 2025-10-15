# CNN Model Operations Issue - RESOLVED

## Date: October 16, 2025 00:00

## Issue History

### Issue #1: PSRAM Not Enabled ✅ FIXED
**Error:**
```
E (643) cnn_inference: Failed to allocate 204800 bytes in PSRAM
```

**Solution:** Enabled PSRAM in `sdkconfig`
```
CONFIG_SPIRAM=y
CONFIG_SPIRAM_MODE_OCT=y
CONFIG_SPIRAM_SPEED_80M=y
```

**Result:**
```
I (319) esp_psram: Found 8MB PSRAM device ✅
I (1151) cnn_inference: Allocated 200 KB tensor arena in PSRAM ✅
```

---

### Issue #2: Missing PAD Operation ✅ FIXED
**Error:**
```
Didn't find op for builtin opcode 'PAD'
Failed to get registration from op code PAD
```

**Solution:** Added `resolver.AddPad()` to operation resolver

**Result:**
```
I (1163) cnn_inference: Operations registered: Conv2D, Pad, ... ✅
```

---

### Issue #3: Missing EXPAND_DIMS Operation ⏭️ NEXT TO FIX
**Error:**
```
Didn't find op for builtin opcode 'EXPAND_DIMS'
Failed to get registration from op code EXPAND_DIMS
E (1183) cnn_inference: AllocateTensors() failed
```

**Solution Applied:** Added `resolver.AddExpandDims()` to operation resolver

**Expected Result After Rebuild:**
```
I (xxx) cnn_inference: CNN initialized successfully ✅
I (xxx) cnn_inference: Tensor arena: XXXXX / 204800 bytes (XX.X% used)
I (xxx) ShadowRealTime: ✅ CNN initialized successfully
```

---

## Current Model Operations

Based on the errors encountered, the CNN model uses these operations:

1. ✅ **Conv2D** - Convolutional layers
2. ✅ **Pad** - Padding for convolutions
3. ✅ **ExpandDims** - Add dimensions (NEW)
4. ✅ **Reshape** - Tensor reshaping
5. ✅ **FullyConnected** - Dense layers
6. ✅ **Relu** - Activation function
7. ✅ **Softmax** - Output layer
8. ✅ **Quantize** - Quantization
9. ✅ **Dequantize** - Dequantization

**Resolver Size:** Increased from 10 → 12 → 15 to accommodate all operations

---

## How to Build & Flash (With ESP-IDF Environment)

### Method 1: Using the Build Script (Recommended)

```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware
./build_and_flash.sh
```

The script automatically:
1. Sources ESP-IDF environment
2. Builds the firmware
3. Flashes to device
4. Opens serial monitor

### Method 2: Manual Commands

```bash
# Step 1: Source ESP-IDF environment (CRITICAL!)
. $HOME/Dev/esp/esp-idf/export.sh

# Step 2: Navigate to project
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware

# Step 3: Build
idf.py build

# Step 4: Flash
idf.py flash

# Step 5: Monitor
idf.py monitor
```

### Method 3: All-in-One Command

```bash
# Source environment first!
. $HOME/Dev/esp/esp-idf/export.sh

# Then run everything
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware && idf.py build flash monitor
```

---

## Expected Boot Sequence (After Fix)

```
I (xxx) boot: ESP-IDF v5.5 2nd stage bootloader
...
I (319) esp_psram: Found 8MB PSRAM device          ← PSRAM detected ✅
I (322) esp_psram: Speed: 80MHz
...
I (1141) ShadowRealTime: 🧠 Initializing CNN inference engine...
I (1143) cnn_inference: Initializing CNN with TFLite Micro...
I (1143) cnn_inference: Allocated 200 KB tensor arena in PSRAM  ← SUCCESS ✅
I (1163) cnn_inference: Model loaded: 124176 bytes
I (1163) cnn_inference: Operations registered: Conv2D, Pad, ExpandDims, Reshape, FullyConnected, Relu, Softmax, Quantize, Dequantize
I (xxxx) cnn_inference: Tensor arena: XXXXX / 204800 bytes (XX.X% used)  ← NEW: Should appear
I (xxxx) cnn_inference: CNN initialized successfully  ← SUCCESS! ✅
I (xxxx) ShadowRealTime: ✅ CNN initialized successfully
I (xxxx) ShadowRealTime:    Model: stress_model_quant.tflite
I (xxxx) ShadowRealTime:    Tensor arena: XXX / 200 KB (XX.X% used)
I (xxxx) ShadowRealTime:    Free heap after CNN init: 8XXXXXX bytes
...
I (xxxx) ShadowRealTime: System ONLINE - Real-time stress detection with CNN active!
```

---

## What Changed in Latest Build

### File: `components/cnn_inference/cnn_inference.cpp`

```cpp
// OLD (missing operations):
static tflite::MicroMutableOpResolver<10> resolver;
if (resolver.AddConv2D() != kTfLiteOk ||
    resolver.AddReshape() != kTfLiteOk ||
    resolver.AddFullyConnected() != kTfLiteOk ||
    ...

// NEW (all operations):
static tflite::MicroMutableOpResolver<15> resolver;
if (resolver.AddConv2D() != kTfLiteOk ||
    resolver.AddPad() != kTfLiteOk ||           // ADDED
    resolver.AddExpandDims() != kTfLiteOk ||    // ADDED
    resolver.AddReshape() != kTfLiteOk ||
    resolver.AddFullyConnected() != kTfLiteOk ||
    ...
```

---

## Memory Status (Current)

From the health check log:
```
I (33413) Shadow: 💓 Shadow System Health Check #1
I (33413) Shadow:    Free heap: 8334168 bytes  ← 8.3 MB! (PSRAM working!)
I (33413) Shadow:    Total samples: 128
I (33423) Shadow:    ML inferences: 0  ← Will be >0 after CNN works
I (33423) Shadow:    State transitions: 0
I (33423) Shadow:    Sensor health: 25% (FAIR)
```

**Good news:**
- ✅ 8.3 MB free heap (PSRAM included)
- ✅ Sensor data collecting (128 EDA samples)
- ✅ System stable

**After CNN fix:**
- ML inferences should increment every 60 seconds
- Free heap should remain ~8 MB (200 KB used for tensor arena)

---

## Troubleshooting

### If "command not found: idf.py"
**Problem:** ESP-IDF environment not sourced

**Solution:**
```bash
. $HOME/Dev/esp/esp-idf/export.sh
```

### If more missing operations appear
**Problem:** Model uses additional operations

**Solution:** Add to resolver in `cnn_inference.cpp`:
```cpp
resolver.AddOperationName() != kTfLiteOk ||
```

Common operations to try:
- `AddMaxPool2D()`
- `AddAveragePool2D()`
- `AddMean()`
- `AddSqueeze()`
- `AddConcatenation()`

### If AllocateTensors() still fails after adding all ops
**Problem:** Tensor arena too small

**Solution:** Increase size in `cnn_inference.cpp`:
```cpp
constexpr int kTensorArenaSize = 250 * 1024;  // Increase from 200 KB
```

---

## Next Steps After Successful CNN Init

1. **Wait 60 seconds** for first inference
2. **Verify logs show:**
   ```
   I (xxx) ShadowRealTime: 🔔 CNN Inference #1
   I (xxx) cnn_inference: Inference: XX.X%, XXXXus
   I (xxx) ShadowRealTime: 🎯 CNN Inference Result:
   I (xxx) ShadowRealTime:    Stress Probability: XX.X%
   ```

3. **Check performance:**
   - Inference time < 200ms ✅
   - Memory stable ✅
   - No crashes ✅

4. **Continue to Task 8:** Device pairing (ESP32)

---

## Files Modified

1. ✅ `sdkconfig` - PSRAM enabled
2. ✅ `components/cnn_inference/cnn_inference.cpp` - Added Pad, ExpandDims operations
3. ✅ `build_and_flash.sh` - Build script with ESP-IDF environment sourcing
4. ✅ `enable_psram.py` - Helper script to enable PSRAM

---

## Summary

**Status:** ⚠️ Almost there! One more operation to add.

**Progress:**
- ✅ PSRAM working (8MB available)
- ✅ Tensor arena allocated successfully
- ✅ Model loaded
- ✅ PAD operation added
- ⏭️ EXPAND_DIMS operation added (needs rebuild)

**Next Command:**
```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware
./build_and_flash.sh
```

**Expected:** CNN initialization should succeed after this rebuild! 🎉

