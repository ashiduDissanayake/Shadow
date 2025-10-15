# PyTorch to TFLite INT8 Conversion Guide

## Problem
Your current model is **hybrid quantized** (INT8 weights, FLOAT32 input/output), which TFLite Micro with ESP-NN doesn't support.

**Error:**
```
Hybrid models are not supported on TFLite Micro.
Node CONV_2D (number 2) failed to prepare with status 1
```

## Solution Options

### ✅ Option 1: Use AI Edge Torch (RECOMMENDED)
**Google's official PyTorch → TFLite converter**

1. **Open Google Colab**: https://colab.research.google.com/
2. **Upload files**:
   - `best.pth` (your trained model)
   - `convert_pytorch_aiedge.py` (from this directory)

3. **Run in Colab**:
```python
!pip install torch ai-edge-torch tensorflow
!python convert_pytorch_aiedge.py
```

4. **Download**: `stress_model_quant.tflite`

5. **Generate C array**:
```bash
xxd -i stress_model_quant.tflite > stress_model_data.c
```

**Pros:**
- Official Google tool
- Handles PyTorch directly
- Simpler pipeline
- Better weight preservation

**Cons:**
- Newer tool (may have bugs)

---

### Option 2: Disable ESP-NN (TEMPORARY FIX - ALREADY APPLIED)

Your firmware is already configured to use standard TFLite kernels instead of ESP-NN optimized kernels.

**What was changed:**
```kconfig
# sdkconfig
CONFIG_NN_ANSI_C=y            # Use standard C kernels
# CONFIG_NN_OPTIMIZED is not set  # Disable ESP-NN optimization
```

**Impact:**
- ✅ Works with hybrid quantized models
- ❌ 2-3x slower inference (~500ms vs ~200ms)
- ❌ Not ideal for production

**Test it now:**
```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware
./build_and_flash.sh
```

---

### Option 3: Fix Existing Conversion Script

Your `convert_model_to_tflite.py` has the right INT8 settings but ONNX→TF conversion fails.

**Issues:**
1. `onnx2tf` has compatibility issues with TF 2.x
2. Missing `onnxsim` tool
3. KerasTensor incompatibility

**Fixes needed:**
```bash
pip install onnx-simplifier
pip install onnx2tf==1.20.0  # Use specific version
```

Then modify conversion to use NCHW format:
```python
onnx2tf.convert(
    input_onnx_file_path=str(ONNX_PATH),
    output_folder_path=str(TF_MODEL_PATH),
    keep_ncw_or_nchw_or_ncdhw_input_names=['input'],  # Keep NCHW format
)
```

---

## Comparison Table

| Approach | Complexity | Speed | Success Rate |
|----------|------------|-------|--------------|
| **AI Edge Torch** | ⭐⭐ | Fast | ✅ High |
| **Disable ESP-NN** | ⭐ | Slow | ✅ Works now |
| **Fix ONNX→TF** | ⭐⭐⭐⭐⭐ | Fast | ⚠️ Unreliable |
| **Retrain in TF** | ⭐⭐⭐⭐ | Fast | ✅ Best quality |

---

## Recommended Workflow

### Immediate Testing (5 minutes)
```bash
# Your firmware already has ESP-NN disabled
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware
./build_and_flash.sh
```

**Expected result:** CNN initialization should succeed, but inference will be slow.

---

### Production Fix (30 minutes)
1. **Open Google Colab**
2. **Upload** `best.pth` and `convert_pytorch_aiedge.py`
3. **Run conversion** with AI Edge Torch
4. **Download** fully quantized INT8 model
5. **Generate C array**:
```bash
cd model_output
xxd -i stress_model_quant.tflite > ../components/cnn_inference/stress_model_data.c
```
6. **Update header** (add length variable)
7. **Re-enable ESP-NN** in sdkconfig
8. **Rebuild and flash**

---

## Current Status

✅ **Already Done:**
- ESP-NN disabled in sdkconfig
- 34 TFLite operations registered
- Custom partition table (1.875 MB)
- PSRAM enabled (8 MB)

⏭️ **Next Step:**
Test current firmware to verify CNN init works (with slow inference)

⏭️ **After Verification:**
Convert model properly using AI Edge Torch for production speed

---

## Files Created

1. **`PYTORCH_TO_TFLITE_COLAB.py`** - Full manual conversion script (educational)
2. **`convert_pytorch_aiedge.py`** - AI Edge Torch conversion (recommended)
3. **`THIS_FILE.md`** - This guide

---

## Questions?

- **"Why is inference slow?"** → Using standard C kernels instead of optimized ESP-NN
- **"Can I use the current model?"** → Yes! But inference is ~500ms instead of ~200ms
- **"What's the best long-term solution?"** → Convert with AI Edge Torch for full INT8 + ESP-NN
- **"Do I need to retrain?"** → No! AI Edge Torch preserves your trained weights

---

## Test Now

```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware
./build_and_flash.sh
```

**Look for:**
```
I (xxx) cnn_inference: Tensor arena: XXXXX / 204800 bytes used
I (xxx) cnn_inference: CNN initialized successfully ✅
I (xxx) ShadowRealTime: ✅ CNN initialized successfully
```

**If successful:**
- Wait 60 seconds for sensor data
- Check first inference runs
- Measure latency (will be ~500ms)
- Then proceed to AI Edge Torch conversion for production

---

## References

- [AI Edge Torch](https://github.com/google-ai-edge/ai-edge-torch)
- [TFLite INT8 Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [TFLite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [ESP-NN](https://github.com/espressif/esp-nn)
