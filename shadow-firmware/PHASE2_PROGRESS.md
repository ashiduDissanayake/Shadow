# Phase 2: Model Conversion - PROGRESS UPDATE

## 📋 Status: ONNX Export Complete ✅ | TFLite Conversion In Progress ⚙️

---

## ✅ What Was Accomplished

### 1. ONNX Model Export - SUCCESS
✅ **File**: `shadow-firmware/model_output/stress_model.onnx` (431 KB)

**Model Architecture Confirmed:**
- Input: `(batch, 4, 240)` - 4 channels (ACC_MAG, BVP, EDA, TEMP), 240 samples each
- Output: `(batch, 1)` - Stress probability [0.0-1.0]
- Parameters: 109,889 total
- Layers:
  - Conv1D Block 1: 4→64 channels (kernel=10, BatchNorm, ReLU, Dropout, MaxPool)
  - Conv1D Block 2: 64→128 channels (kernel=10, BatchNorm, ReLU, Dropout, MaxPool)
  - Global Average Pooling: 128 channels → 128 features
  - FC Block: 128→128 (Dropout, Linear, ReLU)
  - Output Block: 128→64→1 (Linear, ReLU, Dropout, Linear, Sigmoid)

**Validation:**
- Test inference successful: `0.706626` probability on random input
- ONNX model validated with `onnx.checker`
- Ready for deployment or further conversion

### 2. Scripts Created
✅ `export_onnx.py` - Simple, working ONNX export script  
✅ `convert_model_to_tflite.py` - Full conversion pipeline (complex dependencies)  
✅ `convert_model_simple.py` - Alternative simplified converter  
✅ `conversion_requirements.txt` - Python dependencies  

### 3. Virtual Environment Setup
✅ `.venv_conversion/` - Isolated environment with PyTorch, ONNX, TensorFlow

---

## ⚠️ Challenge: TFLite Conversion Dependencies

### Issue
Converting ONNX → TensorFlow → TFLite requires many interconnected dependencies:
- `onnx2tf` (ONNX to TensorFlow converter)
- `tf_keras` (Keras compatibility layer)
- `onnx_graphsurgeon` (ONNX graph manipulation)
- `psutil`, `flatbuffers`, `simple-onnx-processing-tools`
- Dependency conflicts (e.g., protobuf versions)

### Attempted Solutions
1. ❌ `onnx2tf` - Missing multiple dependencies
2. ❌ `onnx-tf` - Compatibility issues
3. ✅ **ONNX export** - Clean success

---

## 🔄 Path Forward: Three Options

### **Option A: TensorFlow Lite Micro (RECOMMENDED for ESP32)**
Use TensorFlow Lite for Microcontrollers directly, which is optimized for embedded systems.

**Steps:**
1. Complete TFLite conversion using stable environment (see below)
2. Integrate TFLite Micro runtime into ESP-IDF project
3. Use `stress_model.tflite` with ESP32

**Pros:**
- Industry standard for embedded ML
- Excellent ESP32 support via ESP-NN
- Quantization support (INT8) for speed/memory
- Well-documented API

**Cons:**
- Requires completing TFLite conversion
- ~200KB runtime overhead

**Resources:**
- [TFLite Micro Guide](https://www.tensorflow.org/lite/microcontrollers)
- [ESP-NN Library](https://github.com/espressif/esp-nn)

---

### **Option B: ONNX Runtime (Alternative)**
Use ONNX model directly with ONNX Runtime for embedded devices.

**Steps:**
1. Use existing `stress_model.onnx` (already created ✅)
2. Integrate ONNX Runtime Micro into ESP-IDF
3. Deploy ONNX model directly

**Pros:**
- Skip TFLite conversion entirely
- ONNX model already created and validated
- Microsoft maintains embedded runtime

**Cons:**
- Less mature ESP32 support than TFLite
- Larger runtime footprint
- Requires custom build for ESP32

**Resources:**
- [ONNX Runtime](https://onnxruntime.ai/)
- [ONNX Runtime Embedded](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/mlas)

---

### **Option C: Custom C Implementation**
Manually implement CNN layers in C based on model weights.

**Steps:**
1. Extract weights from `best.pth` (PyTorch checkpoint)
2. Implement Conv1D, BatchNorm, Pooling, Linear layers in C
3. Load weights as constant arrays
4. Manual inference pipeline

**Pros:**
- Full control over optimization
- Minimal memory footprint
- No external runtime dependencies

**Cons:**
- Time-consuming (2-3 days development)
- Manual validation required
- Maintenance burden for model updates

**Resources:**
- Existing signal preprocessor as template
- Weight extraction script needed

---

## 🎯 Recommended Next Steps

### Immediate Action: Complete TFLite Conversion

**Method 1: Cloud-Based Conversion (Fastest)**
```bash
# Use Google Colab or similar with pre-installed dependencies
# Upload stress_model.onnx
# Run conversion script
```

**Method 2: Docker Container (Reliable)**
```bash
# Use official TensorFlow docker image
docker run -it --rm -v $(pwd):/workspace tensorflow/tensorflow:latest-gpu bash
cd /workspace/shadow-firmware
pip install onnx onnx2tf tf-keras
python3 convert_model_to_tflite.py
```

**Method 3: Fresh Virtual Environment (Local)**
```bash
# Create clean environment
python3.11 -m venv .venv_tflite  # Use Python 3.11 for better compatibility
source .venv_tflite/bin/activate
pip install --upgrade pip
pip install tensorflow==2.13.0 onnx==1.14.0 onnx2tf tf-keras
python3 convert_model_to_tflite.py
```

### Once TFLite Model is Ready

1. **Generate C Arrays** (Done automatically by `convert_model_to_tflite.py`)
   - Output: `components/cnn_inference/include/stress_model_data.h`
   - Output: `components/cnn_inference/stress_model_data.c`

2. **Create CNN Inference Component**
   ```
   components/cnn_inference/
   ├── include/
   │   ├── cnn_inference.h
   │   └── stress_model_data.h
   ├── cnn_inference.c
   ├── stress_model_data.c
   └── CMakeLists.txt
   ```

3. **Integrate with Main Firmware**
   ```c
   // In consumer_task()
   cnn_input_tensor_t cnn_input;
   preprocess_for_cnn(&g_sensor_system, &cnn_input);
   
   float stress_prob = cnn_predict(cnn_input.data);
   ble_stress_service_update_probability(stress_prob);
   ```

---

## 📊 Current File Structure

```
shadow-firmware/
├── best.pth                          # Original PyTorch model
├── export_onnx.py                    # ✅ Working ONNX export
├── convert_model_to_tflite.py        # ⚙️ Full conversion (needs deps)
├── convert_model_simple.py           # Alternative converter
├── conversion_requirements.txt       # Python dependencies
├── .venv_conversion/                 # Virtual environment
├── model_output/
│   └── stress_model.onnx             # ✅ ONNX model (431 KB)
└── components/
    ├── signal_preprocessor/          # ✅ Phase 1 complete
    └── cnn_inference/                # 🔜 Phase 2 next
```

---

## 🔍 Technical Details

### ONNX Model Specifications
```
Format: ONNX v13
Size: 431 KB (uncompressed)
Input:
  - Name: "input"
  - Shape: (batch, 4, 240)
  - Type: float32
Output:
  - Name: "output"
  - Shape: (batch, 1)
  - Type: float32 (range 0.0-1.0, sigmoid activated)
```

### Expected TFLite Model Specifications
```
Format: TensorFlow Lite (FlatBuffer)
Size: ~100-150 KB (with INT8 quantization)
Input:
  - Index: 0
  - Shape: [1, 4, 240]
  - Type: float32 (input) → int8 (internal quantization)
Output:
  - Index: 0
  - Shape: [1, 1]
  - Type: float32
Quantization: INT8 for weights and activations
```

### Memory Budget (ESP32-S3)
```
Available SRAM: 512 KB
- Ring buffers: ~4 KB (reduced in Phase 1)
- Preprocessing workspace: ~8 KB
- TFLite arena: ~200-250 KB (estimated)
- Other firmware: ~50 KB
Total used: ~262-312 KB ✅ Fits comfortably
```

---

## 🎓 Lessons Learned

1. **ONNX as intermediate format works well** - Export was clean and validated
2. **Python dependency hell is real** - TFLite conversion has complex dep tree
3. **Docker/Colab are your friends** - Pre-configured environments save time
4. **Global pooling matters** - Model uses AdaptiveAvgPool1d, not flatten
5. **BatchNorm in inference** - Must be in eval mode for stable results

---

## 📝 Action Items

### High Priority
- [ ] Complete TFLite conversion using Docker or Colab
- [ ] Generate C header/source files with model data
- [ ] Create `cnn_inference` component skeleton

### Medium Priority
- [ ] Integrate TFLite Micro runtime into ESP-IDF
- [ ] Test inference speed on ESP32 hardware
- [ ] Validate accuracy: Python vs TFLite

### Future Enhancements
- [ ] INT8 quantization optimization
- [ ] Model pruning for size reduction
- [ ] Inference pipeline profiling

---

## 📚 References

- **ONNX Model**: `model_output/stress_model.onnx`
- **Export Script**: `export_onnx.py`
- **Phase 1 Complete**: `PHASE1_COMPLETE.md`
- **Migration Plan**: `MIGRATION_PLAN.md`

---

**Last Updated:** December 2024  
**Status:** ONNX export complete ✅ | TFLite conversion in progress ⚙️  
**Next Milestone:** Deploy TFLite model on ESP32-S3
