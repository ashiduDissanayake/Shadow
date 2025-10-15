# Shadow Firmware Analysis - Executive Summary

## 📌 What We Found

### Current Model Architecture
Your `best.pth` contains a **CNN model** with this structure:
- **Input:** `(batch, 4 channels, 240 samples)` - resampled sensor signals
- **Architecture:**
  - 2 Conv1D layers (feature extraction)
  - Fully connected layers
  - Sigmoid output
- **Output:** Single probability value `[0.0, 1.0]` indicating stress level

### Preprocessing Requirements
The model expects preprocessed data, NOT raw features:

1. **Collect 60 seconds** of sensor data:
   - ACC (X,Y,Z): 1920 samples @ 32Hz
   - BVP: 3840 samples @ 64Hz
   - EDA: 240 samples @ 4Hz
   - TEMP: 240 samples @ 4Hz

2. **Compute ACC magnitude:** `sqrt(x² + y² + z²)`

3. **Resample to 4Hz:** Linear interpolation to get 240 samples per channel

4. **Z-score normalize:** `(signal - mean) / std` for each channel

5. **Stack:** Create `(4, 240)` tensor with channel order `[ACC, BVP, EDA, TEMP]`

---

## 🔄 What Needs to Change

### Remove (Current System):
- ❌ **Feature extraction** - Computing 30 statistical features
- ❌ **MLP model** - 30 → 64 → 32 → 1 network
- ❌ **FSM** - State machine with consecutive confirmations

### Add (New System):
- ✅ **Signal preprocessing** - Resample + normalize
- ✅ **CNN inference** - Use your trained model
- ✅ **Device pairing** - Owner management via BLE

### Keep (Unchanged):
- ✅ **Producer task** - Sensor data collection
- ✅ **Ring buffers** - 60-second windows
- ✅ **BLE protocol** - Communication structure

---

## 🎯 Key Changes Required

### 1. Replace Preprocessing (consumer_task)

**OLD CODE:**
```c
// Extract 30 statistical features
feature_vector_t features;
extract_features_realtime(&g_sensor_system, &g_feature_workspace, &features);

// Run MLP
float prob = shadow_mlp_predict_probability(features.features);

// Filter through FSM
stress_fsm_process_inference(&g_stress_fsm, prob, now_ms, on_stress_transition);
```

**NEW CODE:**
```c
// Preprocess to (4, 240) tensor
float preprocessed[4][240];
preprocess_for_cnn(&g_sensor_system, preprocessed);

// Run CNN inference
float prob = cnn_model_predict(preprocessed);

// Send probability directly (no FSM)
ble_stress_service_update_probability(prob, now_ms);
```

### 2. Add Device Pairing

**BLE Characteristics to Add:**
```c
- Device UUID (read-only)
- Owner UUID (read/write, stored in NVS)
- Pairing command (write-only)
- Pairing status (read-only)
```

**Pairing Flow:**
```
1. macOS scans for devices
2. User selects unpaired device
3. macOS sends CLAIM command with its UUID
4. ESP32 stores owner UUID in NVS
5. Device now only accepts connections from owner
```

---

## 📂 Files Created

### Analysis & Testing:
- ✅ `test_pytorch_model.py` - Model analysis script
- ✅ `test_data.h` - C arrays for validation
- ✅ `test_data_for_esp32.json` - Full test dataset  
- ✅ `preprocessing_visualization.png` - Visual comparison
- ✅ `MIGRATION_PLAN.md` - Detailed implementation plan

### To Create Next:
- 📝 `components/signal_preprocessor/` - Preprocessing in C
- 📝 `components/cnn_inference/` - TFLite interpreter
- 📝 `components/device_pairing/` - Owner management
- 📝 `convert_model_to_tflite.py` - Model conversion script

---

## 🔧 Implementation Steps

### Phase 1: Preprocessing (2-3 days)
1. Create `signal_preprocessor` component
2. Implement `resample_signal()` - linear interpolation
3. Implement `normalize_signal_zscore()` - z-score normalization
4. Implement `preprocess_for_cnn()` - full pipeline
5. Validate against Python output using `test_data.h`

### Phase 2: Model Conversion (1-2 days)
1. Export PyTorch → ONNX
2. Convert ONNX → TensorFlow
3. Convert TensorFlow → TFLite (int8 quantization)
4. Generate C arrays for embedding
5. Test inference speed and accuracy

### Phase 3: Integration (1 day)
1. Remove FSM component
2. Update consumer task
3. Integrate CNN inference
4. Test end-to-end pipeline

### Phase 4: Device Pairing (2-3 days)
1. Add NVS storage functions
2. Create pairing BLE characteristics
3. Implement pairing logic on ESP32
4. Test with multiple devices

### Phase 5: macOS App (2-3 days)
1. Add device discovery screen
2. Implement pairing flow
3. Update UI for continuous probability
4. Add probability graph visualization

### Phase 6: Testing (2-3 days)
1. Validate preprocessing accuracy
2. Verify CNN outputs match Python
3. Test device pairing
4. End-to-end system test

---

## 💡 Key Insights

### Why This is Better:
1. **More Accurate** - CNN operates on raw signals, captures temporal patterns
2. **Real-Time** - No FSM delay, instant probability updates
3. **Scalable** - Easy to add more sensors or improve model
4. **User-Friendly** - Device pairing makes multi-device management easy

### Potential Challenges:
1. **Memory** - CNN model larger than MLP (~50KB vs ~25KB)
2. **Inference Time** - Need to verify < 100ms
3. **Calibration** - May need to adjust normalization params
4. **BLE Pairing** - Need robust error handling

### Mitigations:
1. **Quantization** - int8 quantization reduces model size 4x
2. **ESP32-S3** - Dual core, plenty of SRAM (512KB)
3. **Test Data** - Use `test_data.h` for validation
4. **NVS Storage** - Reliable ownership persistence

---

## 📊 Memory Budget

```
Component                   Current    New        Delta
─────────────────────────────────────────────────────
Ring Buffers               30 KB      30 KB      0 KB
Feature Extraction         15 KB      -          -15 KB
MLP Model                  25 KB      -          -25 KB
FSM + Event Log           2 KB       -          -2 KB
Signal Preprocessor        -          8 KB       +8 KB
CNN Model (quantized)      -          50 KB      +50 KB
Device Pairing            -          1 KB       +1 KB
─────────────────────────────────────────────────────
Total                      72 KB      89 KB      +17 KB

ESP32-S3 SRAM: 512 KB
Usage: 17% → 17% ✅
```

---

## 🎯 Success Criteria

### Technical:
- [ ] Preprocessing < 0.1% error vs Python
- [ ] CNN inference time < 100ms
- [ ] Model accuracy > 99% (matches PyTorch)
- [ ] BLE connection stable (>99% uptime)
- [ ] Device pairing success rate > 95%

### User Experience:
- [ ] Smooth probability updates (no lag)
- [ ] Easy device discovery
- [ ] Reliable ownership persistence
- [ ] Clear visual feedback on macOS

---

## 🚀 Next Steps

**Immediate Actions:**
1. Review `MIGRATION_PLAN.md` for detailed implementation
2. Start with signal preprocessing implementation
3. Use `test_data.h` to validate preprocessing
4. Move to model conversion once preprocessing validated

**Questions to Answer:**
1. Do you have the model training code? (Need architecture definition)
2. What accuracy threshold is acceptable for TFLite?
3. Any specific requirements for device pairing UX?
4. Should multiple owners be supported?

---

## 📞 Support Files

- **Detailed Plan:** `MIGRATION_PLAN.md`
- **Test Script:** `test_pytorch_model.py`
- **Test Data:** `test_data_for_esp32.json`
- **Validation:** `test_data.h`
- **Visualization:** `preprocessing_visualization.png`

---

**Ready to proceed with implementation? Start with Phase 1: Signal Preprocessing!**
