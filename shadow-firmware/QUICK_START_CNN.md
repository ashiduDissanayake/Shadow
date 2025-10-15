# 🚀 Quick Start Guide - CNN Inference Integration

## What You Have Now

✅ **Phase 2 Complete:** TFLite model (121 KB) converted and embedded as C arrays  
🔄 **Phase 3 Started:** CNN inference component created with stub implementation

---

## Test the Stub Implementation (5 minutes)

### 1. Build Firmware
```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware

# Source ESP-IDF (adjust path if needed)
. $HOME/esp/esp-idf/export.sh

# Build
idf.py build
```

**Expected output:**
```
Building ESP-IDF components
[100/100] Linking CXX executable shadow-firmware.elf
...
Project build complete.
```

### 2. Flash to ESP32-S3
```bash
idf.py -p /dev/cu.usbserial-XXXXXXXX flash monitor
```

### 3. Look for These Logs
```
I cnn_inference: Initializing CNN inference engine (STUB VERSION)...
W cnn_inference: ⚠️  This is a STUB implementation for testing!
I cnn_inference: Model loaded: 124176 bytes (121.27 KB)
I cnn_inference: Tensor arena: 200 KB allocated
I cnn_inference: CNN inference engine initialized (STUB)
```

When stress detection runs:
```
I cnn_inference: Inference (STUB): prob=0.673 (67.3%), time=87us
```

---

## Add Real CNN Inference (Next Session)

### Option A: ESP-NN (Recommended - Hardware Optimized)

```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware/components
git clone --recursive https://github.com/espressif/esp-nn.git
cd ..
```

Update `components/cnn_inference/CMakeLists.txt`:
```cmake
idf_component_register(
    SRCS 
        "cnn_inference.c"
        "stress_model_data.c"
    INCLUDE_DIRS 
        "include"
    REQUIRES 
        "signal_preprocessor"
        "esp-nn"  # Add this
)
```

### Option B: Official TFLite Micro (Simpler)

```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware
idf.py add-dependency "espressif/esp-tflite-micro^1.3.1"
```

Update `components/cnn_inference/CMakeLists.txt`:
```cmake
idf_component_register(
    SRCS 
        "cnn_inference.c"
        "stress_model_data.c"
    INCLUDE_DIRS 
        "include"
    REQUIRES 
        "signal_preprocessor"
        "esp-tflite-micro"  # Add this
)
```

---

## File Summary

### Created Files
```
components/cnn_inference/
├── CMakeLists.txt                  # ESP-IDF build config
├── include/
│   ├── cnn_inference.h             # Public API (your code calls this)
│   └── stress_model_data.h         # Model constants (auto-generated)
├── cnn_inference.c                 # Stub implementation
└── stress_model_data.c             # 121 KB model data (auto-generated)

main/
└── CMakeLists.txt                  # Updated with cnn_inference dependency
```

### Generated Files
- `PHASE3_CNN_INTEGRATION.md` - Full implementation guide
- `PHASE3_PROGRESS.md` - Progress report
- `QUICK_START_CNN.md` - This file

---

## API Usage Example

```c
#include "cnn_inference.h"
#include "signal_preprocessor.h"

void app_main(void) {
    // Initialize CNN inference (once at startup)
    cnn_inference_config_t config = cnn_inference_get_default_config();
    int ret = cnn_inference_init(&config);
    if (ret != 0) {
        ESP_LOGE(TAG, "CNN init failed: %d", ret);
        return;
    }
    
    // ... rest of initialization
}

void consumer_task(void *pvParameters) {
    while (1) {
        // Wait for 60 seconds of data
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        // Preprocess signals
        cnn_input_tensor_t cnn_input;
        preprocess_for_cnn(&g_sensor_system, &cnn_input);
        
        // Run CNN inference
        cnn_inference_result_t result;
        cnn_inference_predict(&cnn_input, &result);
        
        if (result.success) {
            ESP_LOGI(TAG, "Stress: %.1f%% (inference: %uus)",
                     result.stress_probability * 100.0f,
                     result.inference_time_us);
            
            // Update BLE
            ble_stress_service_update_probability(result.stress_probability);
        }
    }
}
```

---

## Memory Usage

```
ESP32-S3 Memory Map:
├── Model (Flash):        121 KB ✅
├── Tensor Arena (SRAM):  200 KB ⏳ (when TFLite added)
├── Stack/Heap:           ~50 KB
└── Other components:     ~140 KB
Total SRAM used:          ~390 KB / 512 KB (76%)
```

---

## Current Status

| Component | Status | Size |
|-----------|--------|------|
| Model conversion | ✅ Complete | 121 KB |
| C arrays generation | ✅ Complete | 758 KB source |
| Component structure | ✅ Complete | - |
| API design | ✅ Complete | - |
| Stub implementation | ✅ Complete | Works now! |
| TFLite Micro | ⏳ Pending | Add next |
| Real CNN inference | ⏳ Pending | After TFLite |

---

## Next Steps

1. **Test stub implementation** (5 min)
   - Build and flash firmware
   - Verify model loads
   - Check heuristic inference works

2. **Add TFLite Micro** (30 min)
   - Choose ESP-NN or official
   - Update CMakeLists.txt
   - Rebuild to verify

3. **Implement real inference** (2-3 hours)
   - Replace stub with TFLite code
   - Add required operations
   - Test and validate

4. **Optimize** (1 hour)
   - Tune tensor arena size
   - Measure inference time
   - Compare with Python model

---

## Troubleshooting

### Build fails with "component not found"
- Check `main/CMakeLists.txt` includes `cnn_inference`
- Verify `components/cnn_inference/CMakeLists.txt` exists

### "Model data not found" error
- Verify `stress_model_quant.tflite` was in `model_output/`
- Re-run `python3 generate_c_arrays.py`

### Out of memory during build
- Model data (758 KB source) is large but compiles to 121 KB binary
- This is expected and normal

### Inference returns wrong values
- Stub uses heuristic (not real CNN)
- Add TFLite Micro for real inference
- See `PHASE3_CNN_INTEGRATION.md` for details

---

## Resources

- **Full guide:** `PHASE3_CNN_INTEGRATION.md`
- **Progress report:** `PHASE3_PROGRESS.md`
- **Model specs:** `components/cnn_inference/include/stress_model_data.h`
- **API docs:** `components/cnn_inference/include/cnn_inference.h`

---

**Ready to test?** Build and flash the firmware now! 🚀

The stub implementation will work immediately and show you that the integration is correct.
Then add TFLite Micro for real CNN inference in the next session.
