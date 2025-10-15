# Phase 3: CNN Integration on ESP32-S3

## 🎯 Goal
Integrate TensorFlow Lite Micro runtime and CNN inference into ESP32-S3 firmware, replacing the existing feature extraction + MLP pipeline.

---

## 📊 Current Status

✅ **Phase 1: Signal Preprocessing** - COMPLETE  
✅ **Phase 2: Model Conversion** - COMPLETE  
🔄 **Phase 3: CNN Integration** - IN PROGRESS

**Completed (Phase 2):**
- ✅ PyTorch model analyzed (109K parameters)
- ✅ Signal preprocessor implemented in C
- ✅ ONNX model exported (431 KB)
- ✅ TFLite model quantized (121 KB)
- ✅ C arrays generated (stress_model_data.h/c)

**Current Task:** Add TFLite Micro runtime and create cnn_inference component

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     ESP32-S3 Firmware                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Sensor Sampling (25 Hz)                                   │
│         │                                                   │
│         ▼                                                   │
│  Signal Preprocessor (4 Hz downsampling)                   │
│    • compute_acc_magnitude()                               │
│    • normalize_signal_zscore()                             │
│    • preprocess_for_cnn() → (4, 240) tensor                │
│         │                                                   │
│         ▼                                                   │
│  CNN Inference (TFLite Micro)    ◄── NEW in Phase 3       │
│    • Load model from g_stress_model_data[]                 │
│    • cnn_predict(input[4][240]) → float                    │
│    • Tensor arena: ~200 KB                                 │
│    • Inference time: <100ms                                │
│         │                                                   │
│         ▼                                                   │
│  BLE Service (stress probability)                          │
│    • Update characteristic with probability [0.0-1.0]      │
│    • Transmit to macOS app                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Implementation Plan

### Task 5: Create cnn_inference Component (4-6 hours)

**Deliverables:**
1. Add TFLite Micro library to ESP-IDF project
2. Create `components/cnn_inference/` with implementation
3. Implement CNN inference API
4. Test on ESP32-S3 hardware

---

## 🔧 Step-by-Step Implementation

### Step 1: Add TFLite Micro to ESP-IDF (30 minutes)

**Option A: Use ESP-NN Optimized TFLite Micro (Recommended)**

ESP-NN provides hardware-optimized TFLite Micro for ESP32:

```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware

# Clone ESP-NN (includes TFLite Micro with ESP32 optimizations)
cd components
git clone --recursive https://github.com/espressif/esp-nn.git
cd ..
```

**Option B: Use Official TFLite Micro (Simpler, less optimized)**

```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware

# Add tensorflow-lite-micro as ESP-IDF component
idf.py add-dependency "espressif/esp-tflite-micro^1.3.1"
```

**Choose Option A for better performance!**

---

### Step 2: Create cnn_inference Component Structure (15 minutes)

```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware/components/cnn_inference

# Files already exist:
# ✅ include/stress_model_data.h
# ✅ stress_model_data.c

# Create new files:
touch include/cnn_inference.h
touch cnn_inference.c
touch CMakeLists.txt
```

**Directory structure:**
```
components/cnn_inference/
├── CMakeLists.txt              ← ESP-IDF build config
├── include/
│   ├── cnn_inference.h         ← Public API
│   └── stress_model_data.h     ✅ Already generated
├── cnn_inference.c             ← Implementation
└── stress_model_data.c         ✅ Already generated (758 KB)
```

---

### Step 3: Implement CMakeLists.txt (10 minutes)

Create `components/cnn_inference/CMakeLists.txt`:

```cmake
idf_component_register(
    SRCS 
        "cnn_inference.c"
        "stress_model_data.c"
    INCLUDE_DIRS 
        "include"
    REQUIRES 
        "esp-tflite-micro"  # or "esp-nn" if using Option A
        "signal_preprocessor"
)
```

---

### Step 4: Implement cnn_inference.h (20 minutes)

Create `components/cnn_inference/include/cnn_inference.h`:

```c
#ifndef CNN_INFERENCE_H
#define CNN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "signal_preprocessor.h"

/**
 * @brief CNN inference result structure
 */
typedef struct {
    float stress_probability;  ///< Stress probability [0.0-1.0]
    uint32_t inference_time_ms;  ///< Inference latency in milliseconds
    bool success;  ///< True if inference succeeded
} cnn_inference_result_t;

/**
 * @brief Initialize CNN inference engine
 * 
 * Loads the TFLite model and allocates tensor arena.
 * Must be called once during firmware initialization.
 * 
 * @return 0 on success, negative error code on failure
 */
int cnn_inference_init(void);

/**
 * @brief Run CNN inference on preprocessed sensor data
 * 
 * @param input Preprocessed CNN input tensor (4 channels x 240 samples)
 * @param result Output structure containing stress probability and metadata
 * @return 0 on success, negative error code on failure
 */
int cnn_inference_predict(const cnn_input_tensor_t *input, cnn_inference_result_t *result);

/**
 * @brief Get memory usage statistics
 * 
 * @param arena_used_bytes Output: bytes used in tensor arena
 * @param arena_total_bytes Output: total tensor arena size
 */
void cnn_inference_get_memory_stats(size_t *arena_used_bytes, size_t *arena_total_bytes);

/**
 * @brief Deinitialize CNN inference engine (free resources)
 */
void cnn_inference_deinit(void);

#ifdef __cplusplus
}
#endif

#endif // CNN_INFERENCE_H
```

---

### Step 5: Implement cnn_inference.c (2-3 hours)

Create `components/cnn_inference/cnn_inference.c`:

```c
#include "cnn_inference.h"
#include "stress_model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_log.h"
#include "esp_timer.h"
#include <string.h>

static const char *TAG = "cnn_inference";

// TFLite Micro globals
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;

// Tensor arena (adjust size based on model requirements)
constexpr int kTensorArenaSize = 200 * 1024;  // 200 KB
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

// Model
static const tflite::Model *model = nullptr;

// Op resolver
static tflite::MicroMutableOpResolver<10> *op_resolver = nullptr;

int cnn_inference_init(void) {
    ESP_LOGI(TAG, "Initializing CNN inference engine...");
    
    // Load model
    model = tflite::GetModel(g_stress_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %d doesn't match supported version %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }
    ESP_LOGI(TAG, "Model loaded: %u bytes", g_stress_model_data_len);
    
    // Create op resolver and add required ops
    static tflite::MicroMutableOpResolver<10> resolver;
    
    // Add operations used by the model
    // Adjust based on actual model operations (check with Netron)
    resolver.AddConv2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    // Add more ops as needed: AddMaxPool2D(), AddRelu(), etc.
    
    op_resolver = &resolver;
    
    // Build interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return -2;
    }
    
    // Get input/output tensors
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    // Validate tensor shapes
    if (input_tensor->dims->size != 3 ||
        input_tensor->dims->data[0] != STRESS_MODEL_INPUT_BATCH ||
        input_tensor->dims->data[1] != STRESS_MODEL_INPUT_CHANNELS ||
        input_tensor->dims->data[2] != STRESS_MODEL_INPUT_TIMESTEPS) {
        ESP_LOGE(TAG, "Input tensor shape mismatch");
        return -3;
    }
    
    if (output_tensor->dims->size != 2 ||
        output_tensor->dims->data[0] != STRESS_MODEL_OUTPUT_BATCH ||
        output_tensor->dims->data[1] != STRESS_MODEL_OUTPUT_SIZE) {
        ESP_LOGE(TAG, "Output tensor shape mismatch");
        return -4;
    }
    
    // Log memory usage
    size_t used_bytes = interpreter->arena_used_bytes();
    ESP_LOGI(TAG, "Tensor arena: %u / %u bytes used (%.1f%%)",
             used_bytes, kTensorArenaSize, (used_bytes * 100.0f) / kTensorArenaSize);
    
    ESP_LOGI(TAG, "CNN inference engine initialized successfully");
    return 0;
}

int cnn_inference_predict(const cnn_input_tensor_t *input, cnn_inference_result_t *result) {
    if (!interpreter || !input || !result) {
        ESP_LOGE(TAG, "Invalid parameters");
        return -1;
    }
    
    // Start timing
    int64_t start_time = esp_timer_get_time();
    
    // Copy input data to tensor (flatten 4x240 to 960 elements)
    float *input_data = input_tensor->data.f;
    memcpy(input_data, input->data, STRESS_MODEL_INPUT_SIZE * sizeof(float));
    
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        result->success = false;
        return -2;
    }
    
    // Get output
    float stress_prob = output_tensor->data.f[0];
    
    // Clamp to [0.0, 1.0]
    if (stress_prob < 0.0f) stress_prob = 0.0f;
    if (stress_prob > 1.0f) stress_prob = 1.0f;
    
    // Calculate inference time
    int64_t end_time = esp_timer_get_time();
    uint32_t inference_time_ms = (uint32_t)((end_time - start_time) / 1000);
    
    // Populate result
    result->stress_probability = stress_prob;
    result->inference_time_ms = inference_time_ms;
    result->success = true;
    
    ESP_LOGD(TAG, "Inference: prob=%.3f, time=%ums", stress_prob, inference_time_ms);
    
    return 0;
}

void cnn_inference_get_memory_stats(size_t *arena_used_bytes, size_t *arena_total_bytes) {
    if (interpreter) {
        *arena_used_bytes = interpreter->arena_used_bytes();
    } else {
        *arena_used_bytes = 0;
    }
    *arena_total_bytes = kTensorArenaSize;
}

void cnn_inference_deinit(void) {
    // TFLite Micro uses static allocation, so nothing to free
    interpreter = nullptr;
    input_tensor = nullptr;
    output_tensor = nullptr;
    model = nullptr;
    ESP_LOGI(TAG, "CNN inference engine deinitialized");
}
```

---

### Step 6: Update Main CMakeLists.txt (5 minutes)

Edit `main/CMakeLists.txt` to add cnn_inference dependency:

```cmake
idf_component_register(
    SRCS 
        "main_realtime.c"
        # ... other source files
    INCLUDE_DIRS 
        "."
        "include"
    REQUIRES 
        "sensor_system"
        "signal_preprocessor"
        "cnn_inference"  # ← ADD THIS
        # ... other dependencies
)
```

---

### Step 7: Integrate with Main Firmware (1 hour)

Edit `main/main_realtime.c` in the consumer task:

```c
#include "cnn_inference.h"

// In app_main():
void app_main(void) {
    // ... existing initialization
    
    // Initialize CNN inference
    if (cnn_inference_init() != 0) {
        ESP_LOGE(TAG, "Failed to initialize CNN inference");
        // Handle error
    }
    
    // ... rest of initialization
}

// In consumer_task():
static void consumer_task(void *pvParameters) {
    cnn_input_tensor_t cnn_input;
    cnn_inference_result_t cnn_result;
    
    while (1) {
        // Wait for 60 seconds of data
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        // Preprocess signals for CNN
        int preprocess_status = preprocess_for_cnn(&g_sensor_system, &cnn_input);
        if (preprocess_status != 0) {
            ESP_LOGW(TAG, "Preprocessing failed: %d", preprocess_status);
            continue;
        }
        
        // Run CNN inference
        int inference_status = cnn_inference_predict(&cnn_input, &cnn_result);
        if (inference_status != 0 || !cnn_result.success) {
            ESP_LOGE(TAG, "Inference failed: %d", inference_status);
            continue;
        }
        
        // Log result
        ESP_LOGI(TAG, "Stress probability: %.1f%% (inference: %ums)",
                 cnn_result.stress_probability * 100.0f,
                 cnn_result.inference_time_ms);
        
        // Update BLE characteristic
        ble_stress_service_update_probability(cnn_result.stress_probability);
        
        // TODO: Remove old FSM code
        // TODO: Remove feature extraction code
    }
}
```

---

## 🧪 Testing Strategy

### Unit Tests (Test on ESP32-S3)

1. **Model Loading Test**
   ```c
   TEST_CASE("CNN inference init", "[cnn]") {
       TEST_ASSERT_EQUAL(0, cnn_inference_init());
       size_t used, total;
       cnn_inference_get_memory_stats(&used, &total);
       TEST_ASSERT_GREATER_THAN(0, used);
       TEST_ASSERT_LESS_THAN(total, used);
   }
   ```

2. **Inference Test with Known Input**
   ```c
   TEST_CASE("CNN inference with zeros", "[cnn]") {
       cnn_input_tensor_t input = {0};
       cnn_inference_result_t result;
       TEST_ASSERT_EQUAL(0, cnn_inference_predict(&input, &result));
       TEST_ASSERT_TRUE(result.success);
       TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, result.stress_probability);
   }
   ```

3. **Performance Test**
   ```c
   TEST_CASE("CNN inference latency", "[cnn]") {
       cnn_input_tensor_t input;
       // Fill with random data
       cnn_inference_result_t result;
       cnn_inference_predict(&input, &result);
       TEST_ASSERT_LESS_THAN(100, result.inference_time_ms);  // <100ms
   }
   ```

### Integration Tests

1. **Full Pipeline Test**
   - Capture 60 seconds of real sensor data
   - Run preprocessing + CNN inference
   - Verify probability in [0.0, 1.0]
   - Check inference time <100ms

2. **Memory Test**
   - Monitor heap usage during inference
   - Verify no memory leaks (run 1000 inferences)
   - Check tensor arena usage

3. **Accuracy Test**
   - Compare ESP32 inference with Python model
   - Use same input data
   - Verify output difference <1%

---

## 📊 Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| **Model Size** | ~121 KB | ✅ Achieved |
| **Tensor Arena** | ~200 KB | May need adjustment |
| **Total RAM** | ~330 KB | Model + arena + stack |
| **Inference Time** | <100ms | ESP32-S3 @ 240MHz |
| **Accuracy** | >90% | Match Python model |

---

## 🐛 Troubleshooting

### Issue: "AllocateTensors() failed"
**Solution:** Increase `kTensorArenaSize` in cnn_inference.c

### Issue: "Operation not supported"
**Solution:** Add missing ops to MicroMutableOpResolver (check Netron)

### Issue: "Inference too slow (>100ms)"
**Solution:** 
- Use ESP-NN optimized ops
- Enable ESP32-S3 vector extensions
- Consider model pruning

### Issue: "Output values out of range"
**Solution:** Check quantization parameters, verify preprocessing normalization

---

## 📝 Next Steps (After Task 5)

**Task 6: Integrate CNN with Main Firmware** (2-3 hours)
- Replace feature extraction code with CNN inference
- Remove FSM state machine
- Update BLE service for continuous probability

**Task 7: Device Pairing** (4-6 hours)
- Add device UUID generation
- Implement pairing BLE characteristics
- Add NVS storage for owner persistence

**Task 8: macOS App Updates** (1-2 days)
- Device discovery screen
- Pairing flow UI
- Continuous probability display

---

## 🎯 Success Criteria

✅ CNN inference runs successfully on ESP32-S3  
✅ Inference latency <100ms  
✅ Memory usage <350 KB total  
✅ Output matches Python model (difference <1%)  
✅ No memory leaks after 1000 inferences  
✅ BLE updates with stress probability every 60 seconds  

---

**Current Phase:** 3 (CNN Integration)  
**Current Task:** 5 (Create cnn_inference component)  
**Estimated Time:** 4-6 hours  
**Blocking:** None (all prerequisites complete)

Let's start implementing! 🚀
