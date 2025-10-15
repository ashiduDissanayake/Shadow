# Phase 3B - Real TFLite Micro Implementation Status

## Current Situation

We successfully created the CNN inference component structure and built/flashed the stub implementation. Now we're adding real TensorFlow Lite Micro inference, but encountered C/C++ linkage issues.

## Issue Encountered

When building the real TFLite Micro implementation, we hit compilation errors related to:
1. C's `_Atomic` keyword conflicting with C++'s `<atomic>` header
2. Headers being included in wrong order (extern "C" wrapping C++ templates)

## Root Cause

The `signal_preprocessor.h` header includes `sensor_buffer.h`, which uses C11 `_Atomic` keyword. When this is included from C++ code that also includes TFLite headers (which use C++ `<atomic>`), there's a conflict.

## Solution Strategy

Create a **minimal** C++ implementation file that:
1. Includes TFLite headers FIRST (before any C headers)
2. Only includes necessary C headers via `extern "C"` blocks
3. Does NOT include `signal_preprocessor.h` directly (use forward declarations)

## Action Required: Create cnn_inference.cpp

Save this as `/Users/ashidudissanayake/Dev/Shadow/shadow-firmware/components/cnn_inference/cnn_inference.cpp`:

```cpp
/**
 * @file cnn_inference.cpp
 * @brief CNN inference using TensorFlow Lite Micro
 */

// TFLite headers - MUST be first
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Standard C++ headers
#include <cstring>
#include <cmath>

// Wrap C headers
extern "C" {
    #include "esp_log.h"
    #include "esp_timer.h"
}

// Forward declare types to avoid including problematic headers
typedef struct {
    float data[4][240];  // [channels][timesteps]
} cnn_input_tensor_t;

typedef struct {
    float stress_probability;
    uint32_t inference_time_us;
    bool success;
} cnn_inference_result_t;

typedef struct {
    size_t tensor_arena_size;
    bool enable_profiling;
} cnn_inference_config_t;

// Model data - extern from stress_model_data.c
extern "C" {
    extern const unsigned char g_stress_model_data[];
    extern const unsigned int g_stress_model_data_len;
}

#define STRESS_MODEL_INPUT_CHANNELS 4
#define STRESS_MODEL_INPUT_TIMESTEPS 240

static const char *TAG = "cnn_inference";

// TFLite globals
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

constexpr int kTensorArenaSize = 200 * 1024;  // 200 KB
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

static const tflite::Model* model = nullptr;
static bool g_initialized = false;
static cnn_inference_config_t g_config;

extern "C" {

cnn_inference_config_t cnn_inference_get_default_config(void) {
    cnn_inference_config_t config = {};
    config.tensor_arena_size = kTensorArenaSize;
    config.enable_profiling = false;
    return config;
}

int cnn_inference_init(const cnn_inference_config_t *config) {
    ESP_LOGI(TAG, "Initializing CNN with TFLite Micro...");
    
    if (config == NULL) {
        g_config = cnn_inference_get_default_config();
    } else {
        g_config = *config;
    }
    
    tflite::InitializeTarget();
    
    model = tflite::GetModel(g_stress_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version mismatch");
        return -1;
    }
    ESP_LOGI(TAG, "Model loaded: %u bytes", g_stress_model_data_len);
    
    static tflite::MicroMutableOpResolver<10> resolver;
    
    // Add required operations
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddRelu() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk ||
        resolver.AddQuantize() != kTfLiteOk ||
        resolver.AddDequantize() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to add operations");
        return -2;
    }
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return -3;
    }
    
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    if (!input_tensor || !output_tensor) {
        ESP_LOGE(TAG, "Failed to get tensors");
        return -4;
    }
    
    ESP_LOGI(TAG, "Tensor arena: %zu / %d bytes", 
             interpreter->arena_used_bytes(), kTensorArenaSize);
    
    g_initialized = true;
    ESP_LOGI(TAG, "CNN initialized successfully");
    return 0;
}

int cnn_inference_predict(const cnn_input_tensor_t *input, 
                          cnn_inference_result_t *result) {
    if (!input || !result || !g_initialized || !interpreter) {
        return -1;
    }
    
    int64_t start_time = esp_timer_get_time();
    
    float* input_data = input_tensor->data.f;
    int idx = 0;
    for (int c = 0; c < 4; c++) {
        for (int t = 0; t < 240; t++) {
            input_data[idx++] = input->data[c][t];
        }
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        result->success = false;
        return -2;
    }
    
    float* output_data = output_tensor->data.f;
    float stress_prob = output_data[0];
    
    if (stress_prob < 0.0f) stress_prob = 0.0f;
    if (stress_prob > 1.0f) stress_prob = 1.0f;
    
    int64_t end_time = esp_timer_get_time();
    
    result->stress_probability = stress_prob;
    result->inference_time_us = (uint32_t)(end_time - start_time);
    result->success = true;
    
    ESP_LOGI(TAG, "Inference: %.1f%%, %uus", 
             stress_prob * 100.0f, result->inference_time_us);
    
    return 0;
}

void cnn_inference_get_memory_stats(size_t *arena_used_bytes, 
                                    size_t *arena_total_bytes) {
    if (arena_used_bytes) {
        *arena_used_bytes = interpreter ? interpreter->arena_used_bytes() : 0;
    }
    if (arena_total_bytes) {
        *arena_total_bytes = kTensorArenaSize;
    }
}

void cnn_inference_get_model_info(size_t *model_size_bytes,
                                  int input_shape[3],
                                  int output_shape[2]) {
    if (model_size_bytes) *model_size_bytes = g_stress_model_data_len;
    if (input_shape) {
        input_shape[0] = 1;   // batch
        input_shape[1] = 4;   // channels
        input_shape[2] = 240; // timesteps
    }
    if (output_shape) {
        output_shape[0] = 1;  // batch
        output_shape[1] = 1;  // classes
    }
}

void cnn_inference_deinit(void) {
    interpreter = nullptr;
    input_tensor = nullptr;
    output_tensor = nullptr;
    model = nullptr;
    g_initialized = false;
    ESP_LOGI(TAG, "Deinitialized");
}

} // extern "C"
```

## Next Steps

1. **Create the file above** - copy the content to `cnn_inference.cpp`

2. **Build**:
```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware
idf.py fullclean
idf.py build
```

3. **If you get "Op not registered" errors during init**:
   - Check the error message for missing op names
   - Add them to the resolver (e.g., `resolver.AddMaxPool2D()`, `resolver.AddLogistic()`, etc.)
   - Common additions:
     ```cpp
     resolver.AddMaxPool2D();      // if using MaxPooling
     resolver.AddAveragePool2D();  // if using AvgPooling  
     resolver.AddLogistic();        // if using Sigmoid
     resolver.AddMean();            // if using GlobalAveragePooling
     resolver.AddPad();             // if using Padding
     ```

4. **If tensor arena is too small**:
   - Increase `kTensorArenaSize` (try 250 KB or 300 KB)
   - Check actual usage with `interpreter->arena_used_bytes()`

5. **Flash and monitor**:
```bash
idf.py flash monitor
```

6. **Expected output on boot**:
```
I (xxx) cnn_inference: Initializing CNN with TFLite Micro...
I (xxx) cnn_inference: Model loaded: 124176 bytes
I (xxx) cnn_inference: Tensor arena: 180000 / 204800 bytes
I (xxx) cnn_inference: CNN initialized successfully
```

## Troubleshooting

### If build fails with "undefined reference to cnn_inference_X"
- Check that all functions are wrapped in `extern "C"` block
- Verify CMakeLists.txt has `.cpp` file listed

### If you get linker errors about TFLite
- Ensure `idf_component.yml` has esp-tflite-micro dependency
- Run `idf.py fullclean` then build again

### If inference gives wrong results
- Check tensor shapes match expected (1,4,240) input
- Verify data layout (channels-first vs channels-last)
- Compare first few predictions with Python model

## File Structure

```
components/cnn_inference/
├── CMakeLists.txt              # ✅ Done (has .cpp reference)
├── cnn_inference.cpp           # ⚠️  CREATE THIS FILE
├── stress_model_data.c         # ✅ Done (model data)
└── include/
    ├── cnn_inference.h         # ✅ Done (API)
    └── stress_model_data.h     # ✅ Done (constants)
```

## Success Criteria

✅ Build completes without errors
✅ Firmware size <900 KB  
✅ Init completes successfully on boot
✅ First inference runs in <200ms
✅ Memory stats show reasonable arena usage (<200 KB)
✅ Stress probability output is in [0.0, 1.0] range

## Estimated Time

- File creation: 2 min
- Build + fix ops: 15-30 min
- Flash + test: 10 min  
- **Total: ~30-45 minutes**

## After Success

Once this builds and runs:
1. Test with real sensor data
2. Compare output with Python model
3. Integrate with main firmware consumer_task
4. Update BLE service to send probabilities
5. Proceed to Task 6 (Main Firmware Integration)
