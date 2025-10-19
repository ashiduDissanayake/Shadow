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
    #include "esp_heap_caps.h"
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

// Model data - include INT8 quantized model
#include "stress_model_int8_esp32.h"

#define STRESS_MODEL_INPUT_CHANNELS 4
#define STRESS_MODEL_INPUT_TIMESTEPS 240

// Quantization parameters from INT8 model
#define INPUT_SCALE      0.118650f
#define INPUT_ZERO_POINT -28
#define OUTPUT_SCALE     0.003906f
#define OUTPUT_ZERO_POINT -128

static const char *TAG = "cnn_inference";

// TFLite globals
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

constexpr int kTensorArenaSize = 200 * 1024;  // 200 KB
// Allocate tensor arena in PSRAM (external RAM) instead of SRAM to avoid DRAM overflow
static uint8_t *tensor_arena = nullptr;

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
    
    // Allocate tensor arena in PSRAM to avoid DRAM overflow
    if (tensor_arena == nullptr) {
        tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (tensor_arena == nullptr) {
            ESP_LOGE(TAG, "Failed to allocate %d bytes in PSRAM", kTensorArenaSize);
            return -1;
        }
        ESP_LOGI(TAG, "Allocated %d KB tensor arena in PSRAM", kTensorArenaSize / 1024);
        // Align to 16-byte boundary
        if (((uintptr_t)tensor_arena & 0xF) != 0) {
            ESP_LOGW(TAG, "Tensor arena not 16-byte aligned");
        }
    }
    
    tflite::InitializeTarget();
    
    model = tflite::GetModel(stress_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version mismatch");
        return -2;
    }
    ESP_LOGI(TAG, "Model loaded: %u bytes (INT8 quantized)", stress_model_tflite_len);
    
    static tflite::MicroMutableOpResolver<45> resolver;  // Large enough for all ops
    
    // Add comprehensive set of operations (discovered through iterative testing)
    // Your model is very complex with many dynamic operations!
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddAveragePool2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        // Reduction ops
        resolver.AddMean() != kTfLiteOk ||
        resolver.AddSum() != kTfLiteOk ||
        resolver.AddReduceMax() != kTfLiteOk ||
        // Reshaping and slicing ops
        resolver.AddPad() != kTfLiteOk ||
        resolver.AddExpandDims() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddSqueeze() != kTfLiteOk ||
        resolver.AddShape() != kTfLiteOk ||
        resolver.AddStridedSlice() != kTfLiteOk ||
        resolver.AddSlice() != kTfLiteOk ||
        resolver.AddPack() != kTfLiteOk ||
        resolver.AddUnpack() != kTfLiteOk ||
        resolver.AddConcatenation() != kTfLiteOk ||
        resolver.AddTranspose() != kTfLiteOk ||
        resolver.AddSplit() != kTfLiteOk ||
        resolver.AddSplitV() != kTfLiteOk ||
        // Type conversion
        resolver.AddCast() != kTfLiteOk ||             // Type casting between dtypes
        // Dense ops
        resolver.AddFullyConnected() != kTfLiteOk ||
        // Activations
        resolver.AddRelu() != kTfLiteOk ||
        resolver.AddRelu6() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk ||
        resolver.AddTanh() != kTfLiteOk ||
        resolver.AddLogistic() != kTfLiteOk ||
        // Quantization
        resolver.AddQuantize() != kTfLiteOk ||
        resolver.AddDequantize() != kTfLiteOk ||
        // Element-wise ops
        resolver.AddAdd() != kTfLiteOk ||
        resolver.AddMul() != kTfLiteOk ||
        resolver.AddSub() != kTfLiteOk ||
        resolver.AddDiv() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to add operations");
        return -3;
    }
    
    ESP_LOGI(TAG, "Operations registered: 34 ops including Conv2D, MaxPool2D, Mean, Shape, Slice, Transpose, Cast, Concatenation, Split, Reshape, FullyConnected, Relu, Softmax, Quantize, etc.");
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return -4;
    }
    
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    if (!input_tensor || !output_tensor) {
        ESP_LOGE(TAG, "Failed to get tensors");
        return -5;
    }
    
    ESP_LOGI(TAG, "Tensor arena: %zu / %d bytes (%.1f%% used)", 
             interpreter->arena_used_bytes(), kTensorArenaSize,
             (interpreter->arena_used_bytes() * 100.0) / kTensorArenaSize);
    
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
    
    // For INT8 model, quantize float input to INT8
    int8_t* input_data = input_tensor->data.int8;
    int idx = 0;
    for (int c = 0; c < 4; c++) {
        for (int t = 0; t < 240; t++) {
            // Quantize: int8_value = clamp(float_value / scale + zero_point)
            float quantized = (input->data[c][t] / INPUT_SCALE) + INPUT_ZERO_POINT;
            // Clamp to INT8 range [-128, 127]
            if (quantized < -128.0f) quantized = -128.0f;
            if (quantized > 127.0f) quantized = 127.0f;
            input_data[idx++] = (int8_t)quantized;
        }
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        result->success = false;
        return -2;
    }
    
    // For INT8 model, dequantize INT8 output to float
    int8_t output_quantized = output_tensor->data.int8[0];
    // Dequantize: float_value = (int8_value - zero_point) * scale
    float stress_prob = (output_quantized - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
    
    // Clamp probability to valid range [0.0, 1.0]
    if (stress_prob < 0.0f) stress_prob = 0.0f;
    if (stress_prob > 1.0f) stress_prob = 1.0f;
    
    int64_t end_time = esp_timer_get_time();
    
    result->stress_probability = stress_prob;
    result->inference_time_us = (uint32_t)(end_time - start_time);
    result->success = true;
    
    ESP_LOGI(TAG, "Inference: %.1f%%, %uus (INT8)", 
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
    if (model_size_bytes) *model_size_bytes = stress_model_tflite_len;
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
    
    // Free PSRAM allocation
    if (tensor_arena != nullptr) {
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
        ESP_LOGI(TAG, "Freed tensor arena from PSRAM");
    }
    
    ESP_LOGI(TAG, "Deinitialized");
}

}