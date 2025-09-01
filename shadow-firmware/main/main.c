/*
 * ESP32-S3 Shadow Stress Detection Firmware
 * Dual-Core Producer-Consumer Architecture
 * 
 * Core 0: Sensor data sampling (Producer)
 * Core 1: ML inference processing (Consumer)
 */

#include <stdio.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_random.h"

// Shadow components
#include "sensor_buffer.h"
#include "feature_extractor.h"
#include "simple_mlp.h"

static const char *TAG = "ShadowMain";

// Global shared buffer and workspaces
static multi_sensor_buffer_t g_sensor_buffer;
static feature_workspace_t g_feature_workspace;

// Task handles
static TaskHandle_t sensor_producer_handle = NULL;
static TaskHandle_t ml_consumer_handle = NULL;

// Simulation: Mock sensor data generators (calibrated with WESAD dataset)
static float generate_mock_bvp(void) {
    // Simulate BVP with realistic WESAD range: 2.99-307.47, mean=53.20, std=40.85
    static float base_bvp = 53.2f;  // Mean from WESAD
    static float phase = 0.0f;
    phase += 0.05f + (esp_random() % 50) / 10000.0f;
    
    // Add realistic variability
    float variation = sinf(phase) * 20.0f + (esp_random() % 4000 - 2000) / 100.0f;
    base_bvp += variation * 0.1f;
    
    // Clamp to realistic WESAD bounds
    if (base_bvp < 2.99f) base_bvp = 2.99f;
    if (base_bvp > 307.47f) base_bvp = 307.47f;
    return base_bvp;
}

static float generate_mock_acc(int axis) {
    // Simulate accelerometer with realistic WESAD ranges
    // ACC_X: -64.97 to 62.49 (mean=15.42), ACC_Y: -59.28 to 63.93 (mean=-6.18), ACC_Z: -61.59 to 58.71 (mean=8.99)
    static float acc_phase[3] = {0.0f, 1.0f, 2.0f};
    static float axis_means[3] = {15.42f, -6.18f, 8.99f};  // WESAD means
    static float axis_ranges[3][2] = {{-64.97f, 62.49f}, {-59.28f, 63.93f}, {-61.59f, 58.71f}};  // WESAD min/max
    
    acc_phase[axis] += 0.02f + (esp_random() % 30) / 10000.0f;
    float movement = sinf(acc_phase[axis]) * 8.0f + (esp_random() % 2000 - 1000) / 1000.0f;
    float acc_value = axis_means[axis] + movement;
    
    // Clamp to realistic WESAD bounds
    if (acc_value < axis_ranges[axis][0]) acc_value = axis_ranges[axis][0];
    if (acc_value > axis_ranges[axis][1]) acc_value = axis_ranges[axis][1];
    return acc_value;
}

static float generate_mock_eda(void) {
    // Simulate EDA with realistic WESAD range: 0.09-15.62 µS, mean=2.08, std=2.79
    static float base_eda = 2.08f;  // Mean from WESAD
    
    // Add realistic slow drift and noise
    base_eda += (esp_random() % 400 - 200) / 10000.0f;  // Slow drift
    float noise = (esp_random() % 200 - 100) / 1000.0f;  // Measurement noise
    float eda_value = base_eda + noise;
    
    // Clamp to realistic WESAD bounds
    if (eda_value < 0.09f) eda_value = 0.09f;
    if (eda_value > 15.62f) eda_value = 15.62f;
    return eda_value;
}

static float generate_mock_temp(void) {
    // Simulate temperature with realistic WESAD range: 29.39-35.93°C, mean=33.09, std=1.45
    static float base_temp = 33.09f;  // Mean from WESAD
    
    // Add realistic slow temperature changes
    base_temp += (esp_random() % 200 - 100) / 100000.0f;  // Very slow drift
    float noise = (esp_random() % 100 - 50) / 10000.0f;   // Small measurement noise
    float temp_value = base_temp + noise;
    
    // Clamp to realistic WESAD bounds
    if (temp_value < 29.39f) temp_value = 29.39f;
    if (temp_value > 35.93f) temp_value = 35.93f;
    return temp_value;
}

/**
 * Core 0 Task: Sensor Data Producer
 * Continuously samples sensors at their respective rates
 */
void sensor_producer_task(void *param) {
    ESP_LOGI(TAG, "🚀 Starting sensor producer on Core %d", xPortGetCoreID());
    
    TickType_t last_bvp_time = 0;
    TickType_t last_acc_time = 0;
    TickType_t last_eda_time = 0;
    TickType_t last_temp_time = 0;
    
    // Calculate sampling intervals in ticks
    const TickType_t bvp_interval = pdMS_TO_TICKS(1000 / BVP_SAMPLE_RATE);    // 64Hz = ~15.6ms
    const TickType_t acc_interval = pdMS_TO_TICKS(1000 / ACC_SAMPLE_RATE);    // 32Hz = 31.25ms
    const TickType_t eda_interval = pdMS_TO_TICKS(1000 / EDA_SAMPLE_RATE);    // 4Hz = 250ms
    const TickType_t temp_interval = pdMS_TO_TICKS(1000 / TEMP_SAMPLE_RATE);  // 4Hz = 250ms
    
    uint32_t sample_count = 0;
    
    while (1) {
        TickType_t current_time = xTaskGetTickCount();
        
        // Sample BVP at 64Hz
        if ((current_time - last_bvp_time) >= bvp_interval) {
            float bvp_value = generate_mock_bvp();
            buffer_add_sample(&g_sensor_buffer, LAYER_BVP, bvp_value);
            last_bvp_time = current_time;
            sample_count++;
        }
        
        // Sample ACC at 32Hz (all three axes)
        if ((current_time - last_acc_time) >= acc_interval) {
            buffer_add_sample(&g_sensor_buffer, LAYER_ACC_X, generate_mock_acc(0));
            buffer_add_sample(&g_sensor_buffer, LAYER_ACC_Y, generate_mock_acc(1));
            buffer_add_sample(&g_sensor_buffer, LAYER_ACC_Z, generate_mock_acc(2));
            last_acc_time = current_time;
        }
        
        // Sample EDA at 4Hz
        if ((current_time - last_eda_time) >= eda_interval) {
            float eda_value = generate_mock_eda();
            buffer_add_sample(&g_sensor_buffer, LAYER_EDA, eda_value);
            last_eda_time = current_time;
        }
        
        // Sample TEMP at 4Hz
        if ((current_time - last_temp_time) >= temp_interval) {
            float temp_value = generate_mock_temp();
            buffer_add_sample(&g_sensor_buffer, LAYER_TEMP, temp_value);
            last_temp_time = current_time;
        }
        
        // Log progress every 1000 samples
        if (sample_count % 1000 == 0) {
            ESP_LOGI(TAG, "📊 Samples collected: %lu | Buffer ready: %s", 
                    sample_count, 
                    buffer_is_ready_for_processing(&g_sensor_buffer) ? "YES" : "NO");
        }
        
        // Small delay to prevent watchdog issues
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

/**
 * Core 1 Task: ML Inference Consumer
 * Processes 60-second windows every 10 seconds
 */
void ml_consumer_task(void *param) {
    ESP_LOGI(TAG, "🧠 Starting ML consumer on Core %d", xPortGetCoreID());
    
    // Wait for initial data accumulation
    vTaskDelay(pdMS_TO_TICKS(5000)); // 5 second initial delay
    
    uint32_t inference_count = 0;
    
    while (1) {
        // Wait 10 seconds between inferences
        vTaskDelay(pdMS_TO_TICKS(10000));
        
        // Check if buffer has enough data
        if (!buffer_is_ready_for_processing(&g_sensor_buffer)) {
            ESP_LOGW(TAG, "⚠️ Buffer not ready for processing, skipping inference");
            continue;
        }
        
        ESP_LOGI(TAG, "🔄 Starting inference #%lu", inference_count++);
        
        // Extract features
        feature_vector_t features;
        int feature_result = extract_features(&g_sensor_buffer, &g_feature_workspace, &features);
        
        if (feature_result != 0) {
            ESP_LOGE(TAG, "❌ Feature extraction failed: %d", feature_result);
            continue;
        }
        
        ESP_LOGI(TAG, "✅ Feature extraction completed in %lu ms", features.extraction_time_ms);
        
        // Run ML inference
        uint32_t ml_start = xTaskGetTickCount() * portTICK_PERIOD_MS;
        float stress_probability = shadow_mlp_predict_probability(features.features);
        int stress_class = shadow_mlp_predict_class(features.features);
        uint32_t ml_time = (xTaskGetTickCount() * portTICK_PERIOD_MS) - ml_start;
        
        // Log results
        ESP_LOGI(TAG, "🎯 ML Inference Results:");
        ESP_LOGI(TAG, "   Stress Probability: %.3f", stress_probability);
        ESP_LOGI(TAG, "   Stress Class: %s", stress_class ? "STRESS" : "NORMAL");
        ESP_LOGI(TAG, "   Inference Time: %lu ms", ml_time);
        ESP_LOGI(TAG, "   Total Processing: %lu ms", features.extraction_time_ms + ml_time);
        
        // Buffer status
        for (int layer = 0; layer < NUM_SENSOR_LAYERS; layer++) {
            uint16_t count = buffer_get_count(&g_sensor_buffer, layer);
            ESP_LOGI(TAG, "   Layer %d samples: %d", layer, count);
        }
        
        ESP_LOGI(TAG, "---");
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "🌟 Shadow Stress Detection Firmware v1.0");
    ESP_LOGI(TAG, "ESP32-S3 Dual-Core Producer-Consumer Architecture");
    
    // Initialize system components
    ESP_LOGI(TAG, "Initializing sensor buffer...");
    if (buffer_init(&g_sensor_buffer) != 0) {
        ESP_LOGE(TAG, "Failed to initialize sensor buffer");
        return;
    }
    
    ESP_LOGI(TAG, "Initializing feature extractor...");
    if (feature_extractor_init(&g_feature_workspace) != 0) {
        ESP_LOGE(TAG, "Failed to initialize feature extractor");
        return;
    }
    
    ESP_LOGI(TAG, "System initialization complete!");
    ESP_LOGI(TAG, "Memory usage - Buffer: %lu bytes, Feature workspace: %lu bytes", 
            buffer_get_memory_usage(), feature_extractor_get_memory_usage());
    
    // Create producer task on Core 0
    xTaskCreatePinnedToCore(
        sensor_producer_task,
        "sensor_producer",
        4096,  // Stack size
        NULL,  // Parameters
        5,     // Priority (high for real-time sampling)
        &sensor_producer_handle,
        0      // Core 0
    );
    
    // Create consumer task on Core 1
    xTaskCreatePinnedToCore(
        ml_consumer_task,
        "ml_consumer",
        8192,  // Larger stack for ML processing
        NULL,  // Parameters
        3,     // Priority (lower than producer)
        &ml_consumer_handle,
        1      // Core 1
    );
    
    ESP_LOGI(TAG, "🚀 Tasks created successfully!");
    ESP_LOGI(TAG, "Producer task running on Core 0");
    ESP_LOGI(TAG, "Consumer task running on Core 1");
    ESP_LOGI(TAG, "Real-time stress detection system active!");
}