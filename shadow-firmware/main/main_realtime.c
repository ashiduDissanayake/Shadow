/*
 * ESP32-S3 Real-Time Producer-Consumer Stress Detection System
 * 
 * ARCHITECTURE IMPLEMENTATION:
 * ============================
 * 
 * PRODUCER (Core 0): ISR-based data ingestion
 * - GDMA ISRs for high-frequency sensors (BVP@64Hz, ACC@32Hz)
 * - Timer ISRs for low-frequency sensors (EDA@4Hz, TEMP@4Hz)
 * - Atomic write pointer updates
 * - Automatic ML task signaling via check_and_signal_ml_ready()
 * 
 * CONSUMER (Core 1): Event-driven ML processing
 * - Waits on semaphore (no polling)
 * - Extracts 60-second windows when signaled
 * - Processes features and runs ML inference
 * - Updates batch counter for next iteration
 * 
 * COORDINATION: Global batch counter + atomic operations
 * - Tracks "weakest link" across all sensors
 * - Guarantees temporal alignment
 * - Implements 10-second sliding window
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
#include "driver/gptimer.h"

// Shadow components
#include "realtime_sensor_buffer.h"
#include "feature_extractor.h"
#include "simple_mlp.h"
#include "stress_fsm.h"
#include "event_log.h"
#include "ble_stress_service.h"

static const char *TAG = "ShadowRealTime";

// Task handles
static TaskHandle_t producer_task_handle = NULL;
static TaskHandle_t consumer_task_handle = NULL;

// Timer handles for ISR-based sampling
static gptimer_handle_t bvp_timer = NULL;
static gptimer_handle_t acc_timer = NULL;
static gptimer_handle_t eda_timer = NULL;
static gptimer_handle_t temp_timer = NULL;

// Feature extraction workspace
static feature_workspace_t g_feature_workspace;

// Stress monitoring system
static stress_fsm_context_t g_stress_fsm;
static event_log_context_t g_event_log;

// Statistics tracking
static uint32_t total_inferences = 0;
static uint32_t total_samples_collected = 0;
static uint32_t total_state_transitions = 0;

// Function declarations
int extract_features_realtime(realtime_sensor_system_t *sensor_system, 
                             feature_workspace_t *workspace, 
                             feature_vector_t *result);

// Stress transition callback function
void on_stress_transition(const stress_state_transition_t *transition);

// Stress FSM event callback
// Stress transition callback function
void on_stress_transition(const stress_state_transition_t *transition);

// === REALISTIC SENSOR SIMULATION ===
// Using dataset ranges for realistic mock data generation
// ISR-SAFE: Integer-only operations, no floating point in ISR context

static int32_t generate_mock_bvp_int(void) {
    // WESAD dataset BVP ranges: mean ≈ 53.2, realistic variation
    static int32_t base_bvp_int = 5320;  // 53.2 * 100 for fixed-point
    static uint32_t counter = 0;
    
    counter++;
    
    // Simple pseudo-random using counter (ISR-safe)
    uint32_t pseudo_rand = (counter * 1103515245 + 12345) & 0x7FFFFFFF;
    int32_t variation = (int32_t)(pseudo_rand % 4000) - 2000;  // -20.00 to +20.00
    base_bvp_int += variation / 10;  // Small incremental change
    
    // Clamp to realistic bounds (fixed-point)
    if (base_bvp_int < 299) base_bvp_int = 299;        // 2.99
    if (base_bvp_int > 30747) base_bvp_int = 30747;    // 307.47
    
    return base_bvp_int;  // Return integer (fixed-point)
}

static int32_t generate_mock_acc_int(int axis) {
    // WESAD dataset ACC ranges - ISR-SAFE integer operations
    static uint32_t acc_counter[3] = {0, 1000, 2000};
    static int32_t axis_means_int[3] = {1542, -618, 899};  // * 100 for fixed-point
    static int32_t axis_ranges_int[3][2] = {{-6497, 6249}, {-5928, 6393}, {-6159, 5871}};  // * 100
    
    acc_counter[axis] += 17 + (axis * 7);  // Different increments for each axis
    
    // Simple pseudo-random using counter (ISR-safe)
    uint32_t pseudo_rand = (acc_counter[axis] * 1103515245 + 12345 + axis * 4567) & 0x7FFFFFFF;
    int32_t movement = ((int32_t)(pseudo_rand % 2000) - 1000);  // -10.00 to +10.00 range
    int32_t acc_value_int = axis_means_int[axis] + movement;
    
    // Clamp to realistic bounds
    if (acc_value_int < axis_ranges_int[axis][0]) acc_value_int = axis_ranges_int[axis][0];
    if (acc_value_int > axis_ranges_int[axis][1]) acc_value_int = axis_ranges_int[axis][1];
    
    return acc_value_int;  // Return integer (fixed-point)
}

static int32_t generate_mock_eda_int(void) {
    // WESAD dataset EDA ranges: 0.09-15.62 µS, mean=2.08 - ISR-SAFE
    static int32_t base_eda_int = 208;  // 2.08 * 100 for fixed-point
    static uint32_t eda_counter = 0;
    
    eda_counter += 23;  // Different increment for EDA
    
    // Simple pseudo-random using counter (ISR-safe)
    uint32_t pseudo_rand = (eda_counter * 1103515245 + 12345 + 7891) & 0x7FFFFFFF;
    int32_t drift = ((int32_t)(pseudo_rand % 400) - 200) / 100;  // Slow drift
    int32_t noise = ((int32_t)((pseudo_rand >> 8) % 200) - 100) / 10;  // Measurement noise
    base_eda_int += drift;
    int32_t eda_value_int = base_eda_int + noise;
    
    // Clamp to realistic bounds (fixed-point)
    if (eda_value_int < 9) eda_value_int = 9;        // 0.09
    if (eda_value_int > 1562) eda_value_int = 1562;  // 15.62
    
    return eda_value_int;  // Return integer (fixed-point)
}

static int32_t generate_mock_temp_int(void) {
    // WESAD dataset TEMP ranges: 29.39-35.93°C, mean=33.09 - ISR-SAFE
    static int32_t base_temp_int = 3309;  // 33.09 * 100 for fixed-point
    static uint32_t temp_counter = 0;
    
    temp_counter += 31;  // Different increment for temperature
    
    // Simple pseudo-random using counter (ISR-safe)
    uint32_t pseudo_rand = (temp_counter * 1103515245 + 12345 + 9876) & 0x7FFFFFFF;
    int32_t drift = ((int32_t)(pseudo_rand % 200) - 100) / 1000;  // Very slow drift
    int32_t noise = ((int32_t)((pseudo_rand >> 8) % 100) - 50) / 100;  // Small measurement noise
    base_temp_int += drift;
    int32_t temp_value_int = base_temp_int + noise;
    
    // Clamp to realistic bounds (fixed-point)
    if (temp_value_int < 2939) temp_value_int = 2939;  // 29.39
    if (temp_value_int > 3593) temp_value_int = 3593;  // 35.93
    
    return temp_value_int;  // Return integer (fixed-point)
}

// === STRESS TRANSITION CALLBACK ===

/**
 * Called whenever the stress FSM confirms a state transition
 * This is where we log events and trigger BLE communications
 */
void on_stress_transition(const stress_state_transition_t *transition) {
    if (!transition) return;
    
    ESP_LOGI(TAG, "🔄 STRESS TRANSITION DETECTED!");
    ESP_LOGI(TAG, "   %s → %s", 
             stress_fsm_state_to_string(transition->from_state),
             stress_fsm_state_to_string(transition->to_state));
    ESP_LOGI(TAG, "   Confidence: %.3f", transition->confidence_score);
    ESP_LOGI(TAG, "   Duration in prev state: %lu ms", transition->duration_prev_state_ms);
    
    // Get current sensor quality and battery voltage
    uint8_t sensor_quality = 85; // TODO: Calculate from actual sensor data
    uint16_t battery_mv = 3300;   // TODO: Read from ADC
    
    // Log the event to our ring buffer
    uint8_t sequence = event_log_add_transition(&g_event_log, transition, 
                                               sensor_quality, battery_mv);
    
    if (sequence != EVENT_LOG_INVALID_SEQUENCE) {
        ESP_LOGI(TAG, "   ✅ Event logged with sequence #%d", sequence);
        total_state_transitions++;
        
        // Update BLE advertisement with new state and sequence
        if (ble_stress_service_update_advertisement(battery_mv, sensor_quality) == 0) {
            ESP_LOGI(TAG, "   📡 BLE advertisement updated");
        }
        
        // If connected, send real-time notification
        if (ble_stress_service_is_connected() && ble_stress_service_notifications_enabled()) {
            ble_stress_service_notify_fsm_state();
            ESP_LOGI(TAG, "   📢 Real-time notification sent to connected client");
        }
    } else {
        ESP_LOGE(TAG, "   ❌ Failed to log event!");
    }
}

// === ISR CALLBACKS (ATOMIC DATA INGESTION) ===

/**
 * BVP Timer ISR - 64Hz sampling
 * CRITICAL: This runs in ISR context, uses atomic operations only
 */
static bool IRAM_ATTR bvp_timer_isr_callback(gptimer_handle_t timer, const gptimer_alarm_event_data_t *edata, void *user_ctx) {
    // Generate mock BVP sample (integer only, ISR-safe)
    int32_t bvp_value_int = generate_mock_bvp_int();
    
    // Add to buffer atomically - NO floating-point operations
    realtime_add_sample_int_isr(SENSOR_BVP, bvp_value_int);
    
    total_samples_collected++;
    
    return false; // Don't request task switch
}

/**
 * ACC Timer ISR - 32Hz sampling (all 3 axes)
 * CRITICAL: This runs in ISR context, uses atomic operations only
 */
static bool IRAM_ATTR acc_timer_isr_callback(gptimer_handle_t timer, const gptimer_alarm_event_data_t *edata, void *user_ctx) {
    // Generate mock ACC samples for all axes (integer only, ISR-safe)
    int32_t acc_x_int = generate_mock_acc_int(0);
    int32_t acc_y_int = generate_mock_acc_int(1);
    int32_t acc_z_int = generate_mock_acc_int(2);
    
    // Add all three samples atomically - NO floating-point operations
    realtime_add_sample_int_isr(SENSOR_ACC_X, acc_x_int);
    realtime_add_sample_int_isr(SENSOR_ACC_Y, acc_y_int);
    realtime_add_sample_int_isr(SENSOR_ACC_Z, acc_z_int);
    
    total_samples_collected += 3;
    
    return false; // Don't request task switch
}

/**
 * EDA Timer ISR - 4Hz sampling
 * CRITICAL: This runs in ISR context, uses atomic operations only
 */
static bool IRAM_ATTR eda_timer_isr_callback(gptimer_handle_t timer, const gptimer_alarm_event_data_t *edata, void *user_ctx) {
    // Generate mock EDA sample (integer only, ISR-safe)
    int32_t eda_value_int = generate_mock_eda_int();
    
    // Add to buffer atomically - NO floating-point operations
    realtime_add_sample_int_isr(SENSOR_EDA, eda_value_int);
    
    total_samples_collected++;
    
    return false; // Don't request task switch
}

/**
 * TEMP Timer ISR - 4Hz sampling  
 * CRITICAL: This runs in ISR context, uses atomic operations only
 */
static bool IRAM_ATTR temp_timer_isr_callback(gptimer_handle_t timer, const gptimer_alarm_event_data_t *edata, void *user_ctx) {
    // Generate mock TEMP sample (integer only, ISR-safe)
    int32_t temp_value_int = generate_mock_temp_int();
    
    // Add to buffer atomically - NO floating-point operations
    realtime_add_sample_int_isr(SENSOR_TEMP, temp_value_int);
    
    total_samples_collected++;
    
    return false; // Don't request task switch
}

// === TIMER SETUP FUNCTIONS ===

/**
 * Setup timer for specific sensor sampling rate
 */
static int setup_sensor_timer(gptimer_handle_t *timer, uint32_t frequency_hz, 
                             gptimer_alarm_cb_t callback, const char *name) {
    
    // Timer configuration
    gptimer_config_t timer_config = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000, // 1MHz resolution
    };
    
    ESP_ERROR_CHECK(gptimer_new_timer(&timer_config, timer));
    
    // Alarm configuration
    gptimer_alarm_config_t alarm_config = {
        .alarm_count = 1000000 / frequency_hz, // Convert Hz to microseconds
        .reload_count = 0,
        .flags.auto_reload_on_alarm = true,
    };
    
    ESP_ERROR_CHECK(gptimer_set_alarm_action(*timer, &alarm_config));
    
    // ISR callback
    gptimer_event_callbacks_t cbs = {
        .on_alarm = callback,
    };
    ESP_ERROR_CHECK(gptimer_register_event_callbacks(*timer, &cbs, NULL));
    
    ESP_LOGI(TAG, "✅ %s timer configured: %lu Hz", name, frequency_hz);
    return 0;
}

/**
 * Start all sensor timers
 */
static int start_sensor_timers(void) {
    ESP_LOGI(TAG, "Starting ISR-based sensor sampling...");
    
    ESP_ERROR_CHECK(gptimer_enable(bvp_timer));
    ESP_ERROR_CHECK(gptimer_start(bvp_timer));
    
    ESP_ERROR_CHECK(gptimer_enable(acc_timer));
    ESP_ERROR_CHECK(gptimer_start(acc_timer));
    
    ESP_ERROR_CHECK(gptimer_enable(eda_timer));
    ESP_ERROR_CHECK(gptimer_start(eda_timer));
    
    ESP_ERROR_CHECK(gptimer_enable(temp_timer));
    ESP_ERROR_CHECK(gptimer_start(temp_timer));
    
    ESP_LOGI(TAG, "🚀 All sensor timers started - ISR-based data ingestion active!");
    return 0;
}

// === PRODUCER TASK (TIMER MANAGEMENT) ===

/**
 * Producer Task (Core 0): Manages timers and monitors system
 * The actual data ingestion happens in ISRs, this task just monitors
 */
void producer_task(void *param) {
    ESP_LOGI(TAG, "🔧 Producer task started on Core %d", xPortGetCoreID());
    ESP_LOGI(TAG, "Setting up ISR-based sensor timers...");
    
    // Setup all sensor timers
    if (setup_sensor_timer(&bvp_timer, BVP_SAMPLE_RATE, bvp_timer_isr_callback, "BVP") != 0 ||
        setup_sensor_timer(&acc_timer, ACC_SAMPLE_RATE, acc_timer_isr_callback, "ACC") != 0 ||
        setup_sensor_timer(&eda_timer, EDA_SAMPLE_RATE, eda_timer_isr_callback, "EDA") != 0 ||
        setup_sensor_timer(&temp_timer, TEMP_SAMPLE_RATE, temp_timer_isr_callback, "TEMP") != 0) {
        ESP_LOGE(TAG, "❌ Failed to setup sensor timers");
        vTaskDelete(NULL);
        return;
    }
    
    // Start all timers
    if (start_sensor_timers() != 0) {
        ESP_LOGE(TAG, "❌ Failed to start sensor timers");
        vTaskDelete(NULL);
        return;
    }
    
    ESP_LOGI(TAG, "✅ ISR-based data ingestion system online!");
    
    // Monitor system performance
    uint32_t last_sample_count = 0;
    
    while (1) {
        // Print system status every 5 seconds
        vTaskDelay(pdMS_TO_TICKS(5000));
        
        uint32_t current_samples = total_samples_collected;
        uint32_t samples_per_sec = (current_samples - last_sample_count) / 5;
        last_sample_count = current_samples;
        
        ESP_LOGI(TAG, "📊 Performance: %lu samples/sec (total: %lu)", 
                samples_per_sec, current_samples);
        ESP_LOGI(TAG, "🧠 ML Inferences: %lu total", total_inferences);
        ESP_LOGI(TAG, "🔄 State Transitions: %lu total", total_state_transitions);
        ESP_LOGI(TAG, "📻 BLE Connected: %s", ble_stress_service_is_connected() ? "YES" : "NO");
        
        // Print detailed system status every 30 seconds
        static uint8_t status_counter = 0;
        status_counter++;
        if (status_counter >= 6) { // 6 * 5 seconds = 30 seconds
            status_counter = 0;
            ESP_LOGI(TAG, "=== DETAILED SYSTEM STATUS ===");
            realtime_print_status();
            event_log_print_status(&g_event_log);
            ble_stress_service_print_status();
            ESP_LOGI(TAG, "==============================");
        }
    }
}

// === CONSUMER TASK (EVENT-DRIVEN ML PROCESSING) ===

/**
 * Consumer Task (Core 1): Event-driven ML processing
 * Waits on semaphore, no polling - true real-time system
 */
void consumer_task(void *param) {
    ESP_LOGI(TAG, "🧠 ML Consumer task started on Core %d", xPortGetCoreID());
    ESP_LOGI(TAG, "Waiting for semaphore signals from ISR coordination...");
    
    // Wait for initial data accumulation
    vTaskDelay(pdMS_TO_TICKS(3000)); // 3 second initial delay
    
    while (1) {
        // WAIT ON SEMAPHORE - NO POLLING!
        // This task remains suspended until check_and_signal_ml_ready() signals it
        if (xSemaphoreTake(g_sensor_system.ml_ready_sem, portMAX_DELAY) == pdTRUE) {
            
            ESP_LOGI(TAG, "🔔 Semaphore signal received - starting ML inference #%lu", total_inferences);
            
            uint32_t inference_start = xTaskGetTickCount() * portTICK_PERIOD_MS;
            
            // Get current coordination state
            uint32_t min_batches = realtime_get_min_batch_count();
            
            ESP_LOGI(TAG, "🎯 Min batches available: %lu seconds", min_batches);
            
            // Extract features from 60-second windows
            feature_vector_t features;
            int feature_result = extract_features_realtime(&g_sensor_system, &g_feature_workspace, &features);
            
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
            
            // Update coordination: mark this batch as processed
            realtime_mark_batch_processed(min_batches);
            
            uint32_t total_time = (xTaskGetTickCount() * portTICK_PERIOD_MS) - inference_start;
            
            // === STRESS FSM PROCESSING ===
            // Process the ML result through the confirmation FSM
            uint32_t current_time_ms = (uint32_t)(esp_timer_get_time() / 1000);
            bool transition_occurred = stress_fsm_process_inference(&g_stress_fsm, 
                                                                   stress_probability, 
                                                                   current_time_ms,
                                                                   on_stress_transition);
            
            // Log results
            ESP_LOGI(TAG, "🎯 ML Inference Results:");
            ESP_LOGI(TAG, "   Stress Probability: %.3f", stress_probability);
            ESP_LOGI(TAG, "   Stress Class: %s", stress_class ? "STRESS" : "NORMAL");
            ESP_LOGI(TAG, "   FSM State: %s", 
                     stress_fsm_state_to_string(stress_fsm_get_current_state(&g_stress_fsm)));
            ESP_LOGI(TAG, "   State Transition: %s", transition_occurred ? "YES" : "NO");
            ESP_LOGI(TAG, "   Feature Time: %lu ms", features.extraction_time_ms);
            ESP_LOGI(TAG, "   ML Time: %lu ms", ml_time);
            ESP_LOGI(TAG, "   Total Processing: %lu ms", total_time);
            ESP_LOGI(TAG, "   Batch Processed: %lu", min_batches);
            
            total_inferences++;
            
            // Update BLE advertisement if state changed
            if (transition_occurred) {
                // Get current battery voltage (mock for now)
                uint16_t battery_mv = 3300 + (esp_random() % 500) - 250; // 3050-3550mV
                uint8_t sensor_quality = 85 + (esp_random() % 20) - 10;  // 75-95%
                
                ble_stress_service_update_advertisement(battery_mv, sensor_quality);
                
                // Send notification if client is connected
                if (ble_stress_service_notifications_enabled()) {
                    ble_stress_service_notify_fsm_state();
                }
            }
            
            ESP_LOGI(TAG, "---");
        }
    }
}

// === COMPATIBILITY BRIDGE FOR FEATURE EXTRACTION ===

/**
 * Bridge function to adapt realtime system to existing feature extractor
 * Extracts data from atomic buffers and creates compatible structure
 */
int extract_features_realtime(realtime_sensor_system_t *sensor_system, 
                             feature_workspace_t *workspace, 
                             feature_vector_t *result) {
    
    if (!sensor_system || !workspace || !result) return -1;
    
    uint32_t start_time = xTaskGetTickCount() * portTICK_PERIOD_MS;
    
    // For now, create mock features based on realistic ranges from dataset
    // In production, this would extract windows using realtime_extract_window()
    // and compute the actual 30 features
    
    // Extract windows from each sensor (example for BVP)
    fixed_point_t bvp_window[BVP_BUFFER_SIZE];
    int bvp_samples = realtime_extract_window(SENSOR_BVP, bvp_window, BVP_BUFFER_SIZE);
    
    // Convert back to float for feature computation
    float bvp_data[BVP_BUFFER_SIZE];
    for (int i = 0; i < bvp_samples; i++) {
        bvp_data[i] = FIXED_TO_FLOAT(bvp_window[i]);
    }
    
    // Simulate processing time
    vTaskDelay(pdMS_TO_TICKS(50));
    
    // Calculate realistic features (simplified version)
    // BVP features (8 features)
    if (bvp_samples > 0) {
        // Basic statistics
        float sum = 0, sum_sq = 0, min_val = bvp_data[0], max_val = bvp_data[0];
        for (int i = 0; i < bvp_samples; i++) {
            sum += bvp_data[i];
            sum_sq += bvp_data[i] * bvp_data[i];
            if (bvp_data[i] < min_val) min_val = bvp_data[i];
            if (bvp_data[i] > max_val) max_val = bvp_data[i];
        }
        
        float mean = sum / bvp_samples;
        float variance = (sum_sq / bvp_samples) - (mean * mean);
        float std = sqrtf(variance);
        
        result->features[0] = mean;        // BVP_MEAN
        result->features[1] = std;         // BVP_STD  
        result->features[2] = min_val;     // BVP_MIN
        result->features[3] = max_val;     // BVP_MAX
        result->features[4] = mean;        // BVP_MEDIAN (approximated)
        result->features[5] = max_val - min_val; // BVP_RANGE
        result->features[6] = std * 1.35f; // BVP_IQR (approximated)
        result->features[7] = sum_sq;      // BVP_ENERGY
    }
    
    // Mock remaining features (ACC, EDA, TEMP) with realistic values
    // ACC features (15 features: 5 per axis)
    for (int axis = 0; axis < 3; axis++) {
        int base_idx = 8 + (axis * 5);
        result->features[base_idx + 0] = (axis == 0) ? 15.42f : (axis == 1) ? -6.18f : 8.99f; // mean
        result->features[base_idx + 1] = 8.0f + (esp_random() % 400) / 100.0f; // std
        result->features[base_idx + 2] = -30.0f - (esp_random() % 3000) / 100.0f; // min
        result->features[base_idx + 3] = 30.0f + (esp_random() % 3000) / 100.0f;  // max
        result->features[base_idx + 4] = 100.0f + (esp_random() % 5000) / 100.0f; // energy
    }
    
    // EDA features (4 features)
    result->features[23] = 2.08f + (esp_random() % 200 - 100) / 1000.0f; // EDA_MEAN
    result->features[24] = 0.5f + (esp_random() % 300) / 1000.0f;        // EDA_STD
    result->features[25] = 0.09f + (esp_random() % 100) / 10000.0f;      // EDA_MIN
    result->features[26] = 5.0f + (esp_random() % 1000) / 100.0f;        // EDA_MAX
    
    // TEMP features (3 features)
    result->features[27] = 33.09f + (esp_random() % 200 - 100) / 1000.0f; // TEMP_MEAN
    result->features[28] = 0.3f + (esp_random() % 200) / 1000.0f;         // TEMP_STD
    result->features[29] = 2.0f + (esp_random() % 400) / 100.0f;          // TEMP_RANGE
    
    result->extraction_time_ms = (xTaskGetTickCount() * portTICK_PERIOD_MS) - start_time;
    result->success = true;
    result->timestamp = xTaskGetTickCount();
    
    return 0;
}

// === MAIN APPLICATION ===

void app_main(void) {
    ESP_LOGI(TAG, "🌟 Shadow Real-Time Stress Detection Firmware v3.0");
    ESP_LOGI(TAG, "ESP32-S3 ISR-based Producer-Consumer Architecture");
    ESP_LOGI(TAG, "Architecture: Atomic coordination + Event-driven processing + BLE Communication");
    
    // Initialize real-time sensor system
    ESP_LOGI(TAG, "Initializing real-time sensor buffer system...");
    if (realtime_sensor_init() != 0) {
        ESP_LOGE(TAG, "❌ Failed to initialize real-time sensor system");
        return;
    }
    
    // Initialize feature extractor workspace
    ESP_LOGI(TAG, "Initializing feature extractor...");
    if (feature_extractor_init(&g_feature_workspace) != 0) {
        ESP_LOGE(TAG, "❌ Failed to initialize feature extractor");
        return;
    }
    
    // Initialize stress FSM
    ESP_LOGI(TAG, "Initializing stress finite state machine...");
    if (stress_fsm_init(&g_stress_fsm) != 0) {
        ESP_LOGE(TAG, "❌ Failed to initialize stress FSM");
        return;
    }
    
    // Initialize event logging system
    ESP_LOGI(TAG, "Initializing event logging system...");
    if (event_log_init(&g_event_log) != 0) {
        ESP_LOGE(TAG, "❌ Failed to initialize event log");
        return;
    }
    
    // Initialize BLE stress service
    ESP_LOGI(TAG, "Initializing BLE stress monitoring service...");
    
    // Initialize BLE stress service
    ESP_LOGI(TAG, "Initializing BLE stress service...");
    if (ble_stress_service_init(&g_stress_fsm, &g_event_log) != 0) {
        ESP_LOGE(TAG, "❌ Failed to initialize BLE service");
        return;
    }
    
    ESP_LOGI(TAG, "✅ System initialization complete!");
    ESP_LOGI(TAG, "Memory usage: %lu bytes", realtime_get_memory_usage());
    
    // Start BLE advertising
    ESP_LOGI(TAG, "Starting BLE advertising...");
    if (ble_stress_service_start_advertising() == 0) {
        ESP_LOGI(TAG, "✅ BLE advertising started");
    } else {
        ESP_LOGW(TAG, "⚠️  BLE advertising failed to start");
    }
    
    // Create producer task on Core 0 (Timer management)
    xTaskCreatePinnedToCore(
        producer_task,
        "producer",
        4096,  // Stack size
        NULL,  // Parameters
        5,     // Priority (high for timer management)
        &producer_task_handle,
        0      // Core 0
    );
    
    // Create consumer task on Core 1 (ML processing)
    xTaskCreatePinnedToCore(
        consumer_task,
        "consumer", 
        32768,  // Much larger stack for ML processing (32KB)
        NULL,  // Parameters
        3,     // Priority (lower than producer)
        &consumer_task_handle,
        1      // Core 1
    );
    
    ESP_LOGI(TAG, "🚀 Real-time tasks created successfully!");
    ESP_LOGI(TAG, "📡 Producer (Core 0): ISR-based data ingestion");
    ESP_LOGI(TAG, "🧠 Consumer (Core 1): Event-driven ML processing + FSM");
    ESP_LOGI(TAG, "📻 BLE Service: Stress monitor communication");
    ESP_LOGI(TAG, "⚡ Coordination: Atomic batch counting + semaphore signaling");
    ESP_LOGI(TAG, "🎯 Real-time stress detection system ONLINE!");
}
