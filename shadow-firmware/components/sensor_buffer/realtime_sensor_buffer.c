/*
 * Real-Time Producer-Consumer Sensor Buffer Implementation
 * ESP32-S3 Atomic ISR-based Multi-Sensor System
 * 
 * CRITICAL: This implements the coordinated multi-buffer architecture
 * with atomic operations and ISR-based data ingestion
 */

#include "realtime_sensor_buffer.h"
#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_attr.h"

static const char *TAG = "RealtimeSensor";

// Static memory allocation (no fragmentation, ISR-safe)
static fixed_point_t bvp_buffer_data[BVP_BUFFER_SIZE];
static fixed_point_t acc_x_buffer_data[ACC_BUFFER_SIZE];
static fixed_point_t acc_y_buffer_data[ACC_BUFFER_SIZE];
static fixed_point_t acc_z_buffer_data[ACC_BUFFER_SIZE];
static fixed_point_t eda_buffer_data[EDA_BUFFER_SIZE];
static fixed_point_t temp_buffer_data[TEMP_BUFFER_SIZE];

// Buffer configuration table
static const struct {
    fixed_point_t *data;
    uint16_t size;
    uint8_t sample_rate;
} buffer_configs[NUM_SENSOR_BUFFERS] = {
    {bvp_buffer_data,   BVP_BUFFER_SIZE,  BVP_SAMPLE_RATE},   // SENSOR_BVP
    {acc_x_buffer_data, ACC_BUFFER_SIZE,  ACC_SAMPLE_RATE},   // SENSOR_ACC_X
    {acc_y_buffer_data, ACC_BUFFER_SIZE,  ACC_SAMPLE_RATE},   // SENSOR_ACC_Y
    {acc_z_buffer_data, ACC_BUFFER_SIZE,  ACC_SAMPLE_RATE},   // SENSOR_ACC_Z
    {eda_buffer_data,   EDA_BUFFER_SIZE,  EDA_SAMPLE_RATE},   // SENSOR_EDA
    {temp_buffer_data,  TEMP_BUFFER_SIZE, TEMP_SAMPLE_RATE}   // SENSOR_TEMP
};

// Global system instance
realtime_sensor_system_t g_sensor_system = {0};

/**
 * Initialize the real-time sensor buffer system
 */
int realtime_sensor_init(void) {
    ESP_LOGI(TAG, "Initializing real-time sensor buffer system...");
    
    // Clear system structure
    memset(&g_sensor_system, 0, sizeof(realtime_sensor_system_t));
    
    // Initialize individual sensor buffers
    for (int i = 0; i < NUM_SENSOR_BUFFERS; i++) {
        realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[i];
        
        // Set buffer configuration
        buf->data = buffer_configs[i].data;
        buf->size = buffer_configs[i].size;
        buf->sample_rate = buffer_configs[i].sample_rate;
        
        // Initialize atomic write pointer to 0
        atomic_store(&buf->write_ptr, 0);
        
        // Clear buffer data
        memset(buf->data, 0, buf->size * sizeof(fixed_point_t));
        
        ESP_LOGI(TAG, "Buffer %d (%s): %d samples, %d Hz", 
                i, get_sensor_name(i), buf->size, buf->sample_rate);
    }
    
    // Initialize coordination mechanism
    atomic_store(&g_sensor_system.last_processed_batch, 0);
    
    // Create ML task semaphore (binary semaphore, initially not available)
    g_sensor_system.ml_ready_sem = xSemaphoreCreateBinary();
    if (g_sensor_system.ml_ready_sem == NULL) {
        ESP_LOGE(TAG, "Failed to create ML ready semaphore");
        return -1;
    }
    
    g_sensor_system.initialized = 1;
    
    ESP_LOGI(TAG, "✅ Real-time sensor system initialized");
    ESP_LOGI(TAG, "Memory usage: %lu bytes", realtime_get_memory_usage());
    ESP_LOGI(TAG, "Coordination: %d-second windows, %d-second steps", 
             WINDOW_SECONDS, STEP_SECONDS);
    
    return 0;
}

/**
 * Add single sample from ISR (ATOMIC, no blocking, INTEGER ONLY)
 * NO floating-point operations - completely ISR-safe
 */
int IRAM_ATTR realtime_add_sample_int_isr(sensor_id_t sensor_id, int32_t value_int) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS) {
        return -1;
    }
    
    realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[sensor_id];
    
    // Use fixed-point value directly (no conversion needed)
    fixed_point_t fixed_value = (fixed_point_t)value_int;
    
    // Get current write position atomically and increment
    uint16_t write_pos = atomic_fetch_add(&buf->write_ptr, 1);
    
    // Write sample at calculated position (circular buffer)
    buf->data[write_pos % buf->size] = fixed_value;
    
    // Check if we need to signal ML processing (coordination)
    check_and_signal_ml_ready();
    
    return 0;
}

/**
 * Add single sample from ISR (ATOMIC, no blocking, FLOAT VERSION)
 * WARNING: Contains floating-point operations - use realtime_add_sample_int_isr() instead
 */
int IRAM_ATTR realtime_add_sample_isr(sensor_id_t sensor_id, float value) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS) {
        return -1;
    }
    
    realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[sensor_id];
    
    // Convert to fixed-point
    fixed_point_t fixed_value = FLOAT_TO_FIXED(value);
    
    // Get current write position atomically and increment
    uint16_t write_pos = atomic_fetch_add(&buf->write_ptr, 1);
    
    // Write to circular buffer
    buf->data[write_pos % buf->size] = fixed_value;
    
    // CRITICAL: Check coordination and signal ML task if ready
    check_and_signal_ml_ready();
    
    return 0;
}

/**
 * Add multiple samples from ISR (ATOMIC batch operation)
 */
int IRAM_ATTR realtime_add_samples_batch_isr(sensor_id_t sensor_id, const float *values, uint16_t count) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS || !values || count == 0) {
        return -1;
    }
    
    realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[sensor_id];
    
    // Get current write position atomically and reserve space
    uint16_t start_pos = atomic_fetch_add(&buf->write_ptr, count);
    
    // Write all samples to circular buffer
    for (uint16_t i = 0; i < count; i++) {
        fixed_point_t fixed_value = FLOAT_TO_FIXED(values[i]);
        uint16_t write_pos = (start_pos + i) % buf->size;
        buf->data[write_pos] = fixed_value;
    }
    
    // CRITICAL: Check coordination and signal ML task if ready
    check_and_signal_ml_ready();
    
    return 0;
}

/**
 * THE HEART OF THE SYSTEM: Check coordination and signal ML task
 * 
 * This is the core coordination function that implements the 
 * "weakest link" algorithm and semaphore signaling.
 */
void IRAM_ATTR check_and_signal_ml_ready(void) {
    if (!g_sensor_system.initialized) return;
    
    // Step 1: Calculate minimum batches across all sensors (weakest link)
    uint32_t min_batches = UINT32_MAX;
    
    for (int i = 0; i < NUM_SENSOR_BUFFERS; i++) {
        realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[i];
        
        // Get current write pointer atomically
        uint16_t current_samples = atomic_load(&buf->write_ptr);
        
        // Calculate complete 1-second batches for this sensor
        uint32_t sensor_batches = current_samples / buf->sample_rate;
        
        // Find minimum (weakest link)
        if (sensor_batches < min_batches) {
            min_batches = sensor_batches;
        }
    }
    
    // Step 2: Check if ML processing conditions are met
    uint32_t last_processed = atomic_load(&g_sensor_system.last_processed_batch);
    
    bool window_complete = (min_batches >= WINDOW_SECONDS);          // 60-second window available
    bool step_ready = (min_batches - last_processed >= STEP_SECONDS); // 10-second step available
    
    // Step 3: Signal ML task if both conditions are met
    if (window_complete && step_ready && g_sensor_system.ml_ready_sem != NULL) {
        BaseType_t xHigherPriorityTaskWoken = pdFALSE;
        
        // Give semaphore from ISR (non-blocking)
        xSemaphoreGiveFromISR(g_sensor_system.ml_ready_sem, &xHigherPriorityTaskWoken);
        
        // Request context switch if higher priority task was woken
        if (xHigherPriorityTaskWoken == pdTRUE) {
            portYIELD_FROM_ISR();
        }
    }
}

/**
 * Extract 60-second window for ML processing
 */
int realtime_extract_window(sensor_id_t sensor_id, fixed_point_t *output, uint16_t max_samples) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS || !output) {
        return -1;
    }
    
    realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[sensor_id];
    
    // Get current write pointer atomically
    uint16_t write_ptr = atomic_load(&buf->write_ptr);
    
    // Calculate how many samples to extract (60 seconds worth)
    uint16_t window_samples = buf->sample_rate * WINDOW_SECONDS;
    uint16_t samples_to_extract = (window_samples < max_samples) ? window_samples : max_samples;
    
    // Ensure we don't extract more than available
    uint16_t available_samples = (write_ptr < buf->size) ? write_ptr : buf->size;
    if (samples_to_extract > available_samples) {
        samples_to_extract = available_samples;
    }
    
    // Extract most recent samples in chronological order
    for (uint16_t i = 0; i < samples_to_extract; i++) {
        // Calculate position: most recent samples first, going backwards
        uint16_t offset = samples_to_extract - 1 - i;
        uint16_t read_pos = (write_ptr - 1 - offset + buf->size) % buf->size;
        output[i] = buf->data[read_pos];
    }
    
    return samples_to_extract;
}

/**
 * Mark current batch as processed by ML task
 */
void realtime_mark_batch_processed(uint32_t processed_batch) {
    if (!g_sensor_system.initialized) return;
    
    // Update last processed batch atomically
    atomic_store(&g_sensor_system.last_processed_batch, processed_batch);
}

/**
 * Get current write pointer value
 */
uint16_t realtime_get_write_ptr(sensor_id_t sensor_id) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS) return 0;
    
    return atomic_load(&g_sensor_system.buffers[sensor_id].write_ptr);
}

/**
 * Get current batch count for sensor
 */
uint32_t realtime_get_batch_count(sensor_id_t sensor_id) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS) return 0;
    
    realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[sensor_id];
    uint16_t samples = atomic_load(&buf->write_ptr);
    
    return samples / buf->sample_rate;
}

/**
 * Get minimum batch count across all sensors (weakest link)
 */
uint32_t realtime_get_min_batch_count(void) {
    if (!g_sensor_system.initialized) return 0;
    
    uint32_t min_batches = UINT32_MAX;
    
    for (int i = 0; i < NUM_SENSOR_BUFFERS; i++) {
        uint32_t sensor_batches = realtime_get_batch_count(i);
        if (sensor_batches < min_batches) {
            min_batches = sensor_batches;
        }
    }
    
    return (min_batches == UINT32_MAX) ? 0 : min_batches;
}

/**
 * Check if system is ready for ML processing
 */
int realtime_is_ml_ready(void) {
    if (!g_sensor_system.initialized) return 0;
    
    uint32_t min_batches = realtime_get_min_batch_count();
    uint32_t last_processed = atomic_load(&g_sensor_system.last_processed_batch);
    
    bool window_complete = (min_batches >= WINDOW_SECONDS);
    bool step_ready = (min_batches - last_processed >= STEP_SECONDS);
    
    return (window_complete && step_ready) ? 1 : 0;
}

/**
 * Get memory usage statistics
 */
uint32_t realtime_get_memory_usage(void) {
    uint32_t total = sizeof(realtime_sensor_system_t);
    
    // Add static buffer memory
    total += sizeof(bvp_buffer_data);
    total += sizeof(acc_x_buffer_data);
    total += sizeof(acc_y_buffer_data);
    total += sizeof(acc_z_buffer_data);
    total += sizeof(eda_buffer_data);
    total += sizeof(temp_buffer_data);
    
    return total;
}

/**
 * Print system status for debugging
 */
void realtime_print_status(void) {
    if (!g_sensor_system.initialized) {
        ESP_LOGI(TAG, "❌ System not initialized");
        return;
    }
    
    uint32_t min_batches = realtime_get_min_batch_count();
    uint32_t last_processed = atomic_load(&g_sensor_system.last_processed_batch);
    
    ESP_LOGI(TAG, "=== Real-Time Sensor System Status ===");
    ESP_LOGI(TAG, "Min batches available: %lu seconds", min_batches);
    ESP_LOGI(TAG, "Last processed batch: %lu", last_processed);
    ESP_LOGI(TAG, "ML ready: %s", realtime_is_ml_ready() ? "YES" : "NO");
    
    ESP_LOGI(TAG, "Individual sensor status:");
    for (int i = 0; i < NUM_SENSOR_BUFFERS; i++) {
        uint16_t write_ptr = realtime_get_write_ptr(i);
        uint32_t batches = realtime_get_batch_count(i);
        ESP_LOGI(TAG, "  %s: %d samples (%lu batches)", 
                get_sensor_name(i), write_ptr, batches);
    }
    ESP_LOGI(TAG, "=======================================");
}

/**
 * Deinitialize the system
 */
void realtime_sensor_deinit(void) {
    if (!g_sensor_system.initialized) return;
    
    // Delete semaphore
    if (g_sensor_system.ml_ready_sem != NULL) {
        vSemaphoreDelete(g_sensor_system.ml_ready_sem);
        g_sensor_system.ml_ready_sem = NULL;
    }
    
    // Reset atomic variables
    for (int i = 0; i < NUM_SENSOR_BUFFERS; i++) {
        atomic_store(&g_sensor_system.buffers[i].write_ptr, 0);
    }
    atomic_store(&g_sensor_system.last_processed_batch, 0);
    
    g_sensor_system.initialized = 0;
    
    ESP_LOGI(TAG, "Real-time sensor system deinitialized");
}
