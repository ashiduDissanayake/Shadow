/*
 * Real-Time Producer-Consumer Sensor Buffer System
 * ESP32-S3 Multi-Sensor Coordinated Architecture
 * 
 * ATOMIC ISR-based data ingestion with coordinated batch counting
 * 
 * Architecture:
 * - Separate circular buffers for each sensor (BVP, ACC_X, ACC_Y, ACC_Z, EDA, TEMP)
 * - Atomic write pointers updated from ISRs
 * - Global batch counter for coordination
 * - Single semaphore for ML task signaling
 * - Event-driven processing (no polling)
 */

#ifndef REALTIME_SENSOR_BUFFER_H
#define REALTIME_SENSOR_BUFFER_H

#include <stdint.h>
#include <stdatomic.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "esp_log.h"

// Configuration constants
#define WINDOW_SECONDS           60
#define STEP_SECONDS            10
#define NUM_SENSOR_BUFFERS      6

// Sampling rates (Hz) - samples per second
#define BVP_SAMPLE_RATE         64
#define ACC_SAMPLE_RATE         32  
#define EDA_SAMPLE_RATE         4
#define TEMP_SAMPLE_RATE        4

// Buffer sizes (samples for 60 seconds)
#define BVP_BUFFER_SIZE         (BVP_SAMPLE_RATE * WINDOW_SECONDS)      // 3840
#define ACC_BUFFER_SIZE         (ACC_SAMPLE_RATE * WINDOW_SECONDS)      // 1920
#define EDA_BUFFER_SIZE         (EDA_SAMPLE_RATE * WINDOW_SECONDS)      // 240
#define TEMP_BUFFER_SIZE        (TEMP_SAMPLE_RATE * WINDOW_SECONDS)     // 240

// Sensor buffer indices
typedef enum {
    SENSOR_BVP   = 0,
    SENSOR_ACC_X = 1,
    SENSOR_ACC_Y = 2,
    SENSOR_ACC_Z = 3,
    SENSOR_EDA   = 4,
    SENSOR_TEMP  = 5
} sensor_id_t;

// Fixed-point data type for memory efficiency
typedef int32_t fixed_point_t;
#define FIXED_POINT_SCALE       65536   // 2^16
#define FLOAT_TO_FIXED(f)       ((fixed_point_t)((f) * FIXED_POINT_SCALE))
#define FIXED_TO_FLOAT(x)       ((float)(x) / FIXED_POINT_SCALE)

// Single sensor buffer (lock-free, ISR-safe)
typedef struct {
    fixed_point_t *data;                    // Data buffer (statically allocated)
    uint16_t size;                          // Buffer size in samples
    uint8_t sample_rate;                    // Samples per second
    volatile atomic_uint_fast16_t write_ptr; // Atomic write pointer (ISR updates)
} realtime_sensor_buffer_t;

// Global coordination system
typedef struct {
    realtime_sensor_buffer_t buffers[NUM_SENSOR_BUFFERS];    // Individual sensor buffers
    
    // Coordination mechanism (ISR-safe)
    volatile atomic_uint_fast32_t last_processed_batch;  // Last batch processed by ML task
    
    // ML task signaling
    SemaphoreHandle_t ml_ready_sem;                 // Semaphore for ML task wakeup
    TaskHandle_t ml_task_handle;                    // ML task handle for notifications
    
    // System state
    uint8_t initialized;                            // Initialization flag
} realtime_sensor_system_t;

// Global system instance
extern realtime_sensor_system_t g_sensor_system;

// === CORE SYSTEM FUNCTIONS ===

/**
 * Initialize the real-time sensor buffer system
 * Creates buffers, semaphores, and sets up coordination
 */
int realtime_sensor_init(void);

/**
 * Deinitialize the system and cleanup resources
 */
void realtime_sensor_deinit(void);

// === ISR DATA INGESTION FUNCTIONS ===

/**
 * Add a sensor sample from ISR (ATOMIC operation, INTEGER ONLY)
 * Critical: This MUST be called from ISR context only
 * NO floating-point operations - uses fixed-point arithmetic
 * 
 * @param sensor_id Which sensor buffer to write to
 * @param value_int Fixed-point integer value (multiply by 100, e.g., 12.34 -> 1234)
 * @return 0 on success, -1 on error
 */
int IRAM_ATTR realtime_add_sample_int_isr(sensor_id_t sensor_id, int32_t value_int);

/**
 * Add a sensor sample from ISR (ATOMIC operation, FLOAT VERSION)
 * Critical: This MUST be called from ISR context only
 * WARNING: Contains floating-point operations - use realtime_add_sample_int_isr() instead
 * 
 * @param sensor_id Which sensor buffer to write to
 * @param value Sample value to add
 * @return 0 on success, -1 on error
 */
int IRAM_ATTR realtime_add_sample_isr(sensor_id_t sensor_id, float value);

/**
 * Add multiple samples from ISR (ATOMIC batch operation)
 * For GDMA transfers that collect multiple samples
 * 
 * @param sensor_id Which sensor buffer to write to  
 * @param values Array of sample values
 * @param count Number of samples
 * @return 0 on success, -1 on error
 */
int IRAM_ATTR realtime_add_samples_batch_isr(sensor_id_t sensor_id, const float *values, uint16_t count);

// === COORDINATION FUNCTION (ISR-CALLED) ===

/**
 * Check coordination state and signal ML task if ready
 * THE HEART OF THE SYSTEM - called from every ISR
 * 
 * This function:
 * 1. Calculates min_batches across all sensors (weakest link)
 * 2. Checks if 60-second window + 10-second step conditions are met
 * 3. Signals ML task via semaphore if ready
 * 
 * MUST be called from ISR context after updating write pointers
 */
void IRAM_ATTR check_and_signal_ml_ready(void);

// === ML TASK DATA EXTRACTION ===

/**
 * Extract 60-second window for ML processing
 * Called by ML task after semaphore signal
 * 
 * @param sensor_id Which sensor to extract from
 * @param output Buffer to store extracted samples
 * @param max_samples Maximum samples to extract
 * @return Number of samples extracted, or -1 on error
 */
int realtime_extract_window(sensor_id_t sensor_id, fixed_point_t *output, uint16_t max_samples);

/**
 * Mark current batch as processed by ML task
 * Updates last_processed_batch counter atomically
 * 
 * @param processed_batch Batch number that was just processed
 */
void realtime_mark_batch_processed(uint32_t processed_batch);

// === SYSTEM STATUS AND DEBUGGING ===

/**
 * Get current write pointer value (for debugging)
 * @param sensor_id Which sensor to query
 * @return Current write pointer value
 */
uint16_t realtime_get_write_ptr(sensor_id_t sensor_id);

/**
 * Get current batch count for sensor (samples / samples_per_second)
 * @param sensor_id Which sensor to query
 * @return Number of complete 1-second batches available
 */
uint32_t realtime_get_batch_count(sensor_id_t sensor_id);

/**
 * Get minimum batch count across all sensors (weakest link)
 * @return Minimum batches available across all sensors
 */
uint32_t realtime_get_min_batch_count(void);

/**
 * Check if system is ready for ML processing
 * @return 1 if ready (>=60 batches + >=10 new), 0 if not
 */
int realtime_is_ml_ready(void);

/**
 * Get memory usage statistics
 * @return Total memory usage in bytes
 */
uint32_t realtime_get_memory_usage(void);

/**
 * Print system status for debugging
 */
void realtime_print_status(void);

// === CONFIGURATION HELPERS ===

/**
 * Get buffer size for specific sensor
 * @param sensor_id Which sensor to query
 * @return Buffer size in samples
 */
static inline uint16_t get_sensor_buffer_size(sensor_id_t sensor_id) {
    const uint16_t sizes[NUM_SENSOR_BUFFERS] = {
        BVP_BUFFER_SIZE, ACC_BUFFER_SIZE, ACC_BUFFER_SIZE, 
        ACC_BUFFER_SIZE, EDA_BUFFER_SIZE, TEMP_BUFFER_SIZE
    };
    return (sensor_id < NUM_SENSOR_BUFFERS) ? sizes[sensor_id] : 0;
}

/**
 * Get sample rate for specific sensor
 * @param sensor_id Which sensor to query
 * @return Samples per second
 */
static inline uint8_t get_sensor_sample_rate(sensor_id_t sensor_id) {
    const uint8_t rates[NUM_SENSOR_BUFFERS] = {
        BVP_SAMPLE_RATE, ACC_SAMPLE_RATE, ACC_SAMPLE_RATE,
        ACC_SAMPLE_RATE, EDA_SAMPLE_RATE, TEMP_SAMPLE_RATE
    };
    return (sensor_id < NUM_SENSOR_BUFFERS) ? rates[sensor_id] : 0;
}

/**
 * Get sensor name for debugging
 * @param sensor_id Which sensor to query
 * @return String name of sensor
 */
static inline const char* get_sensor_name(sensor_id_t sensor_id) {
    const char* names[NUM_SENSOR_BUFFERS] = {
        "BVP", "ACC_X", "ACC_Y", "ACC_Z", "EDA", "TEMP"
    };
    return (sensor_id < NUM_SENSOR_BUFFERS) ? names[sensor_id] : "UNKNOWN";
}

#endif // REALTIME_SENSOR_BUFFER_H
