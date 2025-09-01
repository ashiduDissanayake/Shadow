/*
 * ESP32 Multi-Sensor Circular Buffer System
 * 
 * 6-Layer Circular Buffer for Synchronized Sensor Data Collection
 * Optimized for ESP32: minimal memory, fixed-point math, efficient indexing
 * 
 * Buffer Layout:
 * Layer 0: BVP   (64 Hz) - 3840 samples (60s)
 * Layer 1: ACC_X (32 Hz) - 1920 samples (60s) 
 * Layer 2: ACC_Y (32 Hz) - 1920 samples (60s)
 * Layer 3: ACC_Z (32 Hz) - 1920 samples (60s)
 * Layer 4: EDA   (4 Hz)  - 240 samples (60s)
 * Layer 5: TEMP  (4 Hz)  - 240 samples (60s)
 */

#ifndef SENSOR_BUFFER_H
#define SENSOR_BUFFER_H

#include <stdint.h>
#include <string.h>

// Configuration constants
#define WINDOW_SECONDS           60
#define STEP_SECONDS            10
#define NUM_SENSOR_LAYERS       6

// Sampling rates (Hz)
#define BVP_SAMPLE_RATE         64
#define ACC_SAMPLE_RATE         32  
#define EDA_SAMPLE_RATE         4
#define TEMP_SAMPLE_RATE        4

// Buffer sizes (samples for 60 seconds)
#define BVP_BUFFER_SIZE         (BVP_SAMPLE_RATE * WINDOW_SECONDS)      // 3840
#define ACC_BUFFER_SIZE         (ACC_SAMPLE_RATE * WINDOW_SECONDS)      // 1920
#define EDA_BUFFER_SIZE         (EDA_SAMPLE_RATE * WINDOW_SECONDS)      // 240
#define TEMP_BUFFER_SIZE        (TEMP_SAMPLE_RATE * WINDOW_SECONDS)     // 240

// Sensor layer indices
typedef enum {
    LAYER_BVP   = 0,
    LAYER_ACC_X = 1,
    LAYER_ACC_Y = 2,
    LAYER_ACC_Z = 3,
    LAYER_EDA   = 4,
    LAYER_TEMP  = 5
} sensor_layer_t;

// Fixed-point data type (16.16 format for memory efficiency)
typedef int32_t fixed_point_t;
#define FIXED_POINT_SCALE       65536   // 2^16
#define FLOAT_TO_FIXED(f)       ((fixed_point_t)((f) * FIXED_POINT_SCALE))
#define FIXED_TO_FLOAT(x)       ((float)(x) / FIXED_POINT_SCALE)

// Circular buffer structure for each sensor layer
typedef struct {
    fixed_point_t *data;           // Data buffer
    uint16_t size;                 // Buffer size
    uint16_t head;                 // Write index (next position to write)
    uint16_t count;                // Number of valid samples
    uint8_t sample_rate;           // Sampling rate (Hz)
    uint32_t last_sample_time_ms;  // Last sample timestamp
} sensor_buffer_t;

// Main multi-sensor buffer system
typedef struct {
    sensor_buffer_t layers[NUM_SENSOR_LAYERS];
    uint32_t system_start_time_ms;
    uint8_t initialized;
} multi_sensor_buffer_t;

// Buffer management functions
int buffer_init(multi_sensor_buffer_t *msb);
void buffer_deinit(multi_sensor_buffer_t *msb);

// Data collection functions  
int buffer_add_sample(multi_sensor_buffer_t *msb, sensor_layer_t layer, float value);
int buffer_add_samples_batch(multi_sensor_buffer_t *msb, sensor_layer_t layer, 
                            const float *values, uint16_t count);

// Data retrieval functions
int buffer_read_window(multi_sensor_buffer_t *msb, sensor_layer_t layer, 
                      fixed_point_t *output, uint16_t max_samples);
int buffer_get_latest_samples(multi_sensor_buffer_t *msb, sensor_layer_t layer,
                             fixed_point_t *output, uint16_t num_samples);

// Buffer status functions
uint16_t buffer_get_count(multi_sensor_buffer_t *msb, sensor_layer_t layer);
int buffer_is_ready_for_processing(multi_sensor_buffer_t *msb);
uint32_t buffer_get_memory_usage(void);

// Timing and synchronization
int buffer_should_sample(multi_sensor_buffer_t *msb, sensor_layer_t layer, uint32_t current_time_ms);
void buffer_reset_timing(multi_sensor_buffer_t *msb);

#endif // SENSOR_BUFFER_H
