/*
 * ESP32 Multi-Sensor Circular Buffer Implementation
 * Optimized for minimal memory usage and computational efficiency.
 */

#include "sensor_buffer.h"
#include <stdlib.h>
#include <string.h> // For memset
#include <stdbool.h> // For bool/true/false

// Static memory allocation for buffers (no fragmentation)
static fixed_point_t bvp_buffer_data[BVP_BUFFER_SIZE];
static fixed_point_t acc_x_buffer_data[ACC_BUFFER_SIZE];
static fixed_point_t acc_y_buffer_data[ACC_BUFFER_SIZE];
static fixed_point_t acc_z_buffer_data[ACC_BUFFER_SIZE];
static fixed_point_t eda_buffer_data[EDA_BUFFER_SIZE];
static fixed_point_t temp_buffer_data[TEMP_BUFFER_SIZE];

// Configuration table for sensors
static const struct {
    fixed_point_t *data;
    uint16_t size;
    uint8_t sample_rate;
} buffer_configs[NUM_SENSOR_LAYERS] = {
    {bvp_buffer_data,   BVP_BUFFER_SIZE,  BVP_SAMPLE_RATE},   // LAYER_BVP
    {acc_x_buffer_data, ACC_BUFFER_SIZE,  ACC_SAMPLE_RATE},   // LAYER_ACC_X
    {acc_y_buffer_data, ACC_BUFFER_SIZE,  ACC_SAMPLE_RATE},   // LAYER_ACC_Y
    {acc_z_buffer_data, ACC_BUFFER_SIZE,  ACC_SAMPLE_RATE},   // LAYER_ACC_Z
    {eda_buffer_data,   EDA_BUFFER_SIZE,  EDA_SAMPLE_RATE},   // LAYER_EDA
    {temp_buffer_data,  TEMP_BUFFER_SIZE, TEMP_SAMPLE_RATE}   // LAYER_TEMP
};

/**
 * Initialize the multi-sensor buffer system.
 */
int buffer_init(multi_sensor_buffer_t *msb) {
    if (!msb) return -1;

    // Clear and initialize all buffers
    memset(msb, 0, sizeof(multi_sensor_buffer_t));

    for (int i = 0; i < NUM_SENSOR_LAYERS; i++) {
        sensor_buffer_t *layer = &msb->layers[i];
        layer->data = buffer_configs[i].data;
        layer->size = buffer_configs[i].size;
        layer->sample_rate = buffer_configs[i].sample_rate;
        layer->head = 0;
        layer->count = 0;
        layer->last_sample_time_ms = 0;

        // Clear allocated buffer
        memset(layer->data, 0, layer->size * sizeof(fixed_point_t));
    }

    msb->system_start_time_ms = 0;
    msb->initialized = 1;

    return 0;
}

/**
 * Add a single sample to a specific layer.
 */
int buffer_add_sample(multi_sensor_buffer_t *msb, sensor_layer_t layer, float value) {
    if (!msb || !msb->initialized || layer >= NUM_SENSOR_LAYERS) return -1;

    sensor_buffer_t *buf = &msb->layers[layer];

    // Convert to fixed-point format
    fixed_point_t fixed_value = FLOAT_TO_FIXED(value);

    // Add to circular buffer
    buf->data[buf->head] = fixed_value;

    // Update indices and track the number of valid entries
    buf->head = (buf->head + 1) % buf->size;
    buf->count = (buf->count < buf->size) ? (buf->count + 1) : buf->size;

    return 0;
}

/**
 * Add multiple samples in batch (more efficient for burst data).
 */
int buffer_add_samples_batch(multi_sensor_buffer_t *msb, sensor_layer_t layer, const float *values, uint16_t count) {
    if (!msb || !msb->initialized || layer >= NUM_SENSOR_LAYERS || !values) return -1;

    sensor_buffer_t *buf = &msb->layers[layer];

    for (uint16_t i = 0; i < count; i++) {
        // Convert to fixed-point
        fixed_point_t fixed_value = FLOAT_TO_FIXED(values[i]);

        // Add to circular buffer
        buf->data[buf->head] = fixed_value;
        buf->head = (buf->head + 1) % buf->size;
        buf->count = (buf->count < buf->size) ? (buf->count + 1) : buf->size;
    }

    return 0;
}

/**
 * Read data from buffer (chronological order).
 */
int buffer_read_window(multi_sensor_buffer_t *msb, sensor_layer_t layer, fixed_point_t *output, uint16_t max_samples) {
    if (!msb || !output || layer >= NUM_SENSOR_LAYERS) return -1;

    sensor_buffer_t *buf = &msb->layers[layer];
    if (buf->count == 0) return 0; // No data available

    uint16_t samples_to_read = (buf->count < max_samples) ? buf->count : max_samples;

    for (uint16_t i = 0; i < samples_to_read; i++) {
        uint16_t read_pos = (buf->head + i) % buf->size;
        output[i] = buf->data[read_pos];
    }

    return samples_to_read;
}

/**
 * Get most recent N samples from buffer.
 */
int buffer_get_latest_samples(multi_sensor_buffer_t *msb, sensor_layer_t layer, fixed_point_t *output, uint16_t num_samples) {
    if (!msb || !msb->initialized || layer >= NUM_SENSOR_LAYERS || !output) return -1;

    sensor_buffer_t *buf = &msb->layers[layer];

    uint16_t available = (buf->count < num_samples) ? buf->count : num_samples;
    if (available == 0) return 0;

    // Read backwards from most recent
    for (uint16_t i = 0; i < available; i++) {
        uint16_t pos = (buf->head - 1 - i + buf->size) % buf->size;
        output[available - 1 - i] = buf->data[pos]; // Reverse order for chronological
    }

    return available;
}

/**
 * Check if buffer has enough data for processing.
 */
int buffer_is_ready_for_processing(multi_sensor_buffer_t *msb) {
    if (!msb || !msb->initialized) return 0;

    // Check if all layers have at least 50% of their expected data
    for (int i = 0; i < NUM_SENSOR_LAYERS; i++) {
        sensor_buffer_t *buf = &msb->layers[i];
        if (buf->count < (buf->size / 2)) {
            return 0; // Not enough data yet
        }
    }

    return 1; // Ready for processing
}

/**
 * Get current sample count for a layer.
 */
uint16_t buffer_get_count(multi_sensor_buffer_t *msb, sensor_layer_t layer) {
    if (!msb || !msb->initialized || layer >= NUM_SENSOR_LAYERS) return 0;
    return msb->layers[layer].count;
}

/**
 * Calculate total memory usage.
 */
uint32_t buffer_get_memory_usage(void) {
    uint32_t total = sizeof(multi_sensor_buffer_t);
    total += sizeof(bvp_buffer_data);
    total += sizeof(acc_x_buffer_data);
    total += sizeof(acc_y_buffer_data);
    total += sizeof(acc_z_buffer_data);
    total += sizeof(eda_buffer_data);
    total += sizeof(temp_buffer_data);
    return total;
}

/**
 * Check if it's time to sample for a specific layer based on the sampling rate.
 */
int buffer_should_sample(multi_sensor_buffer_t *msb, sensor_layer_t layer, uint32_t current_time_ms) {
    if (!msb || !msb->initialized || layer >= NUM_SENSOR_LAYERS) return 0;

    sensor_buffer_t *buf = &msb->layers[layer];

    // Initialize system time on first sample
    if (msb->system_start_time_ms == 0) {
        msb->system_start_time_ms = current_time_ms;
        buf->last_sample_time_ms = current_time_ms;
        return 1; // Always sample first time
    }

    // Calculate sampling interval in milliseconds
    uint32_t sample_interval_ms = 1000 / buf->sample_rate;

    // Check if enough time has passed
    if ((current_time_ms - buf->last_sample_time_ms) >= sample_interval_ms) {
        buf->last_sample_time_ms = current_time_ms;
        return 1;
    }

    return 0;
}

/**
 * Reset timing (useful for synchronization).
 */
void buffer_reset_timing(multi_sensor_buffer_t *msb) {
    if (!msb || !msb->initialized) return;

    msb->system_start_time_ms = 0;
    for (int i = 0; i < NUM_SENSOR_LAYERS; i++) {
        msb->layers[i].last_sample_time_ms = 0;
    }
}

/**
 * Deinitialize multi-sensor buffer system
 */
void buffer_deinit(multi_sensor_buffer_t *msb) {
    if (!msb || !msb->initialized) return;
    
    // Reset all buffer states
    for (int i = 0; i < NUM_SENSOR_LAYERS; i++) {
        msb->layers[i].head = 0;
        msb->layers[i].count = 0;
        msb->layers[i].last_sample_time_ms = 0;
    }
    
    msb->system_start_time_ms = 0;
    msb->initialized = false;
}