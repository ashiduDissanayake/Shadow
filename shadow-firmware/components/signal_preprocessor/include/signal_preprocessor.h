/*
 * Signal Preprocessor for CNN Stress Detection
 * ESP32-S3 Shadow Project
 * 
 * Simplified preprocessing pipeline (no resampling needed):
 *  1. Compute ACC magnitude from 3 axes
 *  2. Z-score normalization per channel
 *  3. Stack into (4, 240) tensor for CNN input
 * 
 * NOTE: All sensors configured to sample at 4Hz directly
 */

#ifndef SIGNAL_PREPROCESSOR_H
#define SIGNAL_PREPROCESSOR_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "esp_log.h"
#include "realtime_sensor_buffer.h"

// CNN Model Input Configuration
#define CNN_INPUT_CHANNELS      4       // ACC, BVP, EDA, TEMP
#define CNN_INPUT_SAMPLES       240     // 60 seconds @ 4Hz
#define CNN_WINDOW_DURATION     60      // seconds
#define CNN_SAMPLE_RATE         4       // Hz (all sensors)

// Channel indices in CNN input tensor
typedef enum {
    CNN_CHANNEL_ACC = 0,    // Accelerometer magnitude
    CNN_CHANNEL_BVP = 1,    // Blood volume pulse
    CNN_CHANNEL_EDA = 2,    // Electrodermal activity
    CNN_CHANNEL_TEMP = 3    // Temperature
} cnn_channel_t;

// Preprocessing result structure
typedef struct {
    float data[CNN_INPUT_CHANNELS][CNN_INPUT_SAMPLES];  // (4, 240) tensor
    uint32_t preprocessing_time_ms;                      // Time taken
    bool success;                                        // Processing status
    uint32_t timestamp;                                  // FreeRTOS ticks when created
} cnn_input_tensor_t;

// Statistics for debugging/validation
typedef struct {
    float mean;
    float std;
    float min;
    float max;
} signal_stats_t;

/**
 * Initialize signal preprocessor
 * 
 * @return 0 on success, -1 on error
 */
int signal_preprocessor_init(void);

/**
 * Compute accelerometer magnitude from 3 axes
 * 
 * Formula: magnitude = sqrt(x² + y² + z²)
 * 
 * @param acc_x     X-axis acceleration data
 * @param acc_y     Y-axis acceleration data
 * @param acc_z     Z-axis acceleration data
 * @param output    Output buffer for magnitude (must be pre-allocated)
 * @param length    Number of samples
 * @return 0 on success, negative on error
 */
int compute_acc_magnitude(const float *acc_x, const float *acc_y, const float *acc_z,
                         float *output, uint16_t length);

/**
 * Z-score normalization (in-place)
 * 
 * Formula: normalized = (signal - mean) / std
 * 
 * @param signal    Signal array (modified in-place)
 * @param length    Signal length
 * @return 0 on success, negative on error
 */
int normalize_signal_zscore(float *signal, uint16_t length);

/**
 * Compute signal statistics (for debugging/validation)
 * 
 * @param signal    Input signal
 * @param length    Signal length
 * @param stats     Output statistics structure
 * @return 0 on success, negative on error
 */
int compute_signal_stats(const float *signal, uint16_t length, signal_stats_t *stats);

/**
 * Preprocess sensor data for CNN model
 * 
 * This is the main preprocessing function that:
 * 1. Extracts 240 samples from each sensor buffer
 * 2. Computes ACC magnitude from 3 axes
 * 3. Normalizes each channel using z-score
 * 4. Stacks into (4, 240) tensor
 * 
 * @param sensor_system  Realtime sensor system with ring buffers
 * @param output         Output CNN input tensor
 * @return 0 on success, negative on error
 */
int preprocess_for_cnn(realtime_sensor_system_t *sensor_system,
                       cnn_input_tensor_t *output);

/**
 * Print CNN input tensor for debugging
 * 
 * @param tensor    CNN input tensor to print
 */
void print_cnn_input_tensor(const cnn_input_tensor_t *tensor);

/**
 * Validate preprocessing output against expected values
 * (for testing with test_data.h)
 * 
 * @param actual        Actual preprocessed tensor
 * @param expected      Expected normalized values
 * @param channel       Channel index
 * @param tolerance     Maximum allowed difference
 * @return true if validation passes, false otherwise
 */
bool validate_preprocessing(const cnn_input_tensor_t *actual,
                           const float *expected,
                           cnn_channel_t channel,
                           float tolerance);

/**
 * Get memory usage of signal preprocessor
 * 
 * @return Total memory usage in bytes
 */
uint32_t signal_preprocessor_get_memory_usage(void);

#endif // SIGNAL_PREPROCESSOR_H
