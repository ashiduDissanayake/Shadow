/*
 * Calibration System for Sensor Normalization
 * ESP32-S3 Shadow Project
 * 
 * Collects baseline statistics during calibration period (5-10 minutes)
 * to provide personalized z-score normalization for CNN model.
 */

#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <stdint.h>
#include <stdbool.h>
#include "signal_preprocessor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Calibration configuration
#define CALIBRATION_DURATION_SECONDS    120    // 2 minutes
#define CALIBRATION_SAMPLE_RATE         4      // 4 Hz
#define CALIBRATION_REQUIRED_SAMPLES    (CALIBRATION_DURATION_SECONDS * CALIBRATION_SAMPLE_RATE)  // 480 samples
#define CALIBRATION_MIN_SAMPLES         240    // Minimum 1 minute for valid calibration

// NVS storage key
#define CALIBRATION_NVS_NAMESPACE       "calibration"
#define CALIBRATION_NVS_KEY             "cal_stats"

// Calibration state
typedef enum {
    CAL_STATE_NOT_STARTED = 0,
    CAL_STATE_IN_PROGRESS = 1,
    CAL_STATE_COMPLETED   = 2,
    CAL_STATE_LOADED      = 3,      // Loaded from NVS
    CAL_STATE_FAILED      = 4
} calibration_state_t;

// Per-channel calibration statistics
typedef struct {
    float mean;                    // Computed mean during calibration
    float std;                     // Computed std during calibration
    uint32_t sample_count;         // Number of samples collected
    double running_sum;            // Running sum for mean calculation
    double running_sum_sq;         // Running sum of squares for variance
    bool valid;                    // Whether this channel's calibration is valid
} channel_calibration_t;

// Complete calibration data structure
typedef struct {
    channel_calibration_t channels[CNN_INPUT_CHANNELS];  // ACC, BVP, EDA, TEMP
    calibration_state_t state;
    uint32_t total_samples;
    uint32_t start_time_ms;
    uint32_t end_time_ms;
    uint32_t calibration_version;  // For future compatibility
} calibration_data_t;

/* ==================== INITIALIZATION ==================== */

/**
 * Initialize calibration system
 * Attempts to load calibration from NVS first
 * @return 0 on success, negative on error
 */
int calibration_init(void);

/**
 * Deinitialize calibration system
 */
void calibration_deinit(void);

/* ==================== CALIBRATION CONTROL ==================== */

/**
 * Start new calibration session
 * Clears any existing calibration data and begins collecting samples
 * @return 0 on success, negative on error
 */
int calibration_start(void);

/**
 * Stop calibration session
 * Finalizes statistics and saves to NVS if sufficient samples collected
 * @param force_stop If true, stop even if minimum samples not reached
 * @return 0 on success, negative on error
 */
int calibration_stop(bool force_stop);

/**
 * Reset calibration to factory defaults
 * Clears calibration from NVS and memory
 * @return 0 on success, negative on error
 */
int calibration_reset(void);

/* ==================== DATA COLLECTION ==================== */

/**
 * Update calibration with new samples from a channel
 * Should be called during calibration period with preprocessed data
 * @param samples Array of float samples (after preprocessing but before normalization)
 * @param length Number of samples in array
 * @param channel Channel ID (ACC, BVP, EDA, TEMP)
 * @return 0 on success, negative on error
 */
int calibration_update(const float *samples, uint16_t length, cnn_channel_t channel);

/* ==================== NORMALIZATION ==================== */

/**
 * Normalize signal using calibration statistics
 * If calibration not available, falls back to local z-score
 * @param signal Signal array to normalize (modified in-place)
 * @param length Number of samples in signal
 * @param channel Channel ID for calibration lookup
 * @return 0 on success, negative on error
 */
int calibration_normalize(float *signal, uint16_t length, cnn_channel_t channel);

/* ==================== STATUS & QUERIES ==================== */

/**
 * Check if calibration is complete and valid
 * @return true if calibrated, false otherwise
 */
bool calibration_is_calibrated(void);

/**
 * Get current calibration state
 * @return Current state enum
 */
calibration_state_t calibration_get_state(void);

/**
 * Get calibration progress (0.0 to 1.0)
 * @return Progress fraction
 */
float calibration_get_progress(void);

/**
 * Get calibration statistics for a channel
 * @param channel Channel ID
 * @param mean Output parameter for mean
 * @param std Output parameter for std
 * @return 0 on success, negative if not calibrated
 */
int calibration_get_stats(cnn_channel_t channel, float *mean, float *std);

/**
 * Get remaining time for calibration (seconds)
 * @return Remaining seconds, or 0 if complete/not started
 */
uint32_t calibration_get_remaining_time(void);

/* ==================== PERSISTENCE (NVS) ==================== */

/**
 * Save calibration to NVS (non-volatile storage)
 * @return 0 on success, negative on error
 */
int calibration_save_to_nvs(void);

/**
 * Load calibration from NVS
 * @return 0 on success, negative on error
 */
int calibration_load_from_nvs(void);

/**
 * Check if valid calibration exists in NVS
 * @return true if exists, false otherwise
 */
bool calibration_exists_in_nvs(void);

/* ==================== DEBUGGING ==================== */

/**
 * Print calibration status and statistics
 */
void calibration_print_status(void);

/**
 * Get memory usage of calibration system
 * @return Bytes used
 */
uint32_t calibration_get_memory_usage(void);

#ifdef __cplusplus
}
#endif

#endif /* CALIBRATION_H */
