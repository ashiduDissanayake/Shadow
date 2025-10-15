/*
 * Signal Preprocessor Implementation
 * ESP32-S3 Shadow Project
 */

#include "signal_preprocessor.h"
#include <string.h>

static const char *TAG = "SignalPreprocessor";

/* ==================== INITIALIZATION ==================== */

int signal_preprocessor_init(void) {
    ESP_LOGI(TAG, "Initializing signal preprocessor");
    ESP_LOGI(TAG, "  CNN input shape: (%d channels, %d samples)",
             CNN_INPUT_CHANNELS, CNN_INPUT_SAMPLES);
    ESP_LOGI(TAG, "  Expected sample rate: %dHz for all sensors", CNN_SAMPLE_RATE);
    ESP_LOGI(TAG, "  Window duration: %d seconds", CNN_WINDOW_DURATION);
    ESP_LOGI(TAG, "  Memory usage: %lu bytes", signal_preprocessor_get_memory_usage());
    
    return 0;
}

/* ==================== ACCELEROMETER MAGNITUDE ==================== */

int compute_acc_magnitude(const float *acc_x, const float *acc_y, const float *acc_z,
                         float *output, uint16_t length) {
    if (!acc_x || !acc_y || !acc_z || !output || length == 0) {
        ESP_LOGE(TAG, "Invalid parameters for compute_acc_magnitude");
        return -1;
    }
    
    // Compute magnitude: sqrt(x² + y² + z²)
    for (uint16_t i = 0; i < length; i++) {
        float x = acc_x[i];
        float y = acc_y[i];
        float z = acc_z[i];
        output[i] = sqrtf(x*x + y*y + z*z);
    }
    
    return 0;
}

/* ==================== Z-SCORE NORMALIZATION ==================== */

int normalize_signal_zscore(float *signal, uint16_t length) {
    if (!signal || length == 0) {
        ESP_LOGE(TAG, "Invalid parameters for normalize_signal_zscore");
        return -1;
    }
    
    // Step 1: Compute mean
    float sum = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        sum += signal[i];
    }
    float mean = sum / length;
    
    // Step 2: Compute standard deviation
    float sum_squared_diff = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        float diff = signal[i] - mean;
        sum_squared_diff += diff * diff;
    }
    float variance = sum_squared_diff / length;
    float std = sqrtf(variance);
    
    // Step 3: Avoid division by zero
    if (std < 1e-6f) {
        ESP_LOGW(TAG, "Standard deviation too small (%.6f), using 1.0", std);
        std = 1.0f;
    }
    
    // Step 4: Normalize in-place: (x - mean) / std
    for (uint16_t i = 0; i < length; i++) {
        signal[i] = (signal[i] - mean) / std;
    }
    
    return 0;
}

/* ==================== SIGNAL STATISTICS ==================== */

int compute_signal_stats(const float *signal, uint16_t length, signal_stats_t *stats) {
    if (!signal || length == 0 || !stats) {
        ESP_LOGE(TAG, "Invalid parameters for compute_signal_stats");
        return -1;
    }
    
    // Initialize min/max
    float min_val = signal[0];
    float max_val = signal[0];
    float sum = 0.0f;
    
    // First pass: compute mean, min, max
    for (uint16_t i = 0; i < length; i++) {
        float val = signal[i];
        sum += val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    float mean = sum / length;
    
    // Second pass: compute standard deviation
    float sum_squared_diff = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        float diff = signal[i] - mean;
        sum_squared_diff += diff * diff;
    }
    float std = sqrtf(sum_squared_diff / length);
    
    // Store results
    stats->mean = mean;
    stats->std = std;
    stats->min = min_val;
    stats->max = max_val;
    
    return 0;
}

/* ==================== MAIN PREPROCESSING FUNCTION ==================== */

int preprocess_for_cnn(realtime_sensor_system_t *sensor_system,
                       cnn_input_tensor_t *output) {
    if (!sensor_system || !output) {
        ESP_LOGE(TAG, "Invalid parameters for preprocess_for_cnn");
        return -1;
    }
    
    uint32_t start_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
    
    // Clear output structure
    memset(output, 0, sizeof(cnn_input_tensor_t));
    
    /* ==================== STEP 1: Extract raw sensor data ==================== */
    
    // Temporary buffers for sensor data (240 samples @ 4Hz)
    fixed_point_t acc_x_fixed[CNN_INPUT_SAMPLES];
    fixed_point_t acc_y_fixed[CNN_INPUT_SAMPLES];
    fixed_point_t acc_z_fixed[CNN_INPUT_SAMPLES];
    fixed_point_t bvp_fixed[CNN_INPUT_SAMPLES];
    fixed_point_t eda_fixed[CNN_INPUT_SAMPLES];
    fixed_point_t temp_fixed[CNN_INPUT_SAMPLES];
    
    // Extract from ring buffers
    int acc_x_count = realtime_extract_window(SENSOR_ACC_X, acc_x_fixed, CNN_INPUT_SAMPLES);
    int acc_y_count = realtime_extract_window(SENSOR_ACC_Y, acc_y_fixed, CNN_INPUT_SAMPLES);
    int acc_z_count = realtime_extract_window(SENSOR_ACC_Z, acc_z_fixed, CNN_INPUT_SAMPLES);
    int bvp_count = realtime_extract_window(SENSOR_BVP, bvp_fixed, CNN_INPUT_SAMPLES);
    int eda_count = realtime_extract_window(SENSOR_EDA, eda_fixed, CNN_INPUT_SAMPLES);
    int temp_count = realtime_extract_window(SENSOR_TEMP, temp_fixed, CNN_INPUT_SAMPLES);
    
    // Validate sample counts
    if (acc_x_count < CNN_INPUT_SAMPLES || acc_y_count < CNN_INPUT_SAMPLES ||
        acc_z_count < CNN_INPUT_SAMPLES || bvp_count < CNN_INPUT_SAMPLES ||
        eda_count < CNN_INPUT_SAMPLES || temp_count < CNN_INPUT_SAMPLES) {
        ESP_LOGE(TAG, "Insufficient samples: ACC_X=%d, ACC_Y=%d, ACC_Z=%d, BVP=%d, EDA=%d, TEMP=%d",
                 acc_x_count, acc_y_count, acc_z_count, bvp_count, eda_count, temp_count);
        return -2;
    }
    
    ESP_LOGI(TAG, "Extracted %d samples from each sensor", CNN_INPUT_SAMPLES);
    
    /* ==================== STEP 2: Convert from fixed-point to float ==================== */
    
    float acc_x[CNN_INPUT_SAMPLES];
    float acc_y[CNN_INPUT_SAMPLES];
    float acc_z[CNN_INPUT_SAMPLES];
    float bvp[CNN_INPUT_SAMPLES];
    float eda[CNN_INPUT_SAMPLES];
    float temp[CNN_INPUT_SAMPLES];
    
    for (int i = 0; i < CNN_INPUT_SAMPLES; i++) {
        acc_x[i] = FIXED_TO_FLOAT(acc_x_fixed[i]);
        acc_y[i] = FIXED_TO_FLOAT(acc_y_fixed[i]);
        acc_z[i] = FIXED_TO_FLOAT(acc_z_fixed[i]);
        bvp[i] = FIXED_TO_FLOAT(bvp_fixed[i]);
        eda[i] = FIXED_TO_FLOAT(eda_fixed[i]);
        temp[i] = FIXED_TO_FLOAT(temp_fixed[i]);
    }
    
    /* ==================== STEP 3: Compute ACC magnitude ==================== */
    
    float acc_mag[CNN_INPUT_SAMPLES];
    int ret = compute_acc_magnitude(acc_x, acc_y, acc_z, acc_mag, CNN_INPUT_SAMPLES);
    if (ret != 0) {
        ESP_LOGE(TAG, "Failed to compute ACC magnitude");
        return -3;
    }
    
    ESP_LOGI(TAG, "Computed ACC magnitude from 3 axes");
    
    /* ==================== STEP 4: Copy to output tensor (before normalization) ==================== */
    
    memcpy(output->data[CNN_CHANNEL_ACC], acc_mag, CNN_INPUT_SAMPLES * sizeof(float));
    memcpy(output->data[CNN_CHANNEL_BVP], bvp, CNN_INPUT_SAMPLES * sizeof(float));
    memcpy(output->data[CNN_CHANNEL_EDA], eda, CNN_INPUT_SAMPLES * sizeof(float));
    memcpy(output->data[CNN_CHANNEL_TEMP], temp, CNN_INPUT_SAMPLES * sizeof(float));
    
    /* ==================== STEP 5: Z-score normalization per channel ==================== */
    
    // Normalize ACC magnitude
    ret = normalize_signal_zscore(output->data[CNN_CHANNEL_ACC], CNN_INPUT_SAMPLES);
    if (ret != 0) {
        ESP_LOGE(TAG, "Failed to normalize ACC channel");
        return -4;
    }
    
    // Normalize BVP
    ret = normalize_signal_zscore(output->data[CNN_CHANNEL_BVP], CNN_INPUT_SAMPLES);
    if (ret != 0) {
        ESP_LOGE(TAG, "Failed to normalize BVP channel");
        return -5;
    }
    
    // Normalize EDA
    ret = normalize_signal_zscore(output->data[CNN_CHANNEL_EDA], CNN_INPUT_SAMPLES);
    if (ret != 0) {
        ESP_LOGE(TAG, "Failed to normalize EDA channel");
        return -6;
    }
    
    // Normalize TEMP
    ret = normalize_signal_zscore(output->data[CNN_CHANNEL_TEMP], CNN_INPUT_SAMPLES);
    if (ret != 0) {
        ESP_LOGE(TAG, "Failed to normalize TEMP channel");
        return -7;
    }
    
    ESP_LOGI(TAG, "Applied z-score normalization to all channels");
    
    /* ==================== STEP 6: Compute statistics for debugging ==================== */
    
    signal_stats_t acc_stats, bvp_stats, eda_stats, temp_stats;
    compute_signal_stats(output->data[CNN_CHANNEL_ACC], CNN_INPUT_SAMPLES, &acc_stats);
    compute_signal_stats(output->data[CNN_CHANNEL_BVP], CNN_INPUT_SAMPLES, &bvp_stats);
    compute_signal_stats(output->data[CNN_CHANNEL_EDA], CNN_INPUT_SAMPLES, &eda_stats);
    compute_signal_stats(output->data[CNN_CHANNEL_TEMP], CNN_INPUT_SAMPLES, &temp_stats);
    
    ESP_LOGI(TAG, "Channel statistics:");
    ESP_LOGI(TAG, "  ACC:  mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
             acc_stats.mean, acc_stats.std, acc_stats.min, acc_stats.max);
    ESP_LOGI(TAG, "  BVP:  mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
             bvp_stats.mean, bvp_stats.std, bvp_stats.min, bvp_stats.max);
    ESP_LOGI(TAG, "  EDA:  mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
             eda_stats.mean, eda_stats.std, eda_stats.min, eda_stats.max);
    ESP_LOGI(TAG, "  TEMP: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
             temp_stats.mean, temp_stats.std, temp_stats.min, temp_stats.max);
    
    /* ==================== STEP 7: Finalize output ==================== */
    
    output->preprocessing_time_ms = (xTaskGetTickCount() * portTICK_PERIOD_MS) - start_ms;
    output->success = true;
    output->timestamp = xTaskGetTickCount();
    
    ESP_LOGI(TAG, "✅ Preprocessing completed in %lu ms", output->preprocessing_time_ms);
    
    return 0;
}

/* ==================== DEBUGGING UTILITIES ==================== */

void print_cnn_input_tensor(const cnn_input_tensor_t *tensor) {
    if (!tensor) return;
    
    ESP_LOGI(TAG, "=== CNN Input Tensor ===");
    ESP_LOGI(TAG, "Shape: (%d, %d)", CNN_INPUT_CHANNELS, CNN_INPUT_SAMPLES);
    ESP_LOGI(TAG, "Success: %s", tensor->success ? "YES" : "NO");
    ESP_LOGI(TAG, "Processing time: %lu ms", tensor->preprocessing_time_ms);
    ESP_LOGI(TAG, "Timestamp: %lu", tensor->timestamp);
    
    // Print first 10 samples of each channel
    const char *channel_names[] = {"ACC", "BVP", "EDA", "TEMP"};
    for (int ch = 0; ch < CNN_INPUT_CHANNELS; ch++) {
        ESP_LOGI(TAG, "%s (first 10): ", channel_names[ch]);
        for (int i = 0; i < 10 && i < CNN_INPUT_SAMPLES; i++) {
            ESP_LOGI(TAG, "  [%d] = %.6f", i, tensor->data[ch][i]);
        }
    }
    ESP_LOGI(TAG, "========================");
}

bool validate_preprocessing(const cnn_input_tensor_t *actual,
                           const float *expected,
                           cnn_channel_t channel,
                           float tolerance) {
    if (!actual || !expected || channel >= CNN_INPUT_CHANNELS) {
        ESP_LOGE(TAG, "Invalid parameters for validate_preprocessing");
        return false;
    }
    
    const char *channel_names[] = {"ACC", "BVP", "EDA", "TEMP"};
    ESP_LOGI(TAG, "Validating %s channel (tolerance=%.6f)...", channel_names[channel], tolerance);
    
    int errors = 0;
    float max_error = 0.0f;
    float avg_error = 0.0f;
    
    for (int i = 0; i < CNN_INPUT_SAMPLES; i++) {
        float diff = fabsf(actual->data[channel][i] - expected[i]);
        avg_error += diff;
        
        if (diff > max_error) {
            max_error = diff;
        }
        
        if (diff > tolerance) {
            if (errors < 5) {  // Only print first 5 errors
                ESP_LOGW(TAG, "  Sample[%d]: actual=%.6f, expected=%.6f, diff=%.6f",
                         i, actual->data[channel][i], expected[i], diff);
            }
            errors++;
        }
    }
    
    avg_error /= CNN_INPUT_SAMPLES;
    
    ESP_LOGI(TAG, "Validation results:");
    ESP_LOGI(TAG, "  Errors: %d / %d samples", errors, CNN_INPUT_SAMPLES);
    ESP_LOGI(TAG, "  Max error: %.6f", max_error);
    ESP_LOGI(TAG, "  Avg error: %.6f", avg_error);
    
    if (errors == 0) {
        ESP_LOGI(TAG, "✅ Validation PASSED");
        return true;
    } else {
        ESP_LOGE(TAG, "❌ Validation FAILED (%d errors)", errors);
        return false;
    }
}

uint32_t signal_preprocessor_get_memory_usage(void) {
    // CNN input tensor size
    uint32_t tensor_size = sizeof(cnn_input_tensor_t);
    
    // Temporary buffers in preprocess_for_cnn()
    uint32_t temp_buffers = (CNN_INPUT_SAMPLES * sizeof(fixed_point_t) * 6) +  // Fixed-point buffers
                            (CNN_INPUT_SAMPLES * sizeof(float) * 7);             // Float buffers
    
    return tensor_size + temp_buffers;
}
