/*
 * Calibration System Implementation
 * ESP32-S3 Shadow Project
 */

#include "calibration.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "nvs.h"
#include <string.h>
#include <math.h>

static const char *TAG = "Calibration";

// Global calibration data
static calibration_data_t g_calibration = {0};
static bool g_calibration_initialized = false;

/* ==================== INITIALIZATION ==================== */

int calibration_init(void) {
    if (g_calibration_initialized) {
        ESP_LOGW(TAG, "Calibration already initialized");
        return 0;
    }
    
    ESP_LOGI(TAG, "Initializing calibration system");
    
    // Clear calibration data
    memset(&g_calibration, 0, sizeof(calibration_data_t));
    g_calibration.state = CAL_STATE_NOT_STARTED;
    g_calibration.calibration_version = 1;
    
    // Try to load from NVS
    if (calibration_load_from_nvs() == 0) {
        ESP_LOGI(TAG, "✅ Loaded calibration from NVS");
        g_calibration.state = CAL_STATE_LOADED;
    } else {
        ESP_LOGI(TAG, "No calibration found in NVS");
    }
    
    g_calibration_initialized = true;
    calibration_print_status();
    
    return 0;
}

void calibration_deinit(void) {
    g_calibration_initialized = false;
    memset(&g_calibration, 0, sizeof(calibration_data_t));
}

/* ==================== CALIBRATION CONTROL ==================== */

int calibration_start(void) {
    if (!g_calibration_initialized) {
        ESP_LOGE(TAG, "Calibration not initialized");
        return -1;
    }
    
    ESP_LOGI(TAG, "🎯 Starting calibration session (%d seconds, %d samples required)",
             CALIBRATION_DURATION_SECONDS, CALIBRATION_REQUIRED_SAMPLES);
    
    // Clear previous calibration data
    memset(&g_calibration, 0, sizeof(calibration_data_t));
    
    // Initialize state
    g_calibration.state = CAL_STATE_IN_PROGRESS;
    g_calibration.start_time_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
    g_calibration.calibration_version = 1;
    
    // Initialize channels
    for (int i = 0; i < CNN_INPUT_CHANNELS; i++) {
        g_calibration.channels[i].valid = false;
        g_calibration.channels[i].sample_count = 0;
        g_calibration.channels[i].running_sum = 0.0;
        g_calibration.channels[i].running_sum_sq = 0.0;
    }
    
    return 0;
}

int calibration_stop(bool force_stop) {
    if (!g_calibration_initialized) {
        ESP_LOGE(TAG, "Calibration not initialized");
        return -1;
    }
    
    if (g_calibration.state != CAL_STATE_IN_PROGRESS) {
        ESP_LOGW(TAG, "Calibration not in progress");
        return -2;
    }
    
    g_calibration.end_time_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
    uint32_t duration_ms = g_calibration.end_time_ms - g_calibration.start_time_ms;
    
    ESP_LOGI(TAG, "⏹️ Stopping calibration after %lu ms (%lu samples)",
             duration_ms, g_calibration.total_samples);
    
    // Check if we have enough samples
    if (g_calibration.total_samples < CALIBRATION_MIN_SAMPLES && !force_stop) {
        ESP_LOGE(TAG, "❌ Insufficient samples: %lu < %d (minimum)",
                 g_calibration.total_samples, CALIBRATION_MIN_SAMPLES);
        g_calibration.state = CAL_STATE_FAILED;
        return -3;
    }
    
    // Finalize statistics for each channel
    int valid_channels = 0;
    for (int ch = 0; ch < CNN_INPUT_CHANNELS; ch++) {
        channel_calibration_t *cal = &g_calibration.channels[ch];
        
        if (cal->sample_count < CALIBRATION_MIN_SAMPLES && !force_stop) {
            ESP_LOGW(TAG, "Channel %d: Insufficient samples (%lu)", ch, cal->sample_count);
            cal->valid = false;
            continue;
        }
        
        // Compute final mean
        cal->mean = (float)(cal->running_sum / cal->sample_count);
        
        // Compute final std deviation
        double variance = (cal->running_sum_sq / cal->sample_count) - (cal->mean * cal->mean);
        if (variance < 0.0) variance = 0.0;  // Numerical stability
        cal->std = sqrtf((float)variance);
        
        // Validate std (avoid division by zero later)
        if (cal->std < 1e-6f) {
            ESP_LOGW(TAG, "Channel %d: Std too small (%.6f), setting to 1.0", ch, cal->std);
            cal->std = 1.0f;
        }
        
        cal->valid = true;
        valid_channels++;
        
        ESP_LOGI(TAG, "Channel %d: mean=%.6f, std=%.6f (%lu samples)",
                 ch, cal->mean, cal->std, cal->sample_count);
    }
    
    if (valid_channels == CNN_INPUT_CHANNELS) {
        g_calibration.state = CAL_STATE_COMPLETED;
        ESP_LOGI(TAG, "✅ Calibration completed successfully");
        
        // Save to NVS
        if (calibration_save_to_nvs() == 0) {
            ESP_LOGI(TAG, "✅ Calibration saved to NVS");
        } else {
            ESP_LOGW(TAG, "⚠️ Failed to save calibration to NVS");
        }
        
        return 0;
    } else {
        g_calibration.state = CAL_STATE_FAILED;
        ESP_LOGE(TAG, "❌ Calibration failed: only %d/%d channels valid",
                 valid_channels, CNN_INPUT_CHANNELS);
        return -4;
    }
}

int calibration_reset(void) {
    ESP_LOGI(TAG, "🔄 Resetting calibration");
    
    // Clear memory
    memset(&g_calibration, 0, sizeof(calibration_data_t));
    g_calibration.state = CAL_STATE_NOT_STARTED;
    
    // Clear NVS
    nvs_handle_t nvs_handle;
    esp_err_t err = nvs_open(CALIBRATION_NVS_NAMESPACE, NVS_READWRITE, &nvs_handle);
    if (err == ESP_OK) {
        nvs_erase_key(nvs_handle, CALIBRATION_NVS_KEY);
        nvs_commit(nvs_handle);
        nvs_close(nvs_handle);
        ESP_LOGI(TAG, "✅ Calibration cleared from NVS");
    }
    
    return 0;
}

/* ==================== DATA COLLECTION ==================== */

int calibration_update(const float *samples, uint16_t length, cnn_channel_t channel) {
    if (!g_calibration_initialized) {
        return -1;
    }
    
    if (g_calibration.state != CAL_STATE_IN_PROGRESS) {
        return -2;  // Not collecting samples
    }
    
    if (channel >= CNN_INPUT_CHANNELS || !samples || length == 0) {
        ESP_LOGE(TAG, "Invalid parameters for calibration_update");
        return -3;
    }
    
    channel_calibration_t *cal = &g_calibration.channels[channel];
    
    // Update running statistics
    for (uint16_t i = 0; i < length; i++) {
        float value = samples[i];
        cal->running_sum += value;
        cal->running_sum_sq += (double)(value * value);
        cal->sample_count++;
    }
    
    // Only increment total_samples for channel 0 (ACC) to avoid counting 4x
    // All channels receive the same number of samples per batch
    if (channel == CNN_CHANNEL_ACC) {
        g_calibration.total_samples += length;
        
        // Log progress every 120 samples (30 seconds @ 4Hz)
        if (g_calibration.total_samples % 120 == 0) {
            float progress = calibration_get_progress();
            uint32_t remaining_sec = calibration_get_remaining_time();
            ESP_LOGI(TAG, "📊 Calibration progress: %.1f%% (%lu/%d samples, %lu sec remaining)",
                     progress * 100.0f, g_calibration.total_samples, 
                     CALIBRATION_REQUIRED_SAMPLES, remaining_sec);
        }
        
        // Auto-stop if reached required samples
        if (g_calibration.total_samples >= CALIBRATION_REQUIRED_SAMPLES) {
            ESP_LOGI(TAG, "✅ Calibration auto-complete - required samples reached");
            calibration_stop(false);
        }
    }
    
    return 0;
}

/* ==================== NORMALIZATION ==================== */

int calibration_normalize(float *signal, uint16_t length, cnn_channel_t channel) {
    if (!signal || length == 0 || channel >= CNN_INPUT_CHANNELS) {
        ESP_LOGE(TAG, "Invalid parameters for calibration_normalize");
        return -1;
    }
    
    // Check if we have valid calibration for this channel
    if (!calibration_is_calibrated() || !g_calibration.channels[channel].valid) {
        // Fallback to local z-score normalization
        ESP_LOGW(TAG, "No calibration for channel %d, using local z-score", channel);
        return normalize_signal_zscore(signal, length);
    }
    
    channel_calibration_t *cal = &g_calibration.channels[channel];
    
    // DEBUG: Print first few values before normalization
    ESP_LOGI(TAG, "🔧 Channel %d: Applying calibration normalization (cal_mean=%.6f, cal_std=%.6f)", 
             channel, cal->mean, cal->std);
    ESP_LOGI(TAG, "   Sample values BEFORE: [0]=%.6f, [1]=%.6f, [2]=%.6f", 
             signal[0], signal[1], signal[2]);
    
    // Normalize using calibration statistics: (x - mean) / std
    for (uint16_t i = 0; i < length; i++) {
        signal[i] = (signal[i] - cal->mean) / cal->std;
    }
    
    ESP_LOGI(TAG, "   Sample values AFTER: [0]=%.6f, [1]=%.6f, [2]=%.6f", 
             signal[0], signal[1], signal[2]);
    
    return 0;
}

/* ==================== STATUS & QUERIES ==================== */

bool calibration_is_calibrated(void) {
    return (g_calibration.state == CAL_STATE_COMPLETED || 
            g_calibration.state == CAL_STATE_LOADED);
}

calibration_state_t calibration_get_state(void) {
    return g_calibration.state;
}

float calibration_get_progress(void) {
    if (g_calibration.state != CAL_STATE_IN_PROGRESS) {
        return (g_calibration.state == CAL_STATE_COMPLETED || 
                g_calibration.state == CAL_STATE_LOADED) ? 1.0f : 0.0f;
    }
    
    return (float)g_calibration.total_samples / CALIBRATION_REQUIRED_SAMPLES;
}

int calibration_get_stats(cnn_channel_t channel, float *mean, float *std) {
    if (channel >= CNN_INPUT_CHANNELS || !mean || !std) {
        return -1;
    }
    
    if (!calibration_is_calibrated() || !g_calibration.channels[channel].valid) {
        return -2;
    }
    
    *mean = g_calibration.channels[channel].mean;
    *std = g_calibration.channels[channel].std;
    
    return 0;
}

uint32_t calibration_get_remaining_time(void) {
    if (g_calibration.state != CAL_STATE_IN_PROGRESS) {
        return 0;
    }
    
    uint32_t current_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
    uint32_t elapsed_ms = current_ms - g_calibration.start_time_ms;
    uint32_t target_ms = CALIBRATION_DURATION_SECONDS * 1000;
    
    if (elapsed_ms >= target_ms) {
        return 0;
    }
    
    return (target_ms - elapsed_ms) / 1000;  // Convert to seconds
}

/* ==================== PERSISTENCE (NVS) ==================== */

int calibration_save_to_nvs(void) {
    nvs_handle_t nvs_handle;
    esp_err_t err;
    
    // Open NVS
    err = nvs_open(CALIBRATION_NVS_NAMESPACE, NVS_READWRITE, &nvs_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS: %s", esp_err_to_name(err));
        return -1;
    }
    
    // Save calibration data as blob
    err = nvs_set_blob(nvs_handle, CALIBRATION_NVS_KEY, &g_calibration, sizeof(calibration_data_t));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to write NVS: %s", esp_err_to_name(err));
        nvs_close(nvs_handle);
        return -2;
    }
    
    // Commit
    err = nvs_commit(nvs_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to commit NVS: %s", esp_err_to_name(err));
        nvs_close(nvs_handle);
        return -3;
    }
    
    nvs_close(nvs_handle);
    ESP_LOGI(TAG, "✅ Calibration saved to NVS (%d bytes)", sizeof(calibration_data_t));
    
    return 0;
}

int calibration_load_from_nvs(void) {
    nvs_handle_t nvs_handle;
    esp_err_t err;
    
    // Open NVS
    err = nvs_open(CALIBRATION_NVS_NAMESPACE, NVS_READONLY, &nvs_handle);
    if (err != ESP_OK) {
        if (err == ESP_ERR_NVS_NOT_FOUND) {
            ESP_LOGI(TAG, "No calibration in NVS (namespace not found)");
        } else {
            ESP_LOGE(TAG, "Failed to open NVS: %s", esp_err_to_name(err));
        }
        return -1;
    }
    
    // Read calibration data
    size_t required_size = sizeof(calibration_data_t);
    err = nvs_get_blob(nvs_handle, CALIBRATION_NVS_KEY, &g_calibration, &required_size);
    if (err != ESP_OK) {
        if (err == ESP_ERR_NVS_NOT_FOUND) {
            ESP_LOGI(TAG, "No calibration data in NVS");
        } else {
            ESP_LOGE(TAG, "Failed to read NVS: %s", esp_err_to_name(err));
        }
        nvs_close(nvs_handle);
        return -2;
    }
    
    nvs_close(nvs_handle);
    
    // Validate loaded data
    if (g_calibration.calibration_version != 1) {
        ESP_LOGW(TAG, "Unsupported calibration version: %lu", g_calibration.calibration_version);
        return -3;
    }
    
    // Check if all channels are valid
    int valid_channels = 0;
    for (int i = 0; i < CNN_INPUT_CHANNELS; i++) {
        if (g_calibration.channels[i].valid) {
            valid_channels++;
        }
    }
    
    if (valid_channels != CNN_INPUT_CHANNELS) {
        ESP_LOGW(TAG, "Incomplete calibration: only %d/%d channels valid", 
                 valid_channels, CNN_INPUT_CHANNELS);
        return -4;
    }
    
    ESP_LOGI(TAG, "✅ Loaded valid calibration from NVS");
    calibration_print_status();
    
    return 0;
}

bool calibration_exists_in_nvs(void) {
    nvs_handle_t nvs_handle;
    esp_err_t err;
    
    err = nvs_open(CALIBRATION_NVS_NAMESPACE, NVS_READONLY, &nvs_handle);
    if (err != ESP_OK) {
        return false;
    }
    
    size_t required_size = 0;
    err = nvs_get_blob(nvs_handle, CALIBRATION_NVS_KEY, NULL, &required_size);
    nvs_close(nvs_handle);
    
    return (err == ESP_OK && required_size == sizeof(calibration_data_t));
}

/* ==================== DEBUGGING ==================== */

void calibration_print_status(void) {
    const char *state_names[] = {
        "NOT_STARTED", "IN_PROGRESS", "COMPLETED", "LOADED", "FAILED"
    };
    const char *channel_names[] = {"ACC", "BVP", "EDA", "TEMP"};
    
    ESP_LOGI(TAG, "=== Calibration Status ===");
    ESP_LOGI(TAG, "State: %s", state_names[g_calibration.state]);
    ESP_LOGI(TAG, "Total samples: %lu / %d", g_calibration.total_samples, CALIBRATION_REQUIRED_SAMPLES);
    ESP_LOGI(TAG, "Progress: %.1f%%", calibration_get_progress() * 100.0f);
    
    if (calibration_is_calibrated()) {
        ESP_LOGI(TAG, "Calibration duration: %lu ms", 
                 g_calibration.end_time_ms - g_calibration.start_time_ms);
        ESP_LOGI(TAG, "Channel statistics:");
        for (int i = 0; i < CNN_INPUT_CHANNELS; i++) {
            if (g_calibration.channels[i].valid) {
                ESP_LOGI(TAG, "  %s: mean=%.6f, std=%.6f (%lu samples)",
                         channel_names[i],
                         g_calibration.channels[i].mean,
                         g_calibration.channels[i].std,
                         g_calibration.channels[i].sample_count);
            } else {
                ESP_LOGI(TAG, "  %s: INVALID", channel_names[i]);
            }
        }
    } else if (g_calibration.state == CAL_STATE_IN_PROGRESS) {
        uint32_t remaining = calibration_get_remaining_time();
        ESP_LOGI(TAG, "Remaining time: %lu seconds", remaining);
    }
    
    ESP_LOGI(TAG, "==========================");
}

uint32_t calibration_get_memory_usage(void) {
    return sizeof(calibration_data_t);
}
