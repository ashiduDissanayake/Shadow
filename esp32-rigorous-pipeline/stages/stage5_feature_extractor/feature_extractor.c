/*
 * ESP32 Optimized Feature Extractor Implementation
 * 
 * High-performance feature extraction using fixed-point arithmetic
 * and memory-efficient algorithms optimized for ESP32 constraints
 */

#include "feature_extractor.h"
#include <string.h>
#include <stdio.h>

// Static workspace for feature extraction (prevents fragmentation)
static feature_workspace_t g_workspace;

// millis() function declaration - will be provided by ESP32 framework or test
extern uint32_t millis(void);

int feature_extractor_init(feature_workspace_t *workspace) {
    if (!workspace) {
        return -1;
    }
    
    // Clear workspace memory
    memset(workspace->workspace, 0, sizeof(workspace->workspace));
    memset(workspace->temp_stats, 0, sizeof(workspace->temp_stats));
    workspace->initialized = true;
    
    return 0;
}

int extract_features(multi_sensor_buffer_t *msb, 
                    feature_workspace_t *workspace,
                    feature_vector_t *result) {
    
    if (!msb || !workspace || !result || !workspace->initialized) {
        return -1;
    }
    
    // Check if buffer is ready for processing
    if (!buffer_is_ready_for_processing(msb)) {
        return -2; // Not enough data
    }
    
    uint32_t start_time = millis();
    
    // Initialize result
    memset(result->features, 0, sizeof(result->features));
    result->success = false;
    result->timestamp_ms = start_time;
    
    // Extract features by category
    int ret = 0;
    
    // BVP features (indices 0-7)
    ret = extract_bvp_features(msb, workspace, &result->features[FEATURE_BVP_MEAN]);
    if (ret != 0) return ret;
    
    // ACC features (indices 8-22)
    ret = extract_acc_features(msb, workspace, &result->features[FEATURE_ACC_X_MEAN]);
    if (ret != 0) return ret;
    
    // EDA features (indices 23-26)
    ret = extract_eda_features(msb, workspace, &result->features[FEATURE_EDA_MEAN]);
    if (ret != 0) return ret;
    
    // TEMP features (indices 27-29)
    ret = extract_temp_features(msb, workspace, &result->features[FEATURE_TEMP_MEAN]);
    if (ret != 0) return ret;
    
    // Finalize result
    result->extraction_time_ms = millis() - start_time;
    result->success = true;
    
    return 0;
}

int extract_bvp_features(multi_sensor_buffer_t *msb,
                        feature_workspace_t *workspace,
                        fixed_point_t *features) {
    
    // Read BVP window data
    int sample_count = buffer_read_window(msb, LAYER_BVP, workspace->workspace, BVP_BUFFER_SIZE);
    
    if (sample_count <= 0) {
        return -1;
    }
    
    // Compute comprehensive statistics
    stats_result_t stats;
    int ret = compute_statistics(workspace->workspace, sample_count, 
                               workspace->workspace, &stats);
    
    if (ret != 0) {
        return ret;
    }
    
    // Store BVP features
    features[0] = stats.mean;    // BVP_MEAN
    features[1] = stats.std;     // BVP_STD
    features[2] = stats.min;     // BVP_MIN
    features[3] = stats.max;     // BVP_MAX
    features[4] = stats.median;  // BVP_MEDIAN
    features[5] = stats.range;   // BVP_RANGE
    features[6] = stats.iqr;     // BVP_IQR
    features[7] = stats.energy;  // BVP_ENERGY
    
    return 0;
}

int extract_acc_features(multi_sensor_buffer_t *msb,
                        feature_workspace_t *workspace,
                        fixed_point_t *features) {
    
    // Process each accelerometer axis
    sensor_layer_t acc_layers[] = {LAYER_ACC_X, LAYER_ACC_Y, LAYER_ACC_Z};
    
    for (int axis = 0; axis < 3; axis++) {
        // Read ACC window data
        int sample_count = buffer_read_window(msb, acc_layers[axis], 
                                            workspace->workspace, ACC_BUFFER_SIZE);
        if (sample_count <= 0) {
            return -1;
        }
        
        // Compute statistics for this axis
        stats_result_t stats;
        int ret = compute_statistics(workspace->workspace, sample_count, 
                                   workspace->workspace, &stats);
        if (ret != 0) {
            return ret;
        }
        
        // Store ACC features for this axis (5 features per axis)
        int base_idx = axis * 5;
        features[base_idx + 0] = stats.mean;   // ACC_X/Y/Z_MEAN
        features[base_idx + 1] = stats.std;    // ACC_X/Y/Z_STD
        features[base_idx + 2] = stats.min;    // ACC_X/Y/Z_MIN
        features[base_idx + 3] = stats.max;    // ACC_X/Y/Z_MAX
        features[base_idx + 4] = stats.energy; // ACC_X/Y/Z_ENERGY
    }
    
    return 0;
}

int extract_eda_features(multi_sensor_buffer_t *msb,
                        feature_workspace_t *workspace,
                        fixed_point_t *features) {
    
    // Read EDA window data
    int sample_count = buffer_read_window(msb, LAYER_EDA, workspace->workspace, EDA_BUFFER_SIZE);
    if (sample_count <= 0) {
        return -1;
    }
    
    // Compute basic statistics (no median/IQR needed for EDA)
    fixed_point_t mean = compute_mean(workspace->workspace, sample_count);
    fixed_point_t std = compute_std(workspace->workspace, sample_count, mean);
    
    fixed_point_t min_val, max_val;
    find_min_max(workspace->workspace, sample_count, &min_val, &max_val);
    
    // Store EDA features
    features[0] = mean;                    // EDA_MEAN
    features[1] = std;                     // EDA_STD
    features[2] = min_val;                 // EDA_MIN
    features[3] = max_val;                 // EDA_MAX
    
    return 0;
}

int extract_temp_features(multi_sensor_buffer_t *msb,
                         feature_workspace_t *workspace,
                         fixed_point_t *features) {
    
    // Read TEMP window data
    int sample_count = buffer_read_window(msb, LAYER_TEMP, workspace->workspace, TEMP_BUFFER_SIZE);
    if (sample_count <= 0) {
        return -1;
    }
    
    // Compute basic statistics
    fixed_point_t mean = compute_mean(workspace->workspace, sample_count);
    fixed_point_t std = compute_std(workspace->workspace, sample_count, mean);
    
    fixed_point_t min_val, max_val;
    find_min_max(workspace->workspace, sample_count, &min_val, &max_val);
    fixed_point_t range = max_val - min_val;
    
    // Store TEMP features
    features[0] = mean;                    // TEMP_MEAN
    features[1] = std;                     // TEMP_STD
    features[2] = range;                   // TEMP_RANGE
    
    return 0;
}

int compute_statistics(const fixed_point_t *data, 
                      uint16_t count,
                      fixed_point_t *workspace,
                      stats_result_t *result) {
    
    if (!data || !workspace || !result || count == 0) {
        return -1;
    }
    
    // Compute basic statistics first (no workspace needed)
    result->mean = compute_mean(data, count);
    result->std = compute_std(data, count, result->mean);
    result->energy = compute_energy(data, count);
    
    find_min_max(data, count, &result->min, &result->max);
    result->range = result->max - result->min;
    
    // Copy data to workspace for median/IQR computation
    memcpy(workspace, data, count * sizeof(fixed_point_t));
    
    // Compute median and IQR (requires sorting workspace)
    result->median = quickselect(workspace, count, count / 2);
    
    // Need fresh copy for IQR since quickselect modifies the array
    memcpy(workspace, data, count * sizeof(fixed_point_t));
    
    // IQR computation (Q3 - Q1)
    fixed_point_t q1 = quickselect(workspace, count, count / 4);
    
    // Fresh copy again
    memcpy(workspace, data, count * sizeof(fixed_point_t));
    fixed_point_t q3 = quickselect(workspace, count, (3 * count) / 4);
    result->iqr = q3 - q1;
    
    return 0;
}

// Utility function implementations

fixed_point_t compute_mean(const fixed_point_t *data, uint16_t count) {
    int64_t sum = 0; // Use 64-bit to prevent overflow
    
    for (uint16_t i = 0; i < count; i++) {
        sum += data[i];
    }
    
    return (fixed_point_t)(sum / count);
}

fixed_point_t compute_std(const fixed_point_t *data, uint16_t count, fixed_point_t mean) {
    int64_t variance_sum = 0;
    
    for (uint16_t i = 0; i < count; i++) {
        int64_t diff = (int64_t)(data[i] - mean);
        variance_sum += (diff * diff) / FIXED_POINT_SCALE;
    }
    
    fixed_point_t variance = (fixed_point_t)(variance_sum / count);
    return fixed_sqrt(variance);
}

fixed_point_t compute_energy(const fixed_point_t *data, uint16_t count) {
    int64_t energy_sum = 0;
    
    for (uint16_t i = 0; i < count; i++) {
        int64_t val = data[i];
        energy_sum += (val * val) / FIXED_POINT_SCALE;
    }
    
    return (fixed_point_t)(energy_sum / count);
}

void find_min_max(const fixed_point_t *data, uint16_t count, 
                  fixed_point_t *min_val, fixed_point_t *max_val) {
    
    if (count == 0) {
        *min_val = *max_val = 0;
        return;
    }
    
    *min_val = *max_val = data[0];
    
    for (uint16_t i = 1; i < count; i++) {
        if (data[i] < *min_val) {
            *min_val = data[i];
        }
        if (data[i] > *max_val) {
            *max_val = data[i];
        }
    }
}

fixed_point_t quickselect(fixed_point_t *data, uint16_t count, uint16_t k) {
    if (count == 0 || k >= count) {
        return 0;
    }
    
    // Simple implementation for ESP32 efficiency
    // For small arrays, use insertion sort approach
    if (count <= 10) {
        // Sort small array and return k-th element
        for (uint16_t i = 1; i < count; i++) {
            fixed_point_t key = data[i];
            int j = i - 1;
            while (j >= 0 && data[j] > key) {
                data[j + 1] = data[j];
                j--;
            }
            data[j + 1] = key;
        }
        return data[k];
    }
    
    // For larger arrays, use simplified quickselect
    uint16_t left = 0, right = count - 1;
    
    while (left < right) {
        // Choose pivot (middle element)
        uint16_t pivot_idx = (left + right) / 2;
        fixed_point_t pivot = data[pivot_idx];
        
        // Swap pivot to end
        fixed_point_t temp = data[pivot_idx];
        data[pivot_idx] = data[right];
        data[right] = temp;
        
        // Partition
        uint16_t store_idx = left;
        for (uint16_t i = left; i < right; i++) {
            if (data[i] < pivot) {
                temp = data[i];
                data[i] = data[store_idx];
                data[store_idx] = temp;
                store_idx++;
            }
        }
        
        // Move pivot to final position
        temp = data[store_idx];
        data[store_idx] = data[right];
        data[right] = temp;
        
        // Recurse on appropriate side
        if (k == store_idx) {
            return data[k];
        } else if (k < store_idx) {
            right = store_idx - 1;
        } else {
            left = store_idx + 1;
        }
    }
    
    return data[k];
}

// Fixed-point math utilities

fixed_point_t fixed_sqrt(fixed_point_t x) {
    if (x <= 0) return 0;
    
    // Newton's method for square root
    fixed_point_t result = x;
    fixed_point_t prev_result;
    
    // Limit iterations for ESP32 performance
    for (int i = 0; i < 10; i++) {
        prev_result = result;
        result = (result + fixed_div(x, result)) >> 1;
        
        // Check for convergence
        fixed_point_t diff = (result > prev_result) ? 
                           (result - prev_result) : (prev_result - result);
        if (diff < (FIXED_POINT_SCALE >> 8)) {
            break; // Converged to sufficient precision
        }
    }
    
    return result;
}

fixed_point_t fixed_div(fixed_point_t dividend, fixed_point_t divisor) {
    if (divisor == 0) return 0;
    
    // Shift dividend left for precision, then divide
    int64_t temp = ((int64_t)dividend) * FIXED_POINT_SCALE;
    return (fixed_point_t)(temp / divisor);
}

// Utility functions

void print_feature_vector(const feature_vector_t *features) {
    if (!features) return;
    
    printf("\n📊 Feature Vector (30 features):\n");
    printf("Status: %s | Extraction time: %u ms\n", 
           features->success ? "SUCCESS" : "FAILED", 
           features->extraction_time_ms);
    
    const char* feature_names[] = {
        "BVP_MEAN", "BVP_STD", "BVP_MIN", "BVP_MAX", "BVP_MEDIAN", "BVP_RANGE", "BVP_IQR", "BVP_ENERGY",
        "ACC_X_MEAN", "ACC_X_STD", "ACC_X_MIN", "ACC_X_MAX", "ACC_X_ENERGY",
        "ACC_Y_MEAN", "ACC_Y_STD", "ACC_Y_MIN", "ACC_Y_MAX", "ACC_Y_ENERGY",
        "ACC_Z_MEAN", "ACC_Z_STD", "ACC_Z_MIN", "ACC_Z_MAX", "ACC_Z_ENERGY",
        "EDA_MEAN", "EDA_STD", "EDA_MIN", "EDA_MAX",
        "TEMP_MEAN", "TEMP_STD", "TEMP_RANGE"
    };
    
    for (int i = 0; i < NUM_FEATURES; i++) {
        printf("  [%2d] %-12s = %8.4f\n", i, feature_names[i], 
               FIXED_TO_FLOAT(features->features[i]));
    }
}

uint32_t feature_extractor_get_memory_usage(void) {
    return sizeof(feature_workspace_t);
}