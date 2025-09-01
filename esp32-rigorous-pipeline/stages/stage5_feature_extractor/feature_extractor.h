/*
 * ESP32 Optimized Feature Extractor
 * Calculates 30 features from multi-sensor data for stress detection
 * 
 * Memory-efficient implementation using fixed-point arithmetic
 * and in-place operations for ESP32 deployment
 */

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <stdint.h>
#include <stdbool.h>
#include "sensor_buffer.h"

// Feature vector configuration
#define NUM_FEATURES 30
#define FEATURE_VECTOR_SIZE (NUM_FEATURES * sizeof(fixed_point_t))

// Feature categories (matches Stage 4 model expectations)
typedef enum {
    // BVP-based features (8 features)
    FEATURE_BVP_MEAN = 0,
    FEATURE_BVP_STD,
    FEATURE_BVP_MIN,
    FEATURE_BVP_MAX,
    FEATURE_BVP_MEDIAN,
    FEATURE_BVP_RANGE,
    FEATURE_BVP_IQR,
    FEATURE_BVP_ENERGY,
    
    // ACC-based features (15 features: 5 per axis)
    FEATURE_ACC_X_MEAN,
    FEATURE_ACC_X_STD,
    FEATURE_ACC_X_MIN,
    FEATURE_ACC_X_MAX,
    FEATURE_ACC_X_ENERGY,
    
    FEATURE_ACC_Y_MEAN,
    FEATURE_ACC_Y_STD,
    FEATURE_ACC_Y_MIN,
    FEATURE_ACC_Y_MAX,
    FEATURE_ACC_Y_ENERGY,
    
    FEATURE_ACC_Z_MEAN,
    FEATURE_ACC_Z_STD,
    FEATURE_ACC_Z_MIN,
    FEATURE_ACC_Z_MAX,
    FEATURE_ACC_Z_ENERGY,
    
    // EDA-based features (4 features)
    FEATURE_EDA_MEAN,
    FEATURE_EDA_STD,
    FEATURE_EDA_MIN,
    FEATURE_EDA_MAX,
    
    // TEMP-based features (3 features)
    FEATURE_TEMP_MEAN,
    FEATURE_TEMP_STD,
    FEATURE_TEMP_RANGE
} feature_index_t;

// Feature extraction result
typedef struct {
    fixed_point_t features[NUM_FEATURES];  // 30 features as fixed-point
    uint32_t extraction_time_ms;           // Time taken for extraction
    bool success;                          // Extraction status
    uint32_t timestamp_ms;                 // When features were extracted
} feature_vector_t;

// Computation workspace for feature extraction
// Static allocation to prevent memory fragmentation
typedef struct {
    fixed_point_t workspace[BVP_BUFFER_SIZE * 2];  // Larger workspace for sorting + temp space
    uint32_t temp_stats[4];                        // Temporary statistics storage
    bool initialized;                              // Workspace initialization status
} feature_workspace_t;

// Statistics computation result
typedef struct {
    fixed_point_t mean;
    fixed_point_t std;
    fixed_point_t min;
    fixed_point_t max;
    fixed_point_t median;
    fixed_point_t range;
    fixed_point_t iqr;
    fixed_point_t energy;
} stats_result_t;

// Function declarations

/**
 * Initialize feature extraction workspace
 * @param workspace Pointer to workspace structure
 * @return 0 on success, -1 on failure
 */
int feature_extractor_init(feature_workspace_t *workspace);

/**
 * Extract all 30 features from multi-sensor buffer
 * @param msb Multi-sensor buffer with 60s of data
 * @param workspace Computation workspace
 * @param result Output feature vector
 * @return 0 on success, negative on error
 */
int extract_features(multi_sensor_buffer_t *msb, 
                    feature_workspace_t *workspace,
                    feature_vector_t *result);

/**
 * Compute comprehensive statistics for sensor data
 * @param data Input data array (fixed-point)
 * @param count Number of samples
 * @param workspace Temporary workspace
 * @param result Output statistics
 * @return 0 on success, negative on error
 */
int compute_statistics(const fixed_point_t *data, 
                      uint16_t count,
                      fixed_point_t *workspace,
                      stats_result_t *result);

/**
 * Extract BVP-specific features (8 features)
 * @param msb Multi-sensor buffer
 * @param workspace Computation workspace
 * @param features Output feature array (starting at index 0)
 * @return 0 on success, negative on error
 */
int extract_bvp_features(multi_sensor_buffer_t *msb,
                        feature_workspace_t *workspace,
                        fixed_point_t *features);

/**
 * Extract accelerometer features (15 features)
 * @param msb Multi-sensor buffer
 * @param workspace Computation workspace
 * @param features Output feature array (starting at index 8)
 * @return 0 on success, negative on error
 */
int extract_acc_features(multi_sensor_buffer_t *msb,
                        feature_workspace_t *workspace,
                        fixed_point_t *features);

/**
 * Extract EDA features (4 features)
 * @param msb Multi-sensor buffer
 * @param workspace Computation workspace
 * @param features Output feature array (starting at index 23)
 * @return 0 on success, negative on error
 */
int extract_eda_features(multi_sensor_buffer_t *msb,
                        feature_workspace_t *workspace,
                        fixed_point_t *features);

/**
 * Extract temperature features (3 features)
 * @param msb Multi-sensor buffer
 * @param workspace Computation workspace
 * @param features Output feature array (starting at index 27)
 * @return 0 on success, negative on error
 */
int extract_temp_features(multi_sensor_buffer_t *msb,
                         feature_workspace_t *workspace,
                         fixed_point_t *features);

// Utility functions for efficient computation

/**
 * In-place quickselect for median/percentile computation
 * @param data Data array (will be modified)
 * @param count Number of elements
 * @param k Target index for selection
 * @return Selected element value
 */
fixed_point_t quickselect(fixed_point_t *data, uint16_t count, uint16_t k);

/**
 * Compute mean using fixed-point arithmetic
 * @param data Input data array
 * @param count Number of samples
 * @return Mean value as fixed-point
 */
fixed_point_t compute_mean(const fixed_point_t *data, uint16_t count);

/**
 * Compute standard deviation using fixed-point arithmetic
 * @param data Input data array
 * @param count Number of samples
 * @param mean Pre-computed mean value
 * @return Standard deviation as fixed-point
 */
fixed_point_t compute_std(const fixed_point_t *data, uint16_t count, fixed_point_t mean);

/**
 * Compute energy (sum of squares) using fixed-point arithmetic
 * @param data Input data array
 * @param count Number of samples
 * @return Energy value as fixed-point
 */
fixed_point_t compute_energy(const fixed_point_t *data, uint16_t count);

/**
 * Find minimum and maximum values in array
 * @param data Input data array
 * @param count Number of samples
 * @param min_val Output minimum value
 * @param max_val Output maximum value
 */
void find_min_max(const fixed_point_t *data, uint16_t count, 
                  fixed_point_t *min_val, fixed_point_t *max_val);

/**
 * Print feature vector for debugging
 * @param features Feature vector to print
 */
void print_feature_vector(const feature_vector_t *features);

/**
 * Get memory usage of feature extraction system
 * @return Total memory usage in bytes
 */
uint32_t feature_extractor_get_memory_usage(void);

// Fixed-point math utilities (optimized for ESP32)

/**
 * Fixed-point square root using Newton's method
 * @param x Input value (fixed-point)
 * @return Square root (fixed-point)
 */
fixed_point_t fixed_sqrt(fixed_point_t x);

/**
 * Fixed-point division with rounding
 * @param dividend Numerator (fixed-point)
 * @param divisor Denominator (fixed-point)
 * @return Result (fixed-point)
 */
fixed_point_t fixed_div(fixed_point_t dividend, fixed_point_t divisor);

#endif // FEATURE_EXTRACTOR_H
