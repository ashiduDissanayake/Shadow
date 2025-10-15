/*
 * Signal Preprocessor Validation Test
 * Tests preprocessing against Python-generated test data
 */

#include <stdio.h>
#include "unity.h"
#include "signal_preprocessor.h"
#include "test_data.h"  // Generated C arrays from Python

static const char *TAG = "PreprocessorTest";

// Test ACC magnitude computation
TEST_CASE("Compute ACC magnitude", "[preprocessor]") {
    const int NUM_SAMPLES = 10;
    float acc_x[NUM_SAMPLES] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float acc_y[NUM_SAMPLES] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    float acc_z[NUM_SAMPLES] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    float output[NUM_SAMPLES];
    
    int ret = compute_acc_magnitude(acc_x, acc_y, acc_z, output, NUM_SAMPLES);
    
    TEST_ASSERT_EQUAL(0, ret);
    TEST_ASSERT_FLOAT_WITHIN(0.001, sqrtf(1*1 + 1*1 + 1*1), output[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, sqrtf(2*2 + 1*1 + 1*1), output[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001, sqrtf(10*10 + 1*1 + 1*1), output[9]);
}

// Test Z-score normalization
TEST_CASE("Z-score normalization", "[preprocessor]") {
    // Simple test: all values = 5.0 should normalize to 0.0 (mean=5, std=0)
    const int NUM_SAMPLES = 5;
    float signal[NUM_SAMPLES] = {5.0, 5.0, 5.0, 5.0, 5.0};
    
    int ret = normalize_signal_zscore(signal, NUM_SAMPLES);
    
    TEST_ASSERT_EQUAL(0, ret);
    for (int i = 0; i < NUM_SAMPLES; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.001, 0.0, signal[i]);
    }
}

// Test normalization with known values
TEST_CASE("Z-score normalization with known values", "[preprocessor]") {
    // Data: [1, 2, 3, 4, 5]
    // Mean = 3.0
    // Variance = 2.0
    // Std = sqrt(2.0) = 1.414
    // Normalized[0] = (1 - 3) / 1.414 = -1.414
    const int NUM_SAMPLES = 5;
    float signal[NUM_SAMPLES] = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    int ret = normalize_signal_zscore(signal, NUM_SAMPLES);
    
    TEST_ASSERT_EQUAL(0, ret);
    
    // After normalization, mean should be ~0 and std should be ~1
    float sum = 0.0f;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        sum += signal[i];
    }
    float mean = sum / NUM_SAMPLES;
    TEST_ASSERT_FLOAT_WITHIN(0.001, 0.0, mean);
}

// Test signal statistics
TEST_CASE("Compute signal statistics", "[preprocessor]") {
    const int NUM_SAMPLES = 5;
    float signal[NUM_SAMPLES] = {1.0, 2.0, 3.0, 4.0, 5.0};
    signal_stats_t stats;
    
    int ret = compute_signal_stats(signal, NUM_SAMPLES, &stats);
    
    TEST_ASSERT_EQUAL(0, ret);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 3.0, stats.mean);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 1.0, stats.min);
    TEST_ASSERT_FLOAT_WITHIN(0.001, 5.0, stats.max);
    // Std = sqrt(2.0) = 1.414
    TEST_ASSERT_FLOAT_WITHIN(0.01, 1.414, stats.std);
}

// Test with Python-generated test data (if available)
#ifdef TEST_ACC_SAMPLES
TEST_CASE("Validate against Python test data - ACC", "[preprocessor][validation]") {
    // Compute ACC magnitude from test data
    float acc_mag[TEST_ACC_SAMPLES];
    int ret = compute_acc_magnitude(test_acc_x, test_acc_y, test_acc_z, 
                                    acc_mag, TEST_ACC_SAMPLES);
    TEST_ASSERT_EQUAL(0, ret);
    
    // Normalize
    ret = normalize_signal_zscore(acc_mag, TEST_ACC_SAMPLES);
    TEST_ASSERT_EQUAL(0, ret);
    
    // Compare with expected (only first 240 samples)
    int num_compare = (TEST_ACC_SAMPLES > 240) ? 240 : TEST_ACC_SAMPLES;
    int errors = 0;
    float max_error = 0.0f;
    
    for (int i = 0; i < num_compare; i++) {
        float diff = fabsf(acc_mag[i] - expected_acc_normalized[i]);
        if (diff > max_error) max_error = diff;
        if (diff > 0.001) errors++;  // 0.1% tolerance
    }
    
    ESP_LOGI(TAG, "ACC Validation: %d errors, max error: %.6f", errors, max_error);
    TEST_ASSERT_LESS_THAN(10, errors);  // Allow up to 10 samples with >0.1% error
    TEST_ASSERT_LESS_THAN(0.01, max_error);  // Max 1% error
}

TEST_CASE("Validate against Python test data - BVP", "[preprocessor][validation]") {
    float bvp[TEST_BVP_SAMPLES];
    memcpy(bvp, test_bvp, TEST_BVP_SAMPLES * sizeof(float));
    
    int ret = normalize_signal_zscore(bvp, TEST_BVP_SAMPLES);
    TEST_ASSERT_EQUAL(0, ret);
    
    int num_compare = (TEST_BVP_SAMPLES > 240) ? 240 : TEST_BVP_SAMPLES;
    int errors = 0;
    float max_error = 0.0f;
    
    for (int i = 0; i < num_compare; i++) {
        float diff = fabsf(bvp[i] - expected_bvp_normalized[i]);
        if (diff > max_error) max_error = diff;
        if (diff > 0.001) errors++;
    }
    
    ESP_LOGI(TAG, "BVP Validation: %d errors, max error: %.6f", errors, max_error);
    TEST_ASSERT_LESS_THAN(10, errors);
    TEST_ASSERT_LESS_THAN(0.01, max_error);
}

TEST_CASE("Validate against Python test data - EDA", "[preprocessor][validation]") {
    float eda[TEST_EDA_SAMPLES];
    memcpy(eda, test_eda, TEST_EDA_SAMPLES * sizeof(float));
    
    int ret = normalize_signal_zscore(eda, TEST_EDA_SAMPLES);
    TEST_ASSERT_EQUAL(0, ret);
    
    int num_compare = (TEST_EDA_SAMPLES > 240) ? 240 : TEST_EDA_SAMPLES;
    int errors = 0;
    float max_error = 0.0f;
    
    for (int i = 0; i < num_compare; i++) {
        float diff = fabsf(eda[i] - expected_eda_normalized[i]);
        if (diff > max_error) max_error = diff;
        if (diff > 0.001) errors++;
    }
    
    ESP_LOGI(TAG, "EDA Validation: %d errors, max error: %.6f", errors, max_error);
    TEST_ASSERT_LESS_THAN(10, errors);
    TEST_ASSERT_LESS_THAN(0.01, max_error);
}

TEST_CASE("Validate against Python test data - TEMP", "[preprocessor][validation]") {
    float temp[TEST_TEMP_SAMPLES];
    memcpy(temp, test_temp, TEST_TEMP_SAMPLES * sizeof(float));
    
    int ret = normalize_signal_zscore(temp, TEST_TEMP_SAMPLES);
    TEST_ASSERT_EQUAL(0, ret);
    
    int num_compare = (TEST_TEMP_SAMPLES > 240) ? 240 : TEST_TEMP_SAMPLES;
    int errors = 0;
    float max_error = 0.0f;
    
    for (int i = 0; i < num_compare; i++) {
        float diff = fabsf(temp[i] - expected_temp_normalized[i]);
        if (diff > max_error) max_error = diff;
        if (diff > 0.001) errors++;
    }
    
    ESP_LOGI(TAG, "TEMP Validation: %d errors, max error: %.6f", errors, max_error);
    TEST_ASSERT_LESS_THAN(10, errors);
    TEST_ASSERT_LESS_THAN(0.01, max_error);
}
#endif // TEST_ACC_SAMPLES

// Test memory usage calculation
TEST_CASE("Memory usage", "[preprocessor]") {
    uint32_t memory = signal_preprocessor_get_memory_usage();
    ESP_LOGI(TAG, "Signal preprocessor memory usage: %lu bytes", memory);
    
    // Should be reasonable (less than 50KB)
    TEST_ASSERT_LESS_THAN(50000, memory);
}
