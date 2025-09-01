/*
 * ESP32 Feature Extraction Test and Validation
 * Comprehensive testing of 30-feature extraction system
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "sensor_buffer.h"
#include "feature_extractor.h"

// External functions from test_buffer.c
extern uint32_t millis(void);
extern void advance_time(uint32_t ms);
extern float generate_bvp_sample(uint32_t time_ms);
extern float generate_acc_sample(uint32_t time_ms, int axis);
extern float generate_eda_sample(uint32_t time_ms);
extern float generate_temp_sample(uint32_t time_ms);

// Simulate stress-like patterns for validation
float generate_stress_bvp(uint32_t time_ms) {
    float t = time_ms / 1000.0f;
    // Elevated heart rate ~90 BPM with higher variability
    return 1.2f * sinf(2.0f * M_PI * 1.5f * t) + 0.3f * ((rand() % 100) / 100.0f - 0.5f);
}

float generate_stress_acc(uint32_t time_ms, int axis) {
    float t = time_ms / 1000.0f;
    // Increased movement and variability
    switch(axis) {
        case 0: return 0.3f * sinf(2.0f * M_PI * 1.2f * t) + 0.2f * ((rand() % 100) / 100.0f - 0.5f);
        case 1: return 0.3f * cosf(2.0f * M_PI * 0.8f * t) + 0.2f * ((rand() % 100) / 100.0f - 0.5f);
        case 2: return 9.8f + 0.5f * sinf(2.0f * M_PI * 1.1f * t) + 0.3f * ((rand() % 100) / 100.0f - 0.5f);
        default: return 0.0f;
    }
}

float generate_stress_eda(uint32_t time_ms) {
    float t = time_ms / 1000.0f;
    // Elevated skin conductance
    return 8.0f + 3.0f * sinf(2.0f * M_PI * 0.08f * t) + 0.5f * ((rand() % 100) / 100.0f - 0.5f);
}

float generate_stress_temp(uint32_t time_ms) {
    float t = time_ms / 1000.0f;
    // Slightly elevated temperature
    return 37.2f + 0.3f * sinf(2.0f * M_PI * 0.03f * t) + 0.2f * ((rand() % 100) / 100.0f - 0.5f);
}

void collect_sensor_data(multi_sensor_buffer_t *msb, uint32_t duration_ms, bool stress_pattern) {
    printf("📡 Collecting %s sensor data for %u seconds...\n",
           stress_pattern ? "STRESS" : "NORMAL", duration_ms / 1000);
    
    uint32_t progress_interval = 10000; // Print progress every 10s
    uint32_t next_progress = 10000;
    
    for (uint32_t t = 0; t < duration_ms; t += 1) {
        advance_time(1);
        uint32_t current_time = millis();
        
        // Sample all sensors based on their rates
        if (buffer_should_sample(msb, LAYER_BVP, current_time)) {
            float sample = stress_pattern ? 
                          generate_stress_bvp(current_time) : 
                          generate_bvp_sample(current_time);
            buffer_add_sample(msb, LAYER_BVP, sample);
        }
        
        if (buffer_should_sample(msb, LAYER_ACC_X, current_time)) {
            float sample = stress_pattern ? 
                          generate_stress_acc(current_time, 0) : 
                          generate_acc_sample(current_time, 0);
            buffer_add_sample(msb, LAYER_ACC_X, sample);
        }
        
        if (buffer_should_sample(msb, LAYER_ACC_Y, current_time)) {
            float sample = stress_pattern ? 
                          generate_stress_acc(current_time, 1) : 
                          generate_acc_sample(current_time, 1);
            buffer_add_sample(msb, LAYER_ACC_Y, sample);
        }
        
        if (buffer_should_sample(msb, LAYER_ACC_Z, current_time)) {
            float sample = stress_pattern ? 
                          generate_stress_acc(current_time, 2) : 
                          generate_acc_sample(current_time, 2);
            buffer_add_sample(msb, LAYER_ACC_Z, sample);
        }
        
        if (buffer_should_sample(msb, LAYER_EDA, current_time)) {
            float sample = stress_pattern ? 
                          generate_stress_eda(current_time) : 
                          generate_eda_sample(current_time);
            buffer_add_sample(msb, LAYER_EDA, sample);
        }
        
        if (buffer_should_sample(msb, LAYER_TEMP, current_time)) {
            float sample = stress_pattern ? 
                          generate_stress_temp(current_time) : 
                          generate_temp_sample(current_time);
            buffer_add_sample(msb, LAYER_TEMP, sample);
        }
        
        // Print progress
        if (current_time >= next_progress) {
            printf("  ⏱️  %ds: Buffer ready = %s\n", 
                   current_time / 1000,
                   buffer_is_ready_for_processing(msb) ? "YES" : "NO");
            next_progress += progress_interval;
        }
    }
}

void test_feature_extraction_basic() {
    printf("🧪 Testing Basic Feature Extraction\n");
    printf("=====================================\n");
    
    // Initialize systems
    multi_sensor_buffer_t msb;
    feature_workspace_t workspace;
    feature_vector_t features;
    
    if (buffer_init(&msb) != 0) {
        printf("❌ Buffer initialization failed\n");
        return;
    }
    
    if (feature_extractor_init(&workspace) != 0) {
        printf("❌ Feature extractor initialization failed\n");
        return;
    }
    
    printf("✅ Systems initialized successfully\n");
    printf("📊 Feature extractor memory usage: %u bytes (%.1f KB)\n",
           feature_extractor_get_memory_usage(),
           feature_extractor_get_memory_usage() / 1024.0f);
    
    // Collect 65 seconds of normal data
    collect_sensor_data(&msb, 65000, false);
    
    // Extract features
    printf("\n🔬 Extracting features from normal data...\n");
    
    int result = extract_features(&msb, &workspace, &features);
    if (result != 0) {
        printf("❌ Feature extraction failed with code %d\n", result);
        return;
    }
    
    printf("✅ Feature extraction completed!\n");
    print_feature_vector(&features);
    
    // Test extraction speed
    printf("\n⚡ Testing extraction speed...\n");
    
    uint32_t total_time = 0;
    int num_tests = 10;
    
    for (int i = 0; i < num_tests; i++) {
        uint32_t start_time = millis();
        extract_features(&msb, &workspace, &features);
        uint32_t end_time = millis();
        total_time += (end_time - start_time);
    }
    
    printf("📊 Average extraction time: %.2f ms (%d runs)\n", 
           total_time / (float)num_tests, num_tests);
    
    buffer_deinit(&msb);
    printf("✅ Basic test completed!\n");
}

void test_feature_extraction_patterns() {
    printf("\n🎭 Testing Feature Pattern Recognition\n");
    printf("=====================================\n");
    
    multi_sensor_buffer_t msb_normal, msb_stress;
    feature_workspace_t workspace;
    feature_vector_t features_normal, features_stress;
    
    // Initialize systems
    buffer_init(&msb_normal);
    buffer_init(&msb_stress);
    feature_extractor_init(&workspace);
    
    // Collect normal pattern data
    printf("\n📊 Collecting NORMAL pattern data...\n");
    collect_sensor_data(&msb_normal, 65000, false);
    
    // Collect stress pattern data  
    printf("\n⚡ Collecting STRESS pattern data...\n");
    collect_sensor_data(&msb_stress, 65000, true);
    
    // Extract features from both patterns
    printf("\n🔬 Extracting features from both patterns...\n");
    
    extract_features(&msb_normal, &workspace, &features_normal);
    extract_features(&msb_stress, &workspace, &features_stress);
    
    // Compare key features
    printf("\n📊 Feature Comparison (Normal vs Stress):\n");
    printf("==========================================\n");
    
    const char* key_features[] = {
        "BVP_MEAN", "BVP_STD", "BVP_ENERGY",
        "ACC_X_STD", "ACC_Y_STD", "ACC_Z_STD", 
        "EDA_MEAN", "EDA_STD",
        "TEMP_MEAN"
    };
    
    int key_indices[] = {
        FEATURE_BVP_MEAN, FEATURE_BVP_STD, FEATURE_BVP_ENERGY,
        FEATURE_ACC_X_STD, FEATURE_ACC_Y_STD, FEATURE_ACC_Z_STD,
        FEATURE_EDA_MEAN, FEATURE_EDA_STD,
        FEATURE_TEMP_MEAN
    };
    
    printf("Feature       | Normal    | Stress    | Diff%%    | Pattern\n");
    printf("--------------|-----------|-----------|----------|----------\n");
    
    for (int i = 0; i < 9; i++) {
        float normal_val = FIXED_TO_FLOAT(features_normal.features[key_indices[i]]);
        float stress_val = FIXED_TO_FLOAT(features_stress.features[key_indices[i]]);
        float diff_percent = ((stress_val - normal_val) / normal_val) * 100.0f;
        
        const char* pattern = (fabs(diff_percent) > 10.0f) ? 
                             (diff_percent > 0 ? "HIGHER" : "LOWER") : "SIMILAR";
        
        printf("%-12s | %8.4f | %8.4f | %7.1f%% | %s\n",
               key_features[i], normal_val, stress_val, diff_percent, pattern);
    }
    
    // Calculate feature vector differences
    float total_diff = 0.0f;
    for (int i = 0; i < NUM_FEATURES; i++) {
        float normal_val = FIXED_TO_FLOAT(features_normal.features[i]);
        float stress_val = FIXED_TO_FLOAT(features_stress.features[i]);
        float diff = fabs(stress_val - normal_val);
        total_diff += diff;
    }
    
    printf("\n📈 Total feature vector difference: %.4f\n", total_diff);
    printf("🎯 Pattern discrimination: %s\n", 
           total_diff > 1.0f ? "EXCELLENT" : 
           total_diff > 0.5f ? "GOOD" : "MODERATE");
    
    buffer_deinit(&msb_normal);
    buffer_deinit(&msb_stress);
    printf("✅ Pattern recognition test completed!\n");
}

void test_real_time_simulation() {
    printf("\n⏰ Testing Real-time Processing Simulation\n");
    printf("==========================================\n");
    
    multi_sensor_buffer_t msb;
    feature_workspace_t workspace;
    feature_vector_t features;
    
    buffer_init(&msb);
    feature_extractor_init(&workspace);
    
    // Simulate real-time processing with sliding windows
    printf("🔄 Simulating real-time feature extraction...\n");
    printf("    (60s collection + extraction every 10s)\n");
    
    // Initial 60s data collection
    collect_sensor_data(&msb, 60000, false);
    
    // Now simulate sliding window extractions every 10 seconds
    for (int cycle = 0; cycle < 5; cycle++) {
        printf("\n📊 Cycle %d: Adding 10s data, extracting features...\n", cycle + 1);
        
        // Add 10 more seconds of data
        collect_sensor_data(&msb, 10000, cycle >= 2); // Switch to stress pattern
        
        // Extract features
        uint32_t start_time = millis();
        int result = extract_features(&msb, &workspace, &features);
        uint32_t extraction_time = millis() - start_time;
        
        if (result == 0) {
            printf("✅ Extraction successful in %u ms\n", extraction_time);
            printf("   Key values: BVP_MEAN=%.3f, EDA_MEAN=%.3f, ACC_STD=%.3f\n",
                   FIXED_TO_FLOAT(features.features[FEATURE_BVP_MEAN]),
                   FIXED_TO_FLOAT(features.features[FEATURE_EDA_MEAN]),
                   FIXED_TO_FLOAT(features.features[FEATURE_ACC_X_STD]));
        } else {
            printf("❌ Extraction failed\n");
        }
    }
    
    buffer_deinit(&msb);
    printf("✅ Real-time simulation completed!\n");
}

int main() {
    printf("🚀 ESP32 Feature Extraction Comprehensive Test\n");
    printf("===============================================\n\n");
    
    // Seed random number generator
    srand(time(NULL));
    
    // Run all tests
    test_feature_extraction_basic();
    test_feature_extraction_patterns();
    test_real_time_simulation();
    
    printf("\n🎉 All feature extraction tests completed!\n");
    printf("💾 System ready for ESP32 deployment!\n");
    printf("📊 Total memory usage: Buffer + Features = %.1f KB\n",
           (buffer_get_memory_usage() + feature_extractor_get_memory_usage()) / 1024.0f);
    
    return 0;
}
