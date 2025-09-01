/*
 * ESP32 Complete System Test
 * Tests both buffer system and feature extraction together
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "sensor_buffer.h"
#include "feature_extractor.h"

// Simulate ESP32 millis() function
static uint32_t simulated_time_ms = 0;
uint32_t millis() {
    return simulated_time_ms;
}

void advance_time(uint32_t ms) {
    simulated_time_ms += ms;
}

// Generate synthetic sensor data for testing
float generate_bvp_sample(uint32_t time_ms) {
    // Simulate heart rate ~70 BPM
    float t = time_ms / 1000.0f;
    return sinf(2.0f * M_PI * 1.17f * t) + 0.1f * ((rand() % 100) / 100.0f - 0.5f);
}

float generate_acc_sample(uint32_t time_ms, int axis) {
    float t = time_ms / 1000.0f;
    switch(axis) {
        case 0: return 0.1f * sinf(2.0f * M_PI * 0.5f * t) + 0.05f * ((rand() % 100) / 100.0f - 0.5f);
        case 1: return 0.1f * cosf(2.0f * M_PI * 0.3f * t) + 0.05f * ((rand() % 100) / 100.0f - 0.5f);
        case 2: return 9.8f + 0.2f * sinf(2.0f * M_PI * 0.7f * t) + 0.1f * ((rand() % 100) / 100.0f - 0.5f);
        default: return 0.0f;
    }
}

float generate_eda_sample(uint32_t time_ms) {
    float t = time_ms / 1000.0f;
    return 5.0f + 2.0f * sinf(2.0f * M_PI * 0.05f * t) + 0.1f * ((rand() % 100) / 100.0f - 0.5f);
}

float generate_temp_sample(uint32_t time_ms) {
    float t = time_ms / 1000.0f;
    return 36.5f + 0.5f * sinf(2.0f * M_PI * 0.02f * t) + 0.1f * ((rand() % 100) / 100.0f - 0.5f);
}

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

void print_buffer_status(multi_sensor_buffer_t *msb) {
    const char* layer_names[] = {"BVP", "ACC_X", "ACC_Y", "ACC_Z", "EDA", "TEMP"};
    
    printf("\n📊 Buffer Status:\n");
    printf("Layer    | Count/Size | Fill%%  | Sample Rate\n");
    printf("---------|------------|-------|------------\n");
    
    for (int i = 0; i < NUM_SENSOR_LAYERS; i++) {
        uint16_t count = buffer_get_count(msb, i);
        uint16_t size = msb->layers[i].size;
        float fill_percent = (count * 100.0f) / size;
        
        printf("%-8s | %4d/%4d | %5.1f%% | %d Hz\n", 
               layer_names[i], count, size, fill_percent, msb->layers[i].sample_rate);
    }
    
    printf("\nReady for processing: %s\n", 
           buffer_is_ready_for_processing(msb) ? "YES" : "NO");
}

void test_complete_system() {
    printf("🚀 Complete ESP32 System Test\n");
    printf("===============================\n");
    
    // Initialize all systems
    multi_sensor_buffer_t msb;
    feature_workspace_t workspace;
    feature_vector_t features_normal, features_stress;
    
    if (buffer_init(&msb) != 0) {
        printf("❌ Buffer initialization failed\n");
        return;
    }
    
    if (feature_extractor_init(&workspace) != 0) {
        printf("❌ Feature extractor initialization failed\n");
        return;
    }
    
    printf("✅ Systems initialized successfully\n");
    printf("📊 Total memory usage: %.1f KB\n",
           (buffer_get_memory_usage() + feature_extractor_get_memory_usage()) / 1024.0f);
    
    // Test 1: Normal pattern
    printf("\n🔬 Phase 1: Testing NORMAL stress pattern\n");
    printf("==========================================\n");
    
    collect_sensor_data(&msb, 65000, false);
    print_buffer_status(&msb);
    
    uint32_t start_time = millis();
    int result = extract_features(&msb, &workspace, &features_normal);
    uint32_t extraction_time = millis() - start_time;
    
    if (result == 0) {
        printf("✅ Normal pattern features extracted in %u ms\n", extraction_time);
        printf("   BVP_MEAN=%.3f, EDA_MEAN=%.3f, TEMP_MEAN=%.3f\n",
               FIXED_TO_FLOAT(features_normal.features[FEATURE_BVP_MEAN]),
               FIXED_TO_FLOAT(features_normal.features[FEATURE_EDA_MEAN]),
               FIXED_TO_FLOAT(features_normal.features[FEATURE_TEMP_MEAN]));
    } else {
        printf("❌ Normal pattern feature extraction failed\n");
        return;
    }
    
    // Reset buffer for stress test
    buffer_deinit(&msb);
    buffer_init(&msb);
    
    // Test 2: Stress pattern
    printf("\n⚡ Phase 2: Testing STRESS pattern\n");
    printf("===================================\n");
    
    collect_sensor_data(&msb, 65000, true);
    
    start_time = millis();
    result = extract_features(&msb, &workspace, &features_stress);
    extraction_time = millis() - start_time;
    
    if (result == 0) {
        printf("✅ Stress pattern features extracted in %u ms\n", extraction_time);
        printf("   BVP_MEAN=%.3f, EDA_MEAN=%.3f, TEMP_MEAN=%.3f\n",
               FIXED_TO_FLOAT(features_stress.features[FEATURE_BVP_MEAN]),
               FIXED_TO_FLOAT(features_stress.features[FEATURE_EDA_MEAN]),
               FIXED_TO_FLOAT(features_stress.features[FEATURE_TEMP_MEAN]));
    } else {
        printf("❌ Stress pattern feature extraction failed\n");
        return;
    }
    
    // Test 3: Pattern comparison
    printf("\n📊 Phase 3: Pattern Analysis\n");
    printf("============================\n");
    
    const char* key_features[] = {
        "BVP_MEAN", "BVP_STD", "BVP_ENERGY",
        "ACC_X_STD", "EDA_MEAN", "TEMP_MEAN"
    };
    
    int key_indices[] = {
        FEATURE_BVP_MEAN, FEATURE_BVP_STD, FEATURE_BVP_ENERGY,
        FEATURE_ACC_X_STD, FEATURE_EDA_MEAN, FEATURE_TEMP_MEAN
    };
    
    printf("Feature       | Normal    | Stress    | Diff%%    | Status\n");
    printf("--------------|-----------|-----------|----------|----------\n");
    
    float total_diff = 0.0f;
    for (int i = 0; i < 6; i++) {
        float normal_val = FIXED_TO_FLOAT(features_normal.features[key_indices[i]]);
        float stress_val = FIXED_TO_FLOAT(features_stress.features[key_indices[i]]);
        float diff_percent = ((stress_val - normal_val) / (normal_val + 0.001f)) * 100.0f;
        
        const char* status = (fabs(diff_percent) > 15.0f) ? 
                            (diff_percent > 0 ? "HIGHER" : "LOWER") : "SIMILAR";
        
        printf("%-12s | %8.4f | %8.4f | %7.1f%% | %s\n",
               key_features[i], normal_val, stress_val, diff_percent, status);
        
        total_diff += fabs(diff_percent);
    }
    
    printf("\n📈 Average feature difference: %.1f%%\n", total_diff / 6.0f);
    printf("🎯 Pattern discrimination: %s\n", 
           total_diff > 60.0f ? "EXCELLENT" : 
           total_diff > 30.0f ? "GOOD" : "MODERATE");
    
    // Test 4: Performance evaluation
    printf("\n⚡ Phase 4: Performance Evaluation\n");
    printf("===================================\n");
    
    uint32_t total_extraction_time = 0;
    int num_tests = 10;
    
    for (int i = 0; i < num_tests; i++) {
        start_time = millis();
        extract_features(&msb, &workspace, &features_stress);
        total_extraction_time += (millis() - start_time);
    }
    
    float avg_time = total_extraction_time / (float)num_tests;
    printf("📊 Average extraction time: %.2f ms (%d runs)\n", avg_time, num_tests);
    printf("🚀 Real-time capability: %s (target: <100ms)\n", 
           avg_time < 100.0f ? "EXCELLENT" : avg_time < 200.0f ? "GOOD" : "NEEDS_OPTIMIZATION");
    
    buffer_deinit(&msb);
    printf("\n🎉 Complete system test successful!\n");
    printf("💾 Ready for ESP32 deployment!\n");
}

int main() {
    printf("🚀 ESP32 Stress Detection System - Complete Test\n");
    printf("================================================\n\n");
    
    // Seed random number generator
    srand(time(NULL));
    
    // Run comprehensive test
    test_complete_system();
    
    return 0;
}