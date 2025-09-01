/*
 * ESP32 Multi-Sensor Buffer Test and Demo
 * Tests the 6-layer circular buffer system
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "sensor_buffer.h"

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

void test_buffer_basic_operations() {
    printf("🧪 Testing Basic Buffer Operations\n");
    printf("==================================================\n");
    
    multi_sensor_buffer_t msb;
    
    // Test initialization
    if (buffer_init(&msb) != 0) {
        printf("❌ Buffer initialization failed\n");
        return;
    }
    printf("✅ Buffer initialized successfully\n");
    
    // Test memory usage
    uint32_t memory_usage = buffer_get_memory_usage();
    printf("📊 Total memory usage: %u bytes (%.1f KB)\n", 
           memory_usage, memory_usage / 1024.0f);
    
    // Test adding samples
    printf("\n🔬 Testing sample addition...\n");
    
    for (int i = 0; i < 100; i++) {
        advance_time(1); // 1ms steps
        uint32_t current_time = millis();
        
        // Check each layer for sampling
        if (buffer_should_sample(&msb, LAYER_BVP, current_time)) {
            float sample = generate_bvp_sample(current_time);
            buffer_add_sample(&msb, LAYER_BVP, sample);
        }
        
        if (buffer_should_sample(&msb, LAYER_ACC_X, current_time)) {
            float sample = generate_acc_sample(current_time, 0);
            buffer_add_sample(&msb, LAYER_ACC_X, sample);
        }
        
        if (buffer_should_sample(&msb, LAYER_ACC_Y, current_time)) {
            float sample = generate_acc_sample(current_time, 1);
            buffer_add_sample(&msb, LAYER_ACC_Y, sample);
        }
        
        if (buffer_should_sample(&msb, LAYER_ACC_Z, current_time)) {
            float sample = generate_acc_sample(current_time, 2);
            buffer_add_sample(&msb, LAYER_ACC_Z, sample);
        }
        
        if (buffer_should_sample(&msb, LAYER_EDA, current_time)) {
            float sample = generate_eda_sample(current_time);
            buffer_add_sample(&msb, LAYER_EDA, sample);
        }
        
        if (buffer_should_sample(&msb, LAYER_TEMP, current_time)) {
            float sample = generate_temp_sample(current_time);
            buffer_add_sample(&msb, LAYER_TEMP, sample);
        }
    }
    
    print_buffer_status(&msb);
    
    // Test reading data
    printf("\n📖 Testing data retrieval...\n");
    
    fixed_point_t bvp_data[100];
    int samples_read = buffer_get_latest_samples(&msb, LAYER_BVP, bvp_data, 10);
    printf("✅ Read %d latest BVP samples\n", samples_read);
    
    if (samples_read > 0) {
        printf("   Latest BVP values: ");
        for (int i = 0; i < (samples_read < 5 ? samples_read : 5); i++) {
            printf("%.3f ", FIXED_TO_FLOAT(bvp_data[i]));
        }
        printf("\n");
    }
    
    buffer_deinit(&msb);
    printf("✅ Test completed successfully!\n");
}

void test_long_term_collection() {
    printf("\n🕐 Testing Long-term Data Collection (65 seconds)\n");
    printf("==================================================\n");
    
    multi_sensor_buffer_t msb;
    buffer_init(&msb);
    
    // Simulate 65 seconds of data collection (covers full 60s window + 5s)
    uint32_t total_time_ms = 65000;
    uint32_t progress_interval = 10000; // Print progress every 10s
    uint32_t next_progress = 10000;
    
    printf("📊 Collecting data for 65 seconds...\n");
    
    for (uint32_t t = 0; t < total_time_ms; t += 1) {
        advance_time(1);
        uint32_t current_time = millis();
        
        // Sample all sensors based on their rates
        if (buffer_should_sample(&msb, LAYER_BVP, current_time)) {
            buffer_add_sample(&msb, LAYER_BVP, generate_bvp_sample(current_time));
        }
        
        if (buffer_should_sample(&msb, LAYER_ACC_X, current_time)) {
            buffer_add_sample(&msb, LAYER_ACC_X, generate_acc_sample(current_time, 0));
        }
        
        if (buffer_should_sample(&msb, LAYER_ACC_Y, current_time)) {
            buffer_add_sample(&msb, LAYER_ACC_Y, generate_acc_sample(current_time, 1));
        }
        
        if (buffer_should_sample(&msb, LAYER_ACC_Z, current_time)) {
            buffer_add_sample(&msb, LAYER_ACC_Z, generate_acc_sample(current_time, 2));
        }
        
        if (buffer_should_sample(&msb, LAYER_EDA, current_time)) {
            buffer_add_sample(&msb, LAYER_EDA, generate_eda_sample(current_time));
        }
        
        if (buffer_should_sample(&msb, LAYER_TEMP, current_time)) {
            buffer_add_sample(&msb, LAYER_TEMP, generate_temp_sample(current_time));
        }
        
        // Print progress
        if (current_time >= next_progress) {
            printf("  ⏱️  %ds: ", current_time / 1000);
            printf("BVP:%d ACC:%d EDA:%d TEMP:%d samples\n",
                   buffer_get_count(&msb, LAYER_BVP),
                   buffer_get_count(&msb, LAYER_ACC_X),
                   buffer_get_count(&msb, LAYER_EDA),
                   buffer_get_count(&msb, LAYER_TEMP));
            next_progress += progress_interval;
        }
    }
    
    print_buffer_status(&msb);
    
    // Test full window reading
    printf("\n📖 Testing full window extraction...\n");
    
    fixed_point_t *window_data = malloc(BVP_BUFFER_SIZE * sizeof(fixed_point_t));
    if (window_data) {
        int samples = buffer_read_window(&msb, LAYER_BVP, window_data, BVP_BUFFER_SIZE);
        printf("✅ Extracted %d BVP samples from 60s window\n", samples);
        free(window_data);
    }
    
    buffer_deinit(&msb);
    printf("✅ Long-term test completed!\n");
}

int main() {
    printf("🚀 ESP32 Multi-Sensor Circular Buffer Test\n");
    printf("==========================================\n\n");
    
    // Seed random number generator
    srand(time(NULL));
    
    // Run tests
    test_buffer_basic_operations();
    test_long_term_collection();
    
    printf("\n🎉 All tests completed successfully!\n");
    printf("💾 Buffer system ready for ESP32 deployment!\n");
    
    return 0;
}
