/*
 * ESP32-S3 MLP Model for Stress Detection (ESP-IDF Version)
 * 
 * 30-feature → 64 → 32 → 1 neural network
 * 99.6% accuracy validated, ready for real-time inference
 */

#ifndef SIMPLE_MLP_H
#define SIMPLE_MLP_H

#include <stdint.h>
#include <math.h>
#include "esp_log.h"

// Model architecture constants
#define NUM_FEATURES 30
#define HIDDEN1_SIZE 64
#define HIDDEN2_SIZE 32
#define OUTPUT_SIZE 1
#define CALIBRATION_POINTS 68

// Decision threshold (CRITICAL: apply AFTER calibration)
#define DECISION_THRESHOLD 0.4095238f

// Model data arrays (defined in simple_mlp.c)
extern const float feature_means[NUM_FEATURES];
extern const float feature_scales[NUM_FEATURES];

extern const float weights_layer0[NUM_FEATURES][HIDDEN1_SIZE];
extern const float biases_layer0[HIDDEN1_SIZE];

extern const float weights_layer1[HIDDEN1_SIZE][HIDDEN2_SIZE];
extern const float biases_layer1[HIDDEN2_SIZE];

extern const float weights_layer2[HIDDEN2_SIZE][OUTPUT_SIZE];
extern const float biases_layer2[OUTPUT_SIZE];

extern const float calibration_x[CALIBRATION_POINTS];
extern const float calibration_y[CALIBRATION_POINTS];

extern const char* feature_names[NUM_FEATURES];

// Core inference functions
float shadow_mlp_predict_probability(const float features[NUM_FEATURES]);
int shadow_mlp_predict_class(const float features[NUM_FEATURES]);

// Component functions (for testing)
void standardize_features(float features[NUM_FEATURES]);
void dense_layer_relu(const float* input, int input_size, 
                     const float* weights, const float* biases,
                     float* output, int output_size);
void dense_layer_linear(const float* input, int input_size,
                       const float* weights, const float* biases, 
                       float* output, int output_size);
float sigmoid(float x);
float apply_calibration(float raw_prob);

#endif // SIMPLE_MLP_H
