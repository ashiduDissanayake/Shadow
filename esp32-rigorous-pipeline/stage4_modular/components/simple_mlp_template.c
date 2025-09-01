#include "simple_mlp.h"
#include <string.h>

// Feature names (exact order from training)
const char* feature_names[NUM_FEATURES] = {
    "bvp_BVP_perm_entropy",
    "acc_y_perm_entropy", 
    "acc_l2_ptp",
    "acc_l2_max",
    "acc_z_peaks",
    "eda_l2_lineintegral",
    "acc_l2_peaks",
    "acc_z_perm_entropy",
    "acc_y_lineintegral",
    "eda_EDA_lineintegral",
    "temp_TEMP_min",
    "temp_l2_min",
    "acc_z_rms",
    "acc_z_min",
    "acc_z_energy",
    "acc_z_pct_95",
    "acc_z_mean",
    "bvp_l2_iqr",
    "acc_l2_rms",
    "eda_l2_iqr_5_95",
    "acc_y_peaks",
    "bvp_BVP_n_sign_changes",
    "eda_EDA_iqr_5_95",
    "temp_TEMP_energy",
    "temp_l2_energy",
    "acc_l2_min",
    "temp_TEMP_sum",
    "bvp_l2_peaks",
    "eda_l2_min",
    "eda_EDA_max"
};

// Scaler parameters (exact from sklearn StandardScaler)
const float feature_means[NUM_FEATURES] = {
    // NOTE: Will be filled by code generator
    0.0f  // Placeholder - actual values from model_data.json
};

const float feature_scales[NUM_FEATURES] = {
    // NOTE: Will be filled by code generator  
    1.0f  // Placeholder - actual values from model_data.json
};

// Layer 0: 30 -> 64 (exact sklearn coefs_[0] layout)
const float weights_layer0[NUM_FEATURES][HIDDEN1_SIZE] = {
    // NOTE: Will be filled by code generator
    {0.0f}  // Placeholder - actual weights from model_data.json
};

const float biases_layer0[HIDDEN1_SIZE] = {
    // NOTE: Will be filled by code generator
    0.0f  // Placeholder - actual biases from model_data.json
};

// Layer 1: 64 -> 32 (exact sklearn coefs_[1] layout)
const float weights_layer1[HIDDEN1_SIZE][HIDDEN2_SIZE] = {
    // NOTE: Will be filled by code generator
    {0.0f}  // Placeholder
};

const float biases_layer1[HIDDEN2_SIZE] = {
    // NOTE: Will be filled by code generator
    0.0f  // Placeholder
};

// Layer 2: 32 -> 1 (exact sklearn coefs_[2] layout)
const float weights_layer2[HIDDEN2_SIZE][OUTPUT_SIZE] = {
    // NOTE: Will be filled by code generator
    {0.0f}  // Placeholder
};

const float biases_layer2[OUTPUT_SIZE] = {
    // NOTE: Will be filled by code generator
    0.0f  // Placeholder
};

// Calibration lookup table (exact from IsotonicRegression)
const float calibration_x[CALIBRATION_POINTS] = {
    // NOTE: Will be filled by code generator
    0.0f  // Placeholder - X knots in ascending order
};

const float calibration_y[CALIBRATION_POINTS] = {
    // NOTE: Will be filled by code generator
    0.0f  // Placeholder - corresponding Y values
};

// ============================================================================
// Core Functions
// ============================================================================

void standardize_features(float features[NUM_FEATURES]) {
    // REQUIREMENT A & B: Feature ordering and standardization
    // z_j = (x_j − μ_j)/σ_j with exact 1:1 correspondence
    for (int j = 0; j < NUM_FEATURES; j++) {
        features[j] = (features[j] - feature_means[j]) / feature_scales[j];
    }
}

void dense_layer_relu(const float* input, int input_size,
                     const float weights[][256], const float* biases,
                     float* output, int output_size) {
    // REQUIREMENT C: Weight layout exactly as sklearn coefs_[l][in_index, out_index]
    // Matrix multiplication: output = ReLU(weights^T * input + bias)
    for (int k = 0; k < output_size; k++) {
        float sum = biases[k];
        for (int j = 0; j < input_size; j++) {
            // CRITICAL: weights[j][k] NOT weights[k][j] (avoid transpose bug)
            sum += input[j] * weights[j][k];
        }
        // REQUIREMENT D: ReLU activation for hidden layers
        output[k] = (sum > 0.0f) ? sum : 0.0f;
    }
}

void dense_layer_linear(const float* input, int input_size,
                       const float weights[][256], const float* biases,
                       float* output, int output_size) {
    // Same as dense_layer_relu but NO activation (for output layer)
    for (int k = 0; k < output_size; k++) {
        float sum = biases[k];
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[j][k];
        }
        output[k] = sum;  // No ReLU
    }
}

float sigmoid(float x) {
    // REQUIREMENT E: Standard sigmoid, no clamping for strict replication
    // sigmoid(z) = 1 / (1 + exp(-z))
    // Note: Could clamp x to [-20, 20] for embedded safety if needed
    return 1.0f / (1.0f + expf(-x));
}

float apply_calibration(float raw_prob) {
    // REQUIREMENT F: Exact isotonic calibration with linear interpolation
    
    // Edge cases: clamp to boundaries
    if (raw_prob <= calibration_x[0]) {
        return calibration_y[0];
    }
    if (raw_prob >= calibration_x[CALIBRATION_POINTS - 1]) {
        return calibration_y[CALIBRATION_POINTS - 1];
    }
    
    // Binary search for interpolation interval
    int left = 0;
    int right = CALIBRATION_POINTS - 1;
    
    while (right - left > 1) {
        int mid = (left + right) / 2;
        if (calibration_x[mid] <= raw_prob) {
            left = mid;
        } else {
            right = mid;
        }
    }
    
    // Linear interpolation between calibration_x[left] and calibration_x[right]
    float x0 = calibration_x[left];
    float x1 = calibration_x[right];
    float y0 = calibration_y[left];
    float y1 = calibration_y[right];
    
    // Guard against identical x points (rare but possible)
    if (x1 == x0) {
        return y0;
    }
    
    float t = (raw_prob - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

// ============================================================================
// Main Inference Pipeline
// ============================================================================

float shadow_mlp_predict_probability(const float features[NUM_FEATURES]) {
    // Working buffers for layer outputs
    float standardized[NUM_FEATURES];
    float hidden1[HIDDEN1_SIZE];
    float hidden2[HIDDEN2_SIZE];
    float output[OUTPUT_SIZE];
    
    // Copy input features (non-destructive)
    memcpy(standardized, features, NUM_FEATURES * sizeof(float));
    
    // Step 1: Standardization z_j = (x_j − μ_j)/σ_j
    standardize_features(standardized);
    
    // Step 2: Hidden layer 1 with ReLU
    // h1_k = ReLU(Σ_j z_j * W1_{j,k} + b1_k)
    dense_layer_relu(standardized, NUM_FEATURES, 
                    weights_layer0, biases_layer0,
                    hidden1, HIDDEN1_SIZE);
    
    // Step 3: Hidden layer 2 with ReLU  
    // h2_m = ReLU(Σ_k h1_k * W2_{k,m} + b2_m)
    dense_layer_relu(hidden1, HIDDEN1_SIZE,
                    weights_layer1, biases_layer1, 
                    hidden2, HIDDEN2_SIZE);
    
    // Step 4: Output layer (linear, no activation)
    // ℓ = Σ_m h2_m * W3_m + b3
    dense_layer_linear(hidden2, HIDDEN2_SIZE,
                      weights_layer2, biases_layer2,
                      output, OUTPUT_SIZE);
    
    // Step 5: Sigmoid activation
    // p_raw = 1 / (1 + e^{−ℓ})
    float raw_prob = sigmoid(output[0]);
    
    // Step 6: Isotonic calibration
    // p_cal = ISO(p_raw)
    float calibrated_prob = apply_calibration(raw_prob);
    
    return calibrated_prob;
}

int shadow_mlp_predict_class(const float features[NUM_FEATURES]) {
    // REQUIREMENT G: Apply threshold AFTER calibration, not on raw probability
    float prob = shadow_mlp_predict_probability(features);
    return (prob >= DECISION_THRESHOLD) ? 1 : 0;
}
