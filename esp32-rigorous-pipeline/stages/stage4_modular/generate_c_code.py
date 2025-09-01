#!/usr/bin/env python3
"""
Generate the actual C code with real model weights
"""

import json
import numpy as np

def generate_c_arrays():
    """Generate C arrays from model_data.json"""
    
    # Load the extracted model data
    with open("model_data.json", "r") as f:
        data = json.load(f)
    
    # Generate feature means array
    means_str = "const float feature_means[NUM_FEATURES] = {\n"
    for i, mean in enumerate(data['scaler_means']):
        means_str += f"    {mean:.8f}f"
        if i < len(data['scaler_means']) - 1:
            means_str += ","
        if (i + 1) % 4 == 0:  # 4 per line
            means_str += "\n"
        else:
            means_str += " "
    means_str += "\n};\n"
    
    # Generate feature scales array
    scales_str = "const float feature_scales[NUM_FEATURES] = {\n"
    for i, scale in enumerate(data['scaler_scales']):
        scales_str += f"    {scale:.8f}f"
        if i < len(data['scaler_scales']) - 1:
            scales_str += ","
        if (i + 1) % 4 == 0:
            scales_str += "\n"
        else:
            scales_str += " "
    scales_str += "\n};\n"
    
    # Generate weight arrays for each layer
    weights_str = ""
    biases_str = ""
    
    for layer_idx, (weights, biases) in enumerate(zip(data['weights'], data['biases'])):
        # Convert to numpy for easier indexing
        W = np.array(weights)
        b = np.array(biases)
        
        input_size, output_size = W.shape
        
        # Weight array
        weights_str += f"const float weights_layer{layer_idx}[{input_size}][{output_size}] = {{\n"
        for i in range(input_size):
            weights_str += "    {"
            for j in range(output_size):
                weights_str += f"{W[i,j]:.8f}f"
                if j < output_size - 1:
                    weights_str += ", "
            weights_str += "}"
            if i < input_size - 1:
                weights_str += ","
            weights_str += "\n"
        weights_str += "};\n\n"
        
        # Bias array
        biases_str += f"const float biases_layer{layer_idx}[{output_size}] = {{\n"
        for i, bias in enumerate(b):
            biases_str += f"    {bias:.8f}f"
            if i < len(b) - 1:
                biases_str += ","
            if (i + 1) % 8 == 0:  # 8 per line
                biases_str += "\n"
            else:
                biases_str += " "
        biases_str += "\n};\n\n"
    
    # Generate calibration arrays
    calib_x_str = "const float calibration_x[CALIBRATION_POINTS] = {\n"
    for i, x in enumerate(data['calibration_x']):
        calib_x_str += f"    {x:.8f}f"
        if i < len(data['calibration_x']) - 1:
            calib_x_str += ","
        if (i + 1) % 6 == 0:
            calib_x_str += "\n"
        else:
            calib_x_str += " "
    calib_x_str += "\n};\n"
    
    calib_y_str = "const float calibration_y[CALIBRATION_POINTS] = {\n"
    for i, y in enumerate(data['calibration_y']):
        calib_y_str += f"    {y:.8f}f"
        if i < len(data['calibration_y']) - 1:
            calib_y_str += ","
        if (i + 1) % 6 == 0:
            calib_y_str += "\n"
        else:
            calib_y_str += " "
    calib_y_str += "\n};\n"
    
    return {
        'means': means_str,
        'scales': scales_str,
        'weights': weights_str,
        'biases': biases_str,
        'calib_x': calib_x_str,
        'calib_y': calib_y_str
    }

def generate_simple_mlp_c():
    """Generate the complete simple_mlp.c file"""
    
    # Read the template
    with open("components/simple_mlp_template.c", "r") as f:
        template = f.read()
    
    # Generate arrays
    arrays = generate_c_arrays()
    
    # Replace placeholders in template
    # Replace feature means
    template = template.replace(
        '''const float feature_means[NUM_FEATURES] = {
    // NOTE: Will be filled by code generator
    0.0f  // Placeholder - actual values from model_data.json
};''', arrays['means'])
    
    # Replace feature scales
    template = template.replace(
        '''const float feature_scales[NUM_FEATURES] = {
    // NOTE: Will be filled by code generator  
    1.0f  // Placeholder - actual values from model_data.json
};''', arrays['scales'])
    
    # Replace weights and biases
    template = template.replace(
        '''// Layer 0: 30 -> 64 (exact sklearn coefs_[0] layout)
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
};''', arrays['weights'] + arrays['biases'])
    
    # Replace calibration arrays
    template = template.replace(
        '''// Calibration lookup table (exact from IsotonicRegression)
const float calibration_x[CALIBRATION_POINTS] = {
    // NOTE: Will be filled by code generator
    0.0f  // Placeholder - X knots in ascending order
};

const float calibration_y[CALIBRATION_POINTS] = {
    // NOTE: Will be filled by code generator
    0.0f  // Placeholder - corresponding Y values
};''', arrays['calib_x'] + "\n" + arrays['calib_y'])
    
    # Write the final C file
    with open("components/simple_mlp.c", "w") as f:
        f.write(template)
    
    print("Generated components/simple_mlp.c with actual model weights!")

if __name__ == "__main__":
    generate_simple_mlp_c()
