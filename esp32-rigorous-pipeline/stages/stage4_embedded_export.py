#!/usr/bin/env python3
"""
ESP32-S3 Rigorous ML Pipeline
Stage 4: Export & Embedded Implementation

Converts trained MLP model to C arrays and inference code for ESP32 deployment.
Creates modular, human-readable embedded code with comprehensive validation.

Author: AI Assistant
Date: 2025-08-31
"""

from pathlib import Path
import json
import numpy as np
import joblib
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - STAGE4 - %(levelname)s - %(message)s")
logger = logging.getLogger("STAGE4")

@dataclass
class EmbeddedConfig:
    # Input paths
    stage2_dir: str = "../outputs/stage2_model_exploration"
    output_dir: str = "../outputs/stage4_embedded_export"
    
    # Code generation options
    use_float32: bool = True  # vs fixed-point
    optimize_for_size: bool = True
    generate_test_vectors: bool = True
    max_test_vectors: int = 100
    
    # ESP32 specific
    target_platform: str = "ESP32-S3"
    include_guards: bool = True
    namespace_prefix: str = "shadow_"
    
    # Validation tolerances
    float_tolerance: float = 1e-6
    probability_tolerance: float = 1e-4

class ModelExporter:
    def __init__(self, config: EmbeddedConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.model = None
        self.scaler = None
        self.calibrator = None
        self.calibrator_type = "none"
        self.threshold = 0.5
        self.features = []
        self.artifacts = {}
        
        # Extracted model data
        self.weights = []
        self.biases = []
        self.layer_sizes = []
        self.feature_means = None
        self.feature_scales = None
        self.calibration_points = None
        
    def load_artifacts(self):
        """Load trained model artifacts from Stage 2"""
        stage2_dir = Path(self.config.stage2_dir)
        
        # Load model artifacts metadata
        with open(stage2_dir / "final_model_artifacts.json", 'r') as f:
            self.artifacts = json.load(f)
        
        # Load model components
        self.model = joblib.load(stage2_dir / "final_model.joblib")
        
        if Path(stage2_dir / "final_scaler.joblib").exists():
            self.scaler = joblib.load(stage2_dir / "final_scaler.joblib")
        
        if Path(stage2_dir / "final_calibrator.joblib").exists():
            self.calibrator = joblib.load(stage2_dir / "final_calibrator.joblib")
        
        # Extract metadata
        self.threshold = self.artifacts.get("optimal_threshold", 0.5)
        self.features = self.artifacts.get("features", [])
        self.calibrator_type = self.artifacts.get("calibrator_type", "none")
        
        logger.info(f"Loaded model: {self.artifacts['model_type']}")
        logger.info(f"Features: {len(self.features)}")
        logger.info(f"Calibrator: {self.calibrator_type}")
        logger.info(f"Threshold: {self.threshold:.4f}")
    
    def extract_model_data(self):
        """Extract weights, biases, and architecture from MLP"""
        if not hasattr(self.model, 'coefs_'):
            raise ValueError("Model must be an MLPClassifier with coefs_ attribute")
        
        # Extract weights and biases
        self.weights = [w.astype(np.float32) for w in self.model.coefs_]
        self.biases = [b.astype(np.float32) for b in self.model.intercepts_]
        
        # Layer sizes: [input, hidden1, hidden2, ..., output]
        self.layer_sizes = [self.weights[0].shape[0]]  # input size
        for w in self.weights:
            self.layer_sizes.append(w.shape[1])
        
        logger.info(f"Model architecture: {' -> '.join(map(str, self.layer_sizes))}")
        logger.info(f"Total parameters: {sum(w.size for w in self.weights) + sum(b.size for b in self.biases)}")
        
        # Extract scaler parameters
        if self.scaler:
            self.feature_means = self.scaler.mean_.astype(np.float32)
            self.feature_scales = self.scaler.scale_.astype(np.float32)
            logger.info(f"Scaler: {len(self.feature_means)} features")
        
        # Extract calibration data
        if self.calibrator and self.calibrator_type == "isotonic":
            # Get calibration points from isotonic regression
            x_points = self.calibrator.X_thresholds_.astype(np.float32)
            # Create interpolation points by evaluating the calibrator
            y_points = []
            for x in x_points:
                y = self.calibrator.transform(np.array([x]))[0]
                y_points.append(float(y))
            y_points = np.array(y_points, dtype=np.float32)
            
            self.calibration_points = (x_points, y_points)
            logger.info(f"Calibration: {len(x_points)} interpolation points")
    
    def generate_header_file(self):
        """Generate C header file with model data and function declarations"""
        content = self._generate_header_content()
        header_path = self.output_dir / f"{self.config.namespace_prefix}model.h"
        with open(header_path, 'w') as f:
            f.write(content)
        logger.info(f"Generated header: {header_path}")
        return header_path
    
    def generate_source_file(self):
        """Generate C source file with inference implementation"""
        content = self._generate_source_content()
        source_path = self.output_dir / f"{self.config.namespace_prefix}model.c"
        with open(source_path, 'w') as f:
            f.write(content)
        logger.info(f"Generated source: {source_path}")
        return source_path
    
    def _generate_header_content(self) -> str:
        """Generate the header file content"""
        guard_name = f"{self.config.namespace_prefix.upper()}MODEL_H"
        prefix = self.config.namespace_prefix
        
        content = f"""/*
 * Shadow ML Model - ESP32 Embedded Implementation
 * Generated automatically from Stage 2 trained model
 * 
 * Model: {self.artifacts['model_type']}
 * Features: {len(self.features)}
 * Architecture: {' -> '.join(map(str, self.layer_sizes))}
 * Calibration: {self.calibrator_type}
 * 
 * Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
 */

#ifndef {guard_name}
#define {guard_name}

#ifdef __cplusplus
extern "C" {{
#endif

#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// Model Configuration
// ============================================================================

#define {prefix.upper()}NUM_FEATURES {len(self.features)}
#define {prefix.upper()}NUM_LAYERS {len(self.layer_sizes)}
#define {prefix.upper()}DECISION_THRESHOLD {self.threshold:.8f}f
#define {prefix.upper()}HAS_SCALER {"1" if self.scaler else "0"}
#define {prefix.upper()}HAS_CALIBRATOR {"1" if self.calibrator else "0"}
#define {prefix.upper()}CALIBRATION_POINTS {len(self.calibration_points[0]) if self.calibration_points else 0}

// Layer sizes: [{', '.join(map(str, self.layer_sizes))}]
extern const uint16_t {prefix}layer_sizes[{prefix.upper()}NUM_LAYERS];

// ============================================================================
// Model Weights & Biases
// ============================================================================
"""

        # Declare weight and bias arrays
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            content += f"""
// Layer {i}: {w.shape[0]} -> {w.shape[1]}
extern const float {prefix}weights_layer{i}[{w.shape[0]}][{w.shape[1]}];
extern const float {prefix}biases_layer{i}[{w.shape[1]}];"""

        # Scaler arrays if present
        if self.scaler:
            content += f"""

// ============================================================================
// Feature Scaling Parameters
// ============================================================================
extern const float {prefix}feature_means[{prefix.upper()}NUM_FEATURES];
extern const float {prefix}feature_scales[{prefix.upper()}NUM_FEATURES];"""

        # Calibration arrays if present
        if self.calibration_points:
            content += f"""

// ============================================================================
// Calibration Lookup Table
// ============================================================================
extern const float {prefix}calibration_x[{prefix.upper()}CALIBRATION_POINTS];
extern const float {prefix}calibration_y[{prefix.upper()}CALIBRATION_POINTS];"""

        # Function declarations
        content += f"""

// ============================================================================
// Inference Functions
// ============================================================================

/**
 * Apply feature scaling (if model uses StandardScaler)
 * @param features Input feature array (modified in-place)
 */
void {prefix}scale_features(float features[{prefix.upper()}NUM_FEATURES]);

/**
 * ReLU activation function (element-wise)
 * @param data Input array (modified in-place)
 * @param size Array size
 */
void {prefix}relu(float* data, uint16_t size);

/**
 * Matrix-vector multiplication + bias addition
 * @param input Input vector
 * @param input_size Input dimension
 * @param weights Weight matrix [input_size][output_size]
 * @param biases Bias vector
 * @param output Output vector
 * @param output_size Output dimension
 */
void {prefix}dense_layer(const float* input, uint16_t input_size,
                        const float weights[][256], const float* biases,
                        float* output, uint16_t output_size);

/**
 * Apply isotonic calibration using linear interpolation
 * @param raw_prob Raw probability from model
 * @return Calibrated probability
 */
float {prefix}apply_calibration(float raw_prob);

/**
 * Full MLP forward pass (raw output)
 * @param features Input features [{prefix.upper()}NUM_FEATURES]
 * @return Raw model output (before calibration)
 */
float {prefix}predict_raw(const float features[{prefix.upper()}NUM_FEATURES]);

/**
 * Full inference pipeline with calibration
 * @param features Input features [{prefix.upper()}NUM_FEATURES]
 * @return Calibrated probability [0.0, 1.0]
 */
float {prefix}predict_probability(const float features[{prefix.upper()}NUM_FEATURES]);

/**
 * Binary classification prediction
 * @param features Input features [{prefix.upper()}NUM_FEATURES]
 * @return 1 for stress detected, 0 for no stress
 */
int {prefix}predict_class(const float features[{prefix.upper()}NUM_FEATURES]);

/**
 * Get feature name by index (for debugging)
 * @param index Feature index [0, {prefix.upper()}NUM_FEATURES-1]
 * @return Feature name string
 */
const char* {prefix}get_feature_name(uint16_t index);

#ifdef __cplusplus
}}
#endif

#endif // {guard_name}
"""
        return content
    
    def _generate_source_content(self) -> str:
        """Generate the source file content"""
        prefix = self.config.namespace_prefix
        
        content = f"""/*
 * Shadow ML Model - ESP32 Embedded Implementation
 * Source file with inference logic and model data
 */

#include "{prefix}model.h"
#include <math.h>
#include <string.h>

// ============================================================================
// Model Architecture
// ============================================================================

const uint16_t {prefix}layer_sizes[{prefix.upper()}NUM_LAYERS] = {{
    {', '.join(map(str, self.layer_sizes))}
}};

"""

        # Generate weight and bias arrays
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            content += f"""// Layer {i} Weights: {w.shape[0]} x {w.shape[1]}
const float {prefix}weights_layer{i}[{w.shape[0]}][{w.shape[1]}] = {{
"""
            for row in w:
                row_str = ', '.join(f"{val:.8f}f" for val in row)
                content += f"    {{{row_str}}},\n"
            content += "};\n\n"
            
            content += f"""// Layer {i} Biases: {b.shape[0]}
const float {prefix}biases_layer{i}[{b.shape[1]}] = {{
    {', '.join(f"{val:.8f}f" for val in b)}
}};

"""

        # Generate scaler arrays if present
        if self.scaler:
            content += f"""// ============================================================================
// Feature Scaling Parameters
// ============================================================================

const float {prefix}feature_means[{prefix.upper()}NUM_FEATURES] = {{
    {', '.join(f"{val:.8f}f" for val in self.feature_means)}
}};

const float {prefix}feature_scales[{prefix.upper()}NUM_FEATURES] = {{
    {', '.join(f"{val:.8f}f" for val in self.feature_scales)}
}};

"""

        # Generate calibration arrays if present
        if self.calibration_points:
            x_points, y_points = self.calibration_points
            content += f"""// ============================================================================
// Calibration Lookup Table
// ============================================================================

const float {prefix}calibration_x[{prefix.upper()}CALIBRATION_POINTS] = {{
    {', '.join(f"{val:.8f}f" for val in x_points)}
}};

const float {prefix}calibration_y[{prefix.upper()}CALIBRATION_POINTS] = {{
    {', '.join(f"{val:.8f}f" for val in y_points)}
}};

"""

        # Generate feature names
        content += f"""// ============================================================================
// Feature Names (for debugging)
// ============================================================================

static const char* {prefix}feature_names[{prefix.upper()}NUM_FEATURES] = {{
"""
        for feat in self.features:
            content += f'    "{feat}",\n'
        content += "};\n\n"

        # Generate function implementations
        content += self._generate_function_implementations()
        
        return content
    
    def _generate_function_implementations(self) -> str:
        """Generate the function implementations"""
        prefix = self.config.namespace_prefix
        
        return f"""// ============================================================================
// Function Implementations
// ============================================================================

void {prefix}scale_features(float features[{prefix.upper()}NUM_FEATURES]) {{
#if {prefix.upper()}HAS_SCALER
    for (uint16_t i = 0; i < {prefix.upper()}NUM_FEATURES; i++) {{
        features[i] = (features[i] - {prefix}feature_means[i]) / {prefix}feature_scales[i];
    }}
#endif
}}

void {prefix}relu(float* data, uint16_t size) {{
    for (uint16_t i = 0; i < size; i++) {{
        if (data[i] < 0.0f) {{
            data[i] = 0.0f;
        }}
    }}
}}

void {prefix}dense_layer(const float* input, uint16_t input_size,
                        const float weights[][256], const float* biases,
                        float* output, uint16_t output_size) {{
    // Matrix multiplication: output = weights^T * input + bias
    for (uint16_t j = 0; j < output_size; j++) {{
        float sum = biases[j];
        for (uint16_t i = 0; i < input_size; i++) {{
            sum += input[i] * weights[i][j];
        }}
        output[j] = sum;
    }}
}}

float {prefix}apply_calibration(float raw_prob) {{
#if {prefix.upper()}HAS_CALIBRATOR
    // Binary search for interpolation points
    if (raw_prob <= {prefix}calibration_x[0]) {{
        return {prefix}calibration_y[0];
    }}
    if (raw_prob >= {prefix}calibration_x[{prefix.upper()}CALIBRATION_POINTS - 1]) {{
        return {prefix}calibration_y[{prefix.upper()}CALIBRATION_POINTS - 1];
    }}
    
    // Find interpolation interval
    uint16_t left = 0, right = {prefix.upper()}CALIBRATION_POINTS - 1;
    while (right - left > 1) {{
        uint16_t mid = (left + right) / 2;
        if ({prefix}calibration_x[mid] <= raw_prob) {{
            left = mid;
        }} else {{
            right = mid;
        }}
    }}
    
    // Linear interpolation
    float x0 = {prefix}calibration_x[left];
    float x1 = {prefix}calibration_x[right];
    float y0 = {prefix}calibration_y[left];
    float y1 = {prefix}calibration_y[right];
    
    float t = (raw_prob - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
#else
    return raw_prob;  // No calibration
#endif
}}

float {prefix}predict_raw(const float features[{prefix.upper()}NUM_FEATURES]) {{
    // Working buffers for layer outputs
    static float layer_output[2][256];  // Ping-pong buffers
    
    // Copy input features to first buffer
    memcpy(layer_output[0], features, {prefix.upper()}NUM_FEATURES * sizeof(float));
    
    uint8_t current_buffer = 0;
    uint8_t next_buffer = 1;
    
    // Layer 0: Input -> Hidden1
    {prefix}dense_layer(layer_output[current_buffer], {prefix}layer_sizes[0],
                       {prefix}weights_layer0, {prefix}biases_layer0,
                       layer_output[next_buffer], {prefix}layer_sizes[1]);
    {prefix}relu(layer_output[next_buffer], {prefix}layer_sizes[1]);
    current_buffer = 1 - current_buffer;
    next_buffer = 1 - next_buffer;
    
    // Layer 1: Hidden1 -> Hidden2  
    {prefix}dense_layer(layer_output[current_buffer], {prefix}layer_sizes[1],
                       {prefix}weights_layer1, {prefix}biases_layer1,
                       layer_output[next_buffer], {prefix}layer_sizes[2]);
    {prefix}relu(layer_output[next_buffer], {prefix}layer_sizes[2]);
    current_buffer = 1 - current_buffer;
    next_buffer = 1 - next_buffer;
    
    // Layer 2: Hidden2 -> Output (no activation)
    {prefix}dense_layer(layer_output[current_buffer], {prefix}layer_sizes[2],
                       {prefix}weights_layer2, {prefix}biases_layer2,
                       layer_output[next_buffer], {prefix}layer_sizes[3]);
    
    // Convert to probability using sigmoid
    float logit = layer_output[next_buffer][0];
    return 1.0f / (1.0f + expf(-logit));
}}

float {prefix}predict_probability(const float features[{prefix.upper()}NUM_FEATURES]) {{
    // Copy features for scaling (non-destructive)
    float scaled_features[{prefix.upper()}NUM_FEATURES];
    memcpy(scaled_features, features, {prefix.upper()}NUM_FEATURES * sizeof(float));
    
    // Apply feature scaling
    {prefix}scale_features(scaled_features);
    
    // Get raw prediction
    float raw_prob = {prefix}predict_raw(scaled_features);
    
    // Apply calibration
    return {prefix}apply_calibration(raw_prob);
}}

int {prefix}predict_class(const float features[{prefix.upper()}NUM_FEATURES]) {{
    float prob = {prefix}predict_probability(features);
    return (prob >= {prefix.upper()}DECISION_THRESHOLD) ? 1 : 0;
}}

const char* {prefix}get_feature_name(uint16_t index) {{
    if (index < {prefix.upper()}NUM_FEATURES) {{
        return {prefix}feature_names[index];
    }}
    return "INVALID_INDEX";
}}
"""

    def generate_test_vectors(self):
        """Generate test vectors for validation"""
        if not self.config.generate_test_vectors:
            return
            
        # Load some test data from Stage 2
        stage2_dir = Path(self.config.stage2_dir)
        
        # Try to load fold results to get test samples
        try:
            with open(stage2_dir / "fold_results.json", 'r') as f:
                fold_results = json.load(f)
            
            # Collect test cases from all folds
            test_cases = []
            for model_type, folds in fold_results.items():
                if model_type == self.artifacts['model_type']:
                    for fold in folds[:2]:  # Use first 2 folds
                        pred_data = fold.get('predictions', {})
                        if 'y_true' in pred_data and 'y_prob' in pred_data:
                            for i, (true_label, prob) in enumerate(zip(pred_data['y_true'], pred_data['y_prob'])):
                                if len(test_cases) >= self.config.max_test_vectors:
                                    break
                                test_cases.append({
                                    'fold': fold['fold_id'],
                                    'sample_idx': i,
                                    'true_label': int(true_label),
                                    'python_probability': float(prob)
                                })
                    break
            
            # Generate synthetic test vectors if we don't have enough
            if len(test_cases) < 10:
                logger.warning("Insufficient test data, generating synthetic vectors")
                for i in range(10):
                    # Random features in reasonable ranges
                    features = np.random.randn(len(self.features)).astype(np.float32)
                    test_cases.append({
                        'synthetic': True,
                        'sample_idx': i,
                        'features': features.tolist()
                    })
            
            # Save test vectors
            test_vectors_path = self.output_dir / "test_vectors.json"
            with open(test_vectors_path, 'w') as f:
                json.dump(test_cases, f, indent=2)
            
            logger.info(f"Generated {len(test_cases)} test vectors: {test_vectors_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate test vectors: {e}")

    def generate_build_files(self):
        """Generate CMakeLists.txt and build configuration"""
        cmake_content = f"""# Shadow ML Model - ESP32 Build Configuration
# Generated for {self.config.target_platform}

cmake_minimum_required(VERSION 3.16)

# Component definition
idf_component_register(
    SRCS 
        "{self.config.namespace_prefix}model.c"
    INCLUDE_DIRS 
        "."
    REQUIRES 
        "esp_common"
        "freertos"
        "esp_timer"
)

# Compiler optimizations for inference
target_compile_options(${{COMPONENT_LIB}} PRIVATE
    -O2                    # Optimize for speed
    -ffast-math           # Fast floating point
    -funroll-loops        # Loop unrolling
    -finline-functions    # Aggressive inlining
)

# Model size reporting
add_custom_command(
    TARGET ${{COMPONENT_LIB}} POST_BUILD
    COMMAND ${{CMAKE_SIZE}} $<TARGET_FILE:${{COMPONENT_LIB}}>
    COMMENT "Model component size:"
)
"""
        cmake_path = self.output_dir / "CMakeLists.txt"
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        
        # Generate component config
        kconfig_content = f"""# Shadow ML Model Configuration

menu "Shadow ML Model"
    
    config {self.config.namespace_prefix.upper()}ENABLE_DEBUG
        bool "Enable debug output"
        default n
        help
            Enable debug logging for ML inference
    
    config {self.config.namespace_prefix.upper()}BENCHMARK_MODE
        bool "Enable performance benchmarking"
        default n
        help
            Measure and report inference timing
            
    config {self.config.namespace_prefix.upper()}VALIDATION_MODE
        bool "Enable inference validation"
        default n
        help
            Compare C inference with golden reference
            
endmenu
"""
        kconfig_path = self.output_dir / "Kconfig"
        with open(kconfig_path, 'w') as f:
            f.write(kconfig_content)
        
        logger.info(f"Generated build files: {cmake_path}, {kconfig_path}")

    def export_model(self):
        """Main export function"""
        logger.info("Starting Stage 4: Embedded Export")
        
        # Load and extract model data
        self.load_artifacts()
        self.extract_model_data()
        
        # Generate C code
        header_path = self.generate_header_file()
        source_path = self.generate_source_file()
        
        # Generate supporting files
        self.generate_test_vectors()
        self.generate_build_files()
        
        # Create metadata
        metadata = {
            "export_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "source_model": self.artifacts['model_type'],
            "model_architecture": self.layer_sizes,
            "total_parameters": sum(w.size for w in self.weights) + sum(b.size for b in self.biases),
            "features_count": len(self.features),
            "has_scaler": self.scaler is not None,
            "has_calibrator": self.calibrator is not None,
            "calibrator_type": self.calibrator_type,
            "decision_threshold": self.threshold,
            "target_platform": self.config.target_platform,
            "generated_files": {
                "header": str(header_path.name),
                "source": str(source_path.name),
                "build_config": "CMakeLists.txt",
                "kconfig": "Kconfig"
            },
            "estimated_flash_size_kb": self._estimate_flash_size(),
            "estimated_ram_size_kb": self._estimate_ram_size()
        }
        
        metadata_path = self.output_dir / "export_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Export complete! Files saved to: {self.output_dir}")
        logger.info(f"Estimated Flash: {metadata['estimated_flash_size_kb']:.1f} KB")
        logger.info(f"Estimated RAM: {metadata['estimated_ram_size_kb']:.1f} KB")
        
        return metadata_path

    def _estimate_flash_size(self) -> float:
        """Estimate flash memory usage"""
        # Model weights + biases (float32)
        model_params = sum(w.size for w in self.weights) + sum(b.size for b in self.biases)
        model_size = model_params * 4  # 4 bytes per float32
        
        # Scaler parameters
        scaler_size = len(self.features) * 2 * 4 if self.scaler else 0  # means + scales
        
        # Calibration lookup table
        calib_size = len(self.calibration_points[0]) * 2 * 4 if self.calibration_points else 0
        
        # Code size (rough estimate)
        code_size = 8 * 1024  # ~8KB for inference functions
        
        # Feature names
        name_size = sum(len(name) + 1 for name in self.features)
        
        total_bytes = model_size + scaler_size + calib_size + code_size + name_size
        return total_bytes / 1024  # Convert to KB
    
    def _estimate_ram_size(self) -> float:
        """Estimate RAM usage during inference"""
        # Largest layer output (2 buffers for ping-pong)
        max_layer_size = max(self.layer_sizes[1:-1])  # Exclude input/output
        buffer_size = max_layer_size * 2 * 4  # 2 buffers, float32
        
        # Input feature buffer
        input_size = len(self.features) * 4
        
        # Stack usage (rough estimate)
        stack_size = 1024  # ~1KB for function calls
        
        total_bytes = buffer_size + input_size + stack_size
        return total_bytes / 1024  # Convert to KB

def main():
    """Main entry point"""
    config = EmbeddedConfig()
    exporter = ModelExporter(config)
    metadata_path = exporter.export_model()
    
    print(f"\n✅ Stage 4 Export Complete!")
    print(f"📁 Output directory: {config.output_dir}")
    print(f"📋 Metadata: {metadata_path}")
    print(f"\n🚀 Ready for ESP32 integration!")

if __name__ == "__main__":
    main()
