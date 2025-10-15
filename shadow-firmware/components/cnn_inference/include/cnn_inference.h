/**
 * @file cnn_inference.h
 * @brief CNN inference engine for stress detection using TensorFlow Lite Micro
 * 
 * This component provides CNN-based stress probability inference using a quantized
 * TFLite model. It integrates with the signal_preprocessor component to process
 * 60 seconds of physiological sensor data (ACC, BVP, EDA, TEMP) and output a
 * continuous stress probability [0.0-1.0].
 * 
 * Architecture:
 *   Input:  (1, 4, 240) float32 - [ACC_MAG, BVP, EDA, TEMP] @ 4Hz
 *   Model:  121 KB quantized CNN (INT8 weights, FLOAT32 activations)
 *   Output: (1, 1) float32 - Stress probability
 * 
 * Memory requirements:
 *   - Model: 121 KB (embedded in stress_model_data.c)
 *   - Tensor arena: ~200 KB (configurable)
 *   - Total: ~330 KB
 * 
 * Performance:
 *   - Target inference time: <100ms on ESP32-S3 @ 240MHz
 *   - Inference frequency: Every 60 seconds
 */

#ifndef CNN_INFERENCE_H
#define CNN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "signal_preprocessor.h"

/**
 * @brief CNN inference result structure
 */
typedef struct {
    float stress_probability;     ///< Stress probability [0.0-1.0]
    uint32_t inference_time_us;   ///< Inference latency in microseconds
    bool success;                 ///< True if inference succeeded
} cnn_inference_result_t;

/**
 * @brief CNN inference configuration
 */
typedef struct {
    size_t tensor_arena_size;     ///< Tensor arena size in bytes (default: 200KB)
    bool enable_profiling;        ///< Enable detailed profiling (adds overhead)
} cnn_inference_config_t;

/**
 * @brief Get default CNN inference configuration
 * 
 * @return Default configuration structure
 */
cnn_inference_config_t cnn_inference_get_default_config(void);

/**
 * @brief Initialize CNN inference engine
 * 
 * Loads the embedded TFLite model, creates interpreter, and allocates tensor arena.
 * Must be called once during firmware initialization before any inference.
 * 
 * @param config Configuration (NULL for defaults)
 * @return 0 on success, negative error code on failure
 *         -1: Model version mismatch
 *         -2: Tensor allocation failed
 *         -3: Input tensor shape mismatch
 *         -4: Output tensor shape mismatch
 */
int cnn_inference_init(const cnn_inference_config_t *config);

/**
 * @brief Run CNN inference on preprocessed sensor data
 * 
 * Takes a preprocessed 60-second window of sensor data and runs CNN inference
 * to predict stress probability. This function is thread-safe if called from
 * a single task.
 * 
 * @param input Preprocessed CNN input tensor (4 channels x 240 samples @ 4Hz)
 * @param result Output structure containing stress probability and metadata
 * @return 0 on success, negative error code on failure
 *         -1: Invalid parameters (NULL pointers)
 *         -2: Inference engine not initialized
 *         -3: TFLite Invoke() failed
 */
int cnn_inference_predict(const cnn_input_tensor_t *input, 
                          cnn_inference_result_t *result);

/**
 * @brief Get memory usage statistics
 * 
 * Returns current tensor arena usage. Useful for optimizing tensor_arena_size.
 * 
 * @param arena_used_bytes Output: bytes currently used in tensor arena
 * @param arena_total_bytes Output: total tensor arena size allocated
 */
void cnn_inference_get_memory_stats(size_t *arena_used_bytes, 
                                    size_t *arena_total_bytes);

/**
 * @brief Get model information
 * 
 * @param model_size_bytes Output: embedded model size
 * @param input_shape Output: input tensor shape [batch, channels, timesteps]
 * @param output_shape Output: output tensor shape [batch, outputs]
 */
void cnn_inference_get_model_info(size_t *model_size_bytes,
                                  int input_shape[3],
                                  int output_shape[2]);

/**
 * @brief Deinitialize CNN inference engine
 * 
 * Releases resources allocated by cnn_inference_init().
 * After calling this, cnn_inference_init() must be called again before inference.
 */
void cnn_inference_deinit(void);

#ifdef __cplusplus
}
#endif

#endif // CNN_INFERENCE_H
