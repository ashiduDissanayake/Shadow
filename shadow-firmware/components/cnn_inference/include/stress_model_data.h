/**
 * @file stress_model_data.h
 * @brief TFLite model data for stress detection CNN
 * 
 * This file contains the embedded TFLite model as a constant array.
 * Generated automatically from stress_model_quant.tflite
 * 
 * Model specifications:
 *   Input:  (1, 4, 240) float32 - [ACC_MAG, BVP, EDA, TEMP]
 *   Output: (1, 1) float32 - Stress probability [0.0-1.0]
 *   Size:   121.27 KB
 *   Quantization: Dynamic range (INT8 weights, FLOAT32 activations)
 */

#ifndef STRESS_MODEL_DATA_H
#define STRESS_MODEL_DATA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * @brief Total size of the TFLite model in bytes
 */
#define STRESS_MODEL_SIZE 124176

/**
 * @brief Input tensor shape: [batch, channels, timesteps]
 */
#define STRESS_MODEL_INPUT_BATCH 1
#define STRESS_MODEL_INPUT_CHANNELS 4
#define STRESS_MODEL_INPUT_TIMESTEPS 240
#define STRESS_MODEL_INPUT_SIZE (STRESS_MODEL_INPUT_BATCH * STRESS_MODEL_INPUT_CHANNELS * STRESS_MODEL_INPUT_TIMESTEPS)

/**
 * @brief Output tensor shape: [batch, outputs]
 */
#define STRESS_MODEL_OUTPUT_BATCH 1
#define STRESS_MODEL_OUTPUT_SIZE 1

/**
 * @brief TFLite model data (aligned to 16 bytes for optimal performance)
 * 
 * This array contains the complete TFLite flatbuffer model.
 * Use this with TFLite Micro interpreter.
 */
extern const unsigned char g_stress_model_data[] __attribute__((aligned(16)));

/**
 * @brief Length of the model data array
 */
extern const unsigned int g_stress_model_data_len;

#ifdef __cplusplus
}
#endif

#endif // STRESS_MODEL_DATA_H
