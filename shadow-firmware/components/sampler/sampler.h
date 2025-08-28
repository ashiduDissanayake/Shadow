#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SENSOR_BVP,
    SENSOR_ACC,
    SENSOR_EDA,
    SENSOR_TEMP,
    SENSOR_MAX
} sensor_type_t;

typedef struct {
    uint64_t timestamp_us;
    sensor_type_t type;
    union {
        struct { float ir, red, green; } bvp; // For MAX3010x
        struct { float x, y, z; } acc;        // For MPU6050
        float eda;                             // For GSR
        float temp;                            // For temperature
    } data;
} sensor_sample_t;

// Callback signature for new samples
typedef void (*sampler_callback_t)(const sensor_sample_t *sample, void *ctx);

// Public API
void sampler_init(void);
void sampler_start(void);
void sampler_stop(void);
void sampler_set_rate(sensor_type_t sensor, uint32_t hz);
void sampler_register_callback(sampler_callback_t cb, void *ctx);

#ifdef __cplusplus
}
#endif