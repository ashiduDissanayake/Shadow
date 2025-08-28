#include "sampler.h"
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "driver/i2c.h"
#include "driver/adc.h"
#include "esp_timer.h"

// --- Defaults (can be changed at runtime) ---
static uint32_t sampling_rates[SENSOR_MAX] = {64, 32, 4, 4};

// --- Callback/user context ---
static sampler_callback_t user_cb = NULL;
static void *user_ctx = NULL;

// --- Timer/task handles ---
static TimerHandle_t timers[SENSOR_MAX] = {0};

// --- Sensor initialization stubs ---
static bool bvp_init(void) { /* TODO: Init MAX3010x */ return true; }
static bool acc_init(void) { /* TODO: Init MPU6050 */ return true; }
static bool eda_init(void) { /* TODO: ADC config */ return true; }
static bool temp_init(void) { /* TODO: ADC/I2C config */ return true; }

static bool (*sensor_init_fns[SENSOR_MAX])(void) = {bvp_init, acc_init, eda_init, temp_init};

// --- Sensor read stubs ---
static void bvp_read(sensor_sample_t *out) { out->data.bvp.ir = 0; out->data.bvp.red = 0; out->data.bvp.green = 0; }
static void acc_read(sensor_sample_t *out) { out->data.acc.x = 0; out->data.acc.y = 0; out->data.acc.z = 0; }
static void eda_read(sensor_sample_t *out) { out->data.eda = 0; }
static void temp_read(sensor_sample_t *out) { out->data.temp = 0; }
static void (*sensor_read_fns[SENSOR_MAX])(sensor_sample_t*) = {bvp_read, acc_read, eda_read, temp_read};

// --- Timer callback (calls sensor read and invokes user callback) ---
static void timer_cb(TimerHandle_t xTimer) {
    sensor_type_t type = (sensor_type_t) (uintptr_t) pvTimerGetTimerID(xTimer);
    sensor_sample_t sample = {0};
    sample.timestamp_us = esp_timer_get_time();
    sample.type = type;
    sensor_read_fns[type](&sample);
    if (user_cb) user_cb(&sample, user_ctx);
}

void sampler_init(void) {
    for (sensor_type_t i = 0; i < SENSOR_MAX; ++i) {
        if (!sensor_init_fns[i]()) {
            printf("Sampler: Sensor %d init failed!\n", i);
            // Optionally: retry, set error status, etc.
        }
        if (timers[i]) vTimerDelete(timers[i]);
        timers[i] = NULL;
    }
}

void sampler_start(void) {
    for (sensor_type_t i = 0; i < SENSOR_MAX; ++i) {
        if (timers[i]) vTimerDelete(timers[i]);
        uint32_t period_ms = 1000 / (sampling_rates[i] ? sampling_rates[i] : 1);
        timers[i] = xTimerCreate("samp", period_ms / portTICK_PERIOD_MS, pdTRUE, (void*)(uintptr_t)i, timer_cb);
        if (timers[i]) xTimerStart(timers[i], 0);
    }
}

void sampler_stop(void) {
    for (sensor_type_t i = 0; i < SENSOR_MAX; ++i) {
        if (timers[i]) {
            xTimerStop(timers[i], 0);
            vTimerDelete(timers[i]);
            timers[i] = NULL;
        }
    }
}

void sampler_set_rate(sensor_type_t sensor, uint32_t hz) {
    if (sensor >= SENSOR_MAX) return;
    sampling_rates[sensor] = hz;
    if (timers[sensor]) {
        xTimerStop(timers[sensor], 0);
        vTimerDelete(timers[sensor]);
        timers[sensor] = NULL;
    }
    uint32_t period_ms = 1000 / (hz ? hz : 1);
    timers[sensor] = xTimerCreate("samp", period_ms / portTICK_PERIOD_MS, pdTRUE, (void*)(uintptr_t)sensor, timer_cb);
    if (timers[sensor]) xTimerStart(timers[sensor], 0);
}

void sampler_register_callback(sampler_callback_t cb, void *ctx) {
    user_cb = cb;
    user_ctx = ctx;
}