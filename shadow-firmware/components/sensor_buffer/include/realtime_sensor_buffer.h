/*
 * Real-Time Producer-Consumer Sensor Buffer System
 * ESP32-S3 Multi-Sensor Coordinated Architecture
 *
 * Dual-counter design (Option A):
 *  - ring_index (uint16_t): only for circular addressing
 *  - total_samples (uint32_t): true monotonic sample count (never modulo)
 *
 * Benefits:
 *  - No semantic loss on wrap
 *  - Window & step logic always uses total_samples
 *  - Simple arithmetic (no wrap repair) for batches
 */

#ifndef REALTIME_SENSOR_BUFFER_H
#define REALTIME_SENSOR_BUFFER_H

#include <stdint.h>
#include <stdatomic.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "esp_log.h"

// Window / step coordination
#define WINDOW_SECONDS            60
#define STEP_SECONDS              10
#define NUM_SENSOR_BUFFERS        6

// Sampling rates (Hz)
#define BVP_SAMPLE_RATE           64
#define ACC_SAMPLE_RATE           32
#define EDA_SAMPLE_RATE           4
#define TEMP_SAMPLE_RATE          4

// Buffer sizes (exactly one full 60 s window per sensor)
#define BVP_BUFFER_SIZE           (BVP_SAMPLE_RATE * WINDOW_SECONDS)    // 3840
#define ACC_BUFFER_SIZE           (ACC_SAMPLE_RATE * WINDOW_SECONDS)    // 1920
#define EDA_BUFFER_SIZE           (EDA_SAMPLE_RATE * WINDOW_SECONDS)    // 240
#define TEMP_BUFFER_SIZE          (TEMP_SAMPLE_RATE * WINDOW_SECONDS)   // 240

typedef enum {
    SENSOR_BVP   = 0,
    SENSOR_ACC_X = 1,
    SENSOR_ACC_Y = 2,
    SENSOR_ACC_Z = 3,
    SENSOR_EDA   = 4,
    SENSOR_TEMP  = 5
} sensor_id_t;

// Fixed-point format (Q16.16 style implied scale)
typedef int32_t fixed_point_t;
#define FIXED_POINT_SCALE   65536
#define FLOAT_TO_FIXED(f)   ((fixed_point_t)((f) * FIXED_POINT_SCALE))
#define FIXED_TO_FLOAT(x)   ((float)(x) / FIXED_POINT_SCALE)

/*
 * Per-sensor buffer:
 *  ring_index   : ONLY for modulo writes (16-bit cycles forever)
 *  total_samples: Monotonic (32-bit) sample counter used for batch/window logic
 */
typedef struct {
    fixed_point_t *data;
    uint16_t       size;
    uint8_t        sample_rate;
    _Atomic uint16_t ring_index;       // Modulo address index
    _Atomic uint32_t total_samples;    // True lifetime count
} realtime_sensor_buffer_t;

typedef struct {
    realtime_sensor_buffer_t buffers[NUM_SENSOR_BUFFERS];

    _Atomic uint32_t last_processed_batch;  // Last fully processed batch index (seconds)

    SemaphoreHandle_t ml_ready_sem;
    TaskHandle_t      ml_task_handle;

    uint8_t initialized;
} realtime_sensor_system_t;

extern realtime_sensor_system_t g_sensor_system;

/* === INIT / DEINIT === */
int  realtime_sensor_init(void);
void realtime_sensor_deinit(void);

/* === ISR INGEST === */
int IRAM_ATTR realtime_add_sample_int_isr(sensor_id_t sensor_id, int32_t value_int);
int IRAM_ATTR realtime_add_sample_isr(sensor_id_t sensor_id, float value);
int IRAM_ATTR realtime_add_samples_batch_isr(sensor_id_t sensor_id, const float *values, uint16_t count);

/* === COORDINATION (ISR) === */
void IRAM_ATTR check_and_signal_ml_ready(void);

/* === EXTRACTION (Task Context) === */
int  realtime_extract_window(sensor_id_t sensor_id, fixed_point_t *output, uint16_t max_samples);
void realtime_mark_batch_processed(uint32_t processed_batch);

/* === DEBUG / STATUS === */
uint16_t realtime_get_ring_index(sensor_id_t sensor_id);          // New explicit function
uint32_t realtime_get_total_samples(sensor_id_t sensor_id);       // New
uint32_t realtime_get_batch_count(sensor_id_t sensor_id);
uint32_t realtime_get_min_batch_count(void);
int      realtime_is_ml_ready(void);
uint32_t realtime_get_memory_usage(void);
void     realtime_print_status(void);

/* Backward compatibility (old name) */
static inline uint16_t realtime_get_write_ptr(sensor_id_t sensor_id) {
    return realtime_get_ring_index(sensor_id);
}

/* === CONFIG HELPERS === */
static inline uint16_t get_sensor_buffer_size(sensor_id_t sensor_id) {
    const uint16_t sizes[NUM_SENSOR_BUFFERS] = {
        BVP_BUFFER_SIZE, ACC_BUFFER_SIZE, ACC_BUFFER_SIZE,
        ACC_BUFFER_SIZE, EDA_BUFFER_SIZE, TEMP_BUFFER_SIZE
    };
    return (sensor_id < NUM_SENSOR_BUFFERS) ? sizes[sensor_id] : 0;
}

static inline uint8_t get_sensor_sample_rate(sensor_id_t sensor_id) {
    const uint8_t rates[NUM_SENSOR_BUFFERS] = {
        BVP_SAMPLE_RATE, ACC_SAMPLE_RATE, ACC_SAMPLE_RATE,
        ACC_SAMPLE_RATE, EDA_SAMPLE_RATE, TEMP_SAMPLE_RATE
    };
    return (sensor_id < NUM_SENSOR_BUFFERS) ? rates[sensor_id] : 0;
}

static inline const char* get_sensor_name(sensor_id_t sensor_id) {
    const char* names[NUM_SENSOR_BUFFERS] = {
        "BVP", "ACC_X", "ACC_Y", "ACC_Z", "EDA", "TEMP"
    };
    return (sensor_id < NUM_SENSOR_BUFFERS) ? names[sensor_id] : "UNKNOWN";
}

#endif /* REALTIME_SENSOR_BUFFER_H */