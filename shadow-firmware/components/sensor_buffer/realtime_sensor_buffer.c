#include "realtime_sensor_buffer.h"
#include <string.h>
#include <stdlib.h>

static const char *TAG = "RealtimeSensor";

// Static buffers (exact window size per sensor)
static fixed_point_t bvp_buffer_data[BVP_BUFFER_SIZE];
static fixed_point_t acc_x_buffer_data[ACC_BUFFER_SIZE];
static fixed_point_t acc_y_buffer_data[ACC_BUFFER_SIZE];
static fixed_point_t acc_z_buffer_data[ACC_BUFFER_SIZE];
static fixed_point_t eda_buffer_data[EDA_BUFFER_SIZE];
static fixed_point_t temp_buffer_data[TEMP_BUFFER_SIZE];

static const struct {
    fixed_point_t *data;
    uint16_t size;
    uint8_t sample_rate;
} buffer_configs[NUM_SENSOR_BUFFERS] = {
    { bvp_buffer_data,   BVP_BUFFER_SIZE,  BVP_SAMPLE_RATE },
    { acc_x_buffer_data, ACC_BUFFER_SIZE,  ACC_SAMPLE_RATE },
    { acc_y_buffer_data, ACC_BUFFER_SIZE,  ACC_SAMPLE_RATE },
    { acc_z_buffer_data, ACC_BUFFER_SIZE,  ACC_SAMPLE_RATE },
    { eda_buffer_data,   EDA_BUFFER_SIZE,  EDA_SAMPLE_RATE },
    { temp_buffer_data,  TEMP_BUFFER_SIZE, TEMP_SAMPLE_RATE }
};

realtime_sensor_system_t g_sensor_system = {0};

/* ================= Initialization ================= */

int realtime_sensor_init(void) {
    ESP_LOGI(TAG, "Initializing real-time sensor buffer system (dual-counter Option A)...");

    memset(&g_sensor_system, 0, sizeof(g_sensor_system));

    for (int i = 0; i < NUM_SENSOR_BUFFERS; i++) {
        realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[i];
        buf->data        = buffer_configs[i].data;
        buf->size        = buffer_configs[i].size;
        buf->sample_rate = buffer_configs[i].sample_rate;
        atomic_store(&buf->ring_index, 0);
        atomic_store(&buf->total_samples, 0);
        memset(buf->data, 0, buf->size * sizeof(fixed_point_t));
        ESP_LOGI(TAG, "Buffer %d (%s): size=%u samples, rate=%u Hz",
                 i, get_sensor_name(i), buf->size, buf->sample_rate);
    }

    atomic_store(&g_sensor_system.last_processed_batch, 0);

    g_sensor_system.ml_ready_sem = xSemaphoreCreateBinary();
    if (!g_sensor_system.ml_ready_sem) {
        ESP_LOGE(TAG, "Failed to create ML semaphore");
        return -1;
    }

    g_sensor_system.initialized = 1;

    ESP_LOGI(TAG, "✅ Init complete. Window=%ds Step=%ds", WINDOW_SECONDS, STEP_SECONDS);
    ESP_LOGI(TAG, "Memory usage: %u bytes", realtime_get_memory_usage());
    return 0;
}

void realtime_sensor_deinit(void) {
    if (!g_sensor_system.initialized) return;

    if (g_sensor_system.ml_ready_sem) {
        vSemaphoreDelete(g_sensor_system.ml_ready_sem);
        g_sensor_system.ml_ready_sem = NULL;
    }
    for (int i = 0; i < NUM_SENSOR_BUFFERS; i++) {
        atomic_store(&g_sensor_system.buffers[i].ring_index, 0);
        atomic_store(&g_sensor_system.buffers[i].total_samples, 0);
    }
    atomic_store(&g_sensor_system.last_processed_batch, 0);
    g_sensor_system.initialized = 0;
    ESP_LOGI(TAG, "Deinitialized real-time sensor system");
}

/* ================= ISR Ingestion ================= */

int IRAM_ATTR realtime_add_sample_int_isr(sensor_id_t sensor_id, int32_t value_int) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS) return -1;

    realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[sensor_id];

    uint16_t idx = atomic_fetch_add(&buf->ring_index, 1);
    buf->data[idx % buf->size] = (fixed_point_t)value_int;
    atomic_fetch_add(&buf->total_samples, 1);

    check_and_signal_ml_ready();
    return 0;
}

int IRAM_ATTR realtime_add_sample_isr(sensor_id_t sensor_id, float value) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS) return -1;

    realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[sensor_id];

    uint16_t idx = atomic_fetch_add(&buf->ring_index, 1);
    buf->data[idx % buf->size] = FLOAT_TO_FIXED(value);
    atomic_fetch_add(&buf->total_samples, 1);

    check_and_signal_ml_ready();
    return 0;
}

int IRAM_ATTR realtime_add_samples_batch_isr(sensor_id_t sensor_id, const float *values, uint16_t count) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS || !values || count == 0) return -1;

    realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[sensor_id];

    uint16_t start_idx = atomic_fetch_add(&buf->ring_index, count);
    for (uint16_t i = 0; i < count; i++) {
        uint16_t ring_pos = (start_idx + i) % buf->size;
        buf->data[ring_pos] = FLOAT_TO_FIXED(values[i]);
    }
    atomic_fetch_add(&buf->total_samples, count);

    check_and_signal_ml_ready();
    return 0;
}

/* ================= Coordination (ISR) ================= */

void IRAM_ATTR check_and_signal_ml_ready(void) {
    if (!g_sensor_system.initialized) return;

    uint32_t min_batches = UINT32_MAX;

    for (int i = 0; i < NUM_SENSOR_BUFFERS; i++) {
        realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[i];
        uint32_t samples = atomic_load(&buf->total_samples);
        uint32_t batches = samples / buf->sample_rate;
        if (batches < min_batches) {
            min_batches = batches;
        }
    }

    if (min_batches == UINT32_MAX) return; // Should not happen early, but guard.

    uint32_t last_processed = atomic_load(&g_sensor_system.last_processed_batch);

    bool window_complete = (min_batches >= WINDOW_SECONDS);
    int32_t delta = (int32_t)min_batches - (int32_t)last_processed;
    bool step_ready = (delta >= (int32_t)STEP_SECONDS);

    if (window_complete && step_ready && g_sensor_system.ml_ready_sem) {
        BaseType_t hpw = pdFALSE;
        xSemaphoreGiveFromISR(g_sensor_system.ml_ready_sem, &hpw);
        if (hpw == pdTRUE) {
            portYIELD_FROM_ISR();
        }
    }
}

/* ================= Extraction (Task Context) ================= */

int realtime_extract_window(sensor_id_t sensor_id, fixed_point_t *output, uint16_t max_samples) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS || !output) return -1;

    realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[sensor_id];

    uint32_t total = atomic_load(&buf->total_samples);
    uint32_t window_needed = (uint32_t)buf->sample_rate * WINDOW_SECONDS;

    uint32_t have = (total < window_needed) ? total : window_needed;
    if (have > max_samples) have = max_samples;

    if (have == 0) return 0;

    uint32_t start_global = total - have;

    for (uint32_t i = 0; i < have; i++) {
        uint32_t global_idx = start_global + i;
        uint16_t ring_pos = (uint16_t)(global_idx % buf->size);
        output[i] = buf->data[ring_pos];
    }
    return (int)have;
}

void realtime_mark_batch_processed(uint32_t processed_batch) {
    if (!g_sensor_system.initialized) return;
    atomic_store(&g_sensor_system.last_processed_batch, processed_batch);
}

/* ================= Status / Query ================= */

uint16_t realtime_get_ring_index(sensor_id_t sensor_id) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS) return 0;
    return atomic_load(&g_sensor_system.buffers[sensor_id].ring_index);
}

uint32_t realtime_get_total_samples(sensor_id_t sensor_id) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS) return 0;
    return atomic_load(&g_sensor_system.buffers[sensor_id].total_samples);
}

uint32_t realtime_get_batch_count(sensor_id_t sensor_id) {
    if (!g_sensor_system.initialized || sensor_id >= NUM_SENSOR_BUFFERS) return 0;
    realtime_sensor_buffer_t *buf = &g_sensor_system.buffers[sensor_id];
    uint32_t samples = atomic_load(&buf->total_samples);
    return samples / buf->sample_rate;
}

uint32_t realtime_get_min_batch_count(void) {
    if (!g_sensor_system.initialized) return 0;
    uint32_t min_batches = UINT32_MAX;

    for (int i = 0; i < NUM_SENSOR_BUFFERS; i++) {
        uint32_t bc = realtime_get_batch_count(i);
        if (bc < min_batches) min_batches = bc;
    }
    return (min_batches == UINT32_MAX) ? 0 : min_batches;
}

int realtime_is_ml_ready(void) {
    if (!g_sensor_system.initialized) return 0;
    uint32_t min_batches = realtime_get_min_batch_count();
    uint32_t last_processed = atomic_load(&g_sensor_system.last_processed_batch);

    if (min_batches < WINDOW_SECONDS) return 0;
    int32_t delta = (int32_t)min_batches - (int32_t)last_processed;
    return (delta >= STEP_SECONDS) ? 1 : 0;
}

uint32_t realtime_get_memory_usage(void) {
    uint32_t total = sizeof(realtime_sensor_system_t);
    total += sizeof(bvp_buffer_data);
    total += sizeof(acc_x_buffer_data);
    total += sizeof(acc_y_buffer_data);
    total += sizeof(acc_z_buffer_data);
    total += sizeof(eda_buffer_data);
    total += sizeof(temp_buffer_data);
    return total;
}

void realtime_print_status(void) {
    if (!g_sensor_system.initialized) {
        ESP_LOGI(TAG, "System not initialized");
        return;
    }
    uint32_t min_batches = realtime_get_min_batch_count();
    uint32_t last_processed = atomic_load(&g_sensor_system.last_processed_batch);
    ESP_LOGI(TAG, "=== Real-Time Sensor System Status ===");
    ESP_LOGI(TAG, "Min batches: %u", min_batches);
    ESP_LOGI(TAG, "Last processed batch: %u", last_processed);
    ESP_LOGI(TAG, "ML Ready: %s", realtime_is_ml_ready() ? "YES" : "NO");

    for (int i = 0; i < NUM_SENSOR_BUFFERS; i++) {
        uint32_t ts = realtime_get_total_samples(i);
        uint16_t ri = realtime_get_ring_index(i);
        uint32_t bc = realtime_get_batch_count(i);
        ESP_LOGI(TAG, "  %s: total=%u ring_index=%u batches=%u",
                 get_sensor_name(i), ts, ri, bc);
    }
    ESP_LOGI(TAG, "======================================");
}