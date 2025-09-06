/*
 * ESP32-S3 Real-Time Producer-Consumer Stress Detection System
 *
 * Simplified BLE mode:
 *  - BLE service advertises (sequence,state) via service data (single combined byte)
 *  - On confirmed stable stress FSM transition we log an event and call ble_stress_service_tick()
 *  - No notifications / no explicit advertisement update function / no connection status checks
 *
 * Architecture Summary:
 *  PRODUCER (Core 0):
 *      - Hardware timers (GPTimer) invoke ISR callbacks for mock sensor sampling
 *      - ISRs push fixed-point samples into lock-free ring buffers
 *  CONSUMER (Core 1):
 *      - Blocks on semaphore (ml_ready_sem) signaled when a full aligned window is ready
 *      - Extracts 60 s window, computes features, runs ML inference, feeds FSM
 *      - On confirmed state transition: event logged + BLE tick
 *
 * Coordination:
 *  - Atomic batch counters per sensor (not shown here, inside realtime_sensor_buffer module)
 *  - Min batch determination ensures temporal alignment across sensor rates
 *
 * This file has been cleaned to remove legacy BLE API calls:
 *  - ble_stress_service_is_connected()
 *  - ble_stress_service_update_advertisement()
 *  - ble_stress_service_notifications_enabled()
 *  - ble_stress_service_notify_fsm_state()
 *  - ble_stress_service_print_status()
 *  - ble_stress_service_start_advertising()
 *
 * Only these BLE APIs remain:
 *  - ble_stress_service_init()
 *  - ble_stress_service_tick()
 */

#include <stdio.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_random.h"
#include "driver/gptimer.h"

#include "realtime_sensor_buffer.h"
#include "feature_extractor.h"
#include "simple_mlp.h"
#include "stress_fsm.h"
#include "event_log.h"
#include "ble_stress_service.h"

static const char *TAG = "ShadowRealTime";

/* Task handles */
static TaskHandle_t producer_task_handle = NULL;
static TaskHandle_t consumer_task_handle = NULL;

/* GPTimer handles */
static gptimer_handle_t bvp_timer = NULL;
static gptimer_handle_t acc_timer = NULL;
static gptimer_handle_t eda_timer = NULL;
static gptimer_handle_t temp_timer = NULL;

/* Feature extraction workspace */
static feature_workspace_t g_feature_workspace;

/* FSM + Event Log contexts */
static stress_fsm_context_t g_stress_fsm;
static event_log_context_t g_event_log;

/* Statistics */
static uint32_t total_inferences = 0;
static uint32_t total_samples_collected = 0;
static uint32_t total_state_transitions = 0;

/* Forward declarations */
static bool IRAM_ATTR bvp_timer_isr_callback(gptimer_handle_t, const gptimer_alarm_event_data_t *, void *);
static bool IRAM_ATTR acc_timer_isr_callback(gptimer_handle_t, const gptimer_alarm_event_data_t *, void *);
static bool IRAM_ATTR eda_timer_isr_callback(gptimer_handle_t, const gptimer_alarm_event_data_t *, void *);
static bool IRAM_ATTR temp_timer_isr_callback(gptimer_handle_t, const gptimer_alarm_event_data_t *, void *);
static int  setup_sensor_timer(gptimer_handle_t *timer, uint32_t hz,
                               gptimer_alarm_cb_t cb, const char *name);
static int  start_sensor_timers(void);
void producer_task(void *param);
void consumer_task(void *param);
int extract_features_realtime(realtime_sensor_system_t *sensor_system,
                              feature_workspace_t *workspace,
                              feature_vector_t *result);
void on_stress_transition(const stress_state_transition_t *transition);

/* ================= MOCK SENSOR GENERATORS (fixed-point safe) ================= */

static int32_t  generate_mock_bvp_int(void) {
    static int32_t base_bvp_int = 5320; /* 53.20 * 100 */
    static uint32_t counter = 0;
    counter++;
    uint32_t pseudo_rand = (counter * 1103515245u + 12345u) & 0x7FFFFFFFu;
    int32_t variation = (int32_t)(pseudo_rand % 4000) - 2000;  /* -2000..+1999 */
    base_bvp_int += variation / 10;
    if (base_bvp_int < 299) base_bvp_int = 299;
    if (base_bvp_int > 30747) base_bvp_int = 30747;
    return base_bvp_int;
}

static int32_t generate_mock_acc_int(int axis) {
    static uint32_t acc_counter[3] = {0, 1000, 2000};
    static int32_t axis_means_int[3] = {1542, -618, 899};
    static int32_t axis_ranges_int[3][2] = {
        {-6497, 6249}, {-5928, 6393}, {-6159, 5871}
    };
    acc_counter[axis] += 17 + axis * 7;
    uint32_t pseudo_rand =
        (acc_counter[axis] * 1103515245u + 12345u + (uint32_t)axis * 4567u) & 0x7FFFFFFFu;
    int32_t movement = (int32_t)(pseudo_rand % 2000) - 1000;
    int32_t val = axis_means_int[axis] + movement;
    if (val < axis_ranges_int[axis][0]) val = axis_ranges_int[axis][0];
    if (val > axis_ranges_int[axis][1]) val = axis_ranges_int[axis][1];
    return val;
}

static int32_t generate_mock_eda_int(void) {
    static int32_t base_eda_int = 208; /* 2.08 * 100 */
    static uint32_t eda_counter = 0;
    eda_counter += 23;
    uint32_t pseudo_rand =
        (eda_counter * 1103515245u + 12345u + 7891u) & 0x7FFFFFFFu;
    int32_t drift = (int32_t)(pseudo_rand % 400) - 200;  /* -200..+199 */
    drift /= 100; /* small drift */
    int32_t noise = (int32_t)((pseudo_rand >> 8) % 200) - 100;
    noise /= 10;
    base_eda_int += drift;
    int32_t val = base_eda_int + noise;
    if (val < 9) val = 9;
    if (val > 1562) val = 1562;
    return val;
}

static int32_t generate_mock_temp_int(void) {
    static int32_t base_temp_int = 3309;
    static uint32_t temp_counter = 0;
    temp_counter += 31;
    uint32_t pseudo_rand =
        (temp_counter * 1103515245u + 12345u + 9876u) & 0x7FFFFFFFu;
    int32_t drift = (int32_t)(pseudo_rand % 200) - 100; /* -100..+99 */
    drift /= 1000; /* very slow drift */
    int32_t noise = (int32_t)((pseudo_rand >> 8) % 100) - 50;
    noise /= 100;
    base_temp_int += drift;
    int32_t val = base_temp_int + noise;
    if (val < 2939) val = 2939;
    if (val > 3593) val = 3593;
    return val;
}

/* ================= STRESS TRANSITION CALLBACK ================= */

void on_stress_transition(const stress_state_transition_t *transition) {
    if (!transition) return;

    ESP_LOGI(TAG, "🔄 STRESS TRANSITION DETECTED!");
    ESP_LOGI(TAG, "   %s → %s",
             stress_fsm_state_to_string(transition->from_state),
             stress_fsm_state_to_string(transition->to_state));
    ESP_LOGI(TAG, "   Confidence: %.3f", transition->confidence_score);
    ESP_LOGI(TAG, "   Duration prev state: %lu ms", transition->duration_prev_state_ms);

    uint8_t  sensor_quality = 85;   /* TODO: real calculation */
    uint16_t battery_mv     = 3300; /* TODO: ADC read */

    uint8_t seq = event_log_add_transition(&g_event_log,
                                           transition,
                                           sensor_quality,
                                           battery_mv);

    if (seq != EVENT_LOG_INVALID_SEQUENCE) {
        total_state_transitions++;
        ESP_LOGI(TAG, "   ✅ Event logged (seq=%u)", seq);
        /* Tick BLE so advertisement can reflect new stable state */
        ble_stress_service_tick();
    } else {
        ESP_LOGE(TAG, "   ❌ Failed to log transition");
    }
}

/* ================= ISR CALLBACKS ================= */

static bool IRAM_ATTR bvp_timer_isr_callback(gptimer_handle_t timer,
                                             const gptimer_alarm_event_data_t *edata,
                                             void *user_ctx) {
    realtime_add_sample_int_isr(SENSOR_BVP, generate_mock_bvp_int());
    total_samples_collected++;
    return false;
}

static bool IRAM_ATTR acc_timer_isr_callback(gptimer_handle_t timer,
                                             const gptimer_alarm_event_data_t *edata,
                                             void *user_ctx) {
    realtime_add_sample_int_isr(SENSOR_ACC_X, generate_mock_acc_int(0));
    realtime_add_sample_int_isr(SENSOR_ACC_Y, generate_mock_acc_int(1));
    realtime_add_sample_int_isr(SENSOR_ACC_Z, generate_mock_acc_int(2));
    total_samples_collected += 3;
    return false;
}

static bool IRAM_ATTR eda_timer_isr_callback(gptimer_handle_t timer,
                                             const gptimer_alarm_event_data_t *edata,
                                             void *user_ctx) {
    realtime_add_sample_int_isr(SENSOR_EDA, generate_mock_eda_int());
    total_samples_collected++;
    return false;
}

static bool IRAM_ATTR temp_timer_isr_callback(gptimer_handle_t timer,
                                              const gptimer_alarm_event_data_t *edata,
                                              void *user_ctx) {
    realtime_add_sample_int_isr(SENSOR_TEMP, generate_mock_temp_int());
    total_samples_collected++;
    return false;
}

/* ================= TIMER SETUP ================= */

static int setup_sensor_timer(gptimer_handle_t *timer,
                              uint32_t frequency_hz,
                              gptimer_alarm_cb_t callback,
                              const char *name) {
    gptimer_config_t config = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000
    };
    ESP_ERROR_CHECK(gptimer_new_timer(&config, timer));

    gptimer_alarm_config_t alarm = {
        .alarm_count = 1000000 / frequency_hz,
        .reload_count = 0,
        .flags.auto_reload_on_alarm = true
    };
    ESP_ERROR_CHECK(gptimer_set_alarm_action(*timer, &alarm));

    gptimer_event_callbacks_t cbs = {
        .on_alarm = callback
    };
    ESP_ERROR_CHECK(gptimer_register_event_callbacks(*timer, &cbs, NULL));

    ESP_LOGI(TAG, "✅ %s timer configured @ %lu Hz", name, frequency_hz);
    return 0;
}

static int start_sensor_timers(void) {
    ESP_LOGI(TAG, "Starting ISR timers...");
    ESP_ERROR_CHECK(gptimer_enable(bvp_timer));
    ESP_ERROR_CHECK(gptimer_start(bvp_timer));
    ESP_ERROR_CHECK(gptimer_enable(acc_timer));
    ESP_ERROR_CHECK(gptimer_start(acc_timer));
    ESP_ERROR_CHECK(gptimer_enable(eda_timer));
    ESP_ERROR_CHECK(gptimer_start(eda_timer));
    ESP_ERROR_CHECK(gptimer_enable(temp_timer));
    ESP_ERROR_CHECK(gptimer_start(temp_timer));
    ESP_LOGI(TAG, "🚀 All timers running");
    return 0;
}

/* ================= PRODUCER TASK ================= */

void producer_task(void *param) {
    ESP_LOGI(TAG, "🔧 Producer started (Core %d)", xPortGetCoreID());
    if (setup_sensor_timer(&bvp_timer, BVP_SAMPLE_RATE, bvp_timer_isr_callback, "BVP") ||
        setup_sensor_timer(&acc_timer, ACC_SAMPLE_RATE, acc_timer_isr_callback, "ACC") ||
        setup_sensor_timer(&eda_timer, EDA_SAMPLE_RATE, eda_timer_isr_callback, "EDA") ||
        setup_sensor_timer(&temp_timer, TEMP_SAMPLE_RATE, temp_timer_isr_callback, "TEMP")) {
        ESP_LOGE(TAG, "Timer setup failed");
        vTaskDelete(NULL);
        return;
    }
    if (start_sensor_timers() != 0) {
        ESP_LOGE(TAG, "Timer start failed");
        vTaskDelete(NULL);
        return;
    }

    uint32_t last_sample_count = 0;
    uint8_t  status_counter = 0;

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        uint32_t current_samples = total_samples_collected;
        uint32_t samples_per_sec = (current_samples - last_sample_count) / 5;
        last_sample_count = current_samples;

        ESP_LOGI(TAG, "📊 Samples/sec: %lu (total %lu)",
                 samples_per_sec, current_samples);
        ESP_LOGI(TAG, "🧠 Inferences: %lu", total_inferences);
        ESP_LOGI(TAG, "🔄 State Transitions: %lu", total_state_transitions);

        if (++status_counter >= 6) {
            status_counter = 0;
            ESP_LOGI(TAG, "=== DETAILED STATUS ===");
            realtime_print_status();
            event_log_print_status(&g_event_log);
            ESP_LOGI(TAG, "=======================");
        }

        /* Optional periodic BLE tick (keeps advertisement fresh if something changed) */
        ble_stress_service_tick();
    }
}

/* ================= CONSUMER TASK ================= */

void consumer_task(void *param) {
    ESP_LOGI(TAG, "🧠 Consumer started (Core %d)", xPortGetCoreID());
    vTaskDelay(pdMS_TO_TICKS(3000)); /* warm-up */

    while (1) {
        if (xSemaphoreTake(g_sensor_system.ml_ready_sem, portMAX_DELAY) == pdTRUE) {
            ESP_LOGI(TAG, "🔔 ML Inference #%lu", total_inferences);
            uint32_t t_start = xTaskGetTickCount() * portTICK_PERIOD_MS;

            uint32_t min_batches = realtime_get_min_batch_count();
            ESP_LOGI(TAG, "🎯 Min synchronized batches: %lu sec", min_batches);

            feature_vector_t features;
            int fr = extract_features_realtime(&g_sensor_system,
                                               &g_feature_workspace,
                                               &features);
            if (fr != 0) {
                ESP_LOGE(TAG, "Feature extraction failed (%d)", fr);
                continue;
            }
            ESP_LOGI(TAG, "✅ Features extracted in %lu ms", features.extraction_time_ms);

            uint32_t ml_start = xTaskGetTickCount() * portTICK_PERIOD_MS;
            float prob = shadow_mlp_predict_probability(features.features);
            int   cls  = shadow_mlp_predict_class(features.features);
            uint32_t ml_time = (xTaskGetTickCount() * portTICK_PERIOD_MS) - ml_start;

            realtime_mark_batch_processed(min_batches);

            uint32_t total_time = (xTaskGetTickCount() * portTICK_PERIOD_MS) - t_start;

            uint32_t now_ms = (uint32_t)(esp_timer_get_time() / 1000);
            bool transition = stress_fsm_process_inference(&g_stress_fsm,
                                                           prob,
                                                           now_ms,
                                                           on_stress_transition);

            ESP_LOGI(TAG, "🎯 Inference Result:");
            ESP_LOGI(TAG, "   Probability: %.3f  Class: %s",
                     prob, cls ? "STRESS" : "NORMAL");
            ESP_LOGI(TAG, "   FSM State: %s",
                     stress_fsm_state_to_string(stress_fsm_get_current_state(&g_stress_fsm)));
            ESP_LOGI(TAG, "   Transition: %s", transition ? "YES" : "NO");
            ESP_LOGI(TAG, "   Feature Time: %lu ms", features.extraction_time_ms);
            ESP_LOGI(TAG, "   ML Time: %lu ms", ml_time);
            ESP_LOGI(TAG, "   Total Time: %lu ms", total_time);
            ESP_LOGI(TAG, "   Batch Index Processed: %lu", min_batches);

            total_inferences++;

            if (transition) {
                /* on_stress_transition already called ble_stress_service_tick() */
                /* Optional: tick again (harmless) */
                ble_stress_service_tick();
            }

            ESP_LOGI(TAG, "---");
        }
    }
}

/* ================= FEATURE EXTRACTION BRIDGE ================= */

int extract_features_realtime(realtime_sensor_system_t *sensor_system,
                              feature_workspace_t *workspace,
                              feature_vector_t *result) {
    if (!sensor_system || !workspace || !result) return -1;

    uint32_t start_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;

    /* Example: pull a full BVP window */
    fixed_point_t bvp_window[BVP_BUFFER_SIZE];
    int bvp_samples = realtime_extract_window(SENSOR_BVP, bvp_window, BVP_BUFFER_SIZE);

    float bvp_data[BVP_BUFFER_SIZE];
    for (int i = 0; i < bvp_samples; i++) {
        bvp_data[i] = FIXED_TO_FLOAT(bvp_window[i]);
    }

    vTaskDelay(pdMS_TO_TICKS(50)); /* simulate cost */

    if (bvp_samples > 0) {
        float sum = 0.f, sum_sq = 0.f;
        float minv = bvp_data[0], maxv = bvp_data[0];
        for (int i = 0; i < bvp_samples; i++) {
            float v = bvp_data[i];
            sum += v;
            sum_sq += v * v;
            if (v < minv) minv = v;
            if (v > maxv) maxv = v;
        }
        float mean = sum / bvp_samples;
        float var = (sum_sq / bvp_samples) - mean * mean;
        float std = sqrtf(var > 0.f ? var : 0.f);

        result->features[0] = mean;
        result->features[1] = std;
        result->features[2] = minv;
        result->features[3] = maxv;
        result->features[4] = mean;                 /* approx median */
        result->features[5] = maxv - minv;          /* range */
        result->features[6] = std * 1.35f;          /* pseudo IQR */
        result->features[7] = sum_sq;               /* energy surrogate */
    }

    /* Mock remaining features (ACC, EDA, TEMP) */
    for (int axis = 0; axis < 3; axis++) {
        int base = 8 + axis * 5;
        result->features[base + 0] = (axis == 0) ? 15.42f : (axis == 1) ? -6.18f : 8.99f;
        result->features[base + 1] = 8.0f + (esp_random() % 400) / 100.0f;
        result->features[base + 2] = -30.0f - (esp_random() % 3000) / 100.0f;
        result->features[base + 3] = 30.0f + (esp_random() % 3000) / 100.0f;
        result->features[base + 4] = 100.0f + (esp_random() % 5000) / 100.0f;
    }

    result->features[23] = 2.08f + (esp_random() % 200 - 100) / 1000.0f;
    result->features[24] = 0.5f + (esp_random() % 300) / 1000.0f;
    result->features[25] = 0.09f + (esp_random() % 100) / 10000.0f;
    result->features[26] = 5.0f + (esp_random() % 1000) / 100.0f;

    result->features[27] = 33.09f + (esp_random() % 200 - 100) / 1000.0f;
    result->features[28] = 0.3f + (esp_random() % 200) / 1000.0f;
    result->features[29] = 2.0f + (esp_random() % 400) / 100.0f;

    result->extraction_time_ms = (xTaskGetTickCount() * portTICK_PERIOD_MS) - start_ms;
    result->success = true;
    result->timestamp = xTaskGetTickCount();

    return 0;
}

/* ================= MAIN ENTRY ================= */

void app_main(void) {
    ESP_LOGI(TAG, "🌟 Shadow Real-Time Stress Detection Firmware v3.0");
    ESP_LOGI(TAG, "Initializing subsystems...");

    if (realtime_sensor_init() != 0) {
        ESP_LOGE(TAG, "Failed realtime_sensor_init()");
        return;
    }
    if (feature_extractor_init(&g_feature_workspace) != 0) {
        ESP_LOGE(TAG, "Failed feature_extractor_init()");
        return;
    }
    if (stress_fsm_init(&g_stress_fsm) != 0) {
        ESP_LOGE(TAG, "Failed stress_fsm_init()");
        return;
    }
    if (event_log_init(&g_event_log) != 0) {
        ESP_LOGE(TAG, "Failed event_log_init()");
        return;
    }
    if (ble_stress_service_init(&g_stress_fsm, &g_event_log) != 0) {
        ESP_LOGE(TAG, "Failed ble_stress_service_init()");
        return;
    }

    ESP_LOGI(TAG, "✅ Initialization complete (Memory usage: %lu bytes)",
             realtime_get_memory_usage());

    /* Prime initial advertisement */
    ble_stress_service_tick();

    /* Create producer (Core 0) */
    xTaskCreatePinnedToCore(producer_task,
                            "producer",
                            4096,
                            NULL,
                            5,
                            &producer_task_handle,
                            0);

    /* Create consumer (Core 1) */
    xTaskCreatePinnedToCore(consumer_task,
                            "consumer",
                            32768,
                            NULL,
                            3,
                            &consumer_task_handle,
                            1);

    ESP_LOGI(TAG, "🚀 Tasks started: producer(Core0) / consumer(Core1)");
    ESP_LOGI(TAG, "System ONLINE");
}