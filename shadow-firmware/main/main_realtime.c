/*
 * ESP32-S3 Real-Time Producer-Consumer Stress Detection System
 * Enhanced version with real sensor integration for Shadow Project
 * 
 * Real Sensor Integration:
 *  - MAX30105: Heart rate/BVP sensor via I2C with interrupt-driven FIFO
 *  - MPU6050: 3-axis accelerometer via I2C with data ready interrupt
 *  - GSR/EDA: Galvanic skin response via ADC with timer-based sampling
 *  - Temperature: Mock sensor (can be replaced with DS18B20 or onboard sensor)
 *
 * Architecture:
 *  PRODUCER (Core 0):
 *      - Hardware timers and interrupts drive real sensor data collection
 *      - ISRs and callbacks push samples into lock-free ring buffers
 *  CONSUMER (Core 1):
 *      - Processes sensor data for ML inference pipeline
 *      - Feeds stress detection FSM and BLE notification system
 *
 * Author: ashiduDissanayake
 * Version: 4.0 - Real Sensor Integration
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "driver/i2c.h"
#include "driver/gpio.h"
#include "driver/gptimer.h"
#include "esp_adc/adc_oneshot.h"     // ESP-IDF v5.5 ADC driver
#include "esp_adc/adc_cali.h"        // ADC calibration
#include "esp_adc/adc_cali_scheme.h" // Calibration schemes
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_random.h"

// Include Shadow project headers
#include "realtime_sensor_buffer.h"
#include "signal_preprocessor.h"    // Signal preprocessing for CNN
#include "cnn_inference.h"          // CNN inference engine (replaces MLP+FSM)
#include "calibration.h"            // Calibration system for personalized baseline
#include "feature_extractor.h"      // OLD: Keep for now, will remove later
#include "simple_mlp.h"             // OLD: Keep for now, will remove later
#include "stress_fsm.h"             // OLD: Keep for now, will remove later
#include "event_log.h"
#include "ble_stress_service.h"
#include "ble_pairing.h"            // BLE device pairing protocol
#include "display_manager.h"        // TFT display and QR code

// ==================== PIN CONFIGURATION ====================
#define I2C_SDA_PIN         44    // T-Display S3 SDA pin
#define I2C_SCL_PIN         43    // T-Display S3 SCL pin
#define BUTTON_LEFT_PIN     0     // T-Display S3 left button (GPIO 0) - Calibration control
#define BUTTON_RIGHT_PIN    14    // T-Display S3 right button (GPIO 14) - Display toggle
#define BUTTON_DEBOUNCE_MS  200   // Button debounce time
#define MAX_INT_PIN         1     // MAX30105 interrupt pin
#define MPU_INT_PIN         2     // MPU6050 interrupt pin
#define GSR_ADC_PIN         3     // GSR ADC pin (same as MAX_INT for now)
#define GSR_ADC_CHANNEL     ADC_CHANNEL_0
#define I2C_FREQ_HZ         100000

// ==================== SENSOR ADDRESSES ====================
#define MAX30105_ADDR       0x57
#define MPU6050_ADDR        0x68

// ==================== SAMPLING RATES ====================
// NOTE: All sensors configured to 4Hz for CNN model (no resampling needed)
#define BVP_TARGET_HZ       4
#define ACC_TARGET_HZ       4
#define EDA_TARGET_HZ       4
#define TEMP_TARGET_HZ      4

// ==================== GSR CONFIGURATION ====================
#define GSR_SUPPLY_VOLTAGE  3.3f
#define GSR_AVG_SAMPLES     10
#define GSR_BASELINE_SAMPLES 20

// EDA Voltage Range Configuration (0.1V to 2.5V typical for GSR)
#define EDA_MIN_VOLTAGE     0.1f   // Minimum realistic EDA voltage
#define EDA_MAX_VOLTAGE     2.5f   // Maximum realistic EDA voltage  
#define EDA_DEFAULT_VOLTAGE 1.5f   // Default baseline voltage
#define EDA_NOISE_THRESHOLD 0.05f  // Filter out voltage changes smaller than 50mV

// ==================== MAX30105 REGISTERS ====================
#define MAX_REG_INT_STATUS_1    0x00
#define MAX_REG_INT_ENABLE_1    0x02
#define MAX_REG_FIFO_DATA       0x07
#define MAX_REG_FIFO_CONFIG     0x08
#define MAX_REG_MODE_CONFIG     0x09
#define MAX_REG_SPO2_CONFIG     0x0A
#define MAX_REG_LED1_PA         0x0C

// ==================== MPU6050 REGISTERS ====================
#define MPU_REG_SMPLRT_DIV      0x19
#define MPU_REG_CONFIG          0x1A
#define MPU_REG_ACCEL_CONFIG    0x1C
#define MPU_REG_INT_PIN_CFG     0x37
#define MPU_REG_INT_ENABLE      0x38
#define MPU_REG_INT_STATUS      0x3A
#define MPU_REG_ACCEL_XOUT_H    0x3B
#define MPU_REG_PWR_MGMT_1      0x6B
#define MPU_REG_WHO_AM_I        0x75

static const char *TAG = "ShadowRealTime";
static const char *TAG_MAIN = "Shadow";
static const char *TAG_MAX = "MAX30105";
static const char *TAG_MPU = "MPU6050";
static const char *TAG_GSR = "GSR";
static const char *TAG_DATA = "DataFlow";

/* Task handles */
static TaskHandle_t producer_task_handle = NULL;
static TaskHandle_t consumer_task_handle = NULL;

/* Hardware handles */
static adc_oneshot_unit_handle_t adc_handle;
static adc_cali_handle_t adc_cali_handle = NULL;
static gptimer_handle_t gsr_timer = NULL;
static gptimer_handle_t max_poll_timer = NULL;  // RE-ENABLED - interrupt approach unreliable
static gptimer_handle_t temp_timer = NULL;       // Timer for temperature sampling
static QueueHandle_t sensor_event_queue;

/* Feature extraction workspace */
static feature_workspace_t g_feature_workspace;

/* FSM + Event Log contexts */
static stress_fsm_context_t g_stress_fsm;
static event_log_context_t g_event_log;

/* Sensor availability flags */
static bool max_available = false;
static bool mpu_available = false;
static bool gsr_available = false;

/* Display and device info */
static display_device_info_t g_device_info = {
    .device_name = "Shadow-9026",
    .password = NULL  // No password needed - removed for simplified pairing
};
static volatile int64_t last_button_left_press = 0;
static volatile int64_t last_button_right_press = 0;
static bool adc_calibrated = false;
static bool recalibration_confirm_pending = false;  // Track if waiting for confirmation

/* Sensor statistics */
static uint32_t total_inferences = 0;
static uint32_t total_samples_collected = 0;
static uint32_t total_state_transitions = 0;
static uint32_t bvp_sample_count = 0;
static uint32_t acc_sample_count = 0;
static uint32_t eda_sample_count = 0;
static uint32_t temp_sample_count = 0;

/* EDA processing variables */
static float previous_eda_voltage = EDA_DEFAULT_VOLTAGE;
static bool eda_initialized = false;

/* Dynamic MPU6050 address detection */
static uint8_t mpu6050_addr = 0x68;

/* ================= EVENT SYSTEM ================= */
typedef enum {
    SENSOR_EVENT_MAX_DATA_READY,
    SENSOR_EVENT_MPU_DATA_READY,
    SENSOR_EVENT_GSR_TIMER,
    SENSOR_EVENT_TEMP_TIMER,
    SENSOR_EVENT_MAX_POLL,  // Re-enabled for polling-based sampling
    SENSOR_EVENT_BUTTON_PRESS,  // Button press to toggle display
    SENSOR_EVENT_BUTTON_LEFT,   // Left button (calibration control)
    SENSOR_EVENT_BUTTON_RIGHT,  // Right button (display toggle)
} sensor_event_type_t;

typedef struct {
    sensor_event_type_t type;
    uint64_t timestamp_us;
    uint32_t sequence;
    union {
        uint32_t bvp_value;
        struct { float ax, ay, az; } accel;
        float eda_voltage;
        float temperature;
    } data;
} sensor_event_t;

/* ================= FORWARD DECLARATIONS ================= */
static bool i2c_read_bytes(uint8_t device_addr, uint8_t reg_addr, uint8_t *data, size_t len);
static bool i2c_read_byte(uint8_t device_addr, uint8_t reg_addr, uint8_t *data);
static bool i2c_write_byte(uint8_t device_addr, uint8_t reg_addr, uint8_t data);
static bool gsr_timer_callback(gptimer_handle_t, const gptimer_alarm_event_data_t *, void *);
// static bool max_poll_timer_callback(gptimer_handle_t, const gptimer_alarm_event_data_t *, void *);  // DISABLED
static bool temp_timer_callback(gptimer_handle_t, const gptimer_alarm_event_data_t *, void *);     // Temperature sampling
static void max_interrupt_handler(void *arg);
static void mpu_interrupt_handler(void *arg);
void producer_task(void *param);
void consumer_task(void *param);
static void enhanced_sensor_processing_task(void *param);
void on_stress_transition(const stress_state_transition_t *transition);
int extract_features_realtime(realtime_sensor_system_t *sensor_system,
                              feature_workspace_t *workspace,
                              feature_vector_t *result);

/* ================= INTERRUPT HANDLERS ================= */

// MAX30105 uses hardware averaging (50 SPS / 16 avg = 3.125 Hz effective rate)
// This is close enough to our 4Hz target without software decimation

static void IRAM_ATTR max_interrupt_handler(void *arg) {
    static uint32_t max_sequence = 0;
    
    // Process every interrupt - hardware averaging already reduces rate to ~3.1Hz
    sensor_event_t event = {
        .type = SENSOR_EVENT_MAX_DATA_READY,
        .timestamp_us = esp_timer_get_time(),
        .sequence = ++max_sequence
    };
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xQueueSendFromISR(sensor_event_queue, &event, &xHigherPriorityTaskWoken);
    if (xHigherPriorityTaskWoken) {
        portYIELD_FROM_ISR();
    }
}

static void IRAM_ATTR mpu_interrupt_handler(void *arg) {
    static uint32_t mpu_sequence = 0;
    sensor_event_t event = {
        .type = SENSOR_EVENT_MPU_DATA_READY,
        .timestamp_us = esp_timer_get_time(),
        .sequence = ++mpu_sequence
    };
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xQueueSendFromISR(sensor_event_queue, &event, &xHigherPriorityTaskWoken);
    if (xHigherPriorityTaskWoken) {
        portYIELD_FROM_ISR();
    }
}

/**
 * Left button interrupt handler - Calibration control
 * Starts/stops calibration session
 */
static void IRAM_ATTR button_left_interrupt_handler(void *arg) {
    int64_t now = esp_timer_get_time() / 1000;  // Convert to ms
    
    // Debounce
    if (now - last_button_left_press < BUTTON_DEBOUNCE_MS) {
        return;
    }
    
    last_button_left_press = now;
    
    sensor_event_t event = {
        .type = SENSOR_EVENT_BUTTON_LEFT,
        .timestamp_us = esp_timer_get_time(),
        .sequence = 0
    };
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xQueueSendFromISR(sensor_event_queue, &event, &xHigherPriorityTaskWoken);
    if (xHigherPriorityTaskWoken) {
        portYIELD_FROM_ISR();
    }
}

/**
 * Right button interrupt handler - Display toggle
 * Debounced in ISR to avoid multiple triggers
 */
static void IRAM_ATTR button_right_interrupt_handler(void *arg) {
    int64_t now = esp_timer_get_time() / 1000;  // Convert to ms
    
    // Debounce
    if (now - last_button_right_press < BUTTON_DEBOUNCE_MS) {
        return;
    }
    
    last_button_right_press = now;
    
    sensor_event_t event = {
        .type = SENSOR_EVENT_BUTTON_RIGHT,
        .timestamp_us = esp_timer_get_time(),
        .sequence = 0
    };
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xQueueSendFromISR(sensor_event_queue, &event, &xHigherPriorityTaskWoken);
    if (xHigherPriorityTaskWoken) {
        portYIELD_FROM_ISR();
    }
}

static bool IRAM_ATTR gsr_timer_callback(gptimer_handle_t timer, const gptimer_alarm_event_data_t *edata, void *user_ctx) {
    static uint32_t gsr_sequence = 0;
    sensor_event_t event = {
        .type = SENSOR_EVENT_GSR_TIMER,
        .timestamp_us = esp_timer_get_time(),
        .sequence = ++gsr_sequence
    };
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xQueueSendFromISR(sensor_event_queue, &event, &xHigherPriorityTaskWoken);
    return (xHigherPriorityTaskWoken == pdTRUE);
}

/* MAX30105 polling timer callback - RE-ENABLED (interrupt approach unreliable) */
static bool IRAM_ATTR max_poll_timer_callback(gptimer_handle_t timer, const gptimer_alarm_event_data_t *edata, void *user_ctx) {
    static uint32_t max_poll_sequence = 0;
    sensor_event_t event = {
        .type = SENSOR_EVENT_MAX_POLL,
        .timestamp_us = esp_timer_get_time(),
        .sequence = ++max_poll_sequence
    };
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xQueueSendFromISR(sensor_event_queue, &event, &xHigherPriorityTaskWoken);
    return (xHigherPriorityTaskWoken == pdTRUE);
}


static bool IRAM_ATTR temp_timer_callback(gptimer_handle_t timer, const gptimer_alarm_event_data_t *edata, void *user_ctx) {
    static uint32_t temp_sequence = 0;
    sensor_event_t event = {
        .type = SENSOR_EVENT_TEMP_TIMER,
        .timestamp_us = esp_timer_get_time(),
        .sequence = ++temp_sequence
    };
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xQueueSendFromISR(sensor_event_queue, &event, &xHigherPriorityTaskWoken);
    return (xHigherPriorityTaskWoken == pdTRUE);
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

/* ================= MOCK TEMPERATURE GENERATOR ================= */
static float generate_mock_temperature(void) {
    static uint32_t counter = 0;
    counter++;
    
    // Realistic body temperature simulation
    float base_temp = 36.5f;
    float daily_cycle = sinf((counter * 0.01f)) * 0.8f;  // ±0.8°C variation
    
    // Generate realistic noise
    int32_t random_val = (int32_t)(esp_random() % 200) - 100;  // -100 to +99
    float random_noise = random_val * 0.001f;  // ±0.1°C noise
    
    float final_temp = base_temp + daily_cycle + random_noise;
    
    // Sanity check to prevent unrealistic values
    if (final_temp < 30.0f || final_temp > 45.0f) {
        final_temp = 36.5f;  // Fallback to normal
    }
    
    return final_temp;
}

// ==================== I2C HELPER FUNCTIONS ====================
static esp_err_t i2c_master_init(void) {
    ESP_LOGI(TAG_MAIN, "Initializing I2C bus (SDA:%d, SCL:%d, %dkHz)", 
             I2C_SDA_PIN, I2C_SCL_PIN, I2C_FREQ_HZ/1000);
    
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_SDA_PIN,
        .scl_io_num = I2C_SCL_PIN,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_FREQ_HZ,
    };
    
    esp_err_t err = i2c_param_config(I2C_NUM_0, &conf);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_MAIN, "I2C param config failed: %s", esp_err_to_name(err));
        return err;
    }
    
    err = i2c_driver_install(I2C_NUM_0, conf.mode, 0, 0, 0);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_MAIN, "I2C driver install failed: %s", esp_err_to_name(err));
        return err;
    }
    
    ESP_LOGI(TAG_MAIN, "I2C bus initialized successfully");
    return ESP_OK;
}

static bool i2c_write_byte(uint8_t device_addr, uint8_t reg_addr, uint8_t data) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (device_addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_write_byte(cmd, data, true);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(I2C_NUM_0, cmd, pdMS_TO_TICKS(50));
    i2c_cmd_link_delete(cmd);
    return ret == ESP_OK;
}

static bool i2c_read_bytes(uint8_t device_addr, uint8_t reg_addr, uint8_t *data, size_t len) {
    if (len == 0) return false;
    
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (device_addr << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (device_addr << 1) | I2C_MASTER_READ, true);
    
    if (len > 1) {
        i2c_master_read(cmd, data, len - 1, I2C_MASTER_ACK);
    }
    i2c_master_read_byte(cmd, data + len - 1, I2C_MASTER_LAST_NACK);
    i2c_master_stop(cmd);
    
    esp_err_t ret = i2c_master_cmd_begin(I2C_NUM_0, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    return ret == ESP_OK;
}

// Fixed: Now i2c_read_bytes is declared before i2c_read_byte
static bool i2c_read_byte(uint8_t device_addr, uint8_t reg_addr, uint8_t *data) {
    return i2c_read_bytes(device_addr, reg_addr, data, 1);
}

// I2C device scan function for debugging
static void i2c_scan_devices(void) {
    ESP_LOGI(TAG_MAIN, "Scanning I2C bus for connected devices...");
    int devices_found = 0;
    
    for (uint8_t addr = 1; addr < 127; addr++) {
        i2c_cmd_handle_t cmd = i2c_cmd_link_create();
        i2c_master_start(cmd);
        i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_WRITE, true);
        i2c_master_stop(cmd);
        
        esp_err_t ret = i2c_master_cmd_begin(I2C_NUM_0, cmd, pdMS_TO_TICKS(50));
        i2c_cmd_link_delete(cmd);
        
        if (ret == ESP_OK) {
            ESP_LOGI(TAG_MAIN, "Found I2C device at address: 0x%02X", addr);
            devices_found++;
        }
    }
    
    if (devices_found == 0) {
        ESP_LOGW(TAG_MAIN, "No I2C devices found on the bus");
    } else {
        ESP_LOGI(TAG_MAIN, "I2C scan complete - found %d device(s)", devices_found);
    }
}

// ==================== ADC FUNCTIONS (ESP-IDF v5.5) ====================
static bool gsr_adc_init(void) {
    ESP_LOGI(TAG_GSR, "Initializing GSR ADC on GPIO%d (ESP-IDF v5.5)...", GSR_ADC_PIN);
    
    // ADC unit initialization
    adc_oneshot_unit_init_cfg_t init_config = {
        .unit_id = ADC_UNIT_1,
        .ulp_mode = ADC_ULP_MODE_DISABLE,
    };
    
    esp_err_t err = adc_oneshot_new_unit(&init_config, &adc_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_GSR, "ADC unit init failed: %s", esp_err_to_name(err));
        return false;
    }
    
    // Channel configuration - Fixed: Use ADC_ATTEN_DB_12 instead of deprecated ADC_ATTEN_DB_11
    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ADC_BITWIDTH_DEFAULT,
        .atten = ADC_ATTEN_DB_12,    // Fixed: Use non-deprecated constant
    };
    
    err = adc_oneshot_config_channel(adc_handle, GSR_ADC_CHANNEL, &config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_GSR, "ADC channel config failed: %s", esp_err_to_name(err));
        return false;
    }
    
    // ADC Calibration for ESP-IDF v5.5
    adc_cali_curve_fitting_config_t cali_config = {
        .unit_id = ADC_UNIT_1,
        .atten = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    
    err = adc_cali_create_scheme_curve_fitting(&cali_config, &adc_cali_handle);
    if (err == ESP_OK) {
        adc_calibrated = true;
        ESP_LOGI(TAG_GSR, "ADC calibration successful");
    } else {
        ESP_LOGW(TAG_GSR, "ADC calibration failed: %s, using raw values", esp_err_to_name(err));
        adc_calibrated = false;
    }
    
    ESP_LOGI(TAG_GSR, "GSR ADC initialized successfully");
    return true;
}

static float gsr_read_voltage(void) {
    int total = 0;
    int valid_readings = 0;
    
    // Take multiple readings and average
    for (int i = 0; i < GSR_AVG_SAMPLES; i++) {
        int raw_value;
        esp_err_t err = adc_oneshot_read(adc_handle, GSR_ADC_CHANNEL, &raw_value);
        if (err == ESP_OK) {
            total += raw_value;
            valid_readings++;
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    
    if (valid_readings == 0) {
        ESP_LOGW(TAG_GSR, "No valid ADC readings");
        return -1.0f;
    }
    
    int avg_raw = total / valid_readings;
    
    // Use calibration if available
    if (adc_calibrated && adc_cali_handle != NULL) {
        int voltage_mv;
        esp_err_t err = adc_cali_raw_to_voltage(adc_cali_handle, avg_raw, &voltage_mv);
        if (err == ESP_OK) {
            return voltage_mv / 1000.0f;  // Convert mV to V
        }
    }
    
    // Fallback to manual calculation
    return (avg_raw / 4095.0f) * GSR_SUPPLY_VOLTAGE;
}

static float gsr_process_voltage(float raw_voltage) {
    // Apply range validation and smoothing for more realistic EDA readings
    
    // Clamp voltage to realistic EDA range
    if (raw_voltage < EDA_MIN_VOLTAGE) {
        raw_voltage = EDA_MIN_VOLTAGE;
    } else if (raw_voltage > EDA_MAX_VOLTAGE) {
        raw_voltage = EDA_MAX_VOLTAGE;
    }
    
    // Initialize baseline on first reading
    if (!eda_initialized) {
        previous_eda_voltage = raw_voltage;
        eda_initialized = true;
        ESP_LOGI(TAG_GSR, "EDA baseline initialized: %.3fV", raw_voltage);
        return raw_voltage;
    }
    
    // Apply noise filtering - ignore changes smaller than threshold
    float voltage_diff = fabsf(raw_voltage - previous_eda_voltage);
    if (voltage_diff < EDA_NOISE_THRESHOLD) {
        // Keep previous value to reduce noise
        return previous_eda_voltage;
    }
    
    // Apply light smoothing for remaining changes (80% new, 20% old)
    float smoothed_voltage = (raw_voltage * 0.8f) + (previous_eda_voltage * 0.2f);
    
    // Update previous voltage
    previous_eda_voltage = smoothed_voltage;
    
    return smoothed_voltage;
}

// ==================== MAX30105 FUNCTIONS ====================
// ==================== ENHANCED MAX30105 FUNCTIONS ====================

static bool max30105_debug_status(void) {
    uint8_t status1, status2, fifo_wr, fifo_rd, fifo_ovf;
    
    ESP_LOGI(TAG_MAX, "=== MAX30105 Debug Status ===");
    
    // Read all status registers
    if (!i2c_read_byte(MAX30105_ADDR, 0x00, &status1)) {
        ESP_LOGE(TAG_MAX, "Failed to read INT_STATUS_1");
        return false;
    }
    
    if (!i2c_read_byte(MAX30105_ADDR, 0x01, &status2)) {
        ESP_LOGE(TAG_MAX, "Failed to read INT_STATUS_2");
        return false;
    }
    
    if (!i2c_read_byte(MAX30105_ADDR, 0x04, &fifo_wr)) {
        ESP_LOGE(TAG_MAX, "Failed to read FIFO_WR_PTR");
        return false;
    }
    
    if (!i2c_read_byte(MAX30105_ADDR, 0x06, &fifo_rd)) {
        ESP_LOGE(TAG_MAX, "Failed to read FIFO_RD_PTR");
        return false;
    }
    
    if (!i2c_read_byte(MAX30105_ADDR, 0x05, &fifo_ovf)) {
        ESP_LOGE(TAG_MAX, "Failed to read FIFO_OVF");
        return false;
    }
    
    ESP_LOGI(TAG_MAX, "INT_STATUS_1: 0x%02X", status1);
    ESP_LOGI(TAG_MAX, "INT_STATUS_2: 0x%02X", status2);
    ESP_LOGI(TAG_MAX, "FIFO_WR_PTR:  0x%02X", fifo_wr);
    ESP_LOGI(TAG_MAX, "FIFO_RD_PTR:  0x%02X", fifo_rd);
    ESP_LOGI(TAG_MAX, "FIFO_OVF:     0x%02X", fifo_ovf);
    ESP_LOGI(TAG_MAX, "FIFO_COUNT:   %d", (fifo_wr - fifo_rd) & 0x1F);
    
    return true;
}

static bool max30105_enhanced_init(void) {
    ESP_LOGI(TAG_MAX, "Enhanced MAX30105 initialization...");
    
    // Reset device
    if (!i2c_write_byte(MAX30105_ADDR, 0x09, 0x40)) {
        ESP_LOGE(TAG_MAX, "Device reset failed");
        return false;
    }
    vTaskDelay(pdMS_TO_TICKS(100));  // Increased delay
    
    // Clear FIFO pointers
    if (!i2c_write_byte(MAX30105_ADDR, 0x04, 0x00)) {  // FIFO_WR_PTR
        ESP_LOGE(TAG_MAX, "FIFO_WR_PTR clear failed");
        return false;
    }
    
    if (!i2c_write_byte(MAX30105_ADDR, 0x05, 0x00)) {  // FIFO_OVF_COUNTER
        ESP_LOGE(TAG_MAX, "FIFO_OVF clear failed");
        return false;
    }
    
    if (!i2c_write_byte(MAX30105_ADDR, 0x06, 0x00)) {  // FIFO_RD_PTR
        ESP_LOGE(TAG_MAX, "FIFO_RD_PTR clear failed");
        return false;
    }
    
    // FIFO Configuration: Sample averaging = 8, FIFO rollover enabled, FIFO almost full = 1
    // Averaging 8 samples with 32 SPS base = 4 Hz exact
    // 32 SPS base / 8 averaging = 4.0 Hz (matches other sensors perfectly!)
    if (!i2c_write_byte(MAX30105_ADDR, 0x08, 0x71)) {  // 0x71 = averaging 8 (bits[7:5]=011), rollover enabled, almost full at 1
        ESP_LOGE(TAG_MAX, "FIFO config failed");
        return false;
    }
    
    // Mode Configuration: Heart Rate mode (IR LED for MAX30102)
    // MAX30102 only has RED and IR LEDs (no GREEN!)
    // 0x02 = Heart Rate mode (RED only)
    // 0x03 = SpO2 mode (RED + IR)
    // Using mode 0x02 (RED only) for simplicity and better BVP signal
    if (!i2c_write_byte(MAX30105_ADDR, 0x09, 0x02)) {
        ESP_LOGE(TAG_MAX, "Mode config failed");
        return false;
    }
    
    // SpO2 Configuration: 32 SPS, 411μs pulse width, ADC range 4096nA
    // Note: Despite name, this register controls sample rate for ALL modes
    // 32 SPS with 8x averaging = 4.0Hz exact (perfect match with ACC/EDA/TEMP!)
    if (!i2c_write_byte(MAX30105_ADDR, 0x0A, 0x1F)) {  // 0x1F = 32 SPS (bits[6:2]=00111), 411μs, 4096nA
        ESP_LOGE(TAG_MAX, "Sample rate config failed");
        return false;
    }
    
    // LED1 (RED) Pulse Amplitude: Medium-high intensity for BVP signal
    // MAX30102 uses RED LED for heart rate measurement
    if (!i2c_write_byte(MAX30105_ADDR, 0x0C, 0x3F)) {
        ESP_LOGE(TAG_MAX, "LED1 (RED) config failed");
        return false;
    }
    
    // Enable FIFO Almost Full interrupt (bit 7) and New FIFO Data Ready (bit 6)
    if (!i2c_write_byte(MAX30105_ADDR, 0x02, 0xC0)) {
        ESP_LOGE(TAG_MAX, "Interrupt enable failed");
        return false;
    }
    
    // Clear any pending interrupts
    uint8_t status1, status2;
    i2c_read_byte(MAX30105_ADDR, 0x00, &status1);
    i2c_read_byte(MAX30105_ADDR, 0x01, &status2);
    
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // Debug status after initialization
    max30105_debug_status();
    
    // CRITICAL: Empty FIFO to clear interrupt pin
    // During init, samples accumulate and trigger interrupt
    // We must read them out to allow future interrupts to fire
    ESP_LOGI(TAG_MAX, "Clearing FIFO to reset interrupt pin...");
    for (int i = 0; i < 32; i++) {  // FIFO is 32 samples max
        uint8_t dummy[3];
        i2c_read_bytes(MAX30105_ADDR, 0x07, dummy, 3);  // Read and discard
    }
    
    // Clear interrupt status again after emptying FIFO
    i2c_read_byte(MAX30105_ADDR, 0x00, &status1);
    
    ESP_LOGI(TAG_MAX, "Enhanced MAX30105 initialized successfully (FIFO cleared)");
    return true;
}

// Enhanced data reading with better FIFO handling
static uint32_t max30105_read_fifo_sample(void) {
    // Read FIFO pointers to check if data is available
    uint8_t fifo_wr, fifo_rd;
    
    if (!i2c_read_byte(MAX30105_ADDR, 0x04, &fifo_wr)) {
        return 0;
    }
    
    if (!i2c_read_byte(MAX30105_ADDR, 0x06, &fifo_rd)) {
        return 0;
    }
    
    // Calculate number of samples in FIFO (32-entry circular buffer)
    int samples_available = (fifo_wr - fifo_rd) & 0x1F;
    
    if (samples_available == 0) {
        return 0;  // No data in FIFO
    }
    
    // Clear interrupt status by reading (important for next interrupt)
    uint8_t status1;
    i2c_read_byte(MAX30105_ADDR, 0x00, &status1);
    
    // Read FIFO data (3 bytes for RED LED)
    uint8_t data[3];
    if (!i2c_read_bytes(MAX30105_ADDR, 0x07, data, 3)) {
        ESP_LOGW(TAG_MAX, "FIFO read failed");
        return 0;
    }
    
    // Combine the 18-bit value
    uint32_t ir_value = ((uint32_t)data[0] << 16) | ((uint32_t)data[1] << 8) | data[2];
    ir_value &= 0x3FFFF;  // Mask to 18 bits
    
    return ir_value;
}

// Add manual polling as backup
static void max30105_manual_poll(void) {
    static uint32_t manual_poll_count = 0;
    
    uint32_t sample = max30105_read_fifo_sample();
    if (sample > 0) {
        manual_poll_count++;
        bvp_sample_count++;
        
        ESP_LOGI(TAG_DATA, "[MANUAL] BVP: %lu (#%lu)", sample, bvp_sample_count);
    }
    
    // Debug every 10 manual polls
    if (manual_poll_count % 10 == 0 && manual_poll_count > 0) {
        max30105_debug_status();
    }
}

// ==================== MPU6050 FUNCTIONS ====================
static bool mpu6050_init(void) {
    ESP_LOGI(TAG_MPU, "Initializing MPU6050 accelerometer...");
    
    // Try both possible MPU6050 addresses
    uint8_t addresses[] = {0x68, 0x69};
    uint8_t actual_addr = 0;
    uint8_t who_am_i = 0;
    bool found = false;
    
    for (int i = 0; i < 2; i++) {
        ESP_LOGI(TAG_MPU, "Trying MPU6050 at address 0x%02X...", addresses[i]);
        if (i2c_read_byte(addresses[i], MPU_REG_WHO_AM_I, &who_am_i)) {
            ESP_LOGI(TAG_MPU, "WHO_AM_I at 0x%02X: 0x%02X", addresses[i], who_am_i);
            if (who_am_i == 0x68 || who_am_i == 0x69) {
                actual_addr = addresses[i];
                found = true;
                break;
            }
        } else {
            ESP_LOGW(TAG_MPU, "No response from 0x%02X", addresses[i]);
        }
    }
    
    if (!found) {
        ESP_LOGE(TAG_MPU, "MPU6050 not found at any address. Is it connected?");
        return false;
    }
    
    ESP_LOGI(TAG_MPU, "✅ MPU6050 found at address 0x%02X (WHO_AM_I: 0x%02X)", actual_addr, who_am_i);
    
    // Store the actual address for use in other functions
    mpu6050_addr = actual_addr;
    
    // Wake up
    if (!i2c_write_byte(actual_addr, MPU_REG_PWR_MGMT_1, 0x00)) {
        ESP_LOGE(TAG_MPU, "Wake up failed");
        return false;
    }
    vTaskDelay(pdMS_TO_TICKS(10));
    
    // Set sample rate (4Hz - matching CNN model requirements)
    // Sample Rate = Gyroscope Output Rate / (1 + SMPLRT_DIV)
    // For 4Hz: Using 1kHz gyro output / (1 + 249) = 4Hz
    if (!i2c_write_byte(actual_addr, MPU_REG_SMPLRT_DIV, 249)) {
        ESP_LOGE(TAG_MPU, "Sample rate config failed");
        return false;
    }
    
    // Configure DLPF
    if (!i2c_write_byte(actual_addr, MPU_REG_CONFIG, 0x03)) {
        ESP_LOGE(TAG_MPU, "DLPF config failed");
        return false;
    }
    
    // Configure accelerometer (±2g)
    if (!i2c_write_byte(actual_addr, MPU_REG_ACCEL_CONFIG, 0x00)) {
        ESP_LOGE(TAG_MPU, "Accel config failed");
        return false;
    }
    
    // Enable interrupt
    if (!i2c_write_byte(actual_addr, MPU_REG_INT_ENABLE, 0x01)) {
        ESP_LOGE(TAG_MPU, "Interrupt enable failed");
        return false;
    }
    
    // Clear interrupts
    uint8_t int_status;
    i2c_read_byte(actual_addr, MPU_REG_INT_STATUS, &int_status);
    
    ESP_LOGI(TAG_MPU, "MPU6050 initialized successfully");
    return true;
}

static bool mpu6050_read_accel(float *ax, float *ay, float *az) {
    uint8_t data[6];
    if (!i2c_read_bytes(mpu6050_addr, MPU_REG_ACCEL_XOUT_H, data, 6)) {
        return false;
    }
    
    // Clear interrupt
    uint8_t int_status;
    i2c_read_byte(mpu6050_addr, MPU_REG_INT_STATUS, &int_status);
    
    // Convert to signed values
    int16_t raw_ax = (data[0] << 8) | data[1];
    int16_t raw_ay = (data[2] << 8) | data[3];
    int16_t raw_az = (data[4] << 8) | data[5];
    
    // Convert to g units
    *ax = (float)raw_ax / 16384.0f;
    *ay = (float)raw_ay / 16384.0f;
    *az = (float)raw_az / 16384.0f;
    
    return true;
}

// ==================== TIMER FUNCTIONS ====================
static bool gsr_timer_init(void) {
    ESP_LOGI(TAG_GSR, "Setting up GSR timer for %dHz sampling...", EDA_TARGET_HZ);
    
    gptimer_config_t timer_config = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000,
    };
    
    esp_err_t err = gptimer_new_timer(&timer_config, &gsr_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_GSR, "Timer creation failed: %s", esp_err_to_name(err));
        return false;
    }
    
    gptimer_event_callbacks_t cbs = {
        .on_alarm = gsr_timer_callback,
    };
    
    err = gptimer_register_event_callbacks(gsr_timer, &cbs, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_GSR, "Callback registration failed: %s", esp_err_to_name(err));
        return false;
    }
    
    uint64_t period_us = 1000000 / EDA_TARGET_HZ;
    gptimer_alarm_config_t alarm_config = {
        .alarm_count = period_us,
        .reload_count = 0,
        .flags.auto_reload_on_alarm = true,
    };
    
    err = gptimer_set_alarm_action(gsr_timer, &alarm_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_GSR, "Alarm config failed: %s", esp_err_to_name(err));
        return false;
    }
    
    err = gptimer_enable(gsr_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_GSR, "Timer enable failed: %s", esp_err_to_name(err));
        return false;
    }
    
    err = gptimer_start(gsr_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_GSR, "Timer start failed: %s", esp_err_to_name(err));
        return false;
    }
    
    ESP_LOGI(TAG_GSR, "Timer started successfully");
    return true;
}

/* ================= MAX30105 POLLING TIMER (RE-ENABLED - INTERRUPT APPROACH UNRELIABLE) ================= */
// NOTE: Switched from interrupt to polling due to timing issues
// INT pin stays LOW after FIFO clear, NEGEDGE interrupt never fires
// Polling approach matches GSR/TEMP timers (proven working at 4Hz)
static bool max_poll_timer_init(void) {
    ESP_LOGI(TAG_MAX, "Setting up MAX30105 polling timer for %dHz...", BVP_TARGET_HZ);
    
    gptimer_config_t timer_config = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000,
    };
    
    esp_err_t err = gptimer_new_timer(&timer_config, &max_poll_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_MAX, "MAX poll timer creation failed: %s", esp_err_to_name(err));
        return false;
    }
    
    gptimer_event_callbacks_t cbs = {
        .on_alarm = max_poll_timer_callback,
    };
    
    err = gptimer_register_event_callbacks(max_poll_timer, &cbs, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_MAX, "MAX poll callback registration failed: %s", esp_err_to_name(err));
        return false;
    }
    
    // Calculate period for target Hz (4Hz = 250000 microseconds)
    uint64_t period_us = 1000000 / BVP_TARGET_HZ;
    gptimer_alarm_config_t alarm_config = {
        .alarm_count = period_us,
        .reload_count = 0,
        .flags.auto_reload_on_alarm = true,
    };
    
    err = gptimer_set_alarm_action(max_poll_timer, &alarm_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_MAX, "MAX poll alarm config failed: %s", esp_err_to_name(err));
        return false;
    }
    
    err = gptimer_enable(max_poll_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_MAX, "MAX poll timer enable failed: %s", esp_err_to_name(err));
        return false;
    }
    
    err = gptimer_start(max_poll_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_MAX, "MAX poll timer start failed: %s", esp_err_to_name(err));
        return false;
    }
    
    ESP_LOGI(TAG_MAX, "MAX30105 polling timer started at %dHz", BVP_TARGET_HZ);
    return true;
}


/* ================= TEMPERATURE TIMER (MOCK/ESP32 INTERNAL) ================= */
static bool temp_timer_init(void) {
    ESP_LOGI(TAG, "Setting up temperature timer for %dHz sampling...", TEMP_TARGET_HZ);
    
    gptimer_config_t timer_config = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000,
    };
    
    esp_err_t err = gptimer_new_timer(&timer_config, &temp_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Temperature timer creation failed: %s", esp_err_to_name(err));
        return false;
    }
    
    gptimer_event_callbacks_t cbs = {
        .on_alarm = temp_timer_callback,
    };
    
    err = gptimer_register_event_callbacks(temp_timer, &cbs, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Temperature callback registration failed: %s", esp_err_to_name(err));
        return false;
    }
    
    // Calculate period for target Hz (4Hz = 250000 microseconds)
    uint64_t period_us = 1000000 / TEMP_TARGET_HZ;
    gptimer_alarm_config_t alarm_config = {
        .alarm_count = period_us,
        .reload_count = 0,
        .flags.auto_reload_on_alarm = true,
    };
    
    err = gptimer_set_alarm_action(temp_timer, &alarm_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Temperature alarm config failed: %s", esp_err_to_name(err));
        return false;
    }
    
    err = gptimer_enable(temp_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Temperature timer enable failed: %s", esp_err_to_name(err));
        return false;
    }
    
    err = gptimer_start(temp_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Temperature timer start failed: %s", esp_err_to_name(err));
        return false;
    }
    
    ESP_LOGI(TAG, "Temperature timer started at %dHz (mock/ESP32 internal sensor)", TEMP_TARGET_HZ);
    return true;
}

/* ================= ENHANCED SENSOR PROCESSING WITH ML INTEGRATION ================= */
static void enhanced_sensor_processing_task(void *pvParameters) {
    ESP_LOGI(TAG_DATA, "Enhanced Shadow sensor processing with ML integration started");
    
    // Check stack watermark
    UBaseType_t stack_high_water = uxTaskGetStackHighWaterMark(NULL);
    ESP_LOGI(TAG_DATA, "Producer task stack: %u bytes free", stack_high_water * sizeof(StackType_t));
    
    sensor_event_t event;
    TickType_t last_stats = xTaskGetTickCount();
    uint32_t loop_count = 0;
    // Note: last_manual_poll removed - manual polling disabled for interrupt-only sampling
    
    while (1) {
        loop_count++;
        
        // Check stack every 1000 iterations
        if (loop_count % 1000 == 0) {
            stack_high_water = uxTaskGetStackHighWaterMark(NULL);
            if (stack_high_water < 256) {  // Less than 1KB free
                ESP_LOGW(TAG_DATA, "⚠️ Producer stack low: %u bytes free", stack_high_water * sizeof(StackType_t));
            }
        }
        if (xQueueReceive(sensor_event_queue, &event, pdMS_TO_TICKS(100)) == pdTRUE) {
            
            switch (event.type) {
                case SENSOR_EVENT_MAX_DATA_READY:
                case SENSOR_EVENT_MAX_POLL:  // Polling timer event (now primary)
                    if (max_available) {
                        uint32_t ir_value = max30105_read_fifo_sample();
                        if (ir_value > 0) {
                            // Convert to fixed point and add to ring buffer
                            fixed_point_t bvp_fixed = FLOAT_TO_FIXED((float)ir_value);
                            realtime_add_sample_int_isr(SENSOR_BVP, bvp_fixed);
                            bvp_sample_count++;
                            total_samples_collected++;
                            
                            // Update calibration with this single sample (if in progress)
                            if (calibration_get_state() == CAL_STATE_IN_PROGRESS) {
                                float bvp_float = (float)ir_value;
                                calibration_update(&bvp_float, 1, CNN_CHANNEL_BVP);
                            }
                            
                            // Log EVERY BVP sample (now decimated to ~4Hz)
                            ESP_LOGI(TAG_DATA, "[%" PRIu64 "] BVP: %lu → %.2f (#%lu)", 
                                     event.timestamp_us, ir_value, (float)ir_value, bvp_sample_count);
                        }
                    }
                    break;
                
                case SENSOR_EVENT_MPU_DATA_READY:
                    if (mpu_available) {
                        float ax, ay, az;
                        if (mpu6050_read_accel(&ax, &ay, &az)) {
                            // Convert to fixed point and add to ring buffers
                            realtime_add_sample_int_isr(SENSOR_ACC_X, FLOAT_TO_FIXED(ax));
                            realtime_add_sample_int_isr(SENSOR_ACC_Y, FLOAT_TO_FIXED(ay));
                            realtime_add_sample_int_isr(SENSOR_ACC_Z, FLOAT_TO_FIXED(az));
                            acc_sample_count++;
                            total_samples_collected += 3;
                            
                            // Update calibration with this single sample (if in progress)
                            if (calibration_get_state() == CAL_STATE_IN_PROGRESS) {
                                float magnitude = sqrtf(ax*ax + ay*ay + az*az);
                                calibration_update(&magnitude, 1, CNN_CHANNEL_ACC);
                            }
                            
                            // Log EVERY ACC sample (now at ~4Hz from MPU)
                            float magnitude = sqrtf(ax*ax + ay*ay + az*az);
                            ESP_LOGI(TAG_DATA, "[%" PRIu64 "] ACC: %.3f,%.3f,%.3f |%.3f| (#%lu)", 
                                     event.timestamp_us, ax, ay, az, magnitude, acc_sample_count);
                        }
                    }
                    break;
                
                case SENSOR_EVENT_GSR_TIMER:
                    if (gsr_available) {
                        float raw_voltage = gsr_read_voltage();
                        if (raw_voltage >= 0) {
                            // Process voltage with smoothing and range validation
                            float processed_voltage = gsr_process_voltage(raw_voltage);
                            
                            // Convert to fixed point and add to ring buffer
                            realtime_add_sample_int_isr(SENSOR_EDA, FLOAT_TO_FIXED(processed_voltage));
                            eda_sample_count++;
                            total_samples_collected++;
                            
                            // Update calibration with this single sample (if in progress)
                            if (calibration_get_state() == CAL_STATE_IN_PROGRESS) {
                                calibration_update(&processed_voltage, 1, CNN_CHANNEL_EDA);
                            }
                            
                            ESP_LOGI(TAG_DATA, "[%" PRIu64 "] EDA: %.3fV (#%lu)", 
                                     event.timestamp_us, processed_voltage, eda_sample_count);
                        }
                    }
                    break;
                    
                case SENSOR_EVENT_TEMP_TIMER:
                    {
                        float temp = generate_mock_temperature();
                        // Convert to fixed point and add to ring buffer
                        realtime_add_sample_int_isr(SENSOR_TEMP, FLOAT_TO_FIXED(temp));
                        temp_sample_count++;
                        total_samples_collected++;
                        
                        // Update calibration with this single sample (if in progress)
                        if (calibration_get_state() == CAL_STATE_IN_PROGRESS) {
                            calibration_update(&temp, 1, CNN_CHANNEL_TEMP);
                        }
                        
                        ESP_LOGI(TAG_DATA, "[%" PRIu64 "] TEMP: %.2f°C (#%lu)", 
                                 event.timestamp_us, temp, temp_sample_count);
                    }
                    break;
                
                case SENSOR_EVENT_BUTTON_LEFT:
                    // Left button: Start calibration (auto-completes after 2 minutes)
                    ESP_LOGI(TAG_MAIN, "🔘 Left button pressed - calibration control");
                    if (calibration_get_state() == CAL_STATE_IN_PROGRESS) {
                        ESP_LOGI(TAG_MAIN, "⏳ Calibration already in progress (%.1f%% complete)",
                                 calibration_get_progress() * 100.0f);
                        ESP_LOGI(TAG_MAIN, "   Please wait - will auto-complete after collecting enough samples");
                    } else if (calibration_is_calibrated()) {
                        // Device already calibrated - require double-press to re-calibrate
                        if (recalibration_confirm_pending) {
                            // Second press within timeout - start re-calibration
                            ESP_LOGI(TAG_MAIN, "🔄 CONFIRMED: Starting re-calibration");
                            ESP_LOGI(TAG_MAIN, "   ⚠️ This will overwrite existing calibration");
                            calibration_start();
                            recalibration_confirm_pending = false;
                            ESP_LOGI(TAG_MAIN, "🟢 Re-calibration started (2 minutes, auto-completes)");
                            ESP_LOGI(TAG_MAIN, "   ℹ️ Stay calm and still - will finish automatically");
                        } else {
                            // First press - set confirmation flag
                            ESP_LOGI(TAG_MAIN, "⚠️ Device already calibrated");
                            ESP_LOGI(TAG_MAIN, "   Press LEFT button AGAIN within 5 seconds to re-calibrate");
                            ESP_LOGI(TAG_MAIN, "   (This will erase existing calibration)");
                            recalibration_confirm_pending = true;
                            
                            // Start 5-second timeout timer (will be checked in main loop)
                            last_button_left_press = esp_timer_get_time() / 1000;  // Store timestamp in ms
                        }
                    } else {
                        ESP_LOGI(TAG_MAIN, "🟢 Starting calibration (2 minutes, auto-completes)");
                        calibration_start();
                        recalibration_confirm_pending = false;
                        ESP_LOGI(TAG_MAIN, "   ℹ️ Stay calm and still - will finish automatically");
                    }
                    break;
                
                case SENSOR_EVENT_BUTTON_RIGHT:
                    // Right button: Toggle display mode (clock <-> QR code)
                    ESP_LOGI(TAG_MAIN, "🔘 Right button pressed - toggling display");
                    display_toggle_mode(&g_device_info);
                    break;
                
                case SENSOR_EVENT_BUTTON_PRESS:
                    // Legacy button handler - map to right button behavior
                    ESP_LOGI(TAG_MAIN, "🔘 Button pressed - toggling display");
                    display_toggle_mode(&g_device_info);
                    break;
                
                /* REMOVED: SENSOR_EVENT_MAX_POLL - MAX30105 uses interrupts only */
                // Polling timer disabled to prevent double-sampling with interrupts
                
            }
        } else {
            // Timeout - MAX30105 manual polling DISABLED
            // Using interrupt-driven sampling only for precise 3.1Hz rate
            // Manual polling was causing double-sampling and inflated sample rate
        }
        
        // Stats every 10 seconds
        TickType_t now = xTaskGetTickCount();
        if ((now - last_stats) >= pdMS_TO_TICKS(10000)) {
            ESP_LOGI(TAG_MAIN, "📊 Enhanced Samples: BVP:%lu ACC:%lu EDA:%lu TEMP:%lu", 
                     bvp_sample_count, acc_sample_count, eda_sample_count, temp_sample_count);
            ESP_LOGI(TAG_MAIN, "🔄 ML Ready Signals: %u", uxSemaphoreGetCount(g_sensor_system.ml_ready_sem));
            last_stats = now;
        }
        
        // Check re-calibration confirmation timeout (5 seconds)
        if (recalibration_confirm_pending) {
            int64_t current_time_ms = esp_timer_get_time() / 1000;
            int64_t elapsed_ms = current_time_ms - last_button_left_press;
            if (elapsed_ms > 5000) {  // 5 second timeout
                ESP_LOGI(TAG_MAIN, "⏱️ Re-calibration confirmation timeout - cancelled");
                recalibration_confirm_pending = false;
            }
        }
    }
}

// ==================== GPIO SETUP ====================
static esp_err_t setup_gpio_interrupts(void) {
    ESP_LOGI(TAG_MAIN, "Setting up GPIO interrupts...");
    
    esp_err_t err = gpio_install_isr_service(0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG_MAIN, "ISR service install failed: %s", esp_err_to_name(err));
        return err;
    }
    
    // MAX30102: Using polling timer instead of interrupts (interrupt timing issues)
    // Interrupt approach failed: INT pin stays LOW after FIFO clear, NEGEDGE never fires
    if (max_available) {
        ESP_LOGI(TAG_MAX, "MAX30102 using polling timer (interrupt disabled)");
    }
    
    if (mpu_available) {
        gpio_config_t mpu_conf = {
            .pin_bit_mask = (1ULL << MPU_INT_PIN),
            .mode = GPIO_MODE_INPUT,
            .pull_up_en = GPIO_PULLUP_ENABLE,
            .pull_down_en = GPIO_PULLDOWN_DISABLE,
            .intr_type = GPIO_INTR_POSEDGE,
        };
        
        ESP_ERROR_CHECK(gpio_config(&mpu_conf));
        ESP_ERROR_CHECK(gpio_isr_handler_add(MPU_INT_PIN, mpu_interrupt_handler, NULL));
        ESP_LOGI(TAG_MPU, "Interrupt on GPIO%d", MPU_INT_PIN);
    }
    
    // Setup button GPIOs
    // Left button (GPIO 0) - Calibration control
    gpio_config_t btn_left_conf = {
        .pin_bit_mask = (1ULL << BUTTON_LEFT_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE
    };
    ESP_ERROR_CHECK(gpio_config(&btn_left_conf));
    ESP_ERROR_CHECK(gpio_isr_handler_add(BUTTON_LEFT_PIN, button_left_interrupt_handler, NULL));
    ESP_LOGI(TAG_MAIN, "✅ Left button (GPIO %d) configured for calibration control", BUTTON_LEFT_PIN);
    
    // Right button (GPIO 14) - Display toggle
    gpio_config_t btn_right_conf = {
        .pin_bit_mask = (1ULL << BUTTON_RIGHT_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE
    };
    ESP_ERROR_CHECK(gpio_config(&btn_right_conf));
    ESP_ERROR_CHECK(gpio_isr_handler_add(BUTTON_RIGHT_PIN, button_right_interrupt_handler, NULL));
    ESP_LOGI(TAG_MAIN, "✅ Right button (GPIO %d) configured for display toggle", BUTTON_RIGHT_PIN);
    
    return ESP_OK;
}

/* ================= CONSUMER TASK - ML INFERENCE PIPELINE ================= */
void consumer_task(void *param) {
    ESP_LOGI(TAG, "🧠 Consumer started (Core %d)", xPortGetCoreID());
    ESP_LOGI(TAG, "🎯 Real sensor integration: MAX30105 + MPU6050 + GSR + TEMP(mock)");
    ESP_LOGI(TAG, "🧠 CNN Pipeline: Signal preprocessing → CNN inference → BLE");
    
    // Check stack watermark
    UBaseType_t stack_high_water = uxTaskGetStackHighWaterMark(NULL);
    ESP_LOGI(TAG, "Consumer task stack: %u bytes free", stack_high_water * sizeof(StackType_t));
    
    vTaskDelay(pdMS_TO_TICKS(3000)); /* warm-up delay */

    // Allocate CNN input buffer in PSRAM to avoid stack overflow (3.75KB is too large for 8KB stack)
    cnn_input_tensor_t *cnn_input = (cnn_input_tensor_t*)heap_caps_malloc(sizeof(cnn_input_tensor_t), MALLOC_CAP_SPIRAM);
    if (cnn_input == NULL) {
        ESP_LOGE(TAG, "❌ Failed to allocate CNN input buffer in PSRAM");
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "✅ CNN input buffer allocated in PSRAM: %u bytes", sizeof(cnn_input_tensor_t));
    
    cnn_inference_result_t cnn_result;
    
    uint32_t loop_count = 0;

    while (1) {
        loop_count++;
        
        // Check stack every 10 iterations
        if (loop_count % 10 == 0) {
            stack_high_water = uxTaskGetStackHighWaterMark(NULL);
            if (stack_high_water < 512) {  // Less than 2KB free
                ESP_LOGW(TAG, "⚠️ Consumer stack low: %u bytes free", stack_high_water * sizeof(StackType_t));
            }
        }
        // Wait for ML-ready signal from realtime sensor system
        if (xSemaphoreTake(g_sensor_system.ml_ready_sem, portMAX_DELAY) == pdTRUE) {
            uint32_t min_batches = realtime_get_min_batch_count();
            
            // Check if calibration is in progress
            if (calibration_get_state() == CAL_STATE_IN_PROGRESS) {
                ESP_LOGI(TAG, "📊 Calibration in progress - preprocessing for calibration data");
                
                // Run preprocessing to feed calibration system (but skip CNN)
                int preprocess_ret = preprocess_for_cnn(&g_sensor_system, cnn_input);
                
                if (preprocess_ret != 0) {
                    ESP_LOGE(TAG, "❌ Preprocessing failed during calibration (%d)", preprocess_ret);
                }
                
                // Mark batch as processed and skip CNN inference
                realtime_mark_batch_processed(min_batches);
                continue;
            }
            
            total_inferences++;
            ESP_LOGI(TAG, "🔔 CNN Inference #%lu", total_inferences);
            uint32_t t_start = xTaskGetTickCount() * portTICK_PERIOD_MS;

            // Get minimum synchronized batch count across all sensors
            ESP_LOGI(TAG, "🎯 Min synchronized batches: %lu sec", min_batches);

            /* ==================== NEW CNN PIPELINE ==================== */
            
            // Step 1: Preprocess sensor data for CNN input
            uint32_t preprocess_start = xTaskGetTickCount() * portTICK_PERIOD_MS;
            int preprocess_ret = preprocess_for_cnn(&g_sensor_system, cnn_input);
            uint32_t preprocess_time = (xTaskGetTickCount() * portTICK_PERIOD_MS) - preprocess_start;
            
            if (preprocess_ret != 0) {
                ESP_LOGE(TAG, "❌ Preprocessing failed (%d)", preprocess_ret);
                realtime_mark_batch_processed(min_batches);
                continue;
            }
            ESP_LOGI(TAG, "✅ Preprocessing complete in %lu ms", preprocess_time);

            // Step 2: Run CNN inference
            uint32_t cnn_start = xTaskGetTickCount() * portTICK_PERIOD_MS;
            int cnn_ret = cnn_inference_predict(cnn_input, &cnn_result);
            uint32_t cnn_time = (xTaskGetTickCount() * portTICK_PERIOD_MS) - cnn_start;

            // Mark batch as processed
            realtime_mark_batch_processed(min_batches);

            if (cnn_ret != 0 || !cnn_result.success) {
                ESP_LOGE(TAG, "❌ CNN inference failed (ret=%d, success=%d)", cnn_ret, cnn_result.success);
                continue;
            }

            uint32_t total_time = (xTaskGetTickCount() * portTICK_PERIOD_MS) - t_start;

            // Step 3: Process results and update BLE
            float stress_prob = cnn_result.stress_probability;
            const char *stress_class = (stress_prob >= 0.5f) ? "STRESS" : "NORMAL";
            
            ESP_LOGI(TAG, "🎯 CNN Inference Result:");
            ESP_LOGI(TAG, "   Stress Probability: %.1f%%", stress_prob * 100.0f);
            ESP_LOGI(TAG, "   Class: %s (threshold: 0.5)", stress_class);
            ESP_LOGI(TAG, "   Preprocessing: %lu ms", preprocess_time);
            ESP_LOGI(TAG, "   CNN Inference: %lu ms (internal: %lu us)",
                     cnn_time, cnn_result.inference_time_us);
            ESP_LOGI(TAG, "   Total Pipeline: %lu ms", total_time);
            ESP_LOGI(TAG, "   Batch Index: %lu", min_batches);

            // Update FSM with CNN probability (for backward compatibility with BLE service)
            uint32_t now_ms = (uint32_t)(esp_timer_get_time() / 1000);
            bool transition = stress_fsm_process_inference(&g_stress_fsm, stress_prob, now_ms, on_stress_transition);
            
            // Update BLE advertisement
            ble_stress_service_tick();
            
            if (transition) {
                ESP_LOGI(TAG, "🔄 Stress state transition occurred");
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

    /* Extract BVP window for heart rate features */
    fixed_point_t bvp_window[BVP_BUFFER_SIZE];
    int bvp_samples = realtime_extract_window(SENSOR_BVP, bvp_window, BVP_BUFFER_SIZE);

    float bvp_data[BVP_BUFFER_SIZE];
    for (int i = 0; i < bvp_samples; i++) {
        bvp_data[i] = FIXED_TO_FLOAT(bvp_window[i]);
    }

    vTaskDelay(pdMS_TO_TICKS(20)); /* simulate feature computation cost */

    // Extract BVP features (indices 0-7)
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

        result->features[0] = mean;                 // BVP mean
        result->features[1] = std;                  // BVP std
        result->features[2] = minv;                 // BVP min
        result->features[3] = maxv;                 // BVP max
        result->features[4] = mean;                 // BVP median (approx)
        result->features[5] = maxv - minv;          // BVP range
        result->features[6] = std * 1.35f;          // BVP IQR (approx)
        result->features[7] = sum_sq;               // BVP energy
    }

    /* Extract ACC features from real sensor data (indices 8-22) */
    fixed_point_t acc_x_window[ACC_BUFFER_SIZE];
    fixed_point_t acc_y_window[ACC_BUFFER_SIZE];
    fixed_point_t acc_z_window[ACC_BUFFER_SIZE];
    
    int acc_x_samples = realtime_extract_window(SENSOR_ACC_X, acc_x_window, ACC_BUFFER_SIZE);
    int acc_y_samples = realtime_extract_window(SENSOR_ACC_Y, acc_y_window, ACC_BUFFER_SIZE);
    int acc_z_samples = realtime_extract_window(SENSOR_ACC_Z, acc_z_window, ACC_BUFFER_SIZE);
    
    // Convert and compute features for each axis
    for (int axis = 0; axis < 3; axis++) {
        int base = 8 + axis * 5;
        fixed_point_t *axis_data = (axis == 0) ? acc_x_window : 
                                   (axis == 1) ? acc_y_window : acc_z_window;
        int samples = (axis == 0) ? acc_x_samples : 
                     (axis == 1) ? acc_y_samples : acc_z_samples;
        
        if (samples > 0) {
            float sum = 0.f, sum_sq = 0.f;
            float minv = FIXED_TO_FLOAT(axis_data[0]), maxv = FIXED_TO_FLOAT(axis_data[0]);
            
            for (int i = 0; i < samples; i++) {
                float v = FIXED_TO_FLOAT(axis_data[i]);
                sum += v;
                sum_sq += v * v;
                if (v < minv) minv = v;
                if (v > maxv) maxv = v;
            }
            
            float mean = sum / samples;
            float var = (sum_sq / samples) - mean * mean;
            float std = sqrtf(var > 0.f ? var : 0.f);
            
            result->features[base + 0] = mean;
            result->features[base + 1] = std;
            result->features[base + 2] = minv;
            result->features[base + 3] = maxv;
            result->features[base + 4] = sqrtf(sum_sq);  // RMS
        } else {
            // Fallback values if no data available
            result->features[base + 0] = (axis == 0) ? 15.42f : (axis == 1) ? -6.18f : 8.99f;
            result->features[base + 1] = 8.0f;
            result->features[base + 2] = -30.0f;
            result->features[base + 3] = 30.0f;
            result->features[base + 4] = 100.0f;
        }
    }

    /* Extract EDA features from real GSR data (indices 23-26) */
    fixed_point_t eda_window[EDA_BUFFER_SIZE];
    int eda_samples = realtime_extract_window(SENSOR_EDA, eda_window, EDA_BUFFER_SIZE);
    
    if (eda_samples > 0) {
        float sum = 0.f, sum_sq = 0.f;
        float minv = FIXED_TO_FLOAT(eda_window[0]), maxv = FIXED_TO_FLOAT(eda_window[0]);
        
        for (int i = 0; i < eda_samples; i++) {
            float v = FIXED_TO_FLOAT(eda_window[i]);
            sum += v;
            sum_sq += v * v;
            if (v < minv) minv = v;
            if (v > maxv) maxv = v;
        }
        
        float mean = sum / eda_samples;
        float var = (sum_sq / eda_samples) - mean * mean;
        float std = sqrtf(var > 0.f ? var : 0.f);
        
        result->features[23] = mean;         // EDA mean
        result->features[24] = std;          // EDA std  
        result->features[25] = minv;         // EDA min
        result->features[26] = maxv;         // EDA max
    } else {
        // Fallback EDA values
        result->features[23] = 2.08f + (esp_random() % 200 - 100) / 1000.0f;
        result->features[24] = 0.5f + (esp_random() % 300) / 1000.0f;
        result->features[25] = 0.09f + (esp_random() % 100) / 10000.0f;
        result->features[26] = 5.0f + (esp_random() % 1000) / 100.0f;
    }

    /* Extract TEMP features from mock sensor (indices 27-29) */
    fixed_point_t temp_window[TEMP_BUFFER_SIZE];
    int temp_samples = realtime_extract_window(SENSOR_TEMP, temp_window, TEMP_BUFFER_SIZE);
    
    if (temp_samples > 0) {
        float sum = 0.f;
        float minv = FIXED_TO_FLOAT(temp_window[0]), maxv = FIXED_TO_FLOAT(temp_window[0]);
        
        for (int i = 0; i < temp_samples; i++) {
            float v = FIXED_TO_FLOAT(temp_window[i]);
            sum += v;
            if (v < minv) minv = v;
            if (v > maxv) maxv = v;
        }
        
        float mean = sum / temp_samples;
        
        result->features[27] = mean;         // TEMP mean
        result->features[28] = maxv - minv;  // TEMP range
        result->features[29] = mean - 36.5f; // TEMP deviation from normal
    } else {
        // Fallback temperature values
        result->features[27] = 36.5f + (esp_random() % 200 - 100) / 1000.0f;
        result->features[28] = 0.3f + (esp_random() % 200) / 1000.0f;
        result->features[29] = (esp_random() % 400 - 200) / 100.0f;
    }

    result->extraction_time_ms = (xTaskGetTickCount() * portTICK_PERIOD_MS) - start_ms;
    result->success = true;
    result->timestamp = xTaskGetTickCount();

    return 0;
}

// ==================== ENHANCED MAIN APPLICATION ====================
void app_main(void) {
    ESP_LOGI(TAG_MAIN, "========================================");
    ESP_LOGI(TAG_MAIN, "      Shadow Project v4.0 Enhanced     ");
    ESP_LOGI(TAG_MAIN, "    Real-time Stress Detection with    ");
    ESP_LOGI(TAG_MAIN, "      Real Sensor Integration         ");
    ESP_LOGI(TAG_MAIN, "      Author: ashiduDissanayake       ");
    ESP_LOGI(TAG_MAIN, "========================================");
    
    ESP_LOGI(TAG, "🌟 Shadow Real-Time Stress Detection Firmware v4.0");
    ESP_LOGI(TAG, "Initializing subsystems...");

    /* ================= INITIALIZE SHADOW ML PIPELINE ================= */
    
    // Initialize realtime sensor buffer system
    if (realtime_sensor_init() != 0) {
        ESP_LOGE(TAG, "Failed realtime_sensor_init()");
        return;
    }
    
    // Initialize feature extraction workspace
    if (feature_extractor_init(&g_feature_workspace) != 0) {
        ESP_LOGE(TAG, "Failed feature_extractor_init()");
        return;
    }
    
    // Initialize stress FSM
    if (stress_fsm_init(&g_stress_fsm) != 0) {
        ESP_LOGE(TAG, "Failed stress_fsm_init()");
        return;
    }
    
    // Initialize event logging system
    if (event_log_init(&g_event_log) != 0) {
        ESP_LOGE(TAG, "Failed event_log_init()");
        return;
    }
    
    // Initialize BLE stress service
    if (ble_stress_service_init(&g_stress_fsm, &g_event_log) != 0) {
        ESP_LOGE(TAG, "Failed ble_stress_service_init()");
        return;
    }

    // Initialize BLE pairing service (separate service for device management)
    ESP_LOGI(TAG, "🔐 Initializing BLE pairing service...");
    if (ble_pairing_init(g_device_info.device_name) != 0) {  // Use configured device name
        ESP_LOGE(TAG, "❌ Failed to initialize BLE pairing service");
        return;
    }
    ble_pairing_print_status();  // Print pairing status for debugging

    // Initialize TFT display and show clock
    ESP_LOGI(TAG, "🖥️  Initializing display...");
    if (display_init() != ESP_OK) {
        ESP_LOGE(TAG, "❌ Failed to initialize display");
        // Continue without display - not critical
    } else {
        ESP_LOGI(TAG, "✅ Display initialized - showing clock");
        // Show clock immediately
        display_show_clock();
        // QR code will be shown on button press
    }

    ESP_LOGI(TAG, "✅ Shadow ML Pipeline initialized successfully");
    ESP_LOGI(TAG, "Memory usage: %lu bytes", realtime_get_memory_usage());

    /* ================= INITIALIZE CNN INFERENCE ENGINE ================= */
    ESP_LOGI(TAG, "🧠 Initializing CNN inference engine...");
    int cnn_ret = cnn_inference_init(NULL);  // Use default configuration
    if (cnn_ret != 0) {
        ESP_LOGE(TAG, "❌ CNN initialization failed: %d", cnn_ret);
        ESP_LOGE(TAG, "System will continue but ML inference will be disabled");
        // Don't return - allow system to continue without CNN
    } else {
        size_t used_bytes, total_bytes;
        cnn_inference_get_memory_stats(&used_bytes, &total_bytes);
        ESP_LOGI(TAG, "✅ CNN initialized successfully");
        ESP_LOGI(TAG, "   Model: stress_model_quant.tflite");
        ESP_LOGI(TAG, "   Tensor arena: %zu / %zu KB (%.1f%% used)",
                 used_bytes / 1024, total_bytes / 1024,
                 (used_bytes * 100.0f) / total_bytes);
        ESP_LOGI(TAG, "   Free heap after CNN init: %lu bytes", esp_get_free_heap_size());
    }

    /* ================= INITIALIZE CALIBRATION SYSTEM ================= */
    ESP_LOGI(TAG, "🎯 Initializing calibration system...");
    if (calibration_init() != 0) {
        ESP_LOGW(TAG, "⚠️ Calibration initialization failed - will use local normalization");
    } else {
        if (calibration_is_calibrated()) {
            ESP_LOGI(TAG, "✅ Device is calibrated with personalized baseline");
            ESP_LOGI(TAG, "   Press LEFT button to re-calibrate if needed");
        } else {
            ESP_LOGW(TAG, "⚠️ Device NOT calibrated - predictions may be less accurate");
            ESP_LOGI(TAG, "   👉 Press LEFT button when calm to start 2-minute calibration");
        }
    }

    /* ================= INITIALIZE HARDWARE ================= */
    
    // Create event queue for sensor coordination
    sensor_event_queue = xQueueCreate(100, sizeof(sensor_event_t));
    if (!sensor_event_queue) {
        ESP_LOGE(TAG_MAIN, "Failed to create event queue");
        return;
    }
    
    // Initialize I2C bus
    if (i2c_master_init() != ESP_OK) {
        ESP_LOGE(TAG_MAIN, "I2C initialization failed");
        return;
    }
    
    // Scan I2C bus to detect connected devices
    i2c_scan_devices();
    
    // Initialize all sensors
    ESP_LOGI(TAG_MAIN, "Initializing sensors...");
    max_available = max30105_enhanced_init();
    mpu_available = mpu6050_init();
    gsr_available = gsr_adc_init();
    
    ESP_LOGI(TAG_MAIN, "Real sensor status:");
    ESP_LOGI(TAG_MAIN, "  MAX30105 (BVP): %s", max_available ? "✓ ONLINE" : "✗ OFFLINE");
    ESP_LOGI(TAG_MAIN, "  MPU6050 (ACC):  %s", mpu_available ? "✓ ONLINE" : "✗ OFFLINE");
    ESP_LOGI(TAG_MAIN, "  GSR (EDA):      %s", gsr_available ? "✓ ONLINE" : "✗ OFFLINE");
    ESP_LOGI(TAG_MAIN, "  Temperature:    ✓ MOCK ENABLED");
    
    // Setup GPIO interrupts for sensors
    if (setup_gpio_interrupts() != ESP_OK) {
        ESP_LOGE(TAG_MAIN, "GPIO interrupt setup failed");
        return;
    }
    
    // Setup GSR timer if GSR sensor is available
    if (gsr_available && !gsr_timer_init()) {
        ESP_LOGE(TAG_GSR, "GSR timer setup failed, disabling GSR");
        gsr_available = false;
    }
    
    // Setup MAX30105 polling timer if MAX sensor is available
    // Switched from interrupt to polling due to timing issues with INT pin
    if (max_available && !max_poll_timer_init()) {
        ESP_LOGE(TAG_MAX, "MAX poll timer setup failed, disabling MAX sensor");
        max_available = false;
    }
    
    // Setup temperature timer (always enabled - using mock/ESP32 internal sensor)
    if (!temp_timer_init()) {
        ESP_LOGE(TAG, "⚠️  Temperature timer setup failed, temperature data will not be available");
    }
    
    ESP_LOGI(TAG_MAIN, "✅ Hardware initialization complete");
    vTaskDelay(pdMS_TO_TICKS(2000));

    /* ================= CREATE TASK ARCHITECTURE ================= */
    
    ESP_LOGI(TAG_MAIN, "🚀 Starting Shadow real-time architecture...");
    
    // Prime initial BLE advertisement
    ble_stress_service_tick();

    /* Create producer task (Core 0) - Handles sensor data collection */
    BaseType_t producer_result = xTaskCreatePinnedToCore(
        enhanced_sensor_processing_task,
        "shadow_producer",
        4096,                 // Reduced from 8KB to 4KB
        NULL,
        5,                    // High priority for sensor data
        &producer_task_handle,
        0                     // Core 0
    );
    
    if (producer_result != pdPASS) {
        ESP_LOGE(TAG_MAIN, "Failed to create producer task");
        return;
    }

    /* Log heap status before consumer task creation */
    ESP_LOGI(TAG, "Free heap before consumer task: %lu bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "Minimum free heap: %lu bytes", esp_get_minimum_free_heap_size());
    ESP_LOGI(TAG, "Largest free block: %lu bytes", heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    /* Create consumer task (Core 1) - Handles ML inference and BLE */
    BaseType_t consumer_result = xTaskCreatePinnedToCore(
        consumer_task,
        "shadow_consumer", 
        8192,                 // Reduced from 16KB to 8KB (ML tensor arena is in PSRAM)
        NULL,
        3,                    // Lower priority than producer
        &consumer_task_handle,
        1                     // Core 1
    );
    
    if (consumer_result != pdPASS) {
        ESP_LOGE(TAG_MAIN, "Failed to create consumer task");
        return;
    }

    ESP_LOGI(TAG, "🚀 Tasks started: producer(Core0) / consumer(Core1)");
    ESP_LOGI(TAG, "🎯 Real sensor integration: MAX30105 + MPU6050 + GSR + TEMP(mock)");
    ESP_LOGI(TAG, "🧠 CNN Pipeline: Signal preprocessing → CNN inference → BLE");
    ESP_LOGI(TAG, "System ONLINE - Real-time stress detection with CNN active!");

    /* ================= MAIN MONITORING LOOP ================= */
    
    uint32_t monitoring_cycle = 0;
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(30000));  // 30 second intervals
        monitoring_cycle++;
        
        ESP_LOGI(TAG_MAIN, "💓 Shadow System Health Check #%lu", monitoring_cycle);
        ESP_LOGI(TAG_MAIN, "   Free heap: %lu bytes", esp_get_free_heap_size());
        ESP_LOGI(TAG_MAIN, "   Total samples: %lu", total_samples_collected);
        ESP_LOGI(TAG_MAIN, "   ML inferences: %lu", total_inferences);
        ESP_LOGI(TAG_MAIN, "   State transitions: %lu", total_state_transitions);
        
        // Check sensor health
        uint32_t sensor_health_score = 0;
        if (max_available && bvp_sample_count > 0) sensor_health_score += 25;
        if (mpu_available && acc_sample_count > 0) sensor_health_score += 25;
        if (gsr_available && eda_sample_count > 0) sensor_health_score += 25;
        if (temp_sample_count > 0) sensor_health_score += 25;
        
        ESP_LOGI(TAG_MAIN, "   Sensor health: %lu%% (%s)", sensor_health_score,
                 sensor_health_score >= 75 ? "EXCELLENT" :
                 sensor_health_score >= 50 ? "GOOD" : 
                 sensor_health_score >= 25 ? "FAIR" : "POOR");
        
        // Periodic BLE tick to keep advertisements fresh
        ble_stress_service_tick();
        
        if (monitoring_cycle % 10 == 0) {
            ESP_LOGI(TAG, "=== DETAILED SYSTEM STATUS ===");
            realtime_print_status();
            event_log_print_status(&g_event_log);
            ESP_LOGI(TAG, "===============================");
        }
    }
}