/*
 * Calibration BLE Service
 * ESP32-S3 Shadow Project
 * 
 * Service UUID: C000
 * Characteristics:
 *   - C001: Calibration State (read, notify) - state + progress
 *   - C002: Calibration Control (write) - start/stop/reset commands
 *   - C003: Calibration Stats (read) - per-channel mean/std
 */

#ifndef CALIBRATION_BLE_SERVICE_H
#define CALIBRATION_BLE_SERVICE_H

#include <stdint.h>
#include "esp_gatts_api.h"
#include "calibration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Service UUID: C000
#define CALIBRATION_SERVICE_UUID    0xC000

// Characteristic UUIDs
#define CAL_STATE_CHAR_UUID        0xC001  // Calibration state + progress (read, notify)
#define CAL_CONTROL_CHAR_UUID      0xC002  // Control commands (write)
#define CAL_STATS_CHAR_UUID        0xC003  // Statistics (read)

// Control commands
typedef enum {
    CAL_CMD_START = 0x01,
    CAL_CMD_STOP  = 0x02,
    CAL_CMD_RESET = 0x03
} calibration_command_t;

// State packet format (C001 - 8 bytes)
typedef struct __attribute__((packed)) {
    uint8_t state;              // calibration_state_t
    uint8_t reserved;
    uint16_t progress_percent;  // 0-10000 (0.00% to 100.00%)
    uint32_t remaining_seconds; // Time remaining
} calibration_state_packet_t;

// Stats packet format (C003 - 36 bytes)
typedef struct __attribute__((packed)) {
    float acc_mean;
    float acc_std;
    float bvp_mean;
    float bvp_std;
    float eda_mean;
    float eda_std;
    float temp_mean;
    float temp_std;
    uint32_t total_samples;
} calibration_stats_packet_t;

/* ==================== INITIALIZATION ==================== */

/**
 * Initialize calibration BLE service
 * @return 0 on success, negative on error
 */
int calibration_ble_service_init(void);

/**
 * Register calibration service with GATT server
 * @param gatts_if GATT interface
 * @return 0 on success, negative on error
 */
int calibration_ble_service_register(esp_gatt_if_t gatts_if);

/* ==================== NOTIFICATIONS ==================== */

/**
 * Send calibration state update to connected clients
 * Sends state + progress notification on C001
 * @return 0 on success, negative on error
 */
int calibration_ble_notify_state(void);

/* ==================== HANDLERS ==================== */

/**
 * Handle write to calibration control characteristic (C002)
 * Processes start/stop/reset commands
 * @param data Command data
 * @param len Data length
 * @return 0 on success, negative on error
 */
int calibration_ble_handle_control_write(const uint8_t *data, uint16_t len);

/**
 * Handle read of calibration state characteristic (C001)
 * Returns current state + progress
 * @param data Output buffer
 * @param len Buffer length
 * @return Bytes written, or negative on error
 */
int calibration_ble_handle_state_read(uint8_t *data, uint16_t len);

/**
 * Handle read of calibration stats characteristic (C003)
 * Returns per-channel mean/std
 * @param data Output buffer
 * @param len Buffer length
 * @return Bytes written, or negative on error
 */
int calibration_ble_handle_stats_read(uint8_t *data, uint16_t len);

/* ==================== PERIODIC UPDATE TASK ==================== */

/**
 * Start calibration notification task
 * Sends periodic state updates during calibration
 */
void calibration_ble_start_notify_task(void);

/**
 * Stop calibration notification task
 */
void calibration_ble_stop_notify_task(void);

#ifdef __cplusplus
}
#endif

#endif /* CALIBRATION_BLE_SERVICE_H */
