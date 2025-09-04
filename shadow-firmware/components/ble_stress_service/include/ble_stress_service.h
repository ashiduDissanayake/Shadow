/*
 * ESP32-S3 BLE Stress Monitor Service
 * Part of the ESP32-Host Stress Monitor Communication System
 * 
 * This module implements the BLE GATT service for communication between
 * the ESP32 stress monitor and the host device.
 * 
 * Service Architecture:
 * - Custom "StressMonitor" BLE Service
 * - Advertising with FSM state and sequence number
 * - GATT characteristics for event data exchange
 * - Connection-based detailed data transfer
 * - Acknowledgment mechanism for reliable delivery
 */

#ifndef BLE_STRESS_SERVICE_H
#define BLE_STRESS_SERVICE_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_log.h"
#include "esp_bt.h"
#include "esp_gap_ble_api.h"
#include "esp_gatts_api.h"
#include "esp_bt_main.h"
#include "esp_gatt_common_api.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "stress_fsm.h"
#include "event_log.h"

// BLE Configuration
#define BLE_DEVICE_NAME                 "Shadow"
#define BLE_ADV_INTERVAL_MIN            160     // 100ms (160 * 0.625ms)
#define BLE_ADV_INTERVAL_MAX            320     // 200ms (320 * 0.625ms)
#define BLE_MAX_CONNECTION_INTERVAL     20      // 25ms (20 * 1.25ms)
#define BLE_MIN_CONNECTION_INTERVAL     6       // 7.5ms (6 * 1.25ms)

// Service and Characteristic UUIDs (16-bit for efficiency)
#define STRESS_SERVICE_UUID             0x1800  // Custom service UUID
#define FSM_STATE_CHAR_UUID             0x1801  // FSM State and Sequence (NOTIFY)
#define EVENT_DATA_CHAR_UUID            0x1802  // Event Log Data (INDICATE)
#define EVENT_ACK_CHAR_UUID             0x1803  // Event Acknowledge (WRITE)
#define CONTROL_POINT_CHAR_UUID         0x1804  // Control Point (WRITE)

// GATT Attribute Handles
enum {
    STRESS_SERVICE_HANDLE = 0,
    
    // FSM State and Sequence Characteristic (NOTIFY)
    FSM_STATE_CHAR_HANDLE,
    FSM_STATE_VAL_HANDLE,
    FSM_STATE_CFG_HANDLE,
    
    // Event Log Data Characteristic (INDICATE)
    EVENT_DATA_CHAR_HANDLE,
    EVENT_DATA_VAL_HANDLE,
    EVENT_DATA_CFG_HANDLE,
    
    // Event Acknowledge Characteristic (WRITE)
    EVENT_ACK_CHAR_HANDLE,
    EVENT_ACK_VAL_HANDLE,
    
    // Control Point Characteristic (WRITE)
    CONTROL_POINT_CHAR_HANDLE,
    CONTROL_POINT_VAL_HANDLE,
    
    STRESS_SERVICE_HANDLE_COUNT
};

// Advertisement Data Structure (fits within 31-byte limit)
typedef struct __attribute__((packed)) {
    uint8_t fsm_state;              // Current confirmed FSM state
    uint8_t sequence_number;        // Latest event sequence number
    uint16_t battery_mv;            // Battery voltage in millivolts
    uint8_t sensor_quality;         // Sensor quality indicator (0-100)
} ble_adv_payload_t;

// Control Point Commands
typedef enum {
    CTRL_CMD_REPLAY_FROM_SEQUENCE = 0x01,   // Request replay from specific sequence
    CTRL_CMD_GET_SYSTEM_STATUS = 0x02,      // Request system status
    CTRL_CMD_RESET_EVENT_LOG = 0x03,        // Reset event log (for testing)
    CTRL_CMD_ACKNOWLEDGE_TRANSITION = 0x04  // Acknowledge transition sequence (resets counter)
} ble_control_command_t;

// BLE Service Context
typedef struct {
    esp_gatts_cb_t gatts_cb;                    // GATT server callback
    uint16_t gatts_if;                          // GATT interface
    uint16_t service_handle;                    // Service handle
    uint16_t char_handles[STRESS_SERVICE_HANDLE_COUNT]; // Characteristic handles
    uint16_t conn_id;                           // Connection ID
    bool connected;                             // Connection status
    bool notifications_enabled;                 // FSM state notifications enabled
    bool indications_enabled;                   // Event data indications enabled
    SemaphoreHandle_t mutex;                    // Thread safety
    bool initialized;                           // Initialization flag
} ble_stress_service_t;

// BLE Service Statistics
typedef struct {
    uint32_t advertisements_sent;               // Total advertisements sent
    uint32_t connections_established;           // Total connections made
    uint32_t notifications_sent;                // FSM state notifications sent
    uint32_t indications_sent;                  // Event data indications sent
    uint32_t acknowledgments_received;          // Event acknowledgments received
    uint32_t control_commands_received;         // Control commands received
} ble_service_stats_t;

// === CORE BLE SERVICE FUNCTIONS ===

/**
 * Initialize the BLE stress monitoring service
 * 
 * @param fsm_ctx Stress FSM context for state monitoring
 * @param event_ctx Event log context for data access
 * @return 0 on success, -1 on error
 */
int ble_stress_service_init(stress_fsm_context_t *fsm_ctx, event_log_context_t *event_ctx);

/**
 * Deinitialize the BLE stress monitoring service
 */
void ble_stress_service_deinit(void);

/**
 * Start BLE advertising with current FSM state and sequence
 * 
 * @param battery_mv Current battery voltage
 * @param sensor_quality Current sensor quality (0-100)
 * @return 0 on success, -1 on error
 */
int ble_stress_service_start_advertising(void);

/**
 * Stop BLE advertising
 * 
 * @return 0 on success, -1 on error
 */
int ble_stress_service_stop_advertising(void);

/**
 * Update advertisement data with new FSM state and sequence
 * This should be called whenever the FSM state changes or new events are logged
 * 
 * @param battery_mv Current battery voltage
 * @param sensor_quality Current sensor quality (0-100)
 * @return 0 on success, -1 on error
 */
int ble_stress_service_update_advertisement(uint16_t battery_mv, uint8_t sensor_quality);

/**
 * Send FSM state notification to connected client
 * 
 * @return 0 on success, -1 on error
 */
int ble_stress_service_notify_fsm_state(void);

/**
 * Send event data indication to connected client
 * 
 * @param event Event data to send
 * @return 0 on success, -1 on error
 */
int ble_stress_service_indicate_event_data(const stress_event_t *event);

/**
 * Send multiple events to connected client
 * Used for event replay during synchronization
 * 
 * @param events Array of events to send
 * @param count Number of events to send
 * @return Number of events successfully sent
 */
int ble_stress_service_send_event_batch(const stress_event_t *events, uint8_t count);

// === SERVICE STATUS AND CONTROL ===

/**
 * Check if a client is currently connected
 * 
 * @return true if client is connected
 */
bool ble_stress_service_is_connected(void);

/**
 * Check if notifications are enabled by the client
 * 
 * @return true if notifications are enabled
 */
bool ble_stress_service_notifications_enabled(void);

/**
 * Check if indications are enabled by the client
 * 
 * @return true if indications are enabled
 */
bool ble_stress_service_indications_enabled(void);

/**
 * Get BLE service statistics
 * 
 * @param stats Output buffer for statistics
 * @return true if statistics retrieved successfully
 */
bool ble_stress_service_get_statistics(ble_service_stats_t *stats);

/**
 * Disconnect current client
 * 
 * @return 0 on success, -1 on error
 */
int ble_stress_service_disconnect_client(void);

// === UTILITY FUNCTIONS ===

/**
 * Print BLE service status for debugging
 */
void ble_stress_service_print_status(void);

/**
 * Reset BLE service statistics
 */
void ble_stress_service_reset_statistics(void);

/**
 * Enable/disable verbose BLE logging
 * 
 * @param enable true to enable verbose logging
 */
void ble_stress_service_set_verbose_logging(bool enable);

#endif // BLE_STRESS_SERVICE_H
