/**
 * @file ble_pairing.h
 * @brief BLE Device Pairing Protocol for Shadow Stress Detection System
 * 
 * Features:
 * - Secure device pairing with challenge-response authentication
 * - Multi-device support (up to 3 concurrent paired devices)
 * - Persistent pairing storage in NVS
 * - Device identification and management
 * - Pairing state notifications
 * 
 * BLE Service: Device Pairing Service
 * UUID: 0xB000
 * 
 * Characteristics:
 * - Device Info (0xB001): READ - Device ID, Name, Firmware, Hardware
 * - Pairing State (0xB002): READ | NOTIFY - Current state, paired count
 * - Pairing Control (0xB003): WRITE - Pairing commands
 * - Security Challenge (0xB004): READ | WRITE - Challenge-response auth
 */

#ifndef BLE_PAIRING_H
#define BLE_PAIRING_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_gap_ble_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ==================== CONSTANTS ==================== */

#define PAIRING_SERVICE_UUID        0xB000
#define DEVICE_INFO_CHAR_UUID       0xB001
#define PAIRING_STATE_CHAR_UUID     0xB002
#define PAIRING_CONTROL_CHAR_UUID   0xB003
#define SECURITY_CHALLENGE_CHAR_UUID 0xB004

#define MAX_PAIRED_DEVICES          3
#define DEVICE_NAME_MAX_LEN         32
#define DEVICE_ID_LEN               16  // 128-bit UUID
#define CHALLENGE_LEN               16  // 128-bit challenge
#define FIRMWARE_VERSION_LEN        16
#define HARDWARE_REVISION_LEN       16

/* ==================== ENUMS ==================== */

/**
 * @brief Pairing state machine states
 */
typedef enum {
    PAIRING_STATE_IDLE = 0,        // No pairing in progress
    PAIRING_STATE_ADVERTISING,     // Broadcasting availability
    PAIRING_STATE_CHALLENGE_SENT,  // Challenge sent to client
    PAIRING_STATE_VERIFYING,       // Verifying client response
    PAIRING_STATE_PAIRED,          // Successfully paired
    PAIRING_STATE_REJECTED,        // Pairing rejected
    PAIRING_STATE_TIMEOUT,         // Pairing timeout
    PAIRING_STATE_ERROR            // Error occurred
} pairing_state_t;

/**
 * @brief Pairing control commands (write to 0xB003)
 */
typedef enum {
    PAIRING_CMD_PAIR_REQUEST = 0x01,  // Client requests pairing
    PAIRING_CMD_UNPAIR = 0x02,        // Unpair device
    PAIRING_CMD_PAIR_ACCEPT = 0x03,   // Accept pairing (reserved)
    PAIRING_CMD_PAIR_REJECT = 0x04,   // Reject pairing (reserved)
    PAIRING_CMD_LIST_DEVICES = 0x05,  // Request paired devices list
    PAIRING_CMD_CLEAR_ALL = 0xFF      // Clear all paired devices
} pairing_command_t;

/**
 * @brief Pairing result codes
 */
typedef enum {
    PAIRING_RESULT_SUCCESS = 0x00,
    PAIRING_RESULT_PENDING = 0x01,
    PAIRING_RESULT_FULL = 0x02,          // Max devices reached
    PAIRING_RESULT_ALREADY_PAIRED = 0x03,
    PAIRING_RESULT_AUTH_FAILED = 0x04,
    PAIRING_RESULT_TIMEOUT = 0x05,
    PAIRING_RESULT_ERROR = 0xFF
} pairing_result_t;

/* ==================== STRUCTURES ==================== */

/**
 * @brief Paired device information
 */
typedef struct {
    uint8_t device_id[DEVICE_ID_LEN];      // Client's unique device ID
    char device_name[DEVICE_NAME_MAX_LEN]; // Client's device name
    esp_bd_addr_t bd_addr;                 // Bluetooth MAC address
    uint64_t pair_timestamp;               // Unix timestamp when paired
    uint32_t session_count;                // Number of connection sessions
    bool is_active;                        // Currently connected
    bool is_valid;                         // Entry is valid
} paired_device_t;

/**
 * @brief Device information structure (read from 0xB001)
 */
typedef struct {
    uint8_t device_id[DEVICE_ID_LEN];           // Shadow device UUID
    char device_name[DEVICE_NAME_MAX_LEN];      // Shadow device name
    char firmware_version[FIRMWARE_VERSION_LEN]; // e.g., "v1.0.0"
    char hardware_revision[HARDWARE_REVISION_LEN]; // e.g., "ESP32-S3"
} device_info_t;

/**
 * @brief Pairing state structure (read from 0xB002)
 */
typedef struct {
    pairing_state_t state;           // Current pairing state
    uint8_t paired_count;            // Number of paired devices
    uint8_t max_devices;             // Maximum allowed devices
    uint8_t current_conn_id;         // Current connection ID
    pairing_result_t last_result;    // Last operation result
} pairing_state_info_t;

/**
 * @brief Security challenge structure (0xB004)
 */
typedef struct {
    uint8_t challenge[CHALLENGE_LEN]; // Random challenge
    uint8_t response[CHALLENGE_LEN];  // Expected response (internal)
    uint64_t timestamp;               // Challenge generation time
    bool is_valid;                    // Challenge is active
} security_challenge_t;

/**
 * @brief Pairing context (internal state)
 */
typedef struct {
    device_info_t device_info;
    paired_device_t paired_devices[MAX_PAIRED_DEVICES];
    security_challenge_t current_challenge;
    pairing_state_t state;
    uint16_t gatts_if;
    uint16_t service_handle;
    uint16_t device_info_handle;
    uint16_t pairing_state_handle;
    uint16_t pairing_control_handle;
    uint16_t security_challenge_handle;
    uint16_t conn_id;
    bool initialized;
    bool notifications_enabled;
} pairing_context_t;

/* ==================== PUBLIC API ==================== */

/**
 * @brief Initialize BLE pairing service
 * 
 * @param device_name Custom device name (NULL for default "Shadow-XXXX")
 * @return 0 on success, negative on error
 */
int ble_pairing_init(const char *device_name);

/**
 * @brief Deinitialize BLE pairing service
 */
void ble_pairing_deinit(void);

/**
 * @brief Get current pairing state
 * 
 * @param state_info Output parameter for state information
 * @return 0 on success, negative on error
 */
int ble_pairing_get_state(pairing_state_info_t *state_info);

/**
 * @brief Get device information
 * 
 * @param info Output parameter for device information
 * @return 0 on success, negative on error
 */
int ble_pairing_get_device_info(device_info_t *info);

/**
 * @brief Get list of paired devices
 * 
 * @param devices Output array for paired devices
 * @param max_count Maximum number of devices to retrieve
 * @return Number of paired devices retrieved, negative on error
 */
int ble_pairing_get_paired_devices(paired_device_t *devices, uint8_t max_count);

/**
 * @brief Unpair a specific device
 * 
 * @param device_id Device ID to unpair
 * @return 0 on success, negative on error
 */
int ble_pairing_unpair_device(const uint8_t *device_id);

/**
 * @brief Clear all paired devices
 * 
 * @return 0 on success, negative on error
 */
int ble_pairing_clear_all(void);

/**
 * @brief Check if a device is paired
 * 
 * @param bd_addr Bluetooth address to check
 * @return true if paired, false otherwise
 */
bool ble_pairing_is_device_paired(const esp_bd_addr_t bd_addr);

/**
 * @brief Notify pairing state change to connected clients
 * 
 * @return 0 on success, negative on error
 */
int ble_pairing_notify_state_change(void);

/**
 * @brief Set device name (persistent)
 * 
 * @param name New device name
 * @return 0 on success, negative on error
 */
int ble_pairing_set_device_name(const char *name);

/**
 * @brief Get current device name
 * 
 * @param name Output buffer for device name
 * @param max_len Maximum buffer length
 * @return 0 on success, negative on error
 */
int ble_pairing_get_device_name(char *name, size_t max_len);

/**
 * @brief Print pairing status to console (debug)
 */
void ble_pairing_print_status(void);

#ifdef __cplusplus
}
#endif

#endif // BLE_PAIRING_H
