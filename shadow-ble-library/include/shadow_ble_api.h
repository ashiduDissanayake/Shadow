#ifndef SHADOW_BLE_API_H
#define SHADOW_BLE_API_H

#include "shadow_ble_device.h"
#include "shadow_ble_protocol.h"
#include "shadow_ble_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef enum {
    SHADOW_ERR_SUCCESS = 0,
    SHADOW_ERR_INVALID_PARAM = -1,
    SHADOW_ERR_NO_MEMORY = -2,
    SHADOW_ERR_NOT_CONNECTED = -3,
    SHADOW_ERR_TIMEOUT = -4,
    SHADOW_ERR_CRC_MISMATCH = -5,
    SHADOW_ERR_QUEUE_FULL = -6,
    SHADOW_ERR_PLATFORM_ERROR = -7
} shadow_error_t;

// Event callback types
typedef void (*shadow_connected_callback_t)(void);
typedef void (*shadow_disconnected_callback_t)(int reason);
typedef void (*shadow_data_received_callback_t)(const uint8_t* data, uint16_t length);
typedef void (*shadow_status_received_callback_t)(const uint8_t* status, uint16_t length);
typedef void (*shadow_error_callback_t)(shadow_error_t error);

// Library state
typedef enum {
    SHADOW_STATE_UNINITIALIZED = 0,
    SHADOW_STATE_INITIALIZED,
    SHADOW_STATE_ADVERTISING,
    SHADOW_STATE_CONNECTING,
    SHADOW_STATE_CONNECTED,
    SHADOW_STATE_DISCONNECTED
} shadow_state_t;

// Library context
typedef struct {
    shadow_state_t state;
    platform_config_t config;
    shadow_message_queue_t tx_queue;
    shadow_message_queue_t rx_queue;
    
    // Callbacks
    shadow_connected_callback_t connected_callback;
    shadow_disconnected_callback_t disconnected_callback;
    shadow_data_received_callback_t data_received_callback;
    shadow_status_received_callback_t status_received_callback;
    shadow_error_callback_t error_callback;
} shadow_context_t;

// Function prototypes

// Library initialization and cleanup
int shadow_init(const platform_config_t* config);
void shadow_cleanup(void);

// Connection management
int shadow_start_advertising(void);
int shadow_stop_advertising(void);
int shadow_connect(const char* device_address);
int shadow_disconnect(void);
int shadow_is_connected(void);

// Message sending
int shadow_send_data(const uint8_t* data, uint16_t length);
int shadow_send_control_command(shadow_cmd_t command, const uint8_t* payload, uint16_t length);

// Event callback registration
void shadow_set_connected_callback(shadow_connected_callback_t callback);
void shadow_set_disconnected_callback(shadow_disconnected_callback_t callback);
void shadow_set_data_received_callback(shadow_data_received_callback_t callback);
void shadow_set_status_received_callback(shadow_status_received_callback_t callback);
void shadow_set_error_callback(shadow_error_callback_t callback);

// Utility functions
const char* shadow_get_error_string(shadow_error_t error);
shadow_state_t shadow_get_state(void);

#ifdef __cplusplus
}
#endif

#endif // SHADOW_BLE_API_H