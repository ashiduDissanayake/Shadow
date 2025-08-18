#include "shadow_ble_api.h"
#include <stdlib.h>
#include <string.h>

// Global context
static shadow_context_t g_context = {0};

// Initialize the library
int shadow_init(const platform_config_t* config) {
    // Check if already initialized
    if (g_context.state != SHADOW_STATE_UNINITIALIZED) {
        return SHADOW_ERR_INVALID_PARAM;
    }
    
    // Initialize context
    memset(&g_context, 0, sizeof(shadow_context_t));
    
    // Copy configuration
    if (config) {
        g_context.config = *config;
    }
    
    // Initialize message queues
    if (shadow_queue_init(&g_context.tx_queue) != 0) {
        return SHADOW_ERR_PLATFORM_ERROR;
    }
    
    if (shadow_queue_init(&g_context.rx_queue) != 0) {
        return SHADOW_ERR_PLATFORM_ERROR;
    }
    
    // Initialize platform-specific components
    if (shadow_ble_init(config) != 0) {
        return SHADOW_ERR_PLATFORM_ERROR;
    }
    
    g_context.state = SHADOW_STATE_INITIALIZED;
    return SHADOW_ERR_SUCCESS;
}

// Clean up the library
void shadow_cleanup(void) {
    // Disconnect if connected
    if (g_context.state == SHADOW_STATE_CONNECTED) {
        shadow_disconnect();
    }
    
    // Stop advertising if advertising
    if (g_context.state == SHADOW_STATE_ADVERTISING) {
        shadow_stop_advertising();
    }
    
    // Clean up platform-specific components
    // (In a real implementation, shadow_ble_cleanup() would be called here)
    
    // Reset context
    memset(&g_context, 0, sizeof(shadow_context_t));
}

// Start advertising
int shadow_start_advertising(void) {
    if (g_context.state != SHADOW_STATE_INITIALIZED) {
        return SHADOW_ERR_INVALID_PARAM;
    }
    
    int result = shadow_ble_start_advertising();
    if (result == 0) {
        g_context.state = SHADOW_STATE_ADVERTISING;
    }
    
    return result;
}

// Stop advertising
int shadow_stop_advertising(void) {
    if (g_context.state != SHADOW_STATE_ADVERTISING) {
        return SHADOW_ERR_INVALID_PARAM;
    }
    
    int result = shadow_ble_stop_advertising();
    if (result == 0) {
        g_context.state = SHADOW_STATE_INITIALIZED;
    }
    
    return result;
}

// Connect to a device
int shadow_connect(const char* device_address) {
    if (g_context.state != SHADOW_STATE_INITIALIZED && 
        g_context.state != SHADOW_STATE_ADVERTISING) {
        return SHADOW_ERR_INVALID_PARAM;
    }
    
    g_context.state = SHADOW_STATE_CONNECTING;
    int result = shadow_ble_connect(device_address);
    
    if (result == 0) {
        g_context.state = SHADOW_STATE_CONNECTED;
        // Call connected callback if registered
        if (g_context.connected_callback) {
            g_context.connected_callback();
        }
    } else {
        g_context.state = SHADOW_STATE_INITIALIZED;
    }
    
    return result;
}

// Disconnect from device
int shadow_disconnect(void) {
    if (g_context.state != SHADOW_STATE_CONNECTED) {
        return SHADOW_ERR_INVALID_PARAM;
    }
    
    int result = shadow_ble_disconnect();
    if (result == 0) {
        g_context.state = SHADOW_STATE_INITIALIZED;
        // Call disconnected callback if registered
        if (g_context.disconnected_callback) {
            g_context.disconnected_callback(0); // 0 for normal disconnection
        }
    }
    
    return result;
}

// Check if connected
int shadow_is_connected(void) {
    return (g_context.state == SHADOW_STATE_CONNECTED) ? 1 : 0;
}

// Send data
int shadow_send_data(const uint8_t* data, uint16_t length) {
    if (!data || length == 0) {
        return SHADOW_ERR_INVALID_PARAM;
    }
    
    if (g_context.state != SHADOW_STATE_CONNECTED) {
        return SHADOW_ERR_NOT_CONNECTED;
    }
    
    // Create a message
    shadow_message_t* message = shadow_msg_create(SHADOW_MSG_TYPE_DATA, data, length);
    if (!message) {
        return SHADOW_ERR_NO_MEMORY;
    }
    
    // Add to transmit queue
    int result = shadow_queue_push(&g_context.tx_queue, message, SHADOW_MSG_PRIORITY_NORMAL);
    if (result != 0) {
        shadow_msg_destroy(message);
        return SHADOW_ERR_QUEUE_FULL;
    }
    
    // Send the message through platform-specific layer
    result = shadow_ble_send_data(data, length);
    if (result != 0) {
        // Remove from queue on failure
        shadow_queue_remove(&g_context.tx_queue, message->message_id);
        shadow_msg_destroy(message);
        return SHADOW_ERR_PLATFORM_ERROR;
    }
    
    return SHADOW_ERR_SUCCESS;
}

// Send control command
int shadow_send_control_command(shadow_cmd_t command, const uint8_t* payload, uint16_t length) {
    if (g_context.state != SHADOW_STATE_CONNECTED) {
        return SHADOW_ERR_NOT_CONNECTED;
    }
    
    // Create a message
    shadow_message_t* message = shadow_msg_create(SHADOW_MSG_TYPE_CONTROL, payload, length);
    if (!message) {
        return SHADOW_ERR_NO_MEMORY;
    }
    
    // Set the command in the payload
    if (message->payload) {
        // In a real implementation, we would encode the command properly
        // For now, we'll just use the first byte for the command
        message->payload[0] = (uint8_t)command;
    }
    
    // Add to transmit queue with high priority
    int result = shadow_queue_push(&g_context.tx_queue, message, SHADOW_MSG_PRIORITY_HIGH);
    if (result != 0) {
        shadow_msg_destroy(message);
        return SHADOW_ERR_QUEUE_FULL;
    }
    
    // Send the message through platform-specific layer
    // For control commands, we might want to wait for acknowledgment
    result = shadow_ble_send_control_command(command, payload, length);
    if (result != 0) {
        // Remove from queue on failure
        shadow_queue_remove(&g_context.tx_queue, message->message_id);
        shadow_msg_destroy(message);
        return SHADOW_ERR_PLATFORM_ERROR;
    }
    
    return SHADOW_ERR_SUCCESS;
}

// Set connected callback
void shadow_set_connected_callback(shadow_connected_callback_t callback) {
    g_context.connected_callback = callback;
}

// Set disconnected callback
void shadow_set_disconnected_callback(shadow_disconnected_callback_t callback) {
    g_context.disconnected_callback = callback;
}

// Set data received callback
void shadow_set_data_received_callback(shadow_data_received_callback_t callback) {
    g_context.data_received_callback = callback;
    
    // Register with platform layer
    shadow_ble_set_data_callback(callback);
}

// Set status received callback
void shadow_set_status_received_callback(shadow_status_received_callback_t callback) {
    g_context.status_received_callback = callback;
    
    // Register with platform layer
    shadow_ble_set_status_callback(callback);
}

// Set error callback
void shadow_set_error_callback(shadow_error_callback_t callback) {
    g_context.error_callback = callback;
}

// Get error string
const char* shadow_get_error_string(shadow_error_t error) {
    switch (error) {
        case SHADOW_ERR_SUCCESS:
            return "Success";
        case SHADOW_ERR_INVALID_PARAM:
            return "Invalid parameter";
        case SHADOW_ERR_NO_MEMORY:
            return "No memory";
        case SHADOW_ERR_NOT_CONNECTED:
            return "Not connected";
        case SHADOW_ERR_TIMEOUT:
            return "Timeout";
        case SHADOW_ERR_CRC_MISMATCH:
            return "CRC mismatch";
        case SHADOW_ERR_QUEUE_FULL:
            return "Queue full";
        case SHADOW_ERR_PLATFORM_ERROR:
            return "Platform error";
        default:
            return "Unknown error";
    }
}

// Get current state
shadow_state_t shadow_get_state(void) {
    return g_context.state;
}