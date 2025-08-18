#ifndef SHADOW_BLE_DEVICE_H
#define SHADOW_BLE_DEVICE_H

#include "shadow_ble_protocol.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Platform-specific configuration
typedef struct {
    // Common configuration
    uint32_t message_timeout_ms;
    uint8_t max_retries;
    uint32_t connection_interval_min;
    uint32_t connection_interval_max;
    
    // Platform-specific fields would be defined in implementation
} platform_config_t;

// Sensor data structure
typedef struct {
    uint64_t timestamp;
    struct {
        float x, y, z;
    } accelerometer;
    struct {
        uint32_t ir;
        uint32_t red;
    } ppg;
    struct {
        uint16_t raw;
        float voltage;
    } gsr;
} sensor_data_t;

// Function prototypes for device abstraction layer

// Library initialization
int shadow_ble_init(const platform_config_t* config);

// Connection management
int shadow_ble_start_advertising(void);
int shadow_ble_stop_advertising(void);
int shadow_ble_connect(const char* device_address);
int shadow_ble_disconnect(void);
int shadow_ble_is_connected(void);

// Message sending
int shadow_ble_send_data(const uint8_t* data, uint16_t length);
int shadow_ble_send_control_command(shadow_cmd_t command, const uint8_t* payload, uint16_t length);

// Data receiving
int shadow_ble_set_data_callback(void (*callback)(const uint8_t* data, uint16_t length));
int shadow_ble_set_status_callback(void (*callback)(const uint8_t* status, uint16_t length));

// Sensor interface
int shadow_sensor_init(void);
int shadow_sensor_read_data(sensor_data_t* data);
int shadow_sensor_set_sampling_rate(uint32_t rate_hz);

// Power management
int shadow_power_enter_sleep(uint32_t duration_ms);
int shadow_power_wake_up(void);
int shadow_power_optimize_connection_params(void);

#ifdef __cplusplus
}
#endif

#endif // SHADOW_BLE_DEVICE_H