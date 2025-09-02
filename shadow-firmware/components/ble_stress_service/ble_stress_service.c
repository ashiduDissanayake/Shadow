/*
 * ESP32-S3 BLE Stress Monitor Service Implementation
 * 
 * Implements the complete BLE GATT service for stress monitor communication
 * including advertising, GATT server, and data exchange protocols.
 */

#include "ble_stress_service.h"
#include <string.h>
#include "esp_timer.h"
#include "nvs_flash.h"
#include "esp_bt_device.h"

static const char *TAG = "BLEStressService";

// Global service context
static ble_stress_service_t g_service = {0};
static stress_fsm_context_t *g_fsm_ctx = NULL;
static event_log_context_t *g_event_ctx = NULL;
static ble_service_stats_t g_stats = {0};
static bool g_verbose_logging = false;

// === FORWARD DECLARATIONS ===
static void handle_write_event(esp_ble_gatts_cb_param_t *param);
static void handle_control_command(uint8_t *data, uint16_t len);
static void send_system_status(void);

// === PRIVATE HELPER FUNCTIONS ===

/**
 * Get current time in milliseconds
 */
static uint32_t get_current_time_ms(void) {
    return (uint32_t)(esp_timer_get_time() / 1000);
}

/**
 * Prepare advertisement payload with current state and data
 */
static void prepare_advertisement_payload(ble_adv_payload_t *payload, 
                                        uint16_t battery_mv, 
                                        uint8_t sensor_quality) {
    if (!payload) return;
    
    // Get current FSM state
    payload->fsm_state = (uint8_t)stress_fsm_get_current_state(g_fsm_ctx);
    
    // Get latest sequence number
    payload->sequence_number = event_log_get_latest_sequence(g_event_ctx);
    if (payload->sequence_number == EVENT_LOG_INVALID_SEQUENCE) {
        payload->sequence_number = 0; // No events logged yet
    }
    
    payload->battery_mv = battery_mv;
    payload->sensor_quality = sensor_quality;
}

/**
 * GAP event handler
 */
static void gap_event_handler(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param) {
    switch (event) {
        case ESP_GAP_BLE_ADV_DATA_SET_COMPLETE_EVT:
            if (g_verbose_logging) {
                ESP_LOGI(TAG, "Advertisement data set complete");
            }
            break;
            
        case ESP_GAP_BLE_ADV_START_COMPLETE_EVT:
            if (param->adv_start_cmpl.status == ESP_BT_STATUS_SUCCESS) {
                ESP_LOGI(TAG, "🔊 BLE advertising started");
                g_stats.advertisements_sent++;
            } else {
                ESP_LOGE(TAG, "Failed to start advertising: %d", param->adv_start_cmpl.status);
            }
            break;
            
        case ESP_GAP_BLE_ADV_STOP_COMPLETE_EVT:
            ESP_LOGI(TAG, "🔇 BLE advertising stopped");
            break;
            
        default:
            if (g_verbose_logging) {
                ESP_LOGD(TAG, "GAP event: %d", event);
            }
            break;
    }
}

/**
 * GATTS event handler
 */
static void gatts_event_handler(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if, 
                               esp_ble_gatts_cb_param_t *param) {
    
    if (event == ESP_GATTS_REG_EVT) {
        if (param->reg.status == ESP_GATT_OK) {
            g_service.gatts_if = gatts_if;
            ESP_LOGI(TAG, "GATT server registered");
        } else {
            ESP_LOGE(TAG, "GATT server registration failed: %d", param->reg.status);
        }
        return;
    }
    
    if (gatts_if != g_service.gatts_if) {
        return; // Not our interface
    }
    
    switch (event) {
        case ESP_GATTS_CREATE_EVT:
            if (param->create.status == ESP_GATT_OK) {
                g_service.service_handle = param->create.service_handle;
                ESP_LOGI(TAG, "Service created with handle %d", g_service.service_handle);
                
                // Start the service
                esp_ble_gatts_start_service(g_service.service_handle);
            }
            break;
            
        case ESP_GATTS_START_EVT:
            ESP_LOGI(TAG, "✅ GATT service started");
            break;
            
        case ESP_GATTS_CONNECT_EVT:
            g_service.connected = true;
            g_service.conn_id = param->connect.conn_id;
            g_stats.connections_established++;
            ESP_LOGI(TAG, "🔗 Client connected (conn_id: %d)", g_service.conn_id);
            
            // Stop advertising when connected
            esp_ble_gap_stop_advertising();
            break;
            
        case ESP_GATTS_DISCONNECT_EVT:
            g_service.connected = false;
            g_service.notifications_enabled = false;
            g_service.indications_enabled = false;
            ESP_LOGI(TAG, "🔌 Client disconnected");
            
            // Restart advertising after disconnection
            uint16_t battery_mv = 3300; // Default value - should be read from ADC
            uint8_t sensor_quality = 85; // Default value - should be calculated
            ble_stress_service_start_advertising(battery_mv, sensor_quality);
            break;
            
        case ESP_GATTS_WRITE_EVT:
            handle_write_event(param);
            break;
            
        case ESP_GATTS_CONF_EVT:
            if (g_verbose_logging) {
                ESP_LOGI(TAG, "Indication confirmed by client");
            }
            break;
            
        default:
            if (g_verbose_logging) {
                ESP_LOGD(TAG, "GATTS event: %d", event);
            }
            break;
    }
}

/**
 * Handle GATT write events (acknowledgments and control commands)
 */
static void handle_write_event(esp_ble_gatts_cb_param_t *param) {
    if (param->write.handle == g_service.char_handles[EVENT_ACK_VAL_HANDLE]) {
        // Event acknowledgment received
        if (param->write.len == 1) {
            uint8_t ack_sequence = param->write.value[0];
            if (event_log_acknowledge_sequence(g_event_ctx, ack_sequence)) {
                g_stats.acknowledgments_received++;
                ESP_LOGI(TAG, "✅ Host acknowledged sequence %d", ack_sequence);
            }
        }
        
    } else if (param->write.handle == g_service.char_handles[CONTROL_POINT_VAL_HANDLE]) {
        // Control point command received
        handle_control_command(param->write.value, param->write.len);
        g_stats.control_commands_received++;
        
    } else if (param->write.handle == g_service.char_handles[FSM_STATE_CFG_HANDLE]) {
        // FSM state notification configuration
        if (param->write.len == 2) {
            uint16_t config = (param->write.value[1] << 8) | param->write.value[0];
            g_service.notifications_enabled = (config & 0x0001) != 0;
            ESP_LOGI(TAG, "FSM notifications %s", g_service.notifications_enabled ? "enabled" : "disabled");
        }
        
    } else if (param->write.handle == g_service.char_handles[EVENT_DATA_CFG_HANDLE]) {
        // Event data indication configuration
        if (param->write.len == 2) {
            uint16_t config = (param->write.value[1] << 8) | param->write.value[0];
            g_service.indications_enabled = (config & 0x0002) != 0;
            ESP_LOGI(TAG, "Event indications %s", g_service.indications_enabled ? "enabled" : "disabled");
        }
    }
    
    // Send response
    esp_ble_gatts_send_response(g_service.gatts_if, param->write.conn_id, 
                               param->write.trans_id, ESP_GATT_OK, NULL);
}

/**
 * Handle control point commands
 */
static void handle_control_command(uint8_t *data, uint16_t len) {
    if (!data || len == 0) return;
    
    ble_control_command_t cmd = (ble_control_command_t)data[0];
    
    switch (cmd) {
        case CTRL_CMD_REPLAY_FROM_SEQUENCE:
            if (len >= 2) {
                uint8_t start_sequence = data[1];
                ESP_LOGI(TAG, "🔄 Host requested replay from sequence %d", start_sequence);
                
                // Get events starting from requested sequence
                stress_event_t events[16]; // Reasonable batch size
                uint8_t count = event_log_get_events_from_sequence(g_event_ctx, start_sequence, 
                                                                  events, 16);
                
                if (count > 0) {
                    ble_stress_service_send_event_batch(events, count);
                    ESP_LOGI(TAG, "Sent %d events for replay", count);
                } else {
                    ESP_LOGI(TAG, "No events found for replay from sequence %d", start_sequence);
                }
            }
            break;
            
        case CTRL_CMD_GET_SYSTEM_STATUS:
            ESP_LOGI(TAG, "📊 Host requested system status");
            send_system_status();
            break;
            
        case CTRL_CMD_RESET_EVENT_LOG:
            ESP_LOGW(TAG, "⚠️  Host requested event log reset");
            event_log_reset(g_event_ctx);
            break;
            
        default:
            ESP_LOGW(TAG, "Unknown control command: 0x%02X", cmd);
            break;
    }
}

/**
 * Send system status to connected client
 */
static void send_system_status(void) {
    // Create a status payload
    struct {
        uint8_t fsm_state;
        uint8_t current_sequence;
        uint8_t events_available;
        uint8_t events_unacknowledged;
        uint16_t battery_mv;
        uint8_t sensor_quality;
        uint32_t uptime_ms;
    } __attribute__((packed)) status;
    
    status.fsm_state = (uint8_t)stress_fsm_get_current_state(g_fsm_ctx);
    status.current_sequence = event_log_get_latest_sequence(g_event_ctx);
    
    event_log_stats_t event_stats;
    if (event_log_get_statistics(g_event_ctx, &event_stats)) {
        status.events_available = event_stats.events_available;
        status.events_unacknowledged = event_stats.events_unacknowledged;
    } else {
        status.events_available = 0;
        status.events_unacknowledged = 0;
    }
    
    status.battery_mv = 3300; // Should be read from ADC
    status.sensor_quality = 85; // Should be calculated
    status.uptime_ms = get_current_time_ms();
    
    // Send as indication
    if (g_service.connected && g_service.indications_enabled) {
        esp_ble_gatts_send_indicate(g_service.gatts_if, g_service.conn_id,
                                   g_service.char_handles[EVENT_DATA_VAL_HANDLE],
                                   sizeof(status), (uint8_t*)&status, false);
    }
}

// === PUBLIC API IMPLEMENTATION ===

int ble_stress_service_init(stress_fsm_context_t *fsm_ctx, event_log_context_t *event_ctx) {
    if (!fsm_ctx || !event_ctx) {
        ESP_LOGE(TAG, "Invalid context pointers");
        return -1;
    }
    
    g_fsm_ctx = fsm_ctx;
    g_event_ctx = event_ctx;
    
    // Clear service context
    memset(&g_service, 0, sizeof(ble_stress_service_t));
    memset(&g_stats, 0, sizeof(ble_service_stats_t));
    
    // Create mutex for thread safety
    g_service.mutex = xSemaphoreCreateMutex();
    if (g_service.mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create BLE service mutex");
        return -1;
    }
    
    // Initialize NVS for BLE
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // Initialize Bluetooth
    ESP_ERROR_CHECK(esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT));
    
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ret = esp_bt_controller_init(&bt_cfg);
    if (ret) {
        ESP_LOGE(TAG, "Initialize controller failed: %s", esp_err_to_name(ret));
        return -1;
    }
    
    ret = esp_bt_controller_enable(ESP_BT_MODE_BLE);
    if (ret) {
        ESP_LOGE(TAG, "Enable controller failed: %s", esp_err_to_name(ret));
        return -1;
    }
    
    ret = esp_bluedroid_init();
    if (ret) {
        ESP_LOGE(TAG, "Init bluetooth failed: %s", esp_err_to_name(ret));
        return -1;
    }
    
    ret = esp_bluedroid_enable();
    if (ret) {
        ESP_LOGE(TAG, "Enable bluetooth failed: %s", esp_err_to_name(ret));
        return -1;
    }
    
    // Register GAP and GATTS callbacks
    esp_ble_gap_register_callback(gap_event_handler);
    esp_ble_gatts_register_callback(gatts_event_handler);
    
    // Register GATT application
    esp_ble_gatts_app_register(0);
    
    g_service.initialized = true;
    ESP_LOGI(TAG, "✅ BLE Stress Service initialized");
    
    return 0;
}

void ble_stress_service_deinit(void) {
    if (!g_service.initialized) return;
    
    // Stop advertising if active
    esp_ble_gap_stop_advertising();
    
    // Disconnect if connected
    if (g_service.connected) {
        esp_ble_gatts_close(g_service.gatts_if, g_service.conn_id);
    }
    
    // Cleanup Bluetooth
    esp_bluedroid_disable();
    esp_bluedroid_deinit();
    esp_bt_controller_disable();
    esp_bt_controller_deinit();
    
    if (g_service.mutex != NULL) {
        vSemaphoreDelete(g_service.mutex);
        g_service.mutex = NULL;
    }
    
    g_service.initialized = false;
    ESP_LOGI(TAG, "BLE Stress Service deinitialized");
}

int ble_stress_service_start_advertising(uint16_t battery_mv, uint8_t sensor_quality) {
    if (!g_service.initialized) {
        ESP_LOGE(TAG, "Service not initialized");
        return -1;
    }
    
    // Prepare advertisement payload
    ble_adv_payload_t payload;
    prepare_advertisement_payload(&payload, battery_mv, sensor_quality);
    
    // Configure advertisement data
    esp_ble_adv_data_t adv_data = {
        .set_scan_rsp = false,
        .include_name = true,
        .include_txpower = true,
        .min_interval = BLE_ADV_INTERVAL_MIN,
        .max_interval = BLE_ADV_INTERVAL_MAX,
        .appearance = 0x00,
        .manufacturer_len = sizeof(ble_adv_payload_t),
        .p_manufacturer_data = (uint8_t*)&payload,
        .service_data_len = 0,
        .p_service_data = NULL,
        .service_uuid_len = 2,
        .p_service_uuid = (uint8_t[]){STRESS_SERVICE_UUID & 0xFF, (STRESS_SERVICE_UUID >> 8) & 0xFF},
        .flag = (ESP_BLE_ADV_FLAG_GEN_DISC | ESP_BLE_ADV_FLAG_BREDR_NOT_SPT),
    };
    
    // Set advertisement parameters
    esp_ble_adv_params_t adv_params = {
        .adv_int_min = BLE_ADV_INTERVAL_MIN,
        .adv_int_max = BLE_ADV_INTERVAL_MAX,
        .adv_type = ADV_TYPE_IND,
        .own_addr_type = BLE_ADDR_TYPE_PUBLIC,
        .channel_map = ADV_CHNL_ALL,
        .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
    };
    
    // Set device name
    esp_ble_gap_set_device_name(BLE_DEVICE_NAME);
    
    // Configure and start advertising
    esp_ble_gap_config_adv_data(&adv_data);
    esp_ble_gap_start_advertising(&adv_params);
    
    ESP_LOGI(TAG, "🔊 Starting advertisement: FSM=%s, Seq=%d, Battery=%dmV", 
             stress_fsm_state_to_string((stress_fsm_state_t)payload.fsm_state),
             payload.sequence_number, payload.battery_mv);
    
    return 0;
}

int ble_stress_service_stop_advertising(void) {
    return esp_ble_gap_stop_advertising();
}

int ble_stress_service_update_advertisement(uint16_t battery_mv, uint8_t sensor_quality) {
    if (!g_service.initialized) return -1;
    
    // If we're connected, no need to update advertisement
    if (g_service.connected) return 0;
    
    // Stop current advertising
    esp_ble_gap_stop_advertising();
    
    // Start with new data
    return ble_stress_service_start_advertising(battery_mv, sensor_quality);
}

int ble_stress_service_notify_fsm_state(void) {
    if (!g_service.connected || !g_service.notifications_enabled) {
        return -1;
    }
    
    // Prepare FSM state notification data
    struct {
        uint8_t fsm_state;
        uint8_t sequence_number;
    } __attribute__((packed)) state_data;
    
    state_data.fsm_state = (uint8_t)stress_fsm_get_current_state(g_fsm_ctx);
    state_data.sequence_number = event_log_get_latest_sequence(g_event_ctx);
    
    esp_err_t ret = esp_ble_gatts_send_indicate(g_service.gatts_if, g_service.conn_id,
                                               g_service.char_handles[FSM_STATE_VAL_HANDLE],
                                               sizeof(state_data), (uint8_t*)&state_data, false);
    
    if (ret == ESP_OK) {
        g_stats.notifications_sent++;
        if (g_verbose_logging) {
            ESP_LOGI(TAG, "📢 FSM state notification sent");
        }
    }
    
    return (ret == ESP_OK) ? 0 : -1;
}

int ble_stress_service_indicate_event_data(const stress_event_t *event) {
    if (!g_service.connected || !g_service.indications_enabled || !event) {
        return -1;
    }
    
    esp_err_t ret = esp_ble_gatts_send_indicate(g_service.gatts_if, g_service.conn_id,
                                               g_service.char_handles[EVENT_DATA_VAL_HANDLE],
                                               sizeof(stress_event_t), (uint8_t*)event, false);
    
    if (ret == ESP_OK) {
        g_stats.indications_sent++;
        ESP_LOGI(TAG, "📤 Event indication sent: seq=%d, state=%s", 
                 event->sequence_number, 
                 stress_fsm_state_to_string(event->new_state));
    }
    
    return (ret == ESP_OK) ? 0 : -1;
}

int ble_stress_service_send_event_batch(const stress_event_t *events, uint8_t count) {
    if (!events || count == 0) return 0;
    
    int sent = 0;
    for (uint8_t i = 0; i < count; i++) {
        if (ble_stress_service_indicate_event_data(&events[i]) == 0) {
            sent++;
            // Add small delay between indications
            vTaskDelay(pdMS_TO_TICKS(10));
        }
    }
    
    return sent;
}

bool ble_stress_service_is_connected(void) {
    return g_service.connected;
}

bool ble_stress_service_notifications_enabled(void) {
    return g_service.notifications_enabled;
}

bool ble_stress_service_indications_enabled(void) {
    return g_service.indications_enabled;
}

bool ble_stress_service_get_statistics(ble_service_stats_t *stats) {
    if (!stats) return false;
    
    if (xSemaphoreTake(g_service.mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        *stats = g_stats;
        xSemaphoreGive(g_service.mutex);
        return true;
    }
    
    return false;
}

int ble_stress_service_disconnect_client(void) {
    if (!g_service.connected) return -1;
    
    return esp_ble_gatts_close(g_service.gatts_if, g_service.conn_id);
}

void ble_stress_service_print_status(void) {
    ESP_LOGI(TAG, "=== BLE Stress Service Status ===");
    ESP_LOGI(TAG, "Initialized: %s", g_service.initialized ? "YES" : "NO");
    ESP_LOGI(TAG, "Connected: %s", g_service.connected ? "YES" : "NO");
    ESP_LOGI(TAG, "Notifications enabled: %s", g_service.notifications_enabled ? "YES" : "NO");
    ESP_LOGI(TAG, "Indications enabled: %s", g_service.indications_enabled ? "YES" : "NO");
    ESP_LOGI(TAG, "Advertisements sent: %lu", g_stats.advertisements_sent);
    ESP_LOGI(TAG, "Connections established: %lu", g_stats.connections_established);
    ESP_LOGI(TAG, "Notifications sent: %lu", g_stats.notifications_sent);
    ESP_LOGI(TAG, "Indications sent: %lu", g_stats.indications_sent);
    ESP_LOGI(TAG, "Acknowledgments received: %lu", g_stats.acknowledgments_received);
}

void ble_stress_service_reset_statistics(void) {
    if (xSemaphoreTake(g_service.mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        memset(&g_stats, 0, sizeof(ble_service_stats_t));
        xSemaphoreGive(g_service.mutex);
    }
}

void ble_stress_service_set_verbose_logging(bool enable) {
    g_verbose_logging = enable;
    ESP_LOGI(TAG, "Verbose logging %s", enable ? "enabled" : "disabled");
}
