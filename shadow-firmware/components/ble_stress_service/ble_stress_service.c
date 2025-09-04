/**
 * Simplified BLE Stress Service (ESP32)
 *
 * Advertisement (Service Data AD type 0x16):
 *   Bytes placed (service_data_len=3):
 *     [ UUID_LSB, UUID_MSB, combined ]
 *   CoreBluetooth strips the 2-byte UUID when exposing serviceData[UUID],
 *   so the macOS app receives a single byte: combined
 *
 * combined = (sequence << 1) | stateBit
 *   sequence: 7-bit rolling counter (0–127), incremented ONLY on confirmed stable
 *             transitions between FSM_STABLE_CALM and FSM_STABLE_STRESS.
 *   stateBit: 0 = CALM, 1 = STRESS (transitional states reuse last stable bit).
 *
 * GATT:
 *   Service UUID: 0xA000
 *   Characteristic UUID: 0xA002 (READ | WRITE, no notifications)
 *
 *   READ (without prior WRITE or if no missed events scenario):
 *     Returns 2 bytes: [ currentSequence, currentStateBit ]
 *
 *   WRITE (1 byte: clientLastKnownSequence), then READ:
 *     Returns extended structure:
 *       Byte0: currentSequence
 *       Byte1: currentStateBit
 *       Byte2: missedCount (N = number of intervening events excluding the final one)
 *       Then N * 2 bytes: (sequence, stateBit) pairs for missed events
 *     Definitions:
 *       clientLastKnownSequence = the highest sequence the client already has
 *       currentSequence = device's latest stable transition sequence
 *       We gather all logged transitions with sequence > clientLastKnownSequence
 *       Of those, the final one (highest sequence == currentSequence) gives
 *       current state; earlier ones are "missed".
 */

#include "ble_stress_service.h"
#include "stress_fsm.h"
#include "event_log.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "esp_bt.h"
#include "esp_bt_main.h"
#include "esp_gap_ble_api.h"
#include "esp_gatts_api.h"
#include <string.h>

#define TAG "BLEStressSimple"

// UUIDs
#define SIMPLE_SERVICE_UUID      0xA000
#define SIMPLE_CHAR_UUID         0xA002

// Rolling sequence and stable state tracking
static uint8_t g_sequence = 0;  // 0–127
static stress_fsm_state_t g_last_stable_state = FSM_STABLE_CALM;

// Context
static stress_fsm_context_t *g_fsm = NULL;
static event_log_context_t *g_event_log = NULL;

static bool g_initialized = false;
static bool g_connected = false;

static uint16_t g_gatts_if = 0;
static uint16_t g_service_handle = 0;
static uint16_t g_char_handle = 0;
static uint16_t g_conn_id = 0;

// Extended response buffer (prepared after WRITE, consumed on next READ)
static uint8_t g_resp_buffer[80];
static uint16_t g_resp_len = 0;
static bool g_have_extended = false;

// Forward declarations
static void gap_cb(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param);
static void gatts_cb(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if,
                     esp_ble_gatts_cb_param_t *param);
static void update_advertisement(void);
static void prepare_minimal_response(void);
static void prepare_extended_response(uint8_t clientLastSeq);

// --------------------------------------------------
// Utility
// --------------------------------------------------
static void maybe_increment_sequence(void) {
    stress_fsm_state_t current = stress_fsm_get_current_state(g_fsm);
    if (current == FSM_STABLE_CALM || current == FSM_STABLE_STRESS) {
        if (current != g_last_stable_state) {
            g_sequence = (g_sequence + 1) & 0x7F;
            g_last_stable_state = current;
            ESP_LOGI(TAG, "Stable transition -> seq=%u state=%s",
                     g_sequence, stress_fsm_state_to_string(current));
        }
    }
}

static uint8_t stable_state_bit(void) {
    return (g_last_stable_state == FSM_STABLE_STRESS) ? 1 : 0;
}

// --------------------------------------------------
// Advertisement
// --------------------------------------------------
static void update_advertisement(void) {
    maybe_increment_sequence();

    uint8_t combined = (uint8_t)((g_sequence << 1) | stable_state_bit());

    // Service Data payload: [UUID_LSB, UUID_MSB, combinedByte]
    uint8_t service_data[3];
    service_data[0] = SIMPLE_SERVICE_UUID & 0xFF;
    service_data[1] = (SIMPLE_SERVICE_UUID >> 8) & 0xFF;
    service_data[2] = combined;

    esp_ble_adv_data_t adv_data = {
        .set_scan_rsp = false,
        .include_name = true,
        .include_txpower = false,
        .service_data_len = sizeof(service_data),
        .p_service_data = service_data,
        .flag = ESP_BLE_ADV_FLAG_GEN_DISC | ESP_BLE_ADV_FLAG_BREDR_NOT_SPT
    };

    esp_ble_adv_params_t adv_params = {
        .adv_int_min = 160,  // 100ms
        .adv_int_max = 320,  // 200ms
        .adv_type = ADV_TYPE_IND,
        .own_addr_type = BLE_ADDR_TYPE_PUBLIC,
        .channel_map = ADV_CHNL_ALL,
        .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
    };

    ESP_ERROR_CHECK(esp_ble_gap_config_adv_data(&adv_data));
    ESP_ERROR_CHECK(esp_ble_gap_start_advertising(&adv_params));

    ESP_LOGI(TAG, "Advertising combined=0x%02X sequence=%u stateBit=%u",
             combined, g_sequence, stable_state_bit());
}

// --------------------------------------------------
// Responses
// --------------------------------------------------
static void prepare_minimal_response(void) {
    g_resp_buffer[0] = g_sequence;
    g_resp_buffer[1] = stable_state_bit();
    g_resp_len = 2;
    g_have_extended = false;
}

static void prepare_extended_response(uint8_t clientLastSeq) {
    // Collect events strictly greater than clientLastSeq
    // We'll use existing event_log API:
    // event_log_get_events_from_sequence(context, start_sequence, outArray, maxCount)
    // returns events with sequence >= start_sequence.
    //
    // We want events > clientLastSeq, so start_sequence = clientLastSeq + 1.
    // The highest event is the "current" new sequence (if present).
    // All preceding form the "missed" list.
    
    stress_event_t events[20];
    uint8_t startSeq = (clientLastSeq + 1) & 0x7F; // modulo wrap
    uint8_t count = event_log_get_events_from_sequence(g_event_log, startSeq, events, 20);

    if (count == 0) {
        // No new events; fallback to minimal
        prepare_minimal_response();
        return;
    }

    // The highest (last) event corresponds to current stable transition
    // We assume event sequence numbers are strictly increasing modulo 128
    // For simplicity we ignore wrap anomaly inside this window.
    stress_event_t *latest = &events[count - 1];
    uint8_t currentSeq = latest->sequence_number;
    uint8_t currentStateBit = (latest->new_state == FSM_STABLE_STRESS) ? 1 : 0;

    // Missed events exclude the latest one:
    uint8_t missedCount = (count > 1) ? (count - 1) : 0;

    uint16_t idx = 0;
    g_resp_buffer[idx++] = currentSeq;
    g_resp_buffer[idx++] = currentStateBit;
    g_resp_buffer[idx++] = missedCount;

    for (uint8_t i = 0; i < missedCount; i++) {
        g_resp_buffer[idx++] = events[i].sequence_number;
        uint8_t st = (events[i].new_state == FSM_STABLE_STRESS) ? 1 : 0;
        g_resp_buffer[idx++] = st;
    }

    g_resp_len = idx;
    g_have_extended = true;
    ESP_LOGI(TAG, "Prepared extended response currentSeq=%u missedCount=%u", currentSeq, missedCount);
}

// --------------------------------------------------
// GAP Callback
// --------------------------------------------------
static void gap_cb(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param) {
    switch (event) {
        case ESP_GAP_BLE_ADV_DATA_SET_COMPLETE_EVT:
            // Data prepared
            break;
        case ESP_GAP_BLE_ADV_START_COMPLETE_EVT:
            if (param->adv_start_cmpl.status != ESP_BT_STATUS_SUCCESS) {
                ESP_LOGE(TAG, "Failed to start advertising: %d", param->adv_start_cmpl.status);
            }
            break;
        default:
            break;
    }
}

// --------------------------------------------------
// GATTS Callback
// --------------------------------------------------
static void on_write_event(esp_ble_gatts_cb_param_t *param) {
    if (param->write.handle != g_char_handle) {
        return;
    }
    if (param->write.len != 1) {
        ESP_LOGW(TAG, "Unexpected write length=%d", param->write.len);
        // Fallback minimal response
        prepare_minimal_response();
    } else {
        uint8_t clientLastSeq = param->write.value[0] & 0x7F;
        ESP_LOGI(TAG, "WRITE: client lastKnownSequence=%u", clientLastSeq);
        prepare_extended_response(clientLastSeq);
    }

    esp_ble_gatts_send_response(g_gatts_if,
                                param->write.conn_id,
                                param->write.trans_id,
                                ESP_GATT_OK,
                                NULL);
}

static void on_read_event(esp_ble_gatts_cb_param_t *param) {
    if (param->read.handle != g_char_handle) {
        esp_ble_gatts_send_response(g_gatts_if,
                                    param->read.conn_id,
                                    param->read.trans_id,
                                    ESP_GATT_OK,
                                    NULL);
        return;
    }

    // If no extended response prepared, supply minimal
    if (!g_have_extended) {
        prepare_minimal_response();
    }
    
    esp_gatt_rsp_t rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.attr_value.handle = g_char_handle;
    rsp.attr_value.len = g_resp_len;
    if (g_resp_len > sizeof(rsp.attr_value.value)) {
        rsp.attr_value.len = sizeof(rsp.attr_value.value);
    }
    memcpy(rsp.attr_value.value, g_resp_buffer, rsp.attr_value.len);

    esp_ble_gatts_send_response(g_gatts_if,
                                param->read.conn_id,
                                param->read.trans_id,
                                ESP_GATT_OK,
                                &rsp);

    // Extended responses are one-shot; clear after read
    g_have_extended = false;
}

static void gatts_cb(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if,
                     esp_ble_gatts_cb_param_t *param) {
    if (event == ESP_GATTS_REG_EVT) {
        if (param->reg.status == ESP_GATT_OK) {
            g_gatts_if = gatts_if;
            ESP_LOGI(TAG, "GATT registered");
            // Create service
            esp_gatt_srvc_id_t sid = {
                .is_primary = true,
                .id = {
                    .inst_id = 0,
                    .uuid = {
                        .len = ESP_UUID_LEN_16,
                        .uuid = { .uuid16 = SIMPLE_SERVICE_UUID }
                    }
                }
            };
            esp_ble_gatts_create_service(gatts_if, &sid, 4);
        } else {
            ESP_LOGE(TAG, "GATT reg failed: %d", param->reg.status);
        }
        return;
    }

    if (gatts_if != g_gatts_if) return;

    switch (event) {
        case ESP_GATTS_CREATE_EVT:
            if (param->create.status == ESP_GATT_OK) {
                g_service_handle = param->create.service_handle;
                ESP_LOGI(TAG, "Service created handle=%u", g_service_handle);
                esp_ble_gatts_start_service(g_service_handle);

                // Add characteristic
                esp_bt_uuid_t uuid = {
                    .len = ESP_UUID_LEN_16,
                    .uuid = { .uuid16 = SIMPLE_CHAR_UUID }
                };
                esp_gatt_char_prop_t prop = ESP_GATT_CHAR_PROP_BIT_READ | ESP_GATT_CHAR_PROP_BIT_WRITE;
                esp_ble_gatts_add_char(g_service_handle, &uuid,
                                       ESP_GATT_PERM_READ | ESP_GATT_PERM_WRITE,
                                       prop, NULL, NULL);
            }
            break;

        case ESP_GATTS_ADD_CHAR_EVT:
            if (param->add_char.status == ESP_GATT_OK) {
                g_char_handle = param->add_char.attr_handle;
                ESP_LOGI(TAG, "Characteristic added handle=%u", g_char_handle);
                // Start advertising immediately
                update_advertisement();
            } else {
                ESP_LOGE(TAG, "Add char failed: %d", param->add_char.status);
            }
            break;

        case ESP_GATTS_CONNECT_EVT:
            g_connected = true;
            g_conn_id = param->connect.conn_id;
            ESP_LOGI(TAG, "Client connected (conn_id=%u)", g_conn_id);
            // Stop advertising while connected (optional)
            esp_ble_gap_stop_advertising();
            break;

        case ESP_GATTS_DISCONNECT_EVT:
            g_connected = false;
            ESP_LOGI(TAG, "Client disconnected");
            // Resume advertising with up-to-date sequence/state
            update_advertisement();
            break;

        case ESP_GATTS_WRITE_EVT:
            on_write_event(param);
            break;

        case ESP_GATTS_READ_EVT:
            on_read_event(param);
            break;

        default:
            break;
    }
}

// --------------------------------------------------
// Public API (Header)
// --------------------------------------------------
int ble_stress_service_init(stress_fsm_context_t *fsm_ctx, event_log_context_t *event_ctx) {
    if (g_initialized) return 0;
    if (!fsm_ctx || !event_ctx) return -1;

    g_fsm = fsm_ctx;
    g_event_log = event_ctx;

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    ESP_ERROR_CHECK(esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT));
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_bt_controller_init(&bt_cfg));
    ESP_ERROR_CHECK(esp_bt_controller_enable(ESP_BT_MODE_BLE));
    ESP_ERROR_CHECK(esp_bluedroid_init());
    ESP_ERROR_CHECK(esp_bluedroid_enable());

    ESP_ERROR_CHECK(esp_ble_gap_register_callback(gap_cb));
    ESP_ERROR_CHECK(esp_ble_gatts_register_callback(gatts_cb));
    ESP_ERROR_CHECK(esp_ble_gatts_app_register(0));

    g_initialized = true;
    ESP_LOGI(TAG, "BLE Stress Simple Service initialized");
    return 0;
}

void ble_stress_service_deinit(void) {
    if (!g_initialized) return;
    esp_ble_gap_stop_advertising();
    g_initialized = false;
}

void ble_stress_service_tick(void) {
    // Call periodically (e.g. every few seconds or on state transition)
    if (!g_connected) {
        // Rebuild advertisement if stable state changed
        update_advertisement();
    } else {
        // While connected we do not re-advertise
        maybe_increment_sequence(); // Keep internal sequence correct if state changes mid-connection
    }
}