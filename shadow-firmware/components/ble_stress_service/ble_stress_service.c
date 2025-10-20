/**
 * BLE Stress Service (Simplified + Flush/Reset Support)
 *
 * Advertisement (Service Data AD type 0x16):
 *   [UUID_LSB, UUID_MSB, combined]
 *   combined = ((seq7bit << 1) | stateBit)
 *   seq7bit = (latest_event.sequence_number & 0x7F)
 *   stateBit = 1 if current FSM stable state == STABLE_STRESS else 0
 *
 * GATT Service UUID: 0xA000
 * Characteristic UUID: 0xA002 (READ | WRITE)
 *
 * Write Opcodes:
 *   0xFF : RESET -> flush event log via event_log_reset(), respond with reset ACK
 *   other (value V) : treated as clientLastKnownSequence (lower 7 bits), respond with extended replay
 *
 * Responses:
 *   Minimal (2 bytes):
 *      [currentSeq7bit, currentStateBit]
 *
 *   Extended:
 *      Byte0: currentSeq7bit (from latest event)
 *      Byte1: currentStateBit
 *      Byte2: missedCount (M)
 *      Then M * 2 bytes: (seq_i_7bit, state_i_bit) in chronological order (oldest first)
 *
 *   Reset ACK (4 bytes):
 *      [0x00, stateBit, 0x00, 0x52]
 *
 * Notes:
 * - The underlying event log uses 8-bit sequence numbers (wrap 0..255).
 * - For advertisement / protocol we ONLY expose lower 7 bits.
 * - On replay mismatch > buffer capacity host decides to flush (RESET opcode).
 * - last_acknowledged_sequence in event log is ignored in this simplified mode.
 */

#include <string.h>
#include "ble_stress_service.h"
#include "ble_pairing.h"  // For pairing event forwarding
#include "stress_fsm.h"
#include "event_log.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "esp_bt.h"
#include "esp_bt_main.h"
#include "esp_gap_ble_api.h"
#include "esp_gatts_api.h"

#define TAG "BLEStressSimple"

#define SIMPLE_SERVICE_UUID 0xA000
#define SIMPLE_CHAR_UUID    0xA002

// Opcodes
#define BLE_OPCODE_RESET 0xFF
#define BLE_RESET_MAGIC  0x52  /* 'R' marker */

// External contexts
static stress_fsm_context_t *g_fsm       = NULL;
static event_log_context_t  *g_event_log = NULL;

// BLE internal
static bool     g_initialized   = false;
static bool     g_connected     = false;
static uint16_t g_gatts_if      = 0;
static uint16_t g_service_handle= 0;
static uint16_t g_char_handle   = 0;
static uint16_t g_conn_id       = 0;

// Response buffer (one-shot after write)
static uint8_t  g_resp_buffer[96];
static uint16_t g_resp_len = 0;
static bool     g_have_extended = false;

/* Forward declarations */
static void gap_cb(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param);
static void gatts_cb(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if,
                     esp_ble_gatts_cb_param_t *param);
static void update_advertisement(void);
static void prepare_minimal_response(void);
static void prepare_extended_response(uint8_t clientLastSeq7);
static void prepare_reset_response(void);

/* Utility to determine stable state bit from FSM */
static uint8_t stable_state_bit(void) {
    stress_fsm_state_t st = stress_fsm_get_current_state(g_fsm);
    return (st == FSM_STABLE_STRESS) ? 1 : 0;
}

/* Get latest event if any; returns true if available */
static bool get_latest_event(stress_event_t *out) {
    if (!g_event_log) return false;
    return event_log_get_latest_event(g_event_log, out);
}

/* Build current advertisement combined byte */
static uint8_t build_current_adv_combined(void) {
    stress_event_t latest;
    uint8_t seq7 = 0;
    if (get_latest_event(&latest)) {
        seq7 = (latest.sequence_number & 0x7F);
    }
    uint8_t stateBit = stable_state_bit();
    return (uint8_t)((seq7 << 1) | stateBit);
}

/* Push updated advertisement */
static void update_advertisement(void) {
    uint8_t combined = build_current_adv_combined();
    uint8_t service_data[3];
    service_data[0] = (uint8_t)(SIMPLE_SERVICE_UUID & 0xFF);
    service_data[1] = (uint8_t)(SIMPLE_SERVICE_UUID >> 8);
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
        .adv_int_min       = 160,  // 100 ms
        .adv_int_max       = 320,  // 200 ms
        .adv_type          = ADV_TYPE_IND,
        .own_addr_type     = BLE_ADDR_TYPE_PUBLIC,
        .channel_map       = ADV_CHNL_ALL,
        .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
    };

    ESP_ERROR_CHECK(esp_ble_gap_config_adv_data(&adv_data));
    ESP_ERROR_CHECK(esp_ble_gap_start_advertising(&adv_params));

    ESP_LOGI(TAG, "Advertising combined=0x%02X (seq7=%u state=%u)",
             combined, (combined >> 1) & 0x7F, combined & 0x01);
}

/* Minimal response: just latest seq7 + state bit */
static void prepare_minimal_response(void) {
    uint8_t combined = build_current_adv_combined();
    g_resp_buffer[0] = (combined >> 1) & 0x7F;
    g_resp_buffer[1] = combined & 0x01;
    g_resp_len = 2;
    g_have_extended = false;
}

/* Extended response:
 *  - Provide the latest event seq7 / state
 *  - Provide missed events since clientLastSeq7 (lower 7 bits compare)
 *  - We fetch events starting from (clientLastSeq7+1 & 0xFF) because event log full sequences are 8-bit.
 *    Filtering is done by event_log_get_events_from_sequence using 8-bit numbering.
 */
static void prepare_extended_response(uint8_t clientLastSeq7) {
    if (!g_event_log) {
        prepare_minimal_response();
        return;
    }

    /* Map 7-bit client sequence into 8-bit search window.
       Since event log is 8-bit, we assume lower 7 bits match
       (loss of information is acceptable for simplified design).
       We request starting at (clientLastSeq + 1) modulo 8-bit. */
    uint8_t startSeq8 = (uint8_t)((clientLastSeq7 + 1) & 0xFF);

    stress_event_t temp[EVENT_LOG_CAPACITY];
    uint8_t fetched = event_log_get_events_from_sequence(g_event_log,
                                                         startSeq8,
                                                         temp,
                                                         EVENT_LOG_CAPACITY);
    if (fetched == 0) {
        /* No new events accessible */
        prepare_minimal_response();
        return;
    }

    /* Latest (last) fetched event defines "current" state for response */
    stress_event_t *latest = &temp[fetched - 1];
    uint8_t currentSeq7 = latest->sequence_number & 0x7F;
    uint8_t currentStateBit = (latest->new_state == FSM_STABLE_STRESS) ? 1 : 0;

    /* Missed events are everything except the last one */
    uint8_t missedCount = (fetched > 1) ? (uint8_t)(fetched - 1) : 0;

    uint16_t idx = 0;
    g_resp_buffer[idx++] = currentSeq7;
    g_resp_buffer[idx++] = currentStateBit;
    g_resp_buffer[idx++] = missedCount;

    for (uint8_t i = 0; i < missedCount; i++) {
        uint8_t seq7 = temp[i].sequence_number & 0x7F;
        uint8_t st   = (temp[i].new_state == FSM_STABLE_STRESS) ? 1 : 0;
        g_resp_buffer[idx++] = seq7;
        g_resp_buffer[idx++] = st;
    }

    g_resp_len = idx;
    g_have_extended = true;
    ESP_LOGI(TAG, "Extended response: currentSeq7=%u missed=%u rawFetched=%u",
             currentSeq7, missedCount, fetched);
}

/* Reset ACK (flush) */
static void prepare_reset_response(void) {
    if (g_event_log) {
        event_log_reset(g_event_log);
        ESP_LOGW(TAG, "Event log reset (flush)");
    }

    /* After reset: sequence = 0 visible to host */
    g_resp_buffer[0] = 0x00;               // seq7 = 0
    g_resp_buffer[1] = stable_state_bit(); // current stable bit
    g_resp_buffer[2] = 0x00;               // missedCount = 0
    g_resp_buffer[3] = BLE_RESET_MAGIC;    // marker
    g_resp_len = 4;
    g_have_extended = true;
    ESP_LOGW(TAG, "Reset response prepared");
}

/* GAP callback */
static void gap_cb(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param) {
    switch (event) {
    case ESP_GAP_BLE_ADV_DATA_SET_COMPLETE_EVT:
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

/* Write handler */
static void on_write_event(esp_ble_gatts_cb_param_t *param) {
    if (param->write.handle != g_char_handle) return;

    if (param->write.len != 1) {
        ESP_LOGW(TAG, "Unexpected write length=%d", param->write.len);
        prepare_minimal_response();
    } else {
        uint8_t opcode = param->write.value[0];
        if (opcode == BLE_OPCODE_RESET) {
            ESP_LOGW(TAG, "RESET opcode received");
            prepare_reset_response();
        } else {
            uint8_t clientLastSeq7 = opcode & 0x7F;
            ESP_LOGI(TAG, "Replay request clientLastSeq7=%u", clientLastSeq7);
            prepare_extended_response(clientLastSeq7);
        }
    }

    esp_ble_gatts_send_response(g_gatts_if,
                                param->write.conn_id,
                                param->write.trans_id,
                                ESP_GATT_OK,
                                NULL);

    /* Try to send an indication immediately so central receives the prepared response
       without requiring an explicit read. If client hasn't enabled CCCD or indication
       fails, that is handled gracefully. */
    if (g_resp_len > 0) {
        esp_err_t rc = esp_ble_gatts_send_indicate(g_gatts_if,
                                                   param->write.conn_id,
                                                   g_char_handle,
                                                   g_resp_len,
                                                   g_resp_buffer,
                                                   false);
        if (rc != ESP_OK) {
            ESP_LOGW(TAG, "send_indicate failed: %d", rc);
        } else {
            ESP_LOGI(TAG, "Sent indicate (len=%u)", g_resp_len);
        }
    }
}

/* Read handler */
static void on_read_event(esp_ble_gatts_cb_param_t *param) {
    if (param->read.handle != g_char_handle) {
        esp_ble_gatts_send_response(g_gatts_if,
                                    param->read.conn_id,
                                    param->read.trans_id,
                                    ESP_GATT_OK,
                                    NULL);
        return;
    }

    if (!g_have_extended) {
        prepare_minimal_response();
    }

    esp_gatt_rsp_t rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.attr_value.handle = g_char_handle;
    rsp.attr_value.len = g_resp_len;
    if (rsp.attr_value.len > sizeof(rsp.attr_value.value)) {
        rsp.attr_value.len = sizeof(rsp.attr_value.value);
    }
    memcpy(rsp.attr_value.value, g_resp_buffer, rsp.attr_value.len);

    esp_ble_gatts_send_response(g_gatts_if,
                                param->read.conn_id,
                                param->read.trans_id,
                                ESP_GATT_OK,
                                &rsp);

    g_have_extended = false; // one-shot
}

/* GATT server callback - dispatches to both stress and pairing services */
static void gatts_cb(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if,
                     esp_ble_gatts_cb_param_t *param) {
    // Forward ALL events to pairing service (it filters by app_id internally)
    ble_pairing_gatts_handler(event, gatts_if, param);
    
    // Handle stress service events (app_id=0 only)
    if (event == ESP_GATTS_REG_EVT) {
        // Only process registration for our app_id (0)
        if (param->reg.status == ESP_GATT_OK && param->reg.app_id == 0) {
            g_gatts_if = gatts_if;
            ESP_LOGI(TAG, "GATT registered");
            esp_gatt_srvc_id_t sid = {
                .is_primary = true,
                .id = {
                    .inst_id = 0,
                    .uuid = { .len = ESP_UUID_LEN_16,
                              .uuid = { .uuid16 = SIMPLE_SERVICE_UUID } }
                }
            };
            esp_ble_gatts_create_service(gatts_if, &sid, 4);
        } else if (param->reg.status != ESP_GATT_OK && param->reg.app_id == 0) {
            ESP_LOGE(TAG, "GATT reg failed: %d", param->reg.status);
        }
        return;
    }

    // Filter all other events by gatts_if (only process our own)
    if (gatts_if != g_gatts_if) return;

    switch (event) {
    case ESP_GATTS_CREATE_EVT:
        if (param->create.status == ESP_GATT_OK) {
            g_service_handle = param->create.service_handle;
            ESP_LOGI(TAG, "Service created handle=%u", g_service_handle);
            esp_ble_gatts_start_service(g_service_handle);

            esp_bt_uuid_t uuid = {
                .len = ESP_UUID_LEN_16,
                .uuid = { .uuid16 = SIMPLE_CHAR_UUID }
            };
            esp_gatt_char_prop_t prop =
                ESP_GATT_CHAR_PROP_BIT_READ | ESP_GATT_CHAR_PROP_BIT_WRITE | ESP_GATT_CHAR_PROP_BIT_NOTIFY;
            esp_ble_gatts_add_char(g_service_handle, &uuid,
                                   ESP_GATT_PERM_READ | ESP_GATT_PERM_WRITE,
                                   prop, NULL, NULL);
        }
        break;

    case ESP_GATTS_ADD_CHAR_EVT:
        if (param->add_char.status == ESP_GATT_OK) {
            g_char_handle = param->add_char.attr_handle;
            ESP_LOGI(TAG, "Characteristic added handle=%u", g_char_handle);

            /* Add CCCD descriptor so clients can enable notifications/indications */
            esp_bt_uuid_t cccd_uuid = { .len = ESP_UUID_LEN_16, .uuid = { .uuid16 = ESP_GATT_UUID_CHAR_CLIENT_CONFIG } };
            esp_err_t rc = esp_ble_gatts_add_char_descr(g_service_handle, &cccd_uuid,
                                                       ESP_GATT_PERM_READ | ESP_GATT_PERM_WRITE,
                                                       NULL, NULL);
            if (rc != ESP_OK) {
                ESP_LOGW(TAG, "Failed to add CCCD descriptor: %d", rc);
            }

            update_advertisement();
        } else {
            ESP_LOGE(TAG, "Add char failed: %d", param->add_char.status);
        }
        break;

    case ESP_GATTS_CONNECT_EVT:
        g_connected = true;
        g_conn_id = param->connect.conn_id;
        ESP_LOGI(TAG, "Client connected (conn_id=%u)", g_conn_id);
        esp_ble_gap_stop_advertising(); // optional pause
        break;

    case ESP_GATTS_DISCONNECT_EVT:
        g_connected = false;
        ESP_LOGI(TAG, "Client disconnected");
        update_advertisement(); // resume advertising
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

/* Public API */
int ble_stress_service_init(stress_fsm_context_t *fsm_ctx, event_log_context_t *event_ctx) {
    if (g_initialized) return 0;
    if (!fsm_ctx || !event_ctx) return -1;

    g_fsm       = fsm_ctx;
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
    ESP_LOGI(TAG, "BLE Stress Service initialized");
    return 0;
}

void ble_stress_service_deinit(void) {
    if (!g_initialized) return;
    esp_ble_gap_stop_advertising();
    g_initialized = false;
}

void ble_stress_service_tick(void) {
    /* Called after FSM transitions or periodically.
       If not connected, refresh advertisement content. */
    if (!g_connected) {
        update_advertisement();
    }
    /* If connected and transitions happen, they’ll be reflected next time we advertise.
       We keep no separate sequence counter here—event log drives sequence. */
}