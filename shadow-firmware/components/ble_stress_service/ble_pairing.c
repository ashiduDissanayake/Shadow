/**
 * @file ble_pairing.c
 * @brief BLE Device Pairing Protocol Implementation
 * 
 * This implements a secure pairing mechanism for the Shadow stress detection system.
 * Features challenge-response authentication, persistent pairing storage, and multi-device support.
 */

#include "ble_pairing.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_bt.h"
#include "esp_bt_main.h"
#include "esp_gap_ble_api.h"
#include "esp_gatts_api.h"
#include "mbedtls/sha256.h"
#include <string.h>
#include <stdio.h>

#define TAG "BLEPairing"

/* NVS storage keys */
#define NVS_NAMESPACE "ble_pairing"
#define NVS_KEY_DEVICE_ID "device_id"
#define NVS_KEY_DEVICE_NAME "device_name"
#define NVS_KEY_PAIRED_COUNT "paired_count"
#define NVS_KEY_PAIRED_PREFIX "paired_"  // paired_0, paired_1, paired_2

/* Challenge timeout (30 seconds) */
#define CHALLENGE_TIMEOUT_US (30 * 1000000ULL)

/* Default device name */
#define DEFAULT_DEVICE_NAME_PREFIX "Shadow-"

/* Global pairing context */
static pairing_context_t g_pairing_ctx = {0};

/* Forward declarations */
static int load_from_nvs(void);
static int save_to_nvs(void);
static int save_paired_device(uint8_t index);
static int load_paired_device(uint8_t index);
static void generate_device_id(uint8_t *device_id);
static void generate_challenge(security_challenge_t *challenge);
static bool verify_challenge_response(const uint8_t *response);
static int find_free_slot(void);
static int find_paired_device(const uint8_t *device_id);
static void prepare_device_info_response(uint8_t *buffer, uint16_t *len);
static void prepare_pairing_state_response(uint8_t *buffer, uint16_t *len);
static void handle_pairing_command(uint8_t command, const uint8_t *data, uint16_t len);

/* ==================== INITIALIZATION ==================== */

int ble_pairing_init(const char *device_name) {
    if (g_pairing_ctx.initialized) {
        ESP_LOGW(TAG, "Already initialized");
        return 0;
    }

    ESP_LOGI(TAG, "Initializing BLE pairing service...");

    memset(&g_pairing_ctx, 0, sizeof(pairing_context_t));

    /* Load persistent data from NVS */
    if (load_from_nvs() != 0) {
        ESP_LOGW(TAG, "No stored pairing data, generating new device ID");
        generate_device_id(g_pairing_ctx.device_info.device_id);
        
        /* Set default device name if not provided */
        if (device_name) {
            snprintf(g_pairing_ctx.device_info.device_name, DEVICE_NAME_MAX_LEN, "%s", device_name);
        } else {
            /* Use last 4 bytes of device ID for unique name */
            snprintf(g_pairing_ctx.device_info.device_name, DEVICE_NAME_MAX_LEN,
                    "%s%02X%02X", DEFAULT_DEVICE_NAME_PREFIX,
                    g_pairing_ctx.device_info.device_id[14],
                    g_pairing_ctx.device_info.device_id[15]);
        }
        
        save_to_nvs();
    }

    /* Set firmware and hardware info */
    snprintf(g_pairing_ctx.device_info.firmware_version, FIRMWARE_VERSION_LEN, "v1.0.0");
    snprintf(g_pairing_ctx.device_info.hardware_revision, HARDWARE_REVISION_LEN, "ESP32-S3");

    /* Ensure GAP device name is set so scanners show the configured name */
    if (strlen(g_pairing_ctx.device_info.device_name) > 0) {
        esp_err_t rc = esp_ble_gap_set_device_name(g_pairing_ctx.device_info.device_name);
        if (rc != ESP_OK) {
            ESP_LOGW(TAG, "Failed to set GAP device name: %d", rc);
        } else {
            ESP_LOGI(TAG, "GAP device name set: %s", g_pairing_ctx.device_info.device_name);
        }
    }

    /* Set BLE GAP device name for advertising */
    esp_err_t ret = esp_ble_gap_set_device_name(g_pairing_ctx.device_info.device_name);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set BLE device name: %s", esp_err_to_name(ret));
        return -1;
    }
    ESP_LOGI(TAG, "BLE GAP device name set to: %s", g_pairing_ctx.device_info.device_name);

    /* NOTE: Do NOT register GATT callback here - it would overwrite stress service callback!
     * Instead, the stress service callback will forward pairing events based on gatts_if */
    // ret = esp_ble_gatts_register_callback(gatts_pairing_cb);  // REMOVED
    // if (ret != ESP_OK) {
    //     ESP_LOGE(TAG, "GATT callback register failed: %s", esp_err_to_name(ret));
    //     return -1;
    // }

    ret = esp_ble_gatts_app_register(1); // App ID = 1 for pairing service
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "GATT app register failed: %s", esp_err_to_name(ret));
        return -1;
    }

    g_pairing_ctx.state = PAIRING_STATE_ADVERTISING;
    g_pairing_ctx.initialized = true;

    ESP_LOGI(TAG, "BLE pairing service initialized successfully");
    ESP_LOGI(TAG, "Device ID: %02X%02X...%02X%02X",
             g_pairing_ctx.device_info.device_id[0],
             g_pairing_ctx.device_info.device_id[1],
             g_pairing_ctx.device_info.device_id[14],
             g_pairing_ctx.device_info.device_id[15]);
    ESP_LOGI(TAG, "Device Name: %s", g_pairing_ctx.device_info.device_name);

    return 0;
}

void ble_pairing_deinit(void) {
    if (!g_pairing_ctx.initialized) return;
    
    /* Save current state to NVS */
    save_to_nvs();
    
    g_pairing_ctx.initialized = false;
    ESP_LOGI(TAG, "BLE pairing service deinitialized");
}

/* ==================== NVS PERSISTENCE ==================== */

static int load_from_nvs(void) {
    nvs_handle_t nvs_handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READONLY, &nvs_handle);
    if (err != ESP_OK) {
        return -1;
    }

    size_t size;

    /* Load device ID */
    size = DEVICE_ID_LEN;
    err = nvs_get_blob(nvs_handle, NVS_KEY_DEVICE_ID, g_pairing_ctx.device_info.device_id, &size);
    if (err != ESP_OK) {
        nvs_close(nvs_handle);
        return -1;
    }

    /* Load device name */
    size = DEVICE_NAME_MAX_LEN;
    err = nvs_get_str(nvs_handle, NVS_KEY_DEVICE_NAME, g_pairing_ctx.device_info.device_name, &size);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "No stored device name");
    }

    /* Load paired devices count */
    uint8_t paired_count = 0;
    err = nvs_get_u8(nvs_handle, NVS_KEY_PAIRED_COUNT, &paired_count);
    if (err == ESP_OK && paired_count <= MAX_PAIRED_DEVICES) {
        for (uint8_t i = 0; i < paired_count; i++) {
            load_paired_device(i);
        }
    }

    nvs_close(nvs_handle);
    ESP_LOGI(TAG, "Loaded %d paired devices from NVS", paired_count);
    return 0;
}

static int save_to_nvs(void) {
    nvs_handle_t nvs_handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &nvs_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS: %s", esp_err_to_name(err));
        return -1;
    }

    /* Save device ID */
    err = nvs_set_blob(nvs_handle, NVS_KEY_DEVICE_ID, g_pairing_ctx.device_info.device_id, DEVICE_ID_LEN);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to save device ID");
    }

    /* Save device name */
    err = nvs_set_str(nvs_handle, NVS_KEY_DEVICE_NAME, g_pairing_ctx.device_info.device_name);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to save device name");
    }

    /* Count valid paired devices */
    uint8_t paired_count = 0;
    for (uint8_t i = 0; i < MAX_PAIRED_DEVICES; i++) {
        if (g_pairing_ctx.paired_devices[i].is_valid) {
            save_paired_device(i);
            paired_count++;
        }
    }

    /* Save paired count */
    err = nvs_set_u8(nvs_handle, NVS_KEY_PAIRED_COUNT, paired_count);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to save paired count");
    }

    err = nvs_commit(nvs_handle);
    nvs_close(nvs_handle);

    ESP_LOGI(TAG, "Saved %d paired devices to NVS", paired_count);
    return (err == ESP_OK) ? 0 : -1;
}

static int save_paired_device(uint8_t index) {
    if (index >= MAX_PAIRED_DEVICES) return -1;

    nvs_handle_t nvs_handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &nvs_handle);
    if (err != ESP_OK) return -1;

    char key[32];
    snprintf(key, sizeof(key), "%s%d", NVS_KEY_PAIRED_PREFIX, index);

    err = nvs_set_blob(nvs_handle, key, &g_pairing_ctx.paired_devices[index], sizeof(paired_device_t));
    if (err == ESP_OK) {
        err = nvs_commit(nvs_handle);
    }

    nvs_close(nvs_handle);
    return (err == ESP_OK) ? 0 : -1;
}

static int load_paired_device(uint8_t index) {
    if (index >= MAX_PAIRED_DEVICES) return -1;

    nvs_handle_t nvs_handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READONLY, &nvs_handle);
    if (err != ESP_OK) return -1;

    char key[32];
    snprintf(key, sizeof(key), "%s%d", NVS_KEY_PAIRED_PREFIX, index);

    size_t size = sizeof(paired_device_t);
    err = nvs_get_blob(nvs_handle, key, &g_pairing_ctx.paired_devices[index], &size);

    nvs_close(nvs_handle);
    return (err == ESP_OK) ? 0 : -1;
}

/* ==================== UTILITY FUNCTIONS ==================== */

static void generate_device_id(uint8_t *device_id) {
    /* Generate random 128-bit UUID */
    esp_fill_random(device_id, DEVICE_ID_LEN);
    
    /* Set UUID version 4 (random) and variant bits */
    device_id[6] = (device_id[6] & 0x0F) | 0x40;  // Version 4
    device_id[8] = (device_id[8] & 0x3F) | 0x80;  // Variant 10
}

static void generate_challenge(security_challenge_t *challenge) {
    /* Generate random 128-bit challenge */
    esp_fill_random(challenge->challenge, CHALLENGE_LEN);
    challenge->timestamp = esp_timer_get_time();
    challenge->is_valid = true;

    /* Compute expected response: SHA-256(challenge + device_id) */
    uint8_t input[CHALLENGE_LEN + DEVICE_ID_LEN];
    memcpy(input, challenge->challenge, CHALLENGE_LEN);
    memcpy(input + CHALLENGE_LEN, g_pairing_ctx.device_info.device_id, DEVICE_ID_LEN);

    uint8_t hash[32];
    mbedtls_sha256(input, sizeof(input), hash, 0);

    /* Use first 16 bytes of hash as expected response */
    memcpy(challenge->response, hash, CHALLENGE_LEN);

    ESP_LOGI(TAG, "Challenge generated");
}

static bool verify_challenge_response(const uint8_t *response) {
    if (!g_pairing_ctx.current_challenge.is_valid) {
        ESP_LOGW(TAG, "No active challenge");
        return false;
    }

    /* Check timeout */
    uint64_t now = esp_timer_get_time();
    if ((now - g_pairing_ctx.current_challenge.timestamp) > CHALLENGE_TIMEOUT_US) {
        ESP_LOGW(TAG, "Challenge timeout");
        g_pairing_ctx.current_challenge.is_valid = false;
        return false;
    }

    /* Verify response */
    bool valid = (memcmp(response, g_pairing_ctx.current_challenge.response, CHALLENGE_LEN) == 0);
    
    if (valid) {
        ESP_LOGI(TAG, "Challenge response verified");
    } else {
        ESP_LOGW(TAG, "Challenge response verification failed");
    }

    /* Invalidate challenge after use */
    g_pairing_ctx.current_challenge.is_valid = false;
    return valid;
}

static int find_free_slot(void) {
    for (int i = 0; i < MAX_PAIRED_DEVICES; i++) {
        if (!g_pairing_ctx.paired_devices[i].is_valid) {
            return i;
        }
    }
    return -1;
}

static int find_paired_device(const uint8_t *device_id) {
    for (int i = 0; i < MAX_PAIRED_DEVICES; i++) {
        if (g_pairing_ctx.paired_devices[i].is_valid &&
            memcmp(g_pairing_ctx.paired_devices[i].device_id, device_id, DEVICE_ID_LEN) == 0) {
            return i;
        }
    }
    return -1;
}

/* ==================== RESPONSE PREPARATION ==================== */

static void prepare_device_info_response(uint8_t *buffer, uint16_t *len) {
    uint16_t offset = 0;

    /* Device ID (16 bytes) */
    memcpy(buffer + offset, g_pairing_ctx.device_info.device_id, DEVICE_ID_LEN);
    offset += DEVICE_ID_LEN;

    /* Device Name (32 bytes) */
    memcpy(buffer + offset, g_pairing_ctx.device_info.device_name, DEVICE_NAME_MAX_LEN);
    offset += DEVICE_NAME_MAX_LEN;

    /* Firmware Version (16 bytes) */
    memcpy(buffer + offset, g_pairing_ctx.device_info.firmware_version, FIRMWARE_VERSION_LEN);
    offset += FIRMWARE_VERSION_LEN;

    /* Hardware Revision (16 bytes) */
    memcpy(buffer + offset, g_pairing_ctx.device_info.hardware_revision, HARDWARE_REVISION_LEN);
    offset += HARDWARE_REVISION_LEN;

    *len = offset;
}

static void prepare_pairing_state_response(uint8_t *buffer, uint16_t *len) {
    uint16_t offset = 0;

    /* Pairing state (1 byte) */
    buffer[offset++] = (uint8_t)g_pairing_ctx.state;

    /* Paired count (1 byte) */
    uint8_t paired_count = 0;
    for (int i = 0; i < MAX_PAIRED_DEVICES; i++) {
        if (g_pairing_ctx.paired_devices[i].is_valid) {
            paired_count++;
        }
    }
    buffer[offset++] = paired_count;

    /* Max devices (1 byte) */
    buffer[offset++] = MAX_PAIRED_DEVICES;

    /* Current connection ID (1 byte) */
    buffer[offset++] = (uint8_t)(g_pairing_ctx.conn_id & 0xFF);

    /* Last result (1 byte) */
    buffer[offset++] = (uint8_t)PAIRING_RESULT_SUCCESS;

    *len = offset;
}

/* ==================== COMMAND HANDLERS ==================== */

static void handle_pairing_command(uint8_t command, const uint8_t *data, uint16_t len) {
    ESP_LOGI(TAG, "Pairing command: 0x%02X, len=%d", command, len);

    switch (command) {
        case PAIRING_CMD_PAIR_REQUEST: {
            /* Check if we have space */
            int free_slot = find_free_slot();
            if (free_slot < 0) {
                ESP_LOGW(TAG, "Pairing request rejected: max devices reached");
                g_pairing_ctx.state = PAIRING_STATE_REJECTED;
                break;
            }

            /* Generate challenge */
            generate_challenge(&g_pairing_ctx.current_challenge);
            g_pairing_ctx.state = PAIRING_STATE_CHALLENGE_SENT;
            ESP_LOGI(TAG, "Challenge sent, waiting for response");
            break;
        }

        case PAIRING_CMD_UNPAIR: {
            /* Data should contain device_id (16 bytes) */
            if (len < DEVICE_ID_LEN) {
                ESP_LOGW(TAG, "Invalid unpair command length");
                break;
            }

            int index = find_paired_device(data);
            if (index >= 0) {
                memset(&g_pairing_ctx.paired_devices[index], 0, sizeof(paired_device_t));
                save_to_nvs();
                ESP_LOGI(TAG, "Device unpaired successfully");
                ble_pairing_notify_state_change();
            } else {
                ESP_LOGW(TAG, "Device not found for unpairing");
            }
            break;
        }

        case PAIRING_CMD_CLEAR_ALL: {
            ESP_LOGW(TAG, "Clearing all paired devices");
            for (int i = 0; i < MAX_PAIRED_DEVICES; i++) {
                memset(&g_pairing_ctx.paired_devices[i], 0, sizeof(paired_device_t));
            }
            save_to_nvs();
            ble_pairing_notify_state_change();
            break;
        }

        default:
            ESP_LOGW(TAG, "Unknown pairing command: 0x%02X", command);
            break;
    }
}

/* ==================== GATT CALLBACKS ==================== */

static void on_device_info_read(esp_ble_gatts_cb_param_t *param) {
    uint8_t buffer[128];
    uint16_t len = 0;

    prepare_device_info_response(buffer, &len);

    esp_gatt_rsp_t rsp;
    memset(&rsp, 0, sizeof(esp_gatt_rsp_t));
    rsp.attr_value.handle = g_pairing_ctx.device_info_handle;
    rsp.attr_value.len = len;
    memcpy(rsp.attr_value.value, buffer, len);

    esp_ble_gatts_send_response(g_pairing_ctx.gatts_if,
                                param->read.conn_id,
                                param->read.trans_id,
                                ESP_GATT_OK,
                                &rsp);
}

static void on_pairing_state_read(esp_ble_gatts_cb_param_t *param) {
    uint8_t buffer[16];
    uint16_t len = 0;

    prepare_pairing_state_response(buffer, &len);

    esp_gatt_rsp_t rsp;
    memset(&rsp, 0, sizeof(esp_gatt_rsp_t));
    rsp.attr_value.handle = g_pairing_ctx.pairing_state_handle;
    rsp.attr_value.len = len;
    memcpy(rsp.attr_value.value, buffer, len);

    esp_ble_gatts_send_response(g_pairing_ctx.gatts_if,
                                param->read.conn_id,
                                param->read.trans_id,
                                ESP_GATT_OK,
                                &rsp);
}

static void on_pairing_control_write(esp_ble_gatts_cb_param_t *param) {
    if (param->write.len < 1) {
        ESP_LOGW(TAG, "Invalid pairing control write");
        return;
    }

    uint8_t command = param->write.value[0];
    const uint8_t *data = (param->write.len > 1) ? &param->write.value[1] : NULL;
    uint16_t data_len = (param->write.len > 1) ? (param->write.len - 1) : 0;

    handle_pairing_command(command, data, data_len);

    esp_ble_gatts_send_response(g_pairing_ctx.gatts_if,
                                param->write.conn_id,
                                param->write.trans_id,
                                ESP_GATT_OK,
                                NULL);

    /* Notify state change */
    ble_pairing_notify_state_change();
}

static void on_security_challenge_read(esp_ble_gatts_cb_param_t *param) {
    esp_gatt_rsp_t rsp;
    memset(&rsp, 0, sizeof(esp_gatt_rsp_t));
    rsp.attr_value.handle = g_pairing_ctx.security_challenge_handle;
    
    if (g_pairing_ctx.current_challenge.is_valid) {
        rsp.attr_value.len = CHALLENGE_LEN;
        memcpy(rsp.attr_value.value, g_pairing_ctx.current_challenge.challenge, CHALLENGE_LEN);
    } else {
        rsp.attr_value.len = 0;
    }

    esp_ble_gatts_send_response(g_pairing_ctx.gatts_if,
                                param->read.conn_id,
                                param->read.trans_id,
                                ESP_GATT_OK,
                                &rsp);
}

static void on_security_challenge_write(esp_ble_gatts_cb_param_t *param) {
    /* Expected: 16-byte challenge response + 16-byte client device_id + variable device_name */
    if (param->write.len < (CHALLENGE_LEN + DEVICE_ID_LEN)) {
        ESP_LOGW(TAG, "Invalid security challenge response length");
        g_pairing_ctx.state = PAIRING_STATE_REJECTED;
        return;
    }

    const uint8_t *response = param->write.value;
    const uint8_t *client_device_id = param->write.value + CHALLENGE_LEN;
    const char *client_device_name = (param->write.len > (CHALLENGE_LEN + DEVICE_ID_LEN)) ?
                                      (const char *)(param->write.value + CHALLENGE_LEN + DEVICE_ID_LEN) :
                                      "Unknown";

    /* Verify challenge response */
    if (!verify_challenge_response(response)) {
        ESP_LOGW(TAG, "Challenge verification failed");
        g_pairing_ctx.state = PAIRING_STATE_REJECTED;
        esp_ble_gatts_send_response(g_pairing_ctx.gatts_if, param->write.conn_id,
                                    param->write.trans_id, ESP_GATT_OK, NULL);
        return;
    }

    /* Find free slot */
    int slot = find_free_slot();
    if (slot < 0) {
        ESP_LOGW(TAG, "No free slot for pairing");
        g_pairing_ctx.state = PAIRING_STATE_REJECTED;
        esp_ble_gatts_send_response(g_pairing_ctx.gatts_if, param->write.conn_id,
                                    param->write.trans_id, ESP_GATT_OK, NULL);
        return;
    }

    /* Save paired device */
    paired_device_t *device = &g_pairing_ctx.paired_devices[slot];
    memcpy(device->device_id, client_device_id, DEVICE_ID_LEN);
    snprintf(device->device_name, DEVICE_NAME_MAX_LEN, "%s", client_device_name);
    memcpy(device->bd_addr, param->write.bda, sizeof(esp_bd_addr_t));
    device->pair_timestamp = esp_timer_get_time() / 1000000ULL; // Convert to seconds
    device->session_count = 1;
    device->is_active = true;
    device->is_valid = true;

    /* Save to NVS */
    save_to_nvs();

    g_pairing_ctx.state = PAIRING_STATE_PAIRED;
    ESP_LOGI(TAG, "Device paired successfully in slot %d", slot);
    ESP_LOGI(TAG, "Client Name: %s", device->device_name);

    esp_ble_gatts_send_response(g_pairing_ctx.gatts_if,
                                param->write.conn_id,
                                param->write.trans_id,
                                ESP_GATT_OK,
                                NULL);

    /* Notify state change */
    ble_pairing_notify_state_change();
}

/* Public GATT callback - called by stress service dispatcher */
void ble_pairing_gatts_handler(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if,
                                esp_ble_gatts_cb_param_t *param) {
    switch (event) {
        case ESP_GATTS_REG_EVT: {
            if (param->reg.status == ESP_GATT_OK && param->reg.app_id == 1) {
                g_pairing_ctx.gatts_if = gatts_if;
                ESP_LOGI(TAG, "Pairing service registered (app_id=1)");

                /* Create service */
                esp_gatt_srvc_id_t service_id = {
                    .is_primary = true,
                    .id = {
                        .inst_id = 0,
                        .uuid = {
                            .len = ESP_UUID_LEN_16,
                            .uuid = { .uuid16 = PAIRING_SERVICE_UUID }
                        }
                    }
                };
                esp_ble_gatts_create_service(gatts_if, &service_id, 10);
            }
            break;
        }

        case ESP_GATTS_CREATE_EVT: {
            /* Only process events for our GATT interface (app_id=1) */
            if (gatts_if != g_pairing_ctx.gatts_if && gatts_if != ESP_GATT_IF_NONE) {
                break;
            }
            if (param->create.status == ESP_GATT_OK) {
                g_pairing_ctx.service_handle = param->create.service_handle;
                ESP_LOGI(TAG, "Pairing service created (handle=%u)", g_pairing_ctx.service_handle);
                esp_ble_gatts_start_service(g_pairing_ctx.service_handle);

                /* Add Device Info characteristic */
                esp_bt_uuid_t char_uuid = {
                    .len = ESP_UUID_LEN_16,
                    .uuid = { .uuid16 = DEVICE_INFO_CHAR_UUID }
                };
                esp_ble_gatts_add_char(g_pairing_ctx.service_handle, &char_uuid,
                                       ESP_GATT_PERM_READ,
                                       ESP_GATT_CHAR_PROP_BIT_READ,
                                       NULL, NULL);
            }
            break;
        }

        case ESP_GATTS_ADD_CHAR_EVT: {
            /* Only process events for our GATT interface (app_id=1) */
            if (gatts_if != g_pairing_ctx.gatts_if && gatts_if != ESP_GATT_IF_NONE) {
                break;
            }
            if (param->add_char.status == ESP_GATT_OK) {
                ESP_LOGI(TAG, "Characteristic added (UUID=0x%04X, handle=%u)",
                         param->add_char.char_uuid.uuid.uuid16,
                         param->add_char.attr_handle);

                /* Store handle based on UUID */
                if (param->add_char.char_uuid.uuid.uuid16 == DEVICE_INFO_CHAR_UUID) {
                    g_pairing_ctx.device_info_handle = param->add_char.attr_handle;
                    
                    /* Add next characteristic: Pairing State */
                    esp_bt_uuid_t uuid = {
                        .len = ESP_UUID_LEN_16,
                        .uuid = { .uuid16 = PAIRING_STATE_CHAR_UUID }
                    };
                    esp_ble_gatts_add_char(g_pairing_ctx.service_handle, &uuid,
                                           ESP_GATT_PERM_READ,
                                           ESP_GATT_CHAR_PROP_BIT_READ | ESP_GATT_CHAR_PROP_BIT_NOTIFY,
                                           NULL, NULL);
                } else if (param->add_char.char_uuid.uuid.uuid16 == PAIRING_STATE_CHAR_UUID) {
                    g_pairing_ctx.pairing_state_handle = param->add_char.attr_handle;
                    
                    /* Add next characteristic: Pairing Control */
                    esp_bt_uuid_t uuid = {
                        .len = ESP_UUID_LEN_16,
                        .uuid = { .uuid16 = PAIRING_CONTROL_CHAR_UUID }
                    };
                    esp_ble_gatts_add_char(g_pairing_ctx.service_handle, &uuid,
                                           ESP_GATT_PERM_WRITE,
                                           ESP_GATT_CHAR_PROP_BIT_WRITE,
                                           NULL, NULL);
                } else if (param->add_char.char_uuid.uuid.uuid16 == PAIRING_CONTROL_CHAR_UUID) {
                    g_pairing_ctx.pairing_control_handle = param->add_char.attr_handle;
                    
                    /* Add last characteristic: Security Challenge */
                    esp_bt_uuid_t uuid = {
                        .len = ESP_UUID_LEN_16,
                        .uuid = { .uuid16 = SECURITY_CHALLENGE_CHAR_UUID }
                    };
                    esp_ble_gatts_add_char(g_pairing_ctx.service_handle, &uuid,
                                           ESP_GATT_PERM_READ | ESP_GATT_PERM_WRITE,
                                           ESP_GATT_CHAR_PROP_BIT_READ | ESP_GATT_CHAR_PROP_BIT_WRITE,
                                           NULL, NULL);
                } else if (param->add_char.char_uuid.uuid.uuid16 == SECURITY_CHALLENGE_CHAR_UUID) {
                    g_pairing_ctx.security_challenge_handle = param->add_char.attr_handle;
                    ESP_LOGI(TAG, "All pairing characteristics added successfully");
                }
            }
            break;
        }

        case ESP_GATTS_CONNECT_EVT: {
            /* Only process events for our GATT interface (app_id=1) */
            if (gatts_if != g_pairing_ctx.gatts_if && gatts_if != ESP_GATT_IF_NONE) {
                break;
            }
            g_pairing_ctx.conn_id = param->connect.conn_id;
            ESP_LOGI(TAG, "Client connected (conn_id=%u)", g_pairing_ctx.conn_id);
            break;
        }

        case ESP_GATTS_DISCONNECT_EVT: {
            /* Only process events for our GATT interface (app_id=1) */
            if (gatts_if != g_pairing_ctx.gatts_if && gatts_if != ESP_GATT_IF_NONE) {
                break;
            }
            ESP_LOGI(TAG, "Client disconnected");
            /* Reset pairing state */
            g_pairing_ctx.state = PAIRING_STATE_ADVERTISING;
            break;
        }

        case ESP_GATTS_READ_EVT: {
            /* Only process events for our GATT interface (app_id=1) */
            if (gatts_if != g_pairing_ctx.gatts_if && gatts_if != ESP_GATT_IF_NONE) {
                break;
            }
            if (param->read.handle == g_pairing_ctx.device_info_handle) {
                on_device_info_read(param);
            } else if (param->read.handle == g_pairing_ctx.pairing_state_handle) {
                on_pairing_state_read(param);
            } else if (param->read.handle == g_pairing_ctx.security_challenge_handle) {
                on_security_challenge_read(param);
            }
            break;
        }

        case ESP_GATTS_WRITE_EVT: {
            /* Only process events for our GATT interface (app_id=1) */
            if (gatts_if != g_pairing_ctx.gatts_if && gatts_if != ESP_GATT_IF_NONE) {
                break;
            }
            if (param->write.handle == g_pairing_ctx.pairing_control_handle) {
                on_pairing_control_write(param);
            } else if (param->write.handle == g_pairing_ctx.security_challenge_handle) {
                on_security_challenge_write(param);
            }
            break;
        }

        default:
            break;
    }
}

/* ==================== PUBLIC API IMPLEMENTATION ==================== */

int ble_pairing_get_state(pairing_state_info_t *state_info) {
    if (!g_pairing_ctx.initialized || !state_info) return -1;

    state_info->state = g_pairing_ctx.state;
    state_info->max_devices = MAX_PAIRED_DEVICES;
    state_info->current_conn_id = (uint8_t)(g_pairing_ctx.conn_id & 0xFF);
    state_info->last_result = PAIRING_RESULT_SUCCESS;

    /* Count paired devices */
    uint8_t count = 0;
    for (int i = 0; i < MAX_PAIRED_DEVICES; i++) {
        if (g_pairing_ctx.paired_devices[i].is_valid) {
            count++;
        }
    }
    state_info->paired_count = count;

    return 0;
}

int ble_pairing_get_device_info(device_info_t *info) {
    if (!g_pairing_ctx.initialized || !info) return -1;
    memcpy(info, &g_pairing_ctx.device_info, sizeof(device_info_t));
    return 0;
}

int ble_pairing_get_paired_devices(paired_device_t *devices, uint8_t max_count) {
    if (!g_pairing_ctx.initialized || !devices) return -1;

    uint8_t count = 0;
    for (int i = 0; i < MAX_PAIRED_DEVICES && count < max_count; i++) {
        if (g_pairing_ctx.paired_devices[i].is_valid) {
            memcpy(&devices[count], &g_pairing_ctx.paired_devices[i], sizeof(paired_device_t));
            count++;
        }
    }

    return count;
}

int ble_pairing_unpair_device(const uint8_t *device_id) {
    if (!g_pairing_ctx.initialized || !device_id) return -1;

    int index = find_paired_device(device_id);
    if (index < 0) return -1;

    memset(&g_pairing_ctx.paired_devices[index], 0, sizeof(paired_device_t));
    save_to_nvs();
    ble_pairing_notify_state_change();

    return 0;
}

int ble_pairing_clear_all(void) {
    if (!g_pairing_ctx.initialized) return -1;

    for (int i = 0; i < MAX_PAIRED_DEVICES; i++) {
        memset(&g_pairing_ctx.paired_devices[i], 0, sizeof(paired_device_t));
    }
    save_to_nvs();
    ble_pairing_notify_state_change();

    return 0;
}

bool ble_pairing_is_device_paired(const esp_bd_addr_t bd_addr) {
    if (!g_pairing_ctx.initialized) return false;

    for (int i = 0; i < MAX_PAIRED_DEVICES; i++) {
        if (g_pairing_ctx.paired_devices[i].is_valid &&
            memcmp(g_pairing_ctx.paired_devices[i].bd_addr, bd_addr, sizeof(esp_bd_addr_t)) == 0) {
            return true;
        }
    }
    return false;
}

int ble_pairing_notify_state_change(void) {
    if (!g_pairing_ctx.initialized || !g_pairing_ctx.notifications_enabled) {
        return 0;
    }

    uint8_t buffer[16];
    uint16_t len = 0;
    prepare_pairing_state_response(buffer, &len);

    esp_ble_gatts_send_indicate(g_pairing_ctx.gatts_if,
                                 g_pairing_ctx.conn_id,
                                 g_pairing_ctx.pairing_state_handle,
                                 len, buffer, false);

    return 0;
}

int ble_pairing_set_device_name(const char *name) {
    if (!g_pairing_ctx.initialized || !name) return -1;

    snprintf(g_pairing_ctx.device_info.device_name, DEVICE_NAME_MAX_LEN, "%s", name);
    save_to_nvs();
    ESP_LOGI(TAG, "Device name updated: %s", g_pairing_ctx.device_info.device_name);

    return 0;
}

int ble_pairing_get_device_name(char *name, size_t max_len) {
    if (!g_pairing_ctx.initialized || !name) return -1;

    snprintf(name, max_len, "%s", g_pairing_ctx.device_info.device_name);
    return 0;
}

void ble_pairing_print_status(void) {
    if (!g_pairing_ctx.initialized) {
        ESP_LOGI(TAG, "Pairing service not initialized");
        return;
    }

    ESP_LOGI(TAG, "========== BLE Pairing Status ==========");
    ESP_LOGI(TAG, "Device Name: %s", g_pairing_ctx.device_info.device_name);
    ESP_LOGI(TAG, "Device ID: %02X%02X%02X%02X...%02X%02X%02X%02X",
             g_pairing_ctx.device_info.device_id[0],
             g_pairing_ctx.device_info.device_id[1],
             g_pairing_ctx.device_info.device_id[2],
             g_pairing_ctx.device_info.device_id[3],
             g_pairing_ctx.device_info.device_id[12],
             g_pairing_ctx.device_info.device_id[13],
             g_pairing_ctx.device_info.device_id[14],
             g_pairing_ctx.device_info.device_id[15]);
    ESP_LOGI(TAG, "Firmware: %s", g_pairing_ctx.device_info.firmware_version);
    ESP_LOGI(TAG, "Hardware: %s", g_pairing_ctx.device_info.hardware_revision);
    ESP_LOGI(TAG, "State: %d", g_pairing_ctx.state);

    uint8_t count = 0;
    for (int i = 0; i < MAX_PAIRED_DEVICES; i++) {
        if (g_pairing_ctx.paired_devices[i].is_valid) {
            paired_device_t *dev = &g_pairing_ctx.paired_devices[i];
            ESP_LOGI(TAG, "Paired Device #%d:", count + 1);
            ESP_LOGI(TAG, "  Name: %s", dev->device_name);
            ESP_LOGI(TAG, "  ID: %02X%02X...%02X%02X",
                     dev->device_id[0], dev->device_id[1],
                     dev->device_id[14], dev->device_id[15]);
            ESP_LOGI(TAG, "  Active: %s", dev->is_active ? "YES" : "NO");
            ESP_LOGI(TAG, "  Sessions: %lu", dev->session_count);
            count++;
        }
    }
    ESP_LOGI(TAG, "Total Paired Devices: %d / %d", count, MAX_PAIRED_DEVICES);
    ESP_LOGI(TAG, "========================================");
}
