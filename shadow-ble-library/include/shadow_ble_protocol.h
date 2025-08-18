#ifndef SHADOW_BLE_PROTOCOL_H
#define SHADOW_BLE_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Protocol constants
#define SHADOW_MSG_HEADER 0xAA55
#define SHADOW_MSG_FOOTER 0x55AA
#define SHADOW_MSG_MAX_SIZE 247

// Message types
typedef enum {
    SHADOW_MSG_TYPE_DATA = 0x01,
    SHADOW_MSG_TYPE_CONTROL = 0x02,
    SHADOW_MSG_TYPE_STATUS = 0x03,
    SHADOW_MSG_TYPE_ACK = 0x10,
    SHADOW_MSG_TYPE_NACK = 0x11
} shadow_msg_type_t;

// Control commands
typedef enum {
    SHADOW_CMD_START_DATA = 0x01,
    SHADOW_CMD_STOP_DATA = 0x02,
    SHADOW_CMD_SET_CONFIG = 0x03,
    SHADOW_CMD_GET_CONFIG = 0x04,
    SHADOW_CMD_SLEEP = 0x05,
    SHADOW_CMD_WAKEUP = 0x06,
    SHADOW_CMD_DISCONNECT = 0x07
} shadow_cmd_t;

// Message structure
typedef struct {
    uint16_t header;
    uint32_t message_id;
    uint16_t length;
    uint8_t* payload;
    uint16_t crc16;
    uint16_t footer;
} shadow_message_t;

// Function prototypes

// Message creation and parsing
shadow_message_t* shadow_msg_create(shadow_msg_type_t type, const uint8_t* payload, uint16_t length);
int shadow_msg_parse(const uint8_t* raw_data, uint16_t length, shadow_message_t* message);
void shadow_msg_destroy(shadow_message_t* message);

// CRC functions
uint16_t shadow_crc16(const uint8_t* data, uint16_t length);
int shadow_msg_validate(const shadow_message_t* message);

// Message ID management
uint32_t shadow_msg_generate_id(void);

#ifdef __cplusplus
}
#endif

#endif // SHADOW_BLE_PROTOCOL_H