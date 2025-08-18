#include "shadow_ble_protocol.h"
#include <stdlib.h>
#include <string.h>

// Simple CRC16 implementation
uint16_t shadow_crc16(const uint8_t* data, uint16_t length) {
    uint16_t crc = 0xFFFF;
    for (uint16_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 0x0001) {
                crc >>= 1;
                crc ^= 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

// Message ID generator (simple implementation)
static uint32_t current_message_id = 1;

uint32_t shadow_msg_generate_id(void) {
    return current_message_id++;
}

// Create a new message
shadow_message_t* shadow_msg_create(shadow_msg_type_t type, const uint8_t* payload, uint16_t length) {
    shadow_message_t* message = (shadow_message_t*)malloc(sizeof(shadow_message_t));
    if (!message) {
        return NULL;
    }
    
    message->header = SHADOW_MSG_HEADER;
    message->message_id = shadow_msg_generate_id();
    message->length = length;
    message->footer = SHADOW_MSG_FOOTER;
    
    if (length > 0 && payload != NULL) {
        message->payload = (uint8_t*)malloc(length);
        if (!message->payload) {
            free(message);
            return NULL;
        }
        memcpy(message->payload, payload, length);
    } else {
        message->payload = NULL;
    }
    
    // Calculate CRC over header, message_id, length, and payload
    uint16_t crc_data_length = 4 + 2 + length; // 4 bytes for message_id, 2 for length, + payload
    uint8_t* crc_data = (uint8_t*)malloc(crc_data_length);
    if (!crc_data) {
        if (message->payload) {
            free(message->payload);
        }
        free(message);
        return NULL;
    }
    
    // Copy data for CRC calculation
    memcpy(crc_data, &message->message_id, 4);
    memcpy(crc_data + 4, &message->length, 2);
    if (message->payload && length > 0) {
        memcpy(crc_data + 6, message->payload, length);
    }
    
    message->crc16 = shadow_crc16(crc_data, crc_data_length);
    free(crc_data);
    
    return message;
}

// Parse a message from raw data
int shadow_msg_parse(const uint8_t* raw_data, uint16_t length, shadow_message_t* message) {
    if (!raw_data || !message || length < 12) { // Minimum message size
        return -1;
    }
    
    // Parse header
    memcpy(&message->header, raw_data, 2);
    if (message->header != SHADOW_MSG_HEADER) {
        return -2;
    }
    
    // Parse message ID
    memcpy(&message->message_id, raw_data + 2, 4);
    
    // Parse length
    memcpy(&message->length, raw_data + 6, 2);
    
    // Validate length
    if (length < (12 + message->length)) { // 12 = header(2) + msg_id(4) + length(2) + crc(2) + footer(2)
        return -3;
    }
    
    // Parse payload if exists
    if (message->length > 0) {
        message->payload = (uint8_t*)malloc(message->length);
        if (!message->payload) {
            return -4;
        }
        memcpy(message->payload, raw_data + 8, message->length);
    } else {
        message->payload = NULL;
    }
    
    // Parse CRC
    memcpy(&message->crc16, raw_data + 8 + message->length, 2);
    
    // Parse footer
    memcpy(&message->footer, raw_data + 10 + message->length, 2);
    if (message->footer != SHADOW_MSG_FOOTER) {
        if (message->payload) {
            free(message->payload);
            message->payload = NULL;
        }
        return -5;
    }
    
    // Validate CRC
    if (!shadow_msg_validate(message)) {
        if (message->payload) {
            free(message->payload);
            message->payload = NULL;
        }
        return -6;
    }
    
    return 0; // Success
}

// Validate message CRC
int shadow_msg_validate(const shadow_message_t* message) {
    if (!message) {
        return 0;
    }
    
    // Calculate CRC over message_id, length, and payload
    uint16_t crc_data_length = 4 + 2 + message->length; // 4 bytes for message_id, 2 for length, + payload
    uint8_t* crc_data = (uint8_t*)malloc(crc_data_length);
    if (!crc_data) {
        return 0;
    }
    
    // Copy data for CRC calculation
    memcpy(crc_data, &message->message_id, 4);
    memcpy(crc_data + 4, &message->length, 2);
    if (message->payload && message->length > 0) {
        memcpy(crc_data + 6, message->payload, message->length);
    }
    
    uint16_t calculated_crc = shadow_crc16(crc_data, crc_data_length);
    free(crc_data);
    
    return (calculated_crc == message->crc16) ? 1 : 0;
}

// Destroy message and free memory
void shadow_msg_destroy(shadow_message_t* message) {
    if (message) {
        if (message->payload) {
            free(message->payload);
            message->payload = NULL;
        }
        free(message);
    }
}