#ifndef SHADOW_BLE_TRANSPORT_H
#define SHADOW_BLE_TRANSPORT_H

#include "shadow_ble_protocol.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Message queue constants
#define SHADOW_MSG_QUEUE_SIZE 32

// Message priority levels
typedef enum {
    SHADOW_MSG_PRIORITY_HIGH = 0,
    SHADOW_MSG_PRIORITY_NORMAL = 1,
    SHADOW_MSG_PRIORITY_LOW = 2
} shadow_msg_priority_t;

// Queued message structure
typedef struct {
    shadow_message_t* message;
    shadow_msg_priority_t priority;
    uint32_t timestamp;
    uint8_t retry_count;
    uint8_t ack_received;
} shadow_queued_message_t;

// Message queue structure
typedef struct {
    shadow_queued_message_t messages[SHADOW_MSG_QUEUE_SIZE];
    uint16_t head;
    uint16_t tail;
    uint16_t count;
} shadow_message_queue_t;

// Transport configuration
typedef struct {
    uint32_t message_timeout_ms;
    uint8_t max_retries;
    uint32_t ack_timeout_ms;
} shadow_transport_config_t;

// Function prototypes

// Queue management
int shadow_queue_init(shadow_message_queue_t* queue);
int shadow_queue_push(shadow_message_queue_t* queue, shadow_message_t* message, shadow_msg_priority_t priority);
shadow_message_t* shadow_queue_pop(shadow_message_queue_t* queue);
int shadow_queue_remove(shadow_message_queue_t* queue, uint32_t message_id);
int shadow_queue_is_empty(const shadow_message_queue_t* queue);
int shadow_queue_is_full(const shadow_message_queue_t* queue);

// Acknowledgment handling
int shadow_ack_wait(uint32_t message_id, uint32_t timeout_ms);
int shadow_ack_send(uint32_t message_id, int success);

// Retransmission management
int shadow_retransmit_message(uint32_t message_id);
int shadow_handle_timeout(uint32_t message_id);

// Transport configuration
void shadow_transport_set_config(const shadow_transport_config_t* config);
const shadow_transport_config_t* shadow_transport_get_config(void);

#ifdef __cplusplus
}
#endif

#endif // SHADOW_BLE_TRANSPORT_H