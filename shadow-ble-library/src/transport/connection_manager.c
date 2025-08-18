#include "shadow_ble_transport.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Global transport configuration
static shadow_transport_config_t transport_config = {
    .message_timeout_ms = 1000,
    .max_retries = 3,
    .ack_timeout_ms = 500
};

// Simple timestamp function (in milliseconds)
static uint32_t get_timestamp_ms(void) {
    // This is a simplified implementation
    // In a real system, you would use a more accurate timing source
    return (uint32_t)time(NULL) * 1000;
}

// Initialize message queue
int shadow_queue_init(shadow_message_queue_t* queue) {
    if (!queue) {
        return -1;
    }
    
    memset(queue, 0, sizeof(shadow_message_queue_t));
    return 0;
}

// Check if queue is empty
int shadow_queue_is_empty(const shadow_message_queue_t* queue) {
    if (!queue) {
        return 1; // Treat invalid queue as empty
    }
    return (queue->count == 0);
}

// Check if queue is full
int shadow_queue_is_full(const shadow_message_queue_t* queue) {
    if (!queue) {
        return 1; // Treat invalid queue as full
    }
    return (queue->count >= SHADOW_MSG_QUEUE_SIZE);
}

// Push message to queue
int shadow_queue_push(shadow_message_queue_t* queue, shadow_message_t* message, shadow_msg_priority_t priority) {
    if (!queue || !message) {
        return -1;
    }
    
    if (shadow_queue_is_full(queue)) {
        return -2; // Queue full
    }
    
    // For simplicity, we'll add to the end of the queue
    // In a more advanced implementation, we would sort by priority
    uint16_t index = (queue->head + queue->count) % SHADOW_MSG_QUEUE_SIZE;
    
    queue->messages[index].message = message;
    queue->messages[index].priority = priority;
    queue->messages[index].timestamp = get_timestamp_ms();
    queue->messages[index].retry_count = 0;
    queue->messages[index].ack_received = 0;
    
    queue->count++;
    return 0;
}

// Pop message from queue
shadow_message_t* shadow_queue_pop(shadow_message_queue_t* queue) {
    if (!queue || shadow_queue_is_empty(queue)) {
        return NULL;
    }
    
    // For simplicity, we'll pop from the front
    // In a more advanced implementation, we would select by priority
    shadow_message_t* message = queue->messages[queue->head].message;
    
    // Clear the entry
    queue->messages[queue->head].message = NULL;
    queue->messages[queue->head].priority = SHADOW_MSG_PRIORITY_LOW;
    queue->messages[queue->head].timestamp = 0;
    queue->messages[queue->head].retry_count = 0;
    queue->messages[queue->head].ack_received = 0;
    
    queue->head = (queue->head + 1) % SHADOW_MSG_QUEUE_SIZE;
    queue->count--;
    
    return message;
}

// Remove message from queue by ID
int shadow_queue_remove(shadow_message_queue_t* queue, uint32_t message_id) {
    if (!queue || shadow_queue_is_empty(queue)) {
        return -1;
    }
    
    // Search for the message
    for (uint16_t i = 0; i < queue->count; i++) {
        uint16_t index = (queue->head + i) % SHADOW_MSG_QUEUE_SIZE;
        if (queue->messages[index].message && 
            queue->messages[index].message->message_id == message_id) {
            
            // Free the message
            shadow_msg_destroy(queue->messages[index].message);
            
            // Shift remaining messages
            for (uint16_t j = i; j < queue->count - 1; j++) {
                uint16_t current = (queue->head + j) % SHADOW_MSG_QUEUE_SIZE;
                uint16_t next = (queue->head + j + 1) % SHADOW_MSG_QUEUE_SIZE;
                queue->messages[current] = queue->messages[next];
            }
            
            // Clear the last entry
            uint16_t last_index = (queue->head + queue->count - 1) % SHADOW_MSG_QUEUE_SIZE;
            memset(&queue->messages[last_index], 0, sizeof(shadow_queued_message_t));
            
            queue->count--;
            return 0;
        }
    }
    
    return -2; // Message not found
}

// Set transport configuration
void shadow_transport_set_config(const shadow_transport_config_t* config) {
    if (config) {
        transport_config = *config;
    }
}

// Get transport configuration
const shadow_transport_config_t* shadow_transport_get_config(void) {
    return &transport_config;
}

// Wait for acknowledgment (simplified implementation)
int shadow_ack_wait(uint32_t message_id, uint32_t timeout_ms) {
    // In a real implementation, this would wait for an actual ACK
    // For now, we'll simulate a successful acknowledgment
    // In practice, this would involve:
    // 1. Registering a callback for the message ID
    // 2. Waiting for the callback or timeout
    // 3. Returning success or failure
    
    // Simulate some processing time
    // In a real system, you would use a proper synchronization mechanism
    return 0; // Simulate success
}

// Send acknowledgment (simplified implementation)
int shadow_ack_send(uint32_t message_id, int success) {
    // In a real implementation, this would send an actual ACK/NACK
    // For now, we'll just return success
    return 0;
}

// Retransmit message (simplified implementation)
int shadow_retransmit_message(uint32_t message_id) {
    // In a real implementation, this would requeue the message for transmission
    // For now, we'll just return success
    return 0;
}

// Handle timeout (simplified implementation)
int shadow_handle_timeout(uint32_t message_id) {
    // In a real implementation, this would check retry count and retransmit if needed
    // For now, we'll just return success
    return 0;
}