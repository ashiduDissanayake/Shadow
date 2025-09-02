/*
 * ESP32-S3 Event Logging System
 * Part of the ESP32-Host Stress Monitor Communication System
 * 
 * This module implements a ring buffer-based event logging system that stores
 * confirmed stress state transitions for reliable communication with the host.
 * 
 * Key Features:
 * - Ring buffer with 32 event capacity
 * - Atomic sequence number generation
 * - Thread-safe operations
 * - Memory efficient storage
 * - Graceful overflow handling
 */

#ifndef EVENT_LOG_H
#define EVENT_LOG_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "stress_fsm.h"

// Event Log Configuration
#define EVENT_LOG_CAPACITY          32      // Maximum events in ring buffer
#define EVENT_LOG_INVALID_SEQUENCE  0       // Invalid/uninitialized sequence number

// Stress Event Structure (matches design specification)
typedef struct {
    uint32_t timestamp_ms;                  // Device uptime when event occurred
    stress_fsm_state_t new_state;           // The new confirmed state
    uint8_t sequence_number;                // Unique sequence number (wraps at 255)
    uint32_t duration_prev_state_ms;        // Duration of previous state
    float confidence_score;                 // ML model confidence score
    uint8_t sensor_quality;                 // Sensor data quality indicator (0-100)
    uint16_t battery_mv;                    // Battery voltage at event time
} stress_event_t;

// Event Log Context
typedef struct {
    stress_event_t events[EVENT_LOG_CAPACITY];  // Ring buffer of events
    uint8_t head;                              // Write position (next event slot)
    uint8_t count;                             // Number of valid events
    uint8_t current_sequence;                  // Current sequence number
    uint8_t last_acknowledged_sequence;        // Last sequence ACKed by host
    SemaphoreHandle_t mutex;                   // Thread safety
    bool initialized;                          // Initialization flag
    uint32_t total_events_logged;              // Statistics: total events ever logged
    uint32_t events_overwritten;               // Statistics: events lost due to overflow
} event_log_context_t;

// Event Log Statistics
typedef struct {
    uint8_t events_available;                  // Number of events currently stored
    uint8_t events_unacknowledged;             // Number of events not yet ACKed by host
    uint8_t buffer_usage_percent;              // Buffer utilization percentage
    uint32_t total_events_logged;              // Total events ever logged
    uint32_t events_overwritten;               // Events lost due to overflow
    uint8_t current_sequence;                  // Current sequence number
    uint8_t last_acknowledged_sequence;        // Last ACKed sequence
} event_log_stats_t;

// === CORE EVENT LOGGING FUNCTIONS ===

/**
 * Initialize the event logging system
 * 
 * @param ctx Event log context to initialize
 * @return 0 on success, -1 on error
 */
int event_log_init(event_log_context_t *ctx);

/**
 * Deinitialize the event logging system
 * 
 * @param ctx Event log context to cleanup
 */
void event_log_deinit(event_log_context_t *ctx);

/**
 * Log a new stress state transition event
 * 
 * This function should be called whenever the stress FSM confirms a state transition.
 * It automatically assigns a sequence number and handles ring buffer management.
 * 
 * @param ctx Event log context
 * @param transition FSM transition data
 * @param sensor_quality Current sensor quality (0-100)
 * @param battery_mv Current battery voltage in millivolts
 * @return Sequence number of the logged event, or 0 on error
 */
uint8_t event_log_add_transition(event_log_context_t *ctx,
                                const stress_state_transition_t *transition,
                                uint8_t sensor_quality,
                                uint16_t battery_mv);

/**
 * Get the latest event for BLE advertising (thread-safe)
 * 
 * @param ctx Event log context
 * @param event Output buffer for the latest event
 * @return true if event retrieved successfully, false if no events
 */
bool event_log_get_latest_event(event_log_context_t *ctx, stress_event_t *event);

/**
 * Get a specific event by sequence number (thread-safe)
 * 
 * @param ctx Event log context
 * @param sequence_number Sequence number to retrieve
 * @param event Output buffer for the event
 * @return true if event found and retrieved, false otherwise
 */
bool event_log_get_event_by_sequence(event_log_context_t *ctx,
                                     uint8_t sequence_number,
                                     stress_event_t *event);

/**
 * Get events starting from a specific sequence number
 * 
 * This is used for host synchronization when catching up on missed events.
 * 
 * @param ctx Event log context
 * @param start_sequence Starting sequence number (inclusive)
 * @param events Output buffer for events
 * @param max_events Maximum number of events to retrieve
 * @return Number of events retrieved
 */
uint8_t event_log_get_events_from_sequence(event_log_context_t *ctx,
                                          uint8_t start_sequence,
                                          stress_event_t *events,
                                          uint8_t max_events);

/**
 * Acknowledge receipt of events up to a sequence number
 * 
 * This allows the system to track which events have been safely received
 * by the host and can potentially be garbage collected in the future.
 * 
 * @param ctx Event log context
 * @param sequence_number Latest sequence number acknowledged by host
 * @return true if acknowledgment processed successfully
 */
bool event_log_acknowledge_sequence(event_log_context_t *ctx, uint8_t sequence_number);

/**
 * Get current event log statistics (thread-safe)
 * 
 * @param ctx Event log context
 * @param stats Output buffer for statistics
 * @return true if statistics retrieved successfully
 */
bool event_log_get_statistics(event_log_context_t *ctx, event_log_stats_t *stats);

// === UTILITY FUNCTIONS ===

/**
 * Check if the event log has unacknowledged events
 * 
 * @param ctx Event log context
 * @return true if there are events not yet acknowledged by the host
 */
bool event_log_has_unacknowledged_events(event_log_context_t *ctx);

/**
 * Get the sequence number of the latest event
 * 
 * @param ctx Event log context
 * @return Latest sequence number, or EVENT_LOG_INVALID_SEQUENCE if no events
 */
uint8_t event_log_get_latest_sequence(event_log_context_t *ctx);

/**
 * Calculate sequence number difference (handles wraparound)
 * 
 * @param newer_seq Newer sequence number
 * @param older_seq Older sequence number
 * @return Number of events between older_seq and newer_seq
 */
uint8_t event_log_sequence_diff(uint8_t newer_seq, uint8_t older_seq);

/**
 * Reset event log to initial state
 * Useful for testing or system reset scenarios
 * 
 * @param ctx Event log context
 */
void event_log_reset(event_log_context_t *ctx);

/**
 * Print event log status for debugging
 * 
 * @param ctx Event log context
 */
void event_log_print_status(event_log_context_t *ctx);

#endif // EVENT_LOG_H
