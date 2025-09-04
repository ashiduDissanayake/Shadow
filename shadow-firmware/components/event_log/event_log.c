/*
 * ESP32-S3 Event Logging System Implementation
 * 
 * Ring buffer-based event logging system for stress state transitions
 * with thread-safe operations and reliable sequence number management.
 */

#include "event_log.h"
#include <string.h>
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "EventLog";

// === PRIVATE HELPER FUNCTIONS ===

/**
 * Get next sequence number (handles wraparound, skips 0)
 */
static uint8_t get_next_sequence_number(uint8_t current) {
    uint8_t next = current + 1;
    return (next == EVENT_LOG_INVALID_SEQUENCE) ? 1 : next;
}

/**
 * Calculate ring buffer index from sequence number
 */
static uint8_t sequence_to_index(event_log_context_t *ctx, uint8_t sequence) {
    // Simple linear search for now (could be optimized with modular arithmetic)
    for (uint8_t i = 0; i < ctx->count; i++) {
        uint8_t idx = (ctx->head - ctx->count + i + EVENT_LOG_CAPACITY) % EVENT_LOG_CAPACITY;
        if (ctx->events[idx].sequence_number == sequence) {
            return idx;
        }
    }
    return EVENT_LOG_CAPACITY; // Not found
}

/**
 * Check if sequence number is in valid range
 */
static bool is_valid_sequence(uint8_t sequence) {
    return sequence != EVENT_LOG_INVALID_SEQUENCE;
}

// === PUBLIC API IMPLEMENTATION ===

int event_log_init(event_log_context_t *ctx) {
    if (!ctx) {
        ESP_LOGE(TAG, "Invalid context pointer");
        return -1;
    }
    
    // Clear context structure
    memset(ctx, 0, sizeof(event_log_context_t));
    
    // Create mutex for thread safety
    ctx->mutex = xSemaphoreCreateMutex();
    if (ctx->mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create event log mutex");
        return -1;
    }
    
    // Initialize event log state
    ctx->head = 0;
    ctx->count = 0;
    ctx->current_sequence = 0; // Will become 1 on first event
    ctx->last_acknowledged_sequence = EVENT_LOG_INVALID_SEQUENCE;
    ctx->total_events_logged = 0;
    ctx->events_overwritten = 0;
    ctx->initialized = true;
    
    ESP_LOGI(TAG, "✅ Event logging system initialized");
    ESP_LOGI(TAG, "   Ring buffer capacity: %d events", EVENT_LOG_CAPACITY);
    ESP_LOGI(TAG, "   Memory usage: %zu bytes", sizeof(stress_event_t) * EVENT_LOG_CAPACITY);
    
    return 0;
}

void event_log_deinit(event_log_context_t *ctx) {
    if (!ctx || !ctx->initialized) return;
    
    if (ctx->mutex != NULL) {
        vSemaphoreDelete(ctx->mutex);
        ctx->mutex = NULL;
    }
    
    ctx->initialized = false;
    ESP_LOGI(TAG, "Event logging system deinitialized");
}

uint8_t event_log_add_transition(event_log_context_t *ctx,
                                const stress_state_transition_t *transition,
                                uint8_t sensor_quality,
                                uint16_t battery_mv) {
    
    if (!ctx || !ctx->initialized || !transition) {
        ESP_LOGW(TAG, "Invalid parameters for add_transition");
        return EVENT_LOG_INVALID_SEQUENCE;
    }
    
    // Take mutex for thread safety
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) != pdTRUE) {
        ESP_LOGW(TAG, "Failed to take event log mutex");
        return EVENT_LOG_INVALID_SEQUENCE;
    }
    
    // Generate next sequence number
    ctx->current_sequence = get_next_sequence_number(ctx->current_sequence);
    uint8_t new_sequence = ctx->current_sequence;
    
    // Prepare new event
    stress_event_t new_event = {
        .timestamp_ms = transition->timestamp_ms,
        .new_state = transition->to_state,
        .sequence_number = new_sequence,
        .duration_prev_state_ms = transition->duration_prev_state_ms,
        .confidence_score = transition->confidence_score,
        .sensor_quality = sensor_quality,
        .battery_mv = battery_mv
    };
    
    // Check for overflow condition
    bool will_overflow = (ctx->count >= EVENT_LOG_CAPACITY);
    if (will_overflow) {
        ctx->events_overwritten++;
        ESP_LOGW(TAG, "⚠️  Event log overflow! Overwriting oldest event (total overwritten: %lu)",
                 ctx->events_overwritten);
    } else {
        ctx->count++;
    }
    
    // Add event to ring buffer
    ctx->events[ctx->head] = new_event;
    ctx->head = (ctx->head + 1) % EVENT_LOG_CAPACITY;
    ctx->total_events_logged++;
    
    xSemaphoreGive(ctx->mutex);
    
    ESP_LOGI(TAG, "📝 Event logged: seq=%d, %s, quality=%d%%, battery=%dmV", 
             new_sequence,
             stress_fsm_state_to_string(transition->to_state),
             sensor_quality,
             battery_mv);
    
    return new_sequence;
}

bool event_log_get_latest_event(event_log_context_t *ctx, stress_event_t *event) {
    if (!ctx || !ctx->initialized || !event) return false;
    
    bool success = false;
    
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        if (ctx->count > 0) {
            // Get the most recent event (just before head)
            uint8_t latest_idx = (ctx->head - 1 + EVENT_LOG_CAPACITY) % EVENT_LOG_CAPACITY;
            *event = ctx->events[latest_idx];
            success = true;
        }
        xSemaphoreGive(ctx->mutex);
    }
    
    return success;
}

bool event_log_get_event_by_sequence(event_log_context_t *ctx,
                                     uint8_t sequence_number,
                                     stress_event_t *event) {
    
    if (!ctx || !ctx->initialized || !event || !is_valid_sequence(sequence_number)) {
        return false;
    }
    
    bool success = false;
    
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        uint8_t idx = sequence_to_index(ctx, sequence_number);
        if (idx < EVENT_LOG_CAPACITY) {
            *event = ctx->events[idx];
            success = true;
        }
        xSemaphoreGive(ctx->mutex);
    }
    
    return success;
}

uint8_t event_log_get_events_from_sequence(event_log_context_t *ctx,
                                          uint8_t start_sequence,
                                          stress_event_t *events,
                                          uint8_t max_events) {
    
    if (!ctx || !ctx->initialized || !events || max_events == 0) {
        return 0;
    }
    
    uint8_t events_copied = 0;
    
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        
        // Walk through ring buffer from oldest to newest
        for (uint8_t i = 0; i < ctx->count && events_copied < max_events; i++) {
            uint8_t idx = (ctx->head - ctx->count + i + EVENT_LOG_CAPACITY) % EVENT_LOG_CAPACITY;
            stress_event_t *candidate = &ctx->events[idx];
            
            // Check if this event's sequence is >= start_sequence
            // Handle wraparound by checking if sequence difference is reasonable
            if (is_valid_sequence(start_sequence)) {
                uint8_t diff = event_log_sequence_diff(candidate->sequence_number, start_sequence);
                if (diff <= 128) { // Reasonable difference (handles wraparound)
                    events[events_copied] = *candidate;
                    events_copied++;
                }
            } else {
                // No start sequence specified, return all events
                events[events_copied] = *candidate;
                events_copied++;
            }
        }
        
        xSemaphoreGive(ctx->mutex);
    }
    
    ESP_LOGI(TAG, "Retrieved %d events starting from sequence %d", events_copied, start_sequence);
    return events_copied;
}

bool event_log_acknowledge_sequence(event_log_context_t *ctx, uint8_t sequence_number) {
    if (!ctx || !ctx->initialized || !is_valid_sequence(sequence_number)) {
        return false;
    }
    
    bool success = false;
    
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        // Update last acknowledged sequence
        ctx->last_acknowledged_sequence = sequence_number;
        success = true;
        xSemaphoreGive(ctx->mutex);
        
        ESP_LOGI(TAG, "✅ Host acknowledged sequence %d", sequence_number);
    }
    
    return success;
}

bool event_log_get_statistics(event_log_context_t *ctx, event_log_stats_t *stats) {
    if (!ctx || !ctx->initialized || !stats) return false;
    
    bool success = false;
    
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        stats->events_available = ctx->count;
        stats->buffer_usage_percent = (ctx->count * 100) / EVENT_LOG_CAPACITY;
        stats->total_events_logged = ctx->total_events_logged;
        stats->events_overwritten = ctx->events_overwritten;
        stats->current_sequence = ctx->current_sequence;
        stats->last_acknowledged_sequence = ctx->last_acknowledged_sequence;
        
        // Calculate unacknowledged events
        if (is_valid_sequence(ctx->last_acknowledged_sequence) && ctx->count > 0) {
            stats->events_unacknowledged = event_log_sequence_diff(
                ctx->current_sequence, ctx->last_acknowledged_sequence);
        } else {
            stats->events_unacknowledged = ctx->count;
        }
        
        success = true;
        xSemaphoreGive(ctx->mutex);
    }
    
    return success;
}

bool event_log_has_unacknowledged_events(event_log_context_t *ctx) {
    if (!ctx || !ctx->initialized) return false;
    
    bool has_unacked = false;
    
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        if (ctx->count > 0) {
            if (!is_valid_sequence(ctx->last_acknowledged_sequence)) {
                has_unacked = true; // No acknowledgments received yet
            } else {
                // Check if current sequence is ahead of acknowledged
                uint8_t diff = event_log_sequence_diff(ctx->current_sequence, 
                                                      ctx->last_acknowledged_sequence);
                has_unacked = (diff > 0);
            }
        }
        xSemaphoreGive(ctx->mutex);
    }
    
    return has_unacked;
}

uint8_t event_log_get_latest_sequence(event_log_context_t *ctx) {
    if (!ctx || !ctx->initialized) return EVENT_LOG_INVALID_SEQUENCE;
    
    uint8_t sequence = EVENT_LOG_INVALID_SEQUENCE;
    
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        sequence = ctx->current_sequence;
        xSemaphoreGive(ctx->mutex);
    }
    
    return sequence;
}

uint8_t event_log_sequence_diff(uint8_t newer_seq, uint8_t older_seq) {
    // Handle wraparound correctly
    if (newer_seq >= older_seq) {
        return newer_seq - older_seq;
    } else {
        // Wraparound case: newer_seq wrapped around from 255 to 1
        return (255 - older_seq) + newer_seq;
    }
}

void event_log_reset(event_log_context_t *ctx) {
    if (!ctx || !ctx->initialized) return;
    
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        ctx->head = 0;
        ctx->count = 0;
        ctx->current_sequence = 0;
        ctx->last_acknowledged_sequence = EVENT_LOG_INVALID_SEQUENCE;
        // Keep statistics for debugging
        xSemaphoreGive(ctx->mutex);
    }
    
    ESP_LOGI(TAG, "Event log reset");
}

void event_log_print_status(event_log_context_t *ctx) {
    if (!ctx || !ctx->initialized) {
        ESP_LOGI(TAG, "❌ Event log not initialized");
        return;
    }
    
    event_log_stats_t stats;
    if (event_log_get_statistics(ctx, &stats)) {
        ESP_LOGI(TAG, "=== Event Log Status ===");
        ESP_LOGI(TAG, "Events stored: %d/%d (%d%%)", 
                 stats.events_available, EVENT_LOG_CAPACITY, stats.buffer_usage_percent);
        ESP_LOGI(TAG, "Current sequence: %d", stats.current_sequence);
        ESP_LOGI(TAG, "Total logged: %lu (overwritten: %lu)", 
                 stats.total_events_logged, stats.events_overwritten);
    }
}
