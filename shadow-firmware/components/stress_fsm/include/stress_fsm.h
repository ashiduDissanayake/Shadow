/*
 * ESP32-S3 Stress State Machine (FSM) Component
 * Part of the ESP32-Host Stress Monitor Communication System
 * 
 * This module implements the confirmation FSM that stabilizes the ML output
 * and ensures only deliberate, confirmed state changes are reported.
 * 
 * Design Principles:
 * - Minimize noise from temporary ML fluctuations
 * - Only report stable, confirmed state transitions
 * - Track consecutive inference counts for stability
 * - Generate events only on confirmed transitions
 */

#ifndef STRESS_FSM_H
#define STRESS_FSM_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

// FSM Configuration
#define STRESS_THRESHOLD                0.5f        // ML probability threshold for stress detection
#define CONSECUTIVE_REQUIRED            3           // Required consecutive inferences for state confirmation
#define FSM_HYSTERESIS_ENABLED          0           // Enable hysteresis to prevent oscillations

// If hysteresis is enabled, require more confirmations to return to calm
#if FSM_HYSTERESIS_ENABLED
#define CONSECUTIVE_TO_CALM             4           // Slightly more to return to calm (prevents oscillation)
#else
#define CONSECUTIVE_TO_CALM             CONSECUTIVE_REQUIRED
#endif

// FSM States
typedef enum {
    FSM_STABLE_CALM = 0,        // Confirmed calm state (broadcasted)
    FSM_SUSPECT_STRESS = 1,     // Intermediate: potential stress detected (internal only)
    FSM_STABLE_STRESS = 2,      // Confirmed stress state (broadcasted)
    FSM_SUSPECT_CALM = 3        // Intermediate: potential return to calm (internal only)
} stress_fsm_state_t;

// FSM State Transition Event
typedef struct {
    stress_fsm_state_t from_state;      // Previous state
    stress_fsm_state_t to_state;        // New state
    uint32_t timestamp_ms;              // Device uptime when transition occurred
    float confidence_score;             // ML confidence that triggered the transition
    uint32_t duration_prev_state_ms;    // How long we were in the previous state
    uint8_t consecutive_count;          // Number of consecutive confirmations
} stress_state_transition_t;

// FSM Internal Context
typedef struct {
    stress_fsm_state_t current_state;       // Current FSM state
    uint8_t consecutive_count;              // Consecutive inferences in current direction
    uint32_t last_transition_time_ms;       // Timestamp of last confirmed transition
    uint32_t state_entry_time_ms;           // When we entered the current state
    SemaphoreHandle_t mutex;                // Thread safety for FSM updates
    bool initialized;                       // Initialization flag
} stress_fsm_context_t;

// FSM Event Callback Function Type
// Called whenever a confirmed state transition occurs
typedef void (*stress_fsm_event_callback_t)(const stress_state_transition_t *transition);

// === CORE FSM FUNCTIONS ===

/**
 * Initialize the stress FSM system
 * 
 * @param ctx FSM context structure to initialize
 * @return 0 on success, -1 on error
 */
int stress_fsm_init(stress_fsm_context_t *ctx);

/**
 * Deinitialize the stress FSM system
 * 
 * @param ctx FSM context to cleanup
 */
void stress_fsm_deinit(stress_fsm_context_t *ctx);

/**
 * Process a new ML inference result
 * 
 * This is the core function that implements the confirmation logic.
 * It should be called every time the ML model produces a new prediction.
 * 
 * @param ctx FSM context
 * @param ml_probability Stress probability from ML model (0.0 to 1.0)
 * @param timestamp_ms Current device uptime in milliseconds
 * @param callback Optional callback function for state transitions (can be NULL)
 * @return true if a confirmed state transition occurred, false otherwise
 */
bool stress_fsm_process_inference(stress_fsm_context_t *ctx, 
                                 float ml_probability, 
                                 uint32_t timestamp_ms,
                                 stress_fsm_event_callback_t callback);

/**
 * Get the current confirmed FSM state (thread-safe)
 * 
 * @param ctx FSM context
 * @return Current confirmed state (only STABLE_CALM or STABLE_STRESS)
 */
stress_fsm_state_t stress_fsm_get_current_state(stress_fsm_context_t *ctx);

/**
 * Get time spent in current state (thread-safe)
 * 
 * @param ctx FSM context
 * @param current_time_ms Current device uptime
 * @return Duration in current state in milliseconds
 */
uint32_t stress_fsm_get_state_duration(stress_fsm_context_t *ctx, uint32_t current_time_ms);

/**
 * Check if the FSM is in a stable (broadcastable) state
 * 
 * @param state State to check
 * @return true if state is STABLE_CALM or STABLE_STRESS
 */
static inline bool stress_fsm_is_stable_state(stress_fsm_state_t state) {
    return (state == FSM_STABLE_CALM || state == FSM_STABLE_STRESS);
}

// === UTILITY FUNCTIONS ===

/**
 * Convert FSM state to human-readable string
 * 
 * @param state FSM state
 * @return String representation of the state
 */
const char* stress_fsm_state_to_string(stress_fsm_state_t state);

/**
 * Reset FSM to initial state (STABLE_CALM)
 * Useful for testing or system reset scenarios
 * 
 * @param ctx FSM context
 * @param timestamp_ms Current device uptime
 */
void stress_fsm_reset(stress_fsm_context_t *ctx, uint32_t timestamp_ms);

/**
 * Get FSM statistics for debugging
 * 
 * @param ctx FSM context
 * @param consecutive_count Output: current consecutive count
 * @param time_in_state_ms Output: time in current state
 * @return true if statistics retrieved successfully
 */
bool stress_fsm_get_debug_info(stress_fsm_context_t *ctx, 
                              uint8_t *consecutive_count, 
                              uint32_t *time_in_state_ms);

#endif // STRESS_FSM_H
