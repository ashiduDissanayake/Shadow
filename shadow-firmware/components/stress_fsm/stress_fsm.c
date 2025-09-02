/*
 * ESP32-S3 Stress State Machine (FSM) Implementation
 * 
 * This implements the confirmation FSM that stabilizes ML output and ensures
 * only deliberate, confirmed state changes are reported to the host.
 */

#include "stress_fsm.h"
#include <string.h>
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "StressFSM";

// === PRIVATE HELPER FUNCTIONS ===

/**
 * Get current time in milliseconds (device uptime)
 */
static uint32_t get_current_time_ms(void) {
    return (uint32_t)(esp_timer_get_time() / 1000);
}

/**
 * Check if ML probability indicates stress
 */
static bool is_stress_indicated(float ml_probability) {
    return ml_probability >= STRESS_THRESHOLD;
}

/**
 * Determine target state based on ML probability
 */
static stress_fsm_state_t get_target_state_from_ml(float ml_probability) {
    return is_stress_indicated(ml_probability) ? FSM_STABLE_STRESS : FSM_STABLE_CALM;
}

/**
 * Get the intermediate (suspect) state for a given target
 */
static stress_fsm_state_t get_intermediate_state(stress_fsm_state_t target_state) {
    return (target_state == FSM_STABLE_STRESS) ? FSM_SUSPECT_STRESS : FSM_SUSPECT_CALM;
}

/**
 * Get required consecutive count based on target state and hysteresis
 */
static uint8_t get_required_consecutive_count(stress_fsm_state_t target_state) {
    if (target_state == FSM_STABLE_CALM && FSM_HYSTERESIS_ENABLED) {
        return CONSECUTIVE_TO_CALM;
    }
    return CONSECUTIVE_REQUIRED;
}

/**
 * Create and fire a state transition event
 */
static void fire_transition_event(stress_fsm_context_t *ctx, 
                                 stress_fsm_state_t old_state,
                                 stress_fsm_state_t new_state,
                                 float confidence_score,
                                 uint32_t timestamp_ms,
                                 stress_fsm_event_callback_t callback) {
    
    if (!callback) return;
    
    stress_state_transition_t transition = {
        .from_state = old_state,
        .to_state = new_state,
        .timestamp_ms = timestamp_ms,
        .confidence_score = confidence_score,
        .duration_prev_state_ms = timestamp_ms - ctx->state_entry_time_ms,
        .consecutive_count = ctx->consecutive_count
    };
    
    ESP_LOGI(TAG, "🔄 State Transition: %s → %s (confidence: %.3f, duration: %lums)", 
             stress_fsm_state_to_string(old_state),
             stress_fsm_state_to_string(new_state),
             confidence_score,
             transition.duration_prev_state_ms);
    
    callback(&transition);
}

// === PUBLIC API IMPLEMENTATION ===

int stress_fsm_init(stress_fsm_context_t *ctx) {
    if (!ctx) {
        ESP_LOGE(TAG, "Invalid context pointer");
        return -1;
    }
    
    // Clear context structure
    memset(ctx, 0, sizeof(stress_fsm_context_t));
    
    // Create mutex for thread safety
    ctx->mutex = xSemaphoreCreateMutex();
    if (ctx->mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create FSM mutex");
        return -1;
    }
    
    // Initialize FSM to stable calm state
    uint32_t current_time = get_current_time_ms();
    ctx->current_state = FSM_STABLE_CALM;
    ctx->consecutive_count = 0;
    ctx->last_transition_time_ms = current_time;
    ctx->state_entry_time_ms = current_time;
    ctx->initialized = true;
    
    ESP_LOGI(TAG, "✅ Stress FSM initialized");
    ESP_LOGI(TAG, "   Stress threshold: %.2f", STRESS_THRESHOLD);
    ESP_LOGI(TAG, "   Consecutive required: %d", CONSECUTIVE_REQUIRED);
    ESP_LOGI(TAG, "   Hysteresis enabled: %s (to_calm: %d)", 
             FSM_HYSTERESIS_ENABLED ? "YES" : "NO", CONSECUTIVE_TO_CALM);
    
    return 0;
}

void stress_fsm_deinit(stress_fsm_context_t *ctx) {
    if (!ctx || !ctx->initialized) return;
    
    if (ctx->mutex != NULL) {
        vSemaphoreDelete(ctx->mutex);
        ctx->mutex = NULL;
    }
    
    ctx->initialized = false;
    ESP_LOGI(TAG, "Stress FSM deinitialized");
}

bool stress_fsm_process_inference(stress_fsm_context_t *ctx, 
                                 float ml_probability, 
                                 uint32_t timestamp_ms,
                                 stress_fsm_event_callback_t callback) {
    
    if (!ctx || !ctx->initialized) {
        ESP_LOGW(TAG, "FSM not initialized");
        return false;
    }
    
    // Take mutex for thread safety
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) != pdTRUE) {
        ESP_LOGW(TAG, "Failed to take FSM mutex");
        return false;
    }
    
    bool transition_occurred = false;
    stress_fsm_state_t old_state = ctx->current_state;
    
    // Determine what the ML model is suggesting
    stress_fsm_state_t ml_target_state = get_target_state_from_ml(ml_probability);
    
    ESP_LOGD(TAG, "Processing: ML=%.3f → %s, Current=%s, Consecutive=%d", 
             ml_probability,
             stress_fsm_state_to_string(ml_target_state),
             stress_fsm_state_to_string(ctx->current_state),
             ctx->consecutive_count);
    
    switch (ctx->current_state) {
        
        case FSM_STABLE_CALM:
            if (ml_target_state == FSM_STABLE_STRESS) {
                // Start suspecting stress
                ctx->current_state = FSM_SUSPECT_STRESS;
                ctx->consecutive_count = 1;
                ctx->state_entry_time_ms = timestamp_ms;
                ESP_LOGD(TAG, "STABLE_CALM → SUSPECT_STRESS (count=1)");
            }
            // Stay in STABLE_CALM if ML agrees
            break;
            
        case FSM_SUSPECT_STRESS:
            if (ml_target_state == FSM_STABLE_STRESS) {
                // Continue building evidence for stress
                ctx->consecutive_count++;
                if (ctx->consecutive_count >= CONSECUTIVE_REQUIRED) {
                    // Confirmed stress transition!
                    ctx->current_state = FSM_STABLE_STRESS;
                    ctx->state_entry_time_ms = timestamp_ms;
                    ctx->last_transition_time_ms = timestamp_ms;
                    transition_occurred = true;
                    
                    fire_transition_event(ctx, old_state, FSM_STABLE_STRESS, 
                                         ml_probability, timestamp_ms, callback);
                    ESP_LOGI(TAG, "✅ CONFIRMED: CALM → STRESS (after %d consecutive)", 
                             ctx->consecutive_count);
                }
            } else {
                // False alarm - return to calm
                ctx->current_state = FSM_STABLE_CALM;
                ctx->consecutive_count = 0;
                ctx->state_entry_time_ms = timestamp_ms;
                ESP_LOGD(TAG, "False alarm: SUSPECT_STRESS → STABLE_CALM");
            }
            break;
            
        case FSM_STABLE_STRESS:
            if (ml_target_state == FSM_STABLE_CALM) {
                // Start suspecting return to calm
                ctx->current_state = FSM_SUSPECT_CALM;
                ctx->consecutive_count = 1;
                ctx->state_entry_time_ms = timestamp_ms;
                ESP_LOGD(TAG, "STABLE_STRESS → SUSPECT_CALM (count=1)");
            }
            // Stay in STABLE_STRESS if ML agrees
            break;
            
        case FSM_SUSPECT_CALM:
            if (ml_target_state == FSM_STABLE_CALM) {
                // Continue building evidence for calm
                ctx->consecutive_count++;
                uint8_t required = get_required_consecutive_count(FSM_STABLE_CALM);
                if (ctx->consecutive_count >= required) {
                    // Confirmed return to calm!
                    ctx->current_state = FSM_STABLE_CALM;
                    ctx->state_entry_time_ms = timestamp_ms;
                    ctx->last_transition_time_ms = timestamp_ms;
                    transition_occurred = true;
                    
                    fire_transition_event(ctx, old_state, FSM_STABLE_CALM, 
                                         ml_probability, timestamp_ms, callback);
                    ESP_LOGI(TAG, "✅ CONFIRMED: STRESS → CALM (after %d consecutive)", 
                             ctx->consecutive_count);
                }
            } else {
                // Still stressed - return to stable stress
                ctx->current_state = FSM_STABLE_STRESS;
                ctx->consecutive_count = 0;
                ctx->state_entry_time_ms = timestamp_ms;
                ESP_LOGD(TAG, "Still stressed: SUSPECT_CALM → STABLE_STRESS");
            }
            break;
    }
    
    xSemaphoreGive(ctx->mutex);
    return transition_occurred;
}

stress_fsm_state_t stress_fsm_get_current_state(stress_fsm_context_t *ctx) {
    if (!ctx || !ctx->initialized) {
        return FSM_STABLE_CALM; // Safe default
    }
    
    stress_fsm_state_t state = FSM_STABLE_CALM;
    
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        state = ctx->current_state;
        xSemaphoreGive(ctx->mutex);
    }
    
    // Only return stable states for external consumption
    if (!stress_fsm_is_stable_state(state)) {
        // If we're in an intermediate state, return the last stable state
        // This ensures external consumers only see confirmed states
        return (state == FSM_SUSPECT_STRESS) ? FSM_STABLE_CALM : FSM_STABLE_STRESS;
    }
    
    return state;
}

uint32_t stress_fsm_get_state_duration(stress_fsm_context_t *ctx, uint32_t current_time_ms) {
    if (!ctx || !ctx->initialized) {
        return 0;
    }
    
    uint32_t duration = 0;
    
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        duration = current_time_ms - ctx->state_entry_time_ms;
        xSemaphoreGive(ctx->mutex);
    }
    
    return duration;
}

const char* stress_fsm_state_to_string(stress_fsm_state_t state) {
    switch (state) {
        case FSM_STABLE_CALM:    return "STABLE_CALM";
        case FSM_SUSPECT_STRESS: return "SUSPECT_STRESS";
        case FSM_STABLE_STRESS:  return "STABLE_STRESS";
        case FSM_SUSPECT_CALM:   return "SUSPECT_CALM";
        default:                 return "UNKNOWN";
    }
}

void stress_fsm_reset(stress_fsm_context_t *ctx, uint32_t timestamp_ms) {
    if (!ctx || !ctx->initialized) return;
    
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        ctx->current_state = FSM_STABLE_CALM;
        ctx->consecutive_count = 0;
        ctx->last_transition_time_ms = timestamp_ms;
        ctx->state_entry_time_ms = timestamp_ms;
        xSemaphoreGive(ctx->mutex);
    }
    
    ESP_LOGI(TAG, "FSM reset to STABLE_CALM");
}

bool stress_fsm_get_debug_info(stress_fsm_context_t *ctx, 
                              uint8_t *consecutive_count, 
                              uint32_t *time_in_state_ms) {
    if (!ctx || !ctx->initialized || !consecutive_count || !time_in_state_ms) {
        return false;
    }
    
    bool success = false;
    uint32_t current_time = get_current_time_ms();
    
    if (xSemaphoreTake(ctx->mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        *consecutive_count = ctx->consecutive_count;
        *time_in_state_ms = current_time - ctx->state_entry_time_ms;
        success = true;
        xSemaphoreGive(ctx->mutex);
    }
    
    return success;
}
