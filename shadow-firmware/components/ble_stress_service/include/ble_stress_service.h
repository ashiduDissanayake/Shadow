#ifndef BLE_STRESS_SERVICE_H
#define BLE_STRESS_SERVICE_H

#include <stdbool.h>
#include <stdint.h>
#include "stress_fsm.h"
#include "event_log.h"

#ifdef __cplusplus
extern "C" {
#endif

int  ble_stress_service_init(stress_fsm_context_t *fsm_ctx, event_log_context_t *event_ctx);
void ble_stress_service_deinit(void);
void ble_stress_service_tick(void);

/* ===== Compatibility Stubs (legacy API) ===== */
static inline int ble_stress_service_start_advertising(void) {
    ble_stress_service_tick();
    return 0;
}
static inline int ble_stress_service_update_advertisement(uint16_t battery_mv, uint8_t sensor_quality) {
    ble_stress_service_tick();
    return 0;
}
static inline bool ble_stress_service_is_connected(void) {
    return false; /* Simplified model doesn’t expose connection detail */
}
static inline bool ble_stress_service_notifications_enabled(void) {
    return false;
}
static inline int ble_stress_service_notify_fsm_state(void) {
    return 0;
}
static inline void ble_stress_service_print_status(void) {
    /* No-op */
}
/* ============================================ */

#ifdef __cplusplus
}
#endif

#endif // BLE_STRESS_SERVICE_H