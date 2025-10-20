/**
 * @file time_sync.h
 * @brief Time Synchronization System for Shadow Stress Detection
 * 
 * Synchronizes ESP32 system time with macOS host via BLE.
 * Converts boot-time ticks to real-world Unix timestamps.
 * 
 * Features:
 * - Receive Unix timestamp from macOS during BLE connection
 * - Calculate offset between boot time and real time
 * - Provide real-world timestamps for event logging
 * - Sync display RTC with real time
 */

#ifndef TIME_SYNC_H
#define TIME_SYNC_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==================== STRUCTURES ==================== */

/**
 * @brief Time synchronization context
 */
typedef struct {
    bool is_synced;                 // Whether time has been synced
    int64_t unix_epoch_us;          // Unix timestamp in microseconds when synced
    int64_t boot_time_us;           // esp_timer_get_time() when synced
    int32_t timezone_offset_sec;    // Timezone offset in seconds (e.g., -28800 for PST)
} time_sync_context_t;

/* ==================== PUBLIC API ==================== */

/**
 * @brief Initialize time synchronization system
 * @return 0 on success, negative on error
 */
int time_sync_init(void);

/**
 * @brief Set time from macOS host
 * @param unix_timestamp_ms Unix timestamp in milliseconds
 * @param timezone_offset_sec Timezone offset in seconds
 * @return 0 on success, negative on error
 */
int time_sync_set_time(uint64_t unix_timestamp_ms, int32_t timezone_offset_sec);

/**
 * @brief Get current real-world timestamp
 * @return Unix timestamp in milliseconds (0 if not synced)
 */
uint64_t time_sync_get_timestamp_ms(void);

/**
 * @brief Get current real-world timestamp in microseconds
 * @return Unix timestamp in microseconds (0 if not synced)
 */
int64_t time_sync_get_timestamp_us(void);

/**
 * @brief Check if time is synchronized
 * @return true if synced, false otherwise
 */
bool time_sync_is_synced(void);

/**
 * @brief Convert boot-time milliseconds to Unix timestamp
 * @param boot_time_ms Boot time in milliseconds (from xTaskGetTickCount())
 * @return Unix timestamp in milliseconds (0 if not synced)
 */
uint64_t time_sync_boot_to_unix_ms(uint32_t boot_time_ms);

/**
 * @brief Get current local time (with timezone)
 * @param tm Pointer to tm structure to fill
 * @return 0 on success, negative on error
 */
int time_sync_get_local_time(struct tm *tm);

/**
 * @brief Get formatted time string
 * @param buffer Buffer to store formatted string
 * @param size Size of buffer
 * @param format Format string (strftime compatible, NULL for default)
 * @return Number of characters written, or negative on error
 */
int time_sync_format_time(char *buffer, size_t size, const char *format);

/**
 * @brief Reset time synchronization (e.g., on disconnect)
 */
void time_sync_reset(void);

#ifdef __cplusplus
}
#endif

#endif // TIME_SYNC_H
