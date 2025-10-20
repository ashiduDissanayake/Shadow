/**
 * @file time_sync.c
 * @brief Time Synchronization System Implementation
 */

#include "time_sync.h"
#include "esp_timer.h"
#include "esp_log.h"
#include <string.h>

static const char *TAG = "TimeSync";

// Global time sync context
static time_sync_context_t g_time_ctx = {0};

// === PUBLIC API IMPLEMENTATION ===

int time_sync_init(void) {
    memset(&g_time_ctx, 0, sizeof(time_sync_context_t));
    g_time_ctx.is_synced = false;
    
    ESP_LOGI(TAG, "Time synchronization system initialized");
    return 0;
}

int time_sync_set_time(uint64_t unix_timestamp_ms, int32_t timezone_offset_sec) {
    if (unix_timestamp_ms == 0) {
        ESP_LOGE(TAG, "Invalid timestamp (0)");
        return -1;
    }

    // Store Unix epoch and boot time at moment of sync
    g_time_ctx.unix_epoch_us = (int64_t)unix_timestamp_ms * 1000LL;
    g_time_ctx.boot_time_us = esp_timer_get_time();
    g_time_ctx.timezone_offset_sec = timezone_offset_sec;
    g_time_ctx.is_synced = true;

    // Calculate local time for logging
    struct tm timeinfo;
    time_t unix_sec = unix_timestamp_ms / 1000;
    unix_sec += timezone_offset_sec; // Apply timezone
    localtime_r(&unix_sec, &timeinfo);

    ESP_LOGI(TAG, "⏰ Time synchronized!");
    ESP_LOGI(TAG, "   Unix time: %llu ms", unix_timestamp_ms);
    ESP_LOGI(TAG, "   Local time: %04d-%02d-%02d %02d:%02d:%02d",
             timeinfo.tm_year + 1900, timeinfo.tm_mon + 1, timeinfo.tm_mday,
             timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);
    ESP_LOGI(TAG, "   Timezone: UTC%+d hours", timezone_offset_sec / 3600);
    ESP_LOGI(TAG, "   Boot time at sync: %lld us", g_time_ctx.boot_time_us);

    return 0;
}

uint64_t time_sync_get_timestamp_ms(void) {
    if (!g_time_ctx.is_synced) {
        return 0;
    }

    // Calculate elapsed time since sync
    int64_t current_boot_us = esp_timer_get_time();
    int64_t elapsed_us = current_boot_us - g_time_ctx.boot_time_us;
    
    // Add to Unix epoch
    int64_t current_unix_us = g_time_ctx.unix_epoch_us + elapsed_us;
    
    return (uint64_t)(current_unix_us / 1000LL);
}

int64_t time_sync_get_timestamp_us(void) {
    if (!g_time_ctx.is_synced) {
        return 0;
    }

    int64_t current_boot_us = esp_timer_get_time();
    int64_t elapsed_us = current_boot_us - g_time_ctx.boot_time_us;
    
    return g_time_ctx.unix_epoch_us + elapsed_us;
}

bool time_sync_is_synced(void) {
    return g_time_ctx.is_synced;
}

uint64_t time_sync_boot_to_unix_ms(uint32_t boot_time_ms) {
    if (!g_time_ctx.is_synced) {
        return 0;
    }

    // Convert boot time to microseconds
    int64_t boot_time_us = (int64_t)boot_time_ms * 1000LL;
    
    // Calculate offset from sync point
    int64_t offset_us = boot_time_us - (g_time_ctx.boot_time_us / 1000LL) * 1000LL;
    
    // Add to Unix epoch
    int64_t unix_us = g_time_ctx.unix_epoch_us + offset_us;
    
    return (uint64_t)(unix_us / 1000LL);
}

int time_sync_get_local_time(struct tm *tm) {
    if (!tm) {
        return -1;
    }

    uint64_t unix_ms = time_sync_get_timestamp_ms();
    if (unix_ms == 0) {
        ESP_LOGW(TAG, "Time not synced, cannot get local time");
        return -2;
    }

    // Convert to seconds and apply timezone
    time_t unix_sec = (time_t)(unix_ms / 1000);
    unix_sec += g_time_ctx.timezone_offset_sec;
    
    localtime_r(&unix_sec, tm);
    return 0;
}

int time_sync_format_time(char *buffer, size_t size, const char *format) {
    if (!buffer || size == 0) {
        return -1;
    }

    struct tm timeinfo;
    if (time_sync_get_local_time(&timeinfo) != 0) {
        snprintf(buffer, size, "NOT_SYNCED");
        return -2;
    }

    // Use default format if none provided
    const char *fmt = format ? format : "%Y-%m-%d %H:%M:%S";
    
    size_t written = strftime(buffer, size, fmt, &timeinfo);
    return (int)written;
}

void time_sync_reset(void) {
    ESP_LOGI(TAG, "Time synchronization reset");
    g_time_ctx.is_synced = false;
    g_time_ctx.unix_epoch_us = 0;
    g_time_ctx.boot_time_us = 0;
    g_time_ctx.timezone_offset_sec = 0;
}
