/**
 * Display Manager - Clock and QR Code Display
 * Full screen rendering with button toggle
 */

#include "display_manager.h"
#include "time_sync.h"
#include "calibration.h"
#include "esp_log.h"
#include "driver/gpio.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "esp_heap_caps.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "qrcode.h"
#include <string.h>
#include <time.h>
#include <sys/time.h>

// Battery monitoring configuration for LilyGo T-Display S3
#define BATTERY_ADC_PIN         4               // GPIO 4
#define BATTERY_ADC_CHANNEL     ADC_CHANNEL_3   // ADC1_CHANNEL_3
#define BATTERY_VOLTAGE_DIVIDER 2.0f            // Voltage divider ratio (R1=R2)
#define BATTERY_MIN_VOLTAGE     3300            // 3.3V (empty)
#define BATTERY_MAX_VOLTAGE     4200            // 4.2V (fully charged)
#define BATTERY_READ_SAMPLES    10              // Number of samples to average

static const char *TAG = "DISPLAY";

static esp_lcd_panel_handle_t panel_handle = NULL;
static esp_lcd_panel_io_handle_t io_handle = NULL;
static uint16_t *frame_buffer = NULL;
static display_mode_t current_mode = DISPLAY_MODE_CLOCK;
static display_device_info_t saved_device_info = {0};
static bool display_power_on = true;  // Display starts on

// Battery monitoring
static adc_oneshot_unit_handle_t battery_adc_handle = NULL;
static adc_cali_handle_t battery_cali_handle = NULL;
static bool battery_initialized = false;

// Forward declarations
static void clear_screen(uint16_t color);
static void draw_rect(int x, int y, int w, int h, uint16_t color);
static void draw_large_digit(int x, int y, int digit, uint16_t color);
static void draw_char(int x, int y, char c, uint16_t color, int scale);
static void draw_text(int x, int y, const char *text, uint16_t color, int scale);
static void draw_battery_icon(int x, int y, int percentage, uint16_t color);

// Helper: Clear screen with color
static void clear_screen(uint16_t color) {
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = color;
    }
}

// Helper: Draw a filled rectangle
static void draw_rect(int x, int y, int w, int h, uint16_t color) {
    for (int py = y; py < y + h && py < LCD_HEIGHT; py++) {
        for (int px = x; px < x + w && px < LCD_WIDTH; px++) {
            if (px >= 0 && py >= 0) {
                frame_buffer[py * LCD_WIDTH + px] = color;
            }
        }
    }
}

// Helper: Draw large digit (7-segment style, 40x60 pixels)
static void draw_large_digit(int x, int y, int digit, uint16_t color) {
    // Simple block digits for visibility
    int segments[10][7] = {
        {1,1,1,1,1,1,0}, // 0
        {0,1,1,0,0,0,0}, // 1
        {1,1,0,1,1,0,1}, // 2
        {1,1,1,1,0,0,1}, // 3
        {0,1,1,0,0,1,1}, // 4
        {1,0,1,1,0,1,1}, // 5
        {1,0,1,1,1,1,1}, // 6
        {1,1,1,0,0,0,0}, // 7
        {1,1,1,1,1,1,1}, // 8
        {1,1,1,1,0,1,1}  // 9
    };
    
    int w = 35, h = 55;
    int seg_h = h / 2;
    int seg_w = w;
    int thick = 5;
    
    // Top horizontal
    if (segments[digit][0]) draw_rect(x, y, seg_w, thick, color);
    // Top right vertical
    if (segments[digit][1]) draw_rect(x + seg_w - thick, y, thick, seg_h, color);
    // Bottom right vertical
    if (segments[digit][2]) draw_rect(x + seg_w - thick, y + seg_h, thick, seg_h, color);
    // Bottom horizontal
    if (segments[digit][3]) draw_rect(x, y + h - thick, seg_w, thick, color);
    // Bottom left vertical
    if (segments[digit][4]) draw_rect(x, y + seg_h, thick, seg_h, color);
    // Top left vertical
    if (segments[digit][5]) draw_rect(x, y, thick, seg_h, color);
    // Middle horizontal
    if (segments[digit][6]) draw_rect(x, y + seg_h - thick/2, seg_w, thick, color);
}

// Simple 5x7 bitmap font for text rendering
static const uint8_t font_5x7[][5] = {
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, // A
    {0x7F, 0x49, 0x49, 0x49, 0x36}, // B
    {0x3E, 0x41, 0x41, 0x41, 0x22}, // C
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, // D
    {0x7F, 0x49, 0x49, 0x49, 0x41}, // E
    {0x7F, 0x09, 0x09, 0x09, 0x01}, // F
    {0x3E, 0x41, 0x49, 0x49, 0x7A}, // G
    {0x7F, 0x08, 0x08, 0x08, 0x7F}, // H
    {0x00, 0x41, 0x7F, 0x41, 0x00}, // I
    {0x20, 0x40, 0x41, 0x3F, 0x01}, // J
    {0x7F, 0x08, 0x14, 0x22, 0x41}, // K
    {0x7F, 0x40, 0x40, 0x40, 0x40}, // L
    {0x7F, 0x02, 0x0C, 0x02, 0x7F}, // M
    {0x7F, 0x04, 0x08, 0x10, 0x7F}, // N
    {0x3E, 0x41, 0x41, 0x41, 0x3E}, // O
    {0x7F, 0x09, 0x09, 0x09, 0x06}, // P
    {0x3E, 0x41, 0x51, 0x21, 0x5E}, // Q
    {0x7F, 0x09, 0x19, 0x29, 0x46}, // R
    {0x46, 0x49, 0x49, 0x49, 0x31}, // S
    {0x01, 0x01, 0x7F, 0x01, 0x01}, // T
    {0x3F, 0x40, 0x40, 0x40, 0x3F}, // U
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, // V
    {0x3F, 0x40, 0x38, 0x40, 0x3F}, // W
    {0x63, 0x14, 0x08, 0x14, 0x63}, // X
    {0x07, 0x08, 0x70, 0x08, 0x07}, // Y
    {0x61, 0x51, 0x49, 0x45, 0x43}, // Z
    {0x00, 0x00, 0x00, 0x00, 0x00}, // Space (26)
};

// Helper: Draw a character (5x7 bitmap font, scaled 3x)
static void draw_char(int x, int y, char c, uint16_t color, int scale) {
    if (c < 'A' || c > 'Z') {
        if (c == ' ') c = 'Z' + 1;  // Space character
        else return;  // Unknown character
    }
    
    int idx = (c >= 'A' && c <= 'Z') ? (c - 'A') : 26;  // 26 = space
    
    for (int col = 0; col < 5; col++) {
        uint8_t column_data = font_5x7[idx][col];
        for (int row = 0; row < 7; row++) {
            if (column_data & (1 << row)) {
                draw_rect(x + col * scale, y + row * scale, scale, scale, color);
            }
        }
    }
}

// Helper: Draw text string (5x7 bitmap font, scaled)
static void draw_text(int x, int y, const char *text, uint16_t color, int scale) {
    int cursor_x = x;
    for (int i = 0; text[i] != '\0'; i++) {
        char c = text[i];
        if (c >= 'a' && c <= 'z') c = c - 'a' + 'A';  // Convert to uppercase
        draw_char(cursor_x, y, c, color, scale);
        cursor_x += 6 * scale;  // 5 pixels + 1 pixel spacing
    }
}

esp_err_t display_init(void) {
    ESP_LOGI(TAG, "=== Display Initialization ===");
    
    // GPIO 15 - Power
    gpio_config_t pwr_gpio_config = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = 1ULL << 15
    };
    ESP_ERROR_CHECK(gpio_config(&pwr_gpio_config));
    gpio_set_level(15, 1);  // Power ON
    
    // GPIO 9 - RD pin as input with pullup
    gpio_config_t input_conf = {
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pin_bit_mask = 1ULL << 9
    };
    ESP_ERROR_CHECK(gpio_config(&input_conf));
    
    // GPIO 38 - Backlight
    gpio_config_t bk_gpio_config = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = 1ULL << LCD_PIN_NUM_BK_LIGHT
    };
    ESP_ERROR_CHECK(gpio_config(&bk_gpio_config));
    gpio_set_level(LCD_PIN_NUM_BK_LIGHT, 1);  // Backlight ON
    
    ESP_LOGI(TAG, "GPIOs configured");
    
    // Allocate framebuffer (FULL SCREEN: 170x320)
    frame_buffer = heap_caps_malloc(LCD_WIDTH * LCD_HEIGHT * sizeof(uint16_t), MALLOC_CAP_DMA);
    if (!frame_buffer) {
        ESP_LOGE(TAG, "Failed to allocate framebuffer!");
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "Framebuffer allocated: %dx%d = %d bytes", LCD_WIDTH, LCD_HEIGHT, LCD_WIDTH * LCD_HEIGHT * 2);
    
    // Initialize Intel 8080 bus
    ESP_LOGI(TAG, "Initialize Intel 8080 bus");
    esp_lcd_i80_bus_handle_t i80_bus = NULL;
    esp_lcd_i80_bus_config_t bus_config = {
        .clk_src = LCD_CLK_SRC_DEFAULT,
        .dc_gpio_num = LCD_PIN_NUM_DC,
        .wr_gpio_num = LCD_PIN_NUM_PCLK,
        .data_gpio_nums = {
            LCD_PIN_NUM_DATA0, LCD_PIN_NUM_DATA1, LCD_PIN_NUM_DATA2, LCD_PIN_NUM_DATA3,
            LCD_PIN_NUM_DATA4, LCD_PIN_NUM_DATA5, LCD_PIN_NUM_DATA6, LCD_PIN_NUM_DATA7,
        },
        .bus_width = 8,
        .max_transfer_bytes = LCD_WIDTH * LCD_HEIGHT * sizeof(uint16_t)
    };
    ESP_ERROR_CHECK(esp_lcd_new_i80_bus(&bus_config, &i80_bus));
    
    // Panel IO config
    esp_lcd_panel_io_i80_config_t io_config = {
        .cs_gpio_num = LCD_PIN_NUM_CS,
        .pclk_hz = 10 * 1000 * 1000,  // 10MHz
        .trans_queue_depth = 20,
        .dc_levels = {
            .dc_idle_level = 0,
            .dc_cmd_level = 0,
            .dc_dummy_level = 0,
            .dc_data_level = 1,
        },
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_i80(i80_bus, &io_config, &io_handle));
    ESP_LOGI(TAG, "Panel IO created");
    
    // Panel config
    ESP_LOGI(TAG, "Install LCD driver of st7789");
    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = LCD_PIN_NUM_RST,
        .rgb_endian = ESP_LCD_COLOR_SPACE_RGB,
        .bits_per_pixel = 16,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(io_handle, &panel_config, &panel_handle));
    
    // Reset and Init
    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel_handle));
    ESP_LOGI(TAG, "Panel initialized");
    
    // Configuration - match the working example so coordinates/map are correct
    // In particular swap_xy must be enabled so LCD_WIDTH/LCD_HEIGHT map to the
    // physical orientation used by the panel (170x320 portrait after swap).
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel_handle, false, true));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel_handle, 0, 35));
    
    // Gamma correction (CRITICAL!)
    ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(io_handle, 0xF2, (uint8_t[]){0}, 1));
    ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(io_handle, 0x26, (uint8_t[]){1}, 1));
    ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(io_handle, 0xE0, (uint8_t[]){
        0x0F, 0x31, 0x2B, 0x0C, 0x0E, 0x08, 0x4E, 0xF1,
        0x37, 0x07, 0x10, 0x03, 0x0E, 0x09, 0x00
    }, 15));
    ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(io_handle, 0xE1, (uint8_t[]){
        0x00, 0x0E, 0x14, 0x03, 0x11, 0x07, 0x31, 0xC1,
        0x48, 0x08, 0x0F, 0x0C, 0x31, 0x36, 0x0F
    }, 15));
    
    // Display ON
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel_handle, true));
    ESP_LOGI(TAG, "Display ON - Ready to use!");
    
    // Note: Battery monitoring is initialized separately in main after ADC1 is set up for GSR
    // Call battery_monitor_init_shared() from main to share the ADC handle
    
    return ESP_OK;
}

esp_err_t display_show_clock(void) {
    if (!panel_handle || !frame_buffer) {
        ESP_LOGE(TAG, "Display not initialized!");
        return ESP_FAIL;
    }
    
    // Get current time from time_sync if available
    struct tm timeinfo;
    int ret = time_sync_get_local_time(&timeinfo);
    
    if (ret != 0) {
        // Time not synced yet - show "Connect Host" message instead
        return display_show_connect_host();
    }
    
    // Clear screen with black background (professional theme)
    clear_screen(COLOR_BLACK);
    
    // Draw large clock (HH:MM format) - centered on screen
    // Screen is 320 wide × 170 tall (landscape)
    // Each digit is 35 wide, space 5, colon is 8 wide
    // Total width: 35+5+35+8+35+5+35 = 158 pixels, centered = (320-158)/2 = 81
    // Digit height is 55 pixels, center vertically: (170 - 55) / 2 = 57
    int x_start = 81;
    int y_start = 57;  // Vertically centered (with space for full digit height)
    
    int hour = timeinfo.tm_hour;
    int min = timeinfo.tm_min;
    
    // Draw hours (HH) - WHITE on BLACK for professional look
    draw_large_digit(x_start, y_start, hour / 10, COLOR_WHITE);
    draw_large_digit(x_start + 40, y_start, hour % 10, COLOR_WHITE);
    
    // Draw colon
    draw_rect(x_start + 80, y_start + 18, 6, 6, COLOR_WHITE);
    draw_rect(x_start + 80, y_start + 35, 6, 6, COLOR_WHITE);
    
    // Draw minutes (MM)
    draw_large_digit(x_start + 90, y_start, min / 10, COLOR_WHITE);
    draw_large_digit(x_start + 130, y_start, min % 10, COLOR_WHITE);
    
    // Draw battery indicator in top-right corner
    uint16_t battery_mv = 0;
    int battery_percent = battery_get_percentage(&battery_mv);
    if (battery_percent >= 0) {
        int bat_x = LCD_WIDTH - 60;
        int bat_y = 10;
        
        // Draw battery icon
        draw_battery_icon(bat_x, bat_y, battery_percent, COLOR_WHITE);
        
        // Draw percentage text next to icon (small scale)
        char bat_text[16];  // Increased buffer size to avoid truncation warning
        snprintf(bat_text, sizeof(bat_text), "%d", battery_percent);
        draw_text(bat_x + 25, bat_y + 2, bat_text, COLOR_WHITE, 1);
    }
    
    // Update the display
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    
    ESP_LOGI(TAG, "Clock updated: %02d:%02d (Battery: %d%%)", hour, min, battery_percent);
    current_mode = DISPLAY_MODE_CLOCK;
    
    return ESP_OK;
}

esp_err_t display_show_qr_code(const display_device_info_t *info) {
    if (!panel_handle || !frame_buffer) {
        ESP_LOGE(TAG, "Display not initialized!");
        return ESP_FAIL;
    }
    
    if (!info || !info->device_name) {
        ESP_LOGE(TAG, "Invalid device info!");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Save device info for toggle
    saved_device_info = *info;
    
    // Create QR code data string: Just device name (e.g., "Shadow-9026")
    char qr_data[64];
    snprintf(qr_data, sizeof(qr_data), "%s", info->device_name);
    
    // Clear screen with white background
    clear_screen(COLOR_WHITE);
    
    // Generate QR code
    QRCode qrcode;
    uint8_t qrcodeData[qrcode_getBufferSize(6)];  // Version 6
    qrcode_initText(&qrcode, qrcodeData, 6, ECC_LOW, qr_data);
    
    // Calculate QR code size and position (centered)
    int qr_size = qrcode.size;
    int pixel_size = 3;  // Each QR module is 3x3 pixels
    int qr_display_size = qr_size * pixel_size;
    int qr_x = (LCD_WIDTH - qr_display_size) / 2;
    int qr_y = 40;  // Top portion of screen
    
    // Draw QR code
    for (int y = 0; y < qr_size; y++) {
        for (int x = 0; x < qr_size; x++) {
            uint16_t color = qrcode_getModule(&qrcode, x, y) ? COLOR_BLACK : COLOR_WHITE;
            // Draw pixel_size x pixel_size block for each module
            draw_rect(qr_x + x * pixel_size, qr_y + y * pixel_size, pixel_size, pixel_size, color);
        }
    }
    
    // Draw device name and password below QR code (you can add text rendering here)
    // For now, just a separator line
    draw_rect(10, qr_y + qr_display_size + 20, LCD_WIDTH - 20, 2, COLOR_BLACK);
    
    // Update display
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    
    ESP_LOGI(TAG, "QR Code displayed: %s", info->device_name);
    current_mode = DISPLAY_MODE_QR;
    
    return ESP_OK;
}

esp_err_t display_toggle_mode(const display_device_info_t *info) {
    if (!display_power_on) {
        return ESP_OK;  // Don't toggle if display is off
    }
    
    if (current_mode == DISPLAY_MODE_CLOCK) {
        return display_show_qr_code(info ? info : &saved_device_info);
    } else {
        return display_show_clock();
    }
}

esp_err_t display_set_power(bool on) {
    if (!panel_handle) {
        return ESP_FAIL;
    }
    
    display_power_on = on;
    
    if (on) {
        // Turn on backlight
        gpio_set_level(LCD_PIN_NUM_BK_LIGHT, 1);
        ESP_LOGI(TAG, "Display powered ON");
        
        // Refresh current mode
        if (current_mode == DISPLAY_MODE_CLOCK) {
            display_show_clock();
        } else if (current_mode == DISPLAY_MODE_QR) {
            display_show_qr_code(&saved_device_info);
        }
    } else {
        // Turn off backlight
        gpio_set_level(LCD_PIN_NUM_BK_LIGHT, 0);
        ESP_LOGI(TAG, "Display powered OFF");
    }
    
    return ESP_OK;
}

bool display_is_on(void) {
    return display_power_on;
}

esp_err_t display_refresh(void) {
    if (!display_power_on || !panel_handle) {
        return ESP_OK;  // Don't refresh if display is off
    }
    
    // Check calibration state first (highest priority)
    calibration_state_t cal_state = calibration_get_state();
    
    if (cal_state == CAL_STATE_IN_PROGRESS) {
        // Show calibration progress
        float progress = calibration_get_progress();
        return display_show_calibration_progress(progress);
    } else if (cal_state == CAL_STATE_COMPLETED || cal_state == CAL_STATE_LOADED) {
        // Calibration just completed - show "Good to Go" briefly
        // This will be overridden by clock on next refresh cycle
        if (current_mode == DISPLAY_MODE_CALIBRATING) {
            display_show_good_to_go();
            vTaskDelay(pdMS_TO_TICKS(3000));  // Show for 3 seconds
            // Fall through to show clock
        }
    }
    
    // Refresh based on current mode
    if (current_mode == DISPLAY_MODE_CLOCK || current_mode == DISPLAY_MODE_CONNECT_HOST || 
        current_mode == DISPLAY_MODE_GOOD_TO_GO) {
        // Check if time is synced
        if (time_sync_is_synced()) {
            return display_show_clock();
        } else {
            return display_show_connect_host();
        }
    } else if (current_mode == DISPLAY_MODE_CALIBRATING) {
        // Update calibration progress
        float progress = calibration_get_progress();
        return display_show_calibration_progress(progress);
    }
    // QR code doesn't need refresh
    return ESP_OK;
}

display_mode_t display_get_mode(void) {
    return current_mode;
}

esp_err_t display_show_status(const char *status_text) {
    if (!panel_handle || !frame_buffer) {
        return ESP_FAIL;
    }
    
    // Simple status display
    clear_screen(COLOR_BLACK);
    // (Would add text rendering here)
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    
    return ESP_OK;
}

esp_err_t display_show_connect_host(void) {
    if (!panel_handle || !frame_buffer) {
        ESP_LOGE(TAG, "Display not initialized!");
        return ESP_FAIL;
    }
    
    // Clear screen with black background
    clear_screen(COLOR_BLACK);
    
    // Draw "CONNECT HOST" text centered on screen
    // Screen is 320 wide × 170 tall
    // Text "CONNECT HOST" = 12 characters × 6 pixels/char × scale 3 = 216 pixels wide
    // Center: (320 - 216) / 2 = 52
    int text_x = 52;
    int text_y = 60;  // Vertically centered
    
    draw_text(text_x, text_y, "CONNECT HOST", COLOR_WHITE, 3);
    
    // Add a small animated indicator (blinking dots)
    static int blink_state = 0;
    blink_state = (blink_state + 1) % 4;
    int dot_y = text_y + 30;
    for (int i = 0; i < blink_state; i++) {
        draw_rect(text_x + 80 + i * 15, dot_y, 8, 8, COLOR_WHITE);
    }
    
    // Update the display
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    
    ESP_LOGI(TAG, "Connect Host message displayed");
    current_mode = DISPLAY_MODE_CONNECT_HOST;
    
    return ESP_OK;
}

esp_err_t display_show_calibration_progress(float progress) {
    if (!panel_handle || !frame_buffer) {
        ESP_LOGE(TAG, "Display not initialized!");
        return ESP_FAIL;
    }
    
    // Clamp progress to [0.0, 1.0]
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    
    // Clear screen with black background
    clear_screen(COLOR_BLACK);
    
    // Draw "CALIBRATING" text at top
    // Screen is 320 wide × 170 tall
    // Text "CALIBRATING" = 11 characters × 6 pixels/char × scale 2 = 132 pixels wide
    // Center: (320 - 132) / 2 = 94
    int text_x = 94;
    int text_y = 30;
    
    draw_text(text_x, text_y, "CALIBRATING", COLOR_WHITE, 2);
    
    // Draw progress bar with better visibility
    int bar_x = 40;
    int bar_y = 80;
    int bar_w = 240;  // Total bar width
    int bar_h = 30;   // Bar height
    
    // Draw outer border (frame)
    for (int i = 0; i < 3; i++) {
        draw_rect(bar_x - 3 + i, bar_y - 3, bar_w + 6 - i*2, 1, COLOR_WHITE);  // Top
        draw_rect(bar_x - 3 + i, bar_y + bar_h + 2, bar_w + 6 - i*2, 1, COLOR_WHITE);  // Bottom
        draw_rect(bar_x - 3, bar_y - 2 + i, 1, bar_h + 4 - i*2, COLOR_WHITE);  // Left
        draw_rect(bar_x + bar_w + 2, bar_y - 2 + i, 1, bar_h + 4 - i*2, COLOR_WHITE);  // Right
    }
    
    // Fill background of bar with dark gray (empty portion)
    draw_rect(bar_x, bar_y, bar_w, bar_h, COLOR_DARKGRAY);
    
    // Draw filled portion (progressive fill)
    int filled_w = (int)(bar_w * progress);
    if (filled_w > 0) {
        // Use a gradient-like effect by drawing slightly lighter inner portion
        draw_rect(bar_x, bar_y, filled_w, bar_h, COLOR_WHITE);
        // Add inner highlight for 3D effect
        if (filled_w > 4) {
            draw_rect(bar_x + 2, bar_y + 2, filled_w - 4, 2, COLOR_LIGHTGRAY);
        }
    }
    
    // Draw percentage text below bar
    int percent = (int)(progress * 100);
    char percent_str[16];
    snprintf(percent_str, sizeof(percent_str), "%d%%", percent);
    
    // Center percentage text (2 or 3 chars × 6 pixels/char × scale 2)
    int percent_len = strlen(percent_str);
    int percent_x = 160 - (percent_len * 6 * 2) / 2;  // Center horizontally
    int percent_y = bar_y + bar_h + 15;
    
    draw_text(percent_x, percent_y, percent_str, COLOR_WHITE, 2);
    
    // Draw battery indicator in top-right corner
    uint16_t battery_mv = 0;
    int battery_percent = battery_get_percentage(&battery_mv);
    if (battery_percent >= 0) {
        int bat_x = LCD_WIDTH - 60;
        int bat_y = 10;
        draw_battery_icon(bat_x, bat_y, battery_percent, COLOR_WHITE);
        char bat_text[16];  // Increased buffer size to avoid truncation warning
        snprintf(bat_text, sizeof(bat_text), "%d", battery_percent);
        draw_text(bat_x + 25, bat_y + 2, bat_text, COLOR_WHITE, 1);
    }
    
    // Update the display
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    
    ESP_LOGI(TAG, "Calibration progress: %.1f%% (Battery: %d%%)", progress * 100.0f, battery_percent);
    current_mode = DISPLAY_MODE_CALIBRATING;
    
    return ESP_OK;
}

esp_err_t display_show_good_to_go(void) {
    if (!panel_handle || !frame_buffer) {
        ESP_LOGE(TAG, "Display not initialized!");
        return ESP_FAIL;
    }
    
    // Clear screen with black background
    clear_screen(COLOR_BLACK);
    
    // Draw "GOOD TO GO" text centered on screen
    // Screen is 320 wide × 170 tall
    // Text "GOOD TO GO" = 10 characters × 6 pixels/char × scale 3 = 180 pixels wide
    // Center: (320 - 180) / 2 = 70
    int text_x = 70;
    int text_y = 65;  // Vertically centered
    
    draw_text(text_x, text_y, "GOOD TO GO", COLOR_WHITE, 3);
    
    // Add checkmark symbol (simple V shape)
    int check_x = 140;
    int check_y = 110;
    for (int i = 0; i < 15; i++) {
        draw_rect(check_x + i, check_y + i, 4, 4, COLOR_WHITE);
        draw_rect(check_x + 30 - i, check_y + i, 4, 4, COLOR_WHITE);
    }
    
    // Update the display
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    
    ESP_LOGI(TAG, "Good to Go message displayed");
    current_mode = DISPLAY_MODE_GOOD_TO_GO;
    
    return ESP_OK;
}

// ==================== BATTERY MONITORING ====================

esp_err_t battery_monitor_init(void) {
    ESP_LOGI(TAG, "Initializing battery monitoring on GPIO %d...", BATTERY_ADC_PIN);
    
    // ADC unit initialization
    adc_oneshot_unit_init_cfg_t init_config = {
        .unit_id = ADC_UNIT_1,
        .ulp_mode = ADC_ULP_MODE_DISABLE,
    };
    
    esp_err_t err = adc_oneshot_new_unit(&init_config, &battery_adc_handle);
    if (err == ESP_ERR_INVALID_STATE || err == ESP_ERR_NOT_FOUND) {
        // ADC1 is already initialized (likely by GSR sensor)
        // We can't create a new unit, but we can still configure the channel
        ESP_LOGW(TAG, "ADC1 already in use - battery monitoring will be disabled");
        ESP_LOGW(TAG, "To enable battery monitoring, use battery_monitor_init_shared() with existing ADC handle");
        return ESP_OK;  // Return OK but don't set battery_initialized
    } else if (err != ESP_OK) {
        ESP_LOGE(TAG, "Battery ADC unit init failed: %s", esp_err_to_name(err));
        return err;
    }
    
    // Channel configuration
    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ADC_BITWIDTH_DEFAULT,
        .atten = ADC_ATTEN_DB_12,  // Full range (0-3.3V input)
    };
    
    err = adc_oneshot_config_channel(battery_adc_handle, BATTERY_ADC_CHANNEL, &config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Battery ADC channel config failed: %s", esp_err_to_name(err));
        return err;
    }
    
    // ADC Calibration
    adc_cali_curve_fitting_config_t cali_config = {
        .unit_id = ADC_UNIT_1,
        .atten = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    
    err = adc_cali_create_scheme_curve_fitting(&cali_config, &battery_cali_handle);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Battery ADC calibration successful");
        battery_initialized = true;
    } else {
        ESP_LOGW(TAG, "Battery ADC calibration failed: %s, using raw values", esp_err_to_name(err));
        battery_initialized = false;
    }
    
    ESP_LOGI(TAG, "Battery monitoring initialized");
    return ESP_OK;
}

esp_err_t battery_monitor_init_shared(void *adc_handle) {
    if (adc_handle == NULL) {
        ESP_LOGE(TAG, "Invalid ADC handle");
        return ESP_ERR_INVALID_ARG;
    }
    
    ESP_LOGI(TAG, "Initializing battery monitoring with shared ADC handle...");
    
    // Use the provided ADC handle
    battery_adc_handle = (adc_oneshot_unit_handle_t)adc_handle;
    
    // Channel configuration
    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ADC_BITWIDTH_DEFAULT,
        .atten = ADC_ATTEN_DB_12,  // Full range (0-3.3V input)
    };
    
    esp_err_t err = adc_oneshot_config_channel(battery_adc_handle, BATTERY_ADC_CHANNEL, &config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Battery ADC channel config failed: %s", esp_err_to_name(err));
        return err;
    }
    
    // ADC Calibration
    adc_cali_curve_fitting_config_t cali_config = {
        .unit_id = ADC_UNIT_1,
        .atten = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    
    err = adc_cali_create_scheme_curve_fitting(&cali_config, &battery_cali_handle);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Battery ADC calibration successful");
        battery_initialized = true;
    } else {
        ESP_LOGW(TAG, "Battery ADC calibration failed: %s, using raw values", esp_err_to_name(err));
        battery_initialized = true;  // Still mark as initialized, just without calibration
    }
    
    ESP_LOGI(TAG, "Battery monitoring initialized with shared ADC");
    return ESP_OK;
}

int battery_get_percentage(uint16_t *voltage_mv) {
    if (!battery_initialized || battery_adc_handle == NULL) {
        if (voltage_mv) *voltage_mv = 0;
        return -1;  // Not initialized
    }
    
    // Take multiple readings and average
    int total = 0;
    int valid_readings = 0;
    
    for (int i = 0; i < BATTERY_READ_SAMPLES; i++) {
        int raw_value;
        esp_err_t err = adc_oneshot_read(battery_adc_handle, BATTERY_ADC_CHANNEL, &raw_value);
        if (err == ESP_OK) {
            total += raw_value;
            valid_readings++;
        }
        vTaskDelay(pdMS_TO_TICKS(2));
    }
    
    if (valid_readings == 0) {
        ESP_LOGW(TAG, "No valid battery readings");
        if (voltage_mv) *voltage_mv = 0;
        return -1;
    }
    
    int avg_raw = total / valid_readings;
    
    // Convert to voltage
    int adc_voltage_mv = 0;
    if (battery_cali_handle != NULL) {
        esp_err_t err = adc_cali_raw_to_voltage(battery_cali_handle, avg_raw, &adc_voltage_mv);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "ADC calibration conversion failed");
            adc_voltage_mv = (avg_raw * 3300) / 4095;  // Fallback
        }
    } else {
        // Manual calculation
        adc_voltage_mv = (avg_raw * 3300) / 4095;
    }
    
    // Account for voltage divider (battery voltage is 2x ADC voltage)
    uint16_t battery_voltage = (uint16_t)(adc_voltage_mv * BATTERY_VOLTAGE_DIVIDER);
    
    if (voltage_mv) {
        *voltage_mv = battery_voltage;
    }
    
    // Calculate percentage (linear approximation between min and max)
    int percentage;
    if (battery_voltage >= BATTERY_MAX_VOLTAGE) {
        percentage = 100;
    } else if (battery_voltage <= BATTERY_MIN_VOLTAGE) {
        percentage = 0;
    } else {
        percentage = ((battery_voltage - BATTERY_MIN_VOLTAGE) * 100) / 
                     (BATTERY_MAX_VOLTAGE - BATTERY_MIN_VOLTAGE);
    }
    
    // Clamp to 0-100
    if (percentage < 0) percentage = 0;
    if (percentage > 100) percentage = 100;
    
    return percentage;
}

// Helper: Draw battery icon with percentage
static void draw_battery_icon(int x, int y, int percentage, uint16_t color) {
    // Battery outline (20x10 pixels)
    int width = 20;
    int height = 10;
    
    // Draw battery body
    draw_rect(x, y, width, height, color);
    draw_rect(x + 1, y + 1, width - 2, height - 2, COLOR_BLACK);  // Inner hollow
    
    // Draw battery terminal (nipple)
    draw_rect(x + width, y + 3, 2, 4, color);
    
    // Draw fill level
    int fill_width = ((width - 4) * percentage) / 100;
    if (fill_width > 0) {
        draw_rect(x + 2, y + 2, fill_width, height - 4, color);
    }
}
