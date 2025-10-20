/**
 * Display Manager - Clock and QR Code Display
 * Full screen rendering with button toggle
 */

#include "display_manager.h"
#include "time_sync.h"
#include "esp_log.h"
#include "driver/gpio.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "qrcode.h"
#include <string.h>
#include <time.h>
#include <sys/time.h>

static const char *TAG = "DISPLAY";

static esp_lcd_panel_handle_t panel_handle = NULL;
static esp_lcd_panel_io_handle_t io_handle = NULL;
static uint16_t *frame_buffer = NULL;
static display_mode_t current_mode = DISPLAY_MODE_CLOCK;
static display_device_info_t saved_device_info = {0};
static bool display_power_on = true;  // Display starts on

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
        // Time not synced yet - show placeholder
        clear_screen(0x001F);  // Dark blue background
        
        // Display "SYNCING..." message
        int x_start = 60;
        int y_start = 70;
        
        // We don't have text rendering, so just show --:--
        draw_large_digit(x_start, y_start, 10, COLOR_GRAY);  // Dash (assuming 10 draws dash)
        draw_large_digit(x_start + 40, y_start, 10, COLOR_GRAY);
        draw_rect(x_start + 80, y_start + 18, 6, 6, COLOR_GRAY);
        draw_rect(x_start + 80, y_start + 35, 6, 6, COLOR_GRAY);
        draw_large_digit(x_start + 90, y_start, 10, COLOR_GRAY);
        draw_large_digit(x_start + 130, y_start, 10, COLOR_GRAY);
        
        ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
        ESP_LOGI(TAG, "Clock display: Waiting for time sync...");
        current_mode = DISPLAY_MODE_CLOCK;
        return ESP_OK;
    }
    
    // Clear screen with dark blue
    clear_screen(0x001F);  // Dark blue background
    
    // Draw large clock (HH:MM format) - centered on screen
    // Screen is 320 wide × 170 tall (landscape)
    // Each digit is 35 wide, space 5, colon is 8 wide
    // Total width: 35+5+35+8+35+5+35 = 158 pixels, centered = (320-158)/2 = 81
    // Digit height is 55 pixels, center vertically: (170 - 55) / 2 = 57
    int x_start = 81;
    int y_start = 57;  // Vertically centered (with space for full digit height)
    
    int hour = timeinfo.tm_hour;
    int min = timeinfo.tm_min;
    
    // Draw hours (HH)
    draw_large_digit(x_start, y_start, hour / 10, COLOR_CYAN);
    draw_large_digit(x_start + 40, y_start, hour % 10, COLOR_CYAN);
    
    // Draw colon
    draw_rect(x_start + 80, y_start + 18, 6, 6, COLOR_WHITE);
    draw_rect(x_start + 80, y_start + 35, 6, 6, COLOR_WHITE);
    
    // Draw minutes (MM)
    draw_large_digit(x_start + 90, y_start, min / 10, COLOR_CYAN);
    draw_large_digit(x_start + 130, y_start, min % 10, COLOR_CYAN);
    
    // Update the display
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    
    ESP_LOGI(TAG, "Clock updated: %02d:%02d", hour, min);
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
    
    // Refresh based on current mode
    if (current_mode == DISPLAY_MODE_CLOCK) {
        return display_show_clock();
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
