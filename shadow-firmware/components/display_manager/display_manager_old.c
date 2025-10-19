/**
 * Display Manager - Clock and QR Code Display
 * Full screen rendering with button toggle
 */

#include "display_manager.h"
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
static uint16_t *frame_buffer = NULL;
static display_mode_t current_mode = DISPLAY_MODE_CLOCK;
static display_device_info_t device_info = {0};

// Helper: Clear screen with color
static void clear_screen(uint16_t color) {
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = color;
    }
    esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer);
}

// Helper: Draw a filled rectangle
static void draw_rect(int x, int y, int w, int h, uint16_t color) {
    for (int py = y; py < y + h && py < LCD_HEIGHT; py++) {
        for (int px = x; px < x + w && px < LCD_WIDTH; px++) {
            frame_buffer[py * LCD_WIDTH + px] = color;
        }
    }
}

// Helper: Draw large digit (simple 7-segment style, 60x100 pixels)
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
    
    int w = 40, h = 60;
    int seg_h = h / 3;
    int seg_w = w;
    int thick = 6;
    
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
    ESP_LOGI(TAG, "=== EXACT COPY OF WORKING EXAMPLE ===");
    ESP_LOGI(TAG, "Initializing display...");
    
    // GPIO 15 - Power (EXACTLY like working example)
    gpio_config_t pwr_gpio_config = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = 1ULL << 15
    };
    ESP_ERROR_CHECK(gpio_config(&pwr_gpio_config));
    gpio_set_level(15, 1);  // Power ON
    
    // GPIO 9 - RD pin as input with pullup (from working example)
    gpio_config_t input_conf = {
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pin_bit_mask = 1ULL << 9
    };
    ESP_ERROR_CHECK(gpio_config(&input_conf));
    
    // GPIO 38 - Backlight (EXACTLY like working example)
    gpio_config_t bk_gpio_config = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = 1ULL << LCD_PIN_NUM_BK_LIGHT
    };
    ESP_ERROR_CHECK(gpio_config(&bk_gpio_config));
    gpio_set_level(LCD_PIN_NUM_BK_LIGHT, 1);  // Backlight ON
    
    ESP_LOGI(TAG, "GPIOs configured");
    
    // Allocate framebuffer (DMA capable, EXACTLY as working example uses)
    frame_buffer = heap_caps_malloc(LCD_WIDTH * LCD_HEIGHT * sizeof(uint16_t), MALLOC_CAP_DMA);
    if (!frame_buffer) {
        ESP_LOGE(TAG, "Failed to allocate framebuffer!");
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "Framebuffer allocated: %d bytes", LCD_WIDTH * LCD_HEIGHT * 2);
    
    // Initialize Intel 8080 bus (EXACTLY like working example)
    ESP_LOGI(TAG, "Initialize Intel 8080 bus");
    esp_lcd_i80_bus_handle_t i80_bus = NULL;
    esp_lcd_i80_bus_config_t bus_config = {
        .clk_src = LCD_CLK_SRC_DEFAULT,
        .dc_gpio_num = LCD_PIN_NUM_DC,
        .wr_gpio_num = LCD_PIN_NUM_PCLK,
        .data_gpio_nums = {
            LCD_PIN_NUM_DATA0,
            LCD_PIN_NUM_DATA1,
            LCD_PIN_NUM_DATA2,
            LCD_PIN_NUM_DATA3,
            LCD_PIN_NUM_DATA4,
            LCD_PIN_NUM_DATA5,
            LCD_PIN_NUM_DATA6,
            LCD_PIN_NUM_DATA7,
        },
        .bus_width = 8,
        .max_transfer_bytes = LCD_WIDTH * LCD_HEIGHT * sizeof(uint16_t)
    };
    ESP_ERROR_CHECK(esp_lcd_new_i80_bus(&bus_config, &i80_bus));
    
    // Panel IO config (EXACTLY like working example - NOTE: no callback since we don't use LVGL)
    esp_lcd_panel_io_handle_t io_handle = NULL;
    esp_lcd_panel_io_i80_config_t io_config = {
        .cs_gpio_num = LCD_PIN_NUM_CS,
        .pclk_hz = 10 * 1000 * 1000,  // 10MHz - EXACTLY as working example
        .trans_queue_depth = 20,       // EXACTLY as working example
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
    
    // Panel config (EXACTLY like working example!)
    ESP_LOGI(TAG, "Install LCD driver of st7789");
    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = LCD_PIN_NUM_RST,
        .rgb_endian = ESP_LCD_COLOR_SPACE_RGB,  // KEY: Must be rgb_endian, not color_space!
        .bits_per_pixel = 16,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(io_handle, &panel_config, &panel_handle));
    
    // Reset and Init (EXACTLY like working example)
    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel_handle));
    ESP_LOGI(TAG, "Panel initialized");
    
    // Configuration (EXACTLY like working example)
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel_handle, false, true));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel_handle, 0, 35));
    
    // Gamma correction (EXACTLY like working example - CRITICAL!)
    ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(io_handle, 0xF2, (uint8_t[]){0}, 1)); // 3Gamma disable
    ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(io_handle, 0x26, (uint8_t[]){1}, 1)); // Gamma curve 1
    ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(io_handle, 0xE0, (uint8_t[]){  // Positive gamma
        0x0F, 0x31, 0x2B, 0x0C, 0x0E, 0x08, 0x4E, 0xF1,
        0x37, 0x07, 0x10, 0x03, 0x0E, 0x09, 0x00
    }, 15));
    ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(io_handle, 0xE1, (uint8_t[]){  // Negative gamma
        0x00, 0x0E, 0x14, 0x03, 0x11, 0x07, 0x31, 0xC1,
        0x48, 0x08, 0x0F, 0x0C, 0x31, 0x36, 0x0F
    }, 15));
    
    // Display ON
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel_handle, true));
    ESP_LOGI(TAG, "Display ON");
    
    // Clear screen to black first
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = 0x0000;
    }
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    
    ESP_LOGI(TAG, "=== DISPLAY INITIALIZED ===");
    ESP_LOGI(TAG, "You can now call display_show_clock() to show time");
    
    return ESP_OK;
}

// Dummy functions to satisfy linker (matching header signatures)
esp_err_t display_toggle_mode(const display_device_info_t *info) {
    return ESP_OK;
}

esp_err_t display_show_qr(const display_device_info_t *info) {
    return ESP_OK;
}

// Simple 5x7 font for digits (each char is 5 pixels wide, 7 pixels tall)
static const uint8_t font_5x7[][5] = {
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, // 0
    {0x00, 0x42, 0x7F, 0x40, 0x00}, // 1
    {0x42, 0x61, 0x51, 0x49, 0x46}, // 2
    {0x21, 0x41, 0x45, 0x4B, 0x31}, // 3
    {0x18, 0x14, 0x12, 0x7F, 0x10}, // 4
    {0x27, 0x45, 0x45, 0x45, 0x39}, // 5
    {0x3C, 0x4A, 0x49, 0x49, 0x30}, // 6
    {0x01, 0x71, 0x09, 0x05, 0x03}, // 7
    {0x36, 0x49, 0x49, 0x49, 0x36}, // 8
    {0x06, 0x49, 0x49, 0x29, 0x1E}, // 9
    {0x00, 0x36, 0x36, 0x00, 0x00}, // : (colon)
};

// Helper: Draw a single character at position
static void draw_char(uint16_t *buffer, int x, int y, char c, uint16_t color, uint16_t bg_color) {
    int char_index;
    if (c >= '0' && c <= '9') {
        char_index = c - '0';
    } else if (c == ':') {
        char_index = 10;
    } else {
        return;
    }
    
    const uint8_t *glyph = font_5x7[char_index];
    
    for (int col = 0; col < 5; col++) {
        uint8_t column_data = glyph[col];
        for (int row = 0; row < 7; row++) {
            int px = x + col;
            int py = y + row;
            if (px >= 0 && px < LCD_WIDTH && py >= 0 && py < LCD_HEIGHT) {
                uint16_t pixel_color = (column_data & (1 << row)) ? color : bg_color;
                buffer[py * LCD_WIDTH + px] = pixel_color;
            }
        }
    }
}

// Helper: Draw text string (digits and colons only)
static void draw_text(uint16_t *buffer, int x, int y, const char *text, uint16_t color, uint16_t bg_color) {
    int cursor_x = x;
    for (int i = 0; text[i] != '\0'; i++) {
        draw_char(buffer, cursor_x, y, text[i], color, bg_color);
        cursor_x += 6; // 5 pixels wide + 1 pixel spacing
    }
}

esp_err_t display_show_clock(void) {
    if (!panel_handle || !frame_buffer) {
        ESP_LOGE(TAG, "Display not initialized!");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Get current time (you'll need to integrate with RTC or NTP later)
    // For now, let's show a demo time
    time_t now;
    struct tm timeinfo;
    time(&now);
    localtime_r(&now, &timeinfo);
    
    // Format time string HH:MM:SS
    char time_str[9];
    snprintf(time_str, sizeof(time_str), "%02d:%02d:%02d", 
             timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);
    
    // Clear screen to black
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = 0x0000;  // Black background
    }
    
    // Calculate centered position for clock
    // Time string is 8 characters: "HH:MM:SS" = 8 chars * 6 pixels = 48 pixels wide
    int text_width = 8 * 6;
    int text_height = 7;
    int x = (LCD_WIDTH - text_width) / 2;
    int y = (LCD_HEIGHT - text_height) / 2;
    
    // Draw time in white
    draw_text(frame_buffer, x, y, time_str, 0xFFFF, 0x0000);
    
    // Update display
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    
    ESP_LOGI(TAG, "Clock displayed: %s", time_str);
    return ESP_OK;
}

esp_err_t display_show_status(const char *status_text) {
    return ESP_OK;
}
