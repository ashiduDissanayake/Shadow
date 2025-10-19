/**
 * SIMPLE Display Test - NO QR, NO CLOCK, NO BUTTONS
 * Just fill screen with colors to verify esp_lcd works
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
#include <string.h>

static const char *TAG = "DISPLAY";

static esp_lcd_panel_handle_t panel_handle = NULL;
static uint16_t *frame_buffer = NULL;

// Simple init commands for ST7789
static const uint8_t st7789_init[][2] = {
    {0x11, 0},    // Sleep out
    {0x3A, 0x05}, // 16-bit color
    {0x21, 0},    // Display inversion ON
    {0x29, 0},    // Display ON
};

esp_err_t display_init(void) {
    ESP_LOGI(TAG, "=== SIMPLE DISPLAY TEST ===");
    ESP_LOGI(TAG, "NO QR, NO CLOCK - JUST COLORS");
    
    // Enable GPIO 15 power
    gpio_config_t pwr_cfg = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = 1ULL << 15,
    };
    gpio_config(&pwr_cfg);
    gpio_set_level(15, 1);
    vTaskDelay(pdMS_TO_TICKS(100));
    ESP_LOGI(TAG, "Power enabled (GPIO 15)");
    
    // Allocate framebuffer
    frame_buffer = heap_caps_malloc(LCD_WIDTH * LCD_HEIGHT * 2, MALLOC_CAP_DMA);
    if (!frame_buffer) {
        ESP_LOGE(TAG, "Failed to allocate framebuffer!");
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "Framebuffer allocated: %d bytes", LCD_WIDTH * LCD_HEIGHT * 2);
    
    // Backlight
    gpio_config_t bl_cfg = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = 1ULL << LCD_PIN_NUM_BK_LIGHT,
    };
    gpio_config(&bl_cfg);
    gpio_set_level(LCD_PIN_NUM_BK_LIGHT, 1);
    ESP_LOGI(TAG, "Backlight ON (GPIO %d)", LCD_PIN_NUM_BK_LIGHT);
    
    // Configure i80 bus
    esp_lcd_i80_bus_handle_t i80_bus = NULL;
    esp_lcd_i80_bus_config_t bus_cfg = {
        .dc_gpio_num = LCD_PIN_NUM_DC,
        .wr_gpio_num = LCD_PIN_NUM_PCLK,
        .clk_src = LCD_CLK_SRC_DEFAULT,
        .data_gpio_nums = {
            LCD_PIN_NUM_DATA0, LCD_PIN_NUM_DATA1, LCD_PIN_NUM_DATA2, LCD_PIN_NUM_DATA3,
            LCD_PIN_NUM_DATA4, LCD_PIN_NUM_DATA5, LCD_PIN_NUM_DATA6, LCD_PIN_NUM_DATA7,
        },
        .bus_width = 8,
        .max_transfer_bytes = LCD_WIDTH * LCD_HEIGHT * 2,
    };
    ESP_ERROR_CHECK(esp_lcd_new_i80_bus(&bus_cfg, &i80_bus));
    ESP_LOGI(TAG, "i80 bus created");
    
    // Panel IO
    esp_lcd_panel_io_handle_t io_handle = NULL;
    esp_lcd_panel_io_i80_config_t io_cfg = {
        .cs_gpio_num = LCD_PIN_NUM_CS,
        .pclk_hz = 10 * 1000 * 1000,  // Start with 10MHz (safe)
        .trans_queue_depth = 10,
        .dc_levels = {
            .dc_idle_level = 0,
            .dc_cmd_level = 0,
            .dc_dummy_level = 0,
            .dc_data_level = 1,
        },
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_i80(i80_bus, &io_cfg, &io_handle));
    ESP_LOGI(TAG, "Panel IO created");
    
    // Panel
    esp_lcd_panel_dev_config_t panel_cfg = {
        .reset_gpio_num = LCD_PIN_NUM_RST,
        .color_space = ESP_LCD_COLOR_SPACE_RGB,
        .bits_per_pixel = 16,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(io_handle, &panel_cfg, &panel_handle));
    ESP_LOGI(TAG, "Panel created");
    
    // Reset & Init
    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel_handle));
    ESP_LOGI(TAG, "Panel initialized");
    
    // Basic config
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel_handle, false, true));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel_handle, 0, 35));
    ESP_LOGI(TAG, "Panel configured");
    
    // Turn on display
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel_handle, true));
    ESP_LOGI(TAG, "Display ON");
    
    // TEST: Fill with RED
    ESP_LOGI(TAG, "TEST 1: Filling screen RED");
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = 0xF800;  // RED in RGB565
    }
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    // TEST: Fill with GREEN
    ESP_LOGI(TAG, "TEST 2: Filling screen GREEN");
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = 0x07E0;  // GREEN in RGB565
    }
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    // TEST: Fill with BLUE
    ESP_LOGI(TAG, "TEST 3: Filling screen BLUE");
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = 0x001F;  // BLUE in RGB565
    }
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    // TEST: Fill with WHITE
    ESP_LOGI(TAG, "TEST 4: Filling screen WHITE");
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = 0xFFFF;  // WHITE in RGB565
    }
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer));
    
    ESP_LOGI(TAG, "=== COLOR TEST COMPLETE ===");
    ESP_LOGI(TAG, "Did you see: RED -> GREEN -> BLUE -> WHITE?");
    
    return ESP_OK;
}

// Dummy functions to satisfy linker
void display_toggle_mode(void) {}
void display_show_qr(const char *device_name, const char *password) {}
void display_show_clock(int hour, int min, int sec) {}
void display_show_status(const char *message) {}
