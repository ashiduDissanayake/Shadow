/**
 * @file display_manager.c
 * @brief Minimal Display Test for LilyGo T-Display S3
 * 
 * Based on working example: https://github.com/krupis/T-Display-S3-esp-idf
 * 
 * This is a simplified test to verify display hardware is working.
 * It initializes the ST7789 display and shows a GREEN screen.
 * 
 * Hardware: LilyGo T-Display S3 (170x320, ST7789, 8-bit parallel interface)
 */

#include "display_manager.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/gpio.h"
#include "esp_heap_caps.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"

static const char *TAG = "DISPLAY_TEST";

// Power pin for T-Display S3
#define LCD_PIN_NUM_POWER 15

esp_err_t display_init(void) {
    ESP_LOGI(TAG, "=== Display Hardware Test Starting ===");
    
    // Step 1: Enable display power (GPIO 15 - Critical for T-Display S3!)
    ESP_LOGI(TAG, "Step 1: Enabling display power (GPIO 15)...");
    gpio_config_t pwr_gpio_config = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = (1ULL << LCD_PIN_NUM_POWER)
    };
    ESP_ERROR_CHECK(gpio_config(&pwr_gpio_config));
    gpio_set_level(LCD_PIN_NUM_POWER, 1);  // Power ON
    vTaskDelay(pdMS_TO_TICKS(10));  // Wait for power to stabilize
    ESP_LOGI(TAG, "Display power enabled");

    // Step 2: Enable backlight
    ESP_LOGI(TAG, "Step 2: Enabling backlight (GPIO %d)...", LCD_PIN_NUM_BK_LIGHT);
    gpio_config_t bk_gpio_config = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = (1ULL << LCD_PIN_NUM_BK_LIGHT)
    };
    ESP_ERROR_CHECK(gpio_config(&bk_gpio_config));
    gpio_set_level(LCD_PIN_NUM_BK_LIGHT, 1);  // Backlight ON
    ESP_LOGI(TAG, "Backlight enabled");

    // Step 3: Allocate framebuffer (DMA-capable memory)
    ESP_LOGI(TAG, "Step 3: Allocating framebuffer (%d bytes)...", LCD_WIDTH * LCD_HEIGHT * 2);
    size_t buf_size = LCD_WIDTH * LCD_HEIGHT * sizeof(uint16_t);
    uint16_t *framebuffer = heap_caps_malloc(buf_size, MALLOC_CAP_DMA);
    if (!framebuffer) {
        ESP_LOGE(TAG, "Failed to allocate framebuffer!");
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "Framebuffer allocated successfully");

    // Step 4: Initialize 8080 parallel bus
    ESP_LOGI(TAG, "Step 4: Initializing 8080 parallel bus...");
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
        .max_transfer_bytes = LCD_WIDTH * LCD_HEIGHT * sizeof(uint16_t),
    };
    esp_err_t ret = esp_lcd_new_i80_bus(&bus_config, &i80_bus);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create i80 bus: %s", esp_err_to_name(ret));
        heap_caps_free(framebuffer);
        return ret;
    }
    ESP_LOGI(TAG, "8080 bus initialized");

    // Step 5: Configure panel IO
    ESP_LOGI(TAG, "Step 5: Configuring panel IO...");
    esp_lcd_panel_io_handle_t io_handle = NULL;
    esp_lcd_panel_io_i80_config_t io_config = {
        .cs_gpio_num = LCD_PIN_NUM_CS,
        .pclk_hz = 10 * 1000 * 1000,  // 10 MHz pixel clock
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
    ret = esp_lcd_new_panel_io_i80(i80_bus, &io_config, &io_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create panel IO: %s", esp_err_to_name(ret));
        heap_caps_free(framebuffer);
        return ret;
    }
    ESP_LOGI(TAG, "Panel IO configured");

    // Step 6: Initialize ST7789 panel
    ESP_LOGI(TAG, "Step 6: Initializing ST7789 panel...");
    esp_lcd_panel_handle_t panel_handle = NULL;
    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = LCD_PIN_NUM_RST,
        .rgb_endian = ESP_LCD_COLOR_SPACE_RGB,
        .bits_per_pixel = 16,
    };
    ret = esp_lcd_new_panel_st7789(io_handle, &panel_config, &panel_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create ST7789 panel: %s", esp_err_to_name(ret));
        heap_caps_free(framebuffer);
        return ret;
    }
    ESP_LOGI(TAG, "ST7789 panel created");

    // Step 7: Reset and initialize panel
    ESP_LOGI(TAG, "Step 7: Resetting and initializing panel...");
    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel_handle, true));  // T-Display S3 needs inversion
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel_handle, false, true));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel_handle, 0, 35));  // Offset for 170x320 display
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel_handle, true));
    ESP_LOGI(TAG, "Panel initialized and display turned ON");

    // Step 8: Fill screen with GREEN color
    ESP_LOGI(TAG, "Step 8: Drawing GREEN screen for testing...");
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        framebuffer[i] = COLOR_GREEN;  // RGB565: 0x07E0
    }
    esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, framebuffer);
    
    ESP_LOGI(TAG, "=== GREEN screen displayed! ===");
    ESP_LOGI(TAG, "If you see a green screen, your display is working correctly!");

    // Cleanup
    heap_caps_free(framebuffer);
    
    return ESP_OK;
}

// Stub functions - not used in this minimal test
esp_err_t display_show_qr_code(const display_device_info_t *info) {
    (void)info;
    return ESP_ERR_NOT_SUPPORTED;
}

esp_err_t display_show_clock(void) {
    return ESP_ERR_NOT_SUPPORTED;
}

esp_err_t display_toggle_mode(const display_device_info_t *info) {
    (void)info;
    return ESP_ERR_NOT_SUPPORTED;
}

void display_direct_test(void) {
    ESP_LOGI(TAG, "Direct test not implemented in minimal version");
}
