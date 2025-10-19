/**
 * @file display_direct_test.c
 * @brief MINIMAL Direct GPIO parallel display test (TFT_eSPI approach)
 * 
 * This uses the EXACT same approach as Arduino TFT_eSPI library - direct GPIO register writes.
 * NO esp_lcd API! Just raw GPIO manipulation like the working Arduino code.
 */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_system.h"
#include "soc/gpio_struct.h"

static const char *TAG = "DisplayDirectTest";

// Pin definitions - same as TFT_eSPI Setup206
#define TFT_D0   39
#define TFT_D1   40
#define TFT_D2   41
#define TFT_D3   42
#define TFT_D4   45
#define TFT_D5   46
#define TFT_D6   47
#define TFT_D7   48
#define TFT_WR   8
#define TFT_DC   7
#define TFT_CS   6
#define TFT_RST  5
#define TFT_BL   38

// Since D0-D7 are all >= 32, use the high GPIO registers
#define MASK_OFFSET 32
#define GPIO_CLR_REG GPIO.out1_w1tc.val
#define GPIO_SET_REG GPIO.out1_w1ts.val

// WR is GPIO 8 (< 32)
#define WR_L GPIO.out_w1tc = (1 << TFT_WR)
#define WR_H GPIO.out_w1ts = (1 << TFT_WR)

// Lookup table for fast bit setting (like TFT_eSPI)
static uint32_t xset_mask[256];

// GPIO direction mask for data pins (D0-D7)
#define GPIO_DIR_MASK ((1 << (TFT_D0-MASK_OFFSET)) | (1 << (TFT_D1-MASK_OFFSET)) | \
                       (1 << (TFT_D2-MASK_OFFSET)) | (1 << (TFT_D3-MASK_OFFSET)) | \
                       (1 << (TFT_D4-MASK_OFFSET)) | (1 << (TFT_D5-MASK_OFFSET)) | \
                       (1 << (TFT_D6-MASK_OFFSET)) | (1 << (TFT_D7-MASK_OFFSET)))

// Clear mask for data + WR (both registers)
#define GPIO_OUT_CLR_DATA (GPIO_DIR_MASK)

// Fast byte set function
#define set_mask(C) xset_mask[C]

// Write 8 bits to TFT (like TFT_eSPI)
#define tft_Write_8(C) GPIO_CLR_REG = GPIO_OUT_CLR_DATA; WR_L; GPIO_SET_REG = set_mask((uint8_t)(C)); WR_H

// Write 16 bits to TFT (2 bytes, high byte first)
#define tft_Write_16(C) \
    GPIO_CLR_REG = GPIO_OUT_CLR_DATA; WR_L; GPIO_SET_REG = set_mask((uint8_t)((C) >> 8)); WR_H; \
    GPIO_CLR_REG = GPIO_OUT_CLR_DATA; WR_L; GPIO_SET_REG = set_mask((uint8_t)((C) >> 0)); WR_H

/**
 * Initialize lookup table for fast bit setting
 */
static void init_lookup_table(void)
{
    for (int32_t c = 0; c < 256; c++) {
        xset_mask[c] = 0;
        if (c & 0x01) xset_mask[c] |= (1 << (TFT_D0 - MASK_OFFSET));
        if (c & 0x02) xset_mask[c] |= (1 << (TFT_D1 - MASK_OFFSET));
        if (c & 0x04) xset_mask[c] |= (1 << (TFT_D2 - MASK_OFFSET));
        if (c & 0x08) xset_mask[c] |= (1 << (TFT_D3 - MASK_OFFSET));
        if (c & 0x10) xset_mask[c] |= (1 << (TFT_D4 - MASK_OFFSET));
        if (c & 0x20) xset_mask[c] |= (1 << (TFT_D5 - MASK_OFFSET));
        if (c & 0x40) xset_mask[c] |= (1 << (TFT_D6 - MASK_OFFSET));
        if (c & 0x80) xset_mask[c] |= (1 << (TFT_D7 - MASK_OFFSET));
    }
    ESP_LOGI(TAG, "Lookup table initialized");
}

/**
 * Configure GPIO pins
 */
static void init_gpio(void)
{
    // Power GPIO 15
    gpio_config_t power_cfg = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = (1ULL << 15)
    };
    gpio_config(&power_cfg);
    gpio_set_level(15, 1);
    ESP_LOGI(TAG, "GPIO 15 power enabled");

    // Data pins D0-D7 (39,40,41,42,45,46,47,48)
    gpio_config_t data_cfg = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = (1ULL << TFT_D0) | (1ULL << TFT_D1) | (1ULL << TFT_D2) | (1ULL << TFT_D3) |
                        (1ULL << TFT_D4) | (1ULL << TFT_D5) | (1ULL << TFT_D6) | (1ULL << TFT_D7)
    };
    gpio_config(&data_cfg);

    // Control pins
    gpio_config_t ctrl_cfg = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = (1ULL << TFT_WR) | (1ULL << TFT_DC) | (1ULL << TFT_CS) | 
                        (1ULL << TFT_RST) | (1ULL << TFT_BL)
    };
    gpio_config(&ctrl_cfg);

    // Initial states
    gpio_set_level(TFT_CS, 1);   // CS HIGH (inactive)
    gpio_set_level(TFT_WR, 1);   // WR HIGH (inactive)
    gpio_set_level(TFT_RST, 1);  // RST HIGH (not in reset)
    gpio_set_level(TFT_DC, 1);   // DC HIGH (data mode)
    gpio_set_level(TFT_BL, 1);   // Backlight ON
    
    ESP_LOGI(TAG, "GPIO pins configured");
}

/**
 * Send command to ST7789
 */
static void write_command(uint8_t cmd)
{
    gpio_set_level(TFT_DC, 0);  // DC LOW = command
    gpio_set_level(TFT_CS, 0);  // CS LOW = chip selected
    tft_Write_8(cmd);
    gpio_set_level(TFT_CS, 1);  // CS HIGH = deselect
}

/**
 * Send data byte to ST7789
 */
static void write_data(uint8_t data)
{
    gpio_set_level(TFT_DC, 1);  // DC HIGH = data
    gpio_set_level(TFT_CS, 0);  // CS LOW
    tft_Write_8(data);
    gpio_set_level(TFT_CS, 1);  // CS HIGH
}

/**
 * Hardware reset ST7789
 */
static void reset_display(void)
{
    gpio_set_level(TFT_RST, 1);
    vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(TFT_RST, 0);
    vTaskDelay(pdMS_TO_TICKS(20));
    gpio_set_level(TFT_RST, 1);
    vTaskDelay(pdMS_TO_TICKS(150));
    ESP_LOGI(TAG, "Display reset complete");
}

/**
 * Initialize ST7789 display
 */
static void init_st7789(void)
{
    reset_display();

    write_command(0x01);  // Software reset
    vTaskDelay(pdMS_TO_TICKS(150));

    write_command(0x11);  // Sleep out
    vTaskDelay(pdMS_TO_TICKS(120));

    write_command(0x3A);  // COLMOD: Pixel Format Set
    write_data(0x05);     // 16-bit/pixel (RGB565)

    write_command(0x36);  // MADCTL: Memory Data Access Control
    write_data(0x00);     // Normal orientation

    write_command(0x21);  // Display Inversion ON (like Setup206)

    write_command(0x13);  // Normal Display Mode On

    write_command(0x29);  // Display ON
    vTaskDelay(pdMS_TO_TICKS(10));

    ESP_LOGI(TAG, "ST7789 initialized");
}

/**
 * Set display window (column/row addresses)
 */
static void set_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1)
{
    // CASET: Column Address Set
    write_command(0x2A);
    write_data(x0 >> 8);
    write_data(x0 & 0xFF);
    write_data(x1 >> 8);
    write_data(x1 & 0xFF);

    // RASET: Row Address Set
    write_command(0x2B);
    write_data(y0 >> 8);
    write_data(y0 & 0xFF);
    write_data(y1 >> 8);
    write_data(y1 & 0xFF);

    // RAMWR: Memory Write
    write_command(0x2C);
}

/**
 * Fill entire screen with solid color
 */
static void fill_screen(uint16_t color)
{
    ESP_LOGI(TAG, "Filling screen with color 0x%04X", color);
    
    set_window(0, 0, 169, 319);  // Full screen (170x320)
    
    gpio_set_level(TFT_DC, 1);  // DC = data mode
    gpio_set_level(TFT_CS, 0);  // CS = active
    
    // Write pixels (170 * 320 = 54,400 pixels)
    for (int i = 0; i < 54400; i++) {
        tft_Write_16(color);
    }
    
    gpio_set_level(TFT_CS, 1);  // CS = inactive
    ESP_LOGI(TAG, "Screen fill complete");
}

/**
 * Display test - cycle through colors
 */
void display_direct_test(void)
{
    ESP_LOGI(TAG, "=== DIRECT GPIO DISPLAY TEST (TFT_eSPI approach) ===");
    
    init_lookup_table();
    init_gpio();
    init_st7789();
    
    ESP_LOGI(TAG, "Starting color test...");
    
    while (1) {
        ESP_LOGI(TAG, "RED");
        fill_screen(0xF800);  // Red
        vTaskDelay(pdMS_TO_TICKS(2000));
        
        ESP_LOGI(TAG, "GREEN");
        fill_screen(0x07E0);  // Green
        vTaskDelay(pdMS_TO_TICKS(2000));
        
        ESP_LOGI(TAG, "BLUE");
        fill_screen(0x001F);  // Blue
        vTaskDelay(pdMS_TO_TICKS(2000));
        
        ESP_LOGI(TAG, "WHITE");
        fill_screen(0xFFFF);  // White
        vTaskDelay(pdMS_TO_TICKS(2000));
        
        ESP_LOGI(TAG, "BLACK");
        fill_screen(0x0000);  // Black
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
