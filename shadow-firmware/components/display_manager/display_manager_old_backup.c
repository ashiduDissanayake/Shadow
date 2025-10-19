/* Minimal display manager: single-file clean implementation
 * - No QR, no button handling
 * - Initializes i80 + ST7789 and fills colors
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

static const char *TAG = "DISPLAY_MIN";

esp_err_t display_init(void) {
    ESP_LOGI(TAG, "Minimal display_init: starting");

    // Power on (GPIO15) and backlight
    gpio_config_t pwr = { .pin_bit_mask = (1ULL << 15), .mode = GPIO_MODE_OUTPUT };
    gpio_config(&pwr);
    gpio_set_level(15, 1);
    vTaskDelay(pdMS_TO_TICKS(10));

    gpio_config_t bk = { .pin_bit_mask = (1ULL << LCD_PIN_NUM_BK_LIGHT), .mode = GPIO_MODE_OUTPUT };
    gpio_config(&bk);
    gpio_set_level(LCD_PIN_NUM_BK_LIGHT, 1);

    // Framebuffer
    size_t buf_size = LCD_WIDTH * LCD_HEIGHT * sizeof(uint16_t);
    uint16_t *fb = heap_caps_malloc(buf_size, MALLOC_CAP_DMA);
    if (!fb) {
        ESP_LOGE(TAG, "Frame buffer alloc failed (%u bytes)", buf_size);
        return ESP_ERR_NO_MEM;
    }

    // i80 bus
    esp_lcd_i80_bus_handle_t i80_bus = NULL;
    esp_lcd_i80_bus_config_t bus_cfg = {
        .dc_gpio_num = LCD_PIN_NUM_DC,
        .wr_gpio_num = LCD_PIN_NUM_PCLK,
        .clk_src = LCD_CLK_SRC_DEFAULT,
        .data_gpio_nums = { LCD_PIN_NUM_DATA0, LCD_PIN_NUM_DATA1, LCD_PIN_NUM_DATA2, LCD_PIN_NUM_DATA3, LCD_PIN_NUM_DATA4, LCD_PIN_NUM_DATA5, LCD_PIN_NUM_DATA6, LCD_PIN_NUM_DATA7 },
        .bus_width = 8,
        .max_transfer_bytes = 64 * 1024,
    };
    esp_err_t err = esp_lcd_new_i80_bus(&bus_cfg, &i80_bus);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_i80_bus failed: %s", esp_err_to_name(err));
        heap_caps_free(fb);
        return err;
    }

    // panel io
    esp_lcd_panel_io_handle_t io = NULL;
    esp_lcd_panel_io_i80_config_t io_cfg = {
        .cs_gpio_num = LCD_PIN_NUM_CS,
        .pclk_hz = 10 * 1000 * 1000,
        .trans_queue_depth = 2,
        .dc_levels = { .dc_idle_level = 0, .dc_cmd_level = 0, .dc_dummy_level = 0, .dc_data_level = 1 },
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
    };
    err = esp_lcd_new_panel_io_i80(i80_bus, &io_cfg, &io);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_panel_io_i80 failed: %s", esp_err_to_name(err));
        heap_caps_free(fb);
        return err;
    }

    // panel
    esp_lcd_panel_handle_t panel = NULL;
    esp_lcd_panel_dev_config_t dev_cfg = { .reset_gpio_num = LCD_PIN_NUM_RST, .color_space = ESP_LCD_COLOR_SPACE_RGB, .bits_per_pixel = 16 };
    err = esp_lcd_new_panel_st7789(io, &dev_cfg, &panel);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_panel_st7789 failed: %s", esp_err_to_name(err));
        heap_caps_free(fb);
        return err;
    }

    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel, false, true));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel, 0, 35));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel, true));

    uint16_t colors[] = { COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_WHITE, COLOR_BLACK };
    for (size_t ci = 0; ci < sizeof(colors)/sizeof(colors[0]); ++ci) {
        for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; ++i) fb[i] = colors[ci];
        esp_lcd_panel_draw_bitmap(panel, 0, 0, LCD_WIDTH, LCD_HEIGHT, fb);
        vTaskDelay(pdMS_TO_TICKS(800));
    }

    heap_caps_free(fb);
    ESP_LOGI(TAG, "Minimal display test done");
    return ESP_OK;
}

esp_err_t display_show_qr_code(const display_device_info_t *info) { (void)info; return ESP_ERR_NOT_SUPPORTED; }
esp_err_t display_show_clock(void) { return ESP_ERR_NOT_SUPPORTED; }
esp_err_t display_toggle_mode(const display_device_info_t *info) { (void)info; return ESP_ERR_NOT_SUPPORTED; }
/**
 * Minimal display manager for hardware validation.
 * - Initializes 8080 i80 bus + ST7789 panel via esp_lcd
 * - Fills screen with a sequence of colors
 * - No QR, no buttons, no other logic
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

static const char *TAG = "DISPLAY_MIN";

esp_err_t display_init(void) {
    ESP_LOGI(TAG, "Minimal display_init: starting");

    // Power on (GPIO 15) and backlight
    gpio_config_t pwr = { .pin_bit_mask = (1ULL << 15), .mode = GPIO_MODE_OUTPUT };
    gpio_config(&pwr);
    gpio_set_level(15, 1);
    vTaskDelay(pdMS_TO_TICKS(10));

    gpio_config_t bk = { .pin_bit_mask = (1ULL << LCD_PIN_NUM_BK_LIGHT), .mode = GPIO_MODE_OUTPUT };
    gpio_config(&bk);
    gpio_set_level(LCD_PIN_NUM_BK_LIGHT, 1);

    // Allocate framebuffer
    size_t buf_size = LCD_WIDTH * LCD_HEIGHT * sizeof(uint16_t);
    uint16_t *fb = heap_caps_malloc(buf_size, MALLOC_CAP_DMA);
    if (!fb) {
        ESP_LOGE(TAG, "Frame buffer alloc failed (%u bytes)", buf_size);
        return ESP_ERR_NO_MEM;
    }

    // i80 bus
    esp_lcd_i80_bus_handle_t i80_bus = NULL;
    esp_lcd_i80_bus_config_t bus_cfg = {
        .dc_gpio_num = LCD_PIN_NUM_DC,
        .wr_gpio_num = LCD_PIN_NUM_PCLK,
        .clk_src = LCD_CLK_SRC_DEFAULT,
        .data_gpio_nums = { LCD_PIN_NUM_DATA0, LCD_PIN_NUM_DATA1, LCD_PIN_NUM_DATA2, LCD_PIN_NUM_DATA3, LCD_PIN_NUM_DATA4, LCD_PIN_NUM_DATA5, LCD_PIN_NUM_DATA6, LCD_PIN_NUM_DATA7 },
        .bus_width = 8,
        .max_transfer_bytes = 64 * 1024,
    };
    esp_err_t ret = esp_lcd_new_i80_bus(&bus_cfg, &i80_bus);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_i80_bus failed: %s", esp_err_to_name(ret));
        heap_caps_free(fb);
        return ret;
    }

    // Panel IO
    esp_lcd_panel_io_handle_t io = NULL;
    esp_lcd_panel_io_i80_config_t io_cfg = {
        .cs_gpio_num = LCD_PIN_NUM_CS,
        .pclk_hz = 10 * 1000 * 1000,
        .trans_queue_depth = 2,
        .dc_levels = { .dc_idle_level = 0, .dc_cmd_level = 0, .dc_dummy_level = 0, .dc_data_level = 1 },
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
    };
    ret = esp_lcd_new_panel_io_i80(i80_bus, &io_cfg, &io);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_panel_io_i80 failed: %s", esp_err_to_name(ret));
        heap_caps_free(fb);
        return ret;
    }

    // Panel
    esp_lcd_panel_handle_t panel = NULL;
    esp_lcd_panel_dev_config_t dev_cfg = { .reset_gpio_num = LCD_PIN_NUM_RST, .color_space = ESP_LCD_COLOR_SPACE_RGB, .bits_per_pixel = 16 };
    ret = esp_lcd_new_panel_st7789(io, &dev_cfg, &panel);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_panel_st7789 failed: %s", esp_err_to_name(ret));
        heap_caps_free(fb);
        return ret;
    }

    // Init
    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel, false, true));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel, 0, 35));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel, true));

    // Color test
    uint16_t colors[] = { COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_WHITE, COLOR_BLACK };
    for (size_t ci = 0; ci < sizeof(colors)/sizeof(colors[0]); ++ci) {
        for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; ++i) fb[i] = colors[ci];
        esp_lcd_panel_draw_bitmap(panel, 0, 0, LCD_WIDTH, LCD_HEIGHT, fb);
        vTaskDelay(pdMS_TO_TICKS(800));
    }

    heap_caps_free(fb);
    ESP_LOGI(TAG, "Minimal display test done");
    return ESP_OK;
}

// Stubs
esp_err_t display_show_qr_code(const display_device_info_t *info) { (void)info; return ESP_ERR_NOT_SUPPORTED; }
esp_err_t display_show_clock(void) { return ESP_ERR_NOT_SUPPORTED; }
esp_err_t display_toggle_mode(const display_device_info_t *info) { (void)info; return ESP_ERR_NOT_SUPPORTED; }
/**
 * Display Manager Implementation with ESP-IDF esp_lcd
 * Renders QR codes and clock display on ST7789 TFT (170x320)
 */

// Minimal display test implementation
#include "display_manager.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/gpio.h"
#include "esp_heap_caps.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"

static const char *TAG = "DISPLAY_MIN";

static esp_lcd_panel_io_handle_t io_handle = NULL;
static esp_lcd_panel_handle_t panel_handle = NULL;

// Minimal display_init: initialize i80 bus + ST7789 and show color test (no QR, no buttons)
esp_err_t display_init(void) {
    ESP_LOGI(TAG, "Minimal display_init: initializing panel...");

    // Allocate a DMA-capable frame buffer large enough for one full screen
    uint32_t buf_size = LCD_WIDTH * LCD_HEIGHT * sizeof(uint16_t);
    uint16_t *frame_buffer = heap_caps_malloc(buf_size, MALLOC_CAP_DMA);
    if (!frame_buffer) {
        ESP_LOGE(TAG, "Failed to allocate frame buffer (%u bytes)", buf_size);
        return ESP_ERR_NO_MEM;
    }

    // Configure backlight GPIO
    gpio_config_t bk_gpio_config = {
        .pin_bit_mask = (1ULL << LCD_PIN_NUM_BK_LIGHT),
        .mode = GPIO_MODE_OUTPUT,
    };
    ESP_ERROR_CHECK(gpio_config(&bk_gpio_config));
    gpio_set_level(LCD_PIN_NUM_BK_LIGHT, 1);  // Turn on backlight

    // Configure 8080 parallel bus
    esp_lcd_i80_bus_handle_t i80_bus = NULL;
    esp_lcd_i80_bus_config_t bus_config = {
        .dc_gpio_num = LCD_PIN_NUM_DC,
        .wr_gpio_num = LCD_PIN_NUM_PCLK,
        .clk_src = LCD_CLK_SRC_DEFAULT,
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
        .max_transfer_bytes = 64 * 1024, // use moderate chunk
    };
    esp_err_t ret = esp_lcd_new_i80_bus(&bus_config, &i80_bus);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_i80_bus failed: %s", esp_err_to_name(ret));
        heap_caps_free(frame_buffer);
        return ret;
    }

    // Panel IO config (use a conservative pixel clock)
    esp_lcd_panel_io_i80_config_t io_config = {
        .cs_gpio_num = LCD_PIN_NUM_CS,
        .pclk_hz = 10 * 1000 * 1000, // 10 MHz
        .trans_queue_depth = 2,
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
        ESP_LOGE(TAG, "esp_lcd_new_panel_io_i80 failed: %s", esp_err_to_name(ret));
        heap_caps_free(frame_buffer);
        return ret;
    }

    // Panel device config
    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = LCD_PIN_NUM_RST,
        .color_space = ESP_LCD_COLOR_SPACE_RGB,
        .bits_per_pixel = 16,
    };
    ret = esp_lcd_new_panel_st7789(io_handle, &panel_config, &panel_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_panel_st7789 failed: %s", esp_err_to_name(ret));
        heap_caps_free(frame_buffer);
        return ret;
    }

    // Basic init
    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel_handle, false));
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel_handle, false, true));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel_handle, 0, 35));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel_handle, true));

    // Color test: fill screen with red, green, blue, then black
    uint16_t colors[4] = { COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_BLACK };
    for (int c = 0; c < 4; ++c) {
        // fill buffer
        for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; ++i) {
            frame_buffer[i] = colors[c];
        }
        // draw
        esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer);
        vTaskDelay(pdMS_TO_TICKS(700));
    }

    heap_caps_free(frame_buffer);
    ESP_LOGI(TAG, "Minimal display test complete");
    return ESP_OK;
}

// Stub for QR display - intentionally not implemented in minimal test
esp_err_t display_show_qr_code(const display_device_info_t *info) {
    (void)info;
    return ESP_ERR_NOT_SUPPORTED;
}

        }
    }
}

// Draw colon for clock
static void draw_colon(uint16_t x, uint16_t y, uint16_t color, uint16_t bg) {
    // Top dot
    fill_rect(x, y + 20, 8, 8, color);
    // Bottom dot
    fill_rect(x, y + 40, 8, 8, color);
}

// Draw character (simple 5x7 font)
static void draw_char(uint16_t x, uint16_t y, char c, uint16_t color, uint16_t bg, uint8_t size) {
    if (!frame_buffer) return;
    
    uint8_t char_index = 0;
    
    // Map character to font index
    if (c == ' ') char_index = 0;
    else if (c >= '0' && c <= '9') char_index = c - '0' + 1;
    else if (c == ':') char_index = 11;
    else if (c == 'P' || c == 'p') char_index = 12;
    else if (c == 'E' || c == 'e') char_index = 13;
    else if (c == 'A' || c == 'a') char_index = 14;
    else if (c == 'H' || c == 'h') char_index = 15;
    else if (c == 'O' || c == 'o') char_index = 16;
    else if (c == 'L' || c == 'l') char_index = 17;
    else if (c == 'M' || c == 'm') char_index = 18;
    else if (c == 'N' || c == 'n') char_index = 19;
    else char_index = 0; // Default to space for unknown chars
    
    // Draw the character bitmap
    for (uint8_t col = 0; col < 5; col++) {
        uint8_t line = font5x7[char_index][col];
        for (uint8_t row = 0; row < 8; row++) {
            uint16_t pixel_color = (line & 0x01) ? color : bg;
            line >>= 1;
            
            // Draw scaled pixel
            for (uint8_t sy = 0; sy < size; sy++) {
                for (uint8_t sx = 0; sx < size; sx++) {
                    uint16_t px = x + (col * size) + sx;
                    uint16_t py = y + (row * size) + sy;
                    if (px < LCD_WIDTH && py < LCD_HEIGHT) {
                        frame_buffer[py * LCD_WIDTH + px] = pixel_color;
                    }
                }
            }
        }
    }
}

// Draw string
static void draw_string(uint16_t x, uint16_t y, const char *str, uint16_t color, uint16_t bg, uint8_t size) {
    uint16_t cursor_x = x;
    while (*str) {
        draw_char(cursor_x, y, *str++, color, bg, size);
        cursor_x += 6 * size; // 5 pixels + 1 pixel spacing
    }
}

// Update display from frame buffer
static void update_display(void) {
    if (!panel_handle || !frame_buffer) return;
    
    // Draw frame buffer to LCD
    esp_lcd_panel_draw_bitmap(panel_handle, 0, 0, LCD_WIDTH, LCD_HEIGHT, frame_buffer);
}

esp_err_t display_init(void) {
    ESP_LOGI(TAG, "Initializing ST7789 TFT display (170x320, 8-bit parallel)...");
    
    // CRITICAL: Enable power first (GPIO 15 for LilyGo T-Display S3)
    gpio_config_t power_gpio_config = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = 1ULL << 15,  // GPIO 15 is power enable
    };
    ESP_ERROR_CHECK(gpio_config(&power_gpio_config));
    gpio_set_level(15, 1);  // Turn on display power
    vTaskDelay(pdMS_TO_TICKS(10));  // Wait for power to stabilize
    ESP_LOGI(TAG, "Display power enabled (GPIO 15)");
    
    // Allocate frame buffer (170 * 320 * 2 bytes = 108,800 bytes)
    frame_buffer = heap_caps_malloc(LCD_WIDTH * LCD_HEIGHT * sizeof(uint16_t), MALLOC_CAP_DMA);
    if (!frame_buffer) {
        ESP_LOGE(TAG, "Failed to allocate frame buffer");
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "Frame buffer allocated: %d bytes", LCD_WIDTH * LCD_HEIGHT * sizeof(uint16_t));
    
    // Configure backlight GPIO
    gpio_config_t bk_gpio_config = {
        .pin_bit_mask = (1ULL << LCD_PIN_NUM_BK_LIGHT),
        .mode = GPIO_MODE_OUTPUT,
    };
    ESP_ERROR_CHECK(gpio_config(&bk_gpio_config));
    gpio_set_level(LCD_PIN_NUM_BK_LIGHT, 1);  // Turn on backlight
    
    // Configure 8080 parallel bus
    esp_lcd_i80_bus_handle_t i80_bus = NULL;
    esp_lcd_i80_bus_config_t bus_config = {
        .dc_gpio_num = LCD_PIN_NUM_DC,
        .wr_gpio_num = LCD_PIN_NUM_PCLK,
        .clk_src = LCD_CLK_SRC_DEFAULT,
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
    ESP_ERROR_CHECK(esp_lcd_new_i80_bus(&bus_config, &i80_bus));
    
    // Configure panel IO (20MHz for LilyGo T-Display S3)
    esp_lcd_panel_io_handle_t io_handle = NULL;
    esp_lcd_panel_io_i80_config_t io_config = {
        .cs_gpio_num = LCD_PIN_NUM_CS,
        .pclk_hz = 20 * 1000 * 1000,  // 20MHz (LilyGo factory setting)
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
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_i80(i80_bus, &io_config, &io_handle));
    
    // Configure ST7789 panel (RGB color space from factory examples!)
    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = LCD_PIN_NUM_RST,
        .color_space = ESP_LCD_COLOR_SPACE_RGB,  // RGB, not BGR!
        .bits_per_pixel = LCD_BIT_PER_PIXEL,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(io_handle, &panel_config, &panel_handle));
    
    // Initialize panel FIRST (basic reset and init)
    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel_handle));
    
    // Enable color inversion (LilyGo hardware requires this)
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel_handle, true));
    
    // Standard panel configuration
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel_handle, false, true));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel_handle, 0, 35));  // Offset for 170x320
    
    // CRITICAL: Send custom LCD_MODULE_CMD_1 initialization commands AFTER basic setup
    // This is what makes the LilyGo T-Display S3 work properly!
    ESP_LOGI(TAG, "Sending LCD_MODULE_CMD_1 initialization commands...");
    for (uint8_t i = 0; i < (sizeof(lcd_st7789v) / sizeof(lcd_cmd_t)); i++) {
        esp_lcd_panel_io_tx_param(io_handle, lcd_st7789v[i].cmd, 
                                   lcd_st7789v[i].data, 
                                   lcd_st7789v[i].len & 0x7F);
        if (lcd_st7789v[i].len & 0x80) {
            vTaskDelay(pdMS_TO_TICKS(120));  // 120ms delay if bit 7 is set
        }
    }
    ESP_LOGI(TAG, "Custom init commands sent successfully");
    
    // Turn on display
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel_handle, true));
    
    ESP_LOGI(TAG, "ST7789 display initialized successfully");
    
    // ========== SIMPLE DISPLAY TEST ==========
    // Test basic framebuffer rendering with solid colors
    ESP_LOGI(TAG, "========== DISPLAY TEST START ==========");
    
    // Test 1: Fill screen RED
    ESP_LOGI(TAG, "Test 1: Filling screen RED...");
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = COLOR_RED;
    }
    update_display();
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Test 2: Fill screen GREEN
    ESP_LOGI(TAG, "Test 2: Filling screen GREEN...");
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = COLOR_GREEN;
    }
    update_display();
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Test 3: Fill screen BLUE
    ESP_LOGI(TAG, "Test 3: Filling screen BLUE...");
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = COLOR_BLUE;
    }
    update_display();
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Test 4: Fill screen WHITE
    ESP_LOGI(TAG, "Test 4: Filling screen WHITE...");
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = COLOR_WHITE;
    }
    update_display();
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Test 5: Fill screen BLACK
    ESP_LOGI(TAG, "Test 5: Filling screen BLACK...");
    for (int i = 0; i < LCD_WIDTH * LCD_HEIGHT; i++) {
        frame_buffer[i] = COLOR_BLACK;
    }
    update_display();
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    ESP_LOGI(TAG, "========== DISPLAY TEST COMPLETE ==========");
    ESP_LOGI(TAG, "If you saw RED -> GREEN -> BLUE -> WHITE -> BLACK, display is working!");
    
    // Show initial clock (commented out for now - test basic colors first)
    // display_show_clock();
    
    return ESP_OK;
}

esp_err_t display_show_qr_code(const display_device_info_t *info) {
    if (!info || !frame_buffer || !panel_handle) {
        return ESP_ERR_INVALID_ARG;
    }
    
    ESP_LOGI(TAG, "Generating QR code for %s", info->device_name);
    
    // Format: "device_name:password"
    char qr_data[128];
    snprintf(qr_data, sizeof(qr_data), "%s:%s", info->device_name, info->password);
    ESP_LOGI(TAG, "QR Data: %s", qr_data);
    
    // Allocate QR code structure and module buffer
    QRCode qrcode;
    uint8_t qrcode_bytes[qrcode_getBufferSize(QR_VERSION)];
    
    // Generate QR code
    int8_t result = qrcode_initText(&qrcode, qrcode_bytes, QR_VERSION, ECC_MEDIUM, qr_data);
    if (result != 0) {
        ESP_LOGE(TAG, "Failed to generate QR code: %d", result);
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "QR Code generated successfully (%dx%d)", QR_SIZE, QR_SIZE);
    
    // Clear screen (cyan background for visibility)
    fill_rect(0, 0, LCD_WIDTH, LCD_HEIGHT, COLOR_CYAN);
    
    // Draw "SCAN QR" title at top
    draw_string(10, 10, "SCAN QR", COLOR_BLACK, COLOR_CYAN, 3);
    
    // Draw QR code with larger scale (6x6 pixels per module = 174x174)
    const uint8_t qr_scale = 6;
    const uint16_t qr_pixel_size = QR_SIZE * qr_scale;
    const uint16_t qr_x = (LCD_WIDTH - qr_pixel_size) / 2;
    const uint16_t qr_y = 60;
    
    // Draw white background for QR
    fill_rect(qr_x - 2, qr_y - 2, qr_pixel_size + 4, qr_pixel_size + 4, COLOR_WHITE);
    
    // Draw QR code modules
    for (uint8_t y = 0; y < QR_SIZE; y++) {
        for (uint8_t x = 0; x < QR_SIZE; x++) {
            bool module = qrcode_getModule(&qrcode, x, y);
            uint16_t color = module ? COLOR_BLACK : COLOR_WHITE;
            
            // Draw scaled pixel
            uint16_t px = qr_x + (x * qr_scale);
            uint16_t py = qr_y + (y * qr_scale);
            fill_rect(px, py, qr_scale, qr_scale, color);
        }
    }
    
    // Draw device name at bottom
    draw_string(5, 280, info->device_name, COLOR_BLACK, COLOR_CYAN, 2);
    
    // Update display
    update_display();
    
    ESP_LOGI(TAG, "✅ QR code displayed on TFT");
    ESP_LOGI(TAG, "📱 Scan with phone: %s", qr_data);
    
    return ESP_OK;
}

esp_err_t display_show_clock(void) {
    if (!frame_buffer || !panel_handle) {
        return ESP_ERR_INVALID_STATE;
    }
    
    ESP_LOGI(TAG, "Showing clock display");
    
    // Get current time
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm timeinfo;
    localtime_r(&tv.tv_sec, &timeinfo);
    
    // Extract hours and minutes
    uint8_t hour = timeinfo.tm_hour;
    uint8_t minute = timeinfo.tm_min;
    
    // Clear screen with black background
    fill_rect(0, 0, LCD_WIDTH, LCD_HEIGHT, COLOR_BLACK);
    
    // Draw "SHADOW" title at top
    draw_string(40, 20, "SHADOW", 0xFBE0, COLOR_BLACK, 2);  // Orange text
    
    // Draw date below title (using compile date as placeholder)
    draw_string(15, 50, __DATE__, COLOR_GREEN, COLOR_BLACK, 2);
    
    // Calculate position for time display (centered)
    // Each digit is 44 pixels wide (11*4 scale), colon is 8 pixels
    // Total: 44 + 44 + 8 + 44 + 44 = 184 pixels
    uint16_t time_x = (LCD_WIDTH - 184) / 2;
    uint16_t time_y = 120;
    
    // Color for time (orange like Arduino example)
    uint16_t time_color = 0xFBE0;  // Orange
    
    // Draw hours
    draw_large_digit(time_x, time_y, hour / 10, time_color, COLOR_BLACK);       // Tens
    draw_large_digit(time_x + 48, time_y, hour % 10, time_color, COLOR_BLACK);  // Ones
    
    // Draw colon
    draw_colon(time_x + 96, time_y, time_color, COLOR_BLACK);
    
    // Draw minutes
    draw_large_digit(time_x + 112, time_y, minute / 10, time_color, COLOR_BLACK); // Tens
    draw_large_digit(time_x + 160, time_y, minute % 10, time_color, COLOR_BLACK); // Ones
    
    // Draw seconds as small text at bottom
    char sec_str[16];
    snprintf(sec_str, sizeof(sec_str), ":%02d", timeinfo.tm_sec);
    draw_string(60, 250, sec_str, COLOR_WHITE, COLOR_BLACK, 2);
    
    // Update display
    update_display();
    
    ESP_LOGI(TAG, "Clock displayed: %02d:%02d:%02d", hour, minute, timeinfo.tm_sec);
    
    return ESP_OK;
}

esp_err_t display_toggle_mode(const display_device_info_t *info) {
    ESP_LOGI(TAG, "Toggling display mode (current: %s)", 
             current_mode == DISPLAY_MODE_CLOCK ? "CLOCK" : "QR");
    
    if (current_mode == DISPLAY_MODE_CLOCK) {
        current_mode = DISPLAY_MODE_QR;
        return display_show_qr_code(info);
    } else {
        current_mode = DISPLAY_MODE_CLOCK;
        return display_show_clock();
    }
}
