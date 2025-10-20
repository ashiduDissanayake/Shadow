/**
 * @file display_manager.h
 * @brief Display Manager for LilyGo T-Display S3
 * Manages the 1.9" ST7789 TFT display (170x320)
 * - Clock display (default)
 * - QR code display (on button press)
 * - Toggle between modes
 */

#ifndef DISPLAY_MANAGER_H
#define DISPLAY_MANAGER_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// LilyGo T-Display S3 Display Configuration
// After swap_xy(true), dimensions are: 170 wide × 320 tall (portrait mode)
#define LCD_WIDTH           320
#define LCD_HEIGHT          170
#define LCD_BIT_PER_PIXEL   16

// Pin definitions for 8-bit parallel interface
#define LCD_PIN_NUM_DATA0   39
#define LCD_PIN_NUM_DATA1   40
#define LCD_PIN_NUM_DATA2   41
#define LCD_PIN_NUM_DATA3   42
#define LCD_PIN_NUM_DATA4   45
#define LCD_PIN_NUM_DATA5   46
#define LCD_PIN_NUM_DATA6   47
#define LCD_PIN_NUM_DATA7   48
#define LCD_PIN_NUM_PCLK    8
#define LCD_PIN_NUM_CS      6
#define LCD_PIN_NUM_DC      7
#define LCD_PIN_NUM_RST     5
#define LCD_PIN_NUM_BK_LIGHT 38

// Color definitions (RGB565 format)
#define COLOR_BLACK     0x0000
#define COLOR_WHITE     0xFFFF
#define COLOR_RED       0xF800
#define COLOR_GREEN     0x07E0
#define COLOR_BLUE      0x001F
#define COLOR_YELLOW    0xFFE0
#define COLOR_CYAN      0x07FF
#define COLOR_MAGENTA   0xF81F
#define COLOR_GRAY      0x8410
#define COLOR_DARKGRAY  0x4208
#define COLOR_LIGHTGRAY 0xC618

/**
 * Display modes
 */
typedef enum {
    DISPLAY_MODE_CLOCK,      // Show clock
    DISPLAY_MODE_QR,         // Show QR code
    DISPLAY_MODE_STATUS,     // Show status message
    DISPLAY_MODE_OFF         // Screen off (power saving)
} display_mode_t;

/**
 * Display power state
 */
typedef enum {
    DISPLAY_POWER_ON,        // Display is on
    DISPLAY_POWER_OFF        // Display is off (backlight off)
} display_power_state_t;

/**
 * Device information for QR code display
 * Format: QR code contains only device name (e.g., "Shadow-9026")
 */
typedef struct {
    const char *device_name;
    const char *password;  // Deprecated - kept for backward compatibility, can be NULL
} display_device_info_t;

/**
 * Initialize the display
 * 
 * @return ESP_OK on success
 */
esp_err_t display_init(void);

/**
 * Show QR code with device information
 * Format: Just device name (e.g., "Shadow-9026")
 * Password field is no longer used
 * 
 * @param info Device information (only device_name is used)
 * @return ESP_OK on success
 */
esp_err_t display_show_qr_code(const display_device_info_t *info);

/**
 * Show clock display
 * 
 * @return ESP_OK on success
 */
esp_err_t display_show_clock(void);

/**
 * Toggle display mode (clock <-> QR code)
 * Called by button handler
 * 
 * @param info Device information for QR code
 * @return ESP_OK on success
 */
esp_err_t display_toggle_mode(const display_device_info_t *info);

/**
 * Turn display power on or off (backlight control)
 * 
 * @param on true to turn on, false to turn off
 * @return ESP_OK on success
 */
esp_err_t display_set_power(bool on);

/**
 * Get current display power state
 * 
 * @return true if display is on, false if off
 */
bool display_is_on(void);

/**
 * Refresh the current display mode (update clock time, etc.)
 * Should be called periodically when display is on
 * 
 * @return ESP_OK on success
 */
esp_err_t display_refresh(void);

/**
 * Get current display mode
 * 
 * @return Current display mode
 */
display_mode_t display_get_mode(void);

/**
 * DIRECT GPIO TEST - Uses TFT_eSPI approach (no esp_lcd)
 * Minimal test: just cycle through colors
 * This uses the EXACT same register writes as working Arduino code
 */
void display_direct_test(void);

#ifdef __cplusplus
}
#endif

#endif // DISPLAY_MANAGER_H
