# Pin Configuration Comparison: ESP-IDF vs Arduino TFT_eSPI

## CRITICAL FINDING: Your ESP-IDF code has WRONG GPIO pin numbers!

### ❌ Your Current ESP-IDF Configuration (WRONG)
```c
// From display_manager.h
#define LCD_PIN_NUM_DATA0   39  ✅ CORRECT
#define LCD_PIN_NUM_DATA1   40  ✅ CORRECT
#define LCD_PIN_NUM_DATA2   41  ✅ CORRECT
#define LCD_PIN_NUM_DATA3   42  ✅ CORRECT
#define LCD_PIN_NUM_DATA4   43  ❌ WRONG! Should be 45
#define LCD_PIN_NUM_DATA5   44  ❌ WRONG! Should be 46
#define LCD_PIN_NUM_DATA6   45  ❌ WRONG! Should be 47
#define LCD_PIN_NUM_DATA7   46  ❌ WRONG! Should be 48

#define LCD_PIN_NUM_PCLK     8  ✅ CORRECT (WR strobe)
#define LCD_PIN_NUM_CS       6  ✅ CORRECT
#define LCD_PIN_NUM_DC       7  ✅ CORRECT
#define LCD_PIN_NUM_RST      5  ✅ CORRECT
#define LCD_PIN_NUM_BL      38  ✅ CORRECT
```

### ✅ Correct Configuration (from Setup206_LilyGo_T_Display_S3.h)
```c
#define TFT_D0  39  // Data bit 0
#define TFT_D1  40  // Data bit 1
#define TFT_D2  41  // Data bit 2
#define TFT_D3  42  // Data bit 3
#define TFT_D4  45  // Data bit 4 (YOU HAD 43!)
#define TFT_D5  46  // Data bit 5 (YOU HAD 44!)
#define TFT_D6  47  // Data bit 6 (YOU HAD 45!)
#define TFT_D7  48  // Data bit 7 (YOU HAD 46!)

#define TFT_WR   8  // Write strobe (PCLK)
#define TFT_RD   9  // Read strobe (not used in ESP-IDF write-only mode)
#define TFT_CS   6  // Chip select
#define TFT_DC   7  // Data/Command
#define TFT_RST  5  // Reset
#define TFT_BL  38  // Backlight
```

## The Problem

**DATA4, DATA5, DATA6, DATA7 were WRONG!**

You had:
```
DATA4=43, DATA5=44, DATA6=45, DATA7=46
```

Should be:
```
DATA4=45, DATA5=46, DATA6=47, DATA7=48
```

This means the upper 4 bits of every pixel color were going to the **wrong GPIO pins**! This would cause:
- Completely scrambled colors
- Random "pickle" patterns
- Dark blue tint (because upper bits control brightness/color intensity)

## Other Configuration Differences

### Color Space
```c
// Arduino Setup206
#define TFT_RGB_ORDER TFT_RGB  // RGB order
#define TFT_INVERSION_ON       // Inversion enabled

// Your ESP-IDF (CORRECT!)
panel_config.color_space = ESP_LCD_COLOR_SPACE_RGB;
esp_lcd_panel_invert_color(panel_handle, true);
```

### Display Size
```c
// Arduino Setup206
#define TFT_WIDTH 170
#define TFT_HEIGHT 320

// Your ESP-IDF (CORRECT!)
#define LCD_WIDTH  170
#define LCD_HEIGHT 320
```

### GPIO 15 Power Enable
- **NOT mentioned in Setup206** - may not be needed, or handled automatically by Arduino framework
- Try commenting out GPIO 15 code in ESP-IDF if display still doesn't work after pin fix

## Fix for ESP-IDF Code

Update `/Users/ashidudissanayake/Dev/Shadow/shadow-firmware/components/display_manager/include/display_manager.h`:

```c
// 8-bit parallel data pins (CORRECTED!)
#define LCD_PIN_NUM_DATA0   39
#define LCD_PIN_NUM_DATA1   40
#define LCD_PIN_NUM_DATA2   41
#define LCD_PIN_NUM_DATA3   42
#define LCD_PIN_NUM_DATA4   45  // ← CHANGED FROM 43
#define LCD_PIN_NUM_DATA5   46  // ← CHANGED FROM 44
#define LCD_PIN_NUM_DATA6   47  // ← CHANGED FROM 45
#define LCD_PIN_NUM_DATA7   48  // ← CHANGED FROM 46
```

This is almost certainly why your display showed "pickle colors" and dark blue!

## Why GPIO 43-44 Don't Work

Looking at ESP32-S3 datasheet:
- **GPIO 43-44** are strapping pins and may have special restrictions
- **GPIO 45-48** are general-purpose I/O safe for parallel data

LilyGo deliberately skipped GPIO 43-44 in the data bus for this reason.

## Next Steps

1. **First**: Test the Arduino sketch to confirm hardware works
2. **Then**: Apply the GPIO pin fix to ESP-IDF code
3. **Test ESP-IDF** with corrected pins
4. If still issues, try removing GPIO 15 power enable code
