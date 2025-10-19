# LilyGo T-Display S3 - Display Test (Arduino)

## Purpose
This is a minimal Arduino sketch to verify the LilyGo T-Display S3 hardware works correctly using the TFT_eSPI library.

## What This Tests
1. ✅ Display initialization
2. ✅ Color rendering (red, green, blue test)
3. ✅ Text drawing capabilities
4. ✅ Digital clock display
5. ✅ Pin configuration validation

## Setup Instructions

### 1. Install Arduino IDE
If not already installed, download from: https://www.arduino.cc/en/software

### 2. Install ESP32 Board Support
1. Open Arduino IDE
2. Go to `File` → `Preferences`
3. Add to "Additional Boards Manager URLs":
   ```
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
4. Go to `Tools` → `Board` → `Boards Manager`
5. Search for "ESP32" by Espressif Systems
6. Install version 2.0.11 or newer

### 3. Install TFT_eSPI Library
1. Go to `Sketch` → `Include Library` → `Manage Libraries`
2. Search for "TFT_eSPI"
3. Install by Bodmer

### 4. Configure TFT_eSPI for LilyGo T-Display S3

**Option A: Use LilyGo's Setup File (RECOMMENDED)**
1. Copy the TFT_eSPI library from `/Users/ashidudissanayake/Dev/Shadow/TFT_eSPI` to Arduino libraries folder:
   - macOS: `~/Documents/Arduino/libraries/TFT_eSPI/`
2. Open `User_Setup_Select.h` in the TFT_eSPI library folder
3. Comment out the default setup and enable Setup206:
   ```cpp
   //#include <User_Setup.h>           // Default setup - COMMENT THIS OUT
   #include <User_Setups/Setup206_LilyGo_T_Display_S3.h>  // ENABLE THIS
   ```

**Option B: Manual Configuration**
Edit `User_Setup.h` in TFT_eSPI library and set:
```cpp
#define ST7789_DRIVER
#define TFT_WIDTH  170
#define TFT_HEIGHT 320

// LilyGo T-Display S3 pins
#define TFT_MOSI  11  // SDA
#define TFT_SCLK  12  // SCL
#define TFT_CS    10  // Chip select
#define TFT_DC     9  // Data/Command
#define TFT_RST    8  // Reset
#define TFT_BL     6  // Backlight

#define LOAD_GLCD
#define LOAD_FONT2
#define LOAD_FONT4
#define LOAD_FONT6
#define LOAD_FONT7
#define LOAD_FONT8
#define LOAD_GFXFF

#define SMOOTH_FONT
#define SPI_FREQUENCY  40000000  // 40MHz
```

### 5. Upload the Sketch
1. Open `display_test.ino` in Arduino IDE
2. Select board: `Tools` → `Board` → `ESP32 Arduino` → `ESP32S3 Dev Module`
3. Configure board settings:
   - USB CDC On Boot: "Enabled"
   - Flash Mode: "QIO 80MHz"
   - Flash Size: "16MB (128Mb)"
   - Partition Scheme: "16M Flash (3MB APP/9.9MB FATFS)"
   - PSRAM: "OPI PSRAM"
   - Upload Speed: "921600"
4. Select Port: `Tools` → `Port` → `/dev/cu.usbmodem####` (your board's port)
5. Click Upload (→) button

## Expected Behavior

### On Serial Monitor (115200 baud)
```
=== LilyGo T-Display S3 Test ===
Initializing TFT display...
Display initialized successfully!
Display size: 320 x 170
Testing colors...
Color test complete!
Time: 12:34:56
Time: 12:34:57
...
```

### On Display
1. **Color test sequence** (0.5s each):
   - Solid RED screen
   - Solid GREEN screen
   - Solid BLUE screen
   - BLACK background
2. **Clock display**:
   - Large orange time (HH:MM format)
   - Flashing colon
   - Green text: "SHADOW Display Test"
   - Date stamp
   - Centered "LilyGo T-Display S3" text
   - Random color value display

## Troubleshooting

### Display stays dark/blank
- ❌ **Check USB cable** - use a data cable, not charge-only
- ❌ **Check TFT_eSPI configuration** - verify Setup206 is enabled
- ❌ **Check backlight pin** - GPIO 6 should be HIGH
- ❌ **Check power** - board should have 5V via USB

### Colors are wrong
- Try changing `#define TFT_RGB_ORDER TFT_BGR` to `TFT_RGB` in User_Setup.h
- Or vice versa

### Display is upside down / rotated
- Change `tft.setRotation(1);` to `0`, `2`, or `3` in setup()

### Upload fails
- Press and hold BOOT button while clicking Upload
- Release BOOT after "Connecting..." appears

## Next Steps

**If this works:**
✅ Hardware is confirmed working
✅ Pin configuration is correct  
✅ Interface type is identified (SPI in this case)
→ We can now properly configure ESP-IDF with the correct settings

**If this doesn't work:**
❌ Hardware issue (display, board, or wiring)
❌ Wrong TFT_eSPI configuration
→ Need to debug Arduino setup first before moving to ESP-IDF

## Critical Information This Test Reveals

1. **Interface Type**: LilyGo T-Display S3 uses **8-BIT PARALLEL** interface
   - NOT SPI! This is important for ESP-IDF configuration
   
2. **Correct Pins** (from Setup206_LilyGo_T_Display_S3.h):
   ```
   TFT_CS  = 6
   TFT_DC  = 7
   TFT_RST = 5
   TFT_WR  = 8  (Write strobe)
   TFT_RD  = 9  (Read strobe)
   TFT_D0-D7 = 39, 40, 41, 42, 45, 46, 47, 48 (8-bit data bus)
   TFT_BL  = 38 (Backlight)
   ```

3. **Display specs**:
   - 170x320 pixels
   - ST7789 driver
   - **8-bit parallel interface** (i80 bus)
   - RGB color order (not BGR)
   - Color inversion ON

## Contact
If you see this working, we know exactly how to fix the ESP-IDF code!
