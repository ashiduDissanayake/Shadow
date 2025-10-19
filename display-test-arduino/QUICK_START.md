# Quick Start Guide

## 🚀 Test Arduino First (2 minutes)

1. **Open Arduino IDE**

2. **Copy TFT_eSPI library:**
   ```bash
   cp -r /Users/ashidudissanayake/Dev/Shadow/TFT_eSPI ~/Documents/Arduino/libraries/
   ```

3. **Enable Setup206:**
   Edit `~/Documents/Arduino/libraries/TFT_eSPI/User_Setup_Select.h`:
   ```cpp
   // Line ~22 - Comment out default:
   //#include <User_Setup.h>
   
   // Line ~141 - Enable this:
   #include <User_Setups/Setup206_LilyGo_T_Display_S3.h>
   ```

4. **Open sketch:**
   ```
   File → Open → /Users/ashidudissanayake/Dev/Shadow/display-test-arduino/display_test.ino
   ```

5. **Configure board:**
   - Tools → Board → ESP32 Arduino → **ESP32S3 Dev Module**
   - Tools → USB CDC On Boot → **Enabled**  
   - Tools → PSRAM → **OPI PSRAM**
   - Tools → Port → **/dev/cu.usbmodem####**

6. **Upload** (→ button)

7. **Open Serial Monitor** (115200 baud)

### ✅ Success Looks Like:
- Color test: RED → GREEN → BLUE → BLACK
- Clock display with orange time
- Serial: "Display initialized successfully!"

### ❌ Failure Means:
- Hardware issue (unlikely - you said it works in Arduino)
- Wrong TFT_eSPI configuration
- Wrong board selected

---

## 🔧 Fix ESP-IDF (Once Arduino Works)

Edit: `shadow-firmware/components/display_manager/include/display_manager.h`

**Find these lines (~35-38):**
```c
#define LCD_PIN_NUM_DATA4   43  // ← WRONG
#define LCD_PIN_NUM_DATA5   44  // ← WRONG
#define LCD_PIN_NUM_DATA6   45  // ← WRONG
#define LCD_PIN_NUM_DATA7   46  // ← WRONG
```

**Change to:**
```c
#define LCD_PIN_NUM_DATA4   45  // ← CORRECT
#define LCD_PIN_NUM_DATA5   46  // ← CORRECT
#define LCD_PIN_NUM_DATA6   47  // ← CORRECT
#define LCD_PIN_NUM_DATA7   48  // ← CORRECT
```

**Rebuild:**
```bash
cd ~/Dev/Shadow/shadow-firmware
idf.py build flash monitor
```

### ✅ Success Looks Like:
- Clock displays properly
- Colors are correct
- QR code scannable
- Button toggles work

---

## 📊 Summary

| Component | Current | Correct | Status |
|-----------|---------|---------|--------|
| DATA0-3   | 39-42   | 39-42   | ✅ OK  |
| **DATA4** | **43**  | **45**  | ❌ FIX |
| **DATA5** | **44**  | **46**  | ❌ FIX |
| **DATA6** | **45**  | **47**  | ❌ FIX |
| **DATA7** | **46**  | **48**  | ❌ FIX |
| WR (PCLK) | 8       | 8       | ✅ OK  |
| CS        | 6       | 6       | ✅ OK  |
| DC        | 7       | 7       | ✅ OK  |
| RST       | 5       | 5       | ✅ OK  |
| BL        | 38      | 38      | ✅ OK  |

**4 out of 8 data pins were wrong = scrambled display!**

---

## 💡 Why This Happened

LilyGo **deliberately skipped GPIO 43-44** because they're strapping pins. You probably assumed sequential numbering (39,40,41,42,**43,44,45,46**) but they used (39,40,41,42,**45,46,47,48**).

This is the #1 issue the expert predicted! 🎯
