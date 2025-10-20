# Changes Summary - Button Controls & UI Improvements

**Date**: October 20, 2025  
**Changes**: Button handler implementation, UI improvements, calibration integration

---

## 1. macOS App UI Changes ✅

### Removed "Shadow BLE" from Navigation Bar

**File**: `Shadow/Shadow/Shared/ShadowAppNavBar.swift`

**Change**: Removed the BLE status indicator button and popover from the navigation bar for a cleaner UI.

**Before:**
```swift
// BLE status indicator with "Shadow BLE" text
Button(action: { showBLEPopover.toggle() }) {
    HStack {
        bleStatusIcon
        VStack {
            Text("Shadow BLE")
            Text(bleStatusText)
        }
    }
}
```

**After:**
```swift
// Clean navigation: Calendar button and Profile only
Spacer()
Button(action: onCalendarTap) { ... }
Button(action: onProfileTap) { ... }
```

**Result**: Cleaner, more minimal navigation bar

---

### Replaced Debug Button with Forget Button

**File**: `Shadow/Shadow/Features/Dashboard/ShadowDashboardView.swift`

**Change**: Replaced the purple "Debug" button with a red "Forget" button that appears only when a device is paired.

**Before:**
```swift
Button("Debug") { showingDebugLog = true }
    .buttonStyle(ShadowButtonStyle(color: .purple))
```

**After:**
```swift
if syncViewModel.manager.isPairedToDevice {
    Button("Forget") {
        Task {
            try? await syncViewModel.manager.unpairDevice()
            UserDefaults.standard.removeObject(forKey: "paired_device_id")
        }
    }
    .buttonStyle(ShadowButtonStyle(color: .red))
}
```

**Result**: 
- More useful action (forget device instead of debug log)
- Only shown when device is paired
- Properly unpairs device and clears saved data

---

## 2. ESP32 Firmware Button Configuration ✅

### T-Display S3 Button Pin Mapping

| Button | GPIO | Purpose | Function |
|--------|------|---------|----------|
| **LEFT** | 0 | Calibration Control | Start/Stop 10-min calibration |
| **RIGHT** | 14 | Display Toggle | Switch between clock and QR code |

### Pin Definitions Updated

**File**: `shadow-firmware/main/main_realtime.c`

**Before:**
```c
#define BUTTON_PIN          14    // Single button
```

**After:**
```c
#define BUTTON_LEFT_PIN     0     // Calibration control
#define BUTTON_RIGHT_PIN    14    // Display toggle
```

---

### Interrupt Handlers Added

**Added Two Separate ISR Handlers:**

1. **`button_left_interrupt_handler()`** - Calibration control
   - Debounced with 200ms timeout
   - Sends `SENSOR_EVENT_BUTTON_LEFT` to queue

2. **`button_right_interrupt_handler()`** - Display toggle
   - Debounced with 200ms timeout
   - Sends `SENSOR_EVENT_BUTTON_RIGHT` to queue

**Code:**
```c
static void IRAM_ATTR button_left_interrupt_handler(void *arg) {
    int64_t now = esp_timer_get_time() / 1000;
    if (now - last_button_left_press < BUTTON_DEBOUNCE_MS) {
        return;
    }
    last_button_left_press = now;
    
    sensor_event_t event = {
        .type = SENSOR_EVENT_BUTTON_LEFT,
        .timestamp_us = esp_timer_get_time(),
        .sequence = 0
    };
    xQueueSendFromISR(sensor_event_queue, &event, &xHigherPriorityTaskWoken);
}

// Similar implementation for button_right_interrupt_handler()
```

---

### Event Types Extended

**File**: `shadow-firmware/main/main_realtime.c`

**Added:**
```c
typedef enum {
    // ... existing events ...
    SENSOR_EVENT_BUTTON_LEFT,   // Left button (calibration)
    SENSOR_EVENT_BUTTON_RIGHT,  // Right button (display)
} sensor_event_type_t;
```

---

### Event Handlers Implemented

**In Producer Task Event Loop:**

```c
case SENSOR_EVENT_BUTTON_LEFT:
    // Toggle calibration
    ESP_LOGI(TAG_MAIN, "🔘 Left button pressed - calibration control");
    if (calibration_get_state() == CAL_STATE_IN_PROGRESS) {
        ESP_LOGI(TAG_MAIN, "🔴 Stopping calibration");
        calibration_stop(false);
        if (calibration_is_calibrated()) {
            ESP_LOGI(TAG_MAIN, "✅ Calibration completed successfully!");
        } else {
            ESP_LOGW(TAG_MAIN, "⚠️ Calibration failed or incomplete");
        }
    } else {
        ESP_LOGI(TAG_MAIN, "🟢 Starting calibration (10 minutes)");
        calibration_start();
    }
    break;

case SENSOR_EVENT_BUTTON_RIGHT:
    // Toggle display mode
    ESP_LOGI(TAG_MAIN, "🔘 Right button pressed - toggling display");
    display_toggle_mode(&g_device_info);
    break;
```

---

### GPIO Configuration Updated

**Setup Both Buttons:**

```c
// Left button (GPIO 0) - Calibration control
gpio_config_t btn_left_conf = {
    .pin_bit_mask = (1ULL << BUTTON_LEFT_PIN),
    .mode = GPIO_MODE_INPUT,
    .pull_up_en = GPIO_PULLUP_ENABLE,
    .pull_down_en = GPIO_PULLDOWN_DISABLE,
    .intr_type = GPIO_INTR_NEGEDGE
};
ESP_ERROR_CHECK(gpio_config(&btn_left_conf));
ESP_ERROR_CHECK(gpio_isr_handler_add(BUTTON_LEFT_PIN, button_left_interrupt_handler, NULL));

// Right button (GPIO 14) - Display toggle
gpio_config_t btn_right_conf = {
    .pin_bit_mask = (1ULL << BUTTON_RIGHT_PIN),
    .mode = GPIO_MODE_INPUT,
    .pull_up_en = GPIO_PULLUP_ENABLE,
    .pull_down_en = GPIO_PULLDOWN_DISABLE,
    .intr_type = GPIO_INTR_NEGEDGE
};
ESP_ERROR_CHECK(gpio_config(&btn_right_conf));
ESP_ERROR_CHECK(gpio_isr_handler_add(BUTTON_RIGHT_PIN, button_right_interrupt_handler, NULL));
```

---

## 3. Calibration System Integration ✅

### Header Include Added

**File**: `shadow-firmware/main/main_realtime.c`

**Added:**
```c
#include "calibration.h"  // Calibration system for personalized baseline
```

---

### Calibration Initialization in app_main()

**Added after CNN initialization:**

```c
/* ================= INITIALIZE CALIBRATION SYSTEM ================= */
ESP_LOGI(TAG, "🎯 Initializing calibration system...");
if (calibration_init() != 0) {
    ESP_LOGW(TAG, "⚠️ Calibration initialization failed - will use local normalization");
} else {
    if (calibration_is_calibrated()) {
        ESP_LOGI(TAG, "✅ Device is calibrated with personalized baseline");
        ESP_LOGI(TAG, "   Press LEFT button to re-calibrate if needed");
    } else {
        ESP_LOGW(TAG, "⚠️ Device NOT calibrated - predictions may be less accurate");
        ESP_LOGI(TAG, "   👉 Press LEFT button when calm to start 10-minute calibration");
    }
}
```

**Result**: 
- Calibration system loads from NVS on boot
- Shows calibration status in console
- Guides user to calibrate if not done

---

## 4. Documentation Created ✅

### T_DISPLAY_S3_BUTTONS.md

Comprehensive guide covering:
- Button pin mapping (GPIO 0 and GPIO 14)
- Hardware details and physical layout
- Firmware implementation details
- Testing procedures
- Troubleshooting tips
- Pin conflicts to avoid

---

## Summary of Changes

### Files Modified

1. **`Shadow/Shadow/Shared/ShadowAppNavBar.swift`**
   - Removed "Shadow BLE" status indicator

2. **`Shadow/Shadow/Features/Dashboard/ShadowDashboardView.swift`**
   - Replaced Debug button with Forget button
   - Added conditional rendering (only when paired)

3. **`shadow-firmware/main/main_realtime.c`**
   - Updated pin definitions (LEFT=0, RIGHT=14)
   - Added two separate button ISR handlers
   - Added calibration control logic
   - Added calibration initialization
   - Updated GPIO configuration
   - Extended event types enum

### Files Created

4. **`T_DISPLAY_S3_BUTTONS.md`**
   - Complete button documentation
   - Pin mapping and usage guide
   - Testing procedures

---

## Testing Checklist

### macOS App Testing

- [x] Navigation bar no longer shows "Shadow BLE"
- [ ] Forget button appears when device paired
- [ ] Forget button works (unpairs device)
- [ ] Forget button disappears when no device paired
- [ ] UI is cleaner and more minimal

### Firmware Testing

- [ ] Flash firmware: `idf.py build flash monitor`
- [ ] Boot logs show calibration status
- [ ] LEFT button starts calibration
- [ ] LEFT button stops calibration
- [ ] Calibration saves to NVS
- [ ] Calibration loads on reboot
- [ ] RIGHT button toggles display (clock ↔ QR)

---

## Expected Console Output

### On Boot (Not Calibrated)

```
I (1234) Shadow: 🎯 Initializing calibration system...
I (1235) Shadow: ⚠️ Device NOT calibrated - predictions may be less accurate
I (1236) Shadow:    👉 Press LEFT button when calm to start 10-minute calibration
```

### On Boot (Already Calibrated)

```
I (1234) Shadow: 🎯 Initializing calibration system...
I (1235) Calibration: ✅ Loaded calibration from NVS
I (1236) Shadow: ✅ Device is calibrated with personalized baseline
I (1237) Shadow:    Press LEFT button to re-calibrate if needed
```

### LEFT Button Press (Start)

```
I (5678) ShadowRealTime: 🔘 Left button pressed - calibration control
I (5679) ShadowRealTime: 🟢 Starting calibration (10 minutes)
I (5680) Calibration: 🎯 Starting calibration session (600 seconds)
```

### LEFT Button Press (Stop)

```
I (65678) ShadowRealTime: 🔘 Left button pressed - calibration control
I (65679) ShadowRealTime: 🔴 Stopping calibration
I (65680) Calibration: ✅ Calibration complete - required samples reached
I (65681) Calibration: ✅ Calibration saved to NVS
I (65682) ShadowRealTime: ✅ Calibration completed successfully!
```

### RIGHT Button Press

```
I (1234) ShadowRealTime: 🔘 Right button pressed - toggling display
```

---

## Benefits

### User Experience

✅ **Cleaner UI**: Removed clutter from navigation bar  
✅ **Useful Actions**: Forget button more practical than debug  
✅ **Easy Calibration**: Just press LEFT button when calm  
✅ **Visual Feedback**: Display toggle with RIGHT button  

### Technical

✅ **Two-Button Support**: Proper dual-button configuration  
✅ **Debounced Inputs**: No false triggers  
✅ **Calibration Integrated**: Personalized baseline system  
✅ **NVS Persistence**: Calibration saved forever  
✅ **Clean Architecture**: Separate ISRs and event handlers  

---

## Next Steps

1. **Build & Flash Firmware**
   ```bash
   cd shadow-firmware
   . $HOME/Dev/esp/esp-idf/export.sh
   idf.py build flash monitor
   ```

2. **Test Both Buttons**
   - Press LEFT → Calibration control
   - Press RIGHT → Display toggle

3. **Test Calibration Flow**
   - Start calibration when calm
   - Wait 10 minutes or stop early
   - Verify NVS save
   - Reboot and check load

4. **Test macOS App**
   - Verify cleaner navigation
   - Test Forget button when paired

---

**Status**: ✅ **All changes implemented and ready for testing**

*Last Updated: October 20, 2025*
