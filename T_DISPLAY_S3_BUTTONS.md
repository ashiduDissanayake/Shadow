# T-Display S3 Button Configuration

## Button Pin Mapping

The LilyGo T-Display S3 has **two physical buttons**:

### Left Button (Boot Button)
- **GPIO**: 0
- **Purpose**: **Calibration Control** 🎯
- **Function**: Start/Stop 10-minute calibration session
- **Pull-up**: Enabled (active-low)
- **Interrupt**: Negative edge (trigger on press)

**Usage:**
```
Press once  → Start calibration (if not calibrating)
Press again → Stop calibration (if currently calibrating)
```

**Expected Behavior:**
```
1. User is calm and relaxed
2. Press LEFT button ONCE
3. Console: "🟢 Starting calibration (2 minutes, auto-completes)"
4. Wait 2 minutes - system collects data automatically
5. Progress updates every 30 seconds (25%, 50%, 75%, 100%)
6. Auto-completes: "✅ Calibration auto-complete - required samples reached"
7. Baseline saved to NVS automatically
8. Predictions resume automatically
```

**Note:** No need to press stop! System handles everything automatically.

---

### Right Button (Reset Button)
- **GPIO**: 14
- **Purpose**: **Display Toggle** 🖥️
- **Function**: Toggle between clock view and QR code
- **Pull-up**: Enabled (active-low)
- **Interrupt**: Negative edge (trigger on press)

**Usage:**
```
Press → Toggle display: Clock ↔ QR Code
```

**Expected Behavior:**
```
Default: Shows clock with device name
Press:   Shows QR code for pairing
Press:   Back to clock
```

---

## Firmware Implementation

### Pin Definitions (`main_realtime.c`)

```c
#define BUTTON_LEFT_PIN     0     // Calibration control
#define BUTTON_RIGHT_PIN    14    // Display toggle
#define BUTTON_DEBOUNCE_MS  200   // Debounce time
```

### Interrupt Handlers

**Left Button (Calibration):**
```c
static void IRAM_ATTR button_left_interrupt_handler(void *arg) {
    // Debounce
    int64_t now = esp_timer_get_time() / 1000;
    if (now - last_button_left_press < BUTTON_DEBOUNCE_MS) {
        return;
    }
    last_button_left_press = now;
    
    // Send event to queue
    sensor_event_t event = {
        .type = SENSOR_EVENT_BUTTON_LEFT,
        .timestamp_us = esp_timer_get_time(),
        .sequence = 0
    };
    xQueueSendFromISR(sensor_event_queue, &event, &xHigherPriorityTaskWoken);
}
```

**Right Button (Display):**
```c
static void IRAM_ATTR button_right_interrupt_handler(void *arg) {
    // Similar to left button but sends SENSOR_EVENT_BUTTON_RIGHT
}
```

### Event Handling

In producer task event loop:

```c
case SENSOR_EVENT_BUTTON_LEFT:
    // Toggle calibration
    if (calibration_get_state() == CAL_STATE_IN_PROGRESS) {
        calibration_stop(false);
        if (calibration_is_calibrated()) {
            ESP_LOGI(TAG, "✅ Calibration completed!");
        }
    } else {
        calibration_start();
        ESP_LOGI(TAG, "🟢 Starting calibration");
    }
    break;

case SENSOR_EVENT_BUTTON_RIGHT:
    // Toggle display mode
    display_toggle_mode(&g_device_info);
    break;
```

### GPIO Configuration

```c
// Left button (GPIO 0)
gpio_config_t btn_left_conf = {
    .pin_bit_mask = (1ULL << BUTTON_LEFT_PIN),
    .mode = GPIO_MODE_INPUT,
    .pull_up_en = GPIO_PULLUP_ENABLE,
    .pull_down_en = GPIO_PULLDOWN_DISABLE,
    .intr_type = GPIO_INTR_NEGEDGE
};
ESP_ERROR_CHECK(gpio_config(&btn_left_conf));
ESP_ERROR_CHECK(gpio_isr_handler_add(BUTTON_LEFT_PIN, button_left_interrupt_handler, NULL));

// Right button (GPIO 14)
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

## Hardware Details

### T-Display S3 Physical Layout

```
┌─────────────────────────────────┐
│                                 │
│  [TFT Display 240x135]          │
│                                 │
│  ┌──────────────────────┐       │
│  │  Shadow-XXXX         │       │
│  │  00:00:00            │       │
│  └──────────────────────┘       │
│                                 │
│  [LEFT]              [RIGHT]    │
│  GPIO 0              GPIO 14    │
│  (Calibration)       (Display)  │
└─────────────────────────────────┘
```

### Button Electrical Characteristics

- **Type**: Tactile momentary push buttons
- **Active State**: Low (buttons connect GPIO to GND)
- **Idle State**: High (internal pull-up resistor)
- **Debounce**: 200ms in firmware
- **Interrupt**: Negative edge (falling edge on press)

---

## Pin Conflicts to Avoid

### Reserved Pins (DO NOT USE)

- **GPIO 43**: I2C SCL (display communication)
- **GPIO 44**: I2C SDA (display communication)
- **GPIO 1-9**: Display data bus and control
- **GPIO 38**: Display backlight
- **GPIO 15**: Power enable (if used)

### Safe GPIO Pins (Available)

If you need additional buttons or sensors:
- GPIO 10, 11, 12, 13 (available)
- GPIO 16, 17, 18 (available)
- GPIO 21 (available, but check schematic)

---

## Testing Buttons

### Test Left Button (Calibration)

```bash
# Flash firmware
cd shadow-firmware
idf.py build flash monitor

# Expected logs on boot:
I (1234) Shadow: ⚠️ Device NOT calibrated
I (1235) Shadow:    👉 Press LEFT button when calm to start calibration

# Press LEFT button:
I (5678) ShadowRealTime: 🔘 Left button pressed - calibration start
I (5679) ShadowRealTime: 🟢 Starting calibration (2 minutes, auto-completes)
I (5680) Calibration: 🎯 Starting calibration session (120 seconds)

# Progress updates every 30 seconds:
I (35680) Calibration: 📊 Calibration progress: 25.0% (120/480 samples, 90 sec remaining)
I (65680) Calibration: 📊 Calibration progress: 50.0% (240/480 samples, 60 sec remaining)
I (95680) Calibration: 📊 Calibration progress: 75.0% (360/480 samples, 30 sec remaining)

# Automatic completion after 2 minutes:
I (125680) Calibration: ✅ Calibration auto-complete - required samples reached
I (125681) Calibration: ✅ Calibration completed successfully
I (125682) Calibration: ✅ Calibration saved to NVS

# Predictions resume automatically:
I (135680) ShadowRealTime: 🔔 CNN Inference #1
I (135681) SignalPreprocessor: Applied PERSONALIZED z-score normalization
```

### Test Right Button (Display)

```bash
# Press RIGHT button repeatedly:
I (1234) ShadowRealTime: 🔘 Right button pressed - toggling display
# Display switches from clock to QR code

I (5678) ShadowRealTime: 🔘 Right button pressed - toggling display
# Display switches back to clock
```

---

## Troubleshooting

### Button Not Responding

**Check:**
1. GPIO ISR service installed only once
2. Debounce time not too short
3. Interrupt type is NEGEDGE
4. Pull-up is enabled

**Debug:**
```c
// Add to ISR handler
ESP_EARLY_LOGI(TAG, "Button interrupt triggered!");
```

### Multiple Triggers (Bounce)

**Solution:**
- Increase `BUTTON_DEBOUNCE_MS` from 200 to 300-500
- Verify debounce logic in ISR

### Wrong Button Behavior

**Verify:**
```c
// Check GPIO numbers match physical buttons
#define BUTTON_LEFT_PIN     0     // Must be GPIO 0
#define BUTTON_RIGHT_PIN    14    // Must be GPIO 14
```

---

## Changes Made

### Files Modified

1. **`shadow-firmware/main/main_realtime.c`**
   - Added `BUTTON_LEFT_PIN` (GPIO 0)
   - Added `BUTTON_RIGHT_PIN` (GPIO 14)
   - Renamed old `BUTTON_PIN` references
   - Added `button_left_interrupt_handler()`
   - Added `button_right_interrupt_handler()`
   - Added `SENSOR_EVENT_BUTTON_LEFT`
   - Added `SENSOR_EVENT_BUTTON_RIGHT`
   - Updated GPIO configuration
   - Added calibration control logic
   - Added calibration initialization in `app_main()`

2. **`shadow-firmware/components/signal_preprocessor/include/calibration.h`**
   - Already exists (created earlier)

3. **`shadow-firmware/components/signal_preprocessor/calibration.c`**
   - Already exists (created earlier)

---

## Summary

✅ **Two buttons properly configured:**
- **LEFT (GPIO 0)**: Calibration control - press to start/stop 10-min baseline collection
- **RIGHT (GPIO 14)**: Display toggle - press to switch between clock and QR code

✅ **Calibration system integrated:**
- Loads from NVS on boot
- User can start/stop via LEFT button
- Saves personalized baseline forever
- Used by signal preprocessor for accurate normalization

✅ **Ready for testing:**
- Flash firmware and test both buttons
- Verify calibration flow
- Check display toggle

---

*Last Updated: October 20, 2025*
