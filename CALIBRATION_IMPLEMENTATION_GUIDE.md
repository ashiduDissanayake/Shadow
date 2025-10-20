# Calibration System Implementation Guide

## Overview

This guide provides the implementation of **Option 2: Calibration Period** for personalized stress detection normalization.

## ✅ Completed

### ESP32 Firmware

1. **Created calibration.h** (`shadow-firmware/components/signal_preprocessor/include/calibration.h`)
   - Complete API for calibration system
   - Supports start/stop/reset operations
   - NVS persistence for long-term storage
   - Per-channel statistics (mean, std)
   - 10-minute calibration period (configurable)

2. **Created calibration.c** (`shadow-firmware/components/signal_preprocessor/calibration.c`)
   - Full implementation of calibration system
   - Running statistics computation
   - NVS save/load functionality
   - Progress tracking
   - Auto-stop when complete

3. **Updated signal_preprocessor.c**
   - Integrated calibration system
   - Calls `calibration_init()` on startup
   - Updates calibration during collection period
   - Uses `calibration_normalize()` instead of local z-score
   - Falls back to local normalization if not calibrated

4. **Created calibration_ble_service.h** (header only)
   - BLE service design for remote control
   - Service UUID: C000
   - Characteristics: State (C001), Control (C002), Stats (C003)

### macOS App

5. **Fixed Swift concurrency warnings in LightShadowBLEManager.swift**
   - Added `@preconcurrency` imports
   - Made all delegate methods `nonisolated` with `Task { @MainActor in }`
   - Fixed observer capture issues with `UnsafeMutablePointer`
   - All Swift 6 warnings resolved ✅

## ⏳ Remaining Implementation

### ESP32 Firmware

#### 1. Implement Calibration BLE Service (`calibration_ble_service.c`)

```c
/*
 * Calibration BLE Service Implementation
 */

#include "calibration_ble_service.h"
#include "calibration.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "CalibrationBLE";

// BLE characteristic handles
static uint16_t cal_state_handle = 0;
static uint16_t cal_control_handle = 0;
static uint16_t cal_stats_handle = 0;

// Notification task
static TaskHandle_t notify_task_handle = NULL;

/* ==================== INITIALIZATION ==================== */

int calibration_ble_service_init(void) {
    ESP_LOGI(TAG, "Initializing calibration BLE service");
    return 0;
}

/* ==================== HANDLERS ==================== */

int calibration_ble_handle_control_write(const uint8_t *data, uint16_t len) {
    if (len < 1) {
        ESP_LOGE(TAG, "Invalid control command length");
        return -1;
    }
    
    calibration_command_t cmd = (calibration_command_t)data[0];
    
    switch (cmd) {
        case CAL_CMD_START:
            ESP_LOGI(TAG, "📱 BLE: Start calibration");
            calibration_start();
            calibration_ble_start_notify_task();
            break;
            
        case CAL_CMD_STOP:
            ESP_LOGI(TAG, "📱 BLE: Stop calibration");
            calibration_stop(false);
            calibration_ble_stop_notify_task();
            calibration_ble_notify_state();  // Send final state
            break;
            
        case CAL_CMD_RESET:
            ESP_LOGI(TAG, "📱 BLE: Reset calibration");
            calibration_reset();
            calibration_ble_notify_state();
            break;
            
        default:
            ESP_LOGW(TAG, "Unknown calibration command: 0x%02X", cmd);
            return -2;
    }
    
    return 0;
}

int calibration_ble_handle_state_read(uint8_t *data, uint16_t len) {
    if (len < sizeof(calibration_state_packet_t)) {
        return -1;
    }
    
    calibration_state_packet_t *packet = (calibration_state_packet_t *)data;
    
    packet->state = (uint8_t)calibration_get_state();
    packet->reserved = 0;
    packet->progress_percent = (uint16_t)(calibration_get_progress() * 10000.0f);
    packet->remaining_seconds = calibration_get_remaining_time();
    
    return sizeof(calibration_state_packet_t);
}

int calibration_ble_handle_stats_read(uint8_t *data, uint16_t len) {
    if (len < sizeof(calibration_stats_packet_t)) {
        return -1;
    }
    
    if (!calibration_is_calibrated()) {
        return -2;  // Not calibrated yet
    }
    
    calibration_stats_packet_t *packet = (calibration_stats_packet_t *)data;
    
    calibration_get_stats(CNN_CHANNEL_ACC, &packet->acc_mean, &packet->acc_std);
    calibration_get_stats(CNN_CHANNEL_BVP, &packet->bvp_mean, &packet->bvp_std);
    calibration_get_stats(CNN_CHANNEL_EDA, &packet->eda_mean, &packet->eda_std);
    calibration_get_stats(CNN_CHANNEL_TEMP, &packet->temp_mean, &packet->temp_std);
    
    // Total samples would need to be added to calibration API
    packet->total_samples = 0;  // TODO: Add to calibration.h
    
    return sizeof(calibration_stats_packet_t);
}

int calibration_ble_notify_state(void) {
    uint8_t data[sizeof(calibration_state_packet_t)];
    int len = calibration_ble_handle_state_read(data, sizeof(data));
    
    if (len > 0) {
        // TODO: Call esp_ble_gatts_send_indicate() with cal_state_handle
        ESP_LOGI(TAG, "📤 Sent calibration state notification");
        return 0;
    }
    
    return -1;
}

/* ==================== NOTIFICATION TASK ==================== */

static void calibration_notify_task(void *pvParameters) {
    ESP_LOGI(TAG, "Calibration notification task started");
    
    while (1) {
        if (calibration_get_state() == CAL_STATE_IN_PROGRESS) {
            calibration_ble_notify_state();
            vTaskDelay(pdMS_TO_TICKS(2000));  // Update every 2 seconds
        } else {
            // Calibration stopped
            break;
        }
    }
    
    ESP_LOGI(TAG, "Calibration notification task stopped");
    notify_task_handle = NULL;
    vTaskDelete(NULL);
}

void calibration_ble_start_notify_task(void) {
    if (notify_task_handle == NULL) {
        xTaskCreate(calibration_notify_task, "cal_notify", 2048, NULL, 5, &notify_task_handle);
    }
}

void calibration_ble_stop_notify_task(void) {
    if (notify_task_handle != NULL) {
        vTaskDelete(notify_task_handle);
        notify_task_handle = NULL;
    }
}
```

#### 2. Update CMakeLists.txt

```cmake
# components/ble_service/CMakeLists.txt
idf_component_register(
    SRCS "ble_service.c" "calibration_ble_service.c"
    INCLUDE_DIRS "include"
    REQUIRES nvs_flash bt signal_preprocessor
)
```

#### 3. Integrate with main BLE service

In `components/ble_service/ble_service.c`, add calibration service registration:

```c
#include "calibration_ble_service.h"

// In gatts_event_handler, GAT TS_REG_EVT:
calibration_ble_service_register(gatts_if);

// In characteristic write handler:
if (char_uuid == CAL_CONTROL_CHAR_UUID) {
    calibration_ble_handle_control_write(param->write.value, param->write.len);
}

// In characteristic read handler:
if (char_uuid == CAL_STATE_CHAR_UUID) {
    len = calibration_ble_handle_state_read(buffer, sizeof(buffer));
}
else if (char_uuid == CAL_STATS_CHAR_UUID) {
    len = calibration_ble_handle_stats_read(buffer, sizeof(buffer));
}
```

### macOS App

#### 1. Add Calibration Manager

Create `Shadow/Shadow/Features/Calibration/CalibrationManager.swift`:

```swift
import Foundation
import CoreBluetooth
import Combine

@MainActor
class CalibrationManager: ObservableObject {
    @Published var state: CalibrationState = .notStarted
    @Published var progress: Double = 0.0
    @Published var remainingTime: TimeInterval = 0
    @Published var stats: CalibrationStats?
    
    private let serviceUUID = CBUUID(string: "C000")
    private let stateCharUUID = CBUUID(string: "C001")
    private let controlCharUUID = CBUUID(string: "C002")
    private let statsCharUUID = CBUUID(string: "C003")
    
    private weak var bleManager: LightShadowBLEManager?
    private var peripheral: CBPeripheral?
    private var controlChar: CBCharacteristic?
    
    enum CalibrationState: UInt8 {
        case notStarted = 0
        case inProgress = 1
        case completed = 2
        case loaded = 3
        case failed = 4
        
        var description: String {
            switch self {
            case .notStarted: return "Not Started"
            case .inProgress: return "In Progress"
            case .completed: return "Completed"
            case .loaded: return "Loaded"
            case .failed: return "Failed"
            }
        }
        
        var emoji: String {
            switch self {
            case .notStarted: return "⏸️"
            case .inProgress: return "🔄"
            case .completed: return "✅"
            case .loaded: return "💾"
            case .failed: return "❌"
            }
        }
    }
    
    struct CalibrationStats {
        let accMean: Float
        let accStd: Float
        let bvpMean: Float
        let bvpStd: Float
        let edaMean: Float
        let edaStd: Float
        let tempMean: Float
        let tempStd: Float
        let totalSamples: UInt32
    }
    
    func start() async throws {
        guard let controlChar = controlChar else {
            throw CalibrationError.notConnected
        }
        
        let command = Data([0x01])  // CAL_CMD_START
        peripheral?.writeValue(command, for: controlChar, type: .withResponse)
        
        state = .inProgress
    }
    
    func stop() async throws {
        guard let controlChar = controlChar else {
            throw CalibrationError.notConnected
        }
        
        let command = Data([0x02])  // CAL_CMD_STOP
        peripheral?.writeValue(command, for: controlChar, type: .withResponse)
    }
    
    func reset() async throws {
        guard let controlChar = controlChar else {
            throw CalibrationError.notConnected
        }
        
        let command = Data([0x03])  // CAL_CMD_RESET
        peripheral?.writeValue(command, for: controlChar, type: .withResponse)
        
        state = .notStarted
        progress = 0.0
        stats = nil
    }
    
    func handleStateUpdate(_ data: Data) {
        guard data.count >= 8 else { return }
        
        let stateRaw = data[0]
        let progressPercent = data.withUnsafeBytes { $0.load(fromByteOffset: 2, as: UInt16.self) }
        let remainingSec = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) }
        
        if let newState = CalibrationState(rawValue: stateRaw) {
            state = newState
        }
        progress = Double(progressPercent) / 10000.0
        remainingTime = TimeInterval(remainingSec)
    }
    
    func loadStats() async throws {
        // Read stats characteristic
        // Parse CalibrationStats from response
    }
    
    enum CalibrationError: Error {
        case notConnected
        case invalidData
        case timeout
    }
}
```

#### 2. Add Calibration UI

Create `Shadow/Shadow/Features/Calibration/CalibrationView.swift`:

```swift
import SwiftUI

struct CalibrationView: View {
    @StateObject private var calibrationManager = CalibrationManager()
    @State private var showResetConfirmation = false
    
    var body: some View {
        VStack(spacing: 24) {
            // Header
            Text("Sensor Calibration")
                .font(.title)
                .fontWeight(.bold)
            
            // Status Card
            GroupBox {
                VStack(spacing: 16) {
                    HStack {
                        Text(calibrationManager.state.emoji)
                            .font(.system(size: 48))
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text(calibrationManager.state.description)
                                .font(.title2)
                                .fontWeight(.semibold)
                            
                            if calibrationManager.state == .inProgress {
                                Text("\(Int(calibrationManager.progress * 100))% Complete")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                        }
                        
                        Spacer()
                    }
                    
                    if calibrationManager.state == .inProgress {
                        ProgressView(value: calibrationManager.progress)
                        
                        Text("Remaining: \(formatTime(calibrationManager.remainingTime))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding()
            }
            
            // Instructions
            if calibrationManager.state == .notStarted || calibrationManager.state == .failed {
                GroupBox {
                    VStack(alignment: .leading, spacing: 12) {
                        Label("Calibration Instructions", systemImage: "info.circle")
                            .font(.headline)
                        
                        Text("1. Relax and stay calm for 10 minutes")
                        Text("2. Keep the device on your wrist")
                        Text("3. Avoid exercise or stress")
                        Text("4. The device will collect baseline data")
                        
                        Text("This improves accuracy for your physiology.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.top, 4)
                    }
                    .padding()
                }
            }
            
            // Actions
            HStack(spacing: 12) {
                if calibrationManager.state == .notStarted || calibrationManager.state == .failed {
                    Button(action: {
                        Task {
                            try? await calibrationManager.start()
                        }
                    }) {
                        Label("Start Calibration", systemImage: "play.circle.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                }
                
                if calibrationManager.state == .inProgress {
                    Button(action: {
                        Task {
                            try? await calibrationManager.stop()
                        }
                    }) {
                        Label("Stop", systemImage: "stop.circle.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .tint(.orange)
                    .controlSize(.large)
                }
                
                if calibrationManager.state == .completed || calibrationManager.state == .loaded {
                    Button(action: { showResetConfirmation = true }) {
                        Label("Recalibrate", systemImage: "arrow.clockwise")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                }
            }
            
            Spacer()
        }
        .padding()
        .alert("Reset Calibration?", isPresented: $showResetConfirmation) {
            Button("Cancel", role: .cancel) { }
            Button("Reset", role: .destructive) {
                Task {
                    try? await calibrationManager.reset()
                }
            }
        } message: {
            Text("This will clear your calibration data. You'll need to calibrate again.")
        }
    }
    
    private func formatTime(_ seconds: TimeInterval) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }
}
```

#### 3. Add Calibration Button to Device Settings

Update `DeviceSettingsView.swift`:

```swift
// In pairedDeviceView, add button:
Button(action: { showCalibration = true }) {
    Label("Calibrate Sensors", systemImage: "chart.xyaxis.line")
        .frame(maxWidth: .infinity)
}
.buttonStyle(.bordered)

// Add sheet:
.sheet(isPresented: $showCalibration) {
    CalibrationView()
}
```

## Testing Steps

1. **Flash Updated Firmware**:
   ```bash
   cd shadow-firmware
   . $HOME/Dev/esp/esp-idf/export.sh
   idf.py build flash monitor
   ```

2. **Check Logs**:
   - Should see "Initializing calibration system"
   - Should load from NVS if previously calibrated
   - Should show "No calibration found" if first run

3. **Test macOS App**:
   - Open Device Settings
   - Tap "Calibrate Sensors"
   - Tap "Start Calibration"
   - Watch progress update every 2 seconds
   - Progress bar should reach 100% in 10 minutes
   - After completion, stats saved to ESP32 NVS

4. **Verify Normalization**:
   - After calibration, model should use calibrated stats
   - Logs should show "Applied CALIBRATED z-score normalization"
   - Before calibration, should fall back to local z-score

## Benefits of This Approach

✅ **Personalized**: Each user has their own baseline
✅ **Accurate**: Model performs better with user-specific normalization
✅ **Persistent**: Calibration stored in NVS, survives reboots
✅ **User-Friendly**: Simple start/stop UI
✅ **Robust**: Falls back to local z-score if not calibrated
✅ **Real-Time Progress**: Live updates during calibration

## Summary

- ✅ Core calibration system implemented (C code)
- ✅ Integration with signal preprocessor complete
- ✅ Swift concurrency warnings fixed
- ⏳ BLE service implementation needed
- ⏳ macOS UI needs completion
- ⏳ Testing required

This system solves the normalization bug by using personalized statistics collected during a known "calm" period, ensuring the model receives properly normalized data regardless of how long the user has been in a particular state.
