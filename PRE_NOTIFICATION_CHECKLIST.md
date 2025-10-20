# Pre-Notification System Checklist

## ✅ COMPLETED:

### 1. Display Power Management
- ✅ Short press (< 1.5s): Toggle display ON/OFF
- ✅ Long press (≥ 1.5s): Toggle QR code
- ✅ Auto-sleep after 30 seconds
- ✅ Display refresh at 1Hz (clock updates)
- ✅ Activity tracking for auto-sleep reset

### 2. Time Sync on Device Reset
- ✅ Handles seq=0 state=0 (initial boot)
- ✅ Handles large gap (seq reset)
- ✅ Always syncs time on reconnection
- ✅ macOS sends 12-byte time sync payload
- ✅ ESP32 converts boot time to Unix timestamps

## ⏳ TODO BEFORE NOTIFICATIONS:

### 3. Stress State Visualization (macOS App)
**Goal**: Add graph showing stress states over time

**Requirements**:
- [ ] Create `StressHistoryView.swift` SwiftUI view
- [ ] Use Swift Charts framework
- [ ] X-axis: Time (last 24 hours, scrollable)
- [ ] Y-axis: Stress state (0 = Calm, 1 = Stressed)
- [ ] Color coding:
  - Green area: Calm periods
  - Red area: Stressed periods
- [ ] Interactive: Tap to see event details (time, duration, confidence)
- [ ] Filter options: Last hour, 6 hours, 24 hours, week
- [ ] Sync with CoreData stress events

**Data Source**:
```swift
// Fetch stress events from CoreData
let events = repo.fetchStressEvents(
    deviceUUID: deviceUUID,
    startDate: Date().addingTimeInterval(-24*3600),
    endDate: Date()
)
```

**Chart Type**: Area chart with binary states
```
Stress State
    1 ┤  ▄▄▄▄▄          ▄▄▄        ▄▄▄▄
      │                              
    0 ┤▀▀      ▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀    ▀▀▀
      └───────────────────────────────────► Time
       9AM    12PM    3PM    6PM    9PM
```

**Implementation Plan**:
1. Create StressHistoryView with Chart
2. Add to main ContentView as tab or section
3. Implement time range picker
4. Add event detail overlay
5. Style with Shadow theme colors

## 🚀 READY TO BUILD:

### 4. Notification System
**Once graph is complete, proceed with:**
- [ ] Integrate NotificationDecisionEngine
- [ ] Add calendar event monitoring
- [ ] Implement Gemini AI client
- [ ] Test notification timing
- [ ] Test combined notifications

## 📝 NOTES:

### Time Sync Edge Cases Covered:
1. **Normal delta=1**: No connection, local update only ✅
2. **Delta > 1**: Connect and sync time ✅
3. **seq=0 state=0**: Connect and sync time (device reboot) ✅
4. **Large gap**: Reset and sync time ✅

All paths ensure time sync happens on reconnection!

### Display UX User Flow:
```
User Scenario 1: Quick Check
├─ Short press → Wake display
├─ See clock (with real time after sync)
├─ 30s timeout → Auto-sleep
└─ Battery saved ✅

User Scenario 2: Pairing
├─ Long press (hold 2s) → Show QR code
├─ User scans code
├─ Short press → Sleep display manually
└─ Battery saved ✅

User Scenario 3: Continuous Use
├─ Wake display
├─ Use device (calibration, checking time)
├─ Each button press → Reset 30s timer
├─ Natural usage pattern maintained
└─ Auto-sleep when done ✅
```

### Power Consumption Estimates:
- **Display ON**: ~100mA (TFT backlight)
- **Display OFF**: ~10mA (ESP32 + sensors only)
- **Improvement**: 90% reduction when idle
- **Battery life**: ~10x longer (rough estimate)

## TESTING CHECKLIST:

Before building notifications:
- [ ] Flash firmware with new display management
- [ ] Test short press wake/sleep
- [ ] Test long press QR toggle
- [ ] Test auto-sleep (30s timeout)
- [ ] Test activity reset (button resets timer)
- [ ] Test time sync after device reboot
- [ ] Verify clock shows real time after sync
- [ ] Test stress state graph in macOS app
- [ ] Verify graph shows correct timeline
- [ ] Test graph interactivity (tap for details)

## BUILD ORDER:

1. **NOW**: Build and flash firmware ⏳
2. **Test**: Display power management
3. **Build**: Stress history graph (macOS)
4. **Test**: Graph with real stress events
5. **Then**: Notification system 🚀
