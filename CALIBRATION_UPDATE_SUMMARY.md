# Calibration System Update Summary

## Date: October 20, 2025

## Changes Made

### 1. ⏱️ Reduced Calibration Duration

**Problem:** 10 minutes was too long - users might forget whether they started or stopped calibration.

**Solution:** Changed from 10 minutes to **2 minutes**

### 2. 🎯 Fully Automatic Calibration (One-Button)

**Problem:** Users had difficulty knowing when to press the stop button.

**Solution:** **Press LEFT button ONCE to start** - system auto-completes after 2 minutes!

**Behavior:**
- ✅ **No manual stop needed** - fully automatic
- ✅ **Progress updates** every 30 seconds (25%, 50%, 75%, 100%)
- ✅ **Auto-saves** to NVS when complete
- ✅ **Auto-resumes predictions** after calibration

### 3. ⏸️ Pause CNN Predictions During Calibration

**Problem:** Running stress predictions during calibration could interfere with baseline collection.

**Solution:** Skip CNN inference when calibration is in progress.

**Files Modified:**
- `shadow-firmware/main/main_realtime.c`

**Changes:**
```c
// Added check before CNN inference:
if (calibration_get_state() == CAL_STATE_IN_PROGRESS) {
    ESP_LOGI(TAG, "⏸️ Skipping CNN inference - calibration in progress");
    uint32_t min_batches = realtime_get_min_batch_count();
    realtime_mark_batch_processed(min_batches);
    continue;
}
```

**Behavior:**
- ✅ **Before calibration**: Normal CNN predictions every ~10 seconds
- ⏸️ **During calibration**: CNN predictions paused, only sensor data collection
- ✅ **After calibration**: Resume normal CNN predictions with personalized baseline

---

### 3. 📝 Updated User Messages

**Files Modified:**
- `shadow-firmware/main/main_realtime.c`
- `T_DISPLAY_S3_BUTTONS.md`

**Changes:**
- Boot message: "Press LEFT button when calm to start **2-minute** calibration"
- Button press: "Starting calibration (**2 minutes**)"
- Documentation updated with new timing

---

## How It Works Now

### Calibration Flow (2 Minutes - Fully Automatic)

```
1. User presses LEFT button ONCE
   ↓
2. Console: "🟢 Starting calibration (2 minutes, auto-completes)"
   ↓
3. System pauses CNN predictions automatically
   ↓
4. Collects sensor data for 2 minutes with progress updates:
   - 30 sec: "📊 25.0% complete (120/480 samples, 90 sec remaining)"
   - 60 sec: "📊 50.0% complete (240/480 samples, 60 sec remaining)"
   - 90 sec: "📊 75.0% complete (360/480 samples, 30 sec remaining)"
   ↓
5. AUTO-COMPLETES after 2 minutes (480 samples):
   - Console: "✅ Calibration auto-complete - required samples reached"
   - Console: "✅ Calibration completed successfully"
   - Console: "✅ Calibration saved to NVS"
   ↓
6. Predictions AUTOMATICALLY resume with personalized baseline
   - Console: "🔔 CNN Inference #1"
   - Console: "Applied PERSONALIZED z-score normalization"
```

**User Action Required:** Press LEFT button once, then relax for 2 minutes. That's it!

**What if I press LEFT again during calibration?**
- Console shows: "⏳ Calibration already in progress (XX% complete)"
- Console shows: "Please wait - will auto-complete after collecting enough samples"
- Calibration continues normally (no interruption)

---

## Benefits

### User Experience
- ✅ **One-button operation**: Press once and forget - fully automatic!
- ✅ **Shorter wait time**: 2 minutes vs 10 minutes
- ✅ **Progress feedback**: Updates every 30 seconds so you know what's happening
- ✅ **No confusion**: Can't press stop at wrong time - system handles it
- ✅ **Foolproof**: Pressing button during calibration just shows status

### Technical Accuracy
- ✅ **No interference**: CNN predictions paused during calibration
- ✅ **Clean baseline**: Only sensor data, no ML processing
- ✅ **Persistent storage**: Calibration saved to NVS automatically
- ✅ **Graceful fallback**: Uses local z-score if not calibrated

### System Efficiency
- ✅ **Less memory usage**: Fewer samples to store (480 vs 2400)
- ✅ **Faster computation**: Quicker statistical calculations
- ✅ **Better UX**: Users more likely to complete calibration
- ✅ **Auto-resume**: Predictions start automatically after calibration

---

## Expected Console Output

### During Calibration (2 minutes - Fully Automatic)

```
# User presses LEFT button:
I (12345) Shadow: 🔘 Left button pressed - calibration start
I (12346) Shadow: 🟢 Starting calibration (2 minutes, auto-completes)
I (12347) Shadow:    ℹ️ Stay calm and still - will finish automatically
I (12348) Calibration: 🎯 Starting calibration session (120 seconds, 480 samples required)

# CNN predictions automatically pause:
I (13000) ShadowRealTime: ⏸️ Skipping CNN inference - calibration in progress
I (23000) ShadowRealTime: ⏸️ Skipping CNN inference - calibration in progress

# Progress updates every 30 seconds:
I (42348) Calibration: 📊 Calibration progress: 25.0% (120/480 samples, 90 sec remaining)
I (72348) Calibration: 📊 Calibration progress: 50.0% (240/480 samples, 60 sec remaining)
I (102348) Calibration: 📊 Calibration progress: 75.0% (360/480 samples, 30 sec remaining)

# Automatic completion after 2 minutes (no user action needed):
I (132348) Calibration: ✅ Calibration auto-complete - required samples reached
I (132349) Calibration: ⏹️ Stopping calibration after 120000 ms (480 samples)
I (132350) Calibration: Channel 0: mean=0.123456, std=0.234567 (480 samples)
I (132351) Calibration: Channel 1: mean=0.345678, std=0.456789 (480 samples)
I (132352) Calibration: Channel 2: mean=0.567890, std=0.678901 (480 samples)
I (132353) Calibration: Channel 3: mean=0.789012, std=0.890123 (480 samples)
I (132354) Calibration: ✅ Calibration completed successfully
I (132355) Calibration: ✅ Calibration saved to NVS

# Predictions automatically resume with personalized baseline:
I (143000) ShadowRealTime: 🔔 CNN Inference #1
I (143001) SignalPreprocessor: Applied PERSONALIZED z-score normalization to all channels
I (143200) ShadowRealTime: 🎯 CNN Inference Result:
I (143201) ShadowRealTime:    Stress Probability: 15.2%
I (143202) ShadowRealTime:    Class: NORMAL
```

### If User Presses Button During Calibration

```
# User accidentally presses LEFT again while calibrating:
I (52348) Shadow: 🔘 Left button pressed - calibration start
I (52349) Shadow: ⏳ Calibration already in progress (43.8% complete)
I (52350) Shadow:    Please wait - will auto-complete after collecting enough samples

# Calibration continues normally (no interruption):
I (72348) Calibration: 📊 Calibration progress: 50.0% (240/480 samples, 60 sec remaining)
# ... continues until 100%
```

### After Reboot (Calibration Persists)

```
I (1234) Shadow: 🎯 Initializing calibration system...
I (1235) Calibration: 📂 Loading calibration from NVS...
I (1236) Calibration: ✅ Loaded valid calibration (480 samples)
I (1237) Shadow: ✅ Device is calibrated with personalized baseline
I (1238) Shadow:    Press LEFT button to re-calibrate if needed
```

---

## Testing Checklist

### Before Testing
- [ ] Flash new firmware to device
- [ ] Connect serial monitor
- [ ] Device should boot and show: "⚠️ Device NOT calibrated"

### Test 1: Automatic 2-Minute Calibration (Main Test)
1. [ ] Press LEFT button ONCE
2. [ ] Console shows: "🟢 Starting calibration (2 minutes, auto-completes)"
3. [ ] **Stay calm and still for 2 minutes** (do NOT press button again)
4. [ ] Verify progress updates every 30 seconds (25%, 50%, 75%)
5. [ ] Console shows CNN predictions are paused (⏸️ messages)
6. [ ] After exactly 2 minutes: "✅ Calibration auto-complete - required samples reached"
7. [ ] Verify automatic save: "✅ Calibration saved to NVS"
8. [ ] Verify predictions resume automatically with "PERSONALIZED z-score"

### Test 2: Button Press During Calibration
1. [ ] Press LEFT button to start calibration
2. [ ] After ~30 seconds, press LEFT button again
3. [ ] Should show: "⏳ Calibration already in progress (XX% complete)"
4. [ ] Should show: "Please wait - will auto-complete..."
5. [ ] Calibration should continue normally (not interrupted)
6. [ ] Should still complete automatically after 2 minutes

### Test 3: Re-Calibration
1. [ ] Complete calibration successfully (Test 1)
2. [ ] Verify device shows: "✅ Device is calibrated"
3. [ ] Press LEFT button again
4. [ ] Should warn: "⚠️ Device already calibrated"
5. [ ] Should ask: "Press again to re-calibrate (will overwrite...)"
6. [ ] New calibration should start and complete automatically

### Test 4: Calibration Persistence
1. [ ] Complete calibration successfully
2. [ ] Reboot device (press RESET or power cycle)
3. [ ] Console should show: "✅ Device is calibrated with personalized baseline"
4. [ ] Predictions should use personalized baseline without re-calibration
5. [ ] No need to calibrate again!

---

## Technical Details

### Sample Collection
- **Sample rate**: 4 Hz (4 samples per second)
- **Duration**: 120 seconds
- **Total samples**: 480 samples minimum
- **Channels**: ACC, BVP, EDA, TEMP (all 4 channels)
- **Storage**: NVS partition (persistent)

### Statistical Method
- **Algorithm**: Z-score normalization
- **Per-channel**: Separate mean/std for each sensor
- **Formula**: `normalized = (value - mean) / std`
- **Fallback**: Local z-score if not calibrated

### Memory Usage
- **Calibration data**: ~100 bytes per channel
- **Total**: ~400 bytes in NVS
- **Runtime**: Minimal (no buffering during calibration)

---

## Troubleshooting

### "Calibration failed or incomplete"
- **Cause**: Not enough samples collected (<240)
- **Solution**: Wait longer (at least 1 minute) before stopping

### "No calibration for channel X, using local z-score"
- **Cause**: Device not calibrated yet
- **Solution**: Run calibration process (press LEFT button)

### Predictions still running during calibration
- **Cause**: Old firmware version
- **Solution**: Reflash with updated firmware

### Calibration not saved after reboot
- **Cause**: NVS not initialized or corrupted
- **Solution**: Check NVS partition, may need to erase flash

---

## Files Modified

1. **`shadow-firmware/components/signal_preprocessor/include/calibration.h`**
   - Changed calibration duration constants

2. **`shadow-firmware/main/main_realtime.c`**
   - Added prediction pause during calibration
   - Updated button press messages
   - Updated boot messages

3. **`T_DISPLAY_S3_BUTTONS.md`**
   - Updated documentation with new 2-minute duration

---

## Next Steps

1. **Flash firmware** to device
2. **Test calibration** with 2-minute duration
3. **Verify predictions pause** during calibration
4. **Test persistence** by rebooting after calibration
5. **Monitor accuracy** improvement after personalized calibration

---

## Notes

- ✅ Build successful (no compilation errors)
- ⏳ Ready to flash to device
- 📝 Documentation updated
- 🔧 Backward compatible (can still load old 10-minute calibrations)
