# 🐛 Calibration Bug Fix: Sample Counting Error

**Date**: 2025-10-20  
**Issue**: Calibration failed with only 2/4 channels valid  
**Severity**: CRITICAL - Prevented calibration from completing successfully

---

## 📋 Problem Summary

Calibration was **completing prematurely after processing only 2 channels** (ACC and BVP), leaving EDA and TEMP with 0 samples, causing the calibration to fail.

### Error Logs from Terminal:
```
I (134937) ShadowRealTime: 📊 Calibration in progress - preprocessing for calibration data
I (134957) Calibration: 📊 Calibration progress: 50.0% (240/480 samples, 116 sec remaining)
I (134957) Calibration: 📊 Calibration progress: 100.0% (480/480 samples, 116 sec remaining)
I (134967) Calibration: ✅ Calibration auto-complete - required samples reached
I (134987) Calibration: Channel 0: mean=0.941419, std=0.006363 (240 samples) ✅
I (134987) Calibration: Channel 1: mean=14206.454102, std=1622.734253 (240 samples) ✅
W (134997) Calibration: Channel 2: Insufficient samples (0) ❌
W (135007) Calibration: Channel 3: Insufficient samples (0) ❌
E (135007) Calibration: ❌ Calibration failed: only 2/4 channels valid
```

---

## 🔍 Root Cause Analysis

### The Bug:
In `calibration.c`, the `calibration_update()` function was called **4 times per preprocessing batch** (once for each channel: ACC, BVP, EDA, TEMP).

Each call incremented the **global `total_samples` counter**:

```c
// OLD CODE (BROKEN):
g_calibration.total_samples += length;  // Called 4x per batch!
```

### What Happened:
1. First batch extracted 240 samples for each sensor
2. Preprocessing calls `calibration_update()` for each channel:
   - **Channel 0 (ACC)**: `total_samples = 0 + 240 = 240`
   - **Channel 1 (BVP)**: `total_samples = 240 + 240 = 480` ✅ **Threshold reached!**
   - Auto-complete triggered immediately
   - **Channel 2 (EDA)**: Never processed (calibration already stopped)
   - **Channel 3 (TEMP)**: Never processed (calibration already stopped)

### The Math Error:
```
Required: 480 samples per channel
Received: 240 samples × 4 channels = 960 total count (incorrect accounting)
But the 960 was counted as: 240 (ACC) + 480 (BVP) = stopped at 480!

Actual samples per channel:
- ACC: 240 ✅
- BVP: 240 ✅  
- EDA: 0 ❌
- TEMP: 0 ❌
```

---

## ✅ The Fix

Changed the counting logic to only increment `total_samples` **once per batch** (using ACC as the reference channel):

```c
// NEW CODE (FIXED):
// Only increment total_samples for channel 0 (ACC) to avoid counting 4x
// All channels receive the same number of samples per batch
if (channel == CNN_CHANNEL_ACC) {
    g_calibration.total_samples += length;
    
    // Log progress every 120 samples (30 seconds @ 4Hz)
    if (g_calibration.total_samples % 120 == 0) {
        float progress = calibration_get_progress();
        uint32_t remaining_sec = calibration_get_remaining_time();
        ESP_LOGI(TAG, "📊 Calibration progress: %.1f%% (%lu/%d samples, %lu sec remaining)",
                 progress * 100.0f, g_calibration.total_samples, 
                 CALIBRATION_REQUIRED_SAMPLES, remaining_sec);
    }
    
    // Auto-stop if reached required samples
    if (g_calibration.total_samples >= CALIBRATION_REQUIRED_SAMPLES) {
        ESP_LOGI(TAG, "✅ Calibration auto-complete - required samples reached");
        calibration_stop(false);
    }
}
```

### Why This Works:
- All 4 channels are processed in the same preprocessing batch
- They all receive the same 240 samples
- We only need to count once (ACC channel) as the reference
- Progress tracking and auto-complete only run once per batch

---

## 📊 Expected Behavior After Fix

### Timeline:
```
0s:    User presses LEFT button
       🟢 Starting calibration (2 minutes, auto-completes)

~10s:  First batch ready (240 samples)
       📊 Processing all 4 channels:
          - ACC: 240 samples (total_samples = 240)
          - BVP: 240 samples (no increment)
          - EDA: 240 samples (no increment)
          - TEMP: 240 samples (no increment)
       📊 Calibration progress: 50.0% (240/480 samples)

~20s:  Second batch ready (240 samples)
       📊 Processing all 4 channels:
          - ACC: 240 samples (total_samples = 480) ✅
          - BVP: 240 samples (no increment)
          - EDA: 240 samples (no increment)
          - TEMP: 240 samples (no increment)
       📊 Calibration progress: 100.0% (480/480 samples)
       ✅ Calibration auto-complete - required samples reached
       
       ✅ Channel 0 (ACC): mean=X.XX, std=Y.YY (480 samples)
       ✅ Channel 1 (BVP): mean=X.XX, std=Y.YY (480 samples)
       ✅ Channel 2 (EDA): mean=X.XX, std=Y.YY (480 samples)
       ✅ Channel 3 (TEMP): mean=X.XX, std=Y.YY (480 samples)
       ✅ Calibration completed successfully
       ✅ Calibration saved to NVS

~20s:  Predictions resume with personalized baseline
```

---

## 🧪 Testing Checklist

- [ ] Flash updated firmware
- [ ] Press LEFT button to start calibration
- [ ] Verify progress shows 50% at ~10 seconds
- [ ] Verify progress shows 100% at ~20 seconds
- [ ] Verify **all 4 channels** show 480 samples:
  - `Channel 0: ... (480 samples)`
  - `Channel 1: ... (480 samples)`
  - `Channel 2: ... (480 samples)`
  - `Channel 3: ... (480 samples)`
- [ ] Verify "✅ Calibration completed successfully" (not "failed")
- [ ] Verify "✅ Calibration saved to NVS"
- [ ] Verify predictions use "PERSONALIZED" normalization (not "LOCAL")
- [ ] Reboot and verify calibration persists

---

## 📝 Files Modified

1. **`components/signal_preprocessor/calibration.c`** (lines 196-225):
   - Changed `total_samples` increment to only occur for ACC channel
   - Moved progress logging and auto-complete check inside the channel conditional
   - Added comment explaining the fix

---

## 💡 Lessons Learned

1. **Multi-channel accounting**: When processing multiple channels, be careful about shared counters
2. **Batch processing**: All channels in a batch should be treated as a unit
3. **Defensive logging**: The progress logs showed `50%` then `100%` **in the same millisecond**, which was a red flag
4. **Testing importance**: Without real device testing, this bug wasn't caught in code review

---

## 🔧 Build Information

**Build Status**: ✅ SUCCESS  
**Binary Size**: 1,128,512 bytes (1.1 MB)  
**Free Space**: 837,568 bytes (43%)  
**Warnings**: None (unused function warnings are safe)

---

## 🎯 Next Steps

1. Flash firmware to device: `idf.py flash monitor`
2. Test calibration start → completion → persistence
3. Verify all 4 channels calibrated successfully
4. Confirm predictions use personalized baseline
5. Test across device reboot
