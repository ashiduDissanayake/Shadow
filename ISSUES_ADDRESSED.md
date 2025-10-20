# Issues Addressed - Shadow Project

## Issue #1: macOS App UI Polish ✅ COMPLETED

### Problem
- Too many unnecessary UI elements in Dashboard and Settings
- No way to "forget" paired device
- Cluttered interface with debug tools, notifications section, etc.
- Wanted only: Calendar, Profile header, Device pairing section

### Solution
**Files Modified:**
1. `Shadow/Shadow/Features/Settings/DeviceSettingsView.swift`
2. `Shadow/Shadow/Features/Dashboard/ShadowDashboardView.swift`

**Changes Made:**

#### DeviceSettingsView.swift:
- ✅ Removed notifications section (moved to system settings)
- ✅ Simplified to show only device pairing
- ✅ Added **"Forget Device"** button with confirmation alert
- ✅ Created clean paired/unpaired views
- ✅ Paired view shows:
  - Device name with icon
  - Sync status (Syncing/Idle)
  - Last sync time
  - Events received count
  - Current stress state (colored)
  - Start/Stop sync button
  - **Forget button** (red, with trash icon)
- ✅ Unpaired view shows:
  - Large QR icon
  - Clear instructions
  - "Scan QR Code" button

#### ShadowDashboardView.swift:
- ✅ Removed "Debug Tools" section (still accessible via Debug button)
- ✅ Removed "Sequence Info" row
- ✅ Removed CoreData debug sheet
- ✅ Removed pairing alerts
- ✅ Simplified pairing section:
  - Shows "Paired" or "Unpaired" badge
  - Device name if paired
  - "Pair Device" button if not paired
- ✅ Cleaned up button labels: "Start" instead of "Start Sync", "Stop" instead of "Stop Sync"
- ✅ Added compact "Debug" button for BLE log
- ✅ Shows up to 5 recent events (was 3)

**Result:**
- Clean, minimal UI
- Focus on essential info: Device pairing, sync status, recent events
- Profile header already exists (Welcome back message)
- Calendar integration ready (NotificationManager for event reminders)
- Easy device management with Forget functionality

---

## Issue #2: Z-Score Normalization Bug 🚨 CRITICAL - IN PROGRESS

### Problem Discovery
**User's Insight:** "We normalize the values when we do the input to the model from the signal preprocessor but we do it from the data we have in the 60 second window is that okay?? so if we in the same situation for long time then our value will easily 0 as our past are forgotten from the device as this is a ring buffer will this be a problem?"

**Answer:** YES, this is a **HUGE PROBLEM!** 🚨

### Technical Analysis

**Current Broken Implementation:**
```c
// signal_preprocessor.c - normalize_signal_zscore()
float mean = compute_mean(60_second_window);  // ❌ Local mean
float std = compute_std(60_second_window);    // ❌ Local std
normalized = (value - mean) / std;            // ❌ Self-normalization
```

**Why This Fails:**

1. **Ring Buffer Limitations:**
   - Buffer size: 240 samples @ 4Hz = 60 seconds
   - After 60s in same state, buffer contains ONLY that state's samples
   - Old "normal" samples are completely overwritten

2. **Catastrophic Scenario:**
   - User stressed for 70+ seconds
   - First 60s: Buffer has mix → decent normalization works
   - After 60s: Buffer has ONLY stressed samples
   - Mean of buffer = average of stressed values
   - **Normalized = (stressed - stressed_mean) / stressed_std ≈ 0**
   - Model receives zeros instead of high values!
   - **Model thinks everything is "normal" → FAILS to detect stress!**

3. **Training vs. Inference Mismatch:**
   - **Training:** StandardScaler uses global statistics (entire WESAD dataset)
   - **Inference:** ESP32 uses local statistics (60-second window)
   - **Result:** Distribution shift → model failure

### Visual Example:
```
Time:        0s     60s    70s    120s
State:      [Normal] [Stress detected] [Still stressed]
Buffer:     [Normal] [Mix] [Only Stress] [Only Stress]
Mean:       [Low]   [Med] [HIGH]        [HIGH]
Normalized: [~0]    [~0]  [~0!!]        [~0!!]
Model:      [OK]    [OK]  [FAILS!]      [FAILS!]
                           ^^^^^^^^^^^^^^^^^^^^
                           Model sees zeros!
```

### Solution Created: NORMALIZATION_FIX_GUIDE.md

**Document Contents:**
- ✅ Detailed problem explanation
- ✅ Visual examples and scenarios
- ✅ Three solution options with pros/cons:

**Option 1: Use Training Statistics (RECOMMENDED)**
- Use global mean/std from WESAD training data
- Hardcode in firmware
- No memory overhead
- Matches training exactly
- Status: **Ready to implement - need to extract stats from trained model**

**Option 2: Calibration Period**
- Collect baseline during first 5-10 minutes
- Store in NVS (non-volatile storage)
- Personalized per user
- More complex but better accuracy

**Option 3: Longer Rolling Window**
- Keep 5-10 minute history (1200 samples)
- ~19KB PSRAM overhead
- More robust but still can drift

### Action Items

**Immediate (Before Next Test):**
1. ⏳ Extract training statistics from model:
   ```bash
   cd /Users/ashidudissanayake/Dev/Shadow/model
   python -c "
   import joblib
   pipeline = joblib.load('data/output/07_holdout_evaluation/trained_model.joblib')
   scaler = pipeline.named_steps['scaler']
   print('Means:', scaler.mean_)
   print('Stds:', scaler.scale_)
   "
   ```

2. ⏳ Update `signal_preprocessor.c`:
   - Add `GLOBAL_CHANNEL_MEANS` array
   - Add `GLOBAL_CHANNEL_STDS` array
   - Create `normalize_signal_zscore_global()` function
   - Update `preprocess_for_cnn()` to use global stats

3. ⏳ Rebuild and flash firmware

4. ⏳ Test with prolonged stress scenarios (>60s)

**Future Enhancement:**
- Implement Option 2 (Calibration) for personalized experience
- Store user-specific baseline in NVS
- Add recalibration button

### Impact Assessment

**Without Fix:**
- ❌ Model fails after 60s in same state
- ❌ Cannot detect prolonged stress
- ❌ False negatives for long stress episodes
- ❌ System appears to "forget" stress after 1 minute
- ❌ **UNUSABLE IN PRODUCTION**

**With Fix:**
- ✅ Consistent detection regardless of duration
- ✅ Proper normalization matching training
- ✅ Reliable stress detection for any duration
- ✅ Production-ready

---

## Files Modified

### macOS App (Shadow/)
1. `Shadow/Features/Settings/DeviceSettingsView.swift` - Cleaned and polished
2. `Shadow/Features/Dashboard/ShadowDashboardView.swift` - Removed debug clutter

### Documentation
1. `NORMALIZATION_FIX_GUIDE.md` - Comprehensive fix guide with 3 solutions
2. `ISSUES_ADDRESSED.md` - This document

### ESP32 Firmware (shadow-firmware/)
*To be modified after extracting training stats:*
1. `components/signal_preprocessor/signal_preprocessor.c` - Global normalization
2. `components/signal_preprocessor/include/signal_preprocessor.h` - Updated API

---

## Next Steps

1. **Extract Model Statistics** (Priority: CRITICAL)
   - Run the Python script to get scaler parameters
   - Save to JSON for reference

2. **Update Firmware** (Priority: CRITICAL)
   - Implement global normalization
   - Test with prolonged scenarios

3. **Test UI Changes** (Priority: HIGH)
   - Verify Forget device works
   - Check clean layout
   - Ensure all functionality intact

4. **End-to-End Testing** (Priority: HIGH)
   - Flash fixed firmware
   - Test pairing
   - Test stress detection >60s duration
   - Verify BLE sync
   - Test notifications

---

## Questions for User

1. **Do you have the trained model saved?**
   - Path: `model/data/output/07_holdout_evaluation/trained_model.joblib`
   - If not, need to retrain and save

2. **What are your actual sensor value ranges?**
   - ACC: ? g
   - BVP: ? (raw ADC)
   - EDA: ? µS
   - TEMP: ? °C
   - Needed to validate normalization

3. **Preference for future:**
   - Global model (works for all users, less accurate)
   - OR Calibration (per-user, more accurate, requires setup)

---

## Summary

### Issue #1: UI Polish ✅
- **Status:** COMPLETED
- **Impact:** Improved UX, cleaner interface, added forget functionality
- **Files:** 2 Swift files modified
- **Ready for:** User testing

### Issue #2: Normalization Bug 🚨
- **Status:** IN PROGRESS (solution ready, need stats extraction)
- **Impact:** CRITICAL - Model fails without fix
- **Files:** Guide created, firmware changes pending
- **Blocking:** Need trained model statistics
- **Ready for:** Implementation once stats available

---

**Bottom Line:**
1. ✅ UI is polished and ready
2. 🚨 Normalization bug is CRITICAL and must be fixed before production
3. 📖 Comprehensive fix guide created with 3 solutions
4. ⏳ Waiting on: Extract statistics from trained model

**Recommendation:** Fix normalization bug ASAP (Option 1) before extensive testing, as current implementation will fail in real-world usage.
