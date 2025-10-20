# 🚨 CRITICAL: Z-Score Normalization Bug Fix Guide

## Problem Summary

**Current Implementation**: The signal preprocessor computes z-score normalization using statistics (mean, std) from **only the current 60-second ring buffer window**.

**Why This is BROKEN**:
1. Ring buffer = 240 samples @ 4Hz = 60 seconds of history
2. If user is in same physiological state > 60 seconds:
   - All older samples get overwritten (ring buffer wraps)
   - Mean/std computed from **only current state** samples
   - Normalized values → near 0 (normalizing current state to itself!)
   - Model receives all zeros → **fails to detect stress**

**Example Scenario**:
- User stressed for 70 seconds
- First 60s: Buffer has mix (normal + stressed) → decent normalization
- After 60s: Buffer **only** has stressed samples
- Mean = average stressed value
- Normalized = (stressed - stressed_mean) / stressed_std ≈ 0
- **Model sees zeros instead of high stress values!**

## Training vs. Inference Mismatch

Your model training (`06_model_selection_loso.py`, `07_train_final_holdout.py`):
```python
Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),  # Global stats
    ("mlp", MLPClassifier(...))
])
```

- **Training**: StandardScaler computes mean/std **across entire training dataset**
- **Inference**: ESP32 computes mean/std **from 60-second window only**
- **Result**: Distribution shift → model fails!

## Solution Options

### ✅ Option 1: Use Training Statistics (RECOMMENDED)

Use the **same global mean/std from training data**:

**Steps**:
1. Extract scaler statistics from your trained model:
```python
# In model/07_train_final_holdout.py (after training)
scaler = pipeline.named_steps['scaler']
print("Channel Means:", scaler.mean_)
print("Channel Stds:", scaler.scale_)

# Save to JSON
stats = {
    "acc_mean": float(scaler.mean_[acc_features_idx].mean()),
    "bvp_mean": float(scaler.mean_[bvp_features_idx].mean()),
    "eda_mean": float(scaler.mean_[eda_features_idx].mean()),
    "temp_mean": float(scaler.mean_[temp_features_idx].mean()),
    "acc_std": float(scaler.scale_[acc_features_idx].mean()),
    "bvp_std": float(scaler.scale_[bvp_features_idx].mean()),
    "eda_std": float(scaler.scale_[eda_features_idx].mean()),
    "temp_std": float(scaler.scale_[temp_features_idx].mean()),
}
with open('normalization_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

2. Hardcode in ESP32 firmware:
```c
// shadow-firmware/components/signal_preprocessor/signal_preprocessor.c

// Global normalization statistics from training data (WESAD dataset)
// These values MUST match the StandardScaler fitted during model training
static const float GLOBAL_CHANNEL_MEANS[CNN_INPUT_CHANNELS] = {
    1.234f,  // ACC magnitude mean
    0.856f,  // BVP mean
    2.345f,  // EDA mean
    32.50f   // TEMP mean (Celsius)
};

static const float GLOBAL_CHANNEL_STDS[CNN_INPUT_CHANNELS] = {
    0.523f,  // ACC magnitude std
    0.312f,  // BVP std
    1.234f,  // EDA std
    0.876f   // TEMP std
};

int normalize_signal_zscore_global(float *signal, uint16_t length, 
                                   float global_mean, float global_std) {
    if (!signal || length == 0) {
        return -1;
    }
    
    // Avoid division by zero
    if (global_std < 1e-6f) {
        ESP_LOGW(TAG, "Global std too small (%.6f), using 1.0", global_std);
        global_std = 1.0f;
    }
    
    // Normalize: (x - global_mean) / global_std
    for (uint16_t i = 0; i < length; i++) {
        signal[i] = (signal[i] - global_mean) / global_std;
    }
    
    return 0;
}

// Update preprocess_for_cnn() to use global stats:
int preprocess_for_cnn(realtime_sensor_system_t *sensor_system,
                       cnn_input_tensor_t *output) {
    // ... existing extraction code ...
    
    /* ========== STEP 5: Z-score normalization with GLOBAL stats ========== */
    
    // Normalize ACC magnitude
    ret = normalize_signal_zscore_global(
        output->data[CNN_CHANNEL_ACC], 
        CNN_INPUT_SAMPLES,
        GLOBAL_CHANNEL_MEANS[CNN_CHANNEL_ACC],
        GLOBAL_CHANNEL_STDS[CNN_CHANNEL_ACC]
    );
    
    // Normalize BVP
    ret = normalize_signal_zscore_global(
        output->data[CNN_CHANNEL_BVP], 
        CNN_INPUT_SAMPLES,
        GLOBAL_CHANNEL_MEANS[CNN_CHANNEL_BVP],
        GLOBAL_CHANNEL_STDS[CNN_CHANNEL_BVP]
    );
    
    // Normalize EDA
    ret = normalize_signal_zscore_global(
        output->data[CNN_CHANNEL_EDA], 
        CNN_INPUT_SAMPLES,
        GLOBAL_CHANNEL_MEANS[CNN_CHANNEL_EDA],
        GLOBAL_CHANNEL_STDS[CNN_CHANNEL_EDA]
    );
    
    // Normalize TEMP
    ret = normalize_signal_zscore_global(
        output->data[CNN_CHANNEL_TEMP], 
        CNN_INPUT_SAMPLES,
        GLOBAL_CHANNEL_MEANS[CNN_CHANNEL_TEMP],
        GLOBAL_CHANNEL_STDS[CNN_CHANNEL_TEMP]
    );
    
    ESP_LOGI(TAG, "Applied GLOBAL z-score normalization to all channels");
    
    // ... rest of code ...
}
```

**Pros**:
- ✅ Matches training exactly
- ✅ No memory overhead
- ✅ No calibration needed
- ✅ Consistent across all users
- ✅ Works immediately

**Cons**:
- ❌ Not personalized per user
- ❌ Need to extract stats from training

---

### Option 2: Calibration Period

Collect baseline during first 5-10 minutes:

```c
typedef struct {
    float mean;
    float std;
    uint32_t sample_count;
    float running_sum;
    float running_sum_sq;
    bool calibrated;
} calibration_stats_t;

static calibration_stats_t g_calibration[CNN_INPUT_CHANNELS];

void update_calibration_stats(float *samples, uint16_t length, cnn_channel_t channel) {
    calibration_stats_t *cal = &g_calibration[channel];
    
    for (uint16_t i = 0; i < length; i++) {
        cal->running_sum += samples[i];
        cal->running_sum_sq += samples[i] * samples[i];
        cal->sample_count++;
    }
    
    // After N samples (e.g., 2400 = 10 minutes @ 4Hz), finalize calibration
    if (cal->sample_count >= CALIBRATION_SAMPLES && !cal->calibrated) {
        cal->mean = cal->running_sum / cal->sample_count;
        float variance = (cal->running_sum_sq / cal->sample_count) - (cal->mean * cal->mean);
        cal->std = sqrtf(variance);
        cal->calibrated = true;
        
        // Save to NVS (non-volatile storage)
        save_calibration_to_nvs(channel, cal->mean, cal->std);
    }
}
```

**Pros**:
- ✅ Personalized per user
- ✅ Adapts to individual physiology

**Cons**:
- ❌ Requires 5-10 min calibration
- ❌ Needs NVS storage
- ❌ More complex implementation

---

### Option 3: Longer Rolling Window

Keep 5-10 minute history for computing statistics:

```c
#define STATS_WINDOW_SECONDS 300  // 5 minutes
#define STATS_BUFFER_SIZE (STATS_WINDOW_SECONDS * CNN_SAMPLE_RATE)  // 1200 samples

typedef struct {
    float *history;  // Allocated in PSRAM
    uint16_t write_idx;
    uint16_t filled_count;
} stats_history_t;

static stats_history_t g_stats_history[CNN_INPUT_CHANNELS];
```

**Memory cost**: 1200 samples × 4 bytes × 4 channels = **19.2 KB** in PSRAM

**Pros**:
- ✅ More robust than 60s window
- ✅ Adapts slowly over time

**Cons**:
- ❌ 19.2KB PSRAM overhead
- ❌ Still can drift over long sessions
- ❌ More complex

---

## Recommended Action Plan

### Immediate (Before Next Test):

1. **Extract training statistics**:
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

2. **Update firmware** with global stats (Option 1)

3. **Test** with known stress scenarios

### Future Enhancement:

- Implement **Option 2 (Calibration)** for personalized experience
- Store calibration in ESP32 NVS
- Allow recalibration via button press

---

## Testing the Fix

### Before Fix:
```
Scenario: User stressed for 90 seconds
Expected: High ACC/EDA values → Model detects stress
Actual: After 60s, values normalize to ~0 → Model fails
```

### After Fix (Option 1):
```
Scenario: User stressed for 90 seconds
Expected: High ACC/EDA values → Model detects stress
Actual: Values remain high (global normalization) → Model works! ✅
```

### Validation:
1. Flash fixed firmware
2. Induce stress (exercise, cold pressor, mental arithmetic)
3. Monitor normalized CNN input values - should NOT go to zero
4. Verify model predictions remain consistent over time

---

## File Modifications Required

1. `shadow-firmware/components/signal_preprocessor/signal_preprocessor.c`:
   - Add `GLOBAL_CHANNEL_MEANS` and `GLOBAL_CHANNEL_STDS` constants
   - Add `normalize_signal_zscore_global()` function
   - Update `preprocess_for_cnn()` to use global normalization
   - Remove local mean/std computation

2. `shadow-firmware/components/signal_preprocessor/include/signal_preprocessor.h`:
   - Add `normalize_signal_zscore_global()` declaration

3. `model/07_train_final_holdout.py` (optional):
   - Add code to export scaler statistics to JSON

---

## Questions to Answer

1. **Do you have the trained model saved?**
   - Path: `model/data/output/07_holdout_evaluation/trained_model.joblib`
   - If yes: Extract stats from scaler
   - If no: Retrain and save model + scaler stats

2. **What are the raw sensor value ranges?**
   - ACC magnitude: typically 0.8 - 2.0 g
   - BVP: typically 0 - 255 (raw ADC)
   - EDA: typically 0 - 20 µS
   - TEMP: typically 30 - 36°C

3. **Per-user calibration or global model?**
   - Global (Option 1): Works for all users, less accurate
   - Calibrated (Option 2): Per-user, more accurate, requires setup

---

## Priority

🚨 **CRITICAL** - This bug will cause the model to fail in production!

**Must fix before deploying to users.**
