# Shadow Firmware Migration Plan
## From Feature-Based MLP to Raw Signal CNN Model

**Date:** October 15, 2025  
**Author:** Migration Analysis  
**Goal:** Replace feature extraction + MLP with CNN model operating on raw resampled sensor signals

---

## 📊 Model Analysis Results

### Current PyTorch Model Architecture
```
Input: (batch, 4 channels, 240 samples)

Layer Structure:
├── shared_conv (Convolutional Layers)
│   ├── Conv1D:  [4, 64] kernel=10
│   ├── BatchNorm1D: 64
│   ├── ReLU + Dropout
│   ├── Conv1D:  [64, 128] kernel=10
│   ├── BatchNorm1D: 128
│   └── ReLU + Dropout
│
├── shared_fc (Fully Connected)
│   ├── Flatten
│   ├── Linear: [128, 128]
│   └── ReLU + Dropout
│
└── universal_private (Output Head)
    ├── Linear: [128, 64]
    ├── ReLU
    ├── Dropout
    └── Linear: [64, 1]
    └── Sigmoid

Output: (batch, 1) - Stress probability [0.0, 1.0]
```

### Preprocessing Pipeline

**Input:** 60 seconds of raw sensor data  
**Output:** (4, 240) normalized tensor

```
Step 1: Collect Raw Data
├── ACC (X,Y,Z): 1920 samples @ 32Hz
├── BVP:         3840 samples @ 64Hz
├── EDA:         240 samples @ 4Hz
└── TEMP:        240 samples @ 4Hz

Step 2: Compute ACC Magnitude
├── acc_mag = sqrt(acc_x² + acc_y² + acc_z²)
└── Result: 1920 samples

Step 3: Resample to 4Hz (Linear Interpolation)
├── ACC:  1920 → 240 samples
├── BVP:  3840 → 240 samples
├── EDA:  240  → 240 samples (no change)
└── TEMP: 240  → 240 samples (no change)

Step 4: Z-Score Normalization (Per Channel)
├── For each channel:
│   ├── mean = average(signal)
│   ├── std = standard_deviation(signal)
│   └── normalized = (signal - mean) / std
└── Result: 4 channels × 240 samples

Step 5: Stack Channels
└── Tensor: [ACC, BVP, EDA, TEMP] → (4, 240)
```

---

## 🔄 Architecture Changes

### Current Architecture (OLD)
```
┌─────────────────────────────────────────────────────────────┐
│ PRODUCER (Core 0)                                           │
├─────────────────────────────────────────────────────────────┤
│ Sensors → Ring Buffers (60s windows)                        │
│ - BVP: 3840 samples @ 64Hz                                  │
│ - ACC: 1920 samples @ 32Hz (X,Y,Z)                          │
│ - EDA: 240 samples @ 4Hz                                    │
│ - TEMP: 240 samples @ 4Hz                                   │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ CONSUMER (Core 1)                                           │
├─────────────────────────────────────────────────────────────┤
│ Feature Extraction (30 features)                            │
│ ├── BVP: mean, std, min, max, median, range, IQR, energy   │
│ ├── ACC: 5 features × 3 axes                                │
│ ├── EDA: mean, std, min, max                                │
│ └── TEMP: mean, std, range                                  │
│                                                              │
│ MLP Inference (30 → 64 → 32 → 1)                            │
│ ├── Standardization                                         │
│ ├── Dense layers with ReLU                                  │
│ └── Sigmoid output                                          │
│                                                              │
│ Stress FSM (Stability Filter)                               │
│ ├── Requires 3 consecutive confirmations                    │
│ ├── Hysteresis: 4 confirmations to return to calm           │
│ └── Only broadcasts stable states                           │
│                                                              │
│ BLE Service                                                  │
│ └── Transmits: Discrete states (CALM/STRESS)                │
└─────────────────────────────────────────────────────────────┘
```

### New Architecture (NEW)
```
┌─────────────────────────────────────────────────────────────┐
│ PRODUCER (Core 0) - UNCHANGED                               │
├─────────────────────────────────────────────────────────────┤
│ Sensors → Ring Buffers (60s windows)                        │
│ - BVP: 3840 samples @ 64Hz                                  │
│ - ACC: 1920 samples @ 32Hz (X,Y,Z)                          │
│ - EDA: 240 samples @ 4Hz                                    │
│ - TEMP: 240 samples @ 4Hz                                   │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ CONSUMER (Core 1) - MODIFIED                                │
├─────────────────────────────────────────────────────────────┤
│ Signal Preprocessing (NEW)                                  │
│ ├── Compute ACC magnitude from 3 axes                       │
│ ├── Resample all signals to 4Hz (240 samples)               │
│ ├── Z-score normalization per channel                       │
│ └── Stack into (4, 240) tensor                              │
│                                                              │
│ CNN Inference (NEW)                                         │
│ ├── Conv1D layers (feature extraction)                      │
│ ├── Fully connected layers                                  │
│ ├── Sigmoid output                                          │
│ └── Direct probability output [0.0, 1.0]                    │
│                                                              │
│ ~~Stress FSM~~ (REMOVED)                                    │
│                                                              │
│ BLE Service (ENHANCED)                                      │
│ ├── Device Discovery & Pairing (NEW)                        │
│ │   ├── Device UUID broadcast                               │
│ │   ├── Owner pairing commands                              │
│ │   └── NVS storage for owner info                          │
│ └── Transmits: Continuous probability [0.0, 1.0]            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Implementation Tasks

### ✅ Task 1: Model Analysis (COMPLETED)
- [x] Load and inspect best.pth
- [x] Document model architecture
- [x] Generate test data (test_data.h)
- [x] Create visualization (preprocessing_visualization.png)
- [x] Export test dataset (test_data_for_esp32.json)

**Files Created:**
- `test_pytorch_model.py` - Analysis script
- `test_data.h` - C arrays for ESP32 validation
- `test_data_for_esp32.json` - Full test dataset
- `preprocessing_visualization.png` - Visual comparison

---

### 🔨 Task 2: Implement Preprocessing in C (IN PROGRESS)

**Create New Component:** `components/signal_preprocessor/`

#### Files to Create:
```
components/signal_preprocessor/
├── include/
│   └── signal_preprocessor.h
└── src/
    ├── signal_preprocessor.c
    ├── resample.c              # Linear interpolation
    └── normalize.c             # Z-score normalization
```

#### Key Functions:

```c
/**
 * Resample signal using linear interpolation
 * 
 * @param input        Input signal array
 * @param input_len    Input signal length
 * @param input_rate   Original sampling rate (Hz)
 * @param output       Output buffer (must be pre-allocated)
 * @param output_len   Target output length
 * @param target_rate  Target sampling rate (Hz)
 * @return 0 on success, negative on error
 */
int resample_signal(const float *input, uint16_t input_len,
                   uint8_t input_rate,
                   float *output, uint16_t output_len,
                   uint8_t target_rate);

/**
 * Z-score normalization
 * 
 * @param signal  Signal array (modified in-place)
 * @param length  Signal length
 * @return 0 on success, negative on error
 */
int normalize_signal_zscore(float *signal, uint16_t length);

/**
 * Preprocess sensor data for CNN model
 * 
 * @param sensor_system  Real-time sensor system with ring buffers
 * @param output         Output tensor (4 × 240) pre-allocated
 * @return 0 on success, negative on error
 */
int preprocess_for_cnn(realtime_sensor_system_t *sensor_system,
                       float output[4][240]);
```

#### Algorithm: Linear Interpolation
```c
// Pseudo-code for resampling
float resample_linear(float *data, int old_len, int new_len) {
    float *result = allocate(new_len);
    float ratio = (float)(old_len - 1) / (new_len - 1);
    
    for (int i = 0; i < new_len; i++) {
        float src_pos = i * ratio;
        int idx = (int)src_pos;
        float frac = src_pos - idx;
        
        if (idx + 1 < old_len) {
            result[i] = data[idx] * (1 - frac) + data[idx + 1] * frac;
        } else {
            result[i] = data[idx];
        }
    }
    return result;
}
```

#### Algorithm: Z-Score Normalization
```c
// Pseudo-code for normalization
void normalize_zscore(float *data, int len) {
    // Compute mean
    float sum = 0;
    for (int i = 0; i < len; i++) sum += data[i];
    float mean = sum / len;
    
    // Compute std
    float sum_sq = 0;
    for (int i = 0; i < len; i++) {
        float diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    float std = sqrt(sum_sq / len);
    
    // Normalize
    if (std < 1e-6) std = 1.0;  // Avoid division by zero
    for (int i = 0; i < len; i++) {
        data[i] = (data[i] - mean) / std;
    }
}
```

---

### 🔄 Task 3: Convert Model to TFLite

**Steps:**
1. Export PyTorch to ONNX
2. Convert ONNX to TensorFlow
3. Convert TensorFlow to TFLite with quantization
4. Generate C arrays for embedding

**Script to Create:** `convert_model_to_tflite.py`

```python
import torch
import torch.onnx
import tensorflow as tf
import numpy as np

# 1. Load PyTorch model
checkpoint = torch.load('best.pth', weights_only=False)
model = load_model_architecture()  # Define architecture
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Export to ONNX
dummy_input = torch.randn(1, 4, 240)
torch.onnx.export(model, dummy_input, 'stress_model.onnx',
                  input_names=['input'], output_names=['output'])

# 3. Convert ONNX to TensorFlow
import onnx
from onnx_tf.backend import prepare
onnx_model = onnx.load('stress_model.onnx')
tf_model = prepare(onnx_model)

# 4. Convert to TFLite with int8 quantization
converter = tf.lite.TFLiteConverter.from_saved_model('tf_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()

# 5. Save TFLite model
with open('stress_model.tflite', 'wb') as f:
    f.write(tflite_model)

# 6. Generate C array
xxd -i stress_model.tflite > stress_model_data.h
```

---

### 🗑️ Task 4: Remove FSM Component

**Files to Modify:**
- `main/main_realtime.c` - Remove FSM logic from consumer task
- `components/ble_stress_service/` - Update to send probability instead of state

**Changes in `consumer_task()`:**

```c
// OLD CODE (Remove):
bool transition = stress_fsm_process_inference(&g_stress_fsm,
                                               prob,
                                               now_ms,
                                               on_stress_transition);

// NEW CODE (Replace with):
// Send probability directly via BLE
ble_stress_service_update_probability(prob, now_ms);
```

**Update BLE Service:**

```c
// Add new function
void ble_stress_service_update_probability(float probability, uint32_t timestamp_ms);

// Update advertisement to include:
// - Device UUID
// - Current probability
// - Timestamp
// - Owner pairing status
```

---

### 🔗 Task 5: Implement Device Pairing

**New BLE Characteristics:**

```c
// Device identification
#define DEVICE_UUID_CHAR        "12345678-1234-5678-1234-567812345678"
#define OWNER_UUID_CHAR         "12345678-1234-5678-1234-567812345679"
#define PAIRING_COMMAND_CHAR    "12345678-1234-5678-1234-567812345680"
#define PAIRING_STATUS_CHAR     "12345678-1234-5678-1234-567812345681"

// Pairing commands
typedef enum {
    PAIR_CMD_CLAIM_DEVICE = 0x01,    // Claim this device as owner
    PAIR_CMD_RELEASE_DEVICE = 0x02,  // Release ownership
    PAIR_CMD_QUERY_STATUS = 0x03     // Query pairing status
} pairing_command_t;

// Pairing status
typedef enum {
    PAIR_STATUS_UNPAIRED = 0x00,     // No owner
    PAIR_STATUS_PAIRED = 0x01,       // Has owner
    PAIR_STATUS_CONNECTING = 0x02    // Pairing in progress
} pairing_status_t;
```

**NVS Storage:**

```c
#include "nvs_flash.h"
#include "nvs.h"

// Store owner UUID in non-volatile storage
int store_owner_uuid(const char *owner_uuid);
int load_owner_uuid(char *owner_uuid, size_t max_len);
int clear_owner_uuid(void);
bool is_device_paired(void);
```

**Pairing Flow:**

```
macOS App                           ESP32 Device
    |                                    |
    |------ Scan for devices -------->|
    |<----- Device UUID broadcast ----|
    |                                    |
    |------ PAIR_CMD_CLAIM_DEVICE ---->|
    |       (with macOS UUID)            |
    |                                    |
    |<----- PAIR_STATUS_PAIRED --------|
    |       (confirmation)               |
    |                                    |
    |------ Connect to device --------->|
    |<----- Continuous probability -----|
```

---

### 📱 Task 6: Update macOS App

**Changes Required:**

1. **Device Discovery Screen** (NEW)
   ```swift
   - Scan for Shadow devices
   - Show list with:
     * Device name
     * Device UUID
     * Pairing status (Available/Paired/Mine)
     * Signal strength
   - "Claim Device" button for unpaired devices
   ```

2. **Pairing Flow** (NEW)
   ```swift
   - Send pairing command with macOS UUID
   - Store paired devices in UserDefaults
   - Auto-connect only to owned devices
   ```

3. **Stress Probability Display** (MODIFIED)
   ```swift
   - Replace discrete state (CALM/STRESS)
   - Show continuous probability [0-100%]
   - Add probability graph (real-time)
   - Color gradient: Green (0%) → Red (100%)
   ```

---

## 📝 Testing & Validation

### Validation Steps:

1. **Preprocessing Validation**
   ```
   - Load test_data.h in ESP32
   - Run preprocessing
   - Compare output with expected_*_normalized arrays
   - Verify < 0.001 difference
   ```

2. **Model Output Validation**
   ```
   - Run same input through Python model
   - Run same input through TFLite on ESP32
   - Compare outputs
   - Accept < 1% difference
   ```

3. **BLE Pairing Test**
   ```
   - Flash 2 ESP32 devices
   - Pair device 1 with macOS
   - Verify device 2 still shows as unpaired
   - Test ownership persistence after reboot
   ```

4. **End-to-End Test**
   ```
   - Collect real sensor data
   - Verify preprocessing
   - Verify CNN inference
   - Verify BLE transmission
   - Compare with Python predictions
   ```

---

## 📊 Memory Requirements

### Current (Feature-Based MLP):
```
- Feature extraction workspace: ~15 KB
- MLP model weights: ~25 KB
- FSM + Event Log: ~2 KB
Total: ~42 KB
```

### New (CNN Model):
```
- Signal preprocessor workspace: ~8 KB (4 channels × 240 samples × 8 bytes)
- CNN model (TFLite quantized): ~50 KB (estimate)
- NVS owner storage: ~1 KB
Total: ~59 KB (+17 KB)
```

**ESP32-S3 has 512 KB SRAM - plenty of headroom ✅**

---

## 🎯 Success Criteria

- [ ] Preprocessing matches Python output (< 0.1% error)
- [ ] CNN inference time < 100ms
- [ ] Model accuracy matches PyTorch (> 99%)
- [ ] Device pairing works reliably
- [ ] BLE connection stable
- [ ] Owner persistence across reboots
- [ ] macOS app displays continuous probability
- [ ] Multiple devices can be discovered and paired

---

## 📅 Timeline Estimate

| Task | Estimated Time |
|------|----------------|
| 1. Model Analysis | ✅ Complete |
| 2. Preprocessing in C | 2-3 days |
| 3. Model Conversion | 1-2 days |
| 4. Remove FSM | 1 day |
| 5. Device Pairing | 2-3 days |
| 6. macOS App Update | 2-3 days |
| 7. Testing & Validation | 2-3 days |
| **Total** | **10-15 days** |

---

## 🔧 Development Order

1. ✅ Analyze model (Complete)
2. 🔨 Implement preprocessing in C
3. 🔨 Validate preprocessing with test data
4. 🔨 Convert PyTorch → TFLite
5. 🔨 Integrate TFLite interpreter
6. 🔨 Test inference performance
7. 🔨 Remove FSM logic
8. 🔨 Implement device pairing (ESP32)
9. 🔨 Update BLE characteristics
10. 🔨 Implement device discovery (macOS)
11. 🔨 Update UI for continuous probability
12. 🔨 End-to-end testing
13. 🔨 Performance optimization
14. ✅ Production deployment

---

**Next Step:** Begin implementing signal preprocessing in C (`components/signal_preprocessor/`)
