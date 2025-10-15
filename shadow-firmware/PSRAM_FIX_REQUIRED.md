# CRITICAL FIX REQUIRED: Enable PSRAM

## Issue Identified

**CNN initialization is FAILING** because PSRAM (external RAM) is **disabled** in the project configuration.

### Error from Device Log:
```
E (643) cnn_inference: Failed to allocate 204800 bytes in PSRAM
E (653) ShadowRealTime: ❌ CNN initialization failed: -1
E (663) ShadowRealTime: System will continue but ML inference will be disabled
```

### Root Cause:
```bash
# In sdkconfig:
# CONFIG_SPIRAM is not set  ❌ PSRAM DISABLED
```

The ESP32-S3 has **8MB PSRAM** but it's not enabled in the build configuration.

## Solution Applied

✅ **Modified `sdkconfig` to enable PSRAM:**

```
CONFIG_SPIRAM=y
CONFIG_SPIRAM_MODE_OCT=y
CONFIG_SPIRAM_SPEED_80M=y
CONFIG_SPIRAM_BOOT_INIT=y
CONFIG_SPIRAM_USE_MALLOC=y
CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=16384
CONFIG_SPIRAM_MALLOC_RESERVE_INTERNAL=32768
```

**Configuration Details:**
- **Mode:** Octal (OPI) - High-speed 8-line interface
- **Speed:** 80MHz - Maximum speed for ESP32-S3
- **Boot Init:** Yes - Initialize PSRAM during boot
- **Use for malloc:** Yes - Allow dynamic allocation from PSRAM
- **Reserve Internal:** 32KB - Keep some SRAM for critical operations

## Required Actions

### 1. Reconfigure Project (with ESP-IDF environment active)
```bash
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware
idf.py reconfigure
```

### 2. Rebuild Firmware
```bash
idf.py build
```

### 3. Flash to Device
```bash
idf.py flash
```

### 4. Monitor Output
```bash
idf.py monitor
```

## Expected Output After Fix

### During Boot:
```
I (xxx) cpu_start: Pro cpu start user code
I (xxx) cpu_start: cpu freq: 240000000 Hz  ← Should show 240MHz with PSRAM
I (xxx) app_init: Application information:
...
I (xxx) heap_init: Initializing. RAM available for dynamic allocation:
I (xxx) heap_init: At 3FCAB3D8 len 0003E338 (248 KiB): RAM
I (xxx) heap_init: At 3FCE9710 len 00005724 (21 KiB): RAM
I (xxx) heap_init: At 3FCF0000 len 00008000 (32 KiB): DRAM
I (xxx) heap_init: At 600FE020 len 00001FC8 (7 KiB): RTCRAM
I (xxx) SPIRAM: Found 8MB PSRAM device  ← NEW: PSRAM detected
I (xxx) SPIRAM: Speed: 80MHz  ← NEW
I (xxx) SPIRAM: Initialized, cache is in normal mode  ← NEW
...
I (xxx) cnn_inference: Initializing CNN with TFLite Micro...
I (xxx) cnn_inference: Allocated 200 KB tensor arena in PSRAM  ← SUCCESS!
I (xxx) cnn_inference: Model loaded: 124176 bytes
I (xxx) cnn_inference: Tensor arena: XXXXX / 204800 bytes (XX.X% used)
I (xxx) cnn_inference: CNN initialized successfully  ← SUCCESS!
I (xxx) ShadowRealTime: ✅ CNN initialized successfully
```

### CNN Inference (every 60 seconds):
```
I (xxx) ShadowRealTime: 🔔 CNN Inference #1
I (xxx) ShadowRealTime: ✅ Preprocessing complete in XX ms
I (xxx) cnn_inference: Inference: XX.X%, XXXXus
I (xxx) ShadowRealTime: 🎯 CNN Inference Result:
I (xxx) ShadowRealTime:    Stress Probability: XX.X%
I (xxx) ShadowRealTime:    Class: STRESS / NORMAL
```

## Why This Matters

**Without PSRAM:**
- ❌ CNN tensor arena (200 KB) cannot be allocated
- ❌ SRAM exhausted (only ~300 KB total internal RAM)
- ❌ CNN initialization fails
- ❌ System falls back to no ML inference

**With PSRAM:**
- ✅ 8 MB external RAM available
- ✅ 200 KB tensor arena allocated easily
- ✅ CNN initialization succeeds
- ✅ ~250 KB internal SRAM free for other tasks
- ✅ Full ML inference pipeline active

## Memory Architecture (After Fix)

```
ESP32-S3 Memory Layout:
┌─────────────────────────────────────┐
│ Flash (2MB)                         │
│  - Bootloader                       │
│  - Partition table                  │
│  - Application (866 KB)             │
│  - Model data (121 KB in .rodata)  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Internal SRAM (~300 KB)             │
│  - FreeRTOS stack & heap            │
│  - BLE stack                        │
│  - Sensor buffers                   │
│  - Code execution                   │
│  Free: ~190 KB after init           │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ PSRAM (8 MB) ← NEW!                 │
│  - CNN tensor arena (200 KB)       │
│  - Available for expansion          │
│  Free: ~7.8 MB                      │
└─────────────────────────────────────┘
```

## Performance Impact

**PSRAM Access Speed:**
- **Octal mode @ 80MHz:** ~40 MB/s throughput
- **Cache enabled:** First access cached to SRAM
- **Impact on CNN:** Minimal (~5-10% vs internal SRAM)

**Expected inference time:** 50-150ms (well within <200ms target)

## Verification Checklist

After rebuilding and flashing:

- [ ] Boot logs show "SPIRAM: Found 8MB PSRAM device"
- [ ] CNN initialization succeeds (no error -1)
- [ ] "Allocated 200 KB tensor arena in PSRAM" message appears
- [ ] Free heap increases (should be ~8MB total with PSRAM)
- [ ] First CNN inference runs successfully after 60 seconds
- [ ] Inference time <200ms
- [ ] System remains stable (no crashes, no memory leaks)

## Troubleshooting

**If PSRAM still not detected:**
1. Check hardware: ESP32-S3 variant must have PSRAM (yours does: "Embedded PSRAM 8MB")
2. Verify pin configuration in sdkconfig
3. Try different PSRAM modes (Quad vs Octal)

**If allocation still fails:**
1. Check `esp_get_free_heap_size()` and `heap_caps_get_free_size(MALLOC_CAP_SPIRAM)`
2. Reduce tensor arena size temporarily (e.g., 100 KB)
3. Check for memory leaks elsewhere

**If inference is slow:**
1. Enable PSRAM cache optimization
2. Consider reducing model size
3. Profile with ESP-IDF's profiling tools

## Alternative Solutions (if PSRAM still fails)

### Option 1: Reduce Tensor Arena Size
```cpp
constexpr int kTensorArenaSize = 100 * 1024;  // Reduce from 200KB to 100KB
```
Risk: May cause AllocateTensors() to fail if model needs more memory.

### Option 2: Use Internal SRAM (not recommended)
```cpp
// Back to static allocation (will likely still overflow)
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
```
Risk: DRAM overflow, system instability.

### Option 3: Model Optimization
- Re-quantize with full INT8 (activations + weights)
- Prune model to reduce parameters
- Use lighter model architecture

## Commands Quick Reference

```bash
# Enable ESP-IDF environment (if not already)
. $HOME/Dev/esp/esp-idf/export.sh

# Navigate to project
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware

# Reconfigure with new PSRAM settings
idf.py reconfigure

# Build firmware
idf.py build

# Flash to device
idf.py flash

# Monitor serial output
idf.py monitor

# Or do all at once
idf.py reconfigure build flash monitor
```

## Files Modified

- ✅ `sdkconfig` - PSRAM configuration enabled
- ✅ `enable_psram.py` - Helper script to enable PSRAM

## Next Steps

1. **Immediate:** Rebuild firmware with PSRAM enabled
2. **Verify:** CNN initialization succeeds on device
3. **Test:** Run first inference and measure latency
4. **Validate:** Check memory usage and system stability
5. **Continue:** Proceed with remaining tasks (pairing, macOS app, etc.)

---

## Status Update

**Current State:** ⚠️ **BLOCKED** - CNN cannot initialize without PSRAM

**After Fix:** ✅ **READY** - CNN will initialize and run successfully

**Estimated Time to Fix:** 5-10 minutes (rebuild + flash + verify)

