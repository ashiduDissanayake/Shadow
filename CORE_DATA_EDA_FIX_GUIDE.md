# Shadow iOS Core Data Reset & EDA Fix Guide

## Issues Resolved

### 1. Core Data Multiple Entity Description Error
**Problem**: `CoreData: warning: Multiple NSEntityDescriptions claim the NSManagedObject subclass 'StressEvent'`

**Root Cause**: Multiple NSManagedObjectModel instances claiming the same entities, causing Core Data confusion.

### 2. EDA Voltage Large Jumps  
**Problem**: EDA voltage readings jumping between extreme values (0V to 3.3V) unrealistically.

**Root Cause**: No filtering or range validation on raw ADC readings.

## Solutions Implemented

### 📱 iOS Core Data Reset System

#### Files Added:
1. **`CoreDataReset.swift`** - Utility class for complete Core Data management
2. **`CoreDataDebugView.swift`** - UI for managing Core Data state

#### Key Features:
- **Complete Reset**: Deletes all Core Data files (.sqlite, .wal, .shm)
- **Data-Only Reset**: Clears all data but keeps structure
- **Dynamic UUID Management**: Auto-generates device UUID if none exists
- **Debug Interface**: Easy access to Core Data statistics and controls

#### Usage:
1. **Access Debug View**: 
   - Open Shadow app → Dashboard → Debug Tools → "Core Data Manager"

2. **Reset Options**:
   - **Delete All Data**: Clears entries but keeps database structure
   - **Complete Reset**: Removes all Core Data files (requires app restart)

3. **Device UUID**:
   - Auto-generates if none exists
   - Can manually generate new UUID
   - Stored in UserDefaults for persistence

### 🔧 ESP32 EDA Voltage Enhancement

#### Changes Made to `main_realtime.c`:

```c
// New EDA Configuration
#define EDA_MIN_VOLTAGE     0.1f   // Minimum realistic EDA voltage
#define EDA_MAX_VOLTAGE     2.5f   // Maximum realistic EDA voltage  
#define EDA_DEFAULT_VOLTAGE 1.5f   // Default baseline voltage
#define EDA_NOISE_THRESHOLD 0.05f  // Filter out voltage changes smaller than 50mV

// Enhanced processing function
static float gsr_process_voltage(float raw_voltage);
```

#### Features Added:
1. **Range Validation**: Clamps readings to realistic 0.1V-2.5V range
2. **Noise Filtering**: Ignores changes smaller than 50mV threshold  
3. **Smoothing**: Applies 80% new + 20% old value averaging
4. **Baseline Initialization**: Smart startup calibration

#### Expected Results:
- **Before**: EDA readings jumping 0V → 3.141V → 0V
- **After**: Smooth EDA readings in 0.5V-2.0V range with gradual changes

## 🚀 Quick Fix Instructions

### For Core Data Issues:
1. Build and run the updated iOS app
2. Navigate to Dashboard → Debug Tools → "Core Data Manager" 
3. Tap "Delete All Data" to clear corrupted entries
4. Restart the app - Core Data will recreate clean structures

### For EDA Voltage Issues:
1. Flash the updated firmware:
   ```bash
   cd shadow-firmware
   source ~/Dev/esp/esp-idf/export.sh
   idf.py flash monitor
   ```
2. Observe EDA readings in terminal - should see smoother values
3. Connect iOS app to see improved data visualization

## 📊 Validation

### Core Data Health Check:
- No more "Multiple NSEntityDescriptions" warnings
- Clean device UUID assignment
- Proper BLE data persistence

### EDA Data Quality:
- Voltage readings stay within realistic GSR range
- Smooth transitions between values
- Filtered noise for cleaner ML input

## 🔄 Reset Procedures

### Emergency Core Data Reset:
If app crashes or data corruption persists:
1. Delete app from iOS device/simulator
2. Reinstall app (this removes all Core Data files)
3. App will start fresh with auto-generated device UUID

### ESP32 EDA Recalibration:
If EDA readings seem stuck:
1. Power cycle the ESP32 device
2. EDA baseline will reinitialize on first reading
3. Wait ~30 seconds for values to stabilize

## 📝 Technical Notes

### Core Data Architecture:
- Single NSPersistentContainer in `Shadow.swift`
- Repository pattern with safe attribute checking
- Dynamic UUID management with UserDefaults fallback

### EDA Signal Processing:
- 4Hz sampling rate maintained
- 10-sample averaging for stable readings
- Light smoothing preserves real physiological changes
- Range clamping prevents unrealistic values

This comprehensive fix addresses both the iOS Core Data multiple entity issues and the ESP32 EDA voltage instability, providing a stable foundation for the Shadow stress detection system.
