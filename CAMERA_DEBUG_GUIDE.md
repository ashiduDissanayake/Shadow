# Camera QR Scanner Debug Guide

## Changes Made

### 1. Fixed QRScannerView Issues
- ✅ Changed `@Binding var onDeviceScanned` to `var onDeviceScanned` (not a binding)
- ✅ Added camera permission request with `AVCaptureDevice.requestAccess()`
- ✅ Added detailed status messages and console logging
- ✅ Added error handling for camera initialization
- ✅ Split camera setup into separate method with proper error checking

### 2. Added Debug Output
- 📹 Camera status displayed on screen
- 🖨️ Console logs for each step:
  - Camera permission request
  - Camera device detection
  - Video input setup
  - Metadata output configuration
  - Session start/stop

### 3. Fixed Dashboard Integration
- Changed `.constant({ ... })` to direct closure `{ ... }`
- Added console log when device is paired
- Properly dismiss scanner after successful scan

## Testing Steps

### Step 1: Build and Run
```bash
# Open Xcode
open /Users/ashidudissanayake/Dev/Shadow/Shadow/Shadow.xcodeproj

# In Xcode:
# 1. Select "My Mac" target (NOT "My Mac (Designed for iPad)")
# 2. Product > Clean Build Folder (⇧⌘K)
# 3. Product > Build (⌘B)
# 4. Product > Run (⌘R)
```

### Step 2: Check Console Output
When you click "Scan QR Code", you should see:
```
🎥 QRScannerView appeared - starting camera
🎬 Starting QR scanner setup...
```

Then either:
- ✅ **Permission Granted:**
  ```
  ✅ Camera permission granted
  📹 Found camera device: FaceTime HD Camera
  ✅ Added video input
  ✅ Added metadata output for QR codes
  ✅ Created preview layer
  ✅ Camera session started
  📹 Camera status: ✅ Camera ready - Scan QR code
  ```

- ⚠️ **Permission Denied:**
  ```
  ❌ Camera permission denied
  📹 Camera status: ⚠️ Camera permission denied - Please enable in System Preferences
  ```

### Step 3: Grant Camera Permission (if needed)
If permission is denied:

1. **System Preferences > Security & Privacy > Camera**
2. Find "Shadow" in the list
3. Check the box next to it
4. Restart the app

Or via Terminal:
```bash
# Reset camera permissions (will ask again on next launch)
tccutil reset Camera
```

### Step 4: Test QR Scanning
1. Flash your ESP32 with the firmware
2. The ESP32 should show QR code on display
3. Hold QR code in front of Mac camera
4. You should see:
   ```
   📱 QR Code detected: Shadow-XXXX
   ✅ Device paired: Shadow-XXXX
   ```

## Troubleshooting

### Issue: Camera Permission Popup Doesn't Appear
**Solution:**
```bash
# Reset TCC database
tccutil reset Camera

# Rebuild app with clean
# In Xcode: ⇧⌘K then ⌘B
```

### Issue: "No camera found" Error
**Check:**
1. Is FaceTime camera working in Photo Booth?
2. Is any other app using the camera?
3. Try restarting your Mac

**Terminal Check:**
```bash
# List available cameras
system_profiler SPCameraDataType
```

### Issue: Black Camera Screen
**Check Console for:**
- "❌ Cannot add video input" → Camera busy
- "❌ Failed to access camera" → Permission issue
- No "✅ Camera session started" → Setup failed

**Fix:**
```bash
# Kill any process using camera
sudo killall VDCAssistant
sudo killall AppleCameraAssistant
```

### Issue: QR Code Not Detected
**Check:**
1. QR code is well-lit and clear
2. QR code fills about 50% of camera view
3. Hold steady for 1-2 seconds
4. Console shows "✅ Camera ready - Scan QR code"

**ESP32 QR Format:**
- Must be: `Shadow-XXXX` (e.g., "Shadow-9026")
- Case sensitive
- No spaces or extra characters

## Status Messages

| Message | Meaning |
|---------|---------|
| `Initializing...` | App just opened scanner |
| `Requesting camera access...` | Asking system for permission |
| `⚠️ Camera permission denied` | Need to enable in System Preferences |
| `Setting up camera...` | Permission granted, configuring |
| `Initializing camera...` | Creating capture session |
| `❌ No camera found` | Hardware not detected |
| `❌ Failed to access camera` | Camera in use or error |
| `Starting camera feed...` | Almost ready |
| `✅ Camera ready - Scan QR code` | **READY TO SCAN** |

## Expected Flow

```
User clicks "Scan QR Code"
    ↓
Sheet presents QRScannerView
    ↓
onAppear() → startScanning()
    ↓
Request camera permission (first time only)
    ↓
[User grants permission]
    ↓
setupCamera() creates AVCaptureSession
    ↓
Add video input from FaceTime camera
    ↓
Add metadata output for QR detection
    ↓
Create preview layer for display
    ↓
Start capture session
    ↓
Camera feed appears on screen
    ↓
[User shows QR code]
    ↓
metadataOutput() detects QR code
    ↓
handleScannedCode() validates "Shadow-XXXX"
    ↓
Save to UserDefaults
    ↓
Call onDeviceScanned callback
    ↓
Dashboard receives deviceName
    ↓
Show success alert
    ↓
Start BLE scanning
```

## Next Steps After Camera Works

1. ✅ Camera feed shows
2. ✅ QR code detected
3. ✅ Device paired successfully
4. Test BLE connection with paired device
5. Test passive monitoring
6. Test missed event replay
7. Test notifications

---

**Built:** 20 October 2025  
**Status:** Ready for testing 🚀
