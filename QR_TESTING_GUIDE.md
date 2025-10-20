# QR Scanner Testing Guide

## ⚠️ Issue: QR Code Not Being Detected

### What Should Happen (Automatic):
1. ✅ Click "Scan QR Code" button
2. ✅ Camera permission granted
3. ✅ Camera feed appears
4. ✅ **AUTOMATIC** - Just hold QR in view, no button to click
5. ✅ When QR detected → **BEEP** sound
6. ✅ Scanner closes automatically
7. ✅ "Device paired successfully!" alert
8. ✅ BLE scanning starts

### Debug Test - Step by Step

#### Test 1: Check Camera Status

1. **Rebuild the app:**
   ```bash
   # In Xcode
   ⇧⌘K (Clean)
   ⌘B (Build)
   ⌘R (Run)
   ```

2. **Open the scanner:**
   - Click "Scan QR Code" button
   - Grant camera permission if asked

3. **Check Console Output** (View > Debug Area > Show Debug Area):
   ```
   Should see:
   🎥 QRScannerView appeared - starting camera
   🎬 Starting QR scanner setup...
   ✅ Camera permission granted
   📹 Found camera device: FaceTime HD Camera
   ✅ Added video input
   📋 Available metadata types: [...]
   ✅ Added metadata output for QR codes
   ✅ Created preview layer
   ✅ Camera session started
   🖼️ QRScannerCameraView: Adding preview layer to view
   ```

4. **Check On-Screen Status:**
   - Should see: "🔴 SCANNING" indicator (pulsing red dot)
   - Should see: "✅ Camera ready - Scan QR code" (yellow text)
   - Should see: Green scanning frame (250x250 box)
   - Should see: **YOUR FACE** in camera feed (most important!)

#### Test 2: Test with HTML QR Code (No ESP32 Needed!)

1. **Open the test page:**
   ```bash
   open /Users/ashidudissanayake/Dev/Shadow/test-qr.html
   ```

2. **Position the QR code:**
   - Make QR code **LARGE** on screen (zoom browser to 150-200%)
   - Point Mac camera at the **monitor** showing QR code
   - Keep QR code **centered** in green frame
   - Hold **steady** for 2-3 seconds
   - Ensure **good lighting** (no glare/reflections)

3. **Watch console for:**
   ```
   📸 Metadata received: 1 objects
   📸 Metadata type: org.iso.QRCode
   📸 QR Code detected: Shadow-9026
   📸 Calling onCodeScanned callback
   📱 QR Code detected: Shadow-9026
   ✅ Device paired: Shadow-9026
   ```

4. **If you see:**
   - ❌ "📸 No readable metadata objects" → QR not detected yet
   - ❌ Nothing → Camera not scanning or QR out of view
   - ✅ "📸 QR Code detected: Shadow-9026" → **SUCCESS!**

#### Test 3: Check Why Not Detecting

**A. Camera Feed Not Showing?**
```
Check console for:
"🖼️ QRScannerCameraView: Adding preview layer to view"

If missing → Preview layer not created
If present but no video → Check Activity Monitor for camera usage
```

**B. Camera Scanning But Not Detecting?**

Possible issues:
1. **QR code too small** - Make it bigger (300x300px minimum)
2. **QR code blurry** - Hold steady, focus properly
3. **Wrong QR format** - Must be text, not URL
4. **Metadata types wrong** - Check console for available types

**C. Check Available Metadata Types:**

Look for this line in console:
```
📋 Available metadata types: [...]
```

Should include: `org.iso.QRCode`

If QR not available, scanner will use ALL available types as fallback.

#### Test 4: Manual Test with Different QR Codes

Try these device names in `test-qr.html`:
- `Shadow-1234`
- `Shadow-9026` (default)
- `Shadow-ABCD`

Each should:
1. Generate new QR code
2. Be detectable by scanner
3. Pass validation (starts with "Shadow-")
4. Pair successfully

#### Test 5: ESP32 Real Test

After HTML test works:

1. **Flash ESP32:**
   ```bash
   cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware
   idf.py build flash monitor
   ```

2. **Press button to show QR:**
   - ESP32 displays QR on screen
   - QR contains just device name (e.g., "Shadow-9026")
   - White background, black QR code

3. **Hold ESP32 display to Mac camera:**
   - Same process as HTML test
   - Watch console for detection
   - Should pair automatically

### Troubleshooting

#### Issue: "No camera found"
```bash
# Check camera available
system_profiler SPCameraDataType

# Kill camera processes
sudo killall VDCAssistant
sudo killall AppleCameraAssistant
```

#### Issue: Camera shows but doesn't scan
- Check "🔴 SCANNING" indicator is showing
- Check console shows "✅ Camera session started"
- Check `captureSession.isRunning` is true
- Try restarting app

#### Issue: QR detected but validation fails
Check console for:
```
📱 QR Code detected: <value>
Invalid Shadow device QR code. Expected format: Shadow-XXXX
```

QR must start with "Shadow-" (case sensitive)

#### Issue: No metadata callback
Check console for metadata types:
```
📋 Available metadata types: [...]
```

If empty or missing QR type, camera can't detect QR codes.

### Expected Behavior Summary

| Step | What Happens | What You See |
|------|--------------|--------------|
| 1. Click button | Scanner opens | Camera feed appears |
| 2. Camera starts | Metadata scanning begins | "🔴 SCANNING" indicator |
| 3. QR in view | Detection runs automatically | Green frame visible |
| 4. QR detected | **BEEP** sound | Console: "📸 QR Code detected" |
| 5. Validation | Check "Shadow-" prefix | Console: Validation message |
| 6. Save | Store in UserDefaults | Key: "PairedShadowDevice" |
| 7. Callback | Notify dashboard | Alert: "Device paired successfully!" |
| 8. Dismiss | Scanner closes | Return to dashboard |
| 9. BLE start | Begin scanning | Shows connection status |

### Key Points

✅ **No button to click** - Detection is automatic  
✅ **Hold QR steady** - Give it 1-2 seconds  
✅ **Good lighting** - Avoid glare and shadows  
✅ **Centered in frame** - Use green box as guide  
✅ **Large enough** - 200-300px minimum on screen  
✅ **Console is your friend** - Watch for detection logs  

---

**Last Updated:** 20 October 2025  
**Status:** Ready for testing with enhanced debug output 🎯
