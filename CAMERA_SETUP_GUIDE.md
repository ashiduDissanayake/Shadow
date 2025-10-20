# 🎥 Camera Setup & Testing Guide

## ⚠️ Important: Camera in Simulator

**The Xcode Simulator CANNOT access your Mac's camera!**

The QR scanner will **NOT work** in the simulator because:
- Simulators don't have camera hardware
- macOS security doesn't allow simulator camera access
- AVFoundation APIs will fail in simulator

---

## ✅ Solution 1: Run on Real Mac (Production)

### **Step 1: Select "My Mac" as Target**

1. Open Xcode
2. At the top, next to the Run button (▶️), click the device selector
3. Choose **"My Mac"** (not "My Mac (Designed for iPad)")
4. Click **Run** (▶️)

### **Step 2: Grant Camera Permission**

When you first open the QR scanner, macOS will show a popup:

```
"Shadow" Would Like to Access the Camera
[Don't Allow]  [OK]
```

Click **"OK"** to grant permission.

### **Step 3: If Permission Was Denied**

If you accidentally clicked "Don't Allow", you need to manually enable it:

1. Open **System Settings** (⚙️)
2. Go to **Privacy & Security**
3. Click **Camera**
4. Find **Shadow** in the list
5. Toggle it **ON**
6. Restart the Shadow app

---

## ✅ Solution 2: Simulator Fallback Mode (Development)

I've added a **simulator fallback** that lets you test without a camera!

### **How It Works:**

When running in simulator, the QR scanner shows a text input instead:

```
📱 Simulator Mode
Camera not available in simulator

Enter Shadow Device Name:
┌─────────────────────┐
│ Shadow-9026         │
└─────────────────────┘
Default: Shadow-9026

[Cancel]  [Pair Device]

💡 Run on real Mac to use camera
```

### **Testing in Simulator:**

1. Run app in simulator (iPhone/iPad simulator)
2. Tap "Scan QR Code"
3. You'll see the text input instead of camera
4. Type: `Shadow-9026` (or whatever your ESP32 device name is)
5. Click "Pair Device"
6. App will pair as if you scanned the QR code!

---

## 📋 Camera Permission Key (Info.plist)

The following has been added to your `Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>Shadow needs camera access to scan QR codes from your Shadow device for pairing.</string>
```

This is the message users will see when requesting camera access.

---

## 🧪 Testing Workflow

### **For Development (Simulator):**
```
1. Build → Run on iPhone/iPad Simulator
2. Tap "Scan QR Code"
3. Enter "Shadow-9026" manually
4. Continue testing with BLE (if simulated)
```

### **For Real Testing (Mac):**
```
1. Build → Run on "My Mac"
2. Grant camera permission (one-time)
3. Tap "Scan QR Code"
4. Point camera at ESP32 QR code
5. App auto-pairs when QR detected
```

---

## 🔧 Troubleshooting

### **"Camera permission denied"**

**Fix:** Open System Settings → Privacy & Security → Camera → Enable Shadow

### **"No camera found"**

**Check:**
- Running on real Mac (not simulator)?
- Camera not in use by another app?
- Try quitting Xcode and rebuilding

### **"QR code not scanning"**

**Tips:**
- Hold device steady
- Ensure good lighting
- QR code should fill the green frame
- Try moving closer/further away

### **"Device paired but not connecting"**

**Check:**
- ESP32 powered on?
- BLE enabled on Mac? (System Settings → Bluetooth)
- Device name matches (e.g., "Shadow-9026")?
- Try unpairing and re-pairing

---

## 📱 ESP32 QR Code Display

Your ESP32 firmware shows the QR code:

1. **Power on ESP32**
2. **Press button** (GPIO 14) on T-Display S3
3. Display toggles: **Clock ↔ QR Code**
4. QR code shows: **"Shadow-9026"** (device name only, no password)

---

## 🎯 Quick Test Checklist

- [ ] Camera permission added to Info.plist ✅
- [ ] Running on real Mac (not simulator)
- [ ] Camera permission granted in System Settings
- [ ] ESP32 showing QR code on display
- [ ] App camera view shows live preview
- [ ] QR code scans successfully
- [ ] Device name saved ("Shadow-9026")
- [ ] BLE starts scanning automatically
- [ ] Connection status shows "Connected"

---

## 💡 Pro Tips

1. **Use Real Mac for Final Testing**
   - Simulator is fine for UI work
   - Real Mac required for camera + BLE

2. **Keep ESP32 QR Code Visible**
   - Display stays on after button press
   - Press button again to toggle back to clock

3. **One-Time Pairing**
   - Only need to scan QR once
   - Device name saved permanently
   - Can unpair/re-pair in Device Settings

4. **Debug Mode Available**
   - Simulator automatically uses text input
   - No code changes needed
   - Works on both simulator and real Mac

---

## ✅ You're All Set!

The camera setup is complete. To test:

1. **Flash ESP32** ✅ (Already done!)
2. **Build macOS app** in Xcode
3. **Run on "My Mac"** (not simulator)
4. **Grant camera permission**
5. **Scan QR code** from ESP32
6. **Start monitoring!**

Happy testing! 🚀
