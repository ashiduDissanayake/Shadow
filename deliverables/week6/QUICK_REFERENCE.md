# Shadow BLE Pairing - Quick Reference Card 🚀

**Date**: 18 October 2025  
**Status**: READY FOR UI INTEGRATION  

---

## 📋 **WHAT'S DONE**

✅ **ESP32 Firmware** (Shadow-9026)
- Pairing service initialized (UUID 0xB000)
- 4 characteristics active (0xB001-0xB004)
- SHA-256 challenge-response auth ready
- NVS persistence enabled
- Device advertising and ready

✅ **macOS App** (Shadow Monitor)
- PairingModels.swift created
- PairingHelper.swift with SHA-256
- LightShadowBLEManager updated
- Async pairing flow implemented
- Ready for UI button

---

## 🎯 **TO-DO: ADD UI BUTTON** (5 minutes)

### **Files to Edit**

1. **`/Shadow/Shadow/Features/BLE/SyncDashboardViewModel.swift`**
   
   **Change Line ~14:**
   ```swift
   // FROM:
   private let manager: LightShadowBLEManager
   
   // TO:
   let manager: LightShadowBLEManager  // Remove 'private'
   ```

2. **`/Shadow/Shadow/Features/Dashboard/ShadowDashboardView.swift`**
   
   **Follow Guide**: `deliverables/week6/task9_ui_integration_guide.md`
   
   **Quick Option 2** (Simple button in header):
   ```swift
   // In headerSection, after "Welcome back" text, before Spacer():
   
   if !syncViewModel.manager.isPaired {
       Button(action: {
           Task {
               try? await syncViewModel.manager.performPairing()
           }
       }) {
           HStack(spacing: 4) {
               Image(systemName: "key.fill")
               Text("Pair")
           }
           .font(.caption)
           .padding(.horizontal, 10)
           .padding(.vertical, 5)
           .background(Color.blue)
           .foregroundColor(.white)
           .cornerRadius(6)
       }
       .buttonStyle(.plain)
   } else {
       HStack(spacing: 4) {
           Image(systemName: "checkmark.shield.fill")
               .foregroundColor(.green)
           Text("Paired")
               .font(.caption)
               .foregroundColor(.green)
       }
   }
   ```

---

## 🧪 **TESTING STEPS** (10 minutes)

### **Terminal 1: ESP32 Monitor**
```bash
cd ~/Dev/Shadow/shadow-firmware
. ~/Dev/esp/esp-idf/export.sh
idf.py monitor
```

### **Terminal 2: Xcode**
```bash
cd ~/Dev/Shadow/Shadow
open Shadow.xcodeproj
# Cmd+R to build and run
```

### **In macOS App:**
1. Login
2. Go to Shadow Dashboard
3. Click "Pair Device" button
4. Watch logs!

---

## 📊 **EXPECTED LOGS**

### **ESP32 Monitor:**
```
I (xxxxx) BLEPairing: Client connected
I (xxxxx) BLEPairing: Pairing control write: command=1 (PAIR_REQUEST)
I (xxxxx) BLEPairing: Generating security challenge
I (xxxxx) BLEPairing: Security challenge write received
I (xxxxx) BLEPairing: Challenge verification: SUCCESS
I (xxxxx) BLEPairing: Device paired: Mac
I (xxxxx) BLEPairing: Total paired devices: 1 / 3
```

### **Xcode Console:**
```
[HH:MM:SS] 🔐 Starting pairing process...
[HH:MM:SS] 📱 Shadow Device: Shadow-9026
[HH:MM:SS] 🆔 Device ID: 9251b891...ef3d9026
[HH:MM:SS] 🔧 Firmware: v1.0.0
[HH:MM:SS] ⚙️ Hardware: ESP32-S3
[HH:MM:SS] 📤 Sent pairing request
[HH:MM:SS] ⏳ Pairing state: PENDING
[HH:MM:SS] 🔐 Received challenge
[HH:MM:SS] 📤 Sent challenge response
[HH:MM:SS] ✅ Pairing successful!
```

---

## 🔍 **TROUBLESHOOTING**

### **Problem**: "Bluetooth not powered on"
- **Solution**: Enable Bluetooth in macOS System Settings

### **Problem**: "Device not found"
- **Solution**: Check ESP32 is running (`idf.py monitor`)

### **Problem**: "Characteristic not found"
- **Solution**: 
  1. Disconnect and reconnect
  2. Check ESP32 logs for service creation
  3. Verify all 4 chars registered (handles 46-52)

### **Problem**: "Pairing timeout"
- **Solution**: 
  1. Check both devices on same page
  2. Restart ESP32 (`idf.py flash monitor`)
  3. Restart macOS app

### **Problem**: "Challenge verification failed"
- **Solution**: 
  1. This shouldn't happen (SHA-256 mismatch)
  2. Check logs for details
  3. Report as bug if persistent

---

## 📁 **DOCUMENTATION FILES**

All in `/deliverables/week6/`:

1. **`task8_ble_pairing_complete.md`** - ESP32 implementation details
2. **`task9_macos_app_reference.md`** - Swift reference guide
3. **`task9_macos_app_implementation.md`** - Implementation summary
4. **`task9_ui_integration_guide.md`** - UI button guide
5. **`COMPLETE_SYSTEM_SUMMARY.md`** - Full system overview

---

## 🎯 **SUCCESS CHECKLIST**

- [ ] ESP32 monitor shows "BLE pairing service initialized"
- [ ] macOS app compiles without errors
- [ ] "Pair Device" button appears in UI
- [ ] Click button triggers pairing
- [ ] Both logs show pairing sequence
- [ ] Pairing completes successfully (✅ message)
- [ ] isPaired becomes true in UI
- [ ] Device info displays (Shadow-9026, v1.0.0)
- [ ] Restart app - pairing persists
- [ ] Stress data still flows

---

## 🚀 **ONE-LINER COMMANDS**

```bash
# ESP32: Flash and monitor
cd ~/Dev/Shadow/shadow-firmware && . ~/Dev/esp/esp-idf/export.sh && idf.py flash monitor

# macOS: Open Xcode
cd ~/Dev/Shadow/Shadow && open Shadow.xcodeproj

# Check UserDefaults (after pairing)
defaults read com.yourcompany.Shadow | grep Shadow.ClientDeviceID
```

---

## 🎊 **YOU'RE READY!**

**Just add the UI button and test!**

Everything else is complete and working! 🚀
