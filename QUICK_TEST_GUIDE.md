# Quick Testing Guide

## 1. QR Code Display (LONG PRESS)

**How to show QR code:**
1. Press and **HOLD** the right button for **2 seconds**
2. Release
3. QR code should appear

**Expected logs:**
```
I (xxxx) MAIN: 🔘 Long press detected (1XXX ms) - toggling QR code
I (xxxx) DISPLAY: QR Code displayed: Shadow-XXXX
```

**If not working:**
- Make sure you're holding for full 2 seconds
- Check if you see "Long press detected" in logs
- Button might need longer press (try 2.5 seconds)

## 2. Display Sleep/Wake (SHORT PRESS)

**Short press** = Toggle display on/off

**Expected logs:**
```
// When turning OFF:
I (xxxx) MAIN: 💤 Short press - display sleep
I (xxxx) DISPLAY: Display powered OFF

// When waking UP:
I (xxxx) MAIN: 💡 Short press - display wake
I (xxxx) DISPLAY: Display powered ON
```

## 3. Auto-Sleep Test

1. Wake display
2. Wait 30 seconds without touching
3. Should see:
```
I (xxxx) MAIN: 💤 Auto-sleep: Display idle for 30XXX ms, turning off
I (xxxx) DISPLAY: Display powered OFF
```

## 4. Time Sync Issue

**Problem:** Device not receiving time from macOS

**Diagnosis:**
```bash
# Check if macOS app is running and scanning
# Look for these logs in macOS:
[HH:MM:SS] ADV seq=0 state=0 delta=127
[HH:MM:SS] Initial state detected (seq=0 state=0) -> connect & sync
[HH:MM:SS] ⏰ Syncing time: 2025-10-20 HH:MM:SS
```

**Fix:**
1. **Make sure macOS Shadow app is running**
2. Check Bluetooth permissions
3. Device should advertise as "Shadow-XXXX"
4. macOS should auto-connect when it sees seq=0 state=0

**Expected ESP32 logs after sync:**
```
I (xxxx) BLEPairing: Client connected (conn_id=0)
I (xxxx) TimeSync: ⏰ Time synchronized!
I (xxxx) TimeSync:    Unix time: 1729467923000 ms
I (xxxx) TimeSync:    Local time: 2025-10-20 HH:MM:SS
I (xxxx) BLEPairing: ✅ Time synchronized successfully
```

**Then display will show:**
```
I (xxxx) DISPLAY: Clock display: HH:MM  // Real time!
```

## Troubleshooting:

### QR Code Not Showing:
- **Try holding button longer** (2-3 seconds)
- Watch serial monitor for "Long press detected"
- If you see "Short press" instead, you're not holding long enough

### Time Sync Not Working:
- **Is macOS app running?** Check Activity Monitor
- **Bluetooth enabled?** Check System Settings
- **App has permissions?** System Settings → Privacy → Bluetooth
- **Device visible?** Should see "Shadow-XXXX" in Bluetooth settings (don't pair manually!)

### Display Won't Wake:
- **Press button once** (not hold)
- Should see "💡 Short press - display wake"
- If no logs, button might be faulty

## Next Steps:

1. **Test long press QR code** - hold 2+ seconds
2. **Open macOS Shadow app** - it should auto-connect
3. **Verify time sync** - clock should show real time
4. **Then**: Build stress graph in macOS app
5. **Then**: Build notification system

## Button Summary:

| Action | Button | Duration | Result |
|--------|--------|----------|--------|
| Wake/Sleep | Right | <1.5s | Toggle display power |
| QR Code | Right | ≥1.5s | Toggle QR code display |
| Calibrate | Left | Any | Start/stop calibration |
| Auto-sleep | (none) | 30s idle | Display turns off |
