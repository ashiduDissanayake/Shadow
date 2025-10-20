# Time Sync Not Working - Diagnostic Guide

## Status:
✅ Device paired: "Shadow-9026" stored in macOS app  
✅ Device advertising: seq=0 state=0 (initial state)  
✅ Time sync characteristic: 0xB005 registered  
❌ macOS NOT connecting automatically  
❌ Time NOT syncing  

## Problem:
macOS app should auto-connect when it sees seq=0 state=0, but it's not connecting.

## Possible Causes:

### 1. macOS App Not Scanning
**Check**: Is the Shadow app actually running and scanning?

**Open macOS app and check:**
- App should show "Scanning..." status
- Should see device in the UI
- Check console logs for "Scanning for Shadow-9026..."

### 2. Bluetooth Permission Issue
**macOS Bluetooth permissions might have been revoked**

**Fix**:
```bash
# Check System Settings:
System Settings → Privacy & Security → Bluetooth
# Make sure Shadow app has permission
```

### 3. App Not in Foreground
**macOS might suspend background scanning**

**Fix**:
- Bring Shadow app to foreground
- Keep app window visible

### 4. Central Manager Not Ready
**Bluetooth might not be powered on when app starts**

**Check macOS app logs for:**
```
"Bluetooth not powered on"
```

## Quick Fix Steps:

### Step 1: Restart macOS Shadow App
```bash
# Kill and restart
killall Shadow
# Then reopen Shadow app
```

### Step 2: Check App Logs
Look for these logs when app starts:
```
"Manager init, lastKnownSequence=X"
"Scanning for Shadow-9026..."
"ADV seq=0 state=0 delta=127"
"Initial state detected (seq=0 state=0) -> connect & sync"
```

### Step 3: Force Re-scan
In macOS app:
1. Stop scanning (if running)
2. Wait 2 seconds
3. Start scanning again
4. Should see "Shadow-9026" appear
5. Should auto-connect when delta > 0

### Step 4: Check Pairing Status
```bash
# Open macOS Terminal and run:
defaults read com.yourcompany.Shadow PairedShadowDevice
# Should return: Shadow-9026
```

If not found, you need to re-pair by scanning QR code.

## Expected Flow (What SHOULD Happen):

```
ESP32 boots → seq=0 state=0
      ↓
Advertises: "Shadow-9026" with seq=0 state=0
      ↓
macOS app scanning → discovers peripheral
      ↓
Checks: peripheral.name == "Shadow-9026" ✓
      ↓
Calculates delta = modularDelta(old: 5, new: 0) = large gap
      ↓
Detects: isInitialState = (seq==0 && state==0) ✓
      ↓
Triggers: connect(reset: false, syncTime: true)
      ↓
Connects → discovers services
      ↓
Finds 0xB005 time sync characteristic
      ↓
Sends 12-byte time sync payload
      ↓
ESP32 logs: "⏰ Time synchronized!"
      ↓
Display shows: Real time (HH:MM)
```

## What's Probably Happening:

**macOS app is NOT seeing the advertisement at all**

Reasons:
1. App not scanning (not started, or stopped)
2. Bluetooth permission denied
3. Central manager in wrong state
4. App filter rejecting the peripheral

## Debug Steps:

### Add Logging to macOS App

I'll add verbose logging to see exactly what's happening in the didDiscover callback.

### Check if Advertisement is Being Received

Look for this in macOS app console:
```
"Discovered peripheral: Shadow-9026"
```

If you DON'T see this, macOS isn't receiving advertisements at all.

### Check UserDefaults

```swift
// In macOS app, print:
print("Paired device: \(UserDefaults.standard.string(forKey: "PairedShadowDevice") ?? "none")")
```

Should print: "Shadow-9026"

## Temporary Workaround:

### Re-pair the Device
1. In macOS app: Unpair device (if option exists)
2. Scan QR code on device (long press right button for 2 seconds)
3. macOS app should discover and pair
4. Time sync should work

## Next: Add Debug Logging

I'll add detailed logging to the macOS app's `didDiscover` callback to see why it's not connecting.
