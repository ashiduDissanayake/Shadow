# Quick Time Sync Testing Guide

## What to Look For in ESP32 Logs:

### ✅ Expected Logs on Boot:
```
I (xxx) TimeSync: Time synchronization system initialized
I (xxx) BLEPairing: Pairing service created (handle=XX)
I (xxx) BLEPairing: Characteristic added (UUID=0xB001, handle=XX)  ← Device Info
I (xxx) BLEPairing: Characteristic added (UUID=0xB002, handle=XX)  ← Pairing State
I (xxx) BLEPairing: Characteristic added (UUID=0xB003, handle=XX)  ← Pairing Control
I (xxx) BLEPairing: Characteristic added (UUID=0xB004, handle=XX)  ← Security Challenge
I (xxx) BLEPairing: Characteristic added (UUID=0xB005, handle=XX)  ← Time Sync (NEW!)
I (xxx) BLEPairing: All pairing characteristics added successfully (including time sync)
W (xxx) TimeSync: Time not synced, cannot get local time
I (xxx) DISPLAY: Clock display: Waiting for time sync...
```

### ✅ Expected Logs After macOS Connection:
```
I (xxx) BLEPairing: Client connected (conn_id=X)
I (xxx) BLEPairing: ⏰ Time synchronized!
I (xxx) BLEPairing:    Unix time: 1729467923000 ms
I (xxx) BLEPairing:    Local time: 2025-10-20 16:45:23
I (xxx) BLEPairing:    Timezone: UTC-7 hours
I (xxx) BLEPairing:    Boot time at sync: 12345678 us
I (xxx) BLEPairing: ✅ Time synchronized successfully
```

## What to Look For in macOS Logs:

### ✅ Expected Logs on Connection:
```
[HH:MM:SS] ADV seq=0 state=0 delta=127
[HH:MM:SS] Initial state detected (seq=0 state=0) -> connect & sync
[HH:MM:SS] Connecting reset=false syncTime=true delta=127
[HH:MM:SS] Connected -> discover services
[HH:MM:SS] 📡 Found Event characteristic (A002)
[HH:MM:SS] ⏰ Found Time Sync characteristic           ← NEW!
[HH:MM:SS] ⏰ Syncing time: 2025-10-20 16:45:23 (UTC-7.0)
[HH:MM:SS]    Unix: 1729467923000 ms, TZ offset: -25200 sec
```

## Display Testing:

### Before Sync:
- Display shows: `--:--` (placeholder)

### After Sync:
- Display shows: `16:45` (actual local time)
- Updates every minute

## Quick Test Steps:

1. **Flash firmware** (wait for boot logs)
   - Verify: Time sync characteristic 0xB005 appears
   - Verify: Display shows --:--

2. **Connect macOS app**
   - Verify: "⏰ Found Time Sync characteristic" in macOS logs
   - Verify: "⏰ Syncing time..." message in macOS logs
   - Verify: "✅ Time synchronized successfully" in ESP32 logs

3. **Check Display**
   - Press right button to show clock
   - Verify: Time is correct (matches your computer time)

4. **Generate Stress Event**
   - Calibrate if needed (press left button)
   - Wait for stress transition
   - Verify: Event logged with Unix timestamp in macOS CoreData

5. **Verify Timestamps**
   - Open macOS app → Settings/Debug view
   - Check stress events in database
   - Timestamps should be real Unix time (e.g., 1729467923000)
   - NOT boot time (e.g., 45000)

## Troubleshooting:

### ❌ Time Sync Characteristic (0xB005) Not Showing:
- **Cause:** Service handle count too small
- **Fix:** Increase `esp_ble_gatts_create_service(gatts_if, &service_id, 12)`
- **Verify:** Check build logs for increased service size

### ❌ Display Still Shows --:--:
- **Cause:** Time not synced yet
- **Check:** ESP32 logs for "Time synchronized successfully"
- **Check:** macOS logs for "Syncing time"
- **Try:** Reconnect macOS app

### ❌ Events Have Boot Time Instead of Real Time:
- **Cause:** Time sync happened but FSM using old timestamp
- **Check:** `time_sync_is_synced()` should return true
- **Verify:** Logs show "Using real-world Unix timestamp"

### ❌ macOS App Not Finding Time Sync Characteristic:
- **Cause:** Characteristic UUID mismatch
- **Verify:** ESP32 uses 0xB005, macOS discovers 0xB005
- **Check:** Service discovery logs in macOS

## Success Criteria:

✅ ESP32 logs show 5 characteristics added (B001, B002, B003, B004, B005)  
✅ macOS logs show "⏰ Found Time Sync characteristic"  
✅ macOS logs show "⏰ Syncing time..."  
✅ ESP32 logs show "✅ Time synchronized successfully"  
✅ Display shows correct local time (not --:--)  
✅ Stress events have Unix timestamps > 1,000,000,000,000  
✅ Timestamps match calendar event times in CoreData  

## What's Next After Testing:

Once time sync works:
1. Test notification timing with calendar events
2. Implement Gemini AI message generation
3. Test combined notifications (stress + event)
4. Add pattern analysis for recurring stress
5. Deploy to production! 🚀
