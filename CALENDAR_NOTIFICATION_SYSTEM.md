# 📅 Calendar + Stress Notification System

## Overview

The calendar notification system monitors upcoming events and sends **stress-aware notifications** that adapt based on your current stress level.

---

## How It Works

### 1. **Calendar Event Monitor** (`CalendarEventMonitor.swift`)

**Runs in background** checking every 30 seconds:
- Fetches upcoming events (within next hour)
- Gets current stress state from BLE device
- Decides when and how to notify based on both

**Key Features:**
- ✅ Prevents duplicate notifications (tracks notified events)
- ✅ Stress-aware messaging
- ✅ Adaptive timing based on urgency
- ✅ Automatically starts when app launches

### 2. **Notification Logic**

#### When You're **CALM** 😌:
```
📅 Upcoming: Team Meeting
Starts in 10 minutes
```
- Standard notification
- Default sound
- Simple, informational

#### When You're **STRESSED** 😰:

**IMMINENT (< 2 min):**
```
⚠️ Team Meeting in 2 min
You're stressed. Take 30 seconds to breathe before heading there.
```
- **Critical sound** (urgent)
- Acknowledgment + quick intervention

**SOON (2-5 min):**
```
🌿 Quick Break Before Team Meeting
Starts in 5 min. Short walk or breathing exercise?
```
- Suggestion for stress break
- Default sound
- Actionable advice

**MORE TIME (5-10 min):**
```
Heads up: Team Meeting in 8 min
You've been stressed. How about a breather before your event?
```
- Gentle reminder
- Encourages self-care
- Default sound

---

## Configuration

### Default Settings:
- **Monitoring interval**: Every 30 seconds
- **Reminder time**: 10 minutes before event
- **Stress detection window**: Last 5 minutes
- **Upcoming events window**: Next 1 hour

### To Change Reminder Time:

Edit line 89 in `CalendarEventMonitor.swift`:
```swift
let reminderMinutes = 10.0  // Change to 5, 15, 20, etc.
```

### To Change Check Frequency:

Edit line 31 in `CalendarEventMonitor.swift`:
```swift
timer = Timer.scheduledTimer(withTimeInterval: 30.0, ...)  // Change 30.0 to desired seconds
```

---

## Current Stress State Detection

The monitor checks the **most recent stress event** from your wearable:

```swift
private func getCurrentStressState() -> StressState {
    // Get latest stress event
    let recentEvents = stressRepo.recentEvents(limit: 1)
    
    // Must be within last 5 minutes to be considered "current"
    // Older than 5 min = assumed calm
    return latestEvent.stressState == 1 ? .stressed : .calm
}
```

**Why 5 minutes?**
- Prevents stale data (e.g., stressed 30 min ago doesn't mean stressed now)
- Balances responsiveness with accuracy
- Can be adjusted if needed

---

## Integration Points

### 1. **App Startup** (`Shadow.swift`)
```swift
func applicationDidFinishLaunching() {
    CalendarEventMonitor.shared.startMonitoring()
}
```
Automatically starts monitoring when app launches.

### 2. **Event Creation** (Future Enhancement)
When user creates an event in calendar, you can:
```swift
func addEvent(...) {
    // Save event to CoreData
    eventRepo.save(event)
    
    // Monitor will automatically detect it in next check cycle (30s)
    // OR force immediate check:
    Task {
        await CalendarEventMonitor.shared.checkUpcomingEvents()
    }
}
```

### 3. **Stress State Updates** (Already Connected)
Monitor reads from `StressDataRepository.shared` which is updated by:
- `LightShadowBLEManager` (BLE stress events)
- `StressTransitionDomainEvent` (domain model)

---

## Testing

### Test Scenario 1: **Calm User**
1. Create event 10 minutes from now
2. Keep device in calm state (state=0)
3. Wait for notification
4. **Expected**: Standard "📅 Upcoming: [Title]" notification

### Test Scenario 2: **Stressed User (More Time)**
1. Create event 10 minutes from now
2. Trigger stress on device (state=1)
3. Wait for notification (will come within 30s of 10-min mark)
4. **Expected**: "Heads up: [Title] in 10 min. You've been stressed..."

### Test Scenario 3: **Stressed User (Imminent)**
1. Create event 2 minutes from now
2. Trigger stress on device (state=1)
3. Wait for notification
4. **Expected**: "⚠️ [Title] in 2 min" with **critical sound**

### Debug Console Output:
```
📅 [CalendarMonitor] Starting event monitoring...
📅 [CalendarMonitor] Found 1 upcoming events in next hour
📅 [CalendarMonitor] Event 'Team Meeting' in 10 minutes, stress=stressed
📅 [CalendarMonitor] ✅ Notification sent: Heads up: Team Meeting in 10 min
```

---

## Why Your Notification Didn't Show

### Possible Reasons:

1. **Event Not in CoreData**
   - Check: Does the event show in calendar view?
   - Fix: Make sure event was saved to CoreData

2. **Monitoring Not Started**
   - Check console for: `✅ Calendar event monitoring started`
   - Fix: Rebuild app, monitoring starts at launch

3. **Event Time Already Passed**
   - Monitor only notifies for **future** events
   - Must be within 10 minutes but > 0 minutes

4. **Event More Than 1 Hour Away**
   - Default window is 1 hour
   - Events further out won't be checked yet
   - Will be checked when they enter 1-hour window

5. **Already Notified**
   - Each event notified only once
   - Duplicate prevention active

### Quick Fix:
1. Restart app (monitoring restarts)
2. Create new event **exactly 10 minutes from now**
3. Check console for monitoring logs
4. Should see notification within 30 seconds

---

## Future Enhancements

### Planned Features:
- [ ] User-configurable reminder times per event
- [ ] Multiple reminders (e.g., 15 min + 5 min before)
- [ ] Integration with system calendar (EventKit)
- [ ] Snooze functionality
- [ ] "Busy now" mode (delay non-urgent notifications)
- [ ] ML-based optimal notification timing
- [ ] Stress trend analysis (predict best times)

### Advanced Stress Integration:
- [ ] "Pre-meeting stress prep" (breathing exercises)
- [ ] Post-event stress recovery suggestions
- [ ] Context-aware: meeting type → suggestion type
- [ ] Calendar blocking for stress recovery time

---

## Technical Architecture

```
┌─────────────────────────────────────────────┐
│         App Launch (Shadow.swift)           │
│  CalendarEventMonitor.shared.startMonitoring()│
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│      Timer (every 30 seconds)               │
│     checkUpcomingEvents()                   │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
┌──────────────────┐  ┌──────────────────┐
│  EventRepository │  │ StressDataRepo   │
│  (CoreData)      │  │ (BLE Events)     │
│ - Event.date     │  │ - stressState    │
│ - Event.title    │  │ - timestamp      │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  buildNotification() │
         │  (stress-aware)      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  UNNotificationCenter│
         │  (macOS notification)│
         └──────────────────────┘
```

---

## Summary

✅ **System is now active and monitoring**
✅ **Checks every 30 seconds for upcoming events**
✅ **Adapts notifications based on stress state**
✅ **Prevents duplicate notifications**
✅ **Critical sounds for imminent events during stress**

**Next time you create an event 10 minutes from now, you'll get a notification!** 🎉

If you're stressed (state=1), it'll say:
> "Heads up: [Event] in 10 min. You've been stressed. How about a breather before your event?"

If you're calm (state=0), it'll say:
> "📅 Upcoming: [Event]. Starts in 10 minutes"
