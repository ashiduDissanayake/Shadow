# Shadow Notification System - Implementation Plan

## Overview
Complete architecture for intelligent, context-aware notifications with time synchronization.

---

## 1. TIME SYNCHRONIZATION ARCHITECTURE ⏰

### Problem
- ESP32 uses `xTaskGetTickCount()` → milliseconds since boot
- macOS needs Unix timestamps → real-world time
- Display needs accurate time → user-visible clock

### Solution: 3-Layer Time Sync

```
┌─────────────────────────────────────────────────────────┐
│                  macOS App (Swift)                      │
│  ┌────────────────────────────────────────────────┐    │
│  │  On BLE Connection (delta > 32):               │    │
│  │  1. Get current Unix timestamp (ms)             │    │
│  │  2. Get timezone offset (seconds)               │    │
│  │  3. Send via BLE characteristic 0xB005          │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
                          │ BLE Write (12 bytes)
                          ▼
┌─────────────────────────────────────────────────────────┐
│               ESP32 Firmware (C)                        │
│  ┌────────────────────────────────────────────────┐    │
│  │  Time Sync Component                            │    │
│  │  - Receive Unix timestamp + timezone            │    │
│  │  - Calculate offset: unix_epoch_us - boot_us    │    │
│  │  - Convert future events: boot_time → unix_time │    │
│  └────────────────────────────────────────────────┘    │
│                          │                              │
│  ┌────────────────────────────────────────────────┐    │
│  │  Event Logging                                  │    │
│  │  OLD: timestamp = xTaskGetTickCount()           │    │
│  │  NEW: timestamp = time_sync_get_timestamp_ms()  │    │
│  └────────────────────────────────────────────────┘    │
│                          │                              │
│  ┌────────────────────────────────────────────────┐    │
│  │  Display (TFT)                                  │    │
│  │  - Show local time with timezone                │    │
│  │  - Update every second                          │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### BLE Time Sync Protocol

**Characteristic: 0xB005 (Time Sync)**
- **Properties**: WRITE
- **Payload**: 12 bytes
  ```
  Byte 0-7:  Unix timestamp (uint64_t, milliseconds)
  Byte 8-11: Timezone offset (int32_t, seconds from UTC)
  ```

**Example:**
```
macOS time: 2025-10-20 16:45:23 PDT (UTC-7)
Unix ms:    1729467923000
Timezone:   -25200 seconds (-7 hours)

BLE Write: [0x58, 0xB4, 0x5E, 0x92, 0x91, 0x01, 0x00, 0x00,  // timestamp
            0xD0, 0x9D, 0xFF, 0xFF]                           // timezone
```

---

## 2. NOTIFICATION DECISION ENGINE 🎯

### Rule-Based Timing Logic (Local, Fast, Privacy-Safe)

```swift
class NotificationDecisionEngine {
    
    // RULE 1: Calendar Event Timing
    func shouldNotifyForCalendarEvent(_ event: CalendarEvent) -> Decision {
        let timeUntilEvent = event.startTime.timeIntervalSinceNow
        let reminderOffset = event.reminderMinutes ?? 10 // User-adjustable
        
        // Check if it's time to remind
        guard timeUntilEvent <= (Double(reminderOffset) * 60) else {
            return .wait
        }
        
        // Get current stress state from BLE
        let stressState = getCurrentStressState() // CALM or STRESSED
        
        if stressState == .stressed {
            // DON'T interrupt during stress!
            return .delayUntilCalm(
                showMinimal: true,  // Optional: show silent notification
                event: event
            )
        } else {
            return .sendNow(
                message: "📅 \(event.title) in \(reminderOffset) min",
                priority: .normal
            )
        }
    }
    
    // RULE 2: Stress Episode Ended
    func shouldNotifyForStressEpisode(_ episode: StressEpisode) -> Decision {
        // NEVER notify during stress
        if getCurrentStressState() == .stressed {
            return .wait
        }
        
        // Analyze episode duration
        let duration = episode.duration
        
        if duration < 5.minutes {
            // Short episode - minimal intervention
            return .sendNow(
                message: "You seem calmer now 😌",
                priority: .low
            )
        }
        else if duration < 15.minutes {
            // Medium episode - check for patterns
            let pattern = detectPattern(episode)
            
            if pattern.isRecurring {
                // Recurring stress → Use AI for context
                return .generateWithAI(
                    context: pattern,
                    fallback: "That was stressful. Want to take a break? 🌿"
                )
            } else {
                return .sendNow(
                    message: "Time for a short break? 🌿",
                    priority: .medium
                )
            }
        }
        else {
            // Long episode (>15 min) → Deep intervention with AI
            return .generateWithAI(
                context: episode,
                fallback: "You've been stressed for a while. Let's take a proper break 💙"
            )
        }
    }
    
    // RULE 3: Pattern Detection
    func detectPattern(_ episode: StressEpisode) -> StressPattern {
        // Analyze past week of episodes
        let recentEpisodes = fetchRecentEpisodes(days: 7)
        
        // Check for recurring patterns
        let timePattern = analyzeTimePattern(episodes: recentEpisodes)
        // e.g., "Stressed every Monday 2-3pm"
        
        let durationPattern = analyzeDurationPattern(episodes: recentEpisodes)
        // e.g., "Episodes getting longer each day"
        
        return StressPattern(
            isRecurring: timePattern.confidence > 0.7,
            timeOfDay: timePattern.mostCommonHour,
            dayOfWeek: timePattern.mostCommonDay,
            averageDuration: durationPattern.mean,
            trend: durationPattern.trend // .increasing, .stable, .decreasing
        )
    }
}
```

---

## 3. GEMINI AI INTEGRATION 🤖

### Gemini 2.0 Flash Configuration

```swift
import GoogleGenerativeAI

class NotificationAIGenerator {
    private let apiKey = "YOUR_GEMINI_API_KEY" // Store in Keychain!
    private let model: GenerativeModel
    
    init() {
        // Use Gemini 2.0 Flash (fast, cost-effective)
        model = GenerativeModel(
            name: "gemini-2.0-flash-exp",
            apiKey: apiKey,
            generationConfig: GenerationConfig(
                temperature: 0.7,          // Balanced creativity
                topK: 40,
                topP: 0.95,
                maxOutputTokens: 100,      // Keep messages short
                stopSequences: ["\n\n"]
            )
        )
    }
    
    // Generate context-aware notification message
    func generateMessage(
        for episode: StressEpisode,
        pattern: StressPattern? = nil
    ) async throws -> String {
        
        // Build context from local data
        let context = buildContext(episode: episode, pattern: pattern)
        
        // Structured prompt for Gemini
        let prompt = """
        You are a compassionate wellness assistant for a stress monitoring app.
        
        User Context:
        \(context)
        
        Generate a brief notification message (max 100 characters) that:
        1. Acknowledges the stress without being alarmist
        2. Suggests ONE specific, actionable coping strategy
        3. Is encouraging but not patronizing
        4. Uses casual, friendly tone
        
        DO NOT include:
        - Medical advice
        - Multiple suggestions (only ONE action)
        - Questions (use statements)
        - Emojis (we'll add them)
        
        Message:
        """
        
        do {
            let response = try await model.generateContent(prompt)
            let message = response.text?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .prefix(100) ?? ""
            
            return String(message)
        } catch {
            // Fallback to template on error
            print("Gemini API error: \(error)")
            throw error
        }
    }
    
    private func buildContext(episode: StressEpisode, pattern: StressPattern?) -> String {
        var context = """
        Stress Episode:
        - Duration: \(Int(episode.duration / 60)) minutes
        - Time: \(formatTime(episode.startTime))
        - Peak stress level: \(Int(episode.peakProbability * 100))%
        """
        
        if let pattern = pattern, pattern.isRecurring {
            context += """
            
            Recurring Pattern Detected:
            - Happens \(pattern.dayOfWeek ?? "regularly") around \(pattern.timeOfDay ?? "this time")
            - Average duration: \(Int(pattern.averageDuration / 60)) minutes
            - Trend: \(pattern.trend.description)
            """
        }
        
        // Add recent history
        let todayCount = getEpisodesCount(today: true)
        if todayCount > 1 {
            context += "\n- This is episode #\(todayCount) today"
        }
        
        return context
    }
}
```

### Fallback Mechanism (No Internet)

```swift
class NotificationGenerator {
    private let aiGenerator = NotificationAIGenerator()
    
    func generateMessage(for episode: StressEpisode) async -> String {
        // Try AI first
        do {
            let aiMessage = try await aiGenerator.generateMessage(for: episode)
            return addEmoji(aiMessage)
        } catch {
            // Fallback to smart templates
            return getTemplateMessage(for: episode)
        }
    }
    
    private func getTemplateMessage(for episode: StressEpisode) -> String {
        let duration = episode.duration
        
        switch duration {
        case ..<(5 * 60):
            return "That was intense! You're calmer now 😌"
        case ..<(15 * 60):
            return "Time for a breather? Try a short walk 🚶"
        case ..<(30 * 60):
            return "That was a long stretch. How about a proper break? ☕"
        default:
            return "You've been stressed for a while. Let's reset together 💙"
        }
    }
    
    private func addEmoji(_ message: String) -> String {
        // Add appropriate emoji based on message content
        if message.contains("walk") || message.contains("move") {
            return message + " 🚶"
        } else if message.contains("breath") {
            return message + " 🫁"
        } else if message.contains("break") || message.contains("rest") {
            return message + " ☕"
        } else {
            return message + " 💙"
        }
    }
}
```

---

## 4. IMPLEMENTATION PHASES

### Phase 1: Time Synchronization (PRIORITY 🔴)
**Files to modify:**
1. ✅ `shadow-firmware/components/time_sync/*` (CREATED)
2. ⏳ `components/ble_stress_service/ble_pairing.c` (ADD TIME_SYNC characteristic)
3. ⏳ `main/main_realtime.c` (USE time_sync_get_timestamp_ms() for events)
4. ⏳ `Shadow/Shadow/Features/BLE/LightShadowBLEManager.swift` (SEND time on connect)
5. ⏳ Display integration (sync RTC)

**Testing:**
- [ ] ESP32 receives time from macOS
- [ ] Events logged with real Unix timestamps
- [ ] Display shows correct local time
- [ ] Timestamps match between ESP32 and macOS

### Phase 2: Notification Decision Engine (1-2 days)
**Files to create:**
1. `Shadow/Shadow/Features/Notifications/NotificationDecisionEngine.swift`
2. `Shadow/Shadow/Features/Notifications/NotificationScheduler.swift`
3. `Shadow/Shadow/Features/Notifications/StressPatternAnalyzer.swift`

**Integration:**
- Monitor calendar events
- Monitor stress state changes
- Apply timing rules
- Queue notifications

### Phase 3: Gemini AI Integration (1 day)
**Files to create:**
1. `Shadow/Shadow/Features/AI/GeminiClient.swift`
2. `Shadow/Shadow/Features/AI/NotificationAIGenerator.swift`
3. Add API key management (Keychain)

**Testing:**
- [ ] AI generates contextual messages
- [ ] Fallback works without internet
- [ ] API usage within free tier limits

### Phase 4: Polish & Testing (1-2 days)
- User preferences UI
- Notification history
- Effectiveness metrics
- A/B testing different messages

---

## 5. NEXT STEPS

**Immediate Action:**
1. ✅ Time sync component created
2. ⏳ Add BLE characteristic 0xB005 to pairing service
3. ⏳ Modify event logging to use real timestamps
4. ⏳ Add time sync in macOS BLE manager

**Should I proceed with:**
A) Finish time synchronization implementation (firmware + macOS)?
B) Show you the notification engine code first?
C) Both in parallel?

Let me know and I'll continue! 🚀
