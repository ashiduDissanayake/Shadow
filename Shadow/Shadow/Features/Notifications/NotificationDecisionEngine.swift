//
//  NotificationDecisionEngine.swift
//  Shadow
//
//  Intelligent notification timing based on stress state and event priority
//

import Foundation
import UserNotifications
import CoreData

/// Handles intelligent notification timing decisions
@MainActor
class NotificationDecisionEngine: ObservableObject {
    
    static let shared = NotificationDecisionEngine()
    
    // Track stress duration and last notification time
    private var stressStartTime: Date?
    private var lastNotificationTime: Date?
    private var lastNotifiedState: Int16?
    
    // Configuration
    private let stressDurationThreshold: TimeInterval = 30   // 30 seconds for testing (production: 600)
    private let notificationCooldown: TimeInterval = 30      // 30 seconds for testing (production: 300)

    // MARK: - Decision Logic
    
    /// Decide when and how to notify for a calendar event
    func shouldNotifyForEvent(_ event: CalendarEvent, currentStressState: StressState) -> NotificationDecision {
        let timeUntilEvent = event.startTime.timeIntervalSinceNow
        let reminderMinutes = event.reminderMinutes ?? UserPreferences.shared.defaultReminderMinutes
        let reminderTime = Double(reminderMinutes) * 60
        
        // Not yet time to remind
        guard timeUntilEvent <= reminderTime else {
            return .wait
        }
        
        // Check stress state
        switch currentStressState {
        case .calm:
            // Perfect timing - user is calm
            return .sendNow(
                title: "📅 Upcoming: \(event.title)",
                body: "Starts in \(reminderMinutes) minutes",
                sound: .default,
                priority: .normal
            )
            
        case .stressed:
            // USER IS STRESSED - Use hybrid notification approach
            
            // Calculate urgency based on time remaining
            let urgencyLevel = calculateUrgency(timeRemaining: timeUntilEvent)
            
            switch urgencyLevel {
            case .low:  // > 5 minutes
                // Delay until calm, but set a backup timer
                return .delayUntilCalm(
                    showMinimal: true,  // Show minimal notification anyway
                    maxDelay: 300,      // Maximum 5 minutes delay
                    fallbackDecision: .sendCombined(event: event)
                )
                
            case .medium:  // 2-5 minutes
                // Send combined notification (stress break + event reminder)
                return .sendCombined(event: event)
                
            case .high:  // < 2 minutes
                // Event is imminent - MUST notify even during stress
                return .sendNow(
                    title: "⚠️ \(event.title) starting soon!",
                    body: "In \(Int(timeUntilEvent / 60)) minutes. Take a moment to breathe first.",
                    sound: .defaultCritical,  // Critical sound
                    priority: .high
                )
            }
        }
    }
    
    /// Decide when and how to notify for stress episode end
    func shouldNotifyForStressEpisode(_ episode: StressEpisode, upcomingEvents: [CalendarEvent]) -> NotificationDecision {
        // NEVER notify during active stress
        guard episode.hasEnded else {
            return .wait
        }
        
        // Check if there are upcoming events within 15 minutes
        let soonEvents = upcomingEvents.filter { $0.startTime.timeIntervalSinceNow < 15 * 60 }
        
        if !soonEvents.isEmpty {
            // Combine stress recovery message with event reminder
            let nextEvent = soonEvents.first!
            let minutesUntil = Int(nextEvent.startTime.timeIntervalSinceNow / 60)
            
            return .sendNow(
                title: "You're calmer now 😌",
                body: "Take a moment to reset. \(nextEvent.title) starts in \(minutesUntil) min.",
                sound: .default,
                priority: .medium
            )
        }
        
        // No upcoming events - focus on stress recovery
        let duration = episode.duration
        
        if duration < 5 * 60 {
            // Short episode
            return .sendNow(
                title: "You seem calmer now 😌",
                body: nil,
                sound: .default,
                priority: .low
            )
        } else if duration < 15 * 60 {
            // Medium episode
            return .generateWithAI(
                context: episode,
                fallback: .sendNow(
                    title: "Time for a break? 🌿",
                    body: "You've been stressed for \(Int(duration / 60)) minutes",
                    sound: .default,
                    priority: .medium
                )
            )
        } else {
            // Long episode
            return .generateWithAI(
                context: episode,
                fallback: .sendNow(
                    title: "Let's take a proper break 💙",
                    body: "You've been stressed for quite a while",
                    sound: .default,
                    priority: .high
                )
            )
        }
    }
    
    // MARK: - Helper Functions
    
    private func calculateUrgency(timeRemaining: TimeInterval) -> UrgencyLevel {
        let minutes = timeRemaining / 60
        if minutes < 2 {
            return .high
        } else if minutes < 5 {
            return .medium
        } else {
            return .low
        }
    }

    // MARK: - Runtime evaluation API with proactive patterns
    /// Evaluates stress events with 10-min threshold and calendar awareness
    func evaluate(event: StressEvent) {
        print("🔔 [NotificationEngine] ========================================")
        print("🔔 [NotificationEngine] Evaluating event: seq=\(event.sequenceNumber), state=\(event.stressState)")
        print("🔔 [NotificationEngine] Last notified state: \(lastNotifiedState?.description ?? "nil")")
        print("🔔 [NotificationEngine] Stress start time: \(stressStartTime?.description ?? "nil")")
        
        let currentState = event.stressState
        
        // Track stress duration
        if currentState == 1 { // Stressed
            if stressStartTime == nil {
                stressStartTime = Date()
                print("🔔 [NotificationEngine] ⏱️ Stress period started at \(Date())")
            } else {
                let duration = Date().timeIntervalSince(stressStartTime!)
                print("🔔 [NotificationEngine] ⏱️ Stress ongoing for \(Int(duration/60)) minutes")
            }
        } else { // Calm
            if stressStartTime != nil {
                let duration = Date().timeIntervalSince(stressStartTime!)
                print("🔔 [NotificationEngine] ✅ Stress period ended after \(Int(duration/60)) minutes")
                stressStartTime = nil
            }
        }
        
        // PATTERN 1: Prolonged Stress (10+ minutes) → Suggest break
        if currentState == 1, let startTime = stressStartTime {
            let duration = Date().timeIntervalSince(startTime)
            print("🔔 [NotificationEngine] Checking threshold: \(Int(duration))s / \(Int(stressDurationThreshold))s")
            
            if duration >= stressDurationThreshold {
                print("🔔 [NotificationEngine] ⚠️ THRESHOLD REACHED!")
                // Check cooldown to avoid spam
                if shouldSendNotification() {
                    print("🔔 [NotificationEngine] ✅ Cooldown OK - Sending prolonged stress notification")
                    sendProlongedStressNotification(event: event)
                    return
                } else {
                    print("🔔 [NotificationEngine] ⏸️ Cooldown active - Skipping notification")
                }
            }
        }
        
        // PATTERN 2: State Transitions (stressed → calm)
        if let lastState = lastNotifiedState {
            if currentState != lastState {
                if currentState == 0 { // Recovered to calm
                    print("🔔 [NotificationEngine] 🎉 Recovery detected (1→0)")
                    if shouldSendNotification() {
                        print("🔔 [NotificationEngine] ✅ Cooldown OK - Sending recovery notification")
                        sendRecoveryNotification(event: event)
                    } else {
                        print("🔔 [NotificationEngine] ⏸️ Cooldown active - Skipping recovery notification")
                    }
                }
                lastNotifiedState = currentState
            }
        } else {
            lastNotifiedState = currentState
        }
        
        print("🔔 [NotificationEngine] ========================================")
    }
    
    /// Check upcoming calendar events with stress-aware logic
    func checkCalendarEvents(events: [Event], currentStressState: Int) {
        print("📅 [NotificationEngine] Checking \(events.count) calendar events, stress state: \(currentStressState)")
        let now = Date()
        
        for event in events {
            guard let eventDate = event.date else { continue }
            let timeUntilEvent = eventDate.timeIntervalSince(now)
            
            // PATTERN 3: 60-min evacuation (stressed + upcoming event) - Extended for testing
            if currentStressState == 1 && timeUntilEvent > 0 && timeUntilEvent <= 3600 { // 60 min
                print("📅 [NotificationEngine] Found stressed + event in \(Int(timeUntilEvent/60)) min")
                if shouldSendNotification() {
                    print("📅 [NotificationEngine] 🚨 Sending evacuation notification")
                    sendEvacuationNotification()
                    return
                } else {
                    print("📅 [NotificationEngine] ⏸️ Cooldown active - Skipping evacuation")
                }
            }
            
            // PATTERN 4: 60-min reminder (calm + upcoming event) - Extended for testing
            if currentStressState == 0 && timeUntilEvent > 0 && timeUntilEvent <= 3600 { // 60 min
                print("📅 [NotificationEngine] Found calm + event in \(Int(timeUntilEvent/60)) min")
                if shouldSendNotification() {
                    print("📅 [NotificationEngine] 📅 Sending event reminder")
                    sendEventReminder(event: event, minutesUntil: Int(timeUntilEvent/60))
                    return
                } else {
                    print("📅 [NotificationEngine] ⏸️ Cooldown active - Skipping reminder")
                }
            }
        }
    }
    
    // MARK: - Notification Senders
    
    private func sendProlongedStressNotification(event: StressEvent) {
        Task {
            print("🤖 [NotificationEngine] Calling AI for prolonged stress message...")
            let body = await AIDecisionProvider.shared.message(for: event)
            print("🤖 [NotificationEngine] AI generated: '\(body)'")
            
            let content = UNMutableNotificationContent()
            content.title = ""  // Empty title - only AI body
            content.body = body
            content.sound = .default
            
            let req = UNNotificationRequest(identifier: "stress-prolonged-\(Date().timeIntervalSince1970)", content: content, trigger: nil)
            try? await UNUserNotificationCenter.current().add(req)
            
            lastNotificationTime = Date()
            print("✅ [NotificationEngine] Prolonged stress notification sent!")
        }
    }
    
    private func sendRecoveryNotification(event: StressEvent) {
        Task {
            print("🤖 [NotificationEngine] Calling AI for recovery message...")
            let body = await AIDecisionProvider.shared.message(for: event)
            print("🤖 [NotificationEngine] AI generated: '\(body)'")
            
            let content = UNMutableNotificationContent()
            content.title = ""  // Empty title - only AI body
            content.body = body
            content.sound = .default
            
            let req = UNNotificationRequest(identifier: "recovery-\(Date().timeIntervalSince1970)", content: content, trigger: nil)
            try? await UNUserNotificationCenter.current().add(req)
            
            lastNotificationTime = Date()
            print("✅ [NotificationEngine] Recovery notification sent!")
        }
    }
    
    private func sendEvacuationNotification() {
        Task {
            print("🤖 [NotificationEngine] Preparing evacuation notification...")
            // Get recent stressed event for AI-generated break suggestion
            let recentEvents = StressDataRepository.shared.recentEvents(limit: 5)
            if let stressedEvent = recentEvents.first(where: { $0.stressState == 1 }) {
                print("🤖 [NotificationEngine] Calling AI for evacuation message (using recent stressed event)...")
                let body = await AIDecisionProvider.shared.message(for: stressedEvent)
                print("🤖 [NotificationEngine] AI generated: '\(body)'")
                
                let content = UNMutableNotificationContent()
                content.title = ""  // Empty title - only AI body
                content.body = body  // NO mention of upcoming event
                content.sound = .default
                
                let req = UNNotificationRequest(identifier: "evacuation-\(Date().timeIntervalSince1970)", content: content, trigger: nil)
                try? await UNUserNotificationCenter.current().add(req)
                
                lastNotificationTime = Date()
                print("✅ [NotificationEngine] Evacuation notification sent!")
            } else {
                print("⚠️ [NotificationEngine] No recent stressed event found for evacuation")
            }
        }
    }
    
    private func sendEventReminder(event: Event, minutesUntil: Int) {
        let content = UNMutableNotificationContent()
        content.title = "📅 Coming Up"
        content.body = "\(event.title ?? "Event") in \(minutesUntil) minutes"
        content.sound = .default
        
        let req = UNNotificationRequest(identifier: "reminder-\(Date().timeIntervalSince1970)", content: content, trigger: nil)
        
        Task {
            try? await UNUserNotificationCenter.current().add(req)
            lastNotificationTime = Date()
            print("[NotificationEngine] ✅ Event reminder sent (20-min)")
        }
    }
    
    // MARK: - Helper
    
    private func shouldSendNotification() -> Bool {
        guard let lastTime = lastNotificationTime else {
            print("⏱️ [NotificationEngine] No previous notification - OK to send")
            return true
        }
        let timeSince = Date().timeIntervalSince(lastTime)
        let remaining = notificationCooldown - timeSince
        if remaining > 0 {
            print("⏱️ [NotificationEngine] Cooldown active: \(Int(remaining))s remaining")
            return false
        } else {
            print("⏱️ [NotificationEngine] Cooldown expired - OK to send")
            return true
        }
    }
}

// MARK: - Supporting Types

enum StressState {
    case calm
    case stressed
}

enum UrgencyLevel {
    case low, medium, high
}

indirect enum NotificationDecision {
    /// Wait - don't notify yet
    case wait
    
    /// Send notification immediately
    case sendNow(title: String, body: String?, sound: UNNotificationSound, priority: NotificationPriority)
    
    /// Delay until user is calm, with fallback
    case delayUntilCalm(showMinimal: Bool, maxDelay: TimeInterval, fallbackDecision: NotificationDecision)
    
    /// Send combined notification (stress break + event reminder)
    case sendCombined(event: CalendarEvent)
    
    /// Generate message using AI
    case generateWithAI(context: StressEpisode, fallback: NotificationDecision)
}

enum NotificationPriority {
    case low, medium, normal, high
}

// MARK: - Combined Notification Builder

extension NotificationDecisionEngine {
    
    /// Build a combined notification for stress + upcoming event
    func buildCombinedNotification(event: CalendarEvent, stressEpisode: StressEpisode) -> (title: String, body: String, sound: UNNotificationSound) {
        let minutesUntilEvent = Int(event.startTime.timeIntervalSinceNow / 60)
        let stressDuration = Int(stressEpisode.duration / 60)
        
        // Smart message based on context
        let title: String
        let body: String
        
        if minutesUntilEvent <= 2 {
            // Imminent event - be direct
            title = "⚠️ \(event.title) in \(minutesUntilEvent) min"
            body = "You've been stressed. Take 30 seconds to breathe before heading there."
        } else if minutesUntilEvent <= 5 {
            // Soon - give break suggestion
            title = "🌿 Take a quick break"
            body = "\(event.title) starts in \(minutesUntilEvent) min. A short walk might help reset."
        } else {
            // More time - be gentle
            title = "Heads up: \(event.title) in \(minutesUntilEvent) min"
            body = "You've been stressed for \(stressDuration) min. How about a breather before your next event?"
        }
        
        return (title, body, .default)
    }
}

// MARK: - User Preferences

class UserPreferences {
    static let shared = UserPreferences()
    
    private let defaults = UserDefaults.standard
    
    var defaultReminderMinutes: Int {
        get { defaults.integer(forKey: "defaultReminderMinutes") == 0 ? 10 : defaults.integer(forKey: "defaultReminderMinutes") }
        set { defaults.set(newValue, forKey: "defaultReminderMinutes") }
    }
    
    var allowNotificationsDuringStress: Bool {
        get { defaults.bool(forKey: "allowNotificationsDuringStress") }
        set { defaults.set(newValue, forKey: "allowNotificationsDuringStress") }
    }
}

// MARK: - Calendar Event Model

struct CalendarEvent {
    let id: String
    let title: String
    let startTime: Date
    let endTime: Date
    let reminderMinutes: Int?  // Optional custom reminder time
    let priority: EventPriority
}

enum EventPriority {
    case low, normal, high
}

// MARK: - Stress Episode Model

struct StressEpisode {
    let startTime: Date
    let endTime: Date?
    let duration: TimeInterval  // seconds
    let peakProbability: Double
    let hasEnded: Bool
    
    init(startTime: Date, endTime: Date? = nil, peakProbability: Double = 0.0) {
        self.startTime = startTime
        self.endTime = endTime
        self.peakProbability = peakProbability
        self.hasEnded = (endTime != nil)
        self.duration = endTime?.timeIntervalSince(startTime) ?? Date().timeIntervalSince(startTime)
    }
}
