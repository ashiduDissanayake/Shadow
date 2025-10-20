//
//  NotificationDecisionEngine.swift
//  Shadow
//
//  Intelligent notification timing based on stress state and event priority
//

import Foundation
import UserNotifications

/// Handles intelligent notification timing decisions
@MainActor
class NotificationDecisionEngine: ObservableObject {
    
    static let shared = NotificationDecisionEngine()
    
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

    // Track last notified state to prevent duplicates and ensure transitions
    private var lastNotifiedState: Int16?
    
    // MARK: - Runtime evaluation API (simple)
    /// Lightweight evaluator used by the app when a new StressEvent is persisted.
    func evaluate(event: StressEvent) {
        print("[NotificationEngine] 🔔 Evaluating event: seq=\(event.sequenceNumber), state=\(event.stressState)")
        
        let currentState = event.stressState
        
        // ONLY notify on STATE TRANSITIONS (0→1 or 1→0)
        // NOT on consecutive same states
        if let lastState = lastNotifiedState {
            if currentState == lastState {
                print("[NotificationEngine] ⏭️ Same state as last notification (\(currentState)), skipping")
                return
            }
        }
        
        // State 1 (STRESSED) - only if we weren't already stressed
        if currentState == 1 {
            print("[NotificationEngine] 🚨 STRESS DETECTED! (Transition: \(lastNotifiedState ?? -1) → 1)")
            
            // Mark state BEFORE async work
            lastNotifiedState = 1
            
            // Generate AI message asynchronously
            Task {
                let body = await AIDecisionProvider.shared.message(for: event)
                print("[NotificationEngine] 💬 AI message generated: \(body)")
                
                // Schedule immediate local notification
                let content = UNMutableNotificationContent()
                content.title = "Quick Reset"
                content.body = body
                content.sound = .default
                
                let req = UNNotificationRequest(identifier: "stress-\(event.sequenceNumber)", content: content, trigger: nil)
                do {
                    try await UNUserNotificationCenter.current().add(req)
                    print("[NotificationEngine] ✅ Notification scheduled successfully")
                } catch {
                    print("[NotificationEngine] ❌ Scheduling error: \(error)")
                }
            }
        } else if currentState == 0 {
            // State 0 (CALM) - only if we were previously stressed
            print("[NotificationEngine] ✅ CALM STATE (Transition: \(lastNotifiedState ?? -1) → 0)")
            
            lastNotifiedState = 0
            
            Task {
                let body = await AIDecisionProvider.shared.message(for: event)
                print("[NotificationEngine] 💬 AI message generated: \(body)")
                
                let content = UNMutableNotificationContent()
                content.title = "You're Back"
                content.body = body
                content.sound = .default
                
                let req = UNNotificationRequest(identifier: "calm-\(event.sequenceNumber)", content: content, trigger: nil)
                do {
                    try await UNUserNotificationCenter.current().add(req)
                    print("[NotificationEngine] ✅ Notification scheduled successfully")
                } catch {
                    print("[NotificationEngine] ❌ Scheduling error: \(error)")
                }
            }
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
