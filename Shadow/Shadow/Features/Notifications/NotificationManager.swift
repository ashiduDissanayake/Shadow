//
//  NotificationManager.swift
//  Shadow
//
//  Created on 19/10/2025.
//  Manages local notifications for stress alerts and calendar events
//

import Foundation
import UserNotifications

@MainActor
final class NotificationManager: ObservableObject {
    static let shared = NotificationManager()
    
    @Published var isAuthorized = false
    @Published var notificationsEnabled = true
    
    private init() {
        checkAuthorizationStatus()
        loadSettings()
    }
    
    // MARK: - Authorization
    
    func requestAuthorization() async -> Bool {
        do {
            let granted = try await UNUserNotificationCenter.current()
                .requestAuthorization(options: [.alert, .sound, .badge])
            isAuthorized = granted
            return granted
        } catch {
            print("Notification authorization error: \(error)")
            return false
        }
    }
    
    private func checkAuthorizationStatus() {
        Task {
            let settings = await UNUserNotificationCenter.current().notificationSettings()
            isAuthorized = settings.authorizationStatus == .authorized
        }
    }
    
    // MARK: - Settings
    
    private func loadSettings() {
        notificationsEnabled = UserDefaults.standard.bool(forKey: "NotificationsEnabled")
        if UserDefaults.standard.object(forKey: "NotificationsEnabled") == nil {
            // Default to true on first launch
            notificationsEnabled = true
            UserDefaults.standard.set(true, forKey: "NotificationsEnabled")
        }
    }
    
    func toggleNotifications() {
        notificationsEnabled.toggle()
        UserDefaults.standard.set(notificationsEnabled, forKey: "NotificationsEnabled")
    }
    
    // MARK: - Stress Notifications
    
    func sendStressAlert(severity: StressSeverity = .high) {
        guard notificationsEnabled && isAuthorized else { return }
        
        let content = UNMutableNotificationContent()
        content.title = "⚠️ Stress Level Elevated"
        
        switch severity {
        case .high:
            content.body = "Your stress level is high. Consider taking a break to relax."
        case .medium:
            content.body = "Your stress level is rising. Take a moment to breathe."
        case .low:
            content.body = "Mild stress detected. Stay mindful of your wellbeing."
        }
        
        content.sound = .default
        content.categoryIdentifier = "STRESS_ALERT"
        
        // Deliver immediately
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 1, repeats: false)
        let request = UNNotificationRequest(
            identifier: "stress_\(Date().timeIntervalSince1970)",
            content: content,
            trigger: trigger
        )
        
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Failed to send stress notification: \(error)")
            }
        }
    }
    
    func sendStressRecoveryNotification() {
        guard notificationsEnabled && isAuthorized else { return }
        
        let content = UNMutableNotificationContent()
        content.title = "✅ Stress Level Normalized"
        content.body = "Great! Your stress levels are back to normal. Keep it up!"
        content.sound = .default
        content.categoryIdentifier = "STRESS_RECOVERY"
        
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 1, repeats: false)
        let request = UNNotificationRequest(
            identifier: "recovery_\(Date().timeIntervalSince1970)",
            content: content,
            trigger: trigger
        )
        
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Failed to send recovery notification: \(error)")
            }
        }
    }
    
    // MARK: - Calendar Event Notifications
    
    func scheduleEventReminder(event: Event, minutesBefore: Int = 15) {
        guard notificationsEnabled && isAuthorized else { return }
        guard let eventDate = event.date else { return }
        
        let content = UNMutableNotificationContent()
        content.title = "📅 Upcoming Event"
        content.body = event.title ?? "Event reminder"
        content.sound = .default
        content.categoryIdentifier = "CALENDAR_EVENT"
        
        // Calculate trigger date
        let triggerDate = eventDate.addingTimeInterval(TimeInterval(-minutesBefore * 60))
        
        // Only schedule if in the future
        guard triggerDate > Date() else { return }
        
        let components = Calendar.current.dateComponents(
            [.year, .month, .day, .hour, .minute],
            from: triggerDate
        )
        
        let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: false)
        let request = UNNotificationRequest(
            identifier: "event_\(event.id?.uuidString ?? UUID().uuidString)",
            content: content,
            trigger: trigger
        )
        
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Failed to schedule event reminder: \(error)")
            }
        }
    }
    
    func sendMotivationalMessage() {
        guard notificationsEnabled && isAuthorized else { return }
        
        let messages = [
            "💪 Remember to take breaks throughout your day!",
            "🌟 You're doing great! Keep managing your stress well.",
            "☕ Time for a quick break? Your wellbeing matters.",
            "🧘 Consider a short meditation session to recharge.",
            "🌱 Small moments of rest lead to big improvements.",
            "💙 Your mental health is a priority. Take care of yourself."
        ]
        
        let content = UNMutableNotificationContent()
        content.title = "💡 Wellness Reminder"
        content.body = messages.randomElement() ?? messages[0]
        content.sound = .default
        content.categoryIdentifier = "MOTIVATION"
        
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 1, repeats: false)
        let request = UNNotificationRequest(
            identifier: "motivation_\(Date().timeIntervalSince1970)",
            content: content,
            trigger: trigger
        )
        
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Failed to send motivational message: \(error)")
            }
        }
    }
    
    // MARK: - Cancel Notifications
    
    func cancelAllNotifications() {
        UNUserNotificationCenter.current().removeAllPendingNotificationRequests()
        UNUserNotificationCenter.current().removeAllDeliveredNotifications()
    }
    
    func cancelEventNotification(eventId: UUID) {
        UNUserNotificationCenter.current()
            .removePendingNotificationRequests(withIdentifiers: ["event_\(eventId.uuidString)"])
    }
}

// MARK: - Supporting Types

enum StressSeverity {
    case low
    case medium
    case high
}

// MARK: - Notification Categories

extension NotificationManager {
    static func setupNotificationCategories() {
        // Define actions for stress alerts
        let takeBreakAction = UNNotificationAction(
            identifier: "TAKE_BREAK",
            title: "Take a Break",
            options: .foreground
        )
        
        let dismissAction = UNNotificationAction(
            identifier: "DISMISS",
            title: "Dismiss",
            options: []
        )
        
        let stressCategory = UNNotificationCategory(
            identifier: "STRESS_ALERT",
            actions: [takeBreakAction, dismissAction],
            intentIdentifiers: [],
            options: .customDismissAction
        )
        
        // Event reminder category
        let viewEventAction = UNNotificationAction(
            identifier: "VIEW_EVENT",
            title: "View Event",
            options: .foreground
        )
        
        let eventCategory = UNNotificationCategory(
            identifier: "CALENDAR_EVENT",
            actions: [viewEventAction, dismissAction],
            intentIdentifiers: [],
            options: .customDismissAction
        )
        
        UNUserNotificationCenter.current()
            .setNotificationCategories([stressCategory, eventCategory])
    }
}
