//
//  CalendarEventMonitor.swift
//  Shadow
//
//  Monitors upcoming calendar events and triggers stress-aware notifications
//

import Foundation
import CoreData
import Combine
import UserNotifications

@MainActor
class CalendarEventMonitor: ObservableObject {
    static let shared = CalendarEventMonitor()
    
    private var timer: Timer?
    private var cancellables = Set<AnyCancellable>()
    private let stressRepo = StressDataRepository.shared
    private let notificationCenter = UNUserNotificationCenter.current()
    
    // Track notified events to prevent duplicates
    private var notifiedEvents = Set<UUID>()
    
    // Observe CoreData changes for immediate event detection
    private var eventsObserver: NSObjectProtocol?
    
    private init() {
        // Set up CoreData notification observer for new/updated events
        setupEventObserver()
    }
    
    deinit {
        if let observer = eventsObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }
    
    // MARK: - Event Observer
    
    /// Set up observer for CoreData changes to Event entity
    private func setupEventObserver() {
        eventsObserver = NotificationCenter.default.addObserver(
            forName: NSManagedObjectContext.didSaveObjectsNotification,
            object: PersistenceController.shared.viewContext,
            queue: .main
        ) { [weak self] notification in
            guard let self = self else { return }
            
            // Check if any Event entities were inserted or updated
            let insertedObjects = notification.userInfo?[NSInsertedObjectsKey] as? Set<NSManagedObject> ?? []
            let updatedObjects = notification.userInfo?[NSUpdatedObjectsKey] as? Set<NSManagedObject> ?? []
            
            let hasEventChanges = insertedObjects.contains { $0 is Event } || 
                                 updatedObjects.contains { $0 is Event }
            
            if hasEventChanges {
                print("📅 [CalendarMonitor] Event change detected - checking immediately")
                Task { @MainActor in
                    await self.checkUpcomingEvents()
                }
            }
        }
    }
    
    // MARK: - Monitoring
    
    /// Start monitoring calendar events
    func startMonitoring() {
        print("📅 [CalendarMonitor] Starting event monitoring...")
        
        // Check every 5 minutes for upcoming events (aligned with notification cooldown)
        timer = Timer.scheduledTimer(withTimeInterval: 300.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                await self?.checkUpcomingEvents()
            }
        }
        
        // Immediate first check
        Task {
            await checkUpcomingEvents()
        }
    }
    
    /// Stop monitoring
    func stopMonitoring() {
        print("📅 [CalendarMonitor] Stopping event monitoring")
        timer?.invalidate()
        timer = nil
    }
    
    /// Manually trigger event check (useful for testing or when user adds event)
    func checkNow() {
        print("📅 [CalendarMonitor] Manual check triggered")
        Task { @MainActor in
            await checkUpcomingEvents()
        }
    }
    
    // MARK: - Event Checking
    
    private func checkUpcomingEvents() async {
        // Get all events from CoreData
        let allEvents = EventRepository.shared.fetchAllUpcomingEvents(withinHours: 1)
        
        guard !allEvents.isEmpty else {
            return
        }
        
        print("📅 [CalendarMonitor] Found \(allEvents.count) upcoming events in next hour")
        
        // Get current stress state from recent StressEvents
        let recentEvents = stressRepo.recentEvents(limit: 1)
        let currentStressState = recentEvents.first?.stressState ?? 0
        
        print("📅 [CalendarMonitor] Current stress state: \(currentStressState)")
        
        // Pass to NotificationDecisionEngine for intelligent handling
        NotificationDecisionEngine.shared.checkCalendarEvents(
            events: allEvents,
            currentStressState: Int(currentStressState)
        )
    }
    
    private func processEvent(_ event: Event, stressState: StressState) async {
        guard let eventDate = event.date,
              let eventId = event.id else { return }
        
        // Skip if already notified
        guard !notifiedEvents.contains(eventId) else {
            return
        }
        
        let minutesUntilEvent = eventDate.timeIntervalSinceNow / 60
        
        // Default reminder time: 10 minutes before
        let reminderMinutes = 10.0
        
        // Should we notify now?
        guard minutesUntilEvent <= reminderMinutes && minutesUntilEvent > 0 else {
            return
        }
        
        print("📅 [CalendarMonitor] Event '\(event.title ?? "Untitled")' in \(Int(minutesUntilEvent)) minutes, stress=\(stressState)")
        
        // Mark as notified
        notifiedEvents.insert(eventId)
        
        // Build notification based on stress state
        let notification = buildNotification(
            event: event,
            minutesUntil: Int(minutesUntilEvent),
            stressState: stressState
        )
        
        // Send notification
        await sendNotification(
            identifier: "calendar-\(eventId)",
            title: notification.title,
            body: notification.body,
            sound: notification.sound
        )
    }
    
    // MARK: - Stress State
    
    private func getCurrentStressState() -> StressState {
        // Get most recent stress event
        let recentEvents = stressRepo.recentEvents(limit: 1)
        
        guard let latestEvent = recentEvents.first else {
            return .calm
        }
        
        // Check if event is recent (within last 5 minutes)
        guard let timestamp = latestEvent.timestamp,
              Date().timeIntervalSince(timestamp) < 300 else {
            return .calm
        }
        
        return latestEvent.stressState == 1 ? .stressed : .calm
    }
    
    // MARK: - Notification Builder
    
    private func buildNotification(
        event: Event,
        minutesUntil: Int,
        stressState: StressState
    ) -> (title: String, body: String, sound: UNNotificationSound) {
        
        let eventTitle = event.title ?? "Event"
        
        switch stressState {
        case .calm:
            // User is calm - standard reminder
            return (
                title: "📅 Upcoming: \(eventTitle)",
                body: "Starts in \(minutesUntil) minutes",
                sound: .default
            )
            
        case .stressed:
            // User is stressed - adaptive notification
            if minutesUntil <= 2 {
                // Imminent event - critical
                return (
                    title: "⚠️ \(eventTitle) in \(minutesUntil) min",
                    body: "You're stressed. Take 30 seconds to breathe before heading there.",
                    sound: .defaultCritical
                )
            } else if minutesUntil <= 5 {
                // Soon - suggest quick break
                return (
                    title: "🌿 Quick Break Before \(eventTitle)",
                    body: "Starts in \(minutesUntil) min. Short walk or breathing exercise?",
                    sound: .default
                )
            } else {
                // More time - gentle reminder
                return (
                    title: "Heads up: \(eventTitle) in \(minutesUntil) min",
                    body: "You've been stressed. How about a breather before your event?",
                    sound: .default
                )
            }
        }
    }
    
    // MARK: - Notification Sending
    
    private func sendNotification(
        identifier: String,
        title: String,
        body: String,
        sound: UNNotificationSound
    ) async {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = sound
        
        let request = UNNotificationRequest(
            identifier: identifier,
            content: content,
            trigger: nil  // Send immediately
        )
        
        do {
            try await notificationCenter.add(request)
            print("📅 [CalendarMonitor] ✅ Notification sent: \(title)")
        } catch {
            print("📅 [CalendarMonitor] ❌ Failed to send notification: \(error)")
        }
    }
    
    // MARK: - Cleanup
    
    /// Clear notification history (call when events are deleted or past)
    func clearNotifiedEvent(_ eventId: UUID) {
        notifiedEvents.remove(eventId)
    }
    
    func clearOldNotifications() {
        // Clear events older than 1 hour
        notifiedEvents.removeAll()
    }
}

// MARK: - EventRepository Extension

extension EventRepository {
    /// Fetch upcoming events within specified hours
    func fetchAllUpcomingEvents(withinHours hours: Int = 1) -> [Event] {
        // Use ProfileRepository's context since EventRepository.context is private
        let context = ProfileRepository.shared.container.viewContext
        
        let request: NSFetchRequest<Event> = Event.fetchRequest()
        let now = Date()
        let futureDate = Calendar.current.date(byAdding: .hour, value: hours, to: now) ?? now
        
        request.predicate = NSPredicate(
            format: "date >= %@ AND date <= %@",
            now as NSDate,
            futureDate as NSDate
        )
        request.sortDescriptors = [NSSortDescriptor(key: "date", ascending: true)]
        
        return (try? context.fetch(request)) ?? []
    }
}
