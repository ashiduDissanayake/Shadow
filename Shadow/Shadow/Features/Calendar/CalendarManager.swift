//
//  CalendarManager.swift
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-25.
//


import EventKit
import Foundation

/// Handles all direct EventKit interactions for calendars and events.
final class CalendarManager {
    static let shared = CalendarManager()

    private let eventStore = EKEventStore()

    // MARK: - Permissions

    func requestAccess(completion: @escaping (Bool) -> Void) {
        eventStore.requestAccess(to: .event) { granted, _ in
            DispatchQueue.main.async {
                completion(granted)
            }
        }
    }

    // MARK: - Calendars

    func fetchCalendars() -> [EKCalendar] {
        eventStore.calendars(for: .event)
    }

    // MARK: - Events

    func fetchUpcomingEvents(
        calendar: EKCalendar,
        days: Int = 7
    ) -> [EKEvent] {
        let startDate = Date()
        guard let endDate = Calendar.current.date(byAdding: .day, value: days, to: startDate) else { return [] }
        let predicate = eventStore.predicateForEvents(withStart: startDate, end: endDate, calendars: [calendar])
        return eventStore.events(matching: predicate)
    }

    func createEvent(
        title: String,
        notes: String?,
        startDate: Date,
        endDate: Date,
        calendar: EKCalendar
    ) throws {
        let event = EKEvent(eventStore: eventStore)
        event.title = title
        event.notes = notes
        event.startDate = startDate
        event.endDate = endDate
        event.calendar = calendar
        try eventStore.save(event, span: .thisEvent)
    }

    func deleteEvent(_ event: EKEvent) throws {
        try eventStore.remove(event, span: .thisEvent)
    }
}