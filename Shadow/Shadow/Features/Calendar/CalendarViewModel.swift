import Foundation
import SwiftUI

class CalendarViewModel: ObservableObject {
    @Published var selectedDate: Date = Date()
    @Published var selectedEventType: String = "All"
    @Published var events: [Event] = []
    @Published var errorMessage: String?
    @Published var profile: UserProfile?

    private let eventRepo = EventRepository.shared

    func setProfile(_ profile: UserProfile?) {
        self.profile = profile
        loadEvents()
    }

    func loadEvents() {
        guard let profile = profile else { events = []; return }
        events = eventRepo.fetchEvents(for: profile)
    }

    var filteredEvents: [Event] {
        eventRepo.fetchEvents(for: profile, on: selectedDate, type: selectedEventType)
    }

    func addEvent(title: String, notes: String?, date: Date, duration: Double, eventType: String, customField: String?) {
        guard let profile = profile else { return }
        eventRepo.addEvent(for: profile, title: title, notes: notes, date: date, duration: duration, eventType: eventType, customField: customField)
        loadEvents()
    }

    func deleteEvent(_ event: Event) {
        eventRepo.deleteEvent(event)
        loadEvents()
    }

    func deleteAllEventsForCurrentProfile() {
        guard let profile = profile else { return }
        eventRepo.deleteAllEvents(for: profile)
        loadEvents()
    }
}
