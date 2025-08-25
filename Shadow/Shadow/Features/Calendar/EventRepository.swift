//
//  EventRepository.swift
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-25.
//


import Foundation
import CoreData

class EventRepository {
    static let shared = EventRepository()
    private let context: NSManagedObjectContext

    private init() {
        self.context = ProfileRepository.shared.container.viewContext
    }

    func fetchEvents(for profile: UserProfile?, on date: Date? = nil, type: String? = nil) -> [Event] {
        guard let profile = profile else { return [] }
        let request: NSFetchRequest<Event> = Event.fetchRequest()
        var predicates: [NSPredicate] = [NSPredicate(format: "userProfile == %@", profile)]
        if let date = date {
            let calendar = Calendar.current
            let start = calendar.startOfDay(for: date)
            let end = calendar.date(byAdding: .day, value: 1, to: start)!
            predicates.append(NSPredicate(format: "date >= %@ AND date < %@", start as NSDate, end as NSDate))
        }
        if let type = type, type != "All" {
            predicates.append(NSPredicate(format: "eventType == %@", type))
        }
        request.predicate = NSCompoundPredicate(andPredicateWithSubpredicates: predicates)
        request.sortDescriptors = [NSSortDescriptor(key: "date", ascending: true)]
        return (try? context.fetch(request)) ?? []
    }

    func addEvent(for profile: UserProfile, title: String, notes: String?, date: Date, duration: Double, eventType: String, customField: String?) {
        let event = Event(context: context)
        event.id = UUID()
        event.title = title
        event.notes = notes
        event.date = date
        event.duration = duration
        event.eventType = eventType
        event.customField = customField
        event.userProfile = profile
        try? context.save()
    }

    func deleteEvent(_ event: Event) {
        context.delete(event)
        try? context.save()
    }

    func deleteAllEvents(for profile: UserProfile) {
        let events = fetchEvents(for: profile)
        for event in events {
            context.delete(event)
        }
        try? context.save()
    }
}