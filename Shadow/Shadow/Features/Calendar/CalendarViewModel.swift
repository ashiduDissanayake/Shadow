
//
//  CalendarViewModel.swift
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-25.
//

import EventKit
import Foundation
import SwiftUI

@MainActor
final class CalendarViewModel: ObservableObject {
    @Published var calendars: [EKCalendar] = []
    @Published var events: [EKEvent] = []
    @Published var selectedCalendar: EKCalendar?
    @Published var hasPermission = false
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    private let calendarManager = CalendarManager.shared
    
    init() {
        checkPermissionStatus()
    }
    
    // MARK: - Permissions
    
    private func checkPermissionStatus() {
        let status = EKEventStore.authorizationStatus(for: .event)
        hasPermission = status == .authorized
        
        if hasPermission {
            loadCalendars()
        }
    }
    
    func requestAccess() {
        isLoading = true
        calendarManager.requestAccess { [weak self] granted in
            self?.hasPermission = granted
            self?.isLoading = false
            
            if granted {
                self?.loadCalendars()
            } else {
                self?.errorMessage = "Calendar access denied. Please enable in System Preferences > Security & Privacy > Privacy > Calendars."
            }
        }
    }
    
    // MARK: - Data Loading
    
    func loadCalendars() {
        calendars = calendarManager.fetchCalendars()
        
        // Auto-select the first writable calendar
        selectedCalendar = calendars.first { $0.allowsContentModifications }
        
        if selectedCalendar != nil {
            loadEvents()
        }
    }
    
    func loadEvents() {
        guard let calendar = selectedCalendar else { return }
        
        isLoading = true
        events = calendarManager.fetchUpcomingEvents(calendar: calendar, days: 30)
        isLoading = false
    }
    
    // MARK: - Event Management
    
    func addTestEvent() {
        guard let calendar = selectedCalendar else { return }
        
        let now = Date()
        let endDate = Calendar.current.date(byAdding: .hour, value: 1, to: now) ?? now
        
        do {
            try calendarManager.createEvent(
                title: "Shadow Test Event",
                notes: "Created from Shadow app - Stress monitoring session",
                startDate: now,
                endDate: endDate,
                calendar: calendar
            )
            loadEvents() // Refresh the list
        } catch {
            errorMessage = "Failed to create event: \(error.localizedDescription)"
        }
    }
    
    func addCustomEvent(title: String, notes: String?, startDate: Date, duration: TimeInterval) {
        guard let calendar = selectedCalendar else { return }
        
        let endDate = startDate.addingTimeInterval(duration)
        
        do {
            try calendarManager.createEvent(
                title: title,
                notes: notes,
                startDate: startDate,
                endDate: endDate,
                calendar: calendar
            )
            loadEvents()
        } catch {
            errorMessage = "Failed to create event: \(error.localizedDescription)"
        }
    }
    
    func deleteEvent(_ event: EKEvent) {
        do {
            try calendarManager.deleteEvent(event)
            loadEvents() // Refresh the list
        } catch {
            errorMessage = "Failed to delete event: \(error.localizedDescription)"
        }
    }
}
