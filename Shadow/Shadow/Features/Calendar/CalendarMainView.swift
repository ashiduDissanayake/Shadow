//
//  CalendarMainView.swift (macOS Optimized)
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-25.
//

import SwiftUI
import EventKit

struct CalendarMainView: View {
    @StateObject private var viewModel = CalendarViewModel()
    @Environment(\.dismiss) private var dismiss
    @State private var showingAddEvent = false
    
    var body: some View {
        ZStack {
            // Background gradient matching your app theme
            LinearGradient(
                gradient: Gradient(colors: [
                    Color(red: 0.05, green: 0.08, blue: 0.15),
                    Color(red: 0.1, green: 0.15, blue: 0.25)
                ]),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            
            VStack(spacing: 0) {
                // Header with macOS-style controls
                headerView
                
                if viewModel.hasPermission {
                    authorizedContentView
                } else {
                    unauthorizedView
                }
            }
        }
        .frame(minWidth: 600, minHeight: 500)
        .sheet(isPresented: $showingAddEvent) {
            AddEventView { title, notes, date, duration in
                viewModel.addCustomEvent(
                    title: title,
                    notes: notes,
                    startDate: date,
                    duration: duration
                )
            }
        }
        .alert("Error", isPresented: .constant(viewModel.errorMessage != nil)) {
            Button("OK") {
                viewModel.errorMessage = nil
            }
        } message: {
            if let error = viewModel.errorMessage {
                Text(error)
            }
        }
    }
    
    // MARK: - View Components
    
    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Calendar Integration")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                
                Text("Manage your stress monitoring sessions")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.7))
            }
            
            Spacer()
            
            // macOS-style close button
            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2)
                    .foregroundColor(.white.opacity(0.6))
                    .background(Circle().fill(.black.opacity(0.2)))
            }
            .buttonStyle(.borderless)
            .help("Close Calendar")
        }
        .padding(.horizontal, 24)
        .padding(.top, 20)
        .padding(.bottom, 20)
    }
    
    private var authorizedContentView: some View {
        HStack(spacing: 24) {
            // Left sidebar with calendar selection and controls
            leftSidebarView
                .frame(width: 280)
            
            // Main content area
            mainContentView
                .frame(maxWidth: .infinity)
        }
        .padding(.horizontal, 24)
        .padding(.bottom, 24)
    }
    
    private var leftSidebarView: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Calendar Selector
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: "calendar")
                        .foregroundColor(.blue)
                        .font(.title2)
                    Text("Calendars")
                        .font(.headline)
                        .foregroundColor(.white)
                }
                
                VStack(spacing: 8) {
                    ForEach(viewModel.calendars, id: \.calendarIdentifier) { calendar in
                        Button(action: {
                            viewModel.selectedCalendar = calendar
                            viewModel.loadEvents()
                        }) {
                            HStack {
                                Circle()
                                    .fill(Color(calendar.cgColor))
                                    .frame(width: 12, height: 12)
                                
                                Text(calendar.title)
                                    .font(.subheadline)
                                    .foregroundColor(.white)
                                    .multilineTextAlignment(.leading)
                                
                                Spacer()
                                
                                if viewModel.selectedCalendar?.calendarIdentifier == calendar.calendarIdentifier {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.blue)
                                        .font(.caption)
                                }
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(viewModel.selectedCalendar?.calendarIdentifier == calendar.calendarIdentifier ?
                                          .blue.opacity(0.2) : .white.opacity(0.05))
                                    .stroke(viewModel.selectedCalendar?.calendarIdentifier == calendar.calendarIdentifier ?
                                           .blue.opacity(0.4) : .clear, lineWidth: 1)
                            )
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(.white.opacity(0.05))
                    .stroke(.white.opacity(0.1), lineWidth: 1)
            )
            
            // Action Buttons
            VStack(spacing: 12) {
                Button(action: { showingAddEvent = true }) {
                    HStack {
                        Image(systemName: "plus.circle.fill")
                        Text("New Event")
                        Spacer()
                    }
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.blue.opacity(0.6))
                    )
                }
                .buttonStyle(.borderless)
                
                Button(action: { viewModel.addTestEvent() }) {
                    HStack {
                        Image(systemName: "flask.fill")
                        Text("Add Test Event")
                        Spacer()
                    }
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.purple.opacity(0.6))
                    )
                }
                .buttonStyle(.borderless)
                
                Button(action: { viewModel.loadEvents() }) {
                    HStack {
                        Image(systemName: "arrow.clockwise")
                        Text("Refresh")
                        Spacer()
                    }
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.white.opacity(0.1))
                            .stroke(.white.opacity(0.2), lineWidth: 1)
                    )
                }
                .buttonStyle(.borderless)
            }
            
            Spacer()
        }
    }
    
    private var mainContentView: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Events header
            HStack {
                Image(systemName: "calendar.badge.clock")
                    .foregroundColor(.green)
                    .font(.title2)
                
                Text("Upcoming Events")
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                
                Spacer()
                
                Text("\(viewModel.events.count) events")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.6))
            }
            
            // Events content
            if viewModel.isLoading {
                VStack {
                    ProgressView()
                        .scaleEffect(1.2)
                    Text("Loading events...")
                        .font(.subheadline)
                        .foregroundColor(.white.opacity(0.7))
                        .padding(.top, 8)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if viewModel.events.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "calendar.badge.plus")
                        .font(.system(size: 48))
                        .foregroundColor(.white.opacity(0.3))
                    
                    Text("No upcoming events")
                        .font(.title3)
                        .fontWeight(.medium)
                        .foregroundColor(.white.opacity(0.7))
                    
                    Text("Create your first stress monitoring session")
                        .font(.subheadline)
                        .foregroundColor(.white.opacity(0.5))
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(viewModel.events, id: \.eventIdentifier) { event in
                            EventRowView(event: event) {
                                viewModel.deleteEvent(event)
                            }
                        }
                    }
                    .padding(.vertical, 8)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.white.opacity(0.05))
                .stroke(.white.opacity(0.1), lineWidth: 1)
        )
    }
    
    private var unauthorizedView: some View {
        VStack(spacing: 24) {
            Image(systemName: "calendar.badge.exclamationmark")
                .font(.system(size: 80))
                .foregroundColor(.orange)
            
            VStack(spacing: 8) {
                Text("Calendar Access Required")
                    .font(.title)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                
                Text("Shadow needs access to your calendar to help you schedule and track stress monitoring sessions.")
                    .font(.body)
                    .foregroundColor(.white.opacity(0.8))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }
            
            if viewModel.isLoading {
                VStack(spacing: 12) {
                    ProgressView()
                        .scaleEffect(1.2)
                    Text("Requesting permission...")
                        .font(.subheadline)
                        .foregroundColor(.white.opacity(0.7))
                }
            } else {
                VStack(spacing: 16) {
                    Button(action: { viewModel.requestAccess() }) {
                        Text("Grant Calendar Access")
                            .font(.headline)
                            .fontWeight(.medium)
                            .foregroundColor(.white)
                            .padding(.horizontal, 32)
                            .padding(.vertical, 12)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(.blue.opacity(0.7))
                            )
                    }
                    .buttonStyle(.borderless)
                    
                    Text("You can also enable this in System Preferences > Security & Privacy > Privacy > Calendars")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.5))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 60)
                }
            }
        }
        .padding(60)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Supporting Views

struct EventRowView: View {
    let event: EKEvent
    let onDelete: () -> Void
    @State private var isHovered = false
    
    var body: some View {
        HStack(spacing: 16) {
            // Event indicator
            RoundedRectangle(cornerRadius: 3)
                .fill(Color(event.calendar.cgColor))
                .frame(width: 6, height: 60)
            
            VStack(alignment: .leading, spacing: 6) {
                Text(event.title ?? "Untitled Event")
                    .font(.headline)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .lineLimit(2)
                
                HStack(spacing: 12) {
                    HStack(spacing: 4) {
                        Image(systemName: "calendar")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.6))
                        
                        Text(event.startDate, style: .date)
                            .font(.subheadline)
                            .foregroundColor(.white.opacity(0.7))
                    }
                    
                    HStack(spacing: 4) {
                        Image(systemName: "clock")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.6))
                        
                        Text(event.startDate, style: .time)
                            .font(.subheadline)
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
                
                if let notes = event.notes, !notes.isEmpty {
                    Text(notes)
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.5))
                        .lineLimit(2)
                        .padding(.top, 2)
                }
            }
            
            Spacer()
            
            if isHovered {
                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.subheadline)
                        .foregroundColor(.red)
                        .padding(8)
                        .background(Circle().fill(.red.opacity(0.15)))
                }
                .buttonStyle(.borderless)
                .help("Delete Event")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(isHovered ? .white.opacity(0.08) : .white.opacity(0.04))
                .stroke(.white.opacity(isHovered ? 0.15 : 0.08), lineWidth: 1)
        )
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.2)) {
                isHovered = hovering
            }
        }
    }
}

struct AddEventView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var title = ""
    @State private var notes = ""
    @State private var selectedDate = Date()
    @State private var duration: TimeInterval = 3600 // 1 hour
    
    let onAdd: (String, String?, Date, TimeInterval) -> Void
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("New Event")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Spacer()
                
                HStack(spacing: 12) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .buttonStyle(.borderless)
                    
                    Button("Add Event") {
                        onAdd(title, notes.isEmpty ? nil : notes, selectedDate, duration)
                        dismiss()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(title.isEmpty)
                }
            }
            .padding()
            
            Divider()
            
            // Form content
            Form {
                Section("Event Details") {
                    TextField("Event Title", text: $title)
                        .textFieldStyle(.roundedBorder)
                    
                    TextField("Notes (Optional)", text: $notes, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(3...6)
                }
                
                Section("Timing") {
                    DatePicker("Start Date & Time", selection: $selectedDate)
                        .datePickerStyle(.compact)
                    
                    Picker("Duration", selection: $duration) {
                        Text("30 minutes").tag(TimeInterval(1800))
                        Text("1 hour").tag(TimeInterval(3600))
                        Text("1.5 hours").tag(TimeInterval(5400))
                        Text("2 hours").tag(TimeInterval(7200))
                        Text("3 hours").tag(TimeInterval(10800))
                    }
                    .pickerStyle(.menu)
                }
            }
            .formStyle(.grouped)
            .padding()
        }
        .frame(width: 400, height: 300)
    }
}
