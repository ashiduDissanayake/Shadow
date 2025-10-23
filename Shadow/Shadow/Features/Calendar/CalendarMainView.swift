import SwiftUI

struct CalendarMainView: View {
    @ObservedObject var viewModel: CalendarViewModel
    @Binding var showingCalendar: Bool     // <-- Now a binding!
    @State private var showingAddEvent = false

    private let eventTypes = ["All", "Work", "Birthday", "Custom"]

    var body: some View {
        ZStack {
            // The background (gradient)
            Color.shadowWellnessGradient()
                .ignoresSafeArea()

            if showingCalendar {
                // Dismissal overlay - covers entire background
                Color.clear
                    .ignoresSafeArea()
                    .contentShape(Rectangle())
                    .onTapGesture {
                        withAnimation(.easeInOut(duration: 0.28)) {
                            showingCalendar = false
                        }
                    }

                // Main Calendar UI Container
                VStack(spacing: 0) {
                    headerView
                    
                    // Main content area with proper spacing and sizing
                    HStack(alignment: .top, spacing: 2) {
                        // Left sidebar - Event types and stats
                        sidebarView
                            .frame(width: 260)
                        
                        // Center calendar view
                        CustomCalendarView(
                            selectedDate: $viewModel.selectedDate,
                            events: viewModel.events
                        )
                        .frame(width: 450, height: 300)
                        
                        // Right sidebar - Events list
                        eventsSidebar
                            .frame(width: 300)
                    }
                    .padding(.horizontal, 32)
                    .padding(.bottom, 32)
                }
                .frame(maxWidth: 1100, maxHeight: 800)
                .background(
                    RoundedRectangle(cornerRadius: 20)
                        .fill(Color.shadowSurface.opacity(0.95))
                        .shadow(color: Color.shadowElevation3, radius: 30, x: 0, y: 10)
                )
                .background(
                    // Invisible tap blocker
                    Color.clear
                        .contentShape(Rectangle())
                        .onTapGesture { /* Prevents background tap from propagating */ }
                )
                .sheet(isPresented: $showingAddEvent) {
                    AddEventView(eventTypes: Array(eventTypes.dropFirst())) { title, notes, date, duration, eventType, customField in
                        viewModel.addEvent(
                            title: title,
                            notes: notes,
                            date: date,
                            duration: duration,
                            eventType: eventType,
                            customField: customField
                        )
                    }
                }
                .zIndex(1)
            }
        }
    }
    
    private var headerView: some View {
        HStack {
            // Back button when calendar is open
            Button(action: {
                withAnimation(.easeInOut(duration: 0.28)) {
                    showingCalendar = false
                }
            }) {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 18, weight: .medium))
                }
                .foregroundColor(.shadowPrimary)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.shadowBackgroundSecondary)
                )
            }
            .buttonStyle(.borderless)
            .padding(.trailing, 8)

            VStack(alignment: .leading, spacing: 8) {
                Text("Calendar")
                    .font(.system(size: 38, weight: .bold, design: .rounded))
                    .foregroundColor(.shadowTextPrimary)
                
                if let user = viewModel.profile {
                    HStack(spacing: 10) {
                        Circle()
                            .fill(Color.shadowPrimaryGradient())
                            .frame(width: 8, height: 8)
                        
                        Text("Welcome back, \(user.name ?? "User")")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundColor(.shadowTextSecondary)
                    }
                }
            }
            
            Spacer()
            
            Button(action: {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                    showingAddEvent = true
                }
            }) {
                HStack(spacing: 10) {
                    Image(systemName: "plus")
                        .font(.system(size: 16, weight: .semibold))
                    Text("New Event")
                        .font(.system(size: 16, weight: .semibold))
                }
                .foregroundColor(.white)
                .padding(.horizontal, 24)
                .padding(.vertical, 14)
                .background(Color.shadowPrimaryGradient())
                .clipShape(Capsule())
                .shadow(color: Color.shadowPrimary.opacity(0.4), radius: 10, x: 0, y: 5)
            }
            .buttonStyle(.borderless)
            .scaleEffect(showingAddEvent ? 0.95 : 1.0)
        }
        .padding(.horizontal, 32)
        .padding(.top, 32)
        .padding(.bottom, 28)
    }

    
    private var sidebarView: some View {
        VStack(alignment: .leading, spacing: 28) {
            // Event Type Filter
            VStack(alignment: .leading, spacing: 16) {
                Text("Event Types")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(.shadowTextPrimary)
                
                VStack(spacing: 6) {
                    ForEach(eventTypes, id: \.self) { type in
                        eventTypeButton(type: type)
                    }
                }
            }
            
            Spacer()
            
            // Quick Stats
            quickStatsView
        }
        .padding(24)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.shadowBackgroundSecondary)
                .stroke(Color.shadowBorder, lineWidth: 1)
                .shadow(color: Color.shadowElevation2, radius: 10, x: 0, y: 4)
        )
    }
    
    private var eventsSidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Events")
                        .font(.system(size: 20, weight: .bold))
                        .foregroundColor(.shadowTextPrimary)
                    Text(viewModel.selectedDate, formatter: dayFormatter)
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.shadowTextSecondary)
                }
                Spacer()
                let eventCount = viewModel.filteredEvents.count
                if eventCount > 0 {
                    Text("\(eventCount)")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            Capsule()
                                .fill(Color.shadowPrimary)
                        )
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 20)
            .padding(.bottom, 16)

            Divider()
                .background(Color.shadowBorder)
                .padding(.horizontal, 20)

            // Events scroll view
            ScrollView {
                LazyVStack(spacing: 12) {
                    if viewModel.filteredEvents.isEmpty {
                        VStack(spacing: 16) {
                            Image(systemName: "calendar.badge.exclamationmark")
                                .font(.system(size: 28))
                                .foregroundColor(.shadowTextTertiary)
                            VStack(spacing: 6) {
                                Text("No events scheduled")
                                    .font(.system(size: 16, weight: .medium))
                                    .foregroundColor(.shadowTextSecondary)
                                Text("for this day")
                                    .font(.system(size: 14, weight: .regular))
                                    .foregroundColor(.shadowTextTertiary)
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 40)
                    } else {
                        ForEach(viewModel.filteredEvents, id: \.id) { event in
                            EventRowView(event: event) {
                                withAnimation(.easeInOut(duration: 0.3)) {
                                    viewModel.deleteEvent(event)
                                }
                            }
                            .padding(.horizontal, 4)
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 20)
            }
            .frame(maxHeight: 500)
        }
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.shadowBackgroundSecondary)
                .stroke(Color.shadowBorder, lineWidth: 1)
                .shadow(color: Color.shadowElevation2, radius: 10, x: 0, y: 4)
        )
    }

    private var dayFormatter: DateFormatter {
        let f = DateFormatter()
        f.dateStyle = .full
        return f
    }
    
    private func eventTypeButton(type: String) -> some View {
        Button(action: {
            withAnimation(.easeInOut(duration: 0.2)) {
                viewModel.selectedEventType = type
            }
        }) {
            HStack(spacing: 12) {
                Image(systemName: iconForEventType(type))
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(viewModel.selectedEventType == type ? .shadowPrimary : .shadowTextSecondary)
                    .frame(width: 20)
                
                Text(type)
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(viewModel.selectedEventType == type ? .shadowTextPrimary : .shadowTextSecondary)
                
                Spacer()
                
                if viewModel.selectedEventType == type {
                    Circle()
                        .fill(Color.shadowPrimary)
                        .frame(width: 6, height: 6)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(viewModel.selectedEventType == type ? Color.shadowPrimaryLight.opacity(0.2) : Color.clear)
                    .stroke(viewModel.selectedEventType == type ? Color.shadowPrimary.opacity(0.3) : Color.clear, lineWidth: 1)
            )
        }
        .buttonStyle(.borderless)
    }
    
    private func iconForEventType(_ type: String) -> String {
        switch type {
        case "All": return "calendar"
        case "Work": return "briefcase"
        case "Birthday": return "gift"
        case "Custom": return "star"
        default: return "calendar"
        }
    }
    
    private var quickStatsView: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Today's Overview")
                .font(.system(size: 16, weight: .semibold))
                .foregroundColor(.shadowTextPrimary)
            
            VStack(spacing: 12) {
                HStack {
                    Text("Events today")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.shadowTextSecondary)
                    Spacer()
                    Text("\(todaysEventCount)")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundColor(.shadowPrimary)
                }
                
                HStack {
                    Text("This week")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.shadowTextSecondary)
                    Spacer()
                    Text("\(thisWeekEventCount)")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundColor(.shadowAccent)
                }
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.shadowSurface)
                    .stroke(Color.shadowBorder, lineWidth: 1)
            )
        }
    }
    
    private var todaysEventCount: Int {
        viewModel.filteredEvents.filter { event in
            Calendar.current.isDate(event.date ?? Date(), inSameDayAs: Date())
        }.count
    }
    
    private var thisWeekEventCount: Int {
        let calendar = Calendar.current
        let startOfWeek = calendar.dateInterval(of: .weekOfYear, for: Date())?.start ?? Date()
        let endOfWeek = calendar.dateInterval(of: .weekOfYear, for: Date())?.end ?? Date()
        
        return viewModel.events.filter { event in
            guard let eventDate = event.date else { return false }
            return eventDate >= startOfWeek && eventDate <= endOfWeek
        }.count
    }
}
