import SwiftUI

struct CustomCalendarView: View {
    @Binding var selectedDate: Date
    let events: [Event]
    var onBack: (() -> Void)? = nil
    var onEdit: ((Event) -> Void)? = nil
    var onDelete: ((Event) -> Void)? = nil

    @State private var currentMonth = Date()
    @State private var showingBackButton = false
    private let calendar = Calendar.current

    var body: some View {
        GeometryReader { geometry in
            HStack(alignment: .top, spacing: 2) {
                // Main Calendar Section
                VStack(spacing: 2) {
                    // Header with back button integration
                    headerView
                    
                    // Calendar Navigation
                    calendarNavigationView
                    
                    // Weekday Headers
                    weekdayHeadersView
                    
                    // Calendar Grid
                    calendarGridView
                }
                .frame(minWidth: min(400, geometry.size.width * 0.6))
                .animation(.easeInOut(duration: 0.3), value: currentMonth)
                
                // Events Sidebar
                if geometry.size.width > 700 {
                    eventsSidebarView
                        .frame(minWidth: 280, maxWidth: 350)
                        .transition(.move(edge: .trailing).combined(with: .opacity))
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
        }
        .onAppear {
            currentMonth = selectedDate
            showingBackButton = onBack != nil
        }
    }
    
    // MARK: - Header View
    private var headerView: some View {
        HStack {
            // Back Button with smooth integration
            if showingBackButton, let onBack = onBack {
                Button(action: {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        onBack()
                    }
                }) {
                    HStack(spacing: 6) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 14, weight: .semibold))
                        Text("Back")
                            .font(.system(size: 15, weight: .medium))
                    }
                    .foregroundColor(.blue)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.blue.opacity(0.1))
                            .stroke(.blue.opacity(0.2), lineWidth: 1)
                    )
                }
                .buttonStyle(.borderless)
                .transition(.move(edge: .leading).combined(with: .opacity))
            }
            
            Spacer()
            
            // Today Button
            Button(action: {
                withAnimation(.easeInOut(duration: 0.3)) {
                    currentMonth = Date()
                    selectedDate = Date()
                }
            }) {
                Text("Today")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.white.opacity(0.8))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(.white.opacity(0.1))
                            .stroke(.white.opacity(0.2), lineWidth: 1)
                    )
            }
            .buttonStyle(.borderless)
        }
        .padding(.bottom, 8)
    }
    
    // MARK: - Calendar Navigation
    private var calendarNavigationView: some View {
        HStack {
            Button(action: previousMonth) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(width: 40, height: 40)
                    .background(
                        Circle()
                            .fill(.white.opacity(0.1))
                            .stroke(.white.opacity(0.2), lineWidth: 1)
                    )
            }
            .buttonStyle(.borderless)
            .scaleEffect(0.95)
            .animation(.easeInOut(duration: 0.1), value: currentMonth)

            Spacer()

            Text(monthYearString)
                .font(.system(size: 28, weight: .bold))
                .foregroundColor(.white)
                .animation(.none, value: currentMonth)

            Spacer()

            Button(action: nextMonth) {
                Image(systemName: "chevron.right")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(width: 40, height: 40)
                    .background(
                        Circle()
                            .fill(.white.opacity(0.1))
                            .stroke(.white.opacity(0.2), lineWidth: 1)
                    )
            }
            .buttonStyle(.borderless)
            .scaleEffect(0.95)
            .animation(.easeInOut(duration: 0.1), value: currentMonth)
        }
        .padding(.horizontal, 8)
    }
    
    // MARK: - Weekday Headers
    private var weekdayHeadersView: some View {
        HStack(spacing: 0) {
            ForEach(calendar.shortWeekdaySymbols, id: \.self) { weekday in
                Text(weekday.uppercased())
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(.white.opacity(0.7))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(.white.opacity(0.08))
                .stroke(.white.opacity(0.15), lineWidth: 1)
        )
    }
    
    // MARK: - Calendar Grid
    private var calendarGridView: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 3), count: 7), spacing: 3) {
            ForEach(datesInMonth, id: \.self) { date in
                if calendar.isDate(date, equalTo: currentMonth, toGranularity: .month) {
                    CalendarDayView(
                        date: date,
                        isSelected: calendar.isDate(date, inSameDayAs: selectedDate),
                        isToday: calendar.isDateInToday(date),
                        events: eventsForDate(date),
                        onTap: {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                selectedDate = date
                            }
                        }
                    )
                } else {
                    Rectangle()
                        .fill(Color.clear)
                        .frame(height: 10)
                }
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(.white.opacity(0.05))
                .stroke(.white.opacity(0.1), lineWidth: 1)
        )
    }
    
    // MARK: - Events Sidebar
    private var eventsSidebarView: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Events Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Events")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                    
                    Text(selectedDate, formatter: dayFormatter)
                        .font(.subheadline)
                        .foregroundColor(.white.opacity(0.7))
                }
                
                Spacer()
                
                // Event count badge
                let eventCount = events.filter { calendar.isDate($0.date ?? Date(), inSameDayAs: selectedDate) }.count
                if eventCount > 0 {
                    Text("\(eventCount)")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            Capsule()
                                .fill(.blue.opacity(0.6))
                        )
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 16)
            .padding(.bottom, 12)
            
            Divider()
                .background(.white.opacity(0.2))
            
            // Events List
            ScrollView {
                LazyVStack(spacing: 8) {
                    let eventsForSelectedDate = events.filter { calendar.isDate($0.date ?? Date(), inSameDayAs: selectedDate) }
                    
                    if eventsForSelectedDate.isEmpty {
                        VStack(spacing: 12) {
                            Image(systemName: "calendar.badge.exclamationmark")
                                .font(.system(size: 24))
                                .foregroundColor(.white.opacity(0.4))
                            
                            Text("No events scheduled")
                                .font(.subheadline)
                                .foregroundColor(.white.opacity(0.6))
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 32)
                    } else {
                        ForEach(eventsForSelectedDate, id: \.id) { event in
                            EventRowView(
                                event: event,
                                onDelete: { onDelete?(event) }
                            )
                            .padding(.horizontal, 4)
                        }
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 16)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(.white.opacity(0.05))
                .stroke(.white.opacity(0.1), lineWidth: 1)
        )
    }

    // MARK: - Computed Properties
    private var monthYearString: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMMM yyyy"
        return formatter.string(from: currentMonth)
    }

    private var dayFormatter: DateFormatter {
        let f = DateFormatter()
        f.dateStyle = .full
        return f
    }

    private var datesInMonth: [Date] {
        guard let monthInterval = calendar.dateInterval(of: .month, for: currentMonth) else { return [] }
        let startOfMonth = monthInterval.start
        let startOfWeek = calendar.dateInterval(of: .weekOfYear, for: startOfMonth)?.start ?? startOfMonth
        var dates: [Date] = []
        var currentDate = startOfWeek
        for _ in 0..<42 {
            dates.append(currentDate)
            currentDate = calendar.date(byAdding: .day, value: 1, to: currentDate) ?? currentDate
        }
        return dates
    }

    private func eventsForDate(_ date: Date) -> [Event] {
        events.filter { event in
            guard let eventDate = event.date else { return false }
            return calendar.isDate(eventDate, inSameDayAs: date)
        }
    }

    private func previousMonth() {
        withAnimation(.easeInOut(duration: 0.3)) {
            currentMonth = calendar.date(byAdding: .month, value: -1, to: currentMonth) ?? currentMonth
        }
    }

    private func nextMonth() {
        withAnimation(.easeInOut(duration: 0.3)) {
            currentMonth = calendar.date(byAdding: .month, value: 1, to: currentMonth) ?? currentMonth
        }
    }
}

// MARK: - Calendar Day View
struct CalendarDayView: View {
    let date: Date
    let isSelected: Bool
    let isToday: Bool
    let events: [Event]
    let onTap: () -> Void
    
    @State private var isHovered = false
    
    private let dayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "d"
        return formatter
    }()
    
    var body: some View {
        Button(action: onTap) {
            VStack(spacing: 6) {
                // Day number
                Text(dayFormatter.string(from: date))
                    .font(.system(size: 16, weight: isSelected ? .bold : .medium))
                    .foregroundColor(dayTextColor)
                
                // Event indicators
                HStack(spacing: 2) {
                    ForEach(Array(events.prefix(3).enumerated()), id: \.offset) { index, event in
                        Circle()
                            .fill(colorForEventType(event.eventType ?? "Work"))
                            .frame(width: 5, height: 5)
                    }
                    
                    if events.count > 3 {
                        Text("+")
                            .font(.system(size: 8, weight: .bold))
                            .foregroundColor(.white.opacity(0.8))
                    }
                }
                .frame(height: 10)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 55)
            .background(dayBackgroundColor)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(dayBorderColor, lineWidth: dayBorderWidth)
            )
            .scaleEffect(isSelected ? 1.05 : (isHovered ? 1.02 : 1.0))
            .shadow(
                color: shadowColor,
                radius: shadowRadius,
                x: 0,
                y: shadowOffsetY
            )
        }
        .buttonStyle(.borderless)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHovered = hovering
            }
        }
    }
    
    private var dayTextColor: Color {
        if isSelected {
            return .white
        } else if isToday {
            return .blue
        } else {
            return .white.opacity(0.9)
        }
    }
    
    private var dayBackgroundColor: some ShapeStyle {
        if isSelected {
            return AnyShapeStyle(
                LinearGradient(
                    colors: [.blue.opacity(0.9), .purple.opacity(0.7)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
        } else if isToday {
            return AnyShapeStyle(Color.blue.opacity(0.2))
        } else if isHovered {
            return AnyShapeStyle(Color.white.opacity(0.15))
        } else if !events.isEmpty {
            return AnyShapeStyle(Color.white.opacity(0.08))
        } else {
            return AnyShapeStyle(Color.clear)
        }
    }
    
    private var dayBorderColor: Color {
        if isSelected {
            return .blue.opacity(0.6)
        } else if isToday {
            return .blue.opacity(0.8)
        } else if !events.isEmpty {
            return .white.opacity(0.25)
        } else {
            return .white.opacity(0.12)
        }
    }
    
    private var dayBorderWidth: CGFloat {
        if isSelected {
            return 2.5
        } else if isToday {
            return 2
        } else {
            return 1
        }
    }
    
    private var shadowColor: Color {
        if isSelected {
            return .blue.opacity(0.4)
        } else {
            return Color.clear
        }
    }
    
    private var shadowRadius: CGFloat {
        isSelected ? 10 : 0
    }
    
    private var shadowOffsetY: CGFloat {
        isSelected ? 5 : 0
    }
    
    private func colorForEventType(_ type: String) -> Color {
        switch type {
        case "Work": return .blue
        case "Birthday": return .pink
        case "Custom": return .yellow
        default: return .gray
        }
    }
}
