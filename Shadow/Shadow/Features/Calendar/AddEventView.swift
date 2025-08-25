import SwiftUI

struct AddEventView: View {
    let eventTypes: [String]
    let onAdd: (String, String?, Date, Double, String, String?) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var title = ""
    @State private var notes = ""
    @State private var selectedDate = Date()
    @State private var duration: Double = 3600 // 1 hour
    @State private var eventType: String = "Work"
    @State private var customField: String = ""
    @State private var isHoveringCancel = false
    @State private var isHoveringAdd = false
    @FocusState private var titleFocused: Bool

    // Form validation
    private var isFormValid: Bool {
        !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(spacing: 0) {
            headerView
            bodyScrollView
        }
        .frame(width: 520, height: 680)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(Color.black.opacity(0.12), lineWidth: 0.5)
        )
        .shadow(color: .black.opacity(0.15), radius: 24, x: 0, y: 8)
        .shadow(color: .black.opacity(0.08), radius: 4, x: 0, y: 2)
        .onAppear {
            titleFocused = true
        }
    }

    private var bodyScrollView: some View {
        ScrollView {
            VStack(spacing: 32) {
                eventDetailsSection
                timingSection
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 24)
        }
        .background(
            ZStack {
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color(red: 0.98, green: 0.98, blue: 0.99),
                        Color(red: 0.94, green: 0.95, blue: 0.97)
                    ]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                Rectangle()
                    .fill(Color.black.opacity(0.015))
                    .blendMode(.overlay)
            }
        )
        .clipShape(
            UnevenRoundedRectangle(
                topLeadingRadius: 0,
                bottomLeadingRadius: 16,
                bottomTrailingRadius: 16,
                topTrailingRadius: 0
            )
        )
    }

    private var headerView: some View {
        HStack(spacing: 20) {
            VStack(alignment: .leading, spacing: 6) {
                Text("New Event")
                    .font(.system(size: 28, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)

                Text("Create and schedule your event")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            HStack(spacing: 12) {
                Button("Cancel") {
                    dismiss()
                }
                .buttonStyle(SecondaryButtonStyle(isHovered: $isHoveringCancel))
                .onHover { hovering in
                    withAnimation(.easeInOut(duration: 0.15)) {
                        isHoveringCancel = hovering
                    }
                }

                Button("Create Event") {
                    addEvent()
                }
                .buttonStyle(PrimaryButtonStyle(isHovered: $isHoveringAdd, isEnabled: isFormValid))
                .onHover { hovering in
                    withAnimation(.easeInOut(duration: 0.15)) {
                        isHoveringAdd = hovering
                    }
                }
                .disabled(!isFormValid)
            }
        }
        .padding(.horizontal, 32)
        .padding(.vertical, 24)
        .background(
            ZStack {
                Rectangle()
                    .fill(.ultraThinMaterial)
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color.white.opacity(0.8),
                        Color.white.opacity(0.4)
                    ]),
                    startPoint: .top,
                    endPoint: .bottom
                )
                .blendMode(.overlay)
            }
        )
        .overlay(
            Rectangle()
                .frame(height: 0.5)
                .foregroundStyle(.quaternary),
            alignment: .bottom
        )
        .clipShape(
            UnevenRoundedRectangle(
                topLeadingRadius: 16,
                bottomLeadingRadius: 0,
                bottomTrailingRadius: 0,
                topTrailingRadius: 16
            )
        )
    }

    private var eventDetailsSection: some View {
        VStack(alignment: .leading, spacing: 24) {
            sectionHeader("Event Details", icon: "doc.text.fill")
            eventDetailsFields
        }
        .padding(24)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(NSColor.underPageBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Color.secondary.opacity(0.2), lineWidth: 0.5)
                )
        )
        .shadow(color: .black.opacity(0.06), radius: 8, x: 0, y: 2)
    }

    private var eventDetailsFields: some View {
        VStack(spacing: 20) {
            // Title
            VStack(alignment: .leading, spacing: 8) {
                Text("Title")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.primary)

                TextField("Enter event title", text: $title)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color(NSColor.windowBackgroundColor))
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(
                                        titleFocused ? Color.accentColor.opacity(0.6) : Color.secondary.opacity(0.2),
                                        lineWidth: titleFocused ? 1.5 : 0.5
                                    )
                            )
                    )
                    .focused($titleFocused)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .strokeBorder(
                                titleFocused ? Color.accentColor.opacity(0.2) : .clear,
                                lineWidth: 3
                            )
                    )
            }

            // Event Type Picker
            VStack(alignment: .leading, spacing: 12) {
                Text("Category")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.primary)

                LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 10) {
                    ForEach(eventTypes, id: \.self) { type in
                        eventTypeButton(type: type)
                    }
                }
            }

            // Custom Field (if Custom type selected)
            if eventType == "Custom" {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Custom Description")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(.primary)

                    TextField("Describe your event", text: $customField)
                        .font(.system(size: 16, weight: .medium))
                        .foregroundStyle(.primary)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 14)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(Color(NSColor.windowBackgroundColor))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(Color.secondary.opacity(0.2), lineWidth: 0.5)
                                )
                        )
                }
                .transition(
                    .asymmetric(
                        insertion: .move(edge: .top).combined(with: .opacity),
                        removal: .move(edge: .top).combined(with: .opacity)
                    )
                )
            }

            // Notes Field
            VStack(alignment: .leading, spacing: 8) {
                Text("Notes")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.primary)

                TextField("Add notes (optional)", text: $notes, axis: .vertical)
                    .font(.system(size: 16, weight: .regular))
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color(NSColor.windowBackgroundColor))
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.secondary.opacity(0.2), lineWidth: 0.5)
                            )
                    )
                    .lineLimit(2...5)
            }
        }
    }

    private var timingSection: some View {
        VStack(alignment: .leading, spacing: 24) {
            sectionHeader("Schedule", icon: "calendar.badge.clock")

            VStack(spacing: 20) {
                // Date and Time Picker
                VStack(alignment: .leading, spacing: 8) {
                    Text("Date & Time")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(.primary)

                    DatePicker(
                        "Start Date & Time",
                        selection: $selectedDate,
                        displayedComponents: [.date, .hourAndMinute]
                    )
                    .datePickerStyle(.compact)
                    .labelsHidden()
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color(NSColor.windowBackgroundColor))
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.secondary.opacity(0.2), lineWidth: 0.5)
                            )
                    )
                }

                // Duration Picker
                VStack(alignment: .leading, spacing: 12) {
                    Text("Duration")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(.primary)

                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 10) {
                        durationButton(title: "30 min", value: 1800.0)
                        durationButton(title: "1 hour", value: 3600.0)
                        durationButton(title: "1.5 hours", value: 5400.0)
                        durationButton(title: "2 hours", value: 7200.0)
                        durationButton(title: "3 hours", value: 10800.0)
                        durationButton(title: "All day", value: 86400.0)
                    }
                }
            }
        }
        .padding(24)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(NSColor.underPageBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Color.secondary.opacity(0.2), lineWidth: 0.5)
                )
        )
        .shadow(color: .black.opacity(0.06), radius: 8, x: 0, y: 2)
    }

    private func sectionHeader(_ title: String, icon: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(Color.accentColor)
                .symbolRenderingMode(.hierarchical)

            Text(title)
                .font(.system(size: 20, weight: .semibold, design: .rounded))
                .foregroundStyle(.primary)

            Spacer()
        }
    }

    private func eventTypeButton(type: String) -> some View {
        Button(action: {
            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                eventType = type
            }
        }) {
            VStack(spacing: 8) {
                Image(systemName: iconForEventType(type))
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(eventType == type ? .white : colorForEventType(type))
                    .symbolRenderingMode(.hierarchical)

                Text(type)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(eventType == type ? .white : .primary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .padding(.horizontal, 12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(eventType == type ? colorForEventType(type) : Color(NSColor.windowBackgroundColor))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(
                                eventType == type ? colorForEventType(type).opacity(0.6) : Color.secondary.opacity(0.2),
                                lineWidth: eventType == type ? 1.5 : 0.5
                            )
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(
                        eventType == type ? colorForEventType(type).opacity(0.3) : .clear,
                        lineWidth: 2
                    )
            )
            .shadow(color: eventType == type ? colorForEventType(type).opacity(0.3) : .clear, radius: 8, x: 0, y: 2)
        }
        .buttonStyle(.borderless)
        .scaleEffect(eventType == type ? 1.02 : 1.0)
    }

    private func durationButton(title: String, value: Double) -> some View {
        Button(action: {
            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                duration = value
            }
        }) {
            Text(title)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(duration == value ? .white : .primary)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .padding(.horizontal, 8)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(duration == value ? Color.accentColor : Color(NSColor.windowBackgroundColor))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(
                                    duration == value ? Color.accentColor.opacity(0.6) : Color.secondary.opacity(0.2),
                                    lineWidth: duration == value ? 1.5 : 0.5
                                )
                        )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(
                            duration == value ? Color.accentColor.opacity(0.3) : .clear,
                            lineWidth: 2
                        )
                )
                .shadow(color: duration == value ? Color.accentColor.opacity(0.3) : .clear, radius: 6, x: 0, y: 2)
        }
        .buttonStyle(.borderless)
        .scaleEffect(duration == value ? 1.02 : 1.0)
    }

    private func iconForEventType(_ type: String) -> String {
        switch type {
        case "Work": return "briefcase.fill"
        case "Birthday": return "gift.fill"
        case "Custom": return "star.fill"
        default: return "calendar"
        }
    }

    private func colorForEventType(_ type: String) -> Color {
        switch type {
        case "Work": return .blue
        case "Birthday": return .pink
        case "Custom": return .orange
        default: return .gray
        }
    }

    private func addEvent() {
        let trimmedTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedNotes = notes.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedCustom = customField.trimmingCharacters(in: .whitespacesAndNewlines)

        onAdd(
            trimmedTitle,
            trimmedNotes.isEmpty ? nil : trimmedNotes,
            selectedDate,
            duration,
            eventType,
            eventType == "Custom" ? (trimmedCustom.isEmpty ? nil : trimmedCustom) : nil
        )

        dismiss()
    }
}

// MARK: - Button Styles

struct PrimaryButtonStyle: ButtonStyle {
    @Binding var isHovered: Bool
    let isEnabled: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 16, weight: .semibold))
            .foregroundStyle(isEnabled ? .white : .secondary)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(
                        isEnabled
                        ? (isHovered ? Color.accentColor.opacity(0.9) : Color.accentColor)
                        : Color.secondary.opacity(0.2)
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(
                        isEnabled && isHovered ? Color.white.opacity(0.2) : .clear,
                        lineWidth: 1
                    )
            )
            .shadow(
                color: isEnabled ? Color.accentColor.opacity(0.4) : .clear,
                radius: isHovered ? 8 : 4,
                x: 0,
                y: 2
            )
            .scaleEffect(configuration.isPressed ? 0.98 : (isHovered ? 1.02 : 1.0))
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
            .animation(.easeInOut(duration: 0.15), value: isHovered)
    }
}

struct SecondaryButtonStyle: ButtonStyle {
    @Binding var isHovered: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 16, weight: .medium))
            .foregroundStyle(.primary)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(isHovered ? Color.secondary.opacity(0.05) : Color.clear)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color.secondary.opacity(0.2), lineWidth: 0.5)
                    )
            )
            .scaleEffect(configuration.isPressed ? 0.98 : (isHovered ? 1.01 : 1.0))
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
            .animation(.easeInOut(duration: 0.15), value: isHovered)
    }
}
