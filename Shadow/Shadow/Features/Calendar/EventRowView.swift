import SwiftUI

struct EventRowView: View {
    let event: Event
    let onDelete: () -> Void
    @State private var isHovered = false
    @State private var showDeleteConfirmation = false

    var body: some View {
        HStack(spacing: 0) {
            // Color indicator
            RoundedRectangle(cornerRadius: 4)
                .fill(
                    LinearGradient(
                        colors: colorForEventType(event.eventType ?? "Work"),
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(width: 6, height: 75)
                .animation(.easeInOut(duration: 0.25), value: isHovered)

            // Main content
            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 10) {
                    // Title and event type badge
                    HStack(spacing: 12) {
                        Text(event.title ?? "Untitled Event")
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundColor(.shadowTextPrimary)
                            .lineLimit(2)

                        Spacer()
                        eventTypeBadge
                    }

                    HStack(spacing: 20) {
                        timeInfoView
                        durationInfoView
                    }

                    if let notes = event.notes, !notes.isEmpty {
                        Text(notes)
                            .font(.system(size: 13, weight: .regular))
                            .foregroundColor(.shadowTextSecondary)
                            .lineLimit(isHovered ? 3 : 2)
                            .padding(.top, 4)
                            .animation(.easeInOut(duration: 0.2), value: isHovered)
                    }

                    if event.eventType == "Custom", let custom = event.customField, !custom.isEmpty {
                        HStack(spacing: 6) {
                            Image(systemName: "star.fill")
                                .font(.system(size: 10))
                                .foregroundColor(.shadowAccent)
                            Text(custom)
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.shadowAccent)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.shadowAccentLight.opacity(0.5))
                        )
                    }
                }

                Spacer()

                // Delete button - always visible but styled based on platform/hover
                Button(action: { showDeleteConfirmation = true }) {
                    Image(systemName: "trash.fill")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.white)
                        .frame(width: 38, height: 38)
                        .background(
                            Circle()
                                .fill(
                                    LinearGradient(
                                        colors: deleteButtonColors,
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                        )
                        .shadow(color: deleteButtonShadowColor, radius: isHovered ? 7 : 3, x: 0, y: 3)
                }
                .buttonStyle(.borderless)
                .scaleEffect(deleteButtonScale)
                .opacity(deleteButtonOpacity)
                .animation(.easeInOut(duration: 0.18), value: isHovered)
            }
            .padding(.leading, 16)
            .padding(.trailing, 16)
            .padding(.vertical, 16)
        }
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(isHovered ? Color.shadowBackgroundSecondary : Color.shadowSurface.opacity(0.6))
                .stroke(isHovered ? Color.shadowBorder : Color.shadowBorderLight, lineWidth: 1)
                .shadow(color: isHovered ? Color.shadowElevation2 : Color.clear, radius: isHovered ? 8 : 0, x: 0, y: isHovered ? 4 : 0)
        )
        .scaleEffect(isHovered ? 1.02 : 1.0)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.3)) {
                isHovered = hovering
            }
        }
        .confirmationDialog(
            "Delete Event",
            isPresented: $showDeleteConfirmation,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                onDelete()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Are you sure you want to delete \"\(event.title ?? "this event")\"? This action cannot be undone.")
        }
    }

    // MARK: - Delete Button Styling Properties
    private var deleteButtonColors: [Color] {
#if os(macOS)
        // On macOS, use subtle styling when not hovered, prominent when hovered
        isHovered ? [Color.shadowError, Color.shadowError.opacity(0.7)] : [Color.shadowTextTertiary.opacity(0.3), Color.shadowTextTertiary.opacity(0.2)]
#else
        // On iOS/touchOS, always use visible but subtle styling
        [Color.shadowError.opacity(0.8), Color.shadowError.opacity(0.6)]
#endif
    }
    
    private var deleteButtonShadowColor: Color {
#if os(macOS)
        isHovered ? Color.shadowError.opacity(0.3) : .clear
#else
        Color.shadowError.opacity(0.2)
#endif
    }
    
    private var deleteButtonScale: CGFloat {
#if os(macOS)
        isHovered ? 1.0 : 0.85
#else
        1.0
#endif
    }
    
    private var deleteButtonOpacity: Double {
#if os(macOS)
        isHovered ? 1.0 : 0.6
#else
        0.8
#endif
    }

    private var eventTypeBadge: some View {
        HStack(spacing: 4) {
            Image(systemName: iconForEventType(event.eventType ?? "Work"))
                .font(.system(size: 7, weight: .medium))
            Text(event.eventType ?? "Work")
                .font(.system(size: 8, weight: .medium))
        }
        .foregroundColor(.shadowTextSecondary)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.shadowBackgroundSecondary)
                .stroke(Color.shadowBorder, lineWidth: 0.5)
        )
    }

    private var timeInfoView: some View {
        HStack(spacing: 6) {
            Image(systemName: "clock.fill")
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.shadowPrimary)
            VStack(alignment: .leading, spacing: 2) {
                Text(event.date ?? Date(), style: .time)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.shadowTextPrimary)
                Text(event.date ?? Date(), style: .date)
                    .font(.system(size: 11, weight: .regular))
                    .foregroundColor(.shadowTextSecondary)
            }
        }
    }

    private var durationInfoView: some View {
        HStack(spacing: 6) {
            Image(systemName: "hourglass")
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.shadowAccent)
            Text(formatDuration(event.duration))
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(.shadowTextPrimary)
        }
    }

    private func colorForEventType(_ type: String) -> [Color] {
        switch type {
        case "Work":
            return [.shadowPrimary, .shadowPrimaryLight]
        case "Birthday":
            return [.shadowSecondary, .shadowSecondaryLight]
        case "Custom":
            return [.shadowAccent, .shadowAccentLight]
        default:
            return [.shadowTextTertiary, .shadowTextTertiary.opacity(0.6)]
        }
    }

    private func iconForEventType(_ type: String) -> String {
        switch type {
        case "Work": return "briefcase.fill"
        case "Birthday": return "gift.fill"
        case "Custom": return "star.fill"
        default: return "calendar"
        }
    }

    private func formatDuration(_ duration: Double) -> String {
        let hours = Int(duration) / 3600
        let minutes = (Int(duration) % 3600) / 60

        if hours > 0 && minutes > 0 {
            return "\(hours)h \(minutes)m"
        } else if hours > 0 {
            return "\(hours)h"
        } else if minutes > 0 {
            return "\(minutes)m"
        } else {
            return "30m"
        }
    }
}
