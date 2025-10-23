import SwiftUI

struct ShadowAppNavBar: View {
    let title: String
    let subtitle: String
    let profile: UserProfile
    let onProfileTap: () -> Void
    let onLogout: () -> Void
    let showProfileMenu: Bool
    let onCalendarTap: () -> Void
    @ObservedObject var syncViewModel: SyncDashboardViewModel
    
    @State private var showBLEPopover = false
    
    var body: some View {
        HStack {
            // Title section
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.shadowTextPrimary)
                
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.shadowTextSecondary)
            }
            
            Spacer()
            
            // Calendar button
            Button(action: onCalendarTap) {
                Image(systemName: "calendar")
                    .foregroundColor(.shadowPrimary)
                    .font(.title2)
            }
            .padding(.leading, 8)
            
            // Profile section
            if showProfileMenu {
                Button(action: onProfileTap) {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(Color.shadowPrimaryGradient())
                            .frame(width: 32, height: 32)
                            .overlay(
                                Text(profile.name?.prefix(1).uppercased() ?? "U")
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .foregroundColor(.white)
                            )
                        
                        Text(profile.name ?? "User")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .foregroundColor(.shadowTextPrimary)
                    }
                }
                .padding(.leading, 8)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Color.shadowSurface)
        .shadow(color: Color.shadowElevation1, radius: 4, x: 0, y: 2)
    }
    
    @ViewBuilder
    private var bleStatusIcon: some View {
        if syncViewModel.isActive {
            Image(systemName: "bluetooth")
                .foregroundColor(.shadowWarning)
                .font(.caption)
                .symbolEffect(.pulse, options: .repeating)
        } else if syncViewModel.stateText == "Up to Date" {
            Image(systemName: "bluetooth.fill")
                .foregroundColor(.shadowSuccess)
                .font(.caption)
        } else {
            Image(systemName: "bluetooth.slash")
                .foregroundColor(.shadowTextTertiary)
                .font(.caption)
        }
    }
    
    private var bleStatusText: String {
        return syncViewModel.stateText
    }
    
    private var bleStatusColor: Color {
        if syncViewModel.isActive {
            return .shadowWarning
        } else if syncViewModel.stateText == "Up to Date" {
            return .shadowSuccess
        } else {
            return .shadowTextTertiary
        }
    }
}

struct ShadowBLEPopoverView: View {
    @ObservedObject var syncViewModel: SyncDashboardViewModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundColor(.shadowPrimary)
                
                VStack(alignment: .leading) {
                    Text("Shadow Monitor")
                        .font(.headline)
                        .fontWeight(.semibold)
                        .foregroundColor(.shadowTextPrimary)
                    
                    Text("TinyML Stress Detection")
                        .font(.caption)
                        .foregroundColor(.shadowTextSecondary)
                }
                
                Spacer()
            }
            
            Divider()
                .background(Color.shadowBorder)
            
            // Status section
            VStack(alignment: .leading, spacing: 8) {
                Label("System Status", systemImage: "info.circle")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.shadowTextPrimary)
                
                Text(syncViewModel.stateText)
                    .font(.body)
                    .foregroundColor(statusColor)
                    .padding(.leading, 20)
            }
            
            // Sync Info section
            VStack(alignment: .leading, spacing: 8) {
                Label("Sync Info", systemImage: "arrow.clockwise")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.shadowTextPrimary)
                
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Last Sync:")
                            .foregroundColor(.shadowTextSecondary)
                        Spacer()
                        Text(syncViewModel.lastSync)
                            .font(.caption)
                            .foregroundColor(.shadowTextSecondary)
                    }
                    
                    HStack {
                        Text("Sequence:")
                            .foregroundColor(.shadowTextSecondary)
                        Spacer()
                        Text(syncViewModel.sequenceStatus)
                            .font(.caption)
                            .foregroundColor(.shadowTextSecondary)
                    }
                }
                .padding(.leading, 20)
            }
            
            // Statistics
            VStack(alignment: .leading, spacing: 8) {
                Label("Statistics", systemImage: "chart.bar")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.shadowTextPrimary)
                
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Events Received:")
                            .foregroundColor(.shadowTextSecondary)
                        Spacer()
                        Text("\(syncViewModel.eventsReceived)")
                            .fontWeight(.medium)
                            .foregroundColor(.shadowTextPrimary)
                    }
                }
                .font(.caption)
                .padding(.leading, 20)
            }
            
            Divider()
                .background(Color.shadowBorder)
            
            // Control buttons
            HStack(spacing: 12) {
                if syncViewModel.isActive {
                    Button("Stop Sync") {
                        syncViewModel.stop()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .tint(.shadowWarning)
                } else {
                    Button("Start Sync") {
                        syncViewModel.start()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .tint(.shadowPrimary)
                }
                
                Button("Refresh") {
                    // Refresh handled by view model
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .tint(.shadowSuccess)
            }
        }
        .padding()
        .frame(width: 300)
        .background(Color.shadowSurface)
    }
    
    private var statusColor: Color {
        if syncViewModel.isActive {
            return .shadowWarning
        } else if syncViewModel.stateText == "Up to Date" {
            return .shadowSuccess
        } else {
            return .shadowTextTertiary
        }
    }
}
