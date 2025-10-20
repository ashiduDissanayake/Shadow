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
                    .foregroundColor(.white)
                
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
            }
            
            Spacer()
            
            // Calendar button
            Button(action: onCalendarTap) {
                Image(systemName: "calendar")
                    .foregroundColor(.white)
                    .font(.title2)
            }
            .padding(.leading, 8)
            
            // Profile section
            if showProfileMenu {
                Button(action: onProfileTap) {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(LinearGradient(
                                gradient: Gradient(colors: [.blue, .purple]),
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ))
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
                            .foregroundColor(.white)
                    }
                }
                .padding(.leading, 8)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(
            LinearGradient(
                gradient: Gradient(colors: [
                    Color(red: 0.1, green: 0.15, blue: 0.25),
                    Color(red: 0.05, green: 0.08, blue: 0.15)
                ]),
                startPoint: .leading,
                endPoint: .trailing
            )
        )
    }
    
    @ViewBuilder
    private var bleStatusIcon: some View {
        if syncViewModel.isActive {
            Image(systemName: "bluetooth")
                .foregroundColor(.orange)
                .font(.caption)
                .symbolEffect(.pulse, options: .repeating)
        } else if syncViewModel.stateText == "Up to Date" {
            Image(systemName: "bluetooth.fill")
                .foregroundColor(.green)
                .font(.caption)
        } else {
            Image(systemName: "bluetooth.slash")
                .foregroundColor(.gray)
                .font(.caption)
        }
    }
    
    private var bleStatusText: String {
        return syncViewModel.stateText
    }
    
    private var bleStatusColor: Color {
        if syncViewModel.isActive {
            return .orange
        } else if syncViewModel.stateText == "Up to Date" {
            return .green
        } else {
            return .gray
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
                    .foregroundColor(.blue)
                
                VStack(alignment: .leading) {
                    Text("Shadow Monitor")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text("TinyML Stress Detection")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
            
            Divider()
            
            // Status section
            VStack(alignment: .leading, spacing: 8) {
                Label("System Status", systemImage: "info.circle")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
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
                
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Last Sync:")
                        Spacer()
                        Text(syncViewModel.lastSync)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Sequence:")
                        Spacer()
                        Text(syncViewModel.sequenceStatus)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.leading, 20)
            }
            
            // Statistics
            VStack(alignment: .leading, spacing: 8) {
                Label("Statistics", systemImage: "chart.bar")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Events Received:")
                        Spacer()
                        Text("\(syncViewModel.eventsReceived)")
                            .fontWeight(.medium)
                    }
                }
                .font(.caption)
                .padding(.leading, 20)
            }
            
            Divider()
            
            // Control buttons
            HStack(spacing: 12) {
                if syncViewModel.isActive {
                    Button("Stop Sync") {
                        syncViewModel.stop()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                } else {
                    Button("Start Sync") {
                        syncViewModel.start()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }
                
                Button("Refresh") {
                    // Refresh handled by view model
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding()
        .frame(width: 300)
    }
    
    private var statusColor: Color {
        if syncViewModel.isActive {
            return .orange
        } else if syncViewModel.stateText == "Up to Date" {
            return .green
        } else {
            return .gray
        }
    }
}
