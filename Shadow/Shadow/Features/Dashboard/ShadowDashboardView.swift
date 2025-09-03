import SwiftUI

struct ShadowDashboardView: View {
    let profile: UserProfile
    @ObservedObject var shadowBLEManager: ShadowBLEManager
    let onLogout: () -> Void
    let onDeleteAccount: () -> Void
    let onShowProfile: () -> Void
    
    @State private var showingDebugLog = false
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header section
                headerSection
                
                // Shadow monitoring status
                shadowStatusSection
                
                // Recent stress events
                if shadowBLEManager.totalEventsReceived > 0 {
                    recentEventsSection
                }
                
                // Debug section
                debugSection
                
                Spacer(minLength: 20)
            }
            .padding(.horizontal, 20)
            .padding(.top, 10)
        }
        .background(
            LinearGradient(
                gradient: Gradient(colors: [
                    Color(red: 0.05, green: 0.08, blue: 0.15),
                    Color(red: 0.1, green: 0.15, blue: 0.25)
                ]),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
        )
        .sheet(isPresented: $showingDebugLog) {
            ShadowDebugLogView(shadowBLEManager: shadowBLEManager)
        }
        .onAppear {
            // Start scanning when dashboard appears
            if shadowBLEManager.isBluetoothPoweredOn && !shadowBLEManager.isScanning {
                shadowBLEManager.startScanning()
            }
        }
    }
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Welcome back, \(profile.name ?? "User")!")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                
                Spacer()
                
                Button(action: onShowProfile) {
                    Image(systemName: "person.circle")
                        .font(.title2)
                        .foregroundColor(.white.opacity(0.8))
                }
            }
            
            Text("Shadow stress monitoring dashboard")
                .font(.subheadline)
                .foregroundColor(.white.opacity(0.7))
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
    }
    
    private var shadowStatusSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundColor(.blue)
                
                Text("Shadow Monitoring")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                
                Spacer()
                
                statusIndicator
            }
            
            // Current status details
            VStack(spacing: 12) {
                statusRow("System Status", shadowBLEManager.currentSystemStatus.displayName, systemColor: systemStatusColor)
                
                if shadowBLEManager.connectedDevice != nil {
                    statusRow("Current State", shadowBLEManager.lastStressEvent, systemColor: .orange)
                    statusRow("Sequence Number", "\(shadowBLEManager.lastSequenceNumber)", systemColor: .secondary)
                }
                
                statusRow("Devices Found", "\(shadowBLEManager.foundShadowDevices.count)", systemColor: .secondary)
                statusRow("Events Received", "\(shadowBLEManager.totalEventsReceived)", systemColor: .secondary)
            }
            
            // Action buttons
            HStack(spacing: 12) {
                if shadowBLEManager.isScanning {
                    Button("Stop Scanning") {
                        shadowBLEManager.stopScanning()
                    }
                    .buttonStyle(ShadowButtonStyle(color: .orange))
                } else {
                    Button("Start Scanning") {
                        shadowBLEManager.startScanning()
                    }
                    .buttonStyle(ShadowButtonStyle(color: .blue))
                }
                
                if shadowBLEManager.connectedDevice != nil {
                    Button("Disconnect") {
                        shadowBLEManager.disconnect()
                    }
                    .buttonStyle(ShadowButtonStyle(color: .red))
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
    }
    
    private var recentEventsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "clock.arrow.circlepath")
                    .font(.title3)
                    .foregroundColor(.green)
                
                Text("Recent Activity")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                
                Spacer()
                
                Text("\(shadowBLEManager.totalEventsReceived) events")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.7))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(.ultraThinMaterial)
                    )
            }
            
            VStack(spacing: 8) {
                ForEach(Array(shadowBLEManager.foundShadowDevices.prefix(3))) { device in
                    RecentEventRow(device: device)
                }
                
                if shadowBLEManager.foundShadowDevices.isEmpty {
                    Text("No recent activity")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.6))
                        .italic()
                        .padding(.vertical, 8)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
    }
    
    private var debugSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "terminal")
                    .font(.title3)
                    .foregroundColor(.purple)
                
                Text("System Debug")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                
                Spacer()
                
                Button("View Full Log") {
                    showingDebugLog = true
                }
                .buttonStyle(ShadowButtonStyle(color: .purple, size: .small))
            }
            
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(shadowBLEManager.debugLog.suffix(3))) { logEntry in
                    Text(logEntry.formattedMessage)
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.8))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(.black.opacity(0.3))
                        )
                }
                
                if shadowBLEManager.debugLog.isEmpty {
                    Text("No debug messages")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.6))
                        .italic()
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
    }
    
    @ViewBuilder
    private var statusIndicator: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(systemStatusColor)
                .frame(width: 8, height: 8)
            
            Text(shadowBLEManager.connectionStatus)
                .font(.caption)
                .foregroundColor(.white.opacity(0.8))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(.ultraThinMaterial)
        )
    }
    
    private var systemStatusColor: Color {
        switch shadowBLEManager.currentSystemStatus {
        case .synchronizing:
            return .green
        case .scanning, .connecting:
            return .orange
        case .disconnected:
            return .gray
        }
    }
    
    private func statusRow(_ title: String, _ value: String, systemColor: Color? = nil) -> some View {
        HStack {
            Text(title)
                .font(.caption)
                .foregroundColor(.white.opacity(0.7))
            
            Spacer()
            
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(systemColor ?? .white)
        }
    }
    
    private func timeAgo(_ date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}

struct ShadowDebugLogView: View {
    @ObservedObject var shadowBLEManager: ShadowBLEManager
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(shadowBLEManager.debugLog) { logEntry in
                        Text(logEntry.formattedMessage)
                            .font(.caption)
                            .foregroundColor(.primary)
                            .textSelection(.enabled)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(.ultraThinMaterial)
                            )
                    }
                    
                    if shadowBLEManager.debugLog.isEmpty {
                        Text("No debug messages")
                            .font(.body)
                            .foregroundColor(.secondary)
                            .italic()
                            .padding()
                    }
                }
                .padding()
            }
            .navigationTitle("Debug Log")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .primaryAction) {
                    Button("Clear") {
                        shadowBLEManager.clearDebugLog()
                    }
                }
            }
        }
    }
}

struct ShadowButtonStyle: ButtonStyle {
    let color: Color
    let size: ButtonSize
    
    enum ButtonSize {
        case normal, small
        
        var padding: EdgeInsets {
            switch self {
            case .normal:
                return EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16)
            case .small:
                return EdgeInsets(top: 4, leading: 8, bottom: 4, trailing: 8)
            }
        }
        
        var font: Font {
            switch self {
            case .normal:
                return .caption
            case .small:
                return .caption2
            }
        }
    }
    
    init(color: Color, size: ButtonSize = .normal) {
        self.color = color
        self.size = size
    }
    
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(size.font)
            .fontWeight(.medium)
            .foregroundColor(.white)
            .padding(size.padding)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(color.opacity(configuration.isPressed ? 0.7 : 1.0))
            )
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

struct RecentEventRow: View {
    let device: DiscoveredShadowDevice
    
    var body: some View {
        HStack {
            Circle()
                .fill(device.advertisedState == .synchronizing ? .green : .gray)
                .frame(width: 10, height: 10)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(device.advertisedState.displayName)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                
                Text("Sequence \(device.advertisedSequence)")
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.6))
            }
            
            Spacer()
            
            Text(timeAgo(device.lastSeen))
                .font(.caption2)
                .foregroundColor(.white.opacity(0.6))
        }
        .padding(.vertical, 4)
    }
    
    private func timeAgo(_ date: Date) -> String {
        let interval = Date().timeIntervalSince(date)
        
        if interval < 60 {
            return "\(Int(interval))s ago"
        } else if interval < 3600 {
            return "\(Int(interval / 60))m ago"
        } else {
            return "\(Int(interval / 3600))h ago"
        }
    }
}
