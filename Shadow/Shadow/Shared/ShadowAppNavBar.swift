import SwiftUI

struct ShadowAppNavBar: View {
    let title: String
    let subtitle: String
    let profile: UserProfile
    let onProfileTap: () -> Void
    let onLogout: () -> Void
    let showProfileMenu: Bool
    let onCalendarTap: () -> Void
    @ObservedObject var shadowBLEManager: ShadowBLEManager
    
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
            
            // BLE status indicator
            Button(action: { showBLEPopover.toggle() }) {
                HStack(spacing: 8) {
                    bleStatusIcon
                    
                    VStack(alignment: .trailing, spacing: 1) {
                        Text("Shadow BLE")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.9))
                        
                        Text(bleStatusText)
                            .font(.caption2)
                            .foregroundColor(bleStatusColor)
                            .fontWeight(.medium)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(.ultraThinMaterial)
                )
            }
            .popover(isPresented: $showBLEPopover) {
                ShadowBLEPopoverView(shadowBLEManager: shadowBLEManager)
            }
            
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
        switch shadowBLEManager.currentSystemStatus {
        case .synchronizing:
            Image(systemName: "bluetooth.fill")
                .foregroundColor(.green)
                .font(.caption)
                
        case .scanning:
            Image(systemName: "bluetooth")
                .foregroundColor(.orange)
                .font(.caption)
                .symbolEffect(.pulse, options: .repeating)
                
        case .connecting:
            Image(systemName: "bluetooth")
                .foregroundColor(.orange)
                .font(.caption)
                .symbolEffect(.pulse, options: .repeating)
                
        case .disconnected:
            Image(systemName: "bluetooth.slash")
                .foregroundColor(.gray)
                .font(.caption)
        }
    }
    
    private var bleStatusText: String {
        switch shadowBLEManager.currentSystemStatus {
        case .synchronizing:
            return "Syncing"
        case .scanning:
            return "Scanning"
        case .connecting:
            return "Connecting"
        case .disconnected:
            return "Disconnected"
        }
    }
    
    private var bleStatusColor: Color {
        switch shadowBLEManager.currentSystemStatus {
        case .synchronizing:
            return .green
        case .scanning, .connecting:
            return .orange
        case .disconnected:
            return .gray
        }
    }
}

struct ShadowBLEPopoverView: View {
    @ObservedObject var shadowBLEManager: ShadowBLEManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack {
                Image(systemName: "antenna.radiowaves.left.and.right")
                    .foregroundColor(.blue)
                    .font(.title2)
                
                Text("Shadow BLE Monitor")
                    .font(.headline)
                    .fontWeight(.semibold)
            }
            
            Divider()
            
            // Status section
            VStack(alignment: .leading, spacing: 8) {
                Label("System Status", systemImage: "info.circle")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(shadowBLEManager.currentSystemStatus.displayName)
                    .font(.body)
                    .foregroundColor(statusColor)
                    .padding(.leading, 20)
            }
            
            // Current state section
            if shadowBLEManager.connectedDevice != nil {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Current State", systemImage: "brain.head.profile")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    HStack {
                        Circle()
                            .fill(shadowBLEManager.lastStressEvent == "No recent events" ? .gray : .orange)
                            .frame(width: 12, height: 12)
                        
                        Text(shadowBLEManager.lastStressEvent)
                            .font(.body)
                        
                        Spacer()
                        
                        Text("Seq: \(shadowBLEManager.lastSequenceNumber)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.leading, 20)
                }
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
                        Text("\(shadowBLEManager.totalEventsReceived)")
                            .fontWeight(.medium)
                    }
                    
                    HStack {
                        Text("Devices Found:")
                        Spacer()
                        Text("\(shadowBLEManager.foundShadowDevices.count)")
                            .fontWeight(.medium)
                    }
                }
                .font(.caption)
                .padding(.leading, 20)
            }
            
            // Action buttons
            HStack(spacing: 12) {
                if shadowBLEManager.isScanning {
                    Button("Stop Scan") {
                        shadowBLEManager.stopScanning()
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button("Start Scan") {
                        shadowBLEManager.startScanning()
                    }
                    .buttonStyle(.bordered)
                }
                
                if shadowBLEManager.connectedDevice != nil {
                    Button("Disconnect") {
                        shadowBLEManager.disconnect()
                    }
                    .buttonStyle(.bordered)
                }
                
                Button("Clear Log") {
                    shadowBLEManager.clearDebugLog()
                }
                .buttonStyle(.bordered)
            }
            
            Divider()
            
            // Device list
            if !shadowBLEManager.foundShadowDevices.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Found Devices", systemImage: "list.bullet")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    ForEach(shadowBLEManager.foundShadowDevices) { device in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(device.name)
                                    .font(.caption)
                                    .fontWeight(.medium)
                                
                                Text("State: \(device.advertisedState.displayName)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            VStack(alignment: .trailing, spacing: 2) {
                                Text("Seq: \(device.advertisedSequence)")
                                    .font(.caption2)
                                
                                Text("\(device.rssi) dBm")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                            
                            if shadowBLEManager.connectedDevice?.id != device.peripheral.identifier {
                                Button("Connect") {
                                    shadowBLEManager.connectToDevice(device)
                                }
                                .buttonStyle(.borderedProminent)
                                .controlSize(.mini)
                            } else {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                    .font(.caption)
                            }
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 4)
                    }
                }
            }
        }
        .padding()
        .frame(minWidth: 300, maxWidth: 400)
    }
    
    private var statusColor: Color {
        switch shadowBLEManager.currentSystemStatus {
        case .synchronizing:
            return .green
        case .scanning, .connecting:
            return .orange
        case .disconnected:
            return .primary
        }
    }
}
