import SwiftUI

struct DeviceConnectView: View {
    @ObservedObject var shadowBLEManager: ShadowBLEManager
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                Text("Shadow Automatic Monitor")
                    .font(.title2.bold())
                    .foregroundColor(.white)

                // Sync status card
                syncStatusCard
                
                // Current device status
                currentDeviceStatusCard
                
                // System data card  
                systemDataCard
                
                Spacer()
            }
            .padding()
            .background(shadowBackground)
            .navigationTitle("Shadow Sync Monitor")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") {
                        dismiss()
                    }
                    .foregroundColor(.white)
                }
            }
        }
        .onAppear {
            // Auto-flow starts automatically when view appears
            if shadowBLEManager.isBluetoothPoweredOn && !shadowBLEManager.isScanning {
                shadowBLEManager.startContinuousScanning()
            }
        }
    }
    
    private var syncStatusCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Circle()
                    .fill(syncStatusColor)
                    .frame(width: 12, height: 12)
                
                Text("Power-Efficient Sync Status")
                    .font(.headline)
                    .foregroundColor(.white)
                
                Spacer()
                
                Text(currentSyncStep)
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
            }
            
            // Power-efficient sync progress visualization
            syncProgressView
            
            Text(syncDescription)
                .font(.caption)
                .foregroundColor(.white.opacity(0.7))
                .multilineTextAlignment(.leading)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
    }
    
    private var syncProgressView: some View {
        HStack(spacing: 8) {
            ForEach(SyncStatus.allCases, id: \.self) { status in
                VStack(spacing: 4) {
                    Circle()
                        .fill(stepColor(for: status))
                        .frame(width: 12, height: 12)
                    
                    Text(status.shortName)
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.7))
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity)
                
                if status != SyncStatus.allCases.last {
                    Rectangle()
                        .fill(.white.opacity(0.3))
                        .frame(height: 1)
                        .frame(maxWidth: 20)
                }
            }
        }
    }
    
    private var currentDeviceStatusCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Circle()
                    .fill(deviceStatusColor)
                    .frame(width: 12, height: 12)
                
                Text("Current Device")
                    .font(.headline)
                    .foregroundColor(.white)
                
                Spacer()
                
                Text(shadowBLEManager.connectionStatus)
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
            }
            
            if let device = shadowBLEManager.connectedDevice {
                VStack(alignment: .leading, spacing: 8) {
                    statusRow("Device Name", device.name)
                    statusRow("Device State", device.advertisedState.displayName)
                    statusRow("Signal Strength", "\(device.rssi) dBm")
                }
            } else if let latestDevice = shadowBLEManager.foundShadowDevices.first {
                VStack(alignment: .leading, spacing: 8) {
                    statusRow("Latest Detected", latestDevice.name)
                    statusRow("Advertised State", latestDevice.advertisedState.displayName)
                    statusRow("Sequence Number", "\(latestDevice.advertisedSequence)")
                }
            } else {
                Text("Scanning for Shadow devices...")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.7))
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
    }
    
    private var systemDataCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Circle()
                    .fill(.green)
                    .frame(width: 12, height: 12)
                
                Text("System Data")
                    .font(.headline)
                    .foregroundColor(.white)
                
                Spacer()
            }
            
            VStack(alignment: .leading, spacing: 8) {
                statusRow("Current State", shadowBLEManager.lastStressEvent)
                statusRow("Sequence Number", "\(shadowBLEManager.lastSequenceNumber)")
                statusRow("Total Events", "\(shadowBLEManager.totalEventsReceived)")
                statusRow("System Status", shadowBLEManager.currentSystemStatus.displayName)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
    }
    
    private var currentSyncStep: String {
        shadowBLEManager.currentSyncStatus.displayName
    }
    
    private var syncDescription: String {
        switch shadowBLEManager.currentSyncStatus {
        case .scanning:
            return "Power-efficient scanning for sequence changes. ESP32 maintains ultra-low power advertisement mode."
        case .connecting:
            return "Sequence change detected! Rapidly connecting for immediate data sync."
        case .synchronizing:
            return "Connected. Performing ultra-fast handshake and data synchronization."
        case .disconnected:
            return "Sync complete. Disconnected to maximize ESP32 battery life. Returning to scan mode."
        }
    }
    
    private var syncStatusColor: Color {
        switch shadowBLEManager.currentSyncStatus {
        case .scanning:
            return .blue
        case .connecting:
            return .orange
        case .synchronizing:
            return .purple
        case .disconnected:
            return .green
        }
    }
    
    private var deviceStatusColor: Color {
        if shadowBLEManager.connectedDevice != nil {
            return .green
        } else if !shadowBLEManager.foundShadowDevices.isEmpty {
            return .orange
        } else {
            return .gray
        }
    }
    
    private func stepColor(for status: SyncStatus) -> Color {
        if status == shadowBLEManager.currentSyncStatus {
            return syncStatusColor
        } else if status.rawValue < shadowBLEManager.currentSyncStatus.rawValue {
            return .green
        } else {
            return .gray.opacity(0.3)
        }
    }
    
    private var shadowBackground: some View {
        LinearGradient(
            gradient: Gradient(colors: [
                Color(red: 0.05, green: 0.08, blue: 0.15),
                Color(red: 0.1, green: 0.15, blue: 0.25)
            ]),
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }
    
    private func statusRow(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title)
                .font(.caption)
                .foregroundColor(.white.opacity(0.7))
            
            Spacer()
            
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.white)
        }
    }
}
