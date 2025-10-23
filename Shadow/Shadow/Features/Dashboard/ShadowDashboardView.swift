import SwiftUI
import Combine
import CoreData

struct ShadowDashboardView: View {
    let profile: UserProfile
    @StateObject private var syncViewModel = SyncDashboardViewModel()
    let onLogout: () -> Void
    let onDeleteAccount: () -> Void
    let onShowProfile: () -> Void
    
    @State private var showingDebugLog = false
    @State private var recentEvents: [StressEvent] = []
    @State private var graphEvents: [StressEvent] = []
    @State private var showQRScanner = false
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerSection
                shadowStatusSection
                
                if !graphEvents.isEmpty {
                    stressGraphSection
                }
                
                Spacer(minLength: 20)
            }
            .padding(.horizontal, 20)
            .padding(.top, 10)
        }
        .background(
            Color.shadowWellnessGradient()
            .ignoresSafeArea()
        )
        .sheet(isPresented: $showingDebugLog) {
            ShadowDebugLogView(syncViewModel: syncViewModel)
        }
        .sheet(isPresented: $showQRScanner) {
            QRScannerView(onDeviceScanned: { deviceName in
                print("✅ Device paired: \(deviceName)")
                showQRScanner = false
                syncViewModel.start()
            })
        }
        .onAppear {
            syncViewModel.start()
            recentEvents = syncViewModel.getRecentEvents()
            graphEvents = syncViewModel.getEventsInLastHours(3)
            print("📊 [Dashboard] Initial load: \(recentEvents.count) recent, \(graphEvents.count) graph events")
        }
        .onReceive(syncViewModel.$eventUpdateTrigger) { uuid in
            recentEvents = syncViewModel.getRecentEvents()
            graphEvents = syncViewModel.getEventsInLastHours(3)
            print("📊 [Dashboard] UI update triggered (uuid=\(uuid)): \(recentEvents.count) recent, \(graphEvents.count) graph events")
        }
    }
    
    // MARK: - Graph Section
    private var stressGraphSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .font(.title3)
                    .foregroundColor(.shadowAccent)
                Text("Stress Timeline (Last 3 Hours)")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.shadowTextPrimary)
                Spacer()
                Text("\(graphEvents.count) events")
                    .font(.caption)
                    .foregroundColor(.shadowTextSecondary)
            }
            
            StressStateGraphView.chartView(for: StressStateGraphView.fromCoreData(graphEvents))
                .frame(height: 200)
                .id(graphEvents.map { "\($0.sequenceNumber)-\($0.stressState)" }.joined())
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.shadowSurface)
                .shadow(color: Color.shadowElevation2, radius: 8, x: 0, y: 2)
        )
    }
    
    // MARK: Header
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Welcome back, \(profile.name ?? "User")!")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.shadowTextPrimary)
            }
            Text("Your wellness companion is here to support you")
                .font(.subheadline)
                .foregroundColor(.shadowTextSecondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.shadowSurface)
                .shadow(color: Color.shadowElevation1, radius: 6, x: 0, y: 2)
        )
    }
    
    // MARK: Status Section
    private var shadowStatusSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundColor(.shadowPrimary)
                Text("Shadow Monitoring")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.shadowTextPrimary)
                Spacer()
                statusIndicator
            }
            
            VStack(spacing: 12) {
                statusRow("System Status", syncViewModel.stateText, systemColor: systemStatusColor)
                statusRow("Last Sync", syncViewModel.lastSync, systemColor: .shadowTextSecondary)
                statusRow("Events Received", "\(syncViewModel.eventsReceived)", systemColor: .shadowTextSecondary)
                
                Divider()
                    .background(Color.shadowBorder)
                
                // Pairing Section
                pairingSection
            }
            
            HStack(spacing: 12) {
                if syncViewModel.isActive {
                    Button("Stop") { syncViewModel.stop() }
                        .buttonStyle(ShadowButtonStyle(color: .shadowWarning))
                } else {
                    Button("Start") { syncViewModel.start() }
                        .buttonStyle(ShadowButtonStyle(color: .shadowPrimary))
                }
                
                Button("Refresh") {
                    recentEvents = syncViewModel.getRecentEvents()
                }
                .buttonStyle(ShadowButtonStyle(color: .shadowSuccess))
                
                if syncViewModel.manager.isPairedToDevice {
                    Button("Forget") {
                        syncViewModel.manager.unpairDevice()
                        UserDefaults.standard.removeObject(forKey: "paired_device_id")
                    }
                    .buttonStyle(ShadowButtonStyle(color: .shadowError))
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.shadowSurface)
                .shadow(color: Color.shadowElevation2, radius: 8, x: 0, y: 2)
        )
    }
    
    // MARK: Recent Events
    private var recentEventsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "clock.arrow.circlepath")
                    .font(.title3)
                    .foregroundColor(.shadowSuccess)
                Text("Recent Activity")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.shadowTextPrimary)
                Spacer()
                Text("\(recentEvents.count)")
                    .font(.caption)
                    .foregroundColor(.shadowTextPrimary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color.shadowBackgroundSecondary)
                    )
            }
            
            VStack(spacing: 8) {
                ForEach(Array(recentEvents.prefix(5)), id: \.objectID) { event in
                    RecentStressEventRow(event: event)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.shadowSurface)
                .shadow(color: Color.shadowElevation2, radius: 8, x: 0, y: 2)
        )
    }
    
    @ViewBuilder
    private var statusIndicator: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(systemStatusColor)
                .frame(width: 8, height: 8)
            Text(syncViewModel.stateText)
                .font(.caption)
                .foregroundColor(.shadowTextPrimary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(Color.shadowBackgroundSecondary)
        )
    }
    
    private var systemStatusColor: Color {
        if syncViewModel.isActive {
            return .shadowWarning
        } else if syncViewModel.stateText == "Up To Date" {
            return .shadowSuccess
        } else {
            return .shadowTextTertiary
        }
    }
    
    private func statusRow(_ title: String, _ value: String, systemColor: Color? = nil) -> some View {
        HStack {
            Text(title)
                .font(.caption)
                .foregroundColor(.shadowTextSecondary)
            Spacer()
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(systemColor ?? .shadowTextPrimary)
        }
    }
    
    // MARK: Pairing Section
    private var pairingSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: syncViewModel.manager.isPairedToDevice ? "checkmark.shield.fill" : "lock.shield")
                    .foregroundColor(syncViewModel.manager.isPairedToDevice ? .shadowSuccess : .shadowWarning)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Device Status")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundColor(.shadowTextPrimary)
                    
                    if let deviceName = syncViewModel.manager.pairedDeviceName {
                        Text(deviceName)
                            .font(.caption)
                            .foregroundColor(.shadowTextSecondary)
                    } else {
                        Text("Not paired")
                            .font(.caption)
                            .foregroundColor(.shadowTextSecondary)
                    }
                }
                
                Spacer()
                
                // Status badge
                Text(syncViewModel.manager.isPairedToDevice ? "Paired" : "Unpaired")
                    .font(.caption2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        Capsule()
                            .fill(syncViewModel.manager.isPairedToDevice ? Color.shadowSuccess : Color.shadowWarning)
                    )
            }
            
            // Pairing button
            if !syncViewModel.manager.isPairedToDevice {
                Button(action: {
                    showQRScanner = true
                }) {
                    HStack {
                        Image(systemName: "qrcode.viewfinder")
                        Text("Pair Device")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .background(Color.shadowPrimary)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)
            }
        }
    }
}

struct ShadowDebugLogView: View {
    @ObservedObject var syncViewModel: SyncDashboardViewModel
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(syncViewModel.log.enumerated()), id: \.offset) { _, entry in
                        Text(entry)
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
                    
                    if syncViewModel.log.isEmpty {
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
                    Button("Close") { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button("Clear") {
                        // Optional: implement clearing logs if desired
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
            case .normal: return EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16)
            case .small:  return EdgeInsets(top: 4, leading: 8, bottom: 4, trailing: 8)
            }
        }
        var font: Font {
            switch self {
            case .normal: return .caption
            case .small:  return .caption2
            }
        }
    }
    
    init(color: Color, size: ButtonSize = .normal) {
        self.color = color
        self.size  = size
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
