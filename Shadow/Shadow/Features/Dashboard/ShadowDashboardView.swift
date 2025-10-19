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
    @State private var showingCoreDataDebug = false
    @State private var recentEvents: [StressEvent] = []
    @State private var showingPairingAlert = false
    @State private var pairingError: String?
    @State private var isPairing = false
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerSection
                shadowStatusSection
                
                if !recentEvents.isEmpty {
                    recentEventsSection
                }
                
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
            ShadowDebugLogView(syncViewModel: syncViewModel)
        }
        .sheet(isPresented: $showingCoreDataDebug) {
            CoreDataDebugView()
        }
        .alert("Device Pairing", isPresented: $showingPairingAlert) {
            Button("OK") { }
        } message: {
            if let error = pairingError {
                Text("Pairing failed: \(error)")
            } else {
                Text("Device paired successfully! ✅")
            }
        }
        .onAppear {
            syncViewModel.start()
            recentEvents = syncViewModel.getRecentEvents()
        }
        .onReceive(syncViewModel.$eventsReceived) { _ in
            recentEvents = syncViewModel.getRecentEvents()
        }
    }
    
    // MARK: Header
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
            RoundedRectangle(cornerRadius: 12).fill(.ultraThinMaterial)
        )
    }
    
    // MARK: Status Section
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
            
            VStack(spacing: 12) {
                statusRow("System Status", syncViewModel.stateText, systemColor: systemStatusColor)
                statusRow("Last Sync", syncViewModel.lastSync, systemColor: .secondary)
                statusRow("Sequence Info", syncViewModel.sequenceStatus, systemColor: .secondary)
                statusRow("Events Received", "\(syncViewModel.eventsReceived)", systemColor: .secondary)
                
                Divider()
                    .background(Color.white.opacity(0.3))
                
                // Pairing Section
                pairingSection
            }
            
            HStack(spacing: 12) {
                if syncViewModel.isActive {
                    Button("Stop Sync") { syncViewModel.stop() }
                        .buttonStyle(ShadowButtonStyle(color: .orange))
                } else {
                    Button("Start Sync") { syncViewModel.start() }
                        .buttonStyle(ShadowButtonStyle(color: .blue))
                }
                
                Button("Refresh Data") {
                    recentEvents = syncViewModel.getRecentEvents()
                }
                .buttonStyle(ShadowButtonStyle(color: .green))
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12).fill(.ultraThinMaterial)
        )
    }
    
    // MARK: Recent Events
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
                Text("\(recentEvents.count) events")
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
                ForEach(Array(recentEvents.prefix(3)), id: \.objectID) { event in
                    RecentStressEventRow(event: event)
                }
                
                if recentEvents.isEmpty {
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
            RoundedRectangle(cornerRadius: 12).fill(.ultraThinMaterial)
        )
    }
    
        // MARK: Debug Section
    private var debugSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "ladybug")
                    .font(.title2)
                    .foregroundColor(.orange)
                Text("Debug Tools")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                Spacer()
            }
            
            VStack(spacing: 8) {
                Button(action: { showingDebugLog = true }) {
                    HStack {
                        Image(systemName: "doc.text")
                        Text("BLE Debug Log")
                        Spacer()
                        Image(systemName: "chevron.right")
                    }
                    .foregroundColor(.white)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.blue.opacity(0.2))
                    )
                }
                
                Button(action: { showingCoreDataDebug = true }) {
                    HStack {
                        Image(systemName: "cylinder.split.1x2")
                        Text("Core Data Manager")
                        Spacer()
                        Image(systemName: "chevron.right")
                    }
                    .foregroundColor(.white)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.purple.opacity(0.2))
                    )
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12).fill(.ultraThinMaterial)
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
                .foregroundColor(.white.opacity(0.85))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 6).fill(.ultraThinMaterial)
        )
    }
    
    private var systemStatusColor: Color {
        if syncViewModel.isActive {
            return .orange
        } else if syncViewModel.stateText == "Up To Date" {
            return .green
        } else {
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
    
    // MARK: Pairing Section
    private var pairingSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "lock.shield")
                    .foregroundColor(syncViewModel.manager.isPaired ? .green : .orange)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Device Pairing")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                    
                    if let deviceInfo = syncViewModel.manager.deviceInfo {
                        Text("\(deviceInfo.deviceName) - \(deviceInfo.firmwareVersion)")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.7))
                    } else {
                        Text("Not paired")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
                
                Spacer()
                
                // Pairing state indicator
                HStack(spacing: 4) {
                    Text(syncViewModel.manager.pairingState.emoji)
                    Text(syncViewModel.manager.pairingState.description)
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                }
            }
            
            // Pairing button
            if !syncViewModel.manager.isPaired {
                Button(action: {
                    isPairing = true
                    Task {
                        do {
                            try await syncViewModel.manager.performPairing()
                            isPairing = false
                            pairingError = nil
                            showingPairingAlert = true
                        } catch {
                            isPairing = false
                            pairingError = error.localizedDescription
                            showingPairingAlert = true
                        }
                    }
                }) {
                    HStack {
                        if isPairing {
                            ProgressView()
                                .scaleEffect(0.8)
                                .tint(.white)
                        } else {
                            Image(systemName: "key.fill")
                        }
                        Text(isPairing ? "Pairing..." : "Pair Device")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .background(isPairing ? Color.blue.opacity(0.6) : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)
                .disabled(isPairing || syncViewModel.manager.pairingState == .pending)
            } else {
                // Show paired status
                HStack {
                    Image(systemName: "checkmark.shield.fill")
                        .foregroundColor(.green)
                    Text("Device Paired")
                        .fontWeight(.semibold)
                        .foregroundColor(.green)
                    Spacer()
                }
                .padding(.vertical, 8)
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