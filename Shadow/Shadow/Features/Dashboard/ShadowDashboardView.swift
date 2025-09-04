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
                Image(systemName: "terminal")
                    .font(.title3)
                    .foregroundColor(.purple)
                Text("System Debug")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                Spacer()
                Button("View Full Log") { showingDebugLog = true }
                    .buttonStyle(ShadowButtonStyle(color: .purple, size: .small))
            }
            
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(syncViewModel.log.suffix(3).enumerated()), id: \.offset) { _, line in
                    Text(line)
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.85))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(.black.opacity(0.3))
                        )
                }
                
                if syncViewModel.log.isEmpty {
                    Text("No debug messages")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.6))
                        .italic()
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