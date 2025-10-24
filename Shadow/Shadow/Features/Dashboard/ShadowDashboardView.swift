import SwiftUI
import Combine
import CoreData
import Charts

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
    @State private var showGraph = false // Toggle for graph visibility
    @State private var selectedTimeRange: TimeRange = .threeHours
    @State private var currentTime = Date()
    
    enum TimeRange: String, CaseIterable {
        case oneHour = "1h"
        case threeHours = "3h"
        case sixHours = "6h"
        case twentyFourHours = "24h"
        
        var hours: Int {
            switch self {
            case .oneHour: return 1
            case .threeHours: return 3
            case .sixHours: return 6
            case .twentyFourHours: return 24
            }
        }
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerSection
                
                // Live Status Card
                liveStatusCard
                
                // Statistics Cards Row
                statisticsRow
                
                shadowStatusSection
                
                // Graph Section with Toggle
                graphToggleSection
                
                if showGraph && !graphEvents.isEmpty {
                    stressGraphSection
                        .transition(.opacity.combined(with: .scale))
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
            refreshData()
            print("📊 [Dashboard] Initial load: \(recentEvents.count) recent, \(graphEvents.count) graph events")
            
            // Start timer for live updates
            Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
                currentTime = Date()
            }
        }
        .onReceive(syncViewModel.$eventUpdateTrigger) { uuid in
            refreshData()
            print("📊 [Dashboard] UI update triggered (uuid=\(uuid)): \(recentEvents.count) recent, \(graphEvents.count) graph events")
        }
    }
    
    private func refreshData() {
        recentEvents = syncViewModel.getRecentEvents()
        graphEvents = syncViewModel.getEventsInLastHours(selectedTimeRange.hours)
    }
    
    // MARK: - Live Status Card
    private var liveStatusCard: some View {
        HStack(spacing: 16) {
            // Animated Status Indicator
            ZStack {
                Circle()
                    .fill(currentStressColor.opacity(0.2))
                    .frame(width: 60, height: 60)
                
                Circle()
                    .fill(currentStressColor)
                    .frame(width: 40, height: 40)
                    .overlay(
                        Image(systemName: currentStressState == 1 ? "bolt.fill" : "leaf.fill")
                            .foregroundColor(.white)
                            .font(.title3)
                    )
            }
            .shadow(color: currentStressColor.opacity(0.3), radius: 8)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(currentStressState == 1 ? "Experiencing Stress" : "Feeling Calm")
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundColor(.shadowTextPrimary)
                
                if let lastEvent = recentEvents.first {
                    Text(timeAgo(from: lastEvent.timestamp ?? Date()))
                        .font(.caption)
                        .foregroundColor(.shadowTextSecondary)
                }
            }
            
            Spacer()
            
            // Pulse animation
            if currentStressState == 1 {
                Circle()
                    .fill(Color.red)
                    .frame(width: 12, height: 12)
                    .overlay(
                        Circle()
                            .stroke(Color.red.opacity(0.4), lineWidth: 4)
                            .scaleEffect(pulseAnimation ? 1.8 : 1.0)
                            .opacity(pulseAnimation ? 0 : 1)
                    )
                    .onAppear {
                        withAnimation(.easeInOut(duration: 1.5).repeatForever(autoreverses: false)) {
                            pulseAnimation = true
                        }
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
    
    @State private var pulseAnimation = false
    
    private var currentStressState: Int {
        Int(recentEvents.first?.stressState ?? 0)
    }
    
    private var currentStressColor: Color {
        currentStressState == 1 ? .red : .green
    }
    
    private func timeAgo(from date: Date) -> String {
        let seconds = Int(Date().timeIntervalSince(date))
        if seconds < 60 {
            return "\(seconds) second\(seconds == 1 ? "" : "s") ago"
        } else if seconds < 3600 {
            let minutes = seconds / 60
            return "\(minutes) minute\(minutes == 1 ? "" : "s") ago"
        } else {
            let hours = seconds / 3600
            return "\(hours) hour\(hours == 1 ? "" : "s") ago"
        }
    }
    
    // MARK: - Statistics Row
    private var statisticsRow: some View {
        HStack(spacing: 12) {
            StatCard(
                icon: "chart.bar.fill",
                title: "Today",
                value: "\(todayStressCount)",
                subtitle: "episodes",
                color: .shadowPrimary
            )
            
            StatCard(
                icon: "clock.fill",
                title: "Avg Duration",
                value: averageStressDuration,
                subtitle: "minutes",
                color: .shadowWarning
            )
            
            StatCard(
                icon: "heart.fill",
                title: "Recovery",
                value: "\(recoveryCount)",
                subtitle: "times",
                color: .shadowSuccess
            )
        }
    }
    
    private var todayStressCount: Int {
        let today = Calendar.current.startOfDay(for: Date())
        return recentEvents.filter { event in
            guard let timestamp = event.timestamp else { return false }
            return timestamp >= today && Int(event.stressState) == 1
        }.count
    }
    
    private var averageStressDuration: String {
        // Calculate real average duration of stress episodes
        guard recentEvents.count > 1 else { return "0" }
        
        var totalDuration: TimeInterval = 0
        var episodeCount = 0
        var stressStartTime: Date?
        
        // Sort events by timestamp (oldest first)
        let sortedEvents = recentEvents.sorted { 
            ($0.timestamp ?? Date.distantPast) < ($1.timestamp ?? Date.distantPast) 
        }
        
        for event in sortedEvents {
            if Int(event.stressState) == 1 {
                // Stress started
                if stressStartTime == nil {
                    stressStartTime = event.timestamp
                }
            } else if Int(event.stressState) == 0, let startTime = stressStartTime {
                // Stress ended - calculate duration
                if let endTime = event.timestamp {
                    let duration = endTime.timeIntervalSince(startTime)
                    totalDuration += duration
                    episodeCount += 1
                }
                stressStartTime = nil
            }
        }
        
        // If still stressed, count duration until now
        if let startTime = stressStartTime {
            let duration = Date().timeIntervalSince(startTime)
            totalDuration += duration
            episodeCount += 1
        }
        
        guard episodeCount > 0 else { return "0" }
        let averageMinutes = Int(totalDuration / Double(episodeCount) / 60)
        return "\(averageMinutes)"
    }
    
    private var recoveryCount: Int {
        guard recentEvents.count > 1 else { return 0 }
        var count = 0
        for i in 0..<(recentEvents.count - 1) {
            if Int(recentEvents[i].stressState) == 0 && Int(recentEvents[i+1].stressState) == 1 {
                count += 1
            }
        }
        return count
    }
    
    // MARK: - Graph Toggle Section
    private var graphToggleSection: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: "chart.xyaxis.line")
                    .font(.title3)
                    .foregroundColor(.shadowAccent)
                Text("Stress Timeline")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.shadowTextPrimary)
                
                Spacer()
                
                // Toggle Button
                Button(action: {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                        showGraph.toggle()
                    }
                }) {
                    HStack(spacing: 4) {
                        Text(showGraph ? "Hide" : "Show")
                            .font(.caption)
                            .fontWeight(.semibold)
                        Image(systemName: showGraph ? "chevron.up" : "chevron.down")
                            .font(.caption)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        Capsule()
                            .fill(Color.shadowPrimary)
                    )
                }
                .buttonStyle(.plain)
            }
            
            if showGraph {
                // Time Range Selector
                HStack(spacing: 8) {
                    ForEach(TimeRange.allCases, id: \.self) { range in
                        Button(action: {
                            selectedTimeRange = range
                            graphEvents = syncViewModel.getEventsInLastHours(range.hours)
                        }) {
                            Text(range.rawValue)
                                .font(.caption)
                                .fontWeight(selectedTimeRange == range ? .bold : .medium)
                                .foregroundColor(selectedTimeRange == range ? .white : .shadowTextSecondary)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(
                                    Capsule()
                                        .fill(selectedTimeRange == range ? Color.shadowAccent : Color.shadowBackgroundSecondary)
                                )
                        }
                        .buttonStyle(.plain)
                    }
                    
                    Spacer()
                    
                    Text("\(graphEvents.count) events")
                        .font(.caption2)
                        .foregroundColor(.shadowTextTertiary)
                }
                .padding(.top, 8)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.shadowSurface)
                .shadow(color: Color.shadowElevation1, radius: 6, x: 0, y: 2)
        )
    }
    
    // MARK: - Graph Section
    private var stressGraphSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            EnhancedStressGraph(events: graphEvents, currentTime: currentTime)
                .frame(height: 220)
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

// MARK: - Stat Card Component
struct StatCard: View {
    let icon: String
    let title: String
    let value: String
    let subtitle: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
            
            Text(value)
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(.shadowTextPrimary)
            
            VStack(spacing: 2) {
                Text(title)
                    .font(.caption2)
                    .foregroundColor(.shadowTextSecondary)
                Text(subtitle)
                    .font(.caption2)
                    .foregroundColor(.shadowTextTertiary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.shadowSurface)
                .shadow(color: Color.shadowElevation1, radius: 4, x: 0, y: 2)
        )
    }
}

// MARK: - Enhanced Stress Graph
struct EnhancedStressGraph: View {
    let events: [StressEvent]
    let currentTime: Date
    
    var body: some View {
        let segments = createStateSegments()
        
        Chart {
            // Draw each segment with purple line/area
            ForEach(segments) { segment in
                // Area fill - always purple gradient
                ForEach(segment.points) { point in
                    AreaMark(
                        x: .value("Time", point.date),
                        y: .value("State", point.state)
                    )
                }
                .foregroundStyle(
                    LinearGradient(
                        colors: [
                            Color.purple.opacity(0.3),
                            Color.purple.opacity(0.05)
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                
                // Line - always purple
                ForEach(segment.points) { point in
                    LineMark(
                        x: .value("Time", point.date),
                        y: .value("State", point.state)
                    )
                }
                .foregroundStyle(Color.purple)
                .lineStyle(StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
                
                // Point marks at transitions - green for calm (0), red for stressed (1)
                if let firstPoint = segment.points.first, segment.isTransition {
                    PointMark(
                        x: .value("Time", firstPoint.date),
                        y: .value("State", firstPoint.state)
                    )
                    .symbolSize(120)
                    .foregroundStyle(segment.state == 1 ? Color.red : Color.green)
                }
            }
        }
        .chartXAxis {
            AxisMarks(values: .stride(by: .hour)) { value in
                AxisGridLine()
                AxisTick()
                AxisValueLabel(format: .dateTime.hour().minute())
            }
        }
        .chartYAxis {
            AxisMarks(values: [0, 1]) { value in
                AxisGridLine()
                AxisTick()
                AxisValueLabel {
                    if let intValue = value.as(Int.self) {
                        Text(intValue == 1 ? "Stressed" : "Calm")
                            .font(.caption2)
                            .foregroundColor(.shadowTextSecondary)
                    }
                }
            }
        }
        .chartYScale(domain: -0.15...1.15)
    }
    
    // Create segments grouped by state (all consecutive stressed = 1 segment, all calm = 1 segment)
    private func createStateSegments() -> [StateSegment] {
        guard !events.isEmpty else { return [] }
        
        let sortedEvents = events.sorted { ($0.timestamp ?? Date.distantPast) < ($1.timestamp ?? Date.distantPast) }
        var segments: [StateSegment] = []
        var currentSegmentPoints: [ContinuousPoint] = []
        var currentState: Int?
        
        for (index, event) in sortedEvents.enumerated() {
            let date = event.timestamp ?? Date()
            let state = Int(event.stressState)
            
            // State changed - save previous segment and start new one
            if let prevState = currentState, prevState != state {
                if !currentSegmentPoints.isEmpty {
                    segments.append(StateSegment(
                        state: prevState,
                        points: currentSegmentPoints,
                        isTransition: segments.isEmpty || segments.last!.state != prevState
                    ))
                }
                currentSegmentPoints = []
            }
            
            currentState = state
            currentSegmentPoints.append(ContinuousPoint(date: date, state: state, isTransition: false))
            
            // Add continuation to next event or current time
            if index < sortedEvents.count - 1 {
                let nextDate = sortedEvents[index + 1].timestamp ?? Date()
                currentSegmentPoints.append(ContinuousPoint(
                    date: nextDate.addingTimeInterval(-1),
                    state: state,
                    isTransition: false
                ))
            } else {
                // Last event - extend to current time
                currentSegmentPoints.append(ContinuousPoint(
                    date: currentTime,
                    state: state,
                    isTransition: false
                ))
            }
        }
        
        // Add final segment
        if !currentSegmentPoints.isEmpty, let state = currentState {
            segments.append(StateSegment(
                state: state,
                points: currentSegmentPoints,
                isTransition: segments.isEmpty || segments.last!.state != state
            ))
        }
        
        return segments
    }
}

// Segment representing a continuous period of same state
struct StateSegment: Identifiable {
    let id = UUID()
    let state: Int
    let points: [ContinuousPoint]
    let isTransition: Bool
}

struct ContinuousPoint: Identifiable {
    let id = UUID()
    let date: Date
    let state: Int
    let isTransition: Bool
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
