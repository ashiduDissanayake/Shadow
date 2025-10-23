import Foundation
import Combine

@MainActor
final class SyncDashboardViewModel: ObservableObject {
    @Published var stateText: String = "Idle"
    @Published var lastSync: String = "-"
    @Published var sequenceStatus: String = "-"
    @Published var log: [String] = []
    @Published var eventsReceived: Int = 0
    @Published var eventUpdateTrigger: UUID = UUID()  // Force UI updates
    @Published var isActive: Bool = false
    @Published var currentStateLabel: String = "CALM"
    
    let manager: LightShadowBLEManager  // Made public for pairing access
    private var cancellables = Set<AnyCancellable>()
    
    private let df: DateFormatter = {
        let d = DateFormatter()
        d.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return d
    }()
    
    // NOTE: manager param is optional; default nil avoids calling a MainActor init
    // from a nonisolated default parameter context.
    init(manager: LightShadowBLEManager? = nil) {
        self.manager = manager ?? LightShadowBLEManager()
        setupBindings()
    }
    
    private func setupBindings() {
        manager.$status
            .map { status in
                switch status {
                case .idle: return "Idle"
                case .scanning: return "Monitoring"
                case .connecting: return "Connecting"
                case .requestingMissed: return "Syncing Missed Events"
                case .upToDate: return "Up To Date"
                case .error: return "Error"
                }
            }
            .assign(to: &$stateText)
        
        manager.$isScanning
            .assign(to: &$isActive)
        
        manager.$lastKnownSequence
            .map { seq -> String in
                guard seq > 0 else { return "-" }
                return self.df.string(from: Date())
            }
            .assign(to: &$lastSync)
        
        Publishers.CombineLatest(manager.$lastKnownSequence, manager.$currentStableState)
            .map { seq, state -> String in
                let label = (state == 1) ? "STRESS" : "CALM"
                return "Sequence: \(seq) | State: \(label)"
            }
            .assign(to: &$sequenceStatus)
        
        manager.$currentStableState
            .map { $0 == 1 ? "STRESS" : "CALM" }
            .assign(to: &$currentStateLabel)
        
        manager.$logLines
            .assign(to: &$log)
        
        manager.$lastKnownSequence
            .map { Int($0) }
            .assign(to: &$eventsReceived)

        // Listen for persisted events and evaluate notifications
        NotificationCenter.default.publisher(for: Notification.Name("Shadow.NewStressEvent"))
            .receive(on: DispatchQueue.main)
            .sink { note in
                print("🔔 [ViewModel] Received Shadow.NewStressEvent notification!")
                guard let userInfo = note.userInfo,
                      let seq = userInfo["sequence"] as? Int,
                      let state = userInfo["state"] as? Int,
                      let idStr = userInfo["deviceID"] as? String,
                      let deviceUUID = UUID(uuidString: idStr) else { 
                    print("❌ [ViewModel] Failed to parse notification userInfo")
                    return
                }

                print("🔔 [ViewModel] Parsed: seq=\(seq), state=\(state), device=\(deviceUUID)")
                
                // Fetch the persisted StressEvent from CoreData by sequence
                let events = StressDataRepository.shared.recentEvents(deviceUUID: deviceUUID, limit: 200)
                print("🔔 [ViewModel] Fetched \(events.count) recent events from CoreData")
                
                if let evt = events.first(where: { Int($0.sequenceNumber) == seq }) {
                    print("🔔 [ViewModel] Found matching event: seq=\(evt.sequenceNumber), state=\(evt.stressState)")
                    
                    // Force UI update by changing trigger UUID
                    Task { @MainActor in
                        self.eventUpdateTrigger = UUID()
                        print("🔄 [ViewModel] Triggered UI update with new UUID")
                        
                        // Evaluate via NotificationDecisionEngine (only once per event)
                        print("➡️ [ViewModel] Passing event to NotificationDecisionEngine: seq=\(evt.sequenceNumber), state=\(evt.stressState)")
                        NotificationDecisionEngine.shared.evaluate(event: evt)
                    }
                } else {
                    print("❌ [ViewModel] Could not find event with seq=\(seq) in CoreData")
                }
            }
            .store(in: &cancellables)
    }
    
    // MARK: - UI Actions
    func start() {
        manager.start()
        log.append("[UI] Start requested")
    }
    
    func stop() {
        manager.stop()
        log.append("[UI] Stop requested")
    }
    
    // MARK: - Data Access
    func getRecentEvents(limit: Int = 50) -> [StressEvent] {
        let events = StressDataRepository.shared.recentEvents(limit: limit)
        print("🔍 [ViewModel] getRecentEvents() fetched \(events.count) events from CoreData")
        return events
    }
    
    /// Get events from last N hours (for graph display)
    func getEventsInLastHours(_ hours: Int = 3) -> [StressEvent] {
        let events = StressDataRepository.shared.eventsInLastHours(hours)
        print("🔍 [ViewModel] getEventsInLastHours(\(hours)) fetched \(events.count) events")
        return events
    }
    
    var recentEvents: [StressEvent] {
        StressDataRepository.shared.recentEvents()
    }
}
