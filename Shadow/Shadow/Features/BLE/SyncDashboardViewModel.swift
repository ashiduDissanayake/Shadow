import Foundation
import Combine

@MainActor
final class SyncDashboardViewModel: ObservableObject {
    @Published var stateText: String = "Idle"
    @Published var lastSync: String = "-"
    @Published var sequenceStatus: String = "-"
    @Published var log: [String] = []
    @Published var eventsReceived: Int = 0
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
        StressDataRepository.shared.recentEvents(limit: limit)
    }
    
    var recentEvents: [StressEvent] {
        StressDataRepository.shared.recentEvents()
    }
}
