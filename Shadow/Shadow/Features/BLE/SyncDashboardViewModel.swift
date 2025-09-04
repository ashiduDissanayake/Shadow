import Foundation
import Combine

/// Simple ViewModel that observes LightShadowBLEManager and formats data for SwiftUI.
/// You can bind your UI directly to this instead of juggling many @Published props.
@MainActor
final class SyncDashboardViewModel: ObservableObject {
    @Published var stateText: String = "Idle"
    @Published var lastSync: String = "-"
    @Published var sequenceStatus: String = "-"
    @Published var log: [String] = []
    @Published var eventsReceived: Int = 0
    @Published var isActive: Bool = false
    
    private let manager: LightShadowBLEManager
    private var cancellables = Set<AnyCancellable>()
    private let df: DateFormatter = {
        let d = DateFormatter()
        d.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return d
    }()
    
    init(manager: LightShadowBLEManager = LightShadowBLEManager()) {
        self.manager = manager
        setupBindings()
    }
    
    private func setupBindings() {
        manager.$currentStatus
            .map { status in
                switch status {
                case .idle: return "Idle"
                case .scanning: return "Scanning for Changes"
                case .connecting: return "Connecting"
                case .handshaking: return "Handshaking"
                case .replaying: return "Syncing Data"
                case .upToDate: return "Up to Date"
                case .disconnecting: return "Disconnecting"
                case .error: return "Error"
                }
            }
            .assign(to: &$stateText)
        
        manager.$currentStatus
            .map { status in
                switch status {
                case .scanning, .connecting, .handshaking, .replaying, .disconnecting:
                    return true
                default:
                    return false
                }
            }
            .assign(to: &$isActive)
        
        manager.$lastSyncDate
            .map { [weak self] date in
                guard let date else { return "-" }
                return self?.df.string(from: date) ?? "-"
            }
            .assign(to: &$lastSync)
        
        Publishers.CombineLatest(manager.$advertisedSequence,
                                 manager.$lastKnownSequence)
            .map { adv, stored in "Adv: \(adv) | Local: \(stored)" }
            .assign(to: &$sequenceStatus)
        
        manager.$debugLog
            .assign(to: &$log)
        
        manager.$eventsReceivedThisSync
            .assign(to: &$eventsReceived)
    }
    
    func start() { 
        manager.start() 
    }
    
    func stop() { 
        manager.stop() 
    }
    
    func getRecentEvents() -> [StressEvent] {
        return StressDataRepository.shared.recentEvents()
    }
    
    var recentEvents: [StressEvent] {
        return StressDataRepository.shared.recentEvents()
    }
}
