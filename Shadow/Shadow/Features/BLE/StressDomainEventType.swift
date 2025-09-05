import Foundation

// MARK: - Domain Event Types

enum StressDomainEventType: String {
    case transition
    case dataLossReset
    case deviceReboot
}

/// Core transition domain event (rich but all optional except IDs & sequence).
struct StressTransitionDomainEvent {
    let deviceID: UUID
    let sequence7: UInt8
    let fullSequence: UInt16?          // future use (if >7 bits)
    let resetCounter: Int32
    let epoch: Int16?                   // future boot epoch
    let stressState: UInt8              // 0=CALM, 1=STRESS
    let receivedAt: Date
    let deviceTimestampMs: UInt64?      // future if firmware supplies
    let confidence: Float?
    let batteryMv: UInt16?
    let sensorQuality: UInt8?
    let durationPrevMs: UInt32?
    let notes: String?
    let type: StressDomainEventType
    let isSynthetic: Bool
}

struct ResetMarkerDomainEvent {
    let deviceID: UUID
    let resetCounter: Int32
    let epoch: Int16?
    let reason: String
    let receivedAt: Date
}
