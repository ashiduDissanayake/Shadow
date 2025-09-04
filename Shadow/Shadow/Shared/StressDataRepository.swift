import Foundation
import CoreData
import CoreBluetooth

/// Unified repository for ShadowDevice + StressEvent using the existing AppModel.
/// Mirrors the simplicity of ProfileRepository and replaces ShadowCoreDataManager.
final class StressDataRepository {
    static let shared = StressDataRepository()
    
    private let container: NSPersistentContainer
    private var context: NSManagedObjectContext { container.viewContext }
    
    private init() {
        container = NSPersistentContainer(name: "AppModel")
        container.loadPersistentStores { _, error in
            if let error = error {
                fatalError("Core Data failed to load: \(error)")
            }
        }
        context.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy
    }
    
    // MARK: - Device
    
    @discardableResult
    func getOrCreateDevice(peripheralID: UUID,
                           name: String? = nil,
                           userProfile: UserProfile? = nil) -> ShadowDevice {
        let identifier = peripheralID.uuidString
        let req: NSFetchRequest<ShadowDevice> = ShadowDevice.fetchRequest()
        req.predicate = NSPredicate(format: "deviceIdentifier == %@", identifier)
        req.fetchLimit = 1
        if let existing = try? context.fetch(req).first {
            if let name, existing.deviceName != name {
                existing.deviceName = name
            }
            return existing
        }
        let device = ShadowDevice(context: context)
        device.deviceIdentifier = identifier
        device.deviceName = name ?? "Shadow Device"
        device.lastKnownSequence = 0
        device.lastKnownState = 0
        device.lastConnectedDate = Date()
        device.userProfile = userProfile
        save()
        return device
    }
    
    func updateDeviceSequence(peripheralID: UUID,
                              sequence: UInt8,
                              state: UInt8) {
        let device = getOrCreateDevice(peripheralID: peripheralID)
        device.lastKnownSequence = Int16(sequence)
        device.lastKnownState = Int16(state)
        device.lastConnectedDate = Date()
        save()
    }
    
    func lastKnownSequence(peripheralID: UUID) -> UInt8 {
        let identifier = peripheralID.uuidString
        let req: NSFetchRequest<ShadowDevice> = ShadowDevice.fetchRequest()
        req.predicate = NSPredicate(format: "deviceIdentifier == %@", identifier)
        req.fetchLimit = 1
        if let existing = try? context.fetch(req).first {
            return UInt8(existing.lastKnownSequence)
        }
        return 0
    }
    
    // MARK: - Stress Events
    
    func lastEventSequence() -> UInt8 {
        let req: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        req.sortDescriptors = [NSSortDescriptor(key: "sequenceNumber", ascending: false)]
        req.fetchLimit = 1
        if let last = try? context.fetch(req).first {
            return UInt8(last.sequenceNumber)
        }
        return 0
    }
    
    func addStressEvent(peripheralID: UUID,
                        sequence: UInt8,
                        stressState: UInt8,
                        eventTimestamp: Date,
                        confidenceScore: Float? = nil,
                        batteryMv: UInt16? = nil,
                        sensorQuality: UInt8? = nil,
                        durationPrev: UInt32? = nil) {
        let device = getOrCreateDevice(peripheralID: peripheralID)
        let ev = StressEvent(context: context)
        ev.device = device
        ev.sequenceNumber = Int16(sequence)
        ev.stressState = Int16(stressState)
        ev.timestamp = eventTimestamp
        ev.receivedTimestamp = Date()
        if let confidenceScore { ev.confidenceScore = confidenceScore }
        if let batteryMv { ev.batteryVoltage = Int16(batteryMv) }
        if let sensorQuality { ev.sensorQuality = Int16(sensorQuality) }
        if let durationPrev { ev.durationPrevState = Int32(durationPrev) }
        save()
    }
    
    func recentEvents(limit: Int = 50) -> [StressEvent] {
        let req: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        req.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
        req.fetchLimit = limit
        return (try? context.fetch(req)) ?? []
    }
    
    // MARK: - Utility
    
    func deleteAllEvents(for peripheralID: UUID) {
        let identifier = peripheralID.uuidString
        let req: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        req.predicate = NSPredicate(format: "device.deviceIdentifier == %@", identifier)
        if let events = try? context.fetch(req) {
            events.forEach(context.delete)
            save()
        }
    }
    
    private func save() {
        guard context.hasChanges else { return }
        do { try context.save() }
        catch { print("StressDataRepository save error: \(error)") }
    }
}
