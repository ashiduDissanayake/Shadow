import Foundation
import CoreData

/// Unified repository for ShadowDevice + StressEvent persistence.
final class StressDataRepository {
    static let shared = StressDataRepository()
    
    // Dynamic device UUID - creates one if none exists, supports multiple devices
    lazy var defaultDeviceUUID: UUID = {
        return CoreDataReset.getOrCreateDefaultDeviceUUID()
    }()
    
    private let container: NSPersistentContainer
    private var context: NSManagedObjectContext { container.viewContext }
    
    private init() {
        container = NSPersistentContainer(name: "AppModel")
        container.loadPersistentStores { _, error in
            if let error { fatalError("Core Data load error: \(error)") }
        }
        context.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy
    }
    
    // MARK: - Device Helpers
    
    @discardableResult
    func getOrCreateDevice(deviceUUID: UUID,
                           name: String? = nil) -> ShadowDevice {
        let idStr = deviceUUID.uuidString
        let req: NSFetchRequest<ShadowDevice> = ShadowDevice.fetchRequest()
        req.predicate = NSPredicate(format: "deviceIdentifier == %@", idStr)
        req.fetchLimit = 1
        if let existing = try? context.fetch(req).first {
            if let name, existing.deviceName != name {
                existing.deviceName = name
            }
            return existing
        }
        let d = ShadowDevice(context: context)
        d.deviceIdentifier = idStr
        d.deviceName = name ?? "Shadow Device"
        d.lastKnownSequence = 0
        d.lastKnownState = 0
        // Safe: Only set if attributes exist (runtime KVC guard)
        setIfKeyExists(object: d, key: "resetCounter", value: 0)
        setIfKeyExists(object: d, key: "epoch", value: -1)
        d.lastConnectedDate = Date()
        save()
        return d
    }
    
    func updateDeviceState(deviceUUID: UUID,
                           sequence: UInt8,
                           state: UInt8,
                           resetCounter: Int32,
                           epoch: Int16?) {
        let d = getOrCreateDevice(deviceUUID: deviceUUID)
        d.lastKnownSequence = Int16(sequence)
        d.lastKnownState = Int16(state)
        setIfKeyExists(object: d, key: "resetCounter", value: resetCounter)
        if let epoch { setIfKeyExists(object: d, key: "epoch", value: epoch) }
        d.lastConnectedDate = Date()
        save()
    }
    
    func loadLastKnownSequence(deviceUUID: UUID) -> UInt8 {
        let idStr = deviceUUID.uuidString
        let req: NSFetchRequest<ShadowDevice> = ShadowDevice.fetchRequest()
        req.predicate = NSPredicate(format: "deviceIdentifier == %@", idStr)
        req.fetchLimit = 1
        if let d = try? context.fetch(req).first {
            return UInt8(d.lastKnownSequence & 0x7F)
        }
        return 0
    }
    
    func currentResetCounter(deviceUUID: UUID) -> Int32 {
        // If model has resetCounter attribute, fetch; else fallback to UserDefaults
        let idStr = deviceUUID.uuidString
        let req: NSFetchRequest<ShadowDevice> = ShadowDevice.fetchRequest()
        req.predicate = NSPredicate(format: "deviceIdentifier == %@", idStr)
        req.fetchLimit = 1
        if let d = try? context.fetch(req).first,
           hasKey(d, key: "resetCounter"),
           let val = d.value(forKey: "resetCounter") as? Int32 {
            return val
        }
        // Fallback
        let key = "Shadow_ResetCounter_" + idStr
        return Int32(UserDefaults.standard.integer(forKey: key))
    }
    
    @discardableResult
    func incrementResetCounter(deviceUUID: UUID) -> Int32 {
        let d = getOrCreateDevice(deviceUUID: deviceUUID)
        if hasKey(d, key: "resetCounter"),
           let old = d.value(forKey: "resetCounter") as? Int32 {
            d.setValue(old + 1, forKey: "resetCounter")
            save()
            return old + 1
        } else {
            // Fallback to UserDefaults path
            let key = "Shadow_ResetCounter_" + deviceUUID.uuidString
            let old = Int32(UserDefaults.standard.integer(forKey: key))
            let newVal = old + 1
            UserDefaults.standard.set(Int(newVal), forKey: key)
            return newVal
        }
    }
    
    // MARK: - Event Persistence
    
    func persistTransition(_ evt: StressTransitionDomainEvent) {
        let device = getOrCreateDevice(deviceUUID: evt.deviceID)
        if exists(sequence: evt.sequence7,
                  resetCounter: evt.resetCounter,
                  device: device) {
            return
        }
        let e = StressEvent(context: context)
        e.device = device
        e.sequenceNumber = Int16(evt.sequence7)
        e.stressState = Int16(evt.stressState)
        e.timestamp = evt.receivedAt
        e.receivedTimestamp = evt.receivedAt
        setIfKeyExists(object: e, key: "resetCounter", value: evt.resetCounter)
        setIfKeyExists(object: e, key: "epoch", value: evt.epoch ?? -1)
        setIfKeyExists(object: e, key: "eventType", value: evt.type.rawValue)
        setIfKeyExists(object: e, key: "isSynthetic", value: evt.isSynthetic)
        setIfKeyExists(object: e, key: "notes", value: evt.notes)
        if let c = evt.confidence { e.confidenceScore = c }
//        if let b = evt.batteryMv { e.batteryVoltage = Int16(b) }
//        if let q = evt.sensorQuality { e.sensorQuality = Int16(q) }
//        if let dPrev = evt.durationPrevMs { e.durationPrevState = Int32(dPrev) }
        save()
    }
    
    func persistResetMarker(_ marker: ResetMarkerDomainEvent,
                            lastKnownState: Int16) {
        let device = getOrCreateDevice(deviceUUID: marker.deviceID)
        let e = StressEvent(context: context)
        e.device = device
        e.sequenceNumber = 0
        e.stressState = lastKnownState
        e.timestamp = marker.receivedAt
        e.receivedTimestamp = marker.receivedAt
        setIfKeyExists(object: e, key: "resetCounter", value: marker.resetCounter)
        setIfKeyExists(object: e, key: "epoch", value: marker.epoch ?? -1)
        setIfKeyExists(object: e, key: "eventType", value: StressDomainEventType.dataLossReset.rawValue)
        setIfKeyExists(object: e, key: "isSynthetic", value: true)
        setIfKeyExists(object: e, key: "notes", value: marker.reason)
        save()
    }
    
    private func exists(sequence: UInt8,
                        resetCounter: Int32,
                        device: ShadowDevice) -> Bool {
        // If no resetCounter attribute, uniqueness becomes unreliable; skip check
        guard hasKey(device, key: "resetCounter") else { return false }
        let req: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        if hasProperty(entity: StressEvent.entity(), name: "resetCounter") {
            req.predicate = NSPredicate(format: "device == %@ AND sequenceNumber == %d AND resetCounter == %d",
                                        device, Int(sequence), resetCounter)
        } else {
            req.predicate = NSPredicate(format: "device == %@ AND sequenceNumber == %d",
                                        device, Int(sequence))
        }
        req.fetchLimit = 1
        return ((try? context.fetch(req))?.isEmpty == false)
    }
    
    // MARK: - Queries
    
    func recentEvents(deviceUUID: UUID, limit: Int = 50) -> [StressEvent] {
        let idStr = deviceUUID.uuidString
        let req: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        req.predicate = NSPredicate(format: "device.deviceIdentifier == %@", idStr)
        
        if hasProperty(entity: StressEvent.entity(), name: "resetCounter") {
            req.sortDescriptors = [
                NSSortDescriptor(key: "resetCounter", ascending: true),
                NSSortDescriptor(key: "sequenceNumber", ascending: true)
            ]
        } else {
            req.sortDescriptors = [
                NSSortDescriptor(key: "sequenceNumber", ascending: true)
            ]
        }
        req.fetchLimit = limit
        return (try? context.fetch(req)) ?? []
    }
    
    func recentEvents(limit: Int = 50) -> [StressEvent] {
        recentEvents(deviceUUID: defaultDeviceUUID, limit: limit)
    }
    
    func deleteAll(deviceUUID: UUID) {
        let idStr = deviceUUID.uuidString
        let req: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        req.predicate = NSPredicate(format: "device.deviceIdentifier == %@", idStr)
        if let arr = try? context.fetch(req) {
            arr.forEach(context.delete)
            save()
        }
    }
    
    // MARK: - Helpers for dynamic attribute presence
    
    private func hasKey(_ obj: NSManagedObject, key: String) -> Bool {
        obj.entity.attributesByName[key] != nil
    }
    
    private func setIfKeyExists(object: NSManagedObject, key: String, value: Any?) {
        guard hasKey(object, key: key), let value else { return }
        object.setValue(value, forKey: key)
    }
    
    private func hasProperty(entity: NSEntityDescription, name: String) -> Bool {
        entity.attributesByName[name] != nil
    }
    
    // MARK: - Save
    private func save() {
        guard context.hasChanges else { return }
        do { try context.save() }
        catch { print("Core Data save error: \(error)") }
    }
}
