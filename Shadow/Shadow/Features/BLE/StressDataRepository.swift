import Foundation
import CoreData

/// Unified repository for ShadowDevice + StressEvent persistence.
final class StressDataRepository {
    static let shared = StressDataRepository()
    
    // Dynamic device UUID - creates one if none exists, supports multiple devices
    lazy var defaultDeviceUUID: UUID = {
        return CoreDataReset.getOrCreateDefaultDeviceUUID()
    }()
    
    // Use shared persistence controller to avoid multiple NSManagedObjectModel instances
    private let container = PersistenceController.shared.container
    private var context: NSManagedObjectContext { container.viewContext }
    
    private init() {
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
        // Check if exact same event already exists (sequence + state + resetCounter)
        if exists(sequence: evt.sequence7,
                  state: Int16(evt.stressState),
                  resetCounter: evt.resetCounter,
                  device: device) {
            print("⚠️ [Repository] Duplicate event (seq=\(evt.sequence7), state=\(evt.stressState), reset=\(evt.resetCounter)), skipping")
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
        
        print("✅ [Repository] Saved event: seq=\(evt.sequence7), state=\(evt.stressState), reset=\(evt.resetCounter)")
        
        // Notify observers about new event (lightweight payload)
        NotificationCenter.default.post(name: Notification.Name("Shadow.NewStressEvent"), object: nil, userInfo: [
            "deviceID": evt.deviceID.uuidString,
            "sequence": Int(evt.sequence7),
            "state": Int(evt.stressState),
            "timestamp": evt.receivedAt
        ])
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
                        state: Int16,
                        resetCounter: Int32,
                        device: ShadowDevice) -> Bool {
        // Check if exact same event exists (sequence + state + resetCounter)
        guard hasKey(device, key: "resetCounter") else { return false }
        let req: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        if hasProperty(entity: StressEvent.entity(), name: "resetCounter") {
            req.predicate = NSPredicate(format: "device == %@ AND sequenceNumber == %d AND stressState == %d AND resetCounter == %d",
                                        device, Int(sequence), Int(state), resetCounter)
        } else {
            req.predicate = NSPredicate(format: "device == %@ AND sequenceNumber == %d AND stressState == %d",
                                        device, Int(sequence), Int(state))
        }
        req.fetchLimit = 1
        return ((try? context.fetch(req))?.isEmpty == false)
    }
    
    // MARK: - Queries
    
    func recentEvents(deviceUUID: UUID, limit: Int = 50) -> [StressEvent] {
        let idStr = deviceUUID.uuidString
        let req: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        req.predicate = NSPredicate(format: "device.deviceIdentifier == %@", idStr)
        
        // Sort by timestamp (most recent first) instead of sequence
        req.sortDescriptors = [NSSortDescriptor(keyPath: \StressEvent.timestamp, ascending: false)]
        
        req.fetchLimit = limit
        let events = (try? context.fetch(req)) ?? []
        
        print("🔍 [Repository] recentEvents(deviceUUID) fetched \(events.count) events")
        return events
    }
    
    func recentEvents(limit: Int = 50) -> [StressEvent] {
        // Refresh context to get latest data
        context.refreshAllObjects()
        
        let events = recentEvents(deviceUUID: defaultDeviceUUID, limit: limit)
        print("📊 [Repository] recentEvents() returned \(events.count) events")
        if events.count > 0 {
            print("   First event: seq=\(events.first!.sequenceNumber), state=\(events.first!.stressState)")
            print("   Last event: seq=\(events.last!.sequenceNumber), state=\(events.last!.stressState)")
            
            // DEBUG: Print ALL events
            print("   ALL EVENTS (sorted by timestamp, newest first):")
            for (index, event) in events.enumerated() {
                let timeStr = event.timestamp?.description ?? "nil"
                let resetStr = (event.value(forKey: "resetCounter") as? Int32).map { String($0) } ?? "?"
                print("     [\(index)] seq=\(event.sequenceNumber), state=\(event.stressState), reset=\(resetStr), time=\(timeStr)")
            }
        }
        return events
    }
    
    /// Get events from last N hours (default 3 hours)
    func eventsInLastHours(_ hours: Int = 3) -> [StressEvent] {
        // Refresh context to get latest data
        context.refreshAllObjects()
        
        let cutoff = Date().addingTimeInterval(-Double(hours) * 3600)
        let idStr = defaultDeviceUUID.uuidString
        let req: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        req.predicate = NSPredicate(format: "device.deviceIdentifier == %@ AND timestamp >= %@", idStr, cutoff as NSDate)
        req.sortDescriptors = [NSSortDescriptor(keyPath: \StressEvent.timestamp, ascending: false)]
        let events = (try? context.fetch(req)) ?? []
        
        print("📊 [Repository] eventsInLastHours(\(hours)) returned \(events.count) events since \(cutoff)")
        
        // DEBUG: Print each event
        for (index, event) in events.enumerated() {
            let timeStr = event.timestamp?.description ?? "nil"
            print("   [\(index)] seq=\(event.sequenceNumber), state=\(event.stressState), time=\(timeStr)")
        }
        
        return events
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
