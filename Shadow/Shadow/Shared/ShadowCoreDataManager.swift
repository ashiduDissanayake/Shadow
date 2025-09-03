import Foundation
import CoreData

class ShadowCoreDataManager: ObservableObject, CoreDataManager {
    private let persistentContainer: NSPersistentContainer
    
    init(persistentContainer: NSPersistentContainer) {
        self.persistentContainer = persistentContainer
    }
    
    var context: NSManagedObjectContext {
        return persistentContainer.viewContext
    }
    
    func saveContext() {
        if context.hasChanges {
            do {
                try context.save()
                print("ShadowCoreData: Context saved successfully")
            } catch {
                print("ShadowCoreData: Failed to save context: \(error)")
            }
        }
    }
    
    // MARK: - ShadowDevice Operations
    
    func getOrCreateShadowDevice(identifier: String, for userProfile: UserProfile) -> ShadowDevice? {
        // First try to find existing device
        let request: NSFetchRequest<ShadowDevice> = ShadowDevice.fetchRequest()
        request.predicate = NSPredicate(format: "deviceIdentifier == %@ AND userProfile == %@", identifier, userProfile)
        request.fetchLimit = 1
        
        do {
            let existingDevices = try context.fetch(request)
            if let existingDevice = existingDevices.first {
                return existingDevice
            }
        } catch {
            print("ShadowCoreData: Error fetching device: \(error)")
            return nil
        }
        
        // Create new device
        let newDevice = ShadowDevice(context: context)
        newDevice.deviceIdentifier = identifier
        newDevice.deviceName = "Shadow Device"
        newDevice.lastKnownSequence = 0
        newDevice.lastKnownState = 0  // CALM
        newDevice.userProfile = userProfile
        
        saveContext()
        print("ShadowCoreData: Created new Shadow device: \(identifier)")
        
        return newDevice
    }
    
    func updateDeviceSequence(_ device: ShadowDevice, sequence: UInt8, state: UInt8) {
        device.lastKnownSequence = Int16(sequence)
        device.lastKnownState = Int16(state)
        device.lastConnectedDate = Date()
        
        saveContext()
        print("ShadowCoreData: Updated device sequence: \(sequence), state: \(state)")
    }
    
    func getLastKnownSequence(for deviceIdentifier: String, userProfile: UserProfile) -> UInt8 {
        let request: NSFetchRequest<ShadowDevice> = ShadowDevice.fetchRequest()
        request.predicate = NSPredicate(format: "deviceIdentifier == %@ AND userProfile == %@", deviceIdentifier, userProfile)
        request.fetchLimit = 1
        
        do {
            let devices = try context.fetch(request)
            if let device = devices.first {
                return UInt8(device.lastKnownSequence)
            }
        } catch {
            print("ShadowCoreData: Error fetching last sequence: \(error)")
        }
        
        return 0  // Default for new devices
    }
    
    // MARK: - StressEvent Operations
    
    func createStressEvent(
        device: ShadowDevice,
        sequenceNumber: UInt8,
        stressState: UInt8,
        timestamp: Date,
        confidenceScore: Float? = nil,
        batteryVoltage: UInt16? = nil,
        sensorQuality: UInt8? = nil,
        durationPrevState: UInt32? = nil
    ) -> StressEvent? {
        
        let event = StressEvent(context: context)
        event.device = device
        event.sequenceNumber = Int16(sequenceNumber)
        event.stressState = Int16(stressState)
        event.timestamp = timestamp
        event.receivedTimestamp = Date()
        
        if let confidence = confidenceScore {
            event.confidenceScore = confidence
        }
        
        if let battery = batteryVoltage {
            event.batteryVoltage = Int16(battery)
        }
        
        if let quality = sensorQuality {
            event.sensorQuality = Int16(quality)
        }
        
        if let duration = durationPrevState {
            event.durationPrevState = Int32(duration)
        }
        
        saveContext()
        print("ShadowCoreData: Created stress event: seq=\(sequenceNumber), state=\(stressState)")
        
        return event
    }
    
    func getStressEvents(for device: ShadowDevice, limit: Int = 100) -> [StressEvent] {
        let request: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        request.predicate = NSPredicate(format: "device == %@", device)
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
        request.fetchLimit = limit
        
        do {
            return try context.fetch(request)
        } catch {
            print("ShadowCoreData: Error fetching stress events: \(error)")
            return []
        }
    }
    
    func getStressEventsCount(for device: ShadowDevice) -> Int {
        let request: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        request.predicate = NSPredicate(format: "device == %@", device)
        
        do {
            return try context.count(for: request)
        } catch {
            print("ShadowCoreData: Error counting stress events: \(error)")
            return 0
        }
    }
    
    func getRecentStressEvents(for userProfile: UserProfile, days: Int = 7) -> [StressEvent] {
        let calendar = Calendar.current
        let endDate = Date()
        guard let startDate = calendar.date(byAdding: .day, value: -days, to: endDate) else {
            return []
        }
        
        let request: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        request.predicate = NSPredicate(format: "device.userProfile == %@ AND timestamp >= %@ AND timestamp <= %@", 
                                       userProfile, startDate as NSDate, endDate as NSDate)
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
        request.fetchLimit = 1000  // Reasonable limit
        
        do {
            return try context.fetch(request)
        } catch {
            print("ShadowCoreData: Error fetching recent stress events: \(error)")
            return []
        }
    }
    
    // MARK: - Statistics
    
    func getStressStatistics(for userProfile: UserProfile, days: Int = 7) -> (totalEvents: Int, stressEvents: Int, calmEvents: Int) {
        let events = getRecentStressEvents(for: userProfile, days: days)
        
        let stressEvents = events.filter { $0.stressState == 1 }.count
        let calmEvents = events.filter { $0.stressState == 0 }.count
        
        return (totalEvents: events.count, stressEvents: stressEvents, calmEvents: calmEvents)
    }
    
    // MARK: - Cleanup
    
    func deleteOldStressEvents(olderThanDays days: Int = 30) {
        let calendar = Calendar.current
        guard let cutoffDate = calendar.date(byAdding: .day, value: -days, to: Date()) else {
            return
        }
        
        let request: NSFetchRequest<StressEvent> = StressEvent.fetchRequest()
        request.predicate = NSPredicate(format: "timestamp < %@", cutoffDate as NSDate)
        
        do {
            let oldEvents = try context.fetch(request)
            for event in oldEvents {
                context.delete(event)
            }
            
            if !oldEvents.isEmpty {
                saveContext()
                print("ShadowCoreData: Deleted \(oldEvents.count) old stress events")
            }
        } catch {
            print("ShadowCoreData: Error deleting old events: \(error)")
        }
    }
}
