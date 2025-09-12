//
//  CoreDataReset.swift
//  Shadow
//
//  Created by AI Assistant on 2025-09-12.
//

import Foundation
import CoreData

/// Utility class to reset Core Data and handle fresh start scenarios
class CoreDataReset {
    
    /// Completely delete all Core Data stores and reset to fresh state
    static func deleteAllCoreDataStores() {
        let container = NSPersistentContainer(name: "AppModel")
        
        // Get all store descriptions
        for storeDescription in container.persistentStoreDescriptions {
            guard let storeURL = storeDescription.url else { continue }
            
            do {
                // Remove the actual store file
                if FileManager.default.fileExists(atPath: storeURL.path) {
                    try FileManager.default.removeItem(at: storeURL)
                    print("✅ Deleted Core Data store: \(storeURL.path)")
                }
                
                // Remove related files (-wal, -shm)
                let walURL = storeURL.appendingPathExtension("wal")
                if FileManager.default.fileExists(atPath: walURL.path) {
                    try FileManager.default.removeItem(at: walURL)
                    print("✅ Deleted WAL file: \(walURL.path)")
                }
                
                let shmURL = storeURL.appendingPathExtension("shm")
                if FileManager.default.fileExists(atPath: shmURL.path) {
                    try FileManager.default.removeItem(at: shmURL)
                    print("✅ Deleted SHM file: \(shmURL.path)")
                }
                
            } catch {
                print("❌ Error deleting Core Data files: \(error)")
            }
        }
        
        // Clear UserDefaults for device tracking
        clearDeviceUserDefaults()
        
        print("🔄 Core Data reset complete - app will start fresh")
    }
    
    /// Delete all data but keep the store structure
    static func deleteAllData() {
        let container = NSPersistentContainer(name: "AppModel")
        container.loadPersistentStores { _, error in
            if let error = error {
                print("❌ Failed to load stores for deletion: \(error)")
                return
            }
            
            let context = container.viewContext
            
            // Delete all entities in correct order (relationships first)
            deleteAllEntities(ofType: "StressEvent", in: context)
            deleteAllEntities(ofType: "Event", in: context)
            deleteAllEntities(ofType: "ShadowDevice", in: context)
            deleteAllEntities(ofType: "UserProfile", in: context)
            
            // Save changes
            do {
                try context.save()
                print("✅ All Core Data entries deleted successfully")
            } catch {
                print("❌ Error saving after deletion: \(error)")
            }
        }
        
        // Clear UserDefaults for device tracking
        clearDeviceUserDefaults()
    }
    
    private static func deleteAllEntities(ofType entityName: String, in context: NSManagedObjectContext) {
        let fetchRequest = NSFetchRequest<NSFetchRequestResult>(entityName: entityName)
        let deleteRequest = NSBatchDeleteRequest(fetchRequest: fetchRequest)
        
        do {
            try context.execute(deleteRequest)
            print("✅ Deleted all \(entityName) entities")
        } catch {
            print("❌ Error deleting \(entityName) entities: \(error)")
        }
    }
    
    private static func clearDeviceUserDefaults() {
        let defaults = UserDefaults.standard
        let keys = defaults.dictionaryRepresentation().keys
        
        for key in keys {
            if key.hasPrefix("Shadow_") {
                defaults.removeObject(forKey: key)
                print("✅ Cleared UserDefaults key: \(key)")
            }
        }
        
        defaults.synchronize()
    }
    
    /// Get default device UUID (creates one if none exists)
    static func getOrCreateDefaultDeviceUUID() -> UUID {
        let key = "ShadowDefaultDeviceUUID"
        
        if let uuidString = UserDefaults.standard.string(forKey: key),
           let uuid = UUID(uuidString: uuidString) {
            print("📱 Using existing device UUID: \(uuid)")
            return uuid
        }
        
        // Create new default UUID
        let newUUID = UUID()
        UserDefaults.standard.set(newUUID.uuidString, forKey: key)
        UserDefaults.standard.synchronize()
        
        print("📱 Created new default device UUID: \(newUUID)")
        return newUUID
    }
}
