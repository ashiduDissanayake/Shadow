//
//  PersistenceController.swift
//  Shadow
//
//  Shared CoreData persistence controller
//  Ensures single NSPersistentContainer instance across entire app
//

import Foundation
import CoreData

/// Singleton CoreData persistence controller
/// Prevents multiple NSManagedObjectModel instances and "Multiple NSEntityDescriptions" errors
class PersistenceController {
    static let shared = PersistenceController()
    
    let container: NSPersistentContainer
    
    private init() {
        container = NSPersistentContainer(name: "AppModel")
        
        container.loadPersistentStores { description, error in
            if let error = error {
                fatalError("Failed to load Core Data stack: \(error)")
            }
            print("✅ [PersistenceController] Core Data loaded: \(description)")
        }
        
        // Configure for better concurrency
        container.viewContext.automaticallyMergesChangesFromParent = true
        container.viewContext.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy
    }
    
    /// Convenience accessor for view context
    var viewContext: NSManagedObjectContext {
        container.viewContext
    }
    
    /// Create a new background context for async operations
    func newBackgroundContext() -> NSManagedObjectContext {
        container.newBackgroundContext()
    }
}
