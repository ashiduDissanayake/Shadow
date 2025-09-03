//
//  YourAppNameApp.swift
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-23.
//


import SwiftUI
import CoreData

@main
struct Shadow: App {
    // Core Data container
    let persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "AppModel")
        container.loadPersistentStores { _, error in
            if let error = error {
                print("Core Data failed to load: \(error.localizedDescription)")
            } else {
                print("Core Data loaded successfully")
            }
        }
        return container
    }()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistentContainer.viewContext)
                .environmentObject(ShadowCoreDataManager(persistentContainer: persistentContainer))
        }
    }
}
