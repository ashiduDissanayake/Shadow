//
//  YourAppNameApp.swift
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-23.
//


import SwiftUI
import CoreData
import UserNotifications

// Notification delegate to show notifications when app is in foreground
class NotificationDelegate: NSObject, UNUserNotificationCenterDelegate {
    func userNotificationCenter(_ center: UNUserNotificationCenter,
                               willPresent notification: UNNotification,
                               withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        print("🔔 [Delegate] Notification will present: \(notification.request.content.title)")
        // Show banner even when app is in foreground
        completionHandler([.banner, .sound, .badge])
    }
    
    func userNotificationCenter(_ center: UNUserNotificationCenter,
                               didReceive response: UNNotificationResponse,
                               withCompletionHandler completionHandler: @escaping () -> Void) {
        print("🔔 [Delegate] Notification tapped: \(response.notification.request.content.title)")
        completionHandler()
    }
}

@main
struct Shadow: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
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
        }
    }
}

// App Delegate to set up notification delegate
class AppDelegate: NSObject, NSApplicationDelegate {
    let notificationDelegate = NotificationDelegate()
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        print("🚀 App launched, setting up notification delegate...")
        UNUserNotificationCenter.current().delegate = notificationDelegate
        print("✅ Notification delegate set")
    }
}
