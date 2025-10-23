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
    
    // Use shared persistence controller to avoid multiple NSManagedObjectModel instances
    let persistentContainer = PersistenceController.shared.container
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistentContainer.viewContext)
        }
    }
}

// App Delegate to set up notification delegate and calendar monitoring
class AppDelegate: NSObject, NSApplicationDelegate {
    let notificationDelegate = NotificationDelegate()
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        print("🚀 App launched, setting up notification delegate...")
        UNUserNotificationCenter.current().delegate = notificationDelegate
        print("✅ Notification delegate set")
        
        // Start calendar event monitoring
        Task { @MainActor in
            CalendarEventMonitor.shared.startMonitoring()
            print("✅ Calendar event monitoring started")
        }
    }
}
