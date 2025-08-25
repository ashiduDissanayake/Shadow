//
//  ProfileRepository.swift
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-23.
//


import CoreData
import Foundation

class ProfileRepository {
    static let shared = ProfileRepository()
    let container: NSPersistentContainer

    private init() {
        container = NSPersistentContainer(name: "AppModel")
        container.loadPersistentStores { _, error in
            if let error = error {
                fatalError("Core Data load failed: \(error)")
            }
        }
    }

    // Save profile
    func saveProfile(email: String, name: String, workRole: String) {
        let context = container.viewContext
        let request: NSFetchRequest<UserProfile> = UserProfile.fetchRequest()
        request.predicate = NSPredicate(format: "email == %@", email)
        if let existing = try? context.fetch(request).first {
            existing.name = name
            existing.workRole = workRole
        } else {
            let profile = UserProfile(context: context)
            profile.email = email
            profile.name = name
            profile.workRole = workRole
        }
        try? context.save()
    }

    // Load profile by email
    func loadProfile(email: String) -> UserProfile? {
        let context = container.viewContext
        let request: NSFetchRequest<UserProfile> = UserProfile.fetchRequest()
        request.predicate = NSPredicate(format: "email == %@", email)
        return try? context.fetch(request).first
    }

    // Remove profile
    func deleteProfile(email: String) {
        let context = container.viewContext
        let request: NSFetchRequest<UserProfile> = UserProfile.fetchRequest()
        request.predicate = NSPredicate(format: "email == %@", email)
        if let profile = try? context.fetch(request).first {
            context.delete(profile)
            try? context.save()
        }
    }

    // Check if any profile exists
    func hasAnyProfile() -> Bool {
        let context = container.viewContext
        let request: NSFetchRequest<UserProfile> = UserProfile.fetchRequest()
        return (try? context.count(for: request)) ?? 0 > 0
    }
}
