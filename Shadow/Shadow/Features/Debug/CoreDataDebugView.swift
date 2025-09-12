//
//  CoreDataDebugView.swift
//  Shadow
//
//  Created by AI Assistant on 2025-09-12.
//

import SwiftUI
import CoreData

struct CoreDataDebugView: View {
    @Environment(\.managedObjectContext) private var context
    @State private var showingResetConfirmation = false
    @State private var showingDataDeleteConfirmation = false
    @State private var currentDeviceUUID: String = ""
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    
                    // Current Device Section
                    deviceSection
                    
                    // Data Statistics Section
                    statisticsSection
                    
                    // Reset Options Section
                    resetSection
                    
                }
                .padding()
            }
            .navigationTitle("Core Data Debug")
            .onAppear {
                loadDeviceUUID()
            }
        }
    }
    
    private var deviceSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Device Configuration")
                .font(.headline)
                .foregroundColor(.white)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Current Device UUID:")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.7))
                
                Text(currentDeviceUUID)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.green)
                    .padding(8)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(.black.opacity(0.3))
                    )
                
                Button(action: generateNewUUID) {
                    Text("Generate New UUID")
                        .font(.caption)
                        .foregroundColor(.blue)
                        .padding(.vertical, 4)
                        .padding(.horizontal, 8)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(.blue.opacity(0.2))
                        )
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
    }
    
    private var statisticsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Data Statistics")
                .font(.headline)
                .foregroundColor(.white)
            
            VStack(spacing: 8) {
                StatRow(title: "Stress Events", count: getEntityCount("StressEvent"))
                StatRow(title: "Shadow Devices", count: getEntityCount("ShadowDevice"))
                StatRow(title: "Calendar Events", count: getEntityCount("Event"))
                StatRow(title: "User Profiles", count: getEntityCount("UserProfile"))
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
    }
    
    private var resetSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Reset Options")
                .font(.headline)
                .foregroundColor(.white)
            
            VStack(spacing: 12) {
                Button(action: { showingDataDeleteConfirmation = true }) {
                    HStack {
                        Image(systemName: "trash")
                        Text("Delete All Data")
                    }
                    .foregroundColor(.orange)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.orange.opacity(0.2))
                    )
                }
                
                Button(action: { showingResetConfirmation = true }) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle")
                        Text("Complete Core Data Reset")
                    }
                    .foregroundColor(.red)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.red.opacity(0.2))
                    )
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
        .alert("Delete All Data?", isPresented: $showingDataDeleteConfirmation) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                CoreDataReset.deleteAllData()
                loadDeviceUUID()
            }
        } message: {
            Text("This will delete all stress events, devices, and user data but keep the database structure.")
        }
        .alert("Complete Reset?", isPresented: $showingResetConfirmation) {
            Button("Cancel", role: .cancel) { }
            Button("Reset", role: .destructive) {
                CoreDataReset.deleteAllCoreDataStores()
                // App will need restart after this
            }
        } message: {
            Text("This will completely delete all Core Data files. The app will need to be restarted.")
        }
    }
    
    private func loadDeviceUUID() {
        currentDeviceUUID = StressDataRepository.shared.defaultDeviceUUID.uuidString
    }
    
    private func generateNewUUID() {
        let newUUID = UUID()
        UserDefaults.standard.set(newUUID.uuidString, forKey: "ShadowDefaultDeviceUUID")
        UserDefaults.standard.synchronize()
        currentDeviceUUID = newUUID.uuidString
    }
    
    private func getEntityCount(_ entityName: String) -> Int {
        let request = NSFetchRequest<NSFetchRequestResult>(entityName: entityName)
        do {
            return try context.count(for: request)
        } catch {
            return 0
        }
    }
}

struct StatRow: View {
    let title: String
    let count: Int
    
    var body: some View {
        HStack {
            Text(title)
                .foregroundColor(.white.opacity(0.8))
            Spacer()
            Text("\(count)")
                .foregroundColor(.white)
                .fontWeight(.medium)
        }
    }
}

#Preview {
    CoreDataDebugView()
        .preferredColorScheme(.dark)
}
