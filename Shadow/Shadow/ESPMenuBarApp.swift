//
//  ESPMenuBarApp.swift
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-18.
//


import SwiftUI

@main
struct ESPMenuBarApp: App {
    @StateObject private var link = ESPLink()

    var body: some Scene {
        MenuBarExtra("ESP Link", systemImage: "antenna.radiowaves.left.and.right") {
            Text("Connected: \(link.connectedCount)")
                .font(.headline)
                .padding(.horizontal)

            Divider()

            Button("LED ON") {
                link.send(json: ["type":"command","action":"led","value":1])
            }
            Button("LED OFF") {
                link.send(json: ["type":"command","action":"led","value":0])
            }

            Button("Ping") {
                link.send(json: ["type":"command","action":"ping"])
            }

            Divider()
            Button("Quit") { NSApp.terminate(nil) }
        }
    }
}
