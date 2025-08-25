//
//  AppNavBar.swift
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-24.
//

import SwiftUI

struct AppNavBar: View {
    let title: String
    let subtitle: String?
    let profile: UserProfile
    let onProfileTap: () -> Void
    let onLogout: (() -> Void)?
    let showProfileMenu: Bool
    let onCalendarTap: (() -> Void)? // <-- Added

    @State private var isHoveringProfile = false
    @State private var showBLEPopover = false

    @ObservedObject var bleManager: BLEManager

    var body: some View {
        HStack {
            // Logo and Title
            HStack(spacing: 16) {
                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                gradient: Gradient(colors: [
                                    Color.purple.opacity(0.8),
                                    Color.blue.opacity(0.7)
                                ]),
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 44, height: 44)
                        .shadow(color: Color.purple.opacity(0.3), radius: 8, x: 0, y: 4)
                    
                    Image(systemName: "figure.walk.motion")
                        .font(.system(size: 18, weight: .medium))
                        .foregroundColor(.white)
                }
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.system(size: 24, weight: .semibold, design: .rounded))
                        .foregroundColor(.white)
                    
                    if let subtitle {
                        Text(subtitle)
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
            }
            
            Spacer()

            // Calendar Button
            if let onCalendarTap = onCalendarTap {
                Button(action: { onCalendarTap() }) {
                    Image(systemName: "calendar")
                        .font(.system(size: 22, weight: .bold))
                        .foregroundColor(.white)
                        .padding(8)
                        .background(Color.white.opacity(0.12))
                        .cornerRadius(12)
                }
                .buttonStyle(PlainButtonStyle())
                .padding(.trailing, 10)
            }

            // BLE Status Icon & Popover
            Button(action: {
                showBLEPopover.toggle()
            }) {
                HStack(spacing: 6) {
                    Image(systemName: "dot.radiowaves.left.and.right")
                        .font(.system(size: 22, weight: .bold))
                        .foregroundColor(bleStatusColor)
                    Text(bleStatusText)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.white.opacity(0.8))
                }
                .padding(8)
                .background(Color.white.opacity(0.12))
                .cornerRadius(12)
            }
            .buttonStyle(PlainButtonStyle())
            .popover(isPresented: $showBLEPopover, arrowEdge: .top) {
                BLEPopoverView(bleManager: bleManager)
                    .frame(width: 340, height: 340)
            }
            .padding(.trailing, 10)

            // Profile Section
            if showProfileMenu {
                Button(action: onProfileTap) {
                    HStack(spacing: 12) {
                        ZStack {
                            Circle()
                                .fill(Color.white.opacity(0.15))
                                .frame(width: 36, height: 36)
                            
                            Text(getInitials(from: profile.name ?? "User"))
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundColor(.white)
                        }
                        
                        VStack(alignment: .leading, spacing: 1) {
                            Text(getFirstName(from: profile.name ?? "User"))
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundColor(.white)
                            
                            Text(profile.workRole ?? "Member")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundColor(.white.opacity(0.7))
                        }
                        
                        Image(systemName: "chevron.down")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(.white.opacity(0.6))
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Color.white.opacity(isHoveringProfile ? 0.15 : 0.1))
                            .overlay(
                                RoundedRectangle(cornerRadius: 16)
                                    .stroke(Color.white.opacity(0.2), lineWidth: 1)
                            )
                    )
                    .scaleEffect(isHoveringProfile ? 1.02 : 1.0)
                }
                .buttonStyle(PlainButtonStyle())
                .onHover { hovering in
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isHoveringProfile = hovering
                    }
                }
            }

            // Optional Logout Button (for Profile screen or others)
            if let onLogout {
                Button(action: onLogout) {
                    Image(systemName: "rectangle.portrait.and.arrow.right")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(.white.opacity(0.8))
                }
                .buttonStyle(PlainButtonStyle())
                .padding(.leading, 16)
            }
        }
        .padding(.horizontal, 32)
        .padding(.vertical, 20)
        .background(Color.clear)
    }

    // MARK: - BLE Status

    private var bleStatusColor: Color {
        if bleManager.connectionStatus.lowercased().contains("connected") {
            return .green
        } else if bleManager.connectionStatus.lowercased().contains("scanning") ||
                    bleManager.connectionStatus.lowercased().contains("connecting") {
            return .blue
        } else if bleManager.connectionStatus.lowercased().contains("not available") {
            return .orange
        } else {
            return .gray
        }
    }

    private var bleStatusText: String {
        if bleManager.connectionStatus.lowercased().contains("connected") {
            return "Connected"
        } else if bleManager.connectionStatus.lowercased().contains("scanning") {
            return "Scanning"
        } else if bleManager.connectionStatus.lowercased().contains("connecting") {
            return "Connecting"
        } else if bleManager.connectionStatus.lowercased().contains("not available") {
            return "Unavailable"
        } else {
            return "BLE"
        }
    }

    // MARK: - Helpers
    private func getInitials(from name: String) -> String {
        let components = name.components(separatedBy: .whitespaces)
        let initials = components.compactMap { $0.first?.uppercased() }.prefix(2).joined()
        return initials.isEmpty ? "U" : initials
    }
    private func getFirstName(from name: String) -> String {
        let components = name.components(separatedBy: .whitespaces)
        return components.first ?? "User"
    }
}

// MARK: - BLE Popover View

struct BLEPopoverView: View {
    @ObservedObject var bleManager: BLEManager

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Bluetooth Status")
                .font(.headline)
                .foregroundColor(.primary)
                .padding(.top, 6)
            Divider()

            HStack {
                Image(systemName: "dot.radiowaves.left.and.right")
                    .foregroundColor(bleManager.connectionStatus.lowercased().contains("connected") ? .green : .gray)
                Text(bleManager.connectionStatus)
                    .font(.subheadline)
            }
            .padding(.bottom, 10)

            if bleManager.isScanning {
                Button("Stop Scanning") {
                    bleManager.stopScanning()
                }
                .buttonStyle(.borderedProminent)
            } else {
                Button("Start Scanning") {
                    bleManager.startScanning()
                }
                .buttonStyle(.bordered)
            }

            List(bleManager.foundDevices, id: \.identifier) { device in
                HStack {
                    Text(device.name ?? "Unknown Device")
                        .font(.subheadline)
                    Spacer()
                    if bleManager.connectedPeripheral?.identifier == device.identifier {
                        Text("Connected").foregroundColor(.green).font(.caption)
                    } else {
                        Button("Connect") {
                            bleManager.connect(to: device)
                        }
                        .font(.caption)
                    }
                }
            }
            .frame(height: 150)
        }
        .padding()
        .frame(maxWidth: .infinity)
    }
}
