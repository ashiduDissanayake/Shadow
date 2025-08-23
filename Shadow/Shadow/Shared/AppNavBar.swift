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

    @State private var isHoveringProfile = false

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