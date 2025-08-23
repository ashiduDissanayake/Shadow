import SwiftUI

struct DashboardView: View {
    let profile: UserProfile
    let onLogout: () -> Void
    let onDeleteAccount: () -> Void
    let onShowProfile: () -> Void

    @State private var showingProfile = false
    @State private var selectedTab = 0
    @State private var searchText = ""
    @State private var showingSearch = false

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background gradient and pattern
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color(red: 0.05, green: 0.08, blue: 0.15),
                        Color(red: 0.1, green: 0.15, blue: 0.25)
                    ]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
                RoundedRectangle(cornerRadius: 0)
                    .fill(
                        RadialGradient(
                            gradient: Gradient(colors: [
                                Color.white.opacity(0.015),
                                Color.clear
                            ]),
                            center: .center,
                            startRadius: 0,
                            endRadius: geometry.size.width * 0.8
                        )
                    )
                    .ignoresSafeArea()

                VStack(spacing: 0) {
                    // AppNavBar replaces the custom nav code!
                    AppNavBar(
                        title: "Shadow",
                        subtitle: "Health Dashboard",
                        profile: profile,
                        onProfileTap: { onShowProfile() },
                        onLogout: { onLogout() },
                        showProfileMenu: true
                    )

                    if showingSearch {
                        searchBar
                    }

                    // Main Content Area
                    ScrollView(showsIndicators: false) {
                        VStack(spacing: 32) {
                            welcomeSection
                            quickActionsGrid
                            mainCardsSection
                            Spacer(minLength: 40)
                        }
                    }
                }
            }
        }
        .frame(minWidth: 1200, minHeight: 800)
        .sheet(isPresented: $showingProfile) {
            ProfileView(
                profile: profile,
                onLogout: {
                    showingProfile = false
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        onLogout()
                    }
                },
                onDeleteAccount: {
                    showingProfile = false
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        onDeleteAccount()
                    }
                },
                onBack: {
                    showingProfile = false
                },
                onProfileUpdated: { _ in }
            )
        }
    }

    // MARK: - Search Bar
    private var searchBar: some View {
        HStack(spacing: 16) {
            HStack(spacing: 12) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.white.opacity(0.6))

                TextField("Search health records, goals, insights...", text: $searchText)
                    .textFieldStyle(PlainTextFieldStyle())
                    .font(.system(size: 16, weight: .regular))
                    .foregroundColor(.white)

                if !searchText.isEmpty {
                    Button(action: { searchText = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundColor(.white.opacity(0.6))
                    }
                    .buttonStyle(PlainButtonStyle())
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
            .background(Color.white.opacity(0.15))
            .cornerRadius(16)
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.3), lineWidth: 1)
            )

            Button("Cancel") {
                withAnimation(.easeInOut(duration: 0.3)) {
                    showingSearch = false
                    searchText = ""
                }
            }
            .font(.system(size: 16, weight: .medium))
            .foregroundColor(.white.opacity(0.8))
        }
        .padding(.horizontal, 32)
        .padding(.bottom, 20)
        .transition(.opacity.combined(with: .move(edge: .top)))
    }

    // MARK: - Welcome Section
    private var welcomeSection: some View {
        VStack(spacing: 24) {
            HStack {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Welcome back, \(getFirstName(from: profile.name ?? "User"))!")
                        .font(.system(size: 32, weight: .light, design: .rounded))
                        .foregroundColor(.white)
                    Text("Ready to continue your health journey?")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.white.opacity(0.8))
                }
                Spacer()
            }
            .padding(.horizontal, 32)
            .padding(.top, 20)
        }
    }

    // MARK: - Quick Actions Grid
    private var quickActionsGrid: some View {
        LazyVGrid(columns: [
            GridItem(.flexible()),
            GridItem(.flexible()),
            GridItem(.flexible())
        ], spacing: 20) {
            QuickActionCard(
                icon: "plus.circle.fill",
                title: "Add Record",
                subtitle: "Log new health data",
                color: .green
            )
            QuickActionCard(
                icon: "chart.line.uptrend.xyaxis",
                title: "View Trends",
                subtitle: "Analyze your progress",
                color: .blue
            )
            QuickActionCard(
                icon: "target",
                title: "Set Goals",
                subtitle: "Define new objectives",
                color: .purple
            )
        }
        .padding(.horizontal, 32)
    }

    // MARK: - Main Cards Section
    private var mainCardsSection: some View {
        VStack(spacing: 20) {
            DashboardCard(
                icon: "heart.fill",
                title: "Health Overview",
                subtitle: "Your wellness at a glance",
                content: {
                    VStack(spacing: 16) {
                        HStack {
                            HealthMetricPill(label: "Ready to start", value: "", color: .green)
                            Spacer()
                            HealthMetricPill(label: "All systems go", value: "", color: .blue)
                        }
                        Text("Begin tracking your health metrics to see personalized insights here.")
                            .font(.system(size: 14, weight: .medium))
                            .foregroundColor(Color(red: 0.5, green: 0.5, blue: 0.6))
                            .multilineTextAlignment(.center)
                            .padding(.top, 8)
                    }
                }
            )
            DashboardCard(
                icon: "clock.fill",
                title: "Recent Activity",
                subtitle: "Your latest health actions",
                content: {
                    VStack(spacing: 12) {
                        ActivityItem(
                            icon: "person.badge.plus",
                            title: "Account Created",
                            time: "Just now",
                            color: .purple
                        )
                        Text("Start logging activities to see your health journey unfold here.")
                            .font(.system(size: 14, weight: .medium))
                            .foregroundColor(Color(red: 0.5, green: 0.5, blue: 0.6))
                            .multilineTextAlignment(.center)
                            .padding(.top, 16)
                    }
                }
            )
            DashboardCard(
                icon: "flag.fill",
                title: "Goals & Progress",
                subtitle: "Track your health objectives",
                content: {
                    VStack(spacing: 16) {
                        Text("Set your first health goal to start tracking progress and celebrating achievements.")
                            .font(.system(size: 14, weight: .medium))
                            .foregroundColor(Color(red: 0.5, green: 0.5, blue: 0.6))
                            .multilineTextAlignment(.center)
                        Button(action: {}) {
                            Text("Create Your First Goal")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundColor(.white)
                                .padding(.horizontal, 20)
                                .padding(.vertical, 10)
                                .background(
                                    LinearGradient(
                                        colors: [Color.purple, Color.blue],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .cornerRadius(20)
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                }
            )
        }
        .padding(.horizontal, 32)
    }

    // MARK: - Helper Functions
    private func getFirstName(from name: String) -> String {
        let components = name.components(separatedBy: .whitespaces)
        return components.first ?? "User"
    }
}

// MARK: - Supporting Views (as before)
struct QuickActionCard: View {
    let icon: String
    let title: String
    let subtitle: String
    let color: Color
    @State private var isHovering = false
    
    var body: some View {
        Button(action: {}) {
            VStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(color.opacity(0.15))
                        .frame(width: 50, height: 50)
                    
                    Image(systemName: icon)
                        .font(.system(size: 22, weight: .semibold))
                        .foregroundColor(color)
                }
                
                VStack(spacing: 4) {
                    Text(title)
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))
                    
                    Text(subtitle)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(Color(red: 0.5, green: 0.5, blue: 0.6))
                        .multilineTextAlignment(.center)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 24)
            .background(Color.white.opacity(0.95))
            .cornerRadius(20)
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .stroke(Color(red: 0.94, green: 0.95, blue: 0.96), lineWidth: 1)
            )
            .scaleEffect(isHovering ? 1.05 : 1.0)
            .shadow(
                color: color.opacity(0.2),
                radius: isHovering ? 15 : 8,
                x: 0,
                y: isHovering ? 8 : 4
            )
        }
        .buttonStyle(PlainButtonStyle())
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.3)) {
                isHovering = hovering
            }
        }
    }
}

struct DashboardCard<Content: View>: View {
    let icon: String
    let title: String
    let subtitle: String
    let content: () -> Content
    
    var body: some View {
        VStack(spacing: 0) {
            // Card Header
            HStack {
                HStack(spacing: 12) {
                    Image(systemName: icon)
                        .font(.system(size: 20, weight: .semibold))
                        .foregroundColor(.purple)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text(title)
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))
                        
                        Text(subtitle)
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(Color(red: 0.5, green: 0.5, blue: 0.6))
                    }
                }
                
                Spacer()
                
                Button(action: {}) {
                    Image(systemName: "ellipsis")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(Color(red: 0.6, green: 0.6, blue: 0.7))
                        .frame(width: 32, height: 32)
                        .background(Color(red: 0.96, green: 0.97, blue: 0.98))
                        .cornerRadius(8)
                }
                .buttonStyle(PlainButtonStyle())
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
            
            Divider()
                .background(Color(red: 0.94, green: 0.95, blue: 0.96))
            
            // Card Content
            VStack {
                content()
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
        }
        .background(Color.white.opacity(0.95))
        .cornerRadius(24)
        .shadow(color: Color.black.opacity(0.06), radius: 20, x: 0, y: 10)
    }
}

struct HealthMetricPill: View {
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            
            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(Color(red: 0.4, green: 0.4, blue: 0.5))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(color.opacity(0.1))
        .cornerRadius(12)
    }
}

struct ActivityItem: View {
    let icon: String
    let title: String
    let time: String
    let color: Color
    
    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(color.opacity(0.15))
                    .frame(width: 36, height: 36)
                
                Image(systemName: icon)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(color)
            }
            
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))
                
                Text(time)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(Color(red: 0.5, green: 0.5, blue: 0.6))
            }
            
            Spacer()
        }
    }
}
