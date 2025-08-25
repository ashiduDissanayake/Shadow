import SwiftUI
import CoreData

struct ContentView: View {
    @StateObject private var authVM = AuthViewModel()
    @StateObject private var bleManager = BLEManager()
    @State private var showingProfilePage = false
    @State private var showingCalendar = false // <-- Add this line

    var body: some View {
        NavigationStack {
            ZStack {
                // Optional: global background
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color(red: 0.05, green: 0.08, blue: 0.15),
                        Color(red: 0.1, green: 0.15, blue: 0.25)
                    ]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                VStack(spacing: 0) {
                    // AppNavBar: Only show when user is logged in
                    if let profile = authVM.profile, authVM.isLoggedIn {
                        AppNavBar(
                            title: "Shadow",
                            subtitle: "TinyML Stress Detection",
                            profile: profile,
                            onProfileTap: { showingProfilePage = true },
                            onLogout: handleLogout,
                            showProfileMenu: true,
                            onCalendarTap: { showingCalendar = true }, // <-- Pass closure here
                            bleManager: bleManager // Pass BLE manager for status/actions
                        )
                    }
                    // Main content
                    Group {
                        content
                    }
                }
            }
            // Present your calendar view as a sheet or via navigation.
            .sheet(isPresented: $showingCalendar) {
                CalendarMainView()
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        if authVM.isLoggedIn, let profile = authVM.profile {
            if showingProfilePage {
                ProfileView(
                    profile: profile,
                    onLogout: handleLogout,
                    onDeleteAccount: { handleDeleteAccount(for: profile) },
                    onBack: { showingProfilePage = false },
                    onProfileUpdated: handleProfileUpdated
                )
            } else {
                DashboardView(
                    profile: profile,
                    onLogout: handleLogout,
                    onDeleteAccount: { handleDeleteAccount(for: profile) },
                    onShowProfile: { showingProfilePage = true }
                )
            }
        } else if authVM.hasExistingProfile {
            LoginView(authViewModel: authVM)
        } else {
            SignupView(authVM: authVM)
        }
    }

    // MARK: - Handlers

    private func handleLogout() {
        showingProfilePage = false
        authVM.logout()
    }

    private func handleDeleteAccount(for profile: UserProfile) {
        showingProfilePage = false
        authVM.deleteAccount(email: profile.email ?? "")
    }

    private func handleProfileUpdated(_ updatedProfile: UserProfile) {
        authVM.profile = updatedProfile
        showingProfilePage = false
    }
}
