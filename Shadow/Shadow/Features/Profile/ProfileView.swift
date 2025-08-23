import SwiftUI

struct ProfileView: View {
    let profile: UserProfile
    let onLogout: () -> Void
    let onDeleteAccount: () -> Void
    let onBack: () -> Void
    let onProfileUpdated: (UserProfile) -> Void

    // Edit mode state
    @State private var isEditingMode = false
    @State private var editedName: String = ""
    @State private var editedEmail: String = ""
    @State private var editedWorkRole: String = ""

    // Alerts and feedback
    @State private var showConfirmDelete = false
    @State private var showSaveSuccess = false
    @State private var showSaveError = false
    @State private var saveErrorMessage = ""

    // Hover states (for macOS)
    @State private var isHoveringSave = false
    @State private var isHoveringCancel = false
    @State private var isHoveringEdit = false
    @State private var isHoveringLogout = false
    @State private var isHoveringDelete = false

    @Environment(\.dismiss) private var dismiss

    var body: some View {
            GeometryReader { geometry in
                ZStack {
                    // App gradient background
                    LinearGradient(
                        gradient: Gradient(colors: [
                            Color(red: 0.05, green: 0.08, blue: 0.15),
                            Color(red: 0.1, green: 0.15, blue: 0.25)
                        ]),
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                    .ignoresSafeArea()

                    // Subtle pattern overlay
                    RoundedRectangle(cornerRadius: 0)
                        .fill(
                            RadialGradient(
                                gradient: Gradient(colors: [
                                    Color.white.opacity(0.02),
                                    Color.clear
                                ]),
                                center: .topTrailing,
                                startRadius: 0,
                                endRadius: geometry.size.width
                            )
                        )
                        .ignoresSafeArea()

                    ScrollView(showsIndicators: false) {
                                        VStack(spacing: 0) {
                                            headerSection
                                            VStack(spacing: 32) {
                                                if isEditingMode {
                                                    editFormSection
                                                } else {
                                                    profileInfoSection
                                                    accountActionsSection
                                                }
                                                Spacer(minLength: 32)
                                            }
                                            .padding(.horizontal, 40)
                                            .padding(.vertical, 40)
                                            .background(
                                                RoundedRectangle(cornerRadius: 32)
                                                    .fill(Color.white.opacity(0.95))
                                                    .shadow(color: Color.black.opacity(0.1), radius: 30, x: 0, y: 20)
                                            )
                                            .frame(maxWidth: 800)
                                        }
                                    }
                                }
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                            }
                            .onAppear(perform: initializeEditFields)
        .alert("Delete Account", isPresented: $showConfirmDelete) {
            Button("Delete", role: .destructive) { onDeleteAccount() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Are you sure you want to delete your account? This action cannot be undone and all your data will be permanently lost.")
        }
        .alert("Changes Saved", isPresented: $showSaveSuccess) {
            Button("OK", role: .cancel) {}
        } message: {
            Text("Your profile has been updated successfully.")
        }
        .alert("Save Failed", isPresented: $showSaveError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(saveErrorMessage)
        }
    }

    // MARK: - Header
    private var headerSection: some View {
        VStack(spacing: 32) {
            // Nav Bar
            HStack {
                Button(action: { onBack()}) {
                    HStack(spacing: 8) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 16, weight: .semibold))
                        Text("Back")
                            .font(.system(size: 16, weight: .semibold))
                    }
                    .foregroundColor(.white.opacity(0.8))
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(Color.white.opacity(0.1))
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.white.opacity(0.2), lineWidth: 1)
                    )
                }
                .buttonStyle(PlainButtonStyle())

                Spacer()

                Text(isEditingMode ? "Edit Profile" : "Profile")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundColor(.white)

                Spacer()

                Button(action: {
                    if isEditingMode {
                        saveChanges()
                    } else {
                        startEditing()
                    }
                }) {
                    HStack(spacing: 8) {
                        Image(systemName: isEditingMode ? "checkmark" : "pencil")
                            .font(.system(size: 16, weight: .semibold))
                        Text(isEditingMode ? "Save" : "Edit")
                            .font(.system(size: 16, weight: .semibold))
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(
                        LinearGradient(
                            colors: isEditingMode ? [.green, .green.opacity(0.8)] : [.purple, .blue],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(12)
                    .scaleEffect(isHoveringEdit ? 1.05 : 1.0)
                    .shadow(color: (isEditingMode ? Color.green : Color.purple ).opacity(0.3), radius: 8, x: 0, y: 4)
                }
                .buttonStyle(PlainButtonStyle())
                .onHover { hovering in
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isHoveringEdit = hovering
                    }
                }
            }
            .padding(.horizontal, 32)
            .padding(.top, 40)

            // Avatar
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
                    .frame(width: 120, height: 120)
                    .shadow(color: Color.purple.opacity(0.3), radius: 25, x: 0, y: 15)

                Text(getInitials(from: isEditingMode ? editedName : (profile.name ?? "User")))
                    .font(.system(size: 42, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)

                if isEditingMode {
                    VStack {
                        Spacer()
                        HStack {
                            Spacer()
                            Button(action: {}) {
                                Image(systemName: "camera.fill")
                                    .font(.system(size: 14, weight: .semibold))
                                    .foregroundColor(.white)
                                    .frame(width: 32, height: 32)
                                    .background(Color.purple.opacity(0.9))
                                    .cornerRadius(16)
                                    .shadow(color: .purple.opacity(0.3), radius: 8, x: 0, y: 4)
                            }
                            .buttonStyle(PlainButtonStyle())
                        }
                    }
                }
            }

            if !isEditingMode {
                VStack(spacing: 8) {
                    Text("Welcome back!")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.white.opacity(0.8))

                    Text(profile.name ?? "User")
                        .font(.system(size: 32, weight: .light, design: .rounded))
                        .foregroundColor(.white)
                        .lineLimit(2)
                        .multilineTextAlignment(.center)
                }
            }
        }
        .padding(.bottom, 40)
    }

    // MARK: - Edit Form
    private var editFormSection: some View {
        VStack(spacing: 24) {
            Text("Edit Your Information")
                .font(.system(size: 24, weight: .semibold, design: .rounded))
                .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))

            VStack(spacing: 20) {
                EditableField(
                    title: "Full Name",
                    icon: "person.fill",
                    placeholder: "Enter your full name",
                    text: $editedName
                )

                EditableField(
                    title: "Email Address",
                    icon: "envelope.fill",
                    placeholder: "Enter your email",
                    text: $editedEmail
                )
                .disabled(true)

                EditableField(
                    title: "Work Role/Title",
                    icon: "briefcase.fill",
                    placeholder: "Enter your work role",
                    text: $editedWorkRole
                )
            }

            // Action Buttons
            HStack(spacing: 16) {
                Button(action: cancelEditing) {
                    HStack(spacing: 8) {
                        Image(systemName: "xmark")
                            .font(.system(size: 16, weight: .semibold))
                        Text("Cancel")
                            .font(.system(size: 16, weight: .semibold))
                    }
                    .foregroundColor(Color(red: 0.5, green: 0.5, blue: 0.6))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(Color(red: 0.96, green: 0.97, blue: 0.98))
                    .cornerRadius(14)
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(Color(red: 0.9, green: 0.9, blue: 0.92), lineWidth: 1)
                    )
                    .scaleEffect(isHoveringCancel ? 1.02 : 1.0)
                }
                .buttonStyle(PlainButtonStyle())
                .onHover { hovering in
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isHoveringCancel = hovering
                    }
                }

                Button(action: saveChanges) {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark")
                            .font(.system(size: 16, weight: .semibold))
                        Text("Save Changes")
                            .font(.system(size: 16, weight: .semibold))
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(
                        LinearGradient(
                            colors: hasChanges() ? [.green, .green.opacity(0.8)] : [.gray.opacity(0.6), .gray.opacity(0.4)],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(14)
                    .scaleEffect(isHoveringSave && hasChanges() ? 1.02 : 1.0)
                    .shadow(
                        color: hasChanges() ? .green.opacity(0.3) : .clear,
                        radius: 8,
                        x: 0,
                        y: 4
                    )
                }
                .buttonStyle(PlainButtonStyle())
                .disabled(!hasChanges())
                .onHover { hovering in
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isHoveringSave = hovering
                    }
                }
            }
        }
    }

    // MARK: - Profile Info Card Section
    private var profileInfoSection: some View {
        VStack(spacing: 28) {
            Text("Profile Information")
                .font(.system(size: 24, weight: .semibold, design: .rounded))
                .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))

            VStack(spacing: 16) {
                ProfileInfoCard(
                    icon: "person.fill",
                    title: "Full Name",
                    value: profile.name ?? "Not provided",
                    color: .purple
                )

                ProfileInfoCard(
                    icon: "envelope.fill",
                    title: "Email Address",
                    value: profile.email ?? "Not provided",
                    color: .blue
                )

                ProfileInfoCard(
                    icon: "briefcase.fill",
                    title: "Work Role",
                    value: profile.workRole ?? "Not specified",
                    color: .green
                )

                ProfileInfoCard(
                    icon: "calendar",
                    title: "Member Since",
                    value: "August 2025",
                    color: .orange
                )
            }
        }
    }

    // MARK: - Account Actions Section
    private var accountActionsSection: some View {
        VStack(spacing: 16) {
            Text("Account Actions")
                .font(.system(size: 20, weight: .semibold))
                .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))
                .padding(.top, 16)

            Button(action: onLogout) {
                HStack(spacing: 12) {
                    Image(systemName: "rectangle.portrait.and.arrow.right")
                        .font(.system(size: 18, weight: .semibold))
                    Text("Sign Out")
                        .font(.system(size: 16, weight: .semibold))
                }
                .foregroundColor(Color(red: 0.4, green: 0.4, blue: 0.5))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(Color(red: 0.96, green: 0.97, blue: 0.98))
                .cornerRadius(14)
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Color(red: 0.9, green: 0.9, blue: 0.92), lineWidth: 1)
                )
                .scaleEffect(isHoveringLogout ? 1.02 : 1.0)
            }
            .buttonStyle(PlainButtonStyle())
            .onHover { hovering in
                withAnimation(.easeInOut(duration: 0.2)) {
                    isHoveringLogout = hovering
                }
            }

            Button(role: .destructive) {
                showConfirmDelete = true
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: "trash.fill")
                        .font(.system(size: 16, weight: .semibold))
                    Text("Delete Account")
                        .font(.system(size: 16, weight: .semibold))
                }
                .foregroundColor(.red)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(Color.red.opacity(0.05))
                .cornerRadius(14)
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Color.red.opacity(0.2), lineWidth: 1)
                )
                .scaleEffect(isHoveringDelete ? 1.02 : 1.0)
            }
            .buttonStyle(PlainButtonStyle())
            .onHover { hovering in
                withAnimation(.easeInOut(duration: 0.2)) {
                    isHoveringDelete = hovering
                }
            }
        }
    }

    // MARK: - Helper Functions

    private func getInitials(from name: String) -> String {
        let components = name.components(separatedBy: .whitespaces)
        let initials = components.compactMap { $0.first?.uppercased() }.prefix(2).joined()
        return initials.isEmpty ? "U" : initials
    }

    private func initializeEditFields() {
        editedName = profile.name ?? ""
        editedEmail = profile.email ?? ""
        editedWorkRole = profile.workRole ?? ""
    }

    private func startEditing() {
        initializeEditFields()
        withAnimation(.easeInOut(duration: 0.3)) {
            isEditingMode = true
        }
    }

    private func cancelEditing() {
        withAnimation(.easeInOut(duration: 0.3)) {
            isEditingMode = false
        }
        initializeEditFields()
    }

    private func saveChanges() {
        if hasChanges() {
            // Save to Core Data
            ProfileRepository.shared.saveProfile(
                email: editedEmail,
                name: editedName,
                workRole: editedWorkRole
            )

            // Reload updated profile from Core Data
            if let updated = ProfileRepository.shared.loadProfile(email: editedEmail) {
                onProfileUpdated(updated) // <--- Notify parent with updated profile
            }

            // Show success, exit edit mode
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                showSaveSuccess = true
                withAnimation(.easeInOut(duration: 0.3)) {
                    isEditingMode = false
                }
            }
        }
    }

    private func hasChanges() -> Bool {
        editedName != (profile.name ?? "") ||
        editedEmail != (profile.email ?? "") ||
        editedWorkRole != (profile.workRole ?? "")
    }
}

// MARK: - Supporting Views

struct EditableField: View {
    let title: String
    let icon: String
    let placeholder: String
    @Binding var text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(Color(red: 0.2, green: 0.2, blue: 0.3))

            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(Color(red: 0.4, green: 0.4, blue: 0.5))
                    .frame(width: 20)

                TextField(placeholder, text: $text)
                    .textFieldStyle(PlainTextFieldStyle())
                    .font(.system(size: 16, weight: .regular))
                    .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(Color(red: 0.96, green: 0.97, blue: 0.98))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(text.isEmpty ? Color(red: 0.9, green: 0.9, blue: 0.92) : Color.blue.opacity(0.5), lineWidth: 1)
            )
        }
    }
}

struct ProfileInfoCard: View {
    let icon: String
    let title: String
    let value: String
    let color: Color

    var body: some View {
        HStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(color.opacity(0.15))
                    .frame(width: 44, height: 44)

                Image(systemName: icon)
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(color)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(Color(red: 0.4, green: 0.4, blue: 0.5))

                Text(value)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))
            }

            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
        .background(Color(red: 0.98, green: 0.99, blue: 0.99))
        .cornerRadius(16)
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color(red: 0.94, green: 0.95, blue: 0.96), lineWidth: 1)
        )
    }
}
