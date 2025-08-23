import SwiftUI

struct ResetPasswordView: View {
    @State private var newPassword: String = ""
    @State private var confirmPassword: String = ""
    @State private var error: String?
    let email: String
    let onComplete: () -> Void

    @State private var isHoveringSave = false

    var body: some View {
        ZStack {
            // Glassy dark card background
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .fill(
                    LinearGradient(
                        gradient: Gradient(colors: [
                            Color.white.opacity(0.10),
                            Color.white.opacity(0.05)
                        ]),
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .background(.ultraThinMaterial)
                .shadow(color: Color.black.opacity(0.19), radius: 30, y: 10)
            
            VStack(spacing: 28) {
                // Title
                VStack(spacing: 8) {
                    ZStack {
                        Circle()
                            .fill(
                                LinearGradient(
                                    colors: [Color.purple.opacity(0.8), Color.blue.opacity(0.7)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 60, height: 60)
                            .shadow(color: Color.purple.opacity(0.25), radius: 12, y: 8)
                        Image(systemName: "lock.rotation")
                            .font(.system(size: 28, weight: .semibold))
                            .foregroundColor(.white)
                    }
                    Text("Reset Your Password")
                        .font(.system(size: 22, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                    Text("Enter and confirm your new password below.")
                        .font(.system(size: 15, weight: .medium))
                        .foregroundColor(.white.opacity(0.65))
                }
                .padding(.bottom, 8)
                
                VStack(spacing: 18) {
                    passwordField(
                        title: "New Password",
                        placeholder: "Enter new password",
                        text: $newPassword
                    )
                    passwordField(
                        title: "Confirm Password",
                        placeholder: "Confirm new password",
                        text: $confirmPassword
                    )
                }
                
                if let error = error {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.system(size: 14, weight: .semibold))
                        .multilineTextAlignment(.center)
                        .padding(.top, 2)
                }
                
                Button(action: savePassword) {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                        Text("Save New Password")
                            .fontWeight(.semibold)
                    }
                    .font(.system(size: 17))
                    .foregroundColor(.white)
                    .padding(.horizontal, 32)
                    .padding(.vertical, 14)
                    .background(
                        LinearGradient(
                            colors: [Color.purple, Color.blue],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(14)
                    .shadow(color: Color.purple.opacity(0.18), radius: 10, x: 0, y: 4)
                    .scaleEffect(isHoveringSave ? 1.04 : 1.0)
                }
                .buttonStyle(PlainButtonStyle())
                .onHover { hovering in
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isHoveringSave = hovering
                    }
                }
                .padding(.top, 6)
            }
            .padding(34)
            .frame(width: 370)
        }
        .padding(.vertical, 54)
        .padding(.horizontal, 10)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.clear) // transparent for modal
    }

    private func passwordField(title: String, placeholder: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(.white.opacity(0.8))
            SecureField(placeholder, text: text)
                .textFieldStyle(PlainTextFieldStyle())
                .padding(.horizontal, 16)
                .padding(.vertical, 14)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.white.opacity(0.13))
                        .background(.ultraThinMaterial)
                        .shadow(color: Color.black.opacity(0.08), radius: 2, y: 1)
                )
                .foregroundColor(.white)
                .font(.system(size: 16, weight: .regular))
        }
    }

    private func savePassword() {
        if newPassword.isEmpty || confirmPassword.isEmpty {
            error = "All fields are required."
        } else if newPassword != confirmPassword {
            error = "Passwords do not match."
        } else {
            // Save new password to Keychain
            let hashed = PasswordHasher.hash(newPassword)
            if KeychainHelper.savePassword(hashed, for: AppKeys.keychainAccountPrefix + email) {
                onComplete()
            } else {
                error = "Failed to update password."
            }
        }
    }
}
