import SwiftUI

struct SignupView: View {
    @ObservedObject var authVM: AuthViewModel
    @State private var email: String = ""
    @State private var password: String = ""
    @State private var confirmPassword: String = ""
    @State private var name: String = ""
    @State private var workRole: String = ""
    @State private var showPassword: Bool = false
    @State private var showConfirmPassword: Bool = false
    @State private var isHoveringSignup: Bool = false
    @State private var showSignupSuccess: Bool = false
    @State private var showSignupError: Bool = false
    @State private var signupErrorMessage: String?

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Gradient background
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color(red: 0.05, green: 0.08, blue: 0.15),
                        Color(red: 0.1, green: 0.15, blue: 0.25)
                    ]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea(.all)
                // Pattern overlay
                RoundedRectangle(cornerRadius: 0)
                    .fill(
                        RadialGradient(
                            gradient: Gradient(colors: [
                                Color.white.opacity(0.02),
                                Color.clear
                            ]),
                            center: .topLeading,
                            startRadius: 0,
                            endRadius: geometry.size.width
                        )
                    )
                    .ignoresSafeArea(.all)
                HStack(spacing: 0) {
                    // Branding
                    VStack(spacing: 32) {
                        Spacer()
                        ZStack {
                            Circle()
                                .fill(
                                    LinearGradient(
                                        gradient: Gradient(colors: [
                                            Color.purple.opacity(0.7),
                                            Color.blue.opacity(0.6)
                                        ]),
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                                .frame(width: 80, height: 80)
                                .shadow(color: Color.purple.opacity(0.3), radius: 20, x: 0, y: 10)
                            Image(systemName: "figure.walk.motion")
                                .font(.system(size: 32, weight: .light))
                                .foregroundColor(.white)
                        }
                        VStack(spacing: 16) {
                            Text("Shadow")
                                .font(.system(size: 48, weight: .ultraLight, design: .rounded))
                                .foregroundColor(.white)
                                .tracking(2)
                            Text("Begin Your Health Journey")
                                .font(.system(size: 16, weight: .light))
                                .foregroundColor(Color.white.opacity(0.7))
                                .multilineTextAlignment(.center)
                                .padding(.horizontal, 20)
                        }
                        VStack(spacing: 12) {
                            FeaturePill(icon: "person.badge.plus", text: "Personalized Profile", color: .purple)
                            FeaturePill(icon: "chart.line.uptrend.xyaxis", text: "Health Analytics", color: .blue)
                            FeaturePill(icon: "lock.shield", text: "Secure & Private", color: .green)
                        }
                        Spacer()
                    }
                    .frame(width: geometry.size.width * 0.4)
                    .padding(.vertical, 40)

                    // Signup Card
                    ScrollView(showsIndicators: false) {
                        VStack {
                            VStack(spacing: 28) {
                                VStack(spacing: 8) {
                                    Text("Create Account")
                                        .font(.system(size: 32, weight: .medium, design: .rounded))
                                        .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))
                                    Text("Join thousands who trust Shadow with their health")
                                        .font(.system(size: 14, weight: .regular))
                                        .foregroundColor(Color(red: 0.4, green: 0.4, blue: 0.5))
                                        .multilineTextAlignment(.center)
                                }
                                .padding(.top, 40)

                                VStack(spacing: 20) {
                                    FormField(title: "Email Address", icon: "envelope", placeholder: "Enter your email", text: $email, isSecure: false)
                                    FormField(title: "Full Name", icon: "person", placeholder: "Enter your full name", text: $name, isSecure: false)
                                    FormField(title: "Work Role/Title", icon: "briefcase", placeholder: "e.g., Software Engineer, Doctor", text: $workRole, isSecure: false)

                                    // Password field with show/hide and strength indicator
                                    VStack(alignment: .leading, spacing: 8) {
                                        Text("Password")
                                            .font(.system(size: 13, weight: .medium))
                                            .foregroundColor(Color(red: 0.2, green: 0.2, blue: 0.3))
                                        HStack(spacing: 12) {
                                            Image(systemName: "lock")
                                                .font(.system(size: 16, weight: .medium))
                                                .foregroundColor(Color(red: 0.4, green: 0.4, blue: 0.5))
                                                .frame(width: 20)
                                            Group {
                                                if showPassword {
                                                    TextField("Create a secure password", text: $password)
                                                } else {
                                                    SecureField("Create a secure password", text: $password)
                                                }
                                            }
                                            .textFieldStyle(PlainTextFieldStyle())
                                            .font(.system(size: 16, weight: .regular))
                                            .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))
                                            Button(action: { showPassword.toggle() }) {
                                                Image(systemName: showPassword ? "eye.slash" : "eye")
                                                    .font(.system(size: 16, weight: .medium))
                                                    .foregroundColor(Color(red: 0.5, green: 0.5, blue: 0.6))
                                            }
                                            .buttonStyle(PlainButtonStyle())
                                        }
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 14)
                                        .background(Color(red: 0.96, green: 0.97, blue: 0.98))
                                        .cornerRadius(12)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke(passwordStrengthColor(), lineWidth: password.isEmpty ? 1 : 2)
                                        )
                                        if !password.isEmpty {
                                            HStack(spacing: 4) {
                                                ForEach(0..<4) { idx in
                                                    Rectangle()
                                                        .fill(idx < passwordStrength() ? passwordStrengthColor() : Color.gray.opacity(0.2))
                                                        .frame(height: 3)
                                                        .cornerRadius(2)
                                                }
                                            }
                                            .padding(.top, 4)
                                            Text(passwordStrengthText())
                                                .font(.system(size: 12, weight: .medium))
                                                .foregroundColor(passwordStrengthColor())
                                                .padding(.top, 2)
                                        }
                                    }

                                    // Confirm password
                                    VStack(alignment: .leading, spacing: 8) {
                                        Text("Confirm Password")
                                            .font(.system(size: 13, weight: .medium))
                                            .foregroundColor(Color(red: 0.2, green: 0.2, blue: 0.3))
                                        HStack(spacing: 12) {
                                            Image(systemName: "lock.fill")
                                                .font(.system(size: 16, weight: .medium))
                                                .foregroundColor(Color(red: 0.4, green: 0.4, blue: 0.5))
                                                .frame(width: 20)
                                            Group {
                                                if showConfirmPassword {
                                                    TextField("Confirm your password", text: $confirmPassword)
                                                } else {
                                                    SecureField("Confirm your password", text: $confirmPassword)
                                                }
                                            }
                                            .textFieldStyle(PlainTextFieldStyle())
                                            .font(.system(size: 16, weight: .regular))
                                            .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))
                                            Button(action: { showConfirmPassword.toggle() }) {
                                                Image(systemName: showConfirmPassword ? "eye.slash" : "eye")
                                                    .font(.system(size: 16, weight: .medium))
                                                    .foregroundColor(Color(red: 0.5, green: 0.5, blue: 0.6))
                                            }
                                            .buttonStyle(PlainButtonStyle())
                                        }
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 14)
                                        .background(Color(red: 0.96, green: 0.97, blue: 0.98))
                                        .cornerRadius(12)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke(passwordMatchBorderColor(), lineWidth: confirmPassword.isEmpty ? 1 : 2)
                                        )
                                        if !confirmPassword.isEmpty {
                                            HStack(spacing: 6) {
                                                Image(systemName: passwordsMatch() ? "checkmark.circle.fill" : "xmark.circle.fill")
                                                    .font(.system(size: 12, weight: .medium))
                                                    .foregroundColor(passwordsMatch() ? .green : .red)
                                                Text(passwordsMatch() ? "Passwords match" : "Passwords don't match")
                                                    .font(.system(size: 12, weight: .medium))
                                                    .foregroundColor(passwordsMatch() ? .green : .red)
                                            }
                                            .padding(.top, 4)
                                        }
                                    }
                                }

                                // Error Message
                                if let error = authVM.loginError {
                                    HStack(spacing: 8) {
                                        Image(systemName: "exclamationmark.triangle.fill")
                                            .font(.system(size: 14, weight: .medium))
                                            .foregroundColor(.red)
                                        Text(error)
                                            .font(.system(size: 14, weight: .medium))
                                            .foregroundColor(.red)
                                    }
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 12)
                                    .background(Color.red.opacity(0.1))
                                    .cornerRadius(10)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 10)
                                            .stroke(Color.red.opacity(0.3), lineWidth: 1)
                                    )
                                }

                                // Signup Button
                                Button(action: { handleSignup() }) {
                                    HStack(spacing: 8) {
                                        Image(systemName: "person.badge.plus")
                                            .font(.system(size: 16, weight: .semibold))
                                        Text("Create Account")
                                            .font(.system(size: 16, weight: .semibold))
                                    }
                                    .foregroundColor(.white)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 16)
                                    .background(
                                        LinearGradient(
                                            gradient: Gradient(colors: canSignUp() ? [
                                                Color(red: 0.4, green: 0.2, blue: 0.8),
                                                Color(red: 0.3, green: 0.3, blue: 0.9)
                                            ] : [
                                                Color.gray.opacity(0.6),
                                                Color.gray.opacity(0.4)
                                            ]),
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        )
                                    )
                                    .cornerRadius(12)
                                    .scaleEffect(isHoveringSignup && canSignUp() ? 1.02 : 1.0)
                                    .shadow(
                                        color: canSignUp() ? Color.purple.opacity(0.3) : Color.clear,
                                        radius: isHoveringSignup && canSignUp() ? 12 : 8,
                                        x: 0,
                                        y: isHoveringSignup && canSignUp() ? 6 : 4
                                    )
                                }
                                .buttonStyle(PlainButtonStyle())
                                .disabled(!canSignUp())
                                .onHover { isHovering in
                                    withAnimation(.easeInOut(duration: 0.2)) {
                                        isHoveringSignup = isHovering
                                    }
                                }
                                Spacer(minLength: 16)
                            }
                            .padding(.horizontal, 40)
                            .padding(.bottom, 30)
                        }
                        .background(
                            RoundedRectangle(cornerRadius: 24)
                                .fill(Color.white.opacity(0.95))
                                .shadow(color: Color.black.opacity(0.1), radius: 30, x: -10, y: 20)
                        )
                        .padding(.vertical, 40)
                        .frame(maxWidth: 500)
                        .fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(width: geometry.size.width * 0.6)
                    .padding(.trailing, 60)
                }
            }
        }
        .frame(minWidth: 1000, minHeight: 800)
        .alert(isPresented: $showSignupError) {
            Alert(
                title: Text("Signup Failed"),
                message: Text(signupErrorMessage ?? "Unable to create account."),
                dismissButton: .default(Text("OK"))
            )
        }
        .alert(isPresented: $showSignupSuccess) {
            Alert(
                title: Text("Account Created"),
                message: Text("Welcome to Shadow! Your account has been created successfully."),
                dismissButton: .default(Text("Get Started")) {
                    showSignupSuccess = false
                }
            )
        }
    }

    // MARK: - Helper Functions

    private func handleSignup() {
        authVM.loginError = nil
        if !passwordsMatch() {
            authVM.loginError = "Passwords do not match."
            return
        }
        let success = authVM.signup(
            email: email.trimmingCharacters(in: .whitespacesAndNewlines),
            password: password,
            name: name.trimmingCharacters(in: .whitespacesAndNewlines),
            workRole: workRole.trimmingCharacters(in: .whitespacesAndNewlines)
        )
        if success {
            showSignupSuccess = true
        } else if let error = authVM.loginError {
            signupErrorMessage = error
            showSignupError = true
        }
    }
    private func passwordsMatch() -> Bool { password == confirmPassword }
    private func canSignUp() -> Bool {
        !email.isEmpty && !password.isEmpty && !confirmPassword.isEmpty && !name.isEmpty && !workRole.isEmpty && passwordsMatch() && passwordStrength() >= 2
    }
    private func passwordStrength() -> Int {
        let len = password.count
        let hasU = password.range(of: "[A-Z]", options: .regularExpression) != nil
        let hasL = password.range(of: "[a-z]", options: .regularExpression) != nil
        let hasN = password.range(of: "[0-9]", options: .regularExpression) != nil
        let hasS = password.range(of: "[^A-Za-z0-9]", options: .regularExpression) != nil
        var s = 0
        if len >= 8 { s += 1 }
        if hasU && hasL { s += 1 }
        if hasN { s += 1 }
        if hasS { s += 1 }
        return min(s, 4)
    }
    private func passwordStrengthText() -> String {
        switch passwordStrength() {
        case 0, 1: return "Weak password"
        case 2: return "Fair password"
        case 3: return "Good password"
        case 4: return "Strong password"
        default: return ""
        }
    }
    private func passwordStrengthColor() -> Color {
        switch passwordStrength() {
        case 0, 1: return .red
        case 2: return .orange
        case 3: return .blue
        case 4: return .green
        default: return Color(red: 0.9, green: 0.9, blue: 0.92)
        }
    }
    private func passwordMatchBorderColor() -> Color {
        if confirmPassword.isEmpty { return Color(red: 0.9, green: 0.9, blue: 0.92) }
        return passwordsMatch() ? .green : .red
    }
}

struct FormField: View {
    let title: String
    let icon: String
    let placeholder: String
    @Binding var text: String
    let isSecure: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(Color(red: 0.2, green: 0.2, blue: 0.3))
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(Color(red: 0.4, green: 0.4, blue: 0.5))
                    .frame(width: 20)
                if isSecure {
                    SecureField(placeholder, text: $text)
                        .textFieldStyle(PlainTextFieldStyle())
                        .font(.system(size: 16, weight: .regular))
                        .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))
                } else {
                    TextField(placeholder, text: $text)
                        .textFieldStyle(PlainTextFieldStyle())
                        .font(.system(size: 16, weight: .regular))
                        .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(Color(red: 0.96, green: 0.97, blue: 0.98))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color(red: 0.9, green: 0.9, blue: 0.92), lineWidth: 1)
            )
        }
    }
}
