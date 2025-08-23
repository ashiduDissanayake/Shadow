import SwiftUI

struct LoginView: View {
    @State private var email: String = ""
    @State private var password: String = ""
    @State private var showResetPassword: Bool = false
    @State private var showAuthError: Bool = false
    @State private var authErrorMessage: String?
    @State private var isHoveringLogin: Bool = false
    @State private var isHoveringBiometric: Bool = false
    @State private var showPassword: Bool = false
    @State private var showLoginSuccess: Bool = false
    @State private var showPasswordResetSuccess: Bool = false
    @State private var isLoggingIn: Bool = false
    @State private var alertType: AlertType? = nil
    @ObservedObject var authViewModel: AuthViewModel
    
    enum AlertType: Identifiable {
        case authError(String)
        case loginSuccess
        case passwordResetSuccess
        case biometricError(String)
        
        var id: String {
            switch self {
            case .authError: return "authError"
            case .loginSuccess: return "loginSuccess"
            case .passwordResetSuccess: return "passwordResetSuccess"
            case .biometricError: return "biometricError"
            }
        }
    }
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background with gradient
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color(red: 0.05, green: 0.08, blue: 0.15),
                        Color(red: 0.1, green: 0.15, blue: 0.25)
                    ]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea(.all)
                
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
                    .ignoresSafeArea(.all)
                
                HStack(spacing: 0) {
                    // Left side - Branding
                    VStack(spacing: 32) {
                        Spacer()
                        
                        // Shadow Logo
                        ZStack {
                            Circle()
                                .fill(
                                    LinearGradient(
                                        gradient: Gradient(colors: [
                                            Color(red: 0.4, green: 0.2, blue: 0.8),
                                            Color(red: 0.2, green: 0.4, blue: 0.9)
                                        ]),
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                                .frame(width: 80, height: 80)
                                .shadow(color: Color(red: 0.4, green: 0.2, blue: 0.8).opacity(0.3), radius: 20, x: 0, y: 10)
                            
                            Image(systemName: "figure.walk.motion")
                                .font(.system(size: 32, weight: .light))
                                .foregroundColor(.white)
                        }
                        
                        VStack(spacing: 16) {
                            Text("Shadow")
                                .font(.system(size: 48, weight: .ultraLight, design: .rounded))
                                .foregroundColor(.white)
                                .tracking(2)
                            
                            Text("Your Personal Health Guardian")
                                .font(.system(size: 16, weight: .light))
                                .foregroundColor(Color.white.opacity(0.7))
                                .multilineTextAlignment(.center)
                                .padding(.horizontal, 20)
                        }
                        
                        VStack(spacing: 12) {
                            FeaturePill(icon: "heart.fill", text: "24/7 Health Monitoring", color: .red)
                            FeaturePill(icon: "brain.head.profile", text: "AI-Powered Insights", color: .blue)
                            FeaturePill(icon: "shield.fill", text: "Privacy Protected", color: .green)
                        }
                        
                        Spacer()
                    }
                    .frame(width: geometry.size.width * 0.45)
                    .padding(.vertical, 40)
                    
                    // Right side - Login Form
                    VStack(spacing: 0) {
                        RoundedRectangle(cornerRadius: 24)
                            .fill(Color.white.opacity(0.95))
                            .overlay(
                                VStack(spacing: 32) {
                                    // Header
                                    VStack(spacing: 8) {
                                        Text("Welcome Back")
                                            .font(.system(size: 32, weight: .medium, design: .rounded))
                                            .foregroundColor(Color(red: 0.1, green: 0.1, blue: 0.2))
                                        
                                        Text("Sign in to continue monitoring your health")
                                            .font(.system(size: 14, weight: .regular))
                                            .foregroundColor(Color(red: 0.4, green: 0.4, blue: 0.5))
                                            .multilineTextAlignment(.center)
                                    }
                                    .padding(.top, 40)
                                    
                                    // Form Fields
                                    VStack(spacing: 24) {
                                        // Email Field
                                        VStack(alignment: .leading, spacing: 8) {
                                            Text("Email Address")
                                                .font(.system(size: 13, weight: .medium))
                                                .foregroundColor(Color(red: 0.2, green: 0.2, blue: 0.3))
                                            
                                            HStack(spacing: 12) {
                                                Image(systemName: "envelope")
                                                    .font(.system(size: 16, weight: .medium))
                                                    .foregroundColor(Color(red: 0.4, green: 0.4, blue: 0.5))
                                                    .frame(width: 20)
                                                
                                                TextField("Enter your email", text: $email)
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
                                                    .stroke(Color(red: 0.9, green: 0.9, blue: 0.92), lineWidth: 1)
                                            )
                                        }
                                        
                                        // Password Field
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
                                                        TextField("Enter your password", text: $password)
                                                    } else {
                                                        SecureField("Enter your password", text: $password)
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
                                                    .stroke(Color(red: 0.9, green: 0.9, blue: 0.92), lineWidth: 1)
                                            )
                                        }
                                    }
                                    
                                    // Login Button
                                    Button(action: {
                                        let result = authViewModel.login(email: email, password: password)
                                        if result {
                                            showLoginSuccess = true
                                        } else {
                                            // Always show error if login failed, use a default message if needed
                                            authErrorMessage = authViewModel.loginError ?? "Login failed. Please try again."
                                            showAuthError = true
                                        }
                                    }) {
                                        HStack(spacing: 8) {
                                            if isLoggingIn {
                                                ProgressView()
                                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                                    .scaleEffect(0.8)
                                                Text("Signing In...")
                                                    .font(.system(size: 16, weight: .semibold))
                                            } else {
                                                Image(systemName: "arrow.right")
                                                    .font(.system(size: 16, weight: .semibold))
                                                Text("Sign In")
                                                    .font(.system(size: 16, weight: .semibold))
                                            }
                                        }
                                        .foregroundColor(.white)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 16)
                                        .background(
                                            LinearGradient(
                                                gradient: Gradient(colors: [
                                                    Color(red: 0.4, green: 0.2, blue: 0.8),
                                                    Color(red: 0.3, green: 0.3, blue: 0.9)
                                                ]),
                                                startPoint: .leading,
                                                endPoint: .trailing
                                            )
                                        )
                                        .cornerRadius(12)
                                        .scaleEffect(isHoveringLogin ? 1.02 : 1.0)
                                        .shadow(
                                            color: Color(red: 0.4, green: 0.2, blue: 0.8).opacity(0.3),
                                            radius: isHoveringLogin ? 12 : 8,
                                            x: 0,
                                            y: isHoveringLogin ? 6 : 4
                                        )
                                    }
                                    .buttonStyle(PlainButtonStyle())
                                    .disabled(isLoggingIn)
                                    .onHover { isHovering in
                                        if !isLoggingIn {
                                            withAnimation(.easeInOut(duration: 0.2)) {
                                                isHoveringLogin = isHovering
                                            }
                                        }
                                    }
                                    
                                    // Biometric Authentication
                                    Button(action: {
                                        BiometricHelper.authenticateForPasswordReset { success, error in
                                            if success {
                                                showResetPassword = true
                                            } else {
                                                authErrorMessage = error?.localizedDescription ?? "Authentication failed."
                                                showAuthError = true
                                            }
                                        }
                                    }) {
                                        HStack(spacing: 12) {
                                            Image(systemName: "touchid")
                                                .font(.system(size: 20, weight: .medium))
                                                .foregroundColor(Color(red: 0.4, green: 0.2, blue: 0.8))
                                            
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text("Use Touch ID")
                                                    .font(.system(size: 16, weight: .medium))
                                                    .foregroundColor(Color(red: 0.2, green: 0.2, blue: 0.3))
                                                Text("Quick and secure access")
                                                    .font(.system(size: 12, weight: .regular))
                                                    .foregroundColor(Color(red: 0.5, green: 0.5, blue: 0.6))
                                            }
                                            
                                            Spacer()
                                        }
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 14)
                                        .background(Color(red: 0.98, green: 0.98, blue: 0.99))
                                        .cornerRadius(12)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke(
                                                    Color(red: 0.4, green: 0.2, blue: 0.8).opacity(0.2),
                                                    lineWidth: isHoveringBiometric ? 2 : 1
                                                )
                                        )
                                        .scaleEffect(isHoveringBiometric ? 1.01 : 1.0)
                                    }
                                    .buttonStyle(PlainButtonStyle())
                                    .onHover { isHovering in
                                        withAnimation(.easeInOut(duration: 0.2)) {
                                            isHoveringBiometric = isHovering
                                        }
                                    }
                                    
                                    // Footer
                                    Text("Forgot your password? Contact support")
                                        .font(.system(size: 12, weight: .regular))
                                        .foregroundColor(Color(red: 0.5, green: 0.5, blue: 0.6))
                                        .padding(.bottom, 40)
                                }
                                .padding(.horizontal, 40)
                            )
                            .frame(maxWidth: 480)
                            .shadow(color: Color.black.opacity(0.1), radius: 30, x: -10, y: 20)
                    }
                    .frame(width: geometry.size.width * 0.55)
                    .padding(.vertical, 40)
                    .padding(.trailing, 60)
                }
            }
        }
        .frame(minWidth: 1000, minHeight: 700)
        .alert(isPresented: $showAuthError) {
            Alert(
                title: Text("Authentication Failed"),
                message: Text(authErrorMessage ?? "Unable to authenticate."),
                dismissButton: .default(Text("OK"))
            )
        }
        .sheet(isPresented: $showResetPassword) {
            ResetPasswordView(email: email) {
                showResetPassword = false
            }
        }
        .onAppear {
            if let savedEmail = UserDefaults.standard.string(forKey: AppKeys.lastProfile) {
                self.email = savedEmail
            }
        }
    }
}
