import Foundation
import CoreData

class AuthViewModel: ObservableObject {
    private let keychainAccountPrefix = AppKeys.keychainAccountPrefix
    private let userdefaultkey = AppKeys.lastProfile
    @Published var isLoggedIn: Bool = false
    @Published var loginError: String? = nil
    @Published var profile: UserProfile? = nil
    @Published var hasExistingProfile: Bool = false

    init() {
        if let email = storedProfileEmail(),
           let loadedProfile = ProfileRepository.shared.loadProfile(email: email),
           KeychainHelper.getPassword(for: keychainAccountPrefix + email) != nil {
            self.profile = loadedProfile
            self.isLoggedIn = true
            self.hasExistingProfile = true
        } else {
            self.isLoggedIn = false
            self.profile = nil
            self.hasExistingProfile = ProfileRepository.shared.hasAnyProfile()
        }
    }

    func signup(email: String, password: String, name: String, workRole: String) -> Bool {
        if hasExistingProfile {
            loginError = "A profile already exists. Please log in."
            return false
        }
        let hashed = PasswordHasher.hash(password)
        guard KeychainHelper.savePassword(hashed, for: keychainAccountPrefix + email) else {
            loginError = "Failed to save password."
            return false
        }
        ProfileRepository.shared.saveProfile(email: email, name: name, workRole: workRole)
        self.profile = ProfileRepository.shared.loadProfile(email: email)
        isLoggedIn = true
        loginError = nil
        hasExistingProfile = true
        storeProfileEmail(email)
        return true
    }

    func login(email: String, password: String) -> Bool {
        loginError = nil
        
        guard let storedHash = KeychainHelper.getPassword(for: keychainAccountPrefix + email) else {
            loginError = "No account for this email."
            return false
        }
        
        let enteredHash = PasswordHasher.hash(password)
        if storedHash == enteredHash {
            self.profile = ProfileRepository.shared.loadProfile(email: email)
            isLoggedIn = true
            storeProfileEmail(email)
            return true
        } else {
            loginError = "Incorrect password. Please try again."
            isLoggedIn = false
            return false
        }
    }

    func logout() {
        isLoggedIn = false
        profile = nil
        loginError = nil
    }

    func deleteAccount(email: String) {
        KeychainHelper.deletePassword(for: keychainAccountPrefix + email)
        ProfileRepository.shared.deleteProfile(email: email)
        isLoggedIn = false
        profile = nil
        loginError = nil
        hasExistingProfile = ProfileRepository.shared.hasAnyProfile()
        clearStoredProfileEmail()
    }

    // Helpers for last used email, to support session and auto-fill
    private func storeProfileEmail(_ email: String) {
        UserDefaults.standard.set(email, forKey: userdefaultkey)
    }
    private func storedProfileEmail() -> String? {
        UserDefaults.standard.string(forKey: userdefaultkey)
    }
    private func clearStoredProfileEmail() {
        UserDefaults.standard.removeObject(forKey: userdefaultkey)
    }
}
