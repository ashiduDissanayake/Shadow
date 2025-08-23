//
//  BiometricHelper.swift
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-24.
//


import LocalAuthentication

class BiometricHelper {
    static func authenticateForPasswordReset(completion: @escaping (Bool, Error?) -> Void) {
        let context = LAContext()
        let reason = "Authenticate to reset your password"
        var error: NSError?

        // Try biometrics, fallback to device password
        if context.canEvaluatePolicy(.deviceOwnerAuthentication, error: &error) {
            context.evaluatePolicy(.deviceOwnerAuthentication, localizedReason: reason) { success, authError in
                DispatchQueue.main.async {
                    completion(success, authError)
                }
            }
        } else {
            DispatchQueue.main.async {
                completion(false, error)
            }
        }
    }
}