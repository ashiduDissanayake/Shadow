//
//  PasswordHasher.swift
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-23.
//


import Foundation
import CryptoKit

struct PasswordHasher {
    static func hash(_ password: String) -> String {
        let data = Data(password.utf8)
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }
}