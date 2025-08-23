//
//  KeychainHelper.swift
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-23.
//


import Foundation
import Security

struct KeychainHelper {
    static func savePassword(_ password: String, for account: String) -> Bool {
        guard let passwordData = password.data(using: .utf8) else { return false }

        let query = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrAccount: account
        ] as CFDictionary
        SecItemDelete(query)

        let attributes = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrAccount: account,
            kSecValueData: passwordData
        ] as CFDictionary

        let status = SecItemAdd(attributes, nil)
        return status == errSecSuccess
    }

    static func getPassword(for account: String) -> String? {
        let query = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrAccount: account,
            kSecReturnData: true,
            kSecMatchLimit: kSecMatchLimitOne
        ] as CFDictionary

        var result: AnyObject?
        let status = SecItemCopyMatching(query, &result)
        if status == errSecSuccess, let data = result as? Data {
            return String(data: data, encoding: .utf8)
        }
        return nil
    }

    static func deletePassword(for account: String) {
        let query = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrAccount: account
        ] as CFDictionary
        SecItemDelete(query)
    }
}