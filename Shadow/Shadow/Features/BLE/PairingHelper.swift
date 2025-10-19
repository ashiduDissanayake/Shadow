//
//  PairingHelper.swift
//  Shadow
//
//  Created on 18/10/2025.
//  SHA-256 challenge-response authentication for BLE pairing
//

import Foundation
import CommonCrypto

/// Helper for BLE device pairing operations
struct PairingHelper {
    
    // MARK: - Client Device ID Management
    
    /// Get or create persistent client device UUID
    static func getOrCreateClientDeviceID() -> Data {
        let key = "Shadow.ClientDeviceID"
        
        if let existingID = UserDefaults.standard.data(forKey: key) {
            return existingID
        }
        
        // Generate new UUID (16 bytes)
        let uuid = UUID()
        var uuidBytes = uuid.uuid
        let data = Data(bytes: &uuidBytes, count: MemoryLayout.size(ofValue: uuidBytes))
        
        UserDefaults.standard.set(data, forKey: key)
        print("✅ Generated new client device ID: \(data.hexString)")
        return data
    }
    
    /// Get client device name (Mac hostname)
    static func getClientDeviceName() -> String {
        return Host.current().localizedName ?? "Mac"
    }
    
    // MARK: - Challenge-Response Authentication
    
    /// Compute SHA-256 challenge response
    /// Formula: response = SHA-256(challenge + shadow_device_id)[0:16]
    ///
    /// - Parameters:
    ///   - challenge: 16-byte random challenge from Shadow device
    ///   - shadowDeviceID: 16-byte Shadow device UUID
    /// - Returns: 16-byte response (first 16 bytes of SHA-256 hash)
    static func computeChallengeResponse(challenge: Data, shadowDeviceID: Data) -> Data {
        // Concatenate challenge + shadow_device_id
        var input = Data()
        input.append(challenge)
        input.append(shadowDeviceID)
        
        // Compute SHA-256 hash
        var hash = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        input.withUnsafeBytes {
            _ = CC_SHA256($0.baseAddress, CC_LONG(input.count), &hash)
        }
        
        // Return first 16 bytes
        return Data(hash.prefix(16))
    }
    
    /// Prepare complete challenge response for writing to Security Challenge characteristic
    ///
    /// - Parameters:
    ///   - challenge: 16-byte challenge from Shadow device
    ///   - shadowDeviceID: 16-byte Shadow device UUID
    ///   - clientDeviceID: 16-byte client UUID
    ///   - clientName: Client device name (variable length)
    /// - Returns: Complete data to write (response + client_id + client_name)
    static func prepareChallengeResponse(
        challenge: Data,
        shadowDeviceID: Data,
        clientDeviceID: Data,
        clientName: String
    ) -> Data {
        // 1. Compute SHA-256 response
        let response = computeChallengeResponse(challenge: challenge, shadowDeviceID: shadowDeviceID)
        
        // 2. Prepare write data: response(16) + client_id(16) + client_name(N)
        var writeData = Data()
        writeData.append(response)                                // 16 bytes
        writeData.append(clientDeviceID)                          // 16 bytes
        writeData.append(clientName.data(using: .utf8) ?? Data()) // Variable length
        
        return writeData
    }
    
    // MARK: - Pairing Info Storage
    
    /// Save pairing information to UserDefaults
    static func savePairingInfo(deviceInfo: DeviceInfo, clientDeviceID: Data) {
        let key = "Shadow.PairingInfo.\(deviceInfo.deviceName)"
        
        let info: [String: Any] = [
            "shadowDeviceID": deviceInfo.deviceID,
            "shadowDeviceName": deviceInfo.deviceName,
            "shadowFirmware": deviceInfo.firmwareVersion,
            "shadowHardware": deviceInfo.hardwareRevision,
            "clientDeviceID": clientDeviceID,
            "pairTimestamp": Date(),
            "lastConnected": Date()
        ]
        
        UserDefaults.standard.set(info, forKey: key)
        print("✅ Saved pairing info for \(deviceInfo.deviceName)")
    }
    
    /// Load pairing information from UserDefaults
    static func loadPairingInfo(deviceName: String) -> [String: Any]? {
        let key = "Shadow.PairingInfo.\(deviceName)"
        return UserDefaults.standard.dictionary(forKey: key)
    }
    
    /// Check if device is already paired
    static func isPaired(deviceName: String) -> Bool {
        return loadPairingInfo(deviceName: deviceName) != nil
    }
    
    /// Remove pairing information
    static func removePairingInfo(deviceName: String) {
        let key = "Shadow.PairingInfo.\(deviceName)"
        UserDefaults.standard.removeObject(forKey: key)
        print("❌ Removed pairing info for \(deviceName)")
    }
}

// MARK: - Data Extension

extension Data {
    /// Convert data to hex string
    var hexString: String {
        map { String(format: "%02x", $0) }.joined()
    }
}
