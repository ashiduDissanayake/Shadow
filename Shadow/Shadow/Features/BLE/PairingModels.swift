//
//  PairingModels.swift
//  Shadow
//
//  Created on 18/10/2025.
//  Data models for BLE device pairing protocol
//

import Foundation

// MARK: - Pairing Service UUIDs

/// Pairing Service UUID (separate from Stress Service)
let PAIRING_SERVICE_UUID_STRING = "0000B000-0000-1000-8000-00805F9B34FB"

/// Device Info Characteristic (READ) - 80 bytes
let CHAR_UUID_DEVICE_INFO_STRING = "0000B001-0000-1000-8000-00805F9B34FB"

/// Pairing State Characteristic (READ, NOTIFY) - 3 bytes
let CHAR_UUID_PAIRING_STATE_STRING = "0000B002-0000-1000-8000-00805F9B34FB"

/// Pairing Control Characteristic (WRITE) - 1-17 bytes
let CHAR_UUID_PAIRING_CONTROL_STRING = "0000B003-0000-1000-8000-00805F9B34FB"

/// Security Challenge Characteristic (READ, WRITE)
let CHAR_UUID_SECURITY_CHALLENGE_STRING = "0000B004-0000-1000-8000-00805F9B34FB"

// MARK: - Pairing Commands

enum PairingCommand: UInt8 {
    case pairRequest = 1
    case unpair = 2
    case clearAll = 3
}

// MARK: - Pairing States

enum PairingState: UInt8, CustomStringConvertible {
    case idle = 0
    case advertising = 1
    case connected = 2
    case pending = 3
    case paired = 4
    case rejected = 5
    
    var description: String {
        switch self {
        case .idle: return "Idle"
        case .advertising: return "Advertising"
        case .connected: return "Connected"
        case .pending: return "Pending"
        case .paired: return "Paired"
        case .rejected: return "Rejected"
        }
    }
    
    var emoji: String {
        switch self {
        case .idle: return "⏸️"
        case .advertising: return "📡"
        case .connected: return "🔗"
        case .pending: return "⏳"
        case .paired: return "✅"
        case .rejected: return "❌"
        }
    }
}

// MARK: - Device Info Model

/// Shadow device identification information
struct DeviceInfo {
    let deviceID: Data           // 16 bytes - Shadow UUID
    let deviceName: String       // 32 bytes - "Shadow-XXXX"
    let firmwareVersion: String  // 16 bytes - "v1.0.0"
    let hardwareRevision: String // 16 bytes - "ESP32-S3"
    
    /// Parse device info from 80-byte characteristic read
    init?(from data: Data) {
        guard data.count >= 80 else { return nil }
        
        self.deviceID = data.subdata(in: 0..<16)
        
        let nameData = data.subdata(in: 16..<48)
        self.deviceName = String(data: nameData, encoding: .utf8)?
            .trimmingCharacters(in: .controlCharacters.union(.whitespaces)) ?? "Unknown"
        
        let fwData = data.subdata(in: 48..<64)
        self.firmwareVersion = String(data: fwData, encoding: .utf8)?
            .trimmingCharacters(in: .controlCharacters.union(.whitespaces)) ?? "Unknown"
        
        let hwData = data.subdata(in: 64..<80)
        self.hardwareRevision = String(data: hwData, encoding: .utf8)?
            .trimmingCharacters(in: .controlCharacters.union(.whitespaces)) ?? "Unknown"
    }
    
    var deviceIDHex: String {
        deviceID.map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Pairing State Info Model

/// Pairing state information
struct PairingStateInfo {
    let state: PairingState
    let pairedCount: UInt8
    let maxPaired: UInt8
    
    /// Parse pairing state from 3-byte characteristic read
    init?(from data: Data) {
        guard data.count >= 3 else { return nil }
        
        self.state = PairingState(rawValue: data[0]) ?? .idle
        self.pairedCount = data[1]
        self.maxPaired = data[2]
    }
}

// MARK: - Security Challenge Model

/// Security challenge for authentication
struct SecurityChallenge {
    let challenge: Data      // 16 bytes - random challenge
    let timestamp: UInt64    // 8 bytes - microseconds
    
    /// Parse challenge from 24-byte characteristic read
    init?(from data: Data) {
        guard data.count >= 24 else { return nil }
        
        self.challenge = data.subdata(in: 0..<16)
        self.timestamp = data.subdata(in: 16..<24).withUnsafeBytes { 
            $0.load(as: UInt64.self)
        }
    }
}

// MARK: - Pairing Error

enum PairingError: Error, LocalizedError {
    case bluetoothNotAvailable
    case deviceNotConnected
    case characteristicNotFound
    case invalidData
    case timeout
    case rejected
    case challengeExpired
    case alreadyPaired
    case maxDevicesReached
    
    var errorDescription: String? {
        switch self {
        case .bluetoothNotAvailable:
            return "Bluetooth is not available"
        case .deviceNotConnected:
            return "Device is not connected"
        case .characteristicNotFound:
            return "Required characteristic not found"
        case .invalidData:
            return "Invalid data received"
        case .timeout:
            return "Pairing timeout"
        case .rejected:
            return "Pairing rejected by device"
        case .challengeExpired:
            return "Security challenge expired"
        case .alreadyPaired:
            return "Device is already paired"
        case .maxDevicesReached:
            return "Maximum paired devices reached"
        }
    }
}

// MARK: - Constants

struct PairingConfig {
    /// Maximum number of devices that can be paired simultaneously
    static let maxPairedDevices: UInt8 = 3
    
    /// Timeout for challenge-response (30 seconds)
    static let challengeTimeoutSeconds: TimeInterval = 30.0
    
    /// Device ID length (UUID)
    static let deviceIDLength = 16
    
    /// Challenge length
    static let challengeLength = 16
    
    /// Device name max length
    static let deviceNameMaxLength = 32
    
    /// Firmware version max length
    static let firmwareVersionMaxLength = 16
    
    /// Hardware revision max length
    static let hardwareRevisionMaxLength = 16
}
