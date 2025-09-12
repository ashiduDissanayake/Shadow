# Shadow BLE Protocol Specification (RFC-Style)

**Document:** Shadow-BLE-001  
**Version:** 1.0  
**Date:** December 2025  
**Author:** Ashidu Dissanayake  
**Status:** Implementation Draft

---

## Abstract

This document specifies the Bluetooth Low Energy (BLE) communication protocol used by the Shadow stress detection system. The protocol enables efficient synchronization of stress state transitions between ESP32-based wearable devices and macOS client applications with support for data loss recovery and ring buffer management.

---

## 1. Introduction

### 1.1 Purpose
The Shadow BLE Protocol provides reliable, low-power communication for real-time stress monitoring data between embedded devices and mobile/desktop applications.

### 1.2 Key Features
- **Ring Buffer Management**: 32-entry circular buffer with 7-bit sequence numbering
- **State Synchronization**: Binary stress state tracking (CALM=0, STRESS=1) 
- **Data Loss Recovery**: Automatic detection and recovery of missed transitions
- **Power Efficiency**: Advertisement-based discovery with connection-on-demand
- **Reset Handling**: Protocol-level reset with sequence counter management

---

## 2. Protocol Architecture

### 2.1 Service Definition
- **Service UUID**: `A000`
- **Event Characteristic UUID**: `A002`
- **Device Name**: `"Shadow"`

### 2.2 Communication Model
```
ESP32 Device     <--> BLE Advertisement <--> macOS Client <--> Core Data
     |                        |                    |              |
 [Ring Buffer]    [State Broadcast]    [Synchronization]   [Persistence]
```

### 2.3 State Machine
```
IDLE → ADVERTISING → CONNECTED → REQUESTING → SYNCING → ADVERTISING
 ↓         ↓            ↓           ↓           ↓          ↓
ERROR ← CONNECTION_FAIL ← PROTOCOL_ERROR ← TIMEOUT ← RETRY
```

---

## 3. Packet Formats

### 3.1 Advertisement Packet
```
Byte 0: SSSSSSS S
        ||||||| |
        ||||||| +-- State bit (0=CALM, 1=STRESS)
        +---------- 7-bit sequence number (0-127)
```

**Example**: `0x0B` = sequence 5, stress state  
**Calculation**: `(5 << 1) | 1 = 0x0B`

### 3.2 Reset Request
```
Byte 0: 0xFF (Reset opcode)
```

### 3.3 Reset Response
```
Byte 0: 0x00 (Reserved)
Byte 1: XXXXXX S (Current state)
Byte 2: 0x00 (Reserved) 
Byte 3: 0x52 (Reset magic confirmation)
```

### 3.4 Missed Data Request
```
Byte 0: LLLLLLL (Last known sequence, 7-bit)
```

### 3.5 Minimal Response (Delta = 1)
```
Byte 0: CCCCCCC (Current sequence)
Byte 1: XXXXXX S (Current state)
```

### 3.6 Extended Response (Delta > 1)
```
Byte 0: CCCCCCC (Current sequence)
Byte 1: XXXXXX S (Current state)
Byte 2: MMMMMMM (Missed count)
Byte 3+: [Sequence, State] pairs for each missed entry
```

---

## 4. Protocol Operations

### 4.1 Device Discovery
1. ESP32 continuously advertises with service data
2. macOS scans for devices named "Shadow" with service A000
3. Advertisement data contains latest sequence and state

### 4.2 Synchronization Flow
```
macOS → ESP32: Connect to BLE device
macOS → ESP32: Discover service A000, characteristic A002  
macOS → ESP32: Write last known sequence number
ESP32 → macOS: Read response with current state + missed data
macOS → ESP32: Disconnect
```

### 4.3 Delta Calculation
```c
uint8_t delta = (new_sequence - last_sequence) & 0x7F;
```
- **Delta = 0**: No new data, ignore
- **Delta = 1**: Single transition, can use advertisement only (optional)
- **Delta > 1**: Multiple transitions, requires connection
- **Delta > 32**: Ring buffer overflow, requires reset

### 4.4 Reset Protocol
Used when data loss exceeds ring buffer capacity:
```
macOS → ESP32: Write 0xFF (reset opcode)
ESP32 → macOS: Read reset confirmation with current state
ESP32: Clear ring buffer, reset sequence to 0
macOS: Increment reset counter, persist reset marker
```

---

## 5. Error Handling

### 5.1 Connection Failures
- **Timeout**: Return to scanning mode
- **Service Discovery Fail**: Retry connection after throttle period (1.5s)
- **Characteristic Missing**: Log error, disconnect

### 5.2 Protocol Errors
- **Malformed Response**: Log error, request retry
- **Unexpected Data Length**: Log warning, attempt to parse
- **Reset Magic Mismatch**: Continue with normal parsing

### 5.3 Data Integrity
- **Sequence Gaps**: Automatic gap detection and recovery
- **State Validation**: Binary validation (0 or 1 only)
- **Duplicate Detection**: Check against existing database entries

---

## 6. Performance Characteristics

### 6.1 Timing Parameters
- **Advertisement Interval**: 1000ms typical
- **Connection Throttle**: 1.5s minimum between attempts
- **Ring Buffer Size**: 32 entries (supports ~32 seconds of transitions)
- **Sequence Wrap**: 128 values (7-bit), wraps every ~2 minutes

### 6.2 Power Consumption
- **Advertisement Mode**: ~50μA continuous
- **Connection Mode**: ~10mA for 100-500ms
- **Total Average**: <100μA with typical usage patterns

### 6.3 Data Throughput
- **Advertisement**: 1 byte per second
- **Sync Burst**: Up to 67 bytes (32 missed entries + headers)
- **Latency**: <2 seconds from state change to app notification

---

## 7. Security Considerations

### 7.1 Data Protection
- **Encryption**: Relies on BLE link-layer encryption
- **Authentication**: Device name verification only
- **Privacy**: No personally identifiable information in broadcasts

### 7.2 Denial of Service
- **Connection Throttling**: Prevents rapid connection attempts
- **Advertisement Validation**: Rejects malformed packets
- **Resource Limits**: Ring buffer prevents memory exhaustion

---

## 8. Implementation Notes

### 8.1 ESP32 Firmware
- **FreeRTOS Tasks**: Separate tasks for sensing, ML inference, and BLE
- **Ring Buffer**: Interrupt-safe circular buffer implementation
- **Power Management**: Sleep modes between advertisements

### 8.2 macOS Client
- **Core Bluetooth**: Native iOS/macOS BLE framework
- **Core Data**: SQLite-backed persistence with relationship management
- **SwiftUI**: Reactive UI updates from published BLE manager state

### 8.3 Cross-Platform Compatibility
- **Standard BLE**: Uses standard GATT services and characteristics
- **Endianness**: Little-endian byte order throughout
- **MTU Requirements**: Minimum 23 bytes (standard BLE minimum)

---

## 9. Future Extensions

### 9.1 Planned Features
- **Firmware Timestamp**: 64-bit device timestamps for events
- **Confidence Scores**: ML model confidence levels
- **Battery Monitoring**: Voltage level reporting
- **Sensor Quality**: Data quality metrics

### 9.2 Protocol Versioning
- **Version Field**: Reserved bit in advertisement packet
- **Capability Negotiation**: Service characteristic for feature discovery
- **Backward Compatibility**: Protocol designed for extensibility

---

## 10. References

- **Bluetooth Core Specification 5.0+**
- **ESP32-S3 Technical Reference Manual**
- **Apple Core Bluetooth Programming Guide**
- **Shadow Firmware Implementation (C/FreeRTOS)**
- **Shadow macOS Application (Swift/SwiftUI)**

---

**End of Document**

*This specification is implemented in the Shadow stress detection system as of December 2025.*
