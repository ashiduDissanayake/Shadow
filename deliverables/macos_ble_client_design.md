# macOS BLE Client Implementation Design

## Overview
This document details the design of the macOS BLE client implementation for the Shadow wellness platform, focusing on reliable communication, performance optimization, and user-friendly interface.

## System Architecture

### Core Components

#### 1. BLE Client Manager
- **Responsibilities**:
  - Device discovery and connection management
  - Service and characteristic discovery
  - Connection state monitoring
  - Handle device pairing and bonding

#### 2. Transport Layer
- **Responsibilities**:
  - Message serialization and deserialization
  - Handle message acknowledgments
  - Implement error detection and recovery
  - Manage message queuing and prioritization

#### 3. Application Interface
- **Responsibilities**:
  - Provide high-level API for data exchange
  - Handle event callbacks for status updates
  - Manage configuration settings
  - Implement user interface bindings

#### 4. Connection Manager
- **Responsibilities**:
  - Optimize connection parameters for performance
  - Implement automatic reconnection logic
  - Manage connection lifecycle
  - Handle power-efficient connection strategies

## BLE Service Structure

### Custom Service (UUID: A000)
- **Name**: Shadow Wellness Service
- **Description**: Main service for Shadow wellness platform communication

#### Characteristics:

1. **Data Characteristic (UUID: A001)**
   - **Properties**: Read, Write, Notify
   - **Description**: Bidirectional data transfer with acknowledgment
   - **Size**: Up to 247 bytes (maximum BLE 5.0 MTU)

2. **Control Characteristic (UUID: A002)**
   - **Properties**: Read, Write
   - **Description**: Device control commands and configuration
   - **Size**: Up to 247 bytes

3. **Status Characteristic (UUID: A003)**
   - **Properties**: Read, Notify
   - **Description**: Device status updates and health information
   - **Size**: Up to 247 bytes

4. **Command Response Characteristic (UUID: A004)**
   - **Properties**: Notify
   - **Description**: Asynchronous command responses
   - **Size**: Up to 247 bytes

## Message Protocol Implementation

### Message Structure

#### Data Message Format
```
+--------+--------+--------+--------+--------+--------+
| Header | Msg ID | Length | Payload | CRC16  | Footer |
+--------+--------+--------+--------+--------+--------+
| 2B     | 4B     | 2B     | Variable| 2B     | 2B     |
+--------+--------+--------+--------+--------+--------+
```

- **Header**: Fixed value (0xAA55) for message identification
- **Message ID**: Unique 32-bit identifier for each message
- **Length**: 16-bit length of payload in bytes
- **Payload**: Variable-length data payload
- **CRC16**: 16-bit cyclic redundancy check for error detection
- **Footer**: Fixed value (0x55AA) for message termination

#### Control Message Format
```
+--------+--------+--------+--------+--------+
| Header | Cmd ID | Length | Payload | CRC16  |
+--------+--------+--------+--------+--------+
| 2B     | 2B     | 2B     | Variable| 2B     |
+--------+--------+--------+--------+--------+
```

### Message Handling

#### Outgoing Message Processing
1. **Message Creation**:
   - Generate unique message ID
   - Format payload according to protocol
   - Calculate and append CRC16 checksum
   - Add header and footer

2. **Transmission**:
   - Queue message for transmission
   - Send through appropriate characteristic
   - Wait for acknowledgment
   - Handle timeouts and retries

#### Incoming Message Processing
1. **Message Reception**:
   - Receive data through characteristic notifications
   - Validate message header and footer
   - Calculate and verify CRC16 checksum
   - Parse message ID and payload

2. **Acknowledgment**:
   - Send ACK for valid messages
   - Send NACK for invalid/corrupted messages
   - Handle duplicate messages

## Connection Management

### Connection States

#### 1. Scanning State
- **Activity**: Searching for Shadow devices
- **Transitions**: 
  - To Connecting when target device found
  - To Idle after scan timeout

#### 2. Connecting State
- **Activity**: Establishing connection with device
- **Transitions**:
  - To Connected_Idle on successful connection
  - To Scanning on connection failure

#### 3. Connected_Idle State
- **Activity**: Connected but no active data transfer
- **Transitions**:
  - To Connected_Active when data transfer begins
  - To Disconnected after idle timeout or error

#### 4. Connected_Active State
- **Activity**: Active data transfer in progress
- **Transitions**:
  - To Connected_Idle when data transfer completes
  - To Disconnected on error or client disconnect

#### 5. Disconnected State
- **Activity**: No connection to device
- **Transitions**:
  - To Scanning when reconnection initiated
  - To Idle after cleanup

### Connection Parameter Optimization

#### Active Connection Parameters
- **Connection Interval**: Request 20ms
- **Slave Latency**: 0
- **Supervision Timeout**: 2000ms

#### Idle Connection Parameters
- **Connection Interval**: Request 100ms
- **Slave Latency**: 4 (allow 4 connection events to be skipped)
- **Supervision Timeout**: 2000ms

## Power Management Implementation

### Connection Lifecycle Management

#### Efficient Connection Strategy
1. **Connect on Demand**: Establish connection only when needed
2. **Disconnect After Idle**: Automatically disconnect after inactivity
3. **Fast Reconnect**: Cache connection parameters for quick reconnection
4. **Background Monitoring**: Monitor for device availability

### System Integration
1. **App Nap Compatibility**: Work with macOS energy saving features
2. **Background Tasks**: Use appropriate background task APIs
3. **Resource Management**: Release resources when not needed

## Message Queuing and Prioritization

### Queue Structure

#### 1. High Priority Queue
- **Content**: Critical control commands
- **Processing**: Bypass normal queue, immediate transmission
- **Examples**: Start/stop data streaming, emergency commands

#### 2. Normal Priority Queue
- **Content**: Regular data and commands
- **Processing**: First-in, first-out processing
- **Examples**: Configuration updates, status requests

#### 3. Low Priority Queue
- **Content**: Non-critical background tasks
- **Processing**: Process when higher priority queues empty
- **Examples**: Firmware version checks, statistics requests

### Queue Management

#### Message Lifecycle
1. **Queued**: Message waiting for transmission
2. **Transmitting**: Message being sent to device
3. **Acknowledged**: Message successfully received by device
4. **Timed Out**: No acknowledgment received within timeout
5. **Failed**: Maximum retry attempts exceeded

#### Retry Mechanism
1. **Exponential Backoff**: Increase delay between retries
2. **Maximum Attempts**: Limit retry attempts (default 3)
3. **Queue Priority**: Maintain priority during retries
4. **Failure Handling**: Move to failed message queue for analysis

## Error Handling and Recovery

### Error Types

#### 1. Communication Errors
- **CRC Mismatch**: Corrupted message detection
- **Timeout**: Unacknowledged message handling
- **Connection Loss**: Automatic reconnection

#### 2. Protocol Errors
- **Invalid Commands**: Unknown command handling
- **Malformed Messages**: Incorrect message format
- **Sequence Errors**: Missing or duplicate messages

#### 3. System Errors
- **Resource Exhaustion**: Memory or buffer issues
- **API Failures**: CoreBluetooth framework errors
- **User Cancellation**: Explicit disconnect requests

### Recovery Mechanisms

#### Automatic Retransmission
- Retry failed transmissions (max 3 attempts)
- Exponential backoff between retries
- Clear queue on persistent failures

#### Connection Recovery
- Exponential backoff for reconnection attempts
- Reset connection parameters on repeated failures
- Enter safe mode after multiple failed attempts

#### Data Recovery
- Maintain transmission queue in memory
- Resume transmission after reconnection
- Handle partial data transfers

## API Design

### Core Classes

#### 1. ShadowBLEClient
- **Primary interface** for BLE communication
- **Manages** connection lifecycle
- **Handles** message queuing and transmission

#### 2. ShadowMessage
- **Represents** a single protocol message
- **Manages** serialization and deserialization
- **Handles** CRC calculation and validation

#### 3. ShadowDevice
- **Represents** a connected Shadow device
- **Manages** device-specific state
- **Handles** sensor data processing

### Key Methods

#### ShadowBLEClient
```swift
// Connection management
func startScanning() -> Void
func stopScanning() -> Void
func connect(to device: ShadowDevice) -> Void
func disconnect() -> Void

// Message handling
func sendMessage(_ message: ShadowMessage) -> Void
func sendControlCommand(_ command: Command, payload: Data?) -> Void

// Event callbacks
var onConnected: (() -> Void)?
var onDisconnected: (() -> Void)?
var onDataReceived: ((ShadowMessage) -> Void)?
var onStatusReceived: ((ShadowMessage) -> Void)?
```

#### ShadowMessage
```swift
// Message creation
init(command: Command, payload: Data?)
init(data: Data)

// Properties
var messageId: UInt32 { get }
var command: Command { get }
var payload: Data? { get }

// Validation
func isValid() -> Bool
func calculateCRC() -> UInt16
```

## User Interface Integration

### SwiftUI Integration
```swift
struct BLEConnectionView: View {
    @ObservedObject var bleManager: ShadowBLEClient
    
    var body: some View {
        VStack {
            if bleManager.isConnected {
                ConnectedView(device: bleManager.connectedDevice)
            } else {
                ScanningView(isScanning: bleManager.isScanning)
            }
        }
        .onAppear {
            bleManager.startScanning()
        }
    }
}
```

### Notification Center Integration
- **User Notifications**: Inform user of connection events
- **Status Updates**: Provide real-time status information
- **Error Alerts**: Notify user of critical errors

## Performance Optimization

### Memory Management
1. **Object Pooling**: Reuse message objects to reduce allocations
2. **Buffer Management**: Pre-allocate buffers for known sizes
3. **Automatic Cleanup**: Release resources when not needed

### Threading Model
1. **Main Queue**: UI updates and callbacks
2. **Background Queue**: BLE operations and processing
3. **Serial Queues**: Ensure thread-safe operations

### Data Processing
1. **Batch Processing**: Combine multiple small messages
2. **Asynchronous Processing**: Non-blocking operations
3. **Efficient Parsing**: Optimized data parsing algorithms

## Configuration Parameters

### Compile-time Configuration
```swift
// Connection parameters
let fastConnectionInterval: TimeInterval = 0.02    // 20ms
let slowConnectionInterval: TimeInterval = 0.1    // 100ms
let supervisionTimeout: TimeInterval = 2.0        // 2000ms

// Message handling
let maxRetransmissionAttempts = 3
let messageTimeout: TimeInterval = 1.0            // 1000ms
let ackTimeout: TimeInterval = 0.5                // 500ms

// Scanning parameters
let scanTimeout: TimeInterval = 10.0              // 10 seconds
```

### Runtime Configuration
1. **Connection Preferences**: User-configurable connection settings
2. **Data Rate Settings**: Adjustable data transmission rates
3. **Power Management**: Battery vs. performance modes
4. **Error Handling**: Retry counts and timeouts

## Testing Strategy

### Unit Testing
1. **Message Parsing**: Validate protocol implementation
2. **CRC Calculation**: Verify error detection
3. **Queue Management**: Test message queuing and prioritization
4. **State Transitions**: Verify connection state machine

### Integration Testing
1. **BLE Communication**: End-to-end message exchange
2. **Device Integration**: Connection and data exchange
3. **Error Recovery**: Failure scenarios and recovery
4. **Performance Testing**: Throughput and latency analysis

### UI Testing
1. **Connection Flow**: Device discovery and connection
2. **Data Display**: Real-time data visualization
3. **Error Handling**: User feedback for errors
4. **State Management**: UI state synchronization

## Security Considerations

### Data Protection
1. **Encryption**: Use BLE link-layer encryption
2. **Authentication**: Device pairing and bonding
3. **Data Integrity**: CRC checks for all messages
4. **Access Control**: Whitelist authorized devices

### Privacy
1. **Device Identity**: Use randomized MAC addresses
2. **Data Anonymization**: Remove personally identifiable information
3. **Local Processing**: Process data on device when possible

## Future Enhancements

### Advanced Features
1. **Multi-device Support**: Connect to multiple Shadow devices
2. **Data Synchronization**: Sync data with cloud services
3. **Advanced Analytics**: Real-time data processing and visualization
4. **Customizable UI**: User-configurable dashboard

### Cross-platform Compatibility
1. **iOS Support**: Extend to iOS devices
2. **macOS Extensions**: Safari extension for web integration
3. **Command Line Tools**: Terminal-based utilities
4. **API Standardization**: RESTful API for external integration

## Conclusion

This macOS BLE client implementation provides a robust, efficient, and user-friendly interface for the Shadow wellness platform. By implementing advanced connection management, message queuing, and error recovery mechanisms, the system ensures reliable communication while providing an excellent user experience. The modular design allows for future enhancements and cross-platform compatibility as the platform evolves.