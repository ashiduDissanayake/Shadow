# ESP32 BLE Server Implementation Design

## Overview
This document details the design of the ESP32 BLE server implementation for the Shadow wellness platform, focusing on reliable communication, power efficiency, and performance optimization.

## System Architecture

### Core Components

#### 1. BLE Server Manager
- **Responsibilities**:
  - Initialize and manage BLE GATT server
  - Handle connection/disconnection events
  - Manage connection parameter negotiation
  - Implement power management state transitions

#### 2. Transport Layer
- **Responsibilities**:
  - Parse incoming messages from client
  - Serialize outgoing messages to client
  - Handle message acknowledgments
  - Implement error detection and recovery
  - Manage message queuing

#### 3. Sensor Data Handler
- **Responsibilities**:
  - Interface with sensor hardware (MPU6050, MAX30102, GSR)
  - Collect and process sensor data
  - Format data according to protocol specification
  - Manage data transmission scheduling

#### 4. Power Management Module
- **Responsibilities**:
  - Implement sleep/wake cycles
  - Optimize connection parameters for power efficiency
  - Manage advertising intervals
  - Handle idle timeout disconnections

## BLE GATT Service Structure

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

#### Incoming Message Processing
1. **Message Reception**:
   - Receive data through BLE write characteristic
   - Validate message header and footer
   - Calculate and verify CRC16 checksum
   - Parse message ID and payload

2. **Acknowledgment**:
   - Send ACK for valid messages
   - Send NACK for invalid/corrupted messages
   - Handle duplicate messages

#### Outgoing Message Processing
1. **Message Queuing**:
   - Queue messages with priority levels
   - Handle automatic retransmission
   - Manage message timeouts

2. **Transmission**:
   - Format messages according to protocol
   - Send through appropriate characteristic (Notify)
   - Wait for acknowledgment before sending next message

## Connection Management

### Connection States

#### 1. Disconnected State
- **Advertising**: Broadcast device availability
- **Power Mode**: Low power advertising
- **Transitions**: 
  - To Connecting when client connects
  - To Sleep after extended idle period

#### 2. Connecting State
- **Negotiation**: Exchange connection parameters
- **Authentication**: Handle pairing/bonding if required
- **Transitions**:
  - To Connected_Idle on successful connection
  - To Disconnected on connection failure

#### 3. Connected_Idle State
- **Monitoring**: Watch for client activity
- **Power Mode**: Optimized connection intervals
- **Transitions**:
  - To Connected_Active when data transfer begins
  - To Disconnected after idle timeout

#### 4. Connected_Active State
- **Data Transfer**: Handle active communication
- **Power Mode**: Fast connection intervals
- **Transitions**:
  - To Connected_Idle when data transfer completes
  - To Disconnected on error or client disconnect

#### 5. Sleep State
- **Power Mode**: Deep sleep with periodic wake-up
- **Monitoring**: Wake on specific events
- **Transitions**:
  - To Disconnected on wake-up event

### Connection Parameter Optimization

#### Active Connection Parameters
- **Connection Interval**: 20ms
- **Slave Latency**: 0
- **Supervision Timeout**: 2000ms

#### Idle Connection Parameters
- **Connection Interval**: 100ms
- **Slave Latency**: 4 (allow 4 connection events to be skipped)
- **Supervision Timeout**: 2000ms

#### Advertising Parameters
- **Fast Advertising**: 20ms interval when client searching
- **Slow Advertising**: 1000ms interval during idle periods

## Power Management Implementation

### Sleep/Wake Strategy

#### Wake-up Triggers
1. **Timer-based**: Periodic wake-up for advertising
2. **Button Press**: User interaction wake-up
3. **Sensor Event**: Significant sensor data change
4. **BLE Activity**: Client connection attempt

#### Sleep Modes
1. **Light Sleep**: CPU sleep, peripherals active
2. **Deep Sleep**: Most peripherals off, RTC active
3. **Hibernation**: Minimal power, wake only on reset

### Power Optimization Techniques

#### Dynamic Connection Parameters
- Adjust intervals based on data requirements
- Use slave latency for periodic data
- Optimize supervision timeout

#### Data Transmission Optimization
- Batch multiple sensor readings
- Compress data when possible
- Use notification instead of indication for high-frequency data

## Sensor Integration

### Supported Sensors

#### 1. MPU6050 (Accelerometer/Gyroscope)
- **Interface**: I2C
- **Data Rate**: Configurable (50-1000 Hz)
- **Power Management**: Low power modes

#### 2. MAX30102 (PPG Sensor)
- **Interface**: I2C
- **Data Rate**: Configurable LED pulse frequency
- **Power Management**: Shutdown mode when not in use

#### 3. GSR Sensor (Analog)
- **Interface**: ADC
- **Data Rate**: Configurable sampling rate
- **Power Management**: Power control through GPIO

### Data Collection Strategy

#### Sampling Synchronization
- Use hardware timers for precise sampling
- Implement buffer management for continuous data
- Handle sensor-specific initialization sequences

#### Data Processing
- Apply basic filtering (moving average, median)
- Convert raw values to physical units
- Package data according to protocol format

## Error Handling and Recovery

### Error Types

#### 1. Communication Errors
- **CRC Mismatch**: Corrupted message detection
- **Timeout**: Unacknowledged message handling
- **Connection Loss**: Automatic reconnection

#### 2. Hardware Errors
- **Sensor Failure**: Detection and reporting
- **Memory Issues**: Buffer overflow handling
- **Power Issues**: Brownout detection

#### 3. Protocol Errors
- **Invalid Commands**: Unknown command handling
- **Malformed Messages**: Incorrect message format
- **Sequence Errors**: Missing or duplicate messages

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
- Maintain transmission queue in non-volatile memory
- Resume transmission after reconnection
- Handle partial data transfers

## Implementation Details

### Required Libraries
1. **BLE Library**: NimBLE-Arduino or ESP32 BLE
2. **Sensor Libraries**: 
   - MPU6050 library
   - MAX30105 library
   - ADC management functions
3. **Utility Libraries**:
   - CRC16 calculation
   - Message queue management
   - Power management functions

### Memory Management
1. **Static Allocation**: Pre-allocate buffers for known sizes
2. **Dynamic Allocation**: Minimal use for variable data
3. **Buffer Management**: Circular buffers for sensor data
4. **Queue Management**: Priority queues for message handling

### Task Scheduling
1. **RTOS Tasks**:
   - BLE communication task (high priority)
   - Sensor data collection task (medium priority)
   - Power management task (low priority)
2. **Interrupt Handling**: 
   - Sensor data ready interrupts
   - Button press interrupts
   - Timer interrupts for scheduling

## Configuration Parameters

### Compile-time Configuration
```cpp
// Connection parameters
#define FAST_CONNECTION_INTERVAL 20    // ms
#define SLOW_CONNECTION_INTERVAL 100   // ms
#define SLAVE_LATENCY 0
#define SUPERVISION_TIMEOUT 2000       // ms

// Advertising parameters
#define FAST_ADVERTISING_INTERVAL 20    // ms
#define SLOW_ADVERTISING_INTERVAL 1000  // ms

// Message handling
#define MAX_RETRANSMISSION_ATTEMPTS 3
#define MESSAGE_TIMEOUT 1000           // ms
#define ACK_TIMEOUT 500               // ms

// Power management
#define IDLE_TIMEOUT 5000              // ms
#define SLEEP_ADVERTISING_INTERVAL 5000 // ms
```

### Runtime Configuration
1. **Data Sampling Rates**: Configurable per sensor
2. **Transmission Intervals**: Client-controlled
3. **Power Settings**: Battery vs. performance modes
4. **Error Handling**: Retry counts and timeouts

## Testing Strategy

### Unit Testing
1. **Message Parsing**: Validate protocol implementation
2. **CRC Calculation**: Verify error detection
3. **Queue Management**: Test message queuing and prioritization
4. **State Transitions**: Verify connection state machine

### Integration Testing
1. **BLE Communication**: End-to-end message exchange
2. **Sensor Integration**: Data collection and processing
3. **Power Management**: Sleep/wake cycles and optimization
4. **Error Recovery**: Failure scenarios and recovery

### Performance Testing
1. **Throughput**: Measure data transfer rates
2. **Latency**: Response time analysis
3. **Power Consumption**: Battery life impact
4. **Reliability**: Long-term stability testing

## Future Enhancements

### Advanced Features
1. **OTA Updates**: Wireless firmware update capability
2. **Data Logging**: Local storage with sync capability
3. **Advanced Analytics**: On-device signal processing
4. **Multi-device Support**: Connect to multiple sensors

### Scalability Improvements
1. **Mesh Networking**: Extend to multiple devices
2. **Protocol Extensions**: Support for additional sensors
3. **Security Enhancements**: Advanced encryption and authentication
4. **Cross-platform Compatibility**: Standardized interfaces

## Conclusion

This ESP32 BLE server implementation provides a robust foundation for reliable, efficient communication in the Shadow wellness platform. By implementing advanced connection management, power optimization, and error recovery mechanisms, the system ensures optimal performance while preserving battery life. The modular design allows for future enhancements and scalability as the platform evolves.