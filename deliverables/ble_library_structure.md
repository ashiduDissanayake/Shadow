# Shadow BLE Library Structure

## Overview
This document outlines the structure for the Shadow BLE communication library, providing a consistent and reusable framework for both ESP32 firmware and macOS client implementations.

## Library Architecture

### Core Components

#### 1. Protocol Layer
- **Responsibilities**:
  - Message serialization and deserialization
  - CRC calculation and validation
  - Message ID generation and tracking
  - Protocol version management

#### 2. Transport Layer
- **Responsibilities**:
  - Message queuing and prioritization
  - Acknowledgment handling
  - Retransmission management
  - Connection state management

#### 3. Device Abstraction Layer
- **Responsibilities**:
  - Platform-specific BLE implementation
  - Hardware interface abstraction
  - Resource management
  - Error handling and recovery

#### 4. Application Interface Layer
- **Responsibilities**:
  - High-level API for application integration
  - Event callback system
  - Configuration management
  - Data processing and formatting

## Directory Structure

```
shadow-ble-library/
├── include/
│   ├── shadow_ble_protocol.h
│   ├── shadow_ble_transport.h
│   ├── shadow_ble_device.h
│   └── shadow_ble_api.h
├── src/
│   ├── protocol/
│   │   ├── message_parser.c
│   │   ├── crc16.c
│   │   └── message_queue.c
│   ├── transport/
│   │   ├── connection_manager.c
│   │   ├── ack_handler.c
│   │   └── retransmission.c
│   ├── device/
│   │   ├── esp32/
│   │   │   ├── ble_server.c
│   │   │   ├── sensor_interface.c
│   │   │   └── power_management.c
│   │   └── macos/
│   │       ├── ble_client.c
│   │       ├── ui_integration.c
│   │       └── background_tasks.c
│   └── api/
│       ├── data_handler.c
│       ├── event_callbacks.c
│       └── configuration.c
├── examples/
│   ├── esp32_example/
│   │   ├── main.c
│   │   └── platformio.ini
│   └── macos_example/
│       ├── main.swift
│       └── Package.swift
├── tests/
│   ├── protocol_tests/
│   ├── transport_tests/
│   ├── device_tests/
│   └── integration_tests/
├── docs/
│   ├── protocol_specification.md
│   ├── api_reference.md
│   └── implementation_guide.md
├── CMakeLists.txt
├── library.json
└── README.md
```

## Protocol Layer Implementation

### Message Format Constants
```c
// Message structure constants
#define SHADOW_MSG_HEADER 0xAA55
#define SHADOW_MSG_FOOTER 0x55AA
#define SHADOW_MSG_MAX_SIZE 247

// Message types
typedef enum {
    SHADOW_MSG_TYPE_DATA = 0x01,
    SHADOW_MSG_TYPE_CONTROL = 0x02,
    SHADOW_MSG_TYPE_STATUS = 0x03,
    SHADOW_MSG_TYPE_ACK = 0x10,
    SHADOW_MSG_TYPE_NACK = 0x11
} shadow_msg_type_t;

// Control commands
typedef enum {
    SHADOW_CMD_START_DATA = 0x01,
    SHADOW_CMD_STOP_DATA = 0x02,
    SHADOW_CMD_SET_CONFIG = 0x03,
    SHADOW_CMD_GET_CONFIG = 0x04,
    SHADOW_CMD_SLEEP = 0x05,
    SHADOW_CMD_WAKEUP = 0x06,
    SHADOW_CMD_DISCONNECT = 0x07
} shadow_cmd_t;
```

### Message Structure
```c
typedef struct {
    uint16_t header;
    uint32_t message_id;
    uint16_t length;
    uint8_t* payload;
    uint16_t crc16;
    uint16_t footer;
} shadow_message_t;
```

### Protocol Functions
```c
// Message creation and parsing
shadow_message_t* shadow_msg_create(shadow_msg_type_t type, uint8_t* payload, uint16_t length);
int shadow_msg_parse(uint8_t* raw_data, uint16_t length, shadow_message_t* message);
void shadow_msg_destroy(shadow_message_t* message);

// CRC functions
uint16_t shadow_crc16(uint8_t* data, uint16_t length);
int shadow_msg_validate(shadow_message_t* message);

// Message ID management
uint32_t shadow_msg_generate_id(void);
```

## Transport Layer Implementation

### Message Queue
```c
typedef enum {
    SHADOW_MSG_PRIORITY_HIGH = 0,
    SHADOW_MSG_PRIORITY_NORMAL = 1,
    SHADOW_MSG_PRIORITY_LOW = 2
} shadow_msg_priority_t;

typedef struct {
    shadow_message_t* message;
    shadow_msg_priority_t priority;
    uint32_t timestamp;
    uint8_t retry_count;
    uint8_t ack_received;
} shadow_queued_message_t;

typedef struct {
    shadow_queued_message_t messages[SHADOW_MSG_QUEUE_SIZE];
    uint16_t head;
    uint16_t tail;
    uint16_t count;
} shadow_message_queue_t;
```

### Transport Functions
```c
// Queue management
int shadow_queue_push(shadow_message_queue_t* queue, shadow_message_t* message, shadow_msg_priority_t priority);
shadow_message_t* shadow_queue_pop(shadow_message_queue_t* queue);
int shadow_queue_remove(shadow_message_queue_t* queue, uint32_t message_id);

// Acknowledgment handling
int shadow_ack_wait(uint32_t message_id, uint32_t timeout_ms);
int shadow_ack_send(uint32_t message_id, int success);

// Retransmission management
int shadow_retransmit_message(uint32_t message_id);
int shadow_handle_timeout(uint32_t message_id);
```

## Device Abstraction Layer

### Platform-specific Interfaces

#### ESP32 Implementation
```c
// BLE server functions
int esp32_ble_init(void);
int esp32_ble_start_advertising(void);
int esp32_ble_send_notification(uint16_t characteristic_uuid, uint8_t* data, uint16_t length);
int esp32_ble_send_indication(uint16_t characteristic_uuid, uint8_t* data, uint16_t length);

// Sensor interface functions
int esp32_sensor_init(void);
int esp32_sensor_read_data(sensor_data_t* data);
int esp32_sensor_set_sampling_rate(uint32_t rate_hz);

// Power management functions
int esp32_power_enter_sleep(uint32_t duration_ms);
int esp32_power_wake_up(void);
int esp32_power_optimize_connection_params(void);
```

#### macOS Implementation
```c
// BLE client functions
int macos_ble_init(void);
int macos_ble_start_scan(void);
int macos_ble_connect(const char* device_address);
int macos_ble_write_characteristic(uint16_t characteristic_uuid, uint8_t* data, uint16_t length);

// UI integration functions
void macos_ui_update_status(const char* status);
void macos_ui_show_error(const char* error_message);
void macos_ui_update_sensor_data(sensor_data_t* data);

// Background task functions
int macos_background_start(void);
int macos_background_stop(void);
```

## Application Interface Layer

### High-level API

#### Common API Functions
```c
// Library initialization
int shadow_ble_init(platform_config_t* config);

// Connection management
int shadow_ble_connect(const char* device_address);
int shadow_ble_disconnect(void);
int shadow_ble_is_connected(void);

// Message sending
int shadow_ble_send_data(uint8_t* data, uint16_t length);
int shadow_ble_send_control_command(shadow_cmd_t command, uint8_t* payload, uint16_t length);

// Event callbacks
typedef void (*shadow_connected_callback_t)(void);
typedef void (*shadow_disconnected_callback_t)(int reason);
typedef void (*shadow_data_received_callback_t)(uint8_t* data, uint16_t length);
typedef void (*shadow_status_received_callback_t)(uint8_t* status, uint16_t length);

void shadow_ble_set_connected_callback(shadow_connected_callback_t callback);
void shadow_ble_set_disconnected_callback(shadow_disconnected_callback_t callback);
void shadow_ble_set_data_received_callback(shadow_data_received_callback_t callback);
void shadow_ble_set_status_received_callback(shadow_status_received_callback_t callback);
```

#### Data Structures
```c
typedef struct {
    uint64_t timestamp;
    struct {
        float x, y, z;
    } accelerometer;
    struct {
        uint32_t ir;
        uint32_t red;
    } ppg;
    struct {
        uint16_t raw;
        float voltage;
    } gsr;
} sensor_data_t;

typedef struct {
    // Platform-specific configuration
    #ifdef ESP32
    uint8_t ble_name[32];
    uint32_t advertising_interval;
    #elif defined(MACOS)
    uint32_t scan_timeout;
    int auto_reconnect;
    #endif
    
    // Common configuration
    uint32_t message_timeout;
    uint8_t max_retries;
    uint32_t connection_interval_min;
    uint32_t connection_interval_max;
} platform_config_t;
```

## Implementation Guidelines

### Cross-platform Compatibility

#### C Standard Compliance
- Use C99 standard for maximum compatibility
- Avoid platform-specific language features
- Use standard library functions when possible

#### Conditional Compilation
```c
#ifdef ESP32
    #include "esp32_specific.h"
    // ESP32-specific implementation
#elif defined(MACOS)
    #include "macos_specific.h"
    // macOS-specific implementation
#endif
```

#### Memory Management
- Provide clear memory allocation/deallocation guidelines
- Use stack allocation when possible
- Document memory ownership for dynamically allocated data

### Error Handling

#### Error Codes
```c
typedef enum {
    SHADOW_ERR_SUCCESS = 0,
    SHADOW_ERR_INVALID_PARAM = -1,
    SHADOW_ERR_NO_MEMORY = -2,
    SHADOW_ERR_NOT_CONNECTED = -3,
    SHADOW_ERR_TIMEOUT = -4,
    SHADOW_ERR_CRC_MISMATCH = -5,
    SHADOW_ERR_QUEUE_FULL = -6,
    SHADOW_ERR_PLATFORM_ERROR = -7
} shadow_error_t;
```

#### Error Reporting
- Return error codes from all functions
- Provide detailed error information in logs
- Implement graceful degradation when possible

## Testing Framework

### Unit Tests
```
tests/
├── protocol_tests/
│   ├── test_message_parser.c
│   ├── test_crc16.c
│   └── test_message_queue.c
├── transport_tests/
│   ├── test_ack_handler.c
│   ├── test_retransmission.c
│   └── test_connection_manager.c
└── device_tests/
    ├── esp32/
    │   ├── test_ble_server.c
    │   └── test_sensor_interface.c
    └── macos/
        ├── test_ble_client.c
        └── test_ui_integration.c
```

### Integration Tests
```
tests/integration_tests/
├── test_end_to_end_communication.c
├── test_power_management.c
├── test_error_recovery.c
└── test_performance_benchmarks.c
```

## Documentation

### API Reference
- Detailed function descriptions with parameters and return values
- Example usage for each API function
- Error code explanations

### Implementation Guide
- Step-by-step integration instructions
- Platform-specific setup guides
- Configuration examples

### Protocol Specification
- Detailed message format specifications
- Command and status definitions
- Sequence diagrams for communication flows

## Build System

### CMake Configuration
```cmake
cmake_minimum_required(VERSION 3.10)
project(ShadowBLELibrary C)

set(CMAKE_C_STANDARD 99)

# Platform detection
if(ESP32)
    set(PLATFORM "ESP32")
    add_definitions(-DESP32)
elseif(MACOS)
    set(PLATFORM "MACOS")
    add_definitions(-DMACOS)
endif()

# Source files
set(SOURCES
    src/protocol/message_parser.c
    src/protocol/crc16.c
    src/protocol/message_queue.c
    src/transport/connection_manager.c
    src/transport/ack_handler.c
    src/transport/retransmission.c
    src/api/data_handler.c
    src/api/event_callbacks.c
    src/api/configuration.c
)

# Platform-specific sources
if(ESP32)
    list(APPEND SOURCES
        src/device/esp32/ble_server.c
        src/device/esp32/sensor_interface.c
        src/device/esp32/power_management.c
    )
elseif(MACOS)
    list(APPEND SOURCES
        src/device/macos/ble_client.c
        src/device/macos/ui_integration.c
        src/device/macos/background_tasks.c
    )
endif()

add_library(shadow_ble ${SOURCES})
target_include_directories(shadow_ble PUBLIC include)
```

### PlatformIO Configuration (ESP32)
```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
lib_deps = 
    nimble-arduino
    Wire
    MAX30105
build_flags = 
    -D ESP32
    -I include
```

### Swift Package Manager (macOS)
```swift
// Package.swift
// swift-tools-version:5.3
import PackageDescription

let package = Package(
    name: "ShadowBLELibrary",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        .library(
            name: "ShadowBLELibrary",
            targets: ["ShadowBLELibrary"])
    ],
    targets: [
        .target(
            name: "ShadowBLELibrary",
            dependencies: [],
            path: "src"),
        .testTarget(
            name: "ShadowBLELibraryTests",
            dependencies: ["ShadowBLELibrary"],
            path: "tests")
    ]
)
```

## Versioning and Release Management

### Version Scheme
- Semantic Versioning (MAJOR.MINOR.PATCH)
- API breaking changes increment MAJOR version
- Backward compatible additions increment MINOR version
- Bug fixes increment PATCH version

### Release Process
1. Update version number in all relevant files
2. Run complete test suite
3. Generate documentation
4. Create release tag
5. Publish to package repositories

## Conclusion

This library structure provides a robust, cross-platform framework for implementing reliable BLE communication in the Shadow wellness platform. By separating concerns into distinct layers and providing clear interfaces, the library enables consistent implementation across different platforms while maintaining flexibility for platform-specific optimizations. The modular design allows for easy testing, maintenance, and future enhancements.