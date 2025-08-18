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

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/protocol/message_parser.c
    src/transport/connection_manager.c
    src/api/data_handler.c
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

# Create library
add_library(shadow_ble ${SOURCES})

# Link libraries (platform-specific)
if(ESP32)
    # ESP32 specific libraries would be linked here
    # target_link_libraries(shadow_ble PRIVATE ...)
elseif(MACOS)
    # macOS specific libraries would be linked here
    # target_link_libraries(shadow_ble PRIVATE ...)
endif()

# Installation
install(TARGETS shadow_ble
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)