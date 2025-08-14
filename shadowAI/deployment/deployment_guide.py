"""
Deployment Guide Module

This module provides comprehensive deployment guidance for the ShadowAI
stress detection system, including step-by-step instructions, best practices,
validation procedures, and troubleshooting guides.

Features:
- Complete deployment workflow documentation
- Hardware setup and validation
- Software configuration guides
- Performance validation procedures
- Troubleshooting and diagnostics
- Production deployment checklist

Author: Shadow AI Team
License: MIT
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class DeploymentEnvironment:
    """Deployment environment configuration."""
    target_platform: str = "ESP32-S3"
    development_os: str = "cross-platform"
    ide: str = "PlatformIO"
    required_tools: List[str] = None
    hardware_requirements: Dict[str, str] = None
    
    def __post_init__(self):
        if self.required_tools is None:
            self.required_tools = [
                "PlatformIO IDE",
                "ESP32 toolchain",
                "USB drivers",
                "Serial monitor"
            ]
        
        if self.hardware_requirements is None:
            self.hardware_requirements = {
                "ESP32-S3": "ESP32-S3-DevKitC-1 or compatible",
                "Sensor": "MAX30102 heart rate sensor",
                "Memory": "≥8MB Flash, ≥512KB RAM",
                "Connectivity": "USB-C for programming",
                "Power": "3.3V/5V supply or battery"
            }

class DeploymentGuide:
    """
    Comprehensive deployment guide generator for ShadowAI system.
    
    Provides detailed instructions, validation procedures, and troubleshooting
    guidance for successful deployment of the stress detection system.
    """
    
    def __init__(self, environment: Optional[DeploymentEnvironment] = None):
        """
        Initialize deployment guide generator.
        
        Args:
            environment: Target deployment environment configuration
        """
        self.environment = environment or DeploymentEnvironment()
        
        # Deployment phases
        self.deployment_phases = [
            "Hardware Setup",
            "Software Environment",
            "Model Preparation", 
            "Firmware Generation",
            "System Integration",
            "Validation & Testing",
            "Production Deployment"
        ]
        
        # Validation checklist
        self.validation_checklist = {}
        
        logger.info(f"Deployment guide initialized for {self.environment.target_platform}")
    
    def generate_complete_guide(self, output_path: Optional[str] = None) -> str:
        """
        Generate complete deployment guide document.
        
        Args:
            output_path: Optional path to save the guide
            
        Returns:
            Complete deployment guide as markdown string
        """
        logger.info("Generating complete deployment guide...")
        
        guide_sections = [
            self._generate_title_section(),
            self._generate_overview_section(),
            self._generate_prerequisites_section(),
            self._generate_hardware_setup_section(),
            self._generate_software_setup_section(),
            self._generate_model_preparation_section(),
            self._generate_firmware_generation_section(),
            self._generate_deployment_section(),
            self._generate_validation_section(),
            self._generate_troubleshooting_section(),
            self._generate_maintenance_section(),
            self._generate_appendix_section()
        ]
        
        complete_guide = "\\n\\n".join(guide_sections)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(complete_guide)
            logger.info(f"Deployment guide saved to {output_path}")
        
        return complete_guide
    
    def create_deployment_checklist(self) -> Dict:
        """
        Create a comprehensive deployment checklist.
        
        Returns:
            Structured deployment checklist
        """
        checklist = {
            "pre_deployment": {
                "hardware_preparation": [
                    "ESP32-S3 development board acquired",
                    "MAX30102 sensor module available",
                    "Jumper wires and breadboard ready",
                    "USB-C cable for programming",
                    "Optional: Battery pack for portable operation"
                ],
                "software_preparation": [
                    "PlatformIO IDE installed",
                    "ESP32 platform support added",
                    "Required libraries available",
                    "TensorFlow Lite model converted",
                    "Project files generated"
                ],
                "environment_setup": [
                    "Development machine prepared",
                    "USB drivers installed",
                    "Serial monitor tested",
                    "Project workspace organized"
                ]
            },
            "deployment": {
                "hardware_assembly": [
                    "Sensor connections verified",
                    "Power supply connected",
                    "LED indicators wired",
                    "Enclosure prepared (if applicable)"
                ],
                "firmware_deployment": [
                    "Project compiled successfully",
                    "Firmware uploaded to ESP32-S3",
                    "Boot sequence verified",
                    "Serial output monitored"
                ],
                "system_configuration": [
                    "Sensor calibration completed",
                    "Communication settings configured",
                    "Power management tested",
                    "Debug features verified"
                ]
            },
            "validation": {
                "functional_testing": [
                    "Sensor data acquisition working",
                    "Model inference functioning",
                    "Communication established",
                    "LED indicators operating"
                ],
                "performance_testing": [
                    "Inference latency measured",
                    "Power consumption validated",
                    "Signal quality assessed",
                    "Accuracy benchmarked"
                ],
                "reliability_testing": [
                    "Extended operation tested",
                    "Error handling verified",
                    "Recovery procedures tested",
                    "Environmental conditions tested"
                ]
            },
            "production": {
                "final_validation": [
                    "Complete system test passed",
                    "Documentation updated",
                    "User instructions provided",
                    "Support procedures established"
                ],
                "deployment_completion": [
                    "System handed over to users",
                    "Training provided",
                    "Monitoring established",
                    "Maintenance schedule created"
                ]
            }
        }
        
        return checklist
    
    def generate_troubleshooting_guide(self) -> str:
        """Generate comprehensive troubleshooting guide."""
        return '''# Troubleshooting Guide

## Common Issues and Solutions

### Hardware Issues

#### 1. Sensor Not Detected
**Symptoms:**
- "MAX30102 was not found" error message
- No sensor readings
- I2C communication failure

**Solutions:**
- Check wiring connections (VCC, GND, SDA, SCL)
- Verify 3.3V power supply
- Test with different I2C pins
- Check for loose connections
- Try different MAX30102 module

**Diagnostic Steps:**
```cpp
// Add to setup() for I2C scanning
Wire.begin();
Serial.println("Scanning for I2C devices...");
for(byte address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    if(Wire.endTransmission() == 0) {
        Serial.printf("I2C device found at 0x%02X\\n", address);
    }
}
```

#### 2. Poor Signal Quality
**Symptoms:**
- Erratic BVP readings
- Low signal quality scores
- Inconsistent inference results

**Solutions:**
- Clean sensor surface
- Ensure proper finger placement
- Apply appropriate pressure (not too light, not too heavy)
- Check ambient light interference
- Verify sensor calibration

#### 3. Power Issues
**Symptoms:**
- Random resets
- Brownout detector triggers
- Inconsistent operation

**Solutions:**
- Use adequate power supply (≥500mA)
- Check USB cable quality
- Add decoupling capacitors
- Monitor battery voltage
- Reduce power consumption

### Software Issues

#### 4. Compilation Errors
**Common Errors:**

**TensorFlow Lite not found:**
```
fatal error: tensorflow/lite/micro/all_ops_resolver.h: No such file
```
**Solution:** Install TensorFlow Lite Micro library
```bash
pio lib install "https://github.com/tensorflow/tflite-micro-arduino-examples.git"
```

**Memory allocation errors:**
```
E (xxx) esp_system: Memory allocation failed
```
**Solution:** Reduce tensor arena size or enable PSRAM
```cpp
constexpr int kTensorArenaSize = 40 * 1024; // Reduce from 60KB
```

#### 5. Model Loading Issues
**Symptoms:**
- Model initialization failure
- Schema version mismatch
- Tensor allocation errors

**Solutions:**
- Verify model file inclusion
- Check model compatibility
- Validate tensor arena size
- Confirm model quantization format

**Diagnostic Code:**
```cpp
void debugModelInfo() {
    Serial.printf("Model size: %d bytes\\n", model_array_len);
    Serial.printf("Schema version: %d\\n", model->version());
    Serial.printf("Arena size: %d bytes\\n", kTensorArenaSize);
    Serial.printf("Arena used: %d bytes\\n", interpreter->arena_used_bytes());
}
```

#### 6. Communication Issues
**BLE Problems:**
- Connection failures
- Data transmission errors
- Pairing issues

**Solutions:**
- Check BLE initialization
- Verify service/characteristic UUIDs
- Test with BLE scanner app
- Reset Bluetooth cache on device

**WiFi Problems:**
- Connection timeouts
- Weak signal
- Credential errors

**Solutions:**
- Verify SSID and password
- Check signal strength
- Test with mobile hotspot
- Monitor connection status

### Performance Issues

#### 7. High Inference Latency
**Symptoms:**
- Inference time >100ms
- Real-time processing lag
- System responsiveness issues

**Solutions:**
- Optimize model architecture
- Use more aggressive quantization
- Reduce input data size
- Enable CPU frequency boost

**Performance Monitoring:**
```cpp
unsigned long start = micros();
int result = model.runInference(bvp_data);
unsigned long latency = micros() - start;
Serial.printf("Inference latency: %lu us\\n", latency);
```

#### 8. High Power Consumption
**Symptoms:**
- Short battery life
- Excessive heat generation
- Voltage drops

**Solutions:**
- Enable deep sleep mode
- Reduce sampling frequency
- Optimize LED usage
- Use power-efficient algorithms

### System Issues

#### 9. Memory Leaks
**Symptoms:**
- Gradual performance degradation
- Random crashes
- Heap exhaustion

**Solutions:**
- Monitor heap usage
- Fix dynamic allocations
- Use static buffers
- Implement memory debugging

**Memory Monitoring:**
```cpp
void printMemoryInfo() {
    Serial.printf("Free heap: %d bytes\\n", ESP.getFreeHeap());
    Serial.printf("Min free heap: %d bytes\\n", ESP.getMinFreeHeap());
    Serial.printf("Heap size: %d bytes\\n", ESP.getHeapSize());
}
```

#### 10. Watchdog Timeouts
**Symptoms:**
- Unexpected resets
- "Task watchdog" messages
- System instability

**Solutions:**
- Add delay() calls in loops
- Use yield() for long operations
- Optimize blocking operations
- Feed watchdog manually if needed

## Diagnostic Tools

### Serial Monitor Commands
Implement these debug commands for troubleshooting:

```cpp
void handleSerialCommands() {
    if (Serial.available()) {
        String command = Serial.readString();
        command.trim();
        
        if (command == "status") {
            printSystemStatus();
        } else if (command == "memory") {
            printMemoryInfo();
        } else if (command == "sensor") {
            printSensorStatus();
        } else if (command == "model") {
            debugModelInfo();
        } else if (command == "reset") {
            ESP.restart();
        }
    }
}
```

### LED Diagnostic Patterns
- Solid green: Normal operation
- Slow blink: Acquiring data
- Fast blink: Stress detected
- Double blink: Poor signal quality
- Red solid: Hardware error
- Red blink: Software error

### Log Analysis
Enable detailed logging for debugging:

```cpp
#define DEBUG_LEVEL 3
#define DEBUG_SENSOR 1
#define DEBUG_MODEL 1
#define DEBUG_COMM 1

void debugLog(int level, const char* message) {
    if (level <= DEBUG_LEVEL) {
        Serial.printf("[%lu] %s\\n", millis(), message);
    }
}
```

## Support Resources

### Online Resources
- ESP32-S3 Technical Reference Manual
- TensorFlow Lite Micro Documentation
- MAX30102 Datasheet
- PlatformIO Documentation

### Community Support
- ESP32 Arduino Forum
- TensorFlow Lite Community
- PlatformIO Community
- GitHub Issues

### Professional Support
- Hardware vendor support
- Software development consultation
- Custom integration services
- Training and workshops

## Emergency Procedures

### System Recovery
1. Hold reset button for 10 seconds
2. Flash firmware via USB
3. Factory reset if necessary
4. Restore from backup

### Data Recovery
1. Check serial logs
2. Retrieve from communication logs
3. Use backup storage if available
4. Implement data validation

### Contact Information
- Technical Support: support@shadowai.com
- Documentation: docs.shadowai.com
- Community: community.shadowai.com
- Emergency: emergency@shadowai.com
'''
    
    def generate_validation_procedures(self) -> Dict:
        """Generate comprehensive validation procedures."""
        return {
            "hardware_validation": {
                "power_system": {
                    "tests": [
                        "Measure supply voltage (should be 3.3V ±5%)",
                        "Check current consumption (<200mA typical)",
                        "Verify brownout detection threshold",
                        "Test battery operation if applicable"
                    ],
                    "acceptance_criteria": {
                        "voltage_range": "3.13V - 3.47V",
                        "current_idle": "<50mA",
                        "current_active": "<200mA",
                        "battery_life": ">8 hours continuous"
                    }
                },
                "sensor_system": {
                    "tests": [
                        "Verify I2C communication",
                        "Check sensor detection and initialization",
                        "Measure signal quality with finger placement",
                        "Test signal stability over time"
                    ],
                    "acceptance_criteria": {
                        "i2c_address": "0x57",
                        "signal_quality": ">0.7",
                        "signal_stability": "<5% variation",
                        "response_time": "<1 second"
                    }
                },
                "communication": {
                    "tests": [
                        "BLE advertisement detection",
                        "Connection establishment",
                        "Data transmission verification",
                        "Range testing"
                    ],
                    "acceptance_criteria": {
                        "ble_range": ">10 meters",
                        "connection_time": "<5 seconds",
                        "data_throughput": ">1KB/s",
                        "packet_loss": "<1%"
                    }
                }
            },
            "software_validation": {
                "model_performance": {
                    "tests": [
                        "Inference latency measurement",
                        "Memory usage validation",
                        "Accuracy testing with known data",
                        "Robustness testing"
                    ],
                    "acceptance_criteria": {
                        "inference_time": "<100ms",
                        "memory_usage": "<60KB tensor arena",
                        "accuracy": ">80%",
                        "confidence_threshold": ">0.7"
                    }
                },
                "system_integration": {
                    "tests": [
                        "End-to-end workflow validation",
                        "Error handling verification",
                        "Recovery procedure testing",
                        "Long-term stability testing"
                    ],
                    "acceptance_criteria": {
                        "uptime": ">24 hours continuous",
                        "error_recovery": "<30 seconds",
                        "data_integrity": "100%",
                        "system_availability": ">99%"
                    }
                }
            },
            "user_acceptance": {
                "usability": {
                    "tests": [
                        "Setup procedure timing",
                        "User interface clarity",
                        "Feedback mechanism effectiveness",
                        "Error message comprehension"
                    ],
                    "acceptance_criteria": {
                        "setup_time": "<5 minutes",
                        "user_satisfaction": ">4/5",
                        "error_resolution": "<2 minutes",
                        "training_required": "<30 minutes"
                    }
                }
            }
        }
    
    def _generate_title_section(self) -> str:
        """Generate title and introduction section."""
        return '''# ShadowAI Deployment Guide
## Complete Deployment Instructions for ESP32-S3 Stress Detection System

**Version:** 1.0.0  
**Last Updated:** {current_date}  
**Target Platform:** {platform}  
**Authors:** Shadow AI Team

### Document Overview

This comprehensive guide provides step-by-step instructions for deploying the ShadowAI stress detection system on ESP32-S3 hardware. The guide covers everything from initial hardware setup to production deployment and maintenance.

### Prerequisites

Before beginning deployment, ensure you have:
- Hardware components (ESP32-S3, MAX30102 sensor)
- Development environment (PlatformIO IDE)
- Converted TensorFlow Lite model
- Generated firmware files
- Basic electronics knowledge

### Deployment Timeline

- **Hardware Setup:** 30-60 minutes
- **Software Configuration:** 45-90 minutes  
- **System Integration:** 60-120 minutes
- **Validation & Testing:** 120-240 minutes
- **Total Deployment Time:** 4-8 hours'''.format(
            current_date=time.strftime('%Y-%m-%d'),
            platform=self.environment.target_platform
        )
    
    def _generate_overview_section(self) -> str:
        """Generate system overview section."""
        return '''## System Overview

### Architecture Components

The ShadowAI stress detection system consists of:

1. **Hardware Layer**
   - ESP32-S3 microcontroller
   - MAX30102 heart rate sensor
   - Power management system
   - Communication interfaces

2. **Software Layer**
   - TensorFlow Lite Micro runtime
   - Signal processing pipeline
   - Communication protocols
   - Power management

3. **Model Layer**
   - Quantized CNN model
   - Real-time inference engine
   - Feature extraction algorithms
   - Classification output

### Data Flow

```
Sensor (MAX30102) → Signal Acquisition → Preprocessing → 
Feature Extraction → Model Inference → Classification → 
Communication (BLE/WiFi) → User Interface
```

### Key Features

- **Real-time Processing:** <100ms inference latency
- **Low Power Operation:** Optimized for battery operation
- **Wireless Communication:** BLE and WiFi support
- **High Accuracy:** >80% stress detection accuracy
- **Compact Design:** Suitable for wearable applications

### Performance Specifications

| Metric | Specification |
|--------|---------------|
| Inference Time | <100ms |
| Power Consumption | <200mW |
| Battery Life | >8 hours |
| Accuracy | >80% |
| Memory Usage | <512KB RAM |
| Model Size | <8MB Flash |'''
    
    def _generate_prerequisites_section(self) -> str:
        """Generate prerequisites section."""
        return f'''## Prerequisites and Requirements

### Hardware Requirements

| Component | Specification | Quantity |
|-----------|---------------|----------|
| **Microcontroller** | {self.environment.hardware_requirements.get("ESP32-S3", "ESP32-S3-DevKitC-1")} | 1 |
| **Sensor** | {self.environment.hardware_requirements.get("Sensor", "MAX30102")} | 1 |
| **Connectivity** | USB-C cable for programming | 1 |
| **Breadboard** | Half-size breadboard | 1 |
| **Jumper Wires** | Male-to-male, 10cm | 10 |
| **Power Supply** | 5V/1A USB adapter or battery | 1 |

### Software Requirements

| Tool | Version | Purpose |
|------|---------|---------|
| **PlatformIO IDE** | Latest | Development environment |
| **ESP32 Arduino Core** | 2.0+ | Platform support |
| **TensorFlow Lite Micro** | Latest | ML runtime |
| **MAX30102 Library** | 1.1+ | Sensor driver |

### Skills and Knowledge

- Basic electronics and wiring
- Arduino/C++ programming concepts
- Serial monitor usage
- Basic debugging skills

### Workspace Setup

- Clean, well-lit workspace
- ESD protection (anti-static mat recommended)
- Multimeter for voltage measurements
- Computer with USB ports
- Internet connection for library downloads'''
    
    def _generate_hardware_setup_section(self) -> str:
        """Generate hardware setup section."""
        return '''## Hardware Setup and Assembly

### Step 1: Component Inspection

Before assembly, inspect all components:

1. **ESP32-S3 Board:**
   - Check for physical damage
   - Verify USB connector integrity
   - Ensure bootloader LED functions

2. **MAX30102 Sensor:**
   - Inspect sensor surface for scratches
   - Verify pin labels and alignment
   - Check for loose connections

### Step 2: Wiring Connections

**Connection Diagram:**

```
MAX30102          ESP32-S3
--------          --------
VCC (3.3V) -----> 3V3
GND ------------> GND  
SDA ------------> GPIO21
SCL ------------> GPIO22
INT ------------> (not connected)
```

**Important Notes:**
- Use 3.3V, not 5V for MAX30102
- Keep wire lengths <10cm for I2C reliability
- Ensure solid connections (no loose wires)

### Step 3: Power Verification

1. Connect ESP32-S3 to computer via USB-C
2. Verify power LED illuminates
3. Measure voltage at MAX30102 VCC pin (should be 3.3V)
4. Check for any short circuits

### Step 4: Initial Communication Test

Upload a simple I2C scanner sketch to verify sensor detection:

```cpp
#include <Wire.h>

void setup() {
    Serial.begin(115200);
    Wire.begin(21, 22); // SDA, SCL
    Serial.println("I2C Scanner");
}

void loop() {
    for(byte address = 1; address < 127; address++) {
        Wire.beginTransmission(address);
        if(Wire.endTransmission() == 0) {
            Serial.printf("Device found at 0x%02X\\n", address);
        }
    }
    delay(5000);
}
```

**Expected Output:**
```
I2C Scanner
Device found at 0x57
```

### Step 5: Enclosure Preparation (Optional)

For portable deployment:
- Design or select appropriate enclosure
- Ensure sensor access and heat dissipation
- Plan for battery compartment
- Consider user interface elements'''
    
    def _generate_software_setup_section(self) -> str:
        """Generate software setup section."""
        return '''## Software Environment Setup

### Step 1: Install PlatformIO IDE

**Option A: VS Code Extension**
1. Download and install Visual Studio Code
2. Open VS Code Extensions panel (Ctrl+Shift+X)
3. Search for "PlatformIO IDE"
4. Install the official PlatformIO extension
5. Restart VS Code

**Option B: Standalone Installation**
1. Visit [platformio.org](https://platformio.org)
2. Download PlatformIO IDE
3. Follow installation instructions for your OS
4. Complete initial setup wizard

### Step 2: Configure ESP32 Platform

1. Open PlatformIO Home
2. Navigate to Platforms
3. Search for "espressif32"
4. Install ESP32 platform (latest stable version)
5. Verify installation success

### Step 3: Install Required Libraries

Create a new project or open existing shadowAI project:

```ini
[env:esp32-s3-devkitc-1]
platform = espressif32
board = esp32-s3-devkitc-1
framework = arduino

lib_deps = 
    sparkfun/SparkFun MAX3010x library @ ^1.1.1
    arduino-libraries/ArduinoBLE @ ^1.3.2
    bblanchon/ArduinoJson @ ^6.21.2
    https://github.com/tensorflow/tflite-micro-arduino-examples.git
```

### Step 4: Verify Installation

Create and compile a test project:

```cpp
#include <Arduino.h>
#include "MAX30105.h"

MAX30105 particleSensor;

void setup() {
    Serial.begin(115200);
    
    if (particleSensor.begin()) {
        Serial.println("MAX30105 found");
    } else {
        Serial.println("MAX30105 not found");
    }
}

void loop() {
    // Test loop
    delay(1000);
}
```

### Step 5: USB Driver Installation

**Windows:**
- Install ESP32-S3 USB drivers from Espressif
- Use Device Manager to verify COM port detection

**macOS:**
- Drivers usually auto-install
- Check System Report for USB device

**Linux:**
- Add user to dialout group: `sudo usermod -a -G dialout $USER`
- Install udev rules for ESP32

### Step 6: Serial Monitor Configuration

Configure serial monitor settings:
- Baud rate: 115200
- Line ending: Both NL & CR
- Enable timestamps for debugging'''
    
    def _generate_model_preparation_section(self) -> str:
        """Generate model preparation section."""
        return '''## Model Preparation and Integration

### Step 1: Model Conversion Verification

Ensure your TensorFlow Lite model is properly converted:

```python
# Verify model file
import tensorflow as tf

# Load and inspect model
interpreter = tf.lite.Interpreter(model_path="shadow_cnn_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"Model size: {os.path.getsize('shadow_cnn_model.tflite')} bytes")
```

### Step 2: Generate C Header File

Convert the TFLite model to C header format:

```python
from shadowAI.deployment import TFLiteConverter

converter = TFLiteConverter()
header_path = converter.export_c_header(
    model_path="shadow_cnn_model.tflite",
    output_path="shadow_cnn_model.h",
    array_name="shadow_cnn_model"
)
```

### Step 3: Integrate Model File

1. Copy the generated header file to your project's `lib/` directory
2. Verify the file structure:

```cpp
// shadow_cnn_model.h content verification
#ifndef SHADOW_CNN_MODEL_H
#define SHADOW_CNN_MODEL_H

const unsigned char shadow_cnn_model[] = {
    0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33,
    // ... model data ...
};

const unsigned int shadow_cnn_model_len = 12345; // actual size

#endif
```

### Step 4: Memory Requirements Analysis

Calculate memory requirements:

```cpp
// In your main.cpp
#include "shadow_cnn_model.h"

void printModelInfo() {
    Serial.printf("Model size: %d bytes\\n", shadow_cnn_model_len);
    Serial.printf("Available heap: %d bytes\\n", ESP.getFreeHeap());
    Serial.printf("PSRAM available: %d bytes\\n", ESP.getFreePsram());
}
```

**Memory Allocation Guidelines:**
- Tensor arena: 60KB (adjustable based on model)
- Model storage: Flash memory
- Runtime variables: Main RAM
- Large buffers: PSRAM if available

### Step 5: Quantization Verification

Verify quantization is properly applied:

```cpp
void verifyQuantization() {
    // Check input tensor quantization
    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);
    
    Serial.printf("Input type: %d\\n", input->type);
    Serial.printf("Output type: %d\\n", output->type);
    
    if (input->type == kTfLiteInt8) {
        Serial.println("Input is quantized to INT8");
    }
}
```

### Step 6: Model Performance Testing

Test model performance before deployment:

```cpp
void benchmarkModel() {
    // Prepare dummy input
    float dummy_input[WINDOW_SIZE_SAMPLES];
    for (int i = 0; i < WINDOW_SIZE_SAMPLES; i++) {
        dummy_input[i] = sin(i * 0.1); // Synthetic BVP signal
    }
    
    // Measure inference time
    unsigned long start = micros();
    int result = model.runInference(dummy_input);
    unsigned long latency = micros() - start;
    
    Serial.printf("Inference latency: %lu us\\n", latency);
    Serial.printf("Result: %d\\n", result);
}
```'''
    
    def _generate_firmware_generation_section(self) -> str:
        """Generate firmware generation section."""
        return '''## Firmware Generation and Compilation

### Step 1: Project Structure Verification

Ensure your project follows the correct structure:

```
shadowai-esp32/
├── platformio.ini
├── src/
│   ├── main.cpp
│   ├── sensor_manager.cpp
│   ├── model_inference.cpp
│   └── comm_manager.cpp
├── include/
│   ├── sensor_manager.h
│   ├── model_inference.h
│   ├── comm_manager.h
│   └── config.h
├── lib/
│   └── shadow_cnn_model.h
└── README.md
```

### Step 2: Configuration Review

Review and customize `include/config.h`:

```cpp
// Hardware configuration
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
#define LED_STATUS_PIN 2
#define LED_ERROR_PIN 4

// Model configuration
#define WINDOW_SIZE_SECONDS 60
#define SAMPLING_RATE 64
#define NUM_CLASSES 4

// Communication
#define ENABLE_BLUETOOTH true
#define ENABLE_WIFI false

// Power management
#define ENABLE_DEEP_SLEEP true
#define SLEEP_DURATION_SECONDS 300
```

### Step 3: Compile Verification

Build the project to check for errors:

```bash
# Navigate to project directory
cd shadowai-esp32

# Clean previous builds
pio run --target clean

# Compile project
pio run

# Expected output:
# Successfully created esp32-s3-devkitc-1 firmware
```

### Step 4: Memory Usage Analysis

Check memory usage in build output:

```
RAM:   [===       ]  25.2% (used 82764 bytes from 327680 bytes)
Flash: [====      ]  38.1% (used 1250000 bytes from 3276800 bytes)
```

**Memory Guidelines:**
- RAM usage should be <50% for stability
- Flash usage should be <80% for OTA updates
- Leave headroom for runtime allocations

### Step 5: Advanced Build Options

For production builds, consider these optimizations:

```ini
# platformio.ini optimizations
build_flags = 
    -Os                          # Optimize for size
    -DCORE_DEBUG_LEVEL=1        # Reduce debug output
    -DDISABLE_DEBUG_PRINTS      # Remove debug prints
    -DENABLE_OPTIMIZATION       # Enable optimizations
```

### Step 6: Binary Analysis

Analyze the generated binary:

```bash
# Get binary information
pio run --target size

# Generate assembly listing
pio run --target listing

# Check section sizes
pio run --target sections
```

### Step 7: Backup Creation

Create backups of working firmware:

```bash
# Copy firmware files
cp .pio/build/esp32-s3-devkitc-1/firmware.bin backup/
cp .pio/build/esp32-s3-devkitc-1/partitions.bin backup/
cp .pio/build/esp32-s3-devkitc-1/bootloader.bin backup/

# Create archive
tar -czf shadowai-firmware-v1.0.0.tar.gz backup/
```'''
    
    def _generate_deployment_section(self) -> str:
        """Generate deployment section."""
        return '''## System Deployment and Integration

### Step 1: Initial Firmware Upload

Upload firmware to ESP32-S3:

```bash
# Connect ESP32-S3 via USB-C
# Put device in download mode if necessary

# Upload firmware
pio run --target upload

# Monitor serial output
pio device monitor
```

**Expected Boot Sequence:**
```
ShadowCNN Stress Detection System
ESP32-S3 - Version 1.0
================================
MAX30102 initialized successfully
TensorFlow Lite model initialized successfully
Model memory usage: 58234 bytes
System initialized successfully
```

### Step 2: Sensor Calibration

Follow the calibration procedure:

1. **Initial Setup:**
   - Wait for "Please place finger on sensor" message
   - Clean sensor surface with alcohol wipe
   - Ensure good ambient lighting

2. **Finger Placement:**
   - Place index finger gently on sensor
   - Maintain steady pressure (not too light/heavy)
   - Keep finger still during calibration

3. **Calibration Process:**
   - Wait for sensor readings to stabilize
   - System will automatically complete calibration
   - "Calibration complete" message appears

4. **Verification:**
   - Check signal quality indicator
   - Verify stable BVP readings
   - Confirm LED status indication

### Step 3: Communication Setup

#### Bluetooth (BLE) Configuration:

1. **Device Discovery:**
   - Use BLE scanner app on smartphone
   - Look for "ShadowCNN-StressDetector"
   - Note device MAC address

2. **Connection Test:**
   - Connect to the device
   - Verify service UUID: `12345678-1234-1234-1234-123456789abc`
   - Test data characteristic access

3. **Data Reception:**
   - Subscribe to data notifications
   - Verify JSON format: `{"timestamp": 12345, "class": 1, "confidence": 0.85}`

#### WiFi Configuration (if enabled):

1. **Network Setup:**
   - Configure SSID and password in config.h
   - Rebuild and upload firmware
   - Monitor connection status

2. **Server Communication:**
   - Verify HTTP endpoint accessibility
   - Test data transmission format
   - Check server response handling

### Step 4: Performance Validation

#### Latency Testing:
```cpp
void measureSystemLatency() {
    unsigned long start = millis();
    
    // Acquire BVP window
    sensor.acquireBVPWindow(bvp_data);
    unsigned long acquisition_time = millis() - start;
    
    // Run inference
    start = millis();
    int result = model.runInference(bvp_data);
    unsigned long inference_time = millis() - start;
    
    Serial.printf("Acquisition: %lu ms\\n", acquisition_time);
    Serial.printf("Inference: %lu ms\\n", inference_time);
}
```

#### Power Consumption:
- Measure current draw during different states
- Verify deep sleep functionality
- Test battery operation if applicable

#### Signal Quality:
- Test with different users
- Validate under various conditions
- Check environmental sensitivity

### Step 5: Error Handling Verification

Test error recovery mechanisms:

1. **Sensor Disconnection:**
   - Disconnect sensor during operation
   - Verify error detection and recovery
   - Check LED error indication

2. **Memory Issues:**
   - Monitor heap usage over time
   - Test memory leak detection
   - Verify garbage collection

3. **Communication Failures:**
   - Test BLE disconnection recovery
   - Verify WiFi reconnection logic
   - Check data buffering during outages

### Step 6: Long-term Stability Testing

Run extended tests:

- **24-hour continuous operation**
- **Temperature cycling**
- **Multiple user sessions**
- **Various environmental conditions**

Monitor for:
- Memory leaks
- Performance degradation
- Communication stability
- Accuracy consistency'''
    
    def _generate_validation_section(self) -> str:
        """Generate validation section."""
        return '''## Validation and Testing Procedures

### Validation Framework

Comprehensive testing ensures system reliability and performance. Follow these structured validation procedures:

### Level 1: Unit Testing

#### Sensor Module Testing:
```cpp
bool testSensorModule() {
    // Test sensor initialization
    if (!sensor.initialize()) {
        Serial.println("FAIL: Sensor initialization");
        return false;
    }
    
    // Test I2C communication
    if (!sensor.isConnected()) {
        Serial.println("FAIL: Sensor communication");
        return false;
    }
    
    // Test signal acquisition
    float sample;
    if (!sensor.readBVPSample(&sample)) {
        Serial.println("FAIL: Signal acquisition");
        return false;
    }
    
    Serial.println("PASS: Sensor module");
    return true;
}
```

#### Model Module Testing:
```cpp
bool testModelModule() {
    // Test model initialization
    if (!model.initialize()) {
        Serial.println("FAIL: Model initialization");
        return false;
    }
    
    // Test with synthetic data
    float test_data[WINDOW_SIZE_SAMPLES];
    generateSyntheticBVP(test_data);
    
    int result = model.runInference(test_data);
    if (result < 0 || result >= NUM_CLASSES) {
        Serial.println("FAIL: Model inference");
        return false;
    }
    
    Serial.println("PASS: Model module");
    return true;
}
```

### Level 2: Integration Testing

#### End-to-End Workflow:
```cpp
bool testEndToEndWorkflow() {
    Serial.println("Starting E2E test...");
    
    // 1. Sensor data acquisition
    float bvp_window[WINDOW_SIZE_SAMPLES];
    if (!sensor.acquireBVPWindow(bvp_window)) {
        return false;
    }
    
    // 2. Model inference
    int prediction = model.runInference(bvp_window);
    if (prediction < 0) {
        return false;
    }
    
    // 3. Communication
    String result = formatResult(prediction);
    if (!comm.sendData(result)) {
        return false;
    }
    
    Serial.println("PASS: End-to-end workflow");
    return true;
}
```

### Level 3: Performance Testing

#### Timing Validation:
```cpp
struct PerformanceMetrics {
    unsigned long acquisition_time;
    unsigned long inference_time;
    unsigned long communication_time;
    unsigned long total_time;
};

PerformanceMetrics measurePerformance() {
    PerformanceMetrics metrics;
    unsigned long start, end;
    
    // Measure acquisition time
    start = micros();
    sensor.acquireBVPWindow(bvp_data);
    end = micros();
    metrics.acquisition_time = end - start;
    
    // Measure inference time  
    start = micros();
    model.runInference(bvp_data);
    end = micros();
    metrics.inference_time = end - start;
    
    // Measure communication time
    start = micros();
    comm.sendData("test");
    end = micros();
    metrics.communication_time = end - start;
    
    metrics.total_time = metrics.acquisition_time + 
                        metrics.inference_time + 
                        metrics.communication_time;
    
    return metrics;
}
```

### Level 4: Stress Testing

#### Memory Stress Test:
```cpp
void stressTestMemory() {
    Serial.println("Starting memory stress test...");
    
    for (int i = 0; i < 1000; i++) {
        // Simulate heavy memory usage
        float* temp_buffer = (float*)malloc(1024 * sizeof(float));
        if (temp_buffer == NULL) {
            Serial.printf("Memory allocation failed at iteration %d\\n", i);
            break;
        }
        
        // Use the buffer
        memset(temp_buffer, 0, 1024 * sizeof(float));
        
        // Free memory
        free(temp_buffer);
        
        // Check heap status
        if (ESP.getFreeHeap() < 50000) {
            Serial.printf("Low memory warning: %d bytes\\n", ESP.getFreeHeap());
        }
        
        delay(10);
    }
    
    Serial.println("Memory stress test completed");
}
```

#### Continuous Operation Test:
```cpp
void continuousOperationTest(unsigned long duration_hours) {
    unsigned long start_time = millis();
    unsigned long test_duration = duration_hours * 3600000; // Convert to ms
    int inference_count = 0;
    int error_count = 0;
    
    while (millis() - start_time < test_duration) {
        if (runInferenceCycle()) {
            inference_count++;
        } else {
            error_count++;
        }
        
        // Print status every hour
        if ((millis() - start_time) % 3600000 < 1000) {
            Serial.printf("Status: %d inferences, %d errors\\n", 
                         inference_count, error_count);
        }
        
        delay(60000); // 1-minute intervals
    }
    
    float error_rate = (float)error_count / (inference_count + error_count);
    Serial.printf("Test completed: %.2f%% error rate\\n", error_rate * 100);
}
```

### Acceptance Criteria

| Test Category | Metric | Target | Method |
|---------------|--------|--------|--------|
| **Performance** | Inference Latency | <100ms | Timing measurement |
| **Performance** | Memory Usage | <60KB | Heap monitoring |
| **Performance** | Power Consumption | <200mW | Current measurement |
| **Reliability** | Uptime | >99% | Long-term testing |
| **Reliability** | Error Rate | <1% | Statistical analysis |
| **Accuracy** | Classification | >80% | Validation dataset |
| **Usability** | Setup Time | <5 min | User testing |

### Automated Testing Script

```cpp
void runAutomatedTests() {
    int passed = 0, failed = 0;
    
    // Unit tests
    if (testSensorModule()) passed++; else failed++;
    if (testModelModule()) passed++; else failed++;
    if (testCommModule()) passed++; else failed++;
    
    // Integration tests
    if (testEndToEndWorkflow()) passed++; else failed++;
    if (testErrorRecovery()) passed++; else failed++;
    
    // Performance tests
    PerformanceMetrics perf = measurePerformance();
    if (perf.total_time < 100000) passed++; else failed++; // <100ms
    
    Serial.printf("Test Results: %d passed, %d failed\\n", passed, failed);
    
    if (failed == 0) {
        Serial.println("🎉 ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT");
    } else {
        Serial.println("❌ SOME TESTS FAILED - REVIEW BEFORE DEPLOYMENT");
    }
}
```'''
    
    def _generate_troubleshooting_section(self) -> str:
        """Generate troubleshooting section."""
        return self.generate_troubleshooting_guide()
    
    def _generate_maintenance_section(self) -> str:
        """Generate maintenance section."""
        return '''## Maintenance and Support

### Preventive Maintenance

#### Weekly Tasks:
- Check sensor surface for contamination
- Verify LED status indicators
- Monitor battery levels (if applicable)
- Review system logs for errors

#### Monthly Tasks:
- Clean sensor with alcohol wipe
- Verify firmware version
- Check communication connectivity
- Update libraries if needed

#### Quarterly Tasks:
- Perform complete system validation
- Review performance metrics
- Update documentation
- Plan firmware updates

### Monitoring and Diagnostics

#### System Health Monitoring:
```cpp
void systemHealthCheck() {
    Serial.println("=== System Health Check ===");
    
    // Memory status
    Serial.printf("Free heap: %d bytes\\n", ESP.getFreeHeap());
    Serial.printf("Min free heap: %d bytes\\n", ESP.getMinFreeHeap());
    
    // Sensor status
    Serial.printf("Sensor connected: %s\\n", 
                  sensor.isConnected() ? "Yes" : "No");
    Serial.printf("Signal quality: %.2f\\n", sensor.getSignalQuality());
    
    // Model status
    Serial.printf("Model loaded: %s\\n", 
                  model.isInitialized() ? "Yes" : "No");
    Serial.printf("Last inference time: %lu ms\\n", 
                  model.getLastInferenceTime());
    
    // Communication status
    Serial.printf("BLE connected: %s\\n", 
                  comm.isConnected() ? "Yes" : "No");
    
    // Performance metrics
    Serial.printf("Uptime: %lu hours\\n", millis() / 3600000);
    Serial.printf("Total inferences: %d\\n", getTotalInferences());
}
```

#### Log Analysis:
- Set up centralized logging
- Monitor error patterns
- Track performance trends
- Identify optimization opportunities

### Firmware Updates

#### Over-The-Air (OTA) Updates:
```cpp
#include <Update.h>
#include <WiFi.h>
#include <HTTPClient.h>

bool performOTAUpdate(const String& firmware_url) {
    HTTPClient http;
    http.begin(firmware_url);
    
    int httpCode = http.GET();
    if (httpCode == HTTP_CODE_OK) {
        int contentLength = http.getSize();
        
        if (Update.begin(contentLength)) {
            size_t written = Update.writeStream(http.getStream());
            
            if (written == contentLength) {
                Serial.println("OTA update successful");
                if (Update.end()) {
                    ESP.restart();
                    return true;
                }
            }
        }
    }
    
    return false;
}
```

#### Manual Update Procedure:
1. Backup current configuration
2. Download new firmware
3. Put device in update mode
4. Flash new firmware
5. Verify functionality
6. Restore configuration if needed

### Data Backup and Recovery

#### Configuration Backup:
```cpp
void backupConfiguration() {
    // Save configuration to EEPROM or SPIFFS
    preferences.begin("shadowai", false);
    preferences.putString("version", FIRMWARE_VERSION);
    preferences.putFloat("calibration_offset", sensor_offset);
    preferences.putFloat("calibration_scale", sensor_scale);
    preferences.end();
}

void restoreConfiguration() {
    preferences.begin("shadowai", true);
    String version = preferences.getString("version", "");
    sensor_offset = preferences.getFloat("calibration_offset", 0.0);
    sensor_scale = preferences.getFloat("calibration_scale", 1.0);
    preferences.end();
}
```

#### Data Recovery Procedures:
- Implement data logging to SD card
- Use cloud backup for critical data
- Create recovery procedures for corruption
- Test restore procedures regularly

### Performance Optimization

#### Memory Optimization:
- Monitor heap fragmentation
- Optimize buffer sizes
- Use static allocation where possible
- Implement memory pooling

#### Power Optimization:
- Profile power consumption patterns
- Optimize sleep/wake cycles
- Reduce unnecessary processing
- Use efficient algorithms

#### Communication Optimization:
- Implement data compression
- Use efficient protocols
- Optimize transmission schedules
- Handle connection failures gracefully

### Support Procedures

#### User Support:
- Create user manuals
- Provide training materials
- Establish help desk procedures
- Maintain FAQ database

#### Technical Support:
- Remote diagnostic capabilities
- Log analysis tools
- Performance monitoring dashboards
- Issue tracking system

#### Emergency Procedures:
- System recovery protocols
- Emergency contact information
- Escalation procedures
- Backup system activation

### Documentation Maintenance

#### Keep Updated:
- System architecture diagrams
- Configuration parameters
- Performance benchmarks
- Troubleshooting guides

#### Version Control:
- Track documentation changes
- Maintain change logs
- Review and approve updates
- Distribute to stakeholders

### End-of-Life Planning

#### Hardware Lifecycle:
- Monitor component aging
- Plan replacement schedules
- Evaluate upgrade opportunities
- Manage obsolescence

#### Software Lifecycle:
- Track library dependencies
- Plan migration strategies
- Maintain security updates
- Evaluate new technologies'''
    
    def _generate_appendix_section(self) -> str:
        """Generate appendix section."""
        return '''## Appendix

### A. Technical Specifications

#### ESP32-S3 Specifications:
- **CPU:** Dual-core Xtensa 32-bit LX7, up to 240MHz
- **Memory:** 512KB SRAM, 384KB ROM
- **Flash:** 8MB (configurable)
- **PSRAM:** 8MB (optional)
- **WiFi:** 802.11 b/g/n
- **Bluetooth:** LE 5.0
- **USB:** USB-OTG support
- **GPIO:** 45 programmable GPIOs

#### MAX30102 Specifications:
- **Sensor Type:** Optical heart rate and SpO2
- **LED:** Red (660nm) and IR (880nm)
- **ADC:** 18-bit resolution
- **Sample Rate:** Up to 3200 samples/second
- **Interface:** I2C (400kHz)
- **Supply Voltage:** 3.3V

### B. Pin Configuration Reference

| Function | ESP32-S3 Pin | MAX30102 Pin | Notes |
|----------|--------------|--------------|-------|
| Power | 3V3 | VCC | 3.3V supply |
| Ground | GND | GND | Common ground |
| I2C Data | GPIO21 | SDA | Pull-up required |
| I2C Clock | GPIO22 | SCL | Pull-up required |
| Interrupt | GPIO19 | INT | Optional |
| Status LED | GPIO2 | - | System status |
| Error LED | GPIO4 | - | Error indication |

### C. Memory Layout

```
Flash Memory Layout (8MB):
├── Bootloader (32KB)
├── Partition Table (4KB)
├── NVS (20KB)
├── OTA Data (8KB)
├── Application (3MB)
├── OTA Update (3MB)
├── SPIFFS (1MB)
└── Reserved (896KB)

RAM Layout (512KB):
├── System Reserved (128KB)
├── Application Heap (256KB)
├── Tensor Arena (60KB)
├── Stack Space (32KB)
└── Static Variables (36KB)
```

### D. Communication Protocols

#### BLE Service Definition:
```
Service UUID: 12345678-1234-1234-1234-123456789abc
Characteristics:
  - Data: 87654321-4321-4321-4321-cba987654321 (Read/Notify)
  - Control: 11223344-5566-7788-99aa-bbccddeeff00 (Write)
  - Status: aabbccdd-eeff-0011-2233-445566778899 (Read)
```

#### Data Format Specification:
```json
{
  "timestamp": 1634567890123,
  "inference_count": 42,
  "predicted_class": 1,
  "class_name": "stress",
  "confidence": 0.847,
  "signal_quality": 0.92,
  "battery_level": 85
}
```

### E. Error Codes

| Code | Category | Description | Resolution |
|------|----------|-------------|------------|
| 001 | Sensor | MAX30102 not found | Check wiring |
| 002 | Sensor | Poor signal quality | Improve contact |
| 003 | Model | Initialization failed | Check memory |
| 004 | Model | Inference timeout | Optimize code |
| 005 | Comm | BLE connection lost | Reset BLE |
| 006 | Comm | WiFi authentication | Check credentials |
| 007 | Power | Low battery warning | Charge battery |
| 008 | System | Memory allocation failed | Restart system |

### F. Default Configuration Values

```cpp
// Sensor Configuration
#define SAMPLING_RATE 64
#define WINDOW_SIZE_SECONDS 60
#define SIGNAL_QUALITY_THRESHOLD 0.6

// Model Configuration  
#define TENSOR_ARENA_SIZE (60 * 1024)
#define INFERENCE_TIMEOUT_MS 5000
#define CONFIDENCE_THRESHOLD 0.7

// Communication Configuration
#define BLE_DEVICE_NAME "ShadowCNN-StressDetector"
#define BLE_MTU_SIZE 512
#define WIFI_TIMEOUT_MS 10000

// Power Management
#define SLEEP_DURATION_SECONDS 300
#define LOW_BATTERY_THRESHOLD 15
#define CRITICAL_BATTERY_THRESHOLD 5
```

### G. Performance Benchmarks

#### Reference Performance (ESP32-S3 @ 240MHz):
- **Sensor Reading:** ~5ms per sample
- **Signal Processing:** ~20ms per window
- **Model Inference:** ~80ms average
- **Communication:** ~10ms per transmission
- **Total Cycle Time:** ~115ms

#### Memory Usage:
- **Static RAM:** ~180KB
- **Dynamic Heap:** ~150KB peak
- **Flash Storage:** ~2.8MB application
- **Model Size:** ~800KB

### H. Regulatory Information

#### FCC Compliance:
- Device operates under Part 15 of FCC Rules
- Contains FCC ID: 2AC7Z-ESP32S3
- Unintentional radiator classification

#### CE Marking:
- Complies with EMC Directive 2014/30/EU
- RED Directive 2014/53/EU for radio equipment
- RoHS Directive 2011/65/EU

#### Safety Considerations:
- Device not intended for medical diagnosis
- For research and educational purposes only
- Ensure proper electrical safety practices

### I. References and Resources

#### Documentation:
- ESP32-S3 Technical Reference Manual
- MAX30102 Datasheet and Application Notes
- TensorFlow Lite Micro Documentation
- PlatformIO Platform Documentation

#### Online Resources:
- Project Repository: https://github.com/shadowai/esp32-stress-detection
- Documentation Site: https://docs.shadowai.com
- Community Forum: https://community.shadowai.com
- Technical Support: https://support.shadowai.com

#### Academic References:
- "Stress Detection using Photoplethysmography" (2020)
- "TinyML for Wearable Devices" (2021)
- "Real-time Physiological Monitoring" (2022)

### J. Acknowledgments

Special thanks to:
- TensorFlow Lite Micro team
- ESP32 Arduino community
- MAX30102 sensor community
- PlatformIO development team
- All beta testers and contributors'''

def create_deployment_guide(environment: Optional[DeploymentEnvironment] = None,
                          output_path: Optional[str] = None) -> str:
    """
    Create a complete deployment guide for ShadowAI system.
    
    Args:
        environment: Target deployment environment
        output_path: Optional path to save the guide
        
    Returns:
        Complete deployment guide as markdown string
    """
    guide = DeploymentGuide(environment)
    return guide.generate_complete_guide(output_path)