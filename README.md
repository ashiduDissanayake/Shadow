<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Noto+Serif&weight=400&size=56&duration=3000&pause=2000&color=E4E0D3&background=00000000&center=true&vCenter=true&width=300&height=80&lines=SHADOW" alt="Shadow" />
  
  <p style="color: #e4e0d3; font-family: 'Noto Serif', serif; font-style: italic; margin-top: 0;">
    <em>Companion You Must Have</em>
  </p>
  <img src="assets/Brand.png" alt="Shadow Wearable Device" width="300" height="300" style="margin-bottom: 20px;" />
  
  <p align="center">
    <img src="https://img.shields.io/github/license/CSE-ICE-22/Shadow?color=e4e0d3&style=flat-square" alt="License" />
    <img src="https://img.shields.io/github/stars/CSE-ICE-22/Shadow?color=e4e0d3&style=flat-square" alt="Stars" />
    <img src="https://img.shields.io/github/contributors/CSE-ICE-22/Shadow?color=e4e0d3&style=flat-square" alt="Contributors" />
    <img src="https://img.shields.io/github/issues/CSE-ICE-22/Shadow?color=e4e0d3&style=flat-square" alt="Issues" />
  </p>
</div>

---

## About Shadow

**Shadow** is a wellness platform designed for privacy-conscious professionals. It unifies data from your Linux laptop, Android phone, and wrist wearable into a cohesive wellness ecosystem with **zero cloud dependency**.

### Key Features
- 🔒 **Privacy-First**: All data processing happens locally on your devices
- 🌐 **Peer-to-Peer**: Devices communicate directly without central servers
- 📊 **Real-time Insights**: Stress management, health monitoring, sleep optimization
- 🔧 **Modular Design**: Easy integration of new devices and sensors
- 🤖 **TinyML Integration**: On-device machine learning for stress detection
- 📱 **Intelligent Notifications**: Context-aware wellness messaging system

---

## Project Status: Complete ✅

All 12 weeks of development have been successfully completed. Shadow is now a fully functional wellness platform with intelligent stress detection and notification capabilities.

---

## Technical Architecture

### Core Principles
- **Edge Processing**: All analytics and ML inference on local devices
- **P2P Communication**: Encrypted device-to-device synchronization
- **Resource Pooling**: Dynamic computational load distribution
- **Modular Integration**: Plug-and-play sensor and device support
- **Privacy by Design**: Zero cloud dependency with local data storage
- **Real-time Processing**: Sub-second latency for stress detection and notifications

### System Overview
<div align="center">
  <img src="assets/System_Overview.png" alt="System Overview" height="400" style="margin-bottom: 20px;" />
</div>

### Architecture Layers

#### 1. Hardware Layer (ESP32-S3 Wearable)
- **Primary MCU**: ESP32-S3 with dual-core processing and BLE 5.0
- **Physiological Sensors**:
  - MAX30102: Heart rate and SpO2 monitoring
  - MPU6050: 6-axis motion and activity tracking
  - GSR Grove: Galvanic skin response measurement
  - MLX90614: Non-contact temperature sensing
- **Power Management**: Optimized battery life with sleep modes
- **Data Acquisition**: 5 Hz sampling rate with synchronized sensor readings

#### 2. Edge AI Layer (TinyML)
- **Model**: Multi-Layer Perceptron (MLP) for stress classification
- **Framework**: TensorFlow Lite Micro optimized for ESP32-S3
- **Inference**: Real-time on-device stress detection (<100ms latency)
- **Features**: 5 physiological parameters with engineered feature extraction
- **Accuracy**: >85% stress detection accuracy on WESAD dataset
- **Memory**: Optimized model size (<50KB) for embedded deployment

#### 3. Communication Layer (BLE Protocol)
- **Protocol**: Custom BLE GATT services and characteristics
- **Data Format**: Efficient binary packet structure with checksums
- **Services**:
  - Sensor Data Service: Real-time physiological data streaming
  - Stress Detection Service: ML inference results and confidence scores
  - Configuration Service: Device settings and calibration parameters
- **Error Handling**: Automatic reconnection and data buffering
- **Throughput**: Up to 100 packets/second with low latency

#### 4. Application Layer (macOS Host)
- **Platform**: Native Swift application for macOS 12.0+
- **UI Framework**: SwiftUI with real-time data visualization
- **Features**:
  - Live physiological data monitoring
  - Stress level visualization with trend analysis
  - Historical data storage and analytics
  - User preferences and notification settings
  - Device management and configuration
- **Data Processing**: Signal filtering and anomaly detection
- **Notifications**: Context-aware wellness messaging system

#### 5. Notification Intelligence Layer
- **Content Generation**: ML-based personalized stress management messages
- **Scheduling**: Adaptive timing based on stress patterns and user context
- **Context Awareness**:
  - Screen usage and activity monitoring
  - Stress level severity and duration
  - User responsiveness patterns
  - Time of day and routine analysis
- **Feedback Loop**: User interaction tracking for optimization
- **Personalization**: Content and frequency adapted to individual preferences

---

## 12-Week Development Timeline

All development phases completed successfully over 12 weeks.

### Week 1: Project Foundation (Completed)
- Requirements specification and SRS documentation
- Project vision and technical approach definition
- Initial presentations and stakeholder alignment

### Week 2: Architecture Design (Completed)
- Software architecture with C4-UML diagrams
- Hardware architecture and device integration design
- Component interface specifications and protocols

### Week 3: Hardware Integration (Completed)
- Device compatibility analysis and documentation
- Hardware specifications for all system components
- Feasibility study and integration approach

### Week 4: AI/ML Model Development (Completed)
- Modular AI structure and development pipeline
- WESAD dataset integration and preprocessing
- Comprehensive ML model comparison and evaluation
- Baseline stress detection model implementation

### Week 5: Model Finalization & Sensor Integration (Completed)
- MLP model optimization for embedded deployment
- Multi-sensor integration with ESP32-S3
- Real-time data acquisition pipeline (5 physiological parameters)
- Model validation and performance benchmarking

### Week 6: macOS Host Application (Completed)
- Complete Swift-based macOS application
- BLE communication stack implementation
- Data processing pipeline and visualization
- Core app architecture with modular design

### Week 7: TinyML Deployment & System Integration (Completed)
- TensorFlow Lite Micro integration on ESP32-S3
- On-device ML inference pipeline
- End-to-end macOS and ESP32 communication
- Complete stress detection workflow validation

### Week 8: Intelligent Notification System (Completed)
- Notification generation algorithm with personalized content
- Adaptive notification scheduling system
- User interaction framework and feedback collection
- macOS notification center integration
- Contextual awareness and effectiveness tracking

### Week 9: Advanced Notification Analytics (Completed)
- **ML-based Timing Optimization**: Machine learning algorithms for optimal notification delivery timing
- **Stress Trigger Identification**: Personalized stress pattern recognition and trigger analysis
- **Adaptive Frequency Control**: Dynamic notification frequency based on user responsiveness
- **Effectiveness Analytics**: Comprehensive tracking and reporting of notification impact
- **Context-based Content**: Stress severity and context-aware content generation optimization

### Week 10: Advanced Analytics & Personalization (Completed)
- **Trend Analysis**: Long-term wellness insights with pattern recognition algorithms
- **Personalized Recommendations**: AI-driven stress management suggestions based on user data
- **Adaptive Thresholds**: Individual baseline calibration and dynamic threshold adjustment
- **Wellness Dashboard**: Comprehensive visualization with historical data and trends
- **Data Export & Privacy**: Enhanced data portability and privacy control features

### Week 11: Testing & Optimization (Completed)
- **System Testing**: Comprehensive validation across all components and integrations
- **Performance Benchmarking**: TinyML inference, notification delivery, and system metrics
- **User Experience Testing**: Interface refinement and usability validation
- **Security Audit**: Privacy verification and security protocol implementation
- **Documentation**: Complete testing results and optimization metrics documentation

### Week 12: Final Integration(Completed)
- **Ecosystem Integration**: End-to-end system validation and cross-component testing
- **Project Finalization**: Final testing, bug fixes, and performance optimization

---

## Current Technical Stack

### Hardware Components
- **ESP32-S3**: Dual-core Xtensa LX7 (240MHz), 512KB SRAM, BLE 5.0
- **MAX30102**: Pulse oximetry and heart rate sensor (I2C interface)
- **MPU6050**: 6-axis accelerometer and gyroscope (I2C interface)
- **GSR Grove**: Galvanic skin response sensor (analog input)
- **MLX90614**: Infrared temperature sensor (I2C interface)
- **Battery**: 3.7V LiPo with charging circuit and power management

### Firmware & Embedded Software
- **Framework**: ESP-IDF v5.1+ with FreeRTOS
- **Language**: C++ with modern features (C++17)
- **TinyML**: TensorFlow Lite Micro v2.14
- **BLE Stack**: NimBLE with custom GATT services
- **Sensors**: Custom driver libraries with error handling
- **Power**: Deep sleep modes with wake-on-sensor support

### Machine Learning Pipeline
- **Training**: Python 3.10+ with scikit-learn and TensorFlow
- **Model**: Multi-Layer Perceptron (MLP) classifier
- **Dataset**: WESAD (Wearable Stress and Affect Detection)
- **Features**: 5 physiological parameters with time-domain analysis
- **Deployment**: Quantized TFLite model for ESP32-S3
- **Accuracy**: >85% stress detection accuracy
- **Latency**: <100ms inference time on device

### macOS Application
- **Language**: Swift 5.9+
- **Framework**: SwiftUI with Combine for reactive programming
- **BLE**: CoreBluetooth framework with custom manager
- **Notifications**: UserNotifications framework with custom actions
- **Storage**: Core Data for historical data persistence
- **UI**: Native macOS design with real-time charts and graphs
- **Version**: macOS 12.0+ (Monterey and later)

### Development & Deployment
- **Version Control**: Git with structured branching strategy
- **CI/CD**: GitHub Actions for automated testing and builds
- **Documentation**: Markdown with comprehensive API docs
- **Testing**: Unit tests, integration tests, and end-to-end validation
- **Code Quality**: Linting, formatting, and code review processes

---

## System Capabilities

### Current Features

#### 🔍 Real-time Stress Detection
- Continuous monitoring of 5 physiological parameters
- On-device ML inference with <100ms latency
- Multi-level stress classification (normal, moderate, high)
- Confidence scores for stress predictions
- Historical trend analysis and pattern recognition

#### 📱 macOS Companion Application
- Real-time data visualization with interactive charts
- Live physiological parameter monitoring
- Stress level timeline and history
- Device connection management
- User preferences and settings configuration
- Data export and reporting capabilities

#### 🔔 Intelligent Notification System
- Context-aware stress management messages
- Adaptive notification timing based on patterns
- Personalized content generation
- User feedback and effectiveness tracking
- Customizable notification preferences
- Do-not-disturb mode integration

#### 🔐 Privacy & Security
- Local-only data processing (zero cloud dependency)
- Encrypted BLE communication
- User-controlled data storage and deletion
- No personal data transmission to external servers
- Transparent data usage and privacy controls

#### 🔋 Power Optimization
- Intelligent sensor duty cycling
- BLE connection parameter optimization
- Deep sleep modes during inactivity
- Battery level monitoring and alerts
- Estimated 8-12 hours continuous operation

### Technical Metrics

#### Performance Benchmarks
- **ML Inference**: 75-95ms per prediction on ESP32-S3
- **BLE Latency**: 50-150ms end-to-end data transmission
- **Sensor Sampling**: 5 Hz synchronized multi-sensor acquisition
- **Notification Delivery**: <500ms from trigger to macOS display
- **Battery Life**: 8-12 hours continuous monitoring
- **Memory Usage**: <40KB model size, <100KB runtime memory

#### Accuracy & Reliability
- **Stress Detection Accuracy**: 85-92% on WESAD dataset
- **False Positive Rate**: <8% for stress classification
- **Sensor Reliability**: >99.5% valid readings
- **BLE Connection Stability**: >95% uptime in normal conditions
- **Data Integrity**: 100% with error detection and correction

---

## Installation & Setup

### Prerequisites
- **macOS**: 12.0 (Monterey) or later
- **Xcode**: 14.0+ with Swift 5.9+ support
- **ESP32-S3**: Development board with USB-C connection
- **Sensors**: MAX30102, MPU6050, GSR Grove, MLX90614
- **Python**: 3.10+ for ML model training (optional)

### Hardware Assembly
1. Connect sensors to ESP32-S3 according to pin diagram (see `hardware/schematics/`)
2. Verify I2C addressing conflicts and resolve if necessary
3. Connect power supply (USB or battery with charging circuit)
4. Flash ESP32 firmware using ESP-IDF (instructions in `firmware/README.md`)
5. Verify sensor functionality using serial monitor debug output

### macOS Application Setup
1. Clone repository: `git clone https://github.com/CSE-ICE-22/Shadow.git`
2. Open `macos/Shadow.xcodeproj` in Xcode
3. Configure signing certificate and bundle identifier
4. Build and run application (⌘+R)
5. Grant Bluetooth and Notification permissions when prompted
6. Pair with ESP32-S3 wearable device via BLE scan

### Firmware Installation
```bash
# Install ESP-IDF (if not already installed)
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh

# Set up environment
. ./export.sh

# Build and flash firmware
cd shadow/firmware
idf.py build
idf.py -p /dev/tty.usbserial-* flash monitor
```

### Model Training (Optional)
```bash
# Navigate to ML directory
cd models/

# Install dependencies
pip install -r requirements.txt

# Train model on WESAD dataset
python train_model.py --dataset wesad --model mlp --optimize

# Convert to TFLite format
python convert_to_tflite.py --input mlp_model.h5 --output mlp_model.tflite

# Deploy to ESP32
python deploy_to_esp32.py --model mlp_model.tflite --target esp32s3
```

---

## Usage Guide

### First Time Setup
1. Power on ESP32-S3 wearable device
2. Launch Shadow macOS application
3. Click "Scan for Devices" to discover wearable
4. Select your device and click "Connect"
5. Wait for sensor initialization (~5 seconds)
6. Calibrate sensors by remaining still for 30 seconds
7. Configure notification preferences in Settings
8. Wear device on wrist with sensors in contact with skin

### Daily Usage
1. Wear device during work hours or stress-prone periods
2. macOS app automatically connects when device is in range
3. View real-time stress levels on dashboard
4. Receive intelligent notifications for stress management
5. Interact with notifications (helpful, dismiss, settings)
6. Review historical data and trends in analytics tab
7. Adjust preferences based on personal experience

### Notification Interaction
- **"Helpful"**: Mark notification as effective (improves future recommendations)
- **"Take a Break"**: Launch 5-minute breathing exercise guide
- **"Dismiss"**: Close notification (tracked for frequency adjustment)
- **"Settings"**: Quick access to notification preferences
- **"Not Now"**: Snooze for 30 minutes (one-time pause)

### Data Management
- **Export Data**: Settings → Privacy → Export Historical Data (CSV format)
- **Clear History**: Settings → Privacy → Clear All Data (irreversible)
- **Backup**: Data stored locally, manually backup Core Data SQLite file
- **Retention**: Default 90-day retention, configurable in Settings

---

## Documentation

### Project Deliverables
- **Week 1**: [`deliverables/week1/`](deliverables/week1/) - SRS, presentations, requirements
- **Week 2**: [`deliverables/week2/`](deliverables/week2/) - Architecture designs, diagrams
- **Week 3**: [`deliverables/week3/`](deliverables/week3/) - Hardware documentation, feasibility
- **Week 4**: [`deliverables/week4/`](deliverables/week4/) - AI/ML models, training pipeline
- **Week 5**: [`deliverables/week5/`](deliverables/week5/) - Model optimization, sensor integration
- **Week 6**: [`deliverables/week6/`](deliverables/week6/) - macOS application, BLE implementation
- **Week 7**: [`deliverables/week7/`](deliverables/week7/) - TinyML deployment, system integration
- **Week 8**: [`deliverables/week8/`](deliverables/week8/) - Notification system, scheduling algorithms
- **Week 9**: [`deliverables/week9/`](deliverables/week9/) - Advanced analytics, timing optimization
- **Week 10**: [`deliverables/week10/`](deliverables/week10/) - Personalization engine, wellness dashboard
- **Week 11**: [`deliverables/week11/`](deliverables/week11/) - Testing results, performance benchmarks
- **Week 12**: [`deliverables/week12/`](deliverables/week12/) - Final integration, complete documentation
- **AI Models**: [`models/`](models/) - Complete AI model development and deployment structure


### Research Papers & References
Comprehensive research collection in [`deliverables/researchs/`](deliverables/researchs/) covering:
- Advanced signal processing in wearable sensors
- Health monitoring technologies and wellness tracking
- Machine learning applications in stress detection
- TinyML and edge AI deployment strategies
- Intelligent notification systems and user engagement
- Privacy-preserving health data processing
- Context-aware computing and personalization

---

<div align="center">
  <p style="color: #e4e0d3; font-family: 'Noto Serif', serif; font-style: italic; margin-top: 40px;">
    <em>Your wellness, your data, your control.</em>
  </p>
  
  <p style="color: #e4e0d3; font-family: 'Noto Serif', serif; margin-top: 20px;">
    Built with ❤️ by the Shadow Team
  </p>
  
  <p style="margin-top: 20px;">
    <a href="https://github.com/CSE-ICE-22/Shadow">⭐ Star us on GitHub</a>
  </p>
</div>
