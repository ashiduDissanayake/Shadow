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

**Shadow** is an open-source wellness platform designed for privacy-conscious professionals. It unifies data from your Linux laptop, Android phone, and wrist wearable into a cohesive wellness ecosystem with **zero cloud dependency**.

### Key Features
- 🔒 **Privacy-First**: All data processing happens locally on your devices
- 🌐 **Peer-to-Peer**: Devices communicate directly without central servers
- 📊 **Real-time Insights**: Stress management, health monitoring, sleep optimization
- 🔧 **Modular Design**: Easy integration of new devices and sensors

---

## Current Status

### Week 8: Intelligent Notification System (Current)
- 🔄 **Notification Generation Algorithm**: Developing intelligent stress management message generation
- 🔄 **Notification Scheduling System**: Implementing adaptive notification timing based on stress patterns
- 🔄 **User Interaction Framework**: Building feedback mechanisms for notification effectiveness
- 🔄 **macOS Notification Integration**: Finalizing notification delivery and user interaction features

### Week 7: Deployment on ESP32-S3 & macOS Integration (Completed)
- ✅ **Model Deployment**: Successfully deployed MLP on ESP32-S3 using a custom pipeline
- ✅ **Real-time Inference Pipeline**: Built on-device ML inference with multi-sensor data fusion
- ✅ **macOS App Integration**: Completed end-to-end communication between ESP32 and macOS app
- ✅ **Stress Detection Pipeline**: Established complete stress detection workflow from sensors to macOS

### Week 6: macOS Host Application (Completed)
- ✅ **Complete macOS Application**: Swift-based companion application with full UI/UX
- ✅ **BLE Communication Stack**: Implemented robust Bluetooth Low Energy communication protocol
- ✅ **Data Processing Pipeline**: Built comprehensive data preprocessing and analysis framework
- ✅ **Core App Architecture**: Established modular app structure with real-time data handling

### Week 5: Model Finalization & Integration (Completed)
- ✅ **ExtraTreeClassifier Optimization**: Achieved required performance with ExtraTreeClassifier (no H-CNN needed) - Laterchanged to MLP model
- ✅ **Model Architecture**: Finalized and optimized ML model structure with satisfactory accuracy
- ✅ **Multi-Sensor Integration**: Three physiological sensors integrated with ESP32-S3
- ✅ **Real-time Data Acquisition**: 5 physiological parameters collected without latency

### Week 4: AI/ML Model Development (Completed)
- ✅ **Modular AI Structure**: Organized AI model development with proper folder structure
- ✅ **WESAD Dataset Integration**: Complete stress detection model training
- ✅ **ML Model Comparison**: Comprehensive analysis of different ML approaches
- ✅ **Baseline Model**: Initial stress detection implementation

### Week 3: Hardware Integration (Completed)
- ✅ **Device Documentation**: Complete hardware specs for MacBook, Android, and wearable devices
- ✅ **Feasibility Report**: End-to-end integration approach and device compatibility analysis
- ✅ **Hardware Architecture**: Detailed system integration design

### Week 2: Architecture Design (Completed)
- ✅ **Software Architecture**: C4-UML inspired system design
- ✅ **Hardware Architecture**: Device integration blueprint
- ✅ **Component Design**: Modular system architecture

### Week 1: Project Foundation (Completed)
- ✅ **Requirements Specification**: Complete SRS documentation
- ✅ **Project Vision**: Core concept and technical approach defined
- ✅ **Initial Presentations**: Project overview and technical specifications

---

## Technical Architecture

### Core Principles
- **Edge Processing**: All analytics and ML inference on local devices
- **P2P Communication**: Encrypted device-to-device synchronization
- **Resource Pooling**: Dynamic computational load distribution
- **Modular Integration**: Plug-and-play sensor and device support

### System Overview
<div align="center">
  <img src="assets/System_Overview.png" alt="System Overview" height="400" style="margin-bottom: 20px;" />
</div>

---

## 12-Week Development Roadmap

### ✅ Completed Weeks (1-7)
- **Week 1**: Project foundation and requirements specification
- **Week 2**: Software and hardware architecture design  
- **Week 3**: Device compatibility analysis and documentation
- **Week 4**: AI/ML model development with WESAD dataset
- **Week 5**: Model finalization with ExtraTreeClassifier optimization
- **Week 6**: Complete macOS application with BLE communication and data processing
- **Week 7**: TinyML deployment on ESP32-S3 and end-to-end macOS integration

### 🔄 Current Focus (Week 8)
**Intelligent Notification System**
- Developing notification generation algorithm for personalized stress management messages
- Implementing adaptive notification scheduling based on stress patterns and user context
- Building user feedback mechanisms for notification effectiveness optimization
- Creating contextual awareness integration with screen usage and physiological data
- Finalizing macOS notification center integration and user interaction flows

### ⏳ Upcoming Development Phase

**Week 9: Advanced Notification Analytics**
- Implement machine learning-based notification timing optimization
- Develop personalized stress triggers identification and response patterns
- Create adaptive notification frequency based on user responsiveness
- Build comprehensive notification effectiveness tracking and analytics
- Optimize notification content generation based on stress severity and context

**Week 10: Advanced Analytics & Personalization**
- Implement trend analysis and pattern recognition for long-term wellness insights
- Develop personalized stress management recommendations
- Create adaptive thresholds based on individual user behavior patterns
- Build comprehensive wellness dashboard with historical data visualization
- Implement data export capabilities and enhanced privacy controls

**Week 11: Testing & Optimization**
- Comprehensive system testing and validation across all components
- Performance benchmarking for TinyML inference and notification delivery
- User experience testing and interface refinement
- Security audit and privacy verification protocols
- Documentation of testing results and performance optimization metrics

**Week 12: Final Integration & Open Source Release**
- Complete ecosystem integration testing and validation
- Finalize comprehensive documentation and user installation guides
- Prepare open-source release with detailed contribution guidelines
- Create demonstration videos and comprehensive tutorial content
- Community outreach, project presentation, and open-source launch

---

## Current Technical Stack

### Hardware Components
- **ESP32-S3**: Main processing unit with TinyML capabilities
- **MAX30102**: Heart rate and SpO2 sensor
- **MPU6050**: 6-axis motion sensor
- **GSR Grove**: Galvanic skin response sensor
- **MLX90614**: Non-contact temperature sensor

### Software Architecture
- **ESP32 Firmware**: C++ with ESP-IDF framework and TensorFlow Lite Micro
- **TinyML**: On-device ExtraTreeClassifier inference
- **macOS Application**: Complete Swift application with BLE and notifications
- **Communication**: Enhanced BLE protocol with real-time data streaming
- **ML Pipeline**: Optimized ExtraTreeClassifier for embedded deployment
- **Notification Engine**: Intelligent stress management messaging system

### Current Capabilities
- Complete macOS companion application with full UI and BLE integration
- Real-time acquisition of 5 physiological parameters via ESP32-S3
- TinyML stress detection with ExtraTreeClassifier deployed on-device
- End-to-end stress detection pipeline from sensors to macOS app
- BLE communication stack with robust error handling and data streaming
- Screen usage monitoring and contextual analysis integration
- Multi-sensor data fusion architecture with real-time inference
- Foundation for intelligent notification system implementation

---

## Week 8 Focus Areas

### 🎯 Primary Objectives
1. **Notification Generation Algorithm**
   - Develop intelligent stress management message generation
   - Create personalized notification content based on stress levels
   - Implement contextual message selection algorithms
   - Build notification content library and categorization system

2. **Notification Scheduling System**
   - Implement adaptive notification timing based on stress patterns
   - Create user context-aware notification delivery
   - Build notification frequency optimization algorithms
   - Integrate with user activity and availability detection

3. **macOS Notification Integration**
   - Complete macOS notification center integration
   - Implement user interaction and feedback collection
   - Build notification effectiveness tracking system
   - Create user preference and customization options

4. **System Optimization & Testing**
   - Test notification delivery reliability and timing accuracy
   - Optimize notification algorithm performance
   - Validate user experience and interaction flows
   - Ensure seamless integration with existing stress detection pipeline

---

## Documentation

### Deliverables
- **Week 1**: [`deliverables/week1/`](deliverables/week1/) - SRS, presentations
- **Week 2**: [`deliverables/week2/`](deliverables/week2/) - Architecture designs
- **Week 3**: [`deliverables/week3/`](deliverables/week3/) - Hardware documentation
- **Week 4**: [`deliverables/week4/`](deliverables/week4/) - AI/ML models
- **Week 5**: [`deliverables/week5/`](deliverables/week5/) - ExtraTreeClassifier optimization and TinyML integration
- **Week 6**: [`deliverables/week6/`](deliverables/week6/) - Complete macOS application and BLE implementation
- **Week 7**: [`deliverables/week7/`](deliverables/week7/) - TinyML deployment and end-to-end system integration
- **Week 8**: [`deliverables/week8/`](deliverables/week8/) - Intelligent notification system and scheduling algorithms
- **AI Models**: [`models/`](models/) - Complete AI model development and deployment structure

### Research Papers
Comprehensive research collection in [`deliverables/researchs/`](deliverables/researchs/) covering:
- Advanced signal processing in wearable sensors
- Health monitoring technologies
- Machine learning applications in wellness
- TinyML and edge AI deployment
- Intelligent notification systems

---

## Contributing

We welcome contributions from developers, researchers, and privacy advocates!

### How to Contribute
1. **Star** this repository
2. **Report Issues**: [Open an issue](https://github.com/CSE-ICE-22/Shadow/issues)
3. **Submit PRs**: Fork, create feature branch, commit changes
4. **Join Discussions**: Share ideas and get community support

### Areas of Interest
- TinyML and edge AI deployment
- Intelligent notification systems
- Device integration and sensor support
- Machine learning and AI algorithms
- Privacy engineering and security
- User experience and interface design
- Documentation and tutorials

---

## Community

<div align="center">

**Join the Shadow Community**

[![GitHub Discussions](https://img.shields.io/badge/GitHub-Discussions-e4e0d3?style=for-the-badge&logo=github)](https://github.com/CSE-ICE-22/Shadow/discussions)
[![Discord](https://img.shields.io/badge/Discord-Community-e4e0d3?style=for-the-badge&logo=discord)](https://discord.gg/shadow-community)
[![Matrix](https://img.shields.io/badge/Matrix-Chat-e4e0d3?style=for-the-badge&logo=matrix)](https://matrix.to/#/#shadow:matrix.org)

</div>

---

## License

**MIT License** - see [LICENSE](LICENSE) for details

**Privacy Commitment**: Shadow processes no personal data on external servers. All processing occurs on user-controlled devices.

---

<div align="center">
  <p style="color: #e4e0d3; font-family: 'Noto Serif', serif; font-style: italic;">
    <em>Your wellness, your data, your control.</em>
  </p>
</div>