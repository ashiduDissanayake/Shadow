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

### Week 6: MLOps Pipeline & NLP Integration (Current)
- 🔄 **MLOps Pipeline**: Building deployment-ready ML model pipeline for production use
- 🔄 **NLP Model Development**: Creating custom message generator for personalized stress insights
- 🔄 **Model Deployment**: Implementing containerized deployment environment for cross-platform compatibility
- 🔄 **Automated Testing**: Creating model validation and testing frameworks

### Week 5: Model Finalization & TinyML Integration (Completed)
- ✅ **ExtraTreeClassifier Optimization**: Achieved required performance with ExtraTreeClassifier (no H-CNN needed)
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

### ✅ Completed Weeks (1-5)
- **Week 1**: Project foundation and requirements specification
- **Week 2**: Software and hardware architecture design  
- **Week 3**: Device compatibility analysis and documentation
- **Week 4**: AI/ML model development with WESAD dataset
- **Week 5**: Model finalization with ExtraTreeClassifier

### 🔄 Current Focus (Week 6)
**MLOps Pipeline Development & NLP Integration**
- Building deployment-ready MLOps pipeline for model versioning and CI/CD
- Developing NLP model for personalized stress management message generation
- Implementing automated model testing and validation frameworks
- Creating containerized deployment environment for cross-platform compatibility

### ⏳ Upcoming Development Phase

**Week 7: ESP32-S3 Firmware & BLE Communication**
- Complete ESP32-S3 firmware with TinyML inference
- Implement BLE enhanced protocol for device communication
- Develop real-time data preprocessing pipeline on ESP32
- Test multi-sensor data fusion and local ML inference

**Week 8: Host Application Development**
- Complete Swift/macOS companion application
- Implement BLE communication stack on host side
- Develop data preprocessing and analysis pipeline
- Integrate screen usage monitoring for contextual stress detection

**Week 9: Stress Detection Algorithm Integration**
- Combine physiological data with screen usage patterns
- Implement comprehensive stress detection algorithm
- Develop feedback mechanisms and user notifications
- Test end-to-end stress detection accuracy

**Week 10-11: System Optimization & Testing**
- Optimize power consumption on ESP32-S3
- Implement low-power modes and duty cycling
- Comprehensive system testing and validation
- Performance tuning and accuracy improvements, benchmarking

**Week 11-12: Final Integration & Documentation**
- Complete ecosystem integration (wearable + phone + laptop)
- Implement advanced analytics and trend visualization
- Finalize documentation and user guides
- Prepare for open-source release

---

## Current Technical Stack

### Hardware Components
- **ESP32-S3**: Main processing unit with TinyML capabilities
- **MAX30102**: Heart rate and SpO2 sensor
- **MPU9250**: 9-axis motion sensor
- **GSR Grove**: Galvanic skin response sensor
- **MLX90614**: Non-contact temperature sensor

### Software Architecture
- **ESP32 Firmware**: C++ with ESP-IDF framework
- **TinyML**: TensorFlow Lite Micro for on-device inference
- **Host Application**: Swift (macOS) / Kotlin (Android)
- **Communication**: BLE enhanced protocol
- **ML Pipeline**: Python with scikit-learn (ExtraTreeClassifier)
- **MLOps**: Docker, CI/CD pipelines, model versioning
- **NLP**: Custom message generation for stress insights

### Current Capabilities
- Real-time acquisition of 5 physiological parameters
- UART communication for development/debugging
- Optimized stress detection using ExtraTreeClassifier
- Multi-sensor data fusion on ESP32-S3
- Production-ready ML model pipeline
- Personalized stress management messaging

---

## Documentation

### Deliverables
- **Week 1**: [`deliverables/week1/`](deliverables/week1/) - SRS, presentations
- **Week 2**: [`deliverables/week2/`](deliverables/week2/) - Architecture designs
- **Week 3**: [`deliverables/week3/`](deliverables/week3/) - Hardware documentation
- **Week 4**: [`deliverables/week4/`](deliverables/week4/) - AI/ML models
- **Week 5**: [`deliverables/week5/`](deliverables/week5/) - ExtraTreeClassifier optimization and TinyML integration
- **Week 6**: [`deliverables/week6/`](deliverables/week6/) - MLOps pipeline and NLP models
- **AI Models**: [`models/`](models/) - Complete AI model development structure

### Research Papers
Comprehensive research collection in [`deliverables/researchs/`](deliverables/researchs/) covering:
- Advanced signal processing in wearable sensors
- Health monitoring technologies
- Machine learning applications in wellness
- MLOps and deployment strategies
- Natural language processing for health applications

---

## Contributing

We welcome contributions from developers, researchers, and privacy advocates!

### How to Contribute
1. **Star** this repository
2. **Report Issues**: [Open an issue](https://github.com/CSE-ICE-22/Shadow/issues)
3. **Submit PRs**: Fork, create feature branch, commit changes
4. **Join Discussions**: Share ideas and get community support

### Areas of Interest
- Device integration and sensor support
- Machine learning and AI algorithms
- Privacy engineering and security
- User experience and interface design
- Documentation and tutorials
- MLOps and deployment automation
- Natural language processing

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