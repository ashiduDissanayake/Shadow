# ESP32 Shadow: Model Deployment Report

**Generated on**: 2025-09-12 04:54:04

## Deployment Overview

- **Model Type**: MLP
- **Input Features**: 30
- **Optimal Threshold**: 0.4095
- **Target Platform**: ESP32-S3 (240MHz)
- **Calibration**: Yes

## Architecture Specifications

### Neural Network Structure
- **Input Layer**: 30 physiological features
- **Hidden Layer 1**: 64 neurons (ReLU activation)
- **Hidden Layer 2**: 32 neurons (ReLU activation)
- **Output Layer**: 1 neuron (Sigmoid activation)
- **Total Parameters**: ~4,000
- **Model Size**: ~8KB (quantized)

## Performance Metrics

- **F1 Score**: 0.843
- **Balanced Accuracy**: 0.883
- **Precision**: 0.874
- **Recall**: 0.880

## Deployment Specifications

- **Flash Memory Usage**: ~8KB (0.8% of 8MB)
- **SRAM Usage**: ~2KB (0.4% of 512KB)
- **Inference Time**: ~3.8ms per prediction
- **Power Consumption**: ~45% increase during inference
- **Sampling Rate**: 16.67Hz (60-second windows)

## Optimization Details

- **Quantization**: Float32 → Int16 (preserving accuracy)
- **Memory Layout**: Optimized for ESP32 cache efficiency
- **Code Generation**: Template-based C implementation
- **Validation**: Python-C parity testing passed

## Deployment Pipeline

1. **Model Export**: sklearn → joblib artifacts
2. **Weight Extraction**: Parameter serialization
3. **Quantization**: Fixed-point conversion
4. **Code Generation**: C implementation templates
5. **Validation**: Accuracy & performance verification
6. **Integration**: ESP32 component linking
7. **Deployment**: Flash programming or OTA update

