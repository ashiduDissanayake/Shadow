# ESP32-S3 TensorFlow Lite Conversion Project

## Overview
Converting ExtraTreesClassifier (100 trees, 73 features) to TensorFlow Lite for ESP32-S3 deployment.

## Project Structure
```
esp32-tflite-conversion/
├── data/                 # Original model files
├── stages/              # Stage-by-stage conversion scripts
├── outputs/             # Generated files for each stage
├── docs/               # Documentation and analysis
└── requirements.txt    # Python dependencies
```

## ESP32-S3 Target Specs
- **CPU**: Dual-core Xtensa LX7 @ 240MHz
- **SRAM**: 512KB total (~300-400KB usable)
- **Flash**: 8MB+
- **Target Memory**: <150KB for ML model
- **Target Inference**: <10ms

## Current Model Specs
- **Type**: ExtraTreesClassifier
- **Trees**: 100 estimators
- **Features**: 73 (BVP, ACC, EDA, TEMP sensors)
- **Memory**: ~295KB (too large)
- **Threshold**: 0.1896

## Conversion Strategy (Option C - Hybrid)

### Stage 1: Feature Selection ✅
- **Goal**: 73 → 25-30 features
- **Method**: Importance-based selection
- **Target**: 90%+ importance retention

### Stage 2: Tree Optimization
- **Goal**: 100 → 50-75 trees  
- **Method**: Intelligent tree selection
- **Target**: Minimal accuracy loss

### Stage 3: Neural Network Conversion
- **Goal**: Trees → TensorFlow model
- **Method**: Synthetic data distillation
- **Target**: <5% accuracy loss

### Stage 4: TensorFlow Lite Conversion
- **Goal**: TF → TFLite with quantization
- **Method**: INT8 quantization
- **Target**: <150KB model size

### Stage 5: ESP32-S3 Integration
- **Goal**: Deploy on hardware
- **Method**: TFLite Micro integration
- **Target**: <10ms inference

## Development Approach
- Build incrementally: Stage → Test → Validate → Next Stage
- No "black box" development
- Full understanding at each step
- Embedded engineer collaboration

## Next Steps
1. Run Stage 1: Feature selection
2. Validate results before Stage 2
3. Proceed stage by stage with testing
