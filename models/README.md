# Shadow AI Models

This directory contains all AI/ML components for the Shadow wellness platform, organized for efficient development and deployment.

## Directory Structure

```
models/
├── README.md                                 # This file
├── requirements.txt                          # Dependencies
├── example_usage.py                          # Usage demonstration
├── 01_Model_Introduction_and_Overview.ipynb  # Project overview and planning
├── 02_Phase1_Data_Preprocessing.ipynb        # Data preprocessing pipeline
├── data/                                     # Data management
│   ├── raw/                                  # Original datasets (WESAD, etc.)
│   ├── processed/                            # Cleaned and preprocessed data
│   └── features/                             # Extracted features
├── preprocessing/                            # Data preprocessing modules
│   ├── __init__.py
│   ├── bvp_processor.py                      # BVP signal processing
│   ├── feature_extractor.py                  # HRV and other feature extraction
│   └── data_loader.py                        # Dataset loading utilities
├── training/                                 # Model training
│   ├── __init__.py
│   ├── hybrid_cnn.py                         # Hybrid CNN model architecture
│   ├── train.py                              # Training script
│   └── validation.py                         # Model validation utilities
├── inference/                                # Model inference
│   ├── __init__.py
│   ├── predictor.py                          # Real-time prediction
│   └── batch_inference.py                    # Batch processing
├── deployment/                               # TinyML deployment
│   ├── __init__.py
│   ├── esp32_integration.py                  # ESP32-specific code
│   ├── model_converter.py                    # TensorFlow Lite conversion
│   └── quantization.py                       # Model quantization
├── utils/                                    # Utility functions
│   ├── __init__.py
│   ├── metrics.py                            # Evaluation metrics
│   ├── visualization.py                      # Plotting utilities
│   └── helpers.py                            # General helper functions
├── configs/                                  # Configuration files
│   ├── model_config.yaml                     # Model hyperparameters
│   ├── data_config.yaml                      # Data processing settings
│   └── deployment_config.yaml                # Deployment settings
└── results/                                  # Output and results
    ├── logs/                                 # Training logs
    ├── plots/                                # Generated plots
    ├── models/                               # Saved model files
    └── metrics/                              # Performance metrics
```

## Development Workflow

### 📋 Phase-by-Phase Approach

1. **Phase 1: Data Preprocessing** → `02_Phase1_Data_Preprocessing.ipynb`
   - Load and understand WESAD dataset
   - Implement BVP signal filtering and segmentation
   - Extract HRV features
   - Apply normalization and quality checks

2. **Phase 2: Model Training** → `03_Phase2_Model_Training.ipynb`
   - Implement Hybrid CNN architecture
   - Train with LOSO cross-validation
   - Evaluate performance metrics

3. **Phase 3: TinyML Conversion** → `04_Phase3_TinyML_Conversion.ipynb`
   - Convert to TensorFlow Lite
   - Apply quantization
   - Optimize for ESP32-S3

4. **Phase 4: ESP32 Deployment** → `05_Phase4_ESP32_Integration.ipynb`
   - Set up ESP32 development environment
   - Implement on-device preprocessing
   - Test real-time inference

## Quick Start

### 1. Data Preparation
```python
from preprocessing.data_loader import WESADLoader
from preprocessing.bvp_processor import BVPProcessor

# Load WESAD dataset
loader = WESADLoader()
data = loader.load_bvp_data()

# Process BVP signals
processor = BVPProcessor()
processed_data = processor.process(data)
```

### 2. Model Training
```python
from training.hybrid_cnn import HybridCNN
from training.train import train_model

# Initialize model
model = HybridCNN()

# Train model
train_model(model, processed_data)
```

### 3. Model Deployment
```python
from deployment.model_converter import convert_to_tflite
from deployment.esp32_integration import ESP32Deployer

# Convert to TensorFlow Lite
tflite_model = convert_to_tflite(model)

# Deploy to ESP32
deployer = ESP32Deployer()
deployer.deploy(tflite_model)
```

## Model Architecture

### Hybrid CNN (H-CNN)
- **Input 1**: 3840x1 BVP segment (60-second window)
- **CNN Branch**: Convolutional layers for signal processing
- **Input 2**: 19 HRV features
- **Dense Branch**: Feature processing
- **Concatenation**: Combine both branches
- **Output**: Stress classification (Baseline, Stress, Amusement, Meditation)

## Data Pipeline

1. **Raw Data**: WESAD dataset BVP signals
2. **Preprocessing**: Filtering, segmentation, feature extraction
3. **Training**: Hybrid CNN with LOSO cross-validation
4. **Optimization**: Quantization for TinyML deployment
5. **Deployment**: ESP32-S3 integration

## Configuration

All model parameters are configurable through YAML files in `configs/`:
- Model architecture and hyperparameters
- Data processing settings
- Training parameters
- Deployment configurations

## Results Tracking

Training results, metrics, and model files are automatically saved to `results/`:
- Training logs and metrics
- Model performance plots
- Saved model checkpoints
- Deployment-ready models

## Getting Started

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Add WESAD Data**: Place dataset in `data/raw/wesad/`
3. **Run Phase 1**: Execute `02_Phase1_Data_Preprocessing.ipynb`
4. **Continue Pipeline**: Follow numbered notebooks sequentially

## Expected Outcomes

- **Phase 1**: Clean, processed BVP segments ready for training
- **Phase 2**: Trained Hybrid CNN model with >85% accuracy
- **Phase 3**: Optimized TinyML model <100KB for ESP32
- **Phase 4**: Real-time stress detection on wearable device
