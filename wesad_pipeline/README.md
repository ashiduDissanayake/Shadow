# WESAD Analysis Pipeline

A comprehensive, modular pipeline for WESAD (Wearable Stress and Affect Detection) dataset analysis with focus on BVP signal processing, quality assessment, and windowing for machine learning applications.

## Features

- **Modular Design**: Independent, reusable components for different analysis stages
- **Comprehensive Analysis**: Signal quality assessment, heart rate estimation, and windowing analysis
- **Robust Processing**: Error handling, data validation, and progress tracking
- **Rich Visualizations**: Signal plots, quality metrics, and dataset-wide analysis
- **Flexible Configuration**: Centralized configuration with validation
- **Multiple Export Formats**: NumPy arrays, CSV, and JSON export options
- **Detailed Reporting**: JSON reports and human-readable summaries

## Directory Structure

```
wesad_pipeline/
├── main.py                    # Main pipeline runner
├── config/
│   ├── __init__.py
│   └── config.py             # Configuration settings
├── data/
│   ├── __init__.py
│   ├── loader.py             # WESAD data loading
│   └── preprocessor.py       # Data alignment & cleaning
├── analysis/
│   ├── __init__.py
│   ├── signal_quality.py     # Signal quality computation
│   ├── heart_rate.py         # Heart rate estimation
│   └── windowing.py          # Window creation & analysis
├── visualization/
│   ├── __init__.py
│   ├── signal_plots.py       # BVP, quality, HR plots
│   ├── window_plots.py       # Windowing analysis plots
│   └── dataset_plots.py      # Full dataset analysis
├── utils/
│   ├── __init__.py
│   ├── helpers.py            # Common utilities
│   └── documentation.py     # Report generation
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # Unit tests
└── example_usage.py          # Usage demonstration
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ashiduDissanayake/Shadow.git
cd Shadow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install tqdm  # Additional dependency for progress bars
```

3. Download the WESAD dataset from: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

4. Extract the dataset to `data/raw/wesad/` directory

## Quick Start

### Command Line Usage

```bash
# Run complete analysis on subjects 2, 3, 4, 5
python wesad_pipeline/main.py --wesad-path data/raw/wesad/ --output-path wesad_analysis/ --subjects 2 3 4 5

# Customize analysis parameters
python wesad_pipeline/main.py \
    --wesad-path data/raw/wesad/ \
    --output-path wesad_analysis/ \
    --window-size 30 \
    --overlap 5 \
    --quality-threshold 0.7
```

### Python API Usage

```python
from wesad_pipeline.main import WESADPipeline

# Initialize pipeline
pipeline = WESADPipeline(
    wesad_path="data/raw/wesad/",
    output_path="wesad_analysis/",
    subjects=[2, 3, 4, 5]  # Optional: specify subjects
)

# Run full analysis
results = pipeline.run_analysis()

# Or run specific components
dataset_results = pipeline.run_data_processing()
dataset_results = pipeline.run_signal_analysis(dataset_results)
dataset_results = pipeline.run_windowing_analysis(dataset_results)
pipeline.generate_visualizations(dataset_results)
pipeline.generate_reports(dataset_results)
```

### Test the Pipeline

Run the example demonstration to test functionality with simulated data:

```bash
python wesad_pipeline/example_usage.py
```

## Configuration

The pipeline uses a centralized configuration system. Key parameters:

### Dataset Configuration
- `bvp_sampling_rate`: BVP signal sampling rate (default: 64 Hz)
- `subjects`: List of subject IDs to process (default: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17])
- `target_conditions`: Conditions to analyze (default: ['baseline', 'stress', 'amusement'])

### Analysis Configuration
- `window_size_seconds`: Window size for analysis (default: 60 seconds)
- `overlap_seconds`: Window overlap (default: 5 seconds)
- `quality_threshold`: Minimum signal quality threshold (default: 0.6)
- `min_heart_rate` / `max_heart_rate`: Physiological heart rate range (default: 40-200 BPM)

### Output Configuration
- `output_path`: Base output directory (default: "wesad_analysis")
- `save_plots`: Enable plot generation (default: True)
- `export_formats`: Data export formats (NumPy, CSV, JSON)

## Core Components

### 1. Data Loading (`data/loader.py`)
- Load WESAD pickle files with error handling
- Subject validation and availability checking
- Progress tracking for batch operations
- Caching support for faster subsequent loads

### 2. Data Preprocessing (`data/preprocessor.py`)
- Align labels from 700Hz (respiration) to 64Hz (BVP)
- Signal cleaning and artifact removal
- Data validation and quality checks
- Timestamp generation

### 3. Signal Quality Assessment (`analysis/signal_quality.py`)
- Multi-metric quality assessment (variance, periodicity, morphology)
- Sliding window quality evaluation
- Quality threshold validation
- Real-time quality monitoring

### 4. Heart Rate Analysis (`analysis/heart_rate.py`)
- BVP peak detection using scipy.signal.find_peaks
- Heart rate estimation from peak intervals
- Heart rate variability (HRV) metrics
- Physiological validation (40-200 BPM range)

### 5. Windowing Analysis (`analysis/windowing.py`)
- Sliding window creation with configurable overlap
- Window label assignment (most common label)
- Window quality and confidence scoring
- Time-domain and frequency-domain feature extraction

### 6. Visualization (`visualization/`)
- **Signal plots**: BVP signals with condition backgrounds
- **Window plots**: Window creation and distribution analysis
- **Dataset plots**: Cross-subject comparisons and dataset overview
- High-quality plots (300 DPI) with customizable styling

### 7. Documentation & Export (`utils/`)
- JSON report generation for subjects and dataset
- Human-readable analysis summaries
- Multi-format data export (NumPy, CSV, JSON)
- Comprehensive statistics and performance metrics

## Output Structure

```
wesad_analysis/
├── plots/
│   ├── signals/              # Signal-level plots
│   ├── windows/              # Window analysis plots
│   └── dataset/              # Dataset-wide plots
├── reports/                  # JSON and text reports
├── processed_data/           # Exported data arrays
└── logs/                     # Pipeline execution logs
```

## Key Features

### Robust Error Handling
- Comprehensive data validation
- Graceful handling of missing subjects
- Detailed error logging and recovery
- Progress tracking with informative messages

### Quality Assessment
- Multi-metric signal quality evaluation
- Adaptive thresholding based on signal characteristics
- Window-level quality filtering
- Quality-based acceptance/rejection criteria

### Heart Rate Analysis
- Adaptive peak detection algorithms
- Physiological validation of heart rate estimates
- Heart rate variability analysis
- Real-time monitoring capabilities

### Modular Architecture
- Independent, reusable components
- Configurable analysis parameters
- Extensible design for new analysis methods
- Clear separation of concerns

### Comprehensive Visualization
- Signal plots with condition annotations
- Quality metrics visualization
- Window analysis and distribution plots
- Dataset-wide comparative analysis

## Technical Specifications

- **Python Version**: 3.8+
- **Dependencies**: numpy, scipy, matplotlib, pandas, tqdm, pathlib
- **Input Format**: WESAD pickle files
- **Output Formats**: PNG plots, JSON reports, NumPy arrays, CSV
- **Processing**: Efficient handling of 15+ subjects
- **Memory**: Optimized for large dataset processing

## Testing

Run unit tests to validate functionality:

```bash
python -m pytest wesad_pipeline/tests/test_pipeline.py -v
```

Or run the built-in tests:

```bash
python wesad_pipeline/tests/test_pipeline.py
```

## Example Results

The pipeline generates:

1. **Signal Analysis**: Quality scores, heart rate estimates, peak detection
2. **Window Analysis**: Segmented data with labels, confidence scores
3. **Visualizations**: Comprehensive plots showing signal characteristics
4. **Reports**: Detailed JSON reports and human-readable summaries
5. **Exported Data**: Processed arrays ready for machine learning

## Contributing

This pipeline is part of the Shadow AI stress detection research project. The modular design allows for easy extension and customization of analysis methods.

## License

MIT License - see the repository for full license details.

## Citation

If you use this pipeline in your research, please cite the Shadow AI project and the original WESAD dataset:

```
Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). 
Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection. 
In Proceedings of the 20th ACM international conference on multimodal interaction (pp. 400-408).
```

## Support

For questions or issues:
1. Check the example usage script for demonstrations
2. Review the configuration options for customization
3. Examine the test files for expected behavior
4. Open an issue in the repository for bugs or feature requests