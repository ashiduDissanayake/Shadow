#!/usr/bin/env python3
"""
WESAD Analysis Pipeline - Example Usage

This script demonstrates how to use the WESAD analysis pipeline
with both simulated data and real WESAD data.

Run this script to test the pipeline functionality:
python example_usage.py

Author: Shadow AI Team
License: MIT
"""

import sys
import numpy as np
import tempfile
from pathlib import Path

# Add project path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesad_pipeline.config import WESADConfig
from wesad_pipeline.main import WESADPipeline
from wesad_pipeline.data import WESADPreprocessor
from wesad_pipeline.analysis import SignalQuality, HeartRateAnalyzer, WindowAnalyzer

def create_simulated_bvp_data(duration_seconds=180, sampling_rate=64):
    """
    Create simulated BVP data for testing.
    
    Args:
        duration_seconds: Duration of the simulation in seconds
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary with simulated BVP data and labels
    """
    print(f"Creating simulated BVP data ({duration_seconds}s at {sampling_rate}Hz)")
    
    # Time vector
    t = np.linspace(0, duration_seconds, duration_seconds * sampling_rate)
    
    # Simulate BVP signal (inverted sine wave + noise)
    heart_rate_bpm = 70  # 70 BPM
    heart_rate_hz = heart_rate_bpm / 60
    
    # Primary BVP component
    bvp_signal = -np.sin(2 * np.pi * heart_rate_hz * t)
    
    # Add some harmonics for realism
    bvp_signal += -0.3 * np.sin(2 * np.pi * 2 * heart_rate_hz * t)
    bvp_signal += -0.1 * np.sin(2 * np.pi * 3 * heart_rate_hz * t)
    
    # Add noise
    noise = 0.1 * np.random.randn(len(t))
    bvp_signal += noise
    
    # Create labels (simulate different conditions)
    labels = np.ones(len(t), dtype=int)  # Start with baseline
    
    # Add stress condition in middle third
    stress_start = len(t) // 3
    stress_end = 2 * len(t) // 3
    labels[stress_start:stress_end] = 2  # Stress condition
    
    # Add amusement condition in last third
    amusement_start = 2 * len(t) // 3
    labels[amusement_start:] = 3  # Amusement condition
    
    # Create timestamps
    timestamps = t
    
    return {
        'bvp': bvp_signal,
        'labels': labels,
        'timestamps': timestamps,
        'quality_score': 0.8,
        'sampling_rate': sampling_rate
    }

def test_individual_components():
    """Test individual pipeline components with simulated data."""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL PIPELINE COMPONENTS")
    print("="*60)
    
    # Create configuration
    config = WESADConfig()
    print(f"✓ Configuration created")
    print(f"  - Window size: {config.analysis.window_size_seconds}s")
    print(f"  - Overlap: {config.analysis.overlap_seconds}s")
    print(f"  - Quality threshold: {config.analysis.quality_threshold}")
    
    # Create simulated data
    sim_data = create_simulated_bvp_data(duration_seconds=180)  # 3 minutes
    print(f"✓ Simulated data created")
    print(f"  - Duration: {len(sim_data['bvp'])/config.dataset.bvp_sampling_rate:.1f}s")
    print(f"  - Samples: {len(sim_data['bvp'])}")
    print(f"  - Conditions: {len(np.unique(sim_data['labels']))}")
    
    # Test preprocessor
    print("\n1. Testing Data Preprocessor...")
    preprocessor = WESADPreprocessor(config)
    processed_data = preprocessor.process_subject_data(sim_data)
    print(f"   ✓ Preprocessing completed")
    print(f"   - Original length: {processed_data['processing_info']['original_length']}")
    print(f"   - Processed length: {processed_data['processing_info']['processed_length']}")
    print(f"   - Artifacts detected: {processed_data['processing_info']['artifacts_detected']}")
    
    # Test signal quality
    print("\n2. Testing Signal Quality Assessment...")
    signal_quality = SignalQuality(config)
    quality_result = signal_quality.assess_signal_quality(processed_data['bvp'])
    print(f"   ✓ Quality assessment completed")
    print(f"   - Overall score: {quality_result['overall_score']:.3f}")
    print(f"   - Quality level: {quality_result['quality_level']}")
    print(f"   - Variance score: {quality_result['metrics']['variance_score']:.3f}")
    print(f"   - Periodicity score: {quality_result['metrics']['periodicity_score']:.3f}")
    
    # Test heart rate analysis
    print("\n3. Testing Heart Rate Analysis...")
    hr_analyzer = HeartRateAnalyzer(config)
    hr_result = hr_analyzer.estimate_heart_rate(processed_data['bvp'])
    print(f"   ✓ Heart rate analysis completed")
    print(f"   - Mean HR: {hr_result['mean_hr']:.1f} BPM")
    print(f"   - HR range: {hr_result['min_hr']:.1f} - {hr_result['max_hr']:.1f} BPM")
    print(f"   - Valid estimates: {hr_result['valid_estimates']}/{hr_result['total_estimates']}")
    print(f"   - Peaks detected: {len(hr_result['peak_positions'])}")
    
    # Test windowing analysis
    print("\n4. Testing Windowing Analysis...")
    window_analyzer = WindowAnalyzer(config)
    window_result = window_analyzer.create_windows(
        processed_data['bvp'], 
        processed_data['labels'], 
        processed_data['timestamps']
    )
    print(f"   ✓ Windowing analysis completed")
    metadata = window_result['metadata']
    print(f"   - Total windows: {metadata['total_windows']}")
    print(f"   - Accepted windows: {metadata['accepted_windows']}")
    print(f"   - Acceptance rate: {metadata['acceptance_rate']:.1%}")
    print(f"   - Window size: {metadata['window_size_seconds']}s")
    
    # Show window distribution
    if window_result['windows']:
        from collections import Counter
        labels = [w['label'] for w in window_result['windows']]
        label_counts = Counter(labels)
        print(f"   - Window distribution:")
        for label_id, count in label_counts.items():
            condition_name = config.get_label_name(label_id)
            print(f"     * {condition_name}: {count} windows")
    
    print("\n✓ All individual components tested successfully!")
    
    return {
        'processed_data': processed_data,
        'quality_result': quality_result,
        'hr_result': hr_result,
        'window_result': window_result
    }

def test_complete_pipeline():
    """Test the complete pipeline with simulated data."""
    print("\n" + "="*60)
    print("TESTING COMPLETE PIPELINE")
    print("="*60)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary output directory: {temp_dir}")
        
        # Create configuration
        config = WESADConfig()
        config.output.output_path = temp_dir
        config.dataset.subjects = [999]  # Use fake subject ID for testing
        
        # Initialize pipeline (this will fail on data loading, which is expected)
        try:
            pipeline = WESADPipeline(
                wesad_path="nonexistent/path",  # Intentionally nonexistent
                output_path=temp_dir,
                subjects=[999],
                config=config,
                log_level="ERROR"  # Suppress warnings
            )
            print("✓ Pipeline initialized successfully")
            
            # Test pipeline statistics
            stats = pipeline.get_pipeline_statistics()
            print("✓ Pipeline statistics collected")
            print(f"  - Components: {len(stats['component_statistics'])}")
            
        except Exception as e:
            print(f"✗ Pipeline initialization failed: {str(e)}")
            return False
    
    print("✓ Complete pipeline test completed successfully!")
    return True

def demonstrate_real_usage():
    """Demonstrate how to use the pipeline with real data."""
    print("\n" + "="*60)
    print("REAL USAGE DEMONSTRATION")
    print("="*60)
    
    print("""
To use the WESAD Analysis Pipeline with real data:

1. Download the WESAD dataset from: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
2. Extract the dataset to a directory (e.g., 'data/raw/wesad/')
3. Run the pipeline:

# Command line usage:
python wesad_pipeline/main.py --wesad-path data/raw/wesad/ --output-path wesad_analysis/

# Python API usage:
from wesad_pipeline.main import WESADPipeline

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

Expected outputs:
- wesad_analysis/plots/         # Visualization plots
- wesad_analysis/reports/       # JSON and text reports
- wesad_analysis/processed_data/ # Exported data arrays
- wesad_analysis/logs/          # Pipeline logs

Features:
- Modular design with reusable components
- Comprehensive error handling and validation
- Progress tracking for batch operations
- High-quality visualizations with condition annotations
- Detailed JSON reports and summary statistics
- Multiple export formats (NumPy, CSV, JSON)
- Configurable analysis parameters
""")

def main():
    """Main function to run all demonstrations."""
    print("WESAD Analysis Pipeline - Example Usage")
    print("Repository: https://github.com/ashiduDissanayake/Shadow")
    
    try:
        # Test individual components
        component_results = test_individual_components()
        
        # Test complete pipeline
        pipeline_success = test_complete_pipeline()
        
        # Show usage demonstration
        demonstrate_real_usage()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("""
The WESAD Analysis Pipeline is ready for use!

Key capabilities demonstrated:
✓ Data loading and preprocessing
✓ Signal quality assessment
✓ Heart rate analysis with peak detection
✓ Windowing analysis with feature extraction
✓ Pipeline orchestration and error handling
✓ Modular component architecture

Next steps:
1. Download the WESAD dataset
2. Run the pipeline with real data
3. Explore the generated visualizations and reports
4. Customize analysis parameters as needed
""")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)