"""
Unit Tests for WESAD Analysis Pipeline

Basic unit tests for pipeline components to ensure functionality
and validate core operations.

Author: Shadow AI Team
License: MIT
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging
import sys

# Add project path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wesad_pipeline.config import WESADConfig
from wesad_pipeline.data import WESADPreprocessor
from wesad_pipeline.analysis import SignalQuality, HeartRateAnalyzer, WindowAnalyzer
from wesad_pipeline.utils import WESADHelpers
from wesad_pipeline.main import WESADPipeline

# Disable logging for tests
logging.getLogger().setLevel(logging.CRITICAL)

class TestWESADConfig(unittest.TestCase):
    """Test configuration module."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = WESADConfig()
        
        # Check basic properties
        self.assertIsInstance(config.dataset.subjects, list)
        self.assertGreater(len(config.dataset.subjects), 0)
        self.assertEqual(config.dataset.bvp_sampling_rate, 64)
        self.assertEqual(config.analysis.window_size_seconds, 60)
        self.assertEqual(config.analysis.overlap_seconds, 5)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = WESADConfig()
        
        # Should pass validation
        self.assertTrue(config.validate())
        
        # Test invalid configuration
        config.analysis.window_size_seconds = -1
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_label_mapping(self):
        """Test label mapping functionality."""
        config = WESADConfig()
        
        # Test label name lookup
        self.assertEqual(config.get_label_name(1), 'baseline')
        self.assertEqual(config.get_label_name(2), 'stress')
        
        # Test label ID lookup
        self.assertEqual(config.get_label_id('baseline'), 1)
        self.assertEqual(config.get_label_id('stress'), 2)


class TestWESADPreprocessor(unittest.TestCase):
    """Test data preprocessing module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = WESADConfig()
        self.preprocessor = WESADPreprocessor(self.config)
    
    def test_label_alignment(self):
        """Test label alignment from 700Hz to 64Hz."""
        # Create test labels (700Hz)
        labels_700hz = np.array([1] * 700 + [2] * 700 + [3] * 700)  # 3 seconds
        bvp_length_64hz = int(len(labels_700hz) / (700/64))  # Expected BVP length at 64Hz
        
        aligned_labels = self.preprocessor.align_labels_to_bvp(labels_700hz, bvp_length_64hz)
        
        # Check output length
        self.assertEqual(len(aligned_labels), bvp_length_64hz)
        
        # Check that labels are preserved
        self.assertIn(1, aligned_labels)
        self.assertIn(2, aligned_labels)
        self.assertIn(3, aligned_labels)
    
    def test_signal_validation(self):
        """Test signal validation."""
        # Valid signal
        bvp = np.random.randn(1000)
        labels = np.random.randint(0, 4, 100)
        
        is_valid, issues = self.preprocessor.validate_signal_data(bvp, labels)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
        # Invalid signal (contains NaN)
        bvp_invalid = bvp.copy()
        bvp_invalid[0] = np.nan
        
        is_valid, issues = self.preprocessor.validate_signal_data(bvp_invalid, labels)
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
    
    def test_signal_cleaning(self):
        """Test signal cleaning functionality."""
        # Create signal with artifacts
        signal = np.random.randn(1000)
        signal[100] = 100  # Add spike artifact
        signal[200] = -100  # Add negative spike
        
        cleaned_signal, artifact_mask = self.preprocessor.clean_signal(signal, method='interpolation')
        
        # Check that artifacts were detected
        self.assertTrue(np.any(artifact_mask))
        
        # Check that cleaned signal is different from original
        self.assertFalse(np.array_equal(signal, cleaned_signal))
        
        # Check that extreme values were reduced
        self.assertLess(np.max(cleaned_signal), np.max(signal))


class TestSignalQuality(unittest.TestCase):
    """Test signal quality assessment module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = WESADConfig()
        self.signal_quality = SignalQuality(self.config)
    
    def test_quality_assessment(self):
        """Test signal quality assessment."""
        # Create a good quality BVP-like signal
        t = np.linspace(0, 10, 640)  # 10 seconds at 64Hz
        freq = 1.2  # 1.2 Hz (72 BPM)
        bvp_signal = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        
        quality_result = self.signal_quality.assess_signal_quality(bvp_signal)
        
        # Check result structure
        self.assertIn('overall_score', quality_result)
        self.assertIn('metrics', quality_result)
        self.assertIn('quality_level', quality_result)
        
        # Quality score should be between 0 and 1
        self.assertGreaterEqual(quality_result['overall_score'], 0.0)
        self.assertLessEqual(quality_result['overall_score'], 1.0)
    
    def test_windowed_quality_assessment(self):
        """Test windowed quality assessment."""
        # Create test signal
        bvp_signal = np.random.randn(2000)  # ~31 seconds at 64Hz
        
        windowed_result = self.signal_quality.assess_windowed_quality(bvp_signal, window_length=640)
        
        # Check result structure
        self.assertIn('window_scores', windowed_result)
        self.assertIn('avg_quality', windowed_result)
        
        # Should have multiple windows
        self.assertGreater(len(windowed_result['window_scores']), 1)
    
    def test_quality_threshold_validation(self):
        """Test quality threshold validation."""
        # High quality signal
        high_quality_signal = np.sin(np.linspace(0, 4*np.pi, 1000)) + 0.05 * np.random.randn(1000)
        
        # Low quality signal (mostly noise)
        low_quality_signal = 10 * np.random.randn(1000)
        
        high_quality_valid = self.signal_quality.validate_quality_threshold(high_quality_signal, threshold=0.3)
        low_quality_valid = self.signal_quality.validate_quality_threshold(low_quality_signal, threshold=0.7)
        
        # High quality signal should pass low threshold
        self.assertTrue(high_quality_valid)
        
        # Low quality signal should not pass high threshold
        self.assertFalse(low_quality_valid)


class TestHeartRateAnalyzer(unittest.TestCase):
    """Test heart rate analysis module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = WESADConfig()
        self.hr_analyzer = HeartRateAnalyzer(self.config)
    
    def test_peak_detection(self):
        """Test BVP peak detection."""
        # Create synthetic BVP signal with known peaks
        t = np.linspace(0, 10, 640)  # 10 seconds at 64Hz
        freq = 1.2  # 1.2 Hz (72 BPM)
        bvp_signal = -np.sin(2 * np.pi * freq * t)  # Negative for BVP peaks
        
        peaks, detection_info = self.hr_analyzer.detect_peaks(bvp_signal, method='adaptive')
        
        # Should detect some peaks
        self.assertGreater(len(peaks), 0)
        
        # Check detection info
        self.assertIn('method', detection_info)
        self.assertEqual(detection_info['method'], 'adaptive')
    
    def test_heart_rate_estimation(self):
        """Test heart rate estimation."""
        # Create synthetic BVP signal
        t = np.linspace(0, 30, 1920)  # 30 seconds at 64Hz
        freq = 1.2  # 1.2 Hz (72 BPM)
        bvp_signal = -np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        
        hr_result = self.hr_analyzer.estimate_heart_rate(bvp_signal)
        
        # Check result structure
        self.assertIn('mean_hr', hr_result)
        self.assertIn('valid_estimates', hr_result)
        self.assertIn('peak_positions', hr_result)
        
        # Should have valid estimates
        self.assertGreater(hr_result['valid_estimates'], 0)
        
        # Heart rate should be in reasonable range
        if hr_result['mean_hr'] > 0:
            self.assertGreaterEqual(hr_result['mean_hr'], self.config.analysis.min_heart_rate)
            self.assertLessEqual(hr_result['mean_hr'], self.config.analysis.max_heart_rate)
    
    def test_heart_rate_validation(self):
        """Test heart rate validation."""
        # Valid heart rates
        self.assertTrue(self.hr_analyzer.validate_heart_rate(70))
        self.assertTrue(self.hr_analyzer.validate_heart_rate(100))
        
        # Invalid heart rates
        self.assertFalse(self.hr_analyzer.validate_heart_rate(20))
        self.assertFalse(self.hr_analyzer.validate_heart_rate(250))


class TestWindowAnalyzer(unittest.TestCase):
    """Test windowing analysis module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = WESADConfig()
        self.window_analyzer = WindowAnalyzer(self.config)
    
    def test_window_creation(self):
        """Test window creation."""
        # Create test data (5 minutes at 64Hz)
        signal_length = 5 * 60 * 64
        bvp_signal = np.random.randn(signal_length)
        labels = np.random.randint(1, 4, signal_length)
        
        windows_result = self.window_analyzer.create_windows(bvp_signal, labels)
        
        # Check result structure
        self.assertIn('windows', windows_result)
        self.assertIn('metadata', windows_result)
        
        # Should create some windows
        self.assertGreater(len(windows_result['windows']), 0)
        
        # Check window structure
        first_window = windows_result['windows'][0]
        self.assertIn('bvp', first_window)
        self.assertIn('label', first_window)
        self.assertIn('quality', first_window)
        self.assertIn('confidence', first_window)
    
    def test_window_label_calculation(self):
        """Test window label calculation."""
        # Create labels with clear majority
        window_labels = np.array([1, 1, 1, 2, 2])  # Majority is 1
        
        label, confidence = self.window_analyzer._calculate_window_label(window_labels)
        
        self.assertEqual(label, 1)
        self.assertEqual(confidence, 0.6)  # 3/5 = 0.6
    
    def test_feature_extraction(self):
        """Test window feature extraction."""
        # Create simple windows
        bvp_signal = np.random.randn(1000)
        labels = np.ones(1000)
        
        windows_result = self.window_analyzer.create_windows(bvp_signal, labels)
        
        if windows_result['windows']:
            features_result = self.window_analyzer.extract_window_features(windows_result)
            
            # Check feature result structure
            self.assertIn('features', features_result)
            self.assertIn('labels', features_result)
            self.assertIn('feature_names', features_result)
            
            # Should have features
            self.assertGreater(len(features_result['feature_names']), 0)


class TestWESADHelpers(unittest.TestCase):
    """Test utilities helpers module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = WESADConfig()
        self.helpers = WESADHelpers(self.config)
    
    def test_safe_label_lookup(self):
        """Test safe label lookup."""
        # Valid lookups
        self.assertEqual(self.helpers.safe_label_lookup(1), 'baseline')
        self.assertEqual(self.helpers.safe_label_lookup(2), 'stress')
        
        # Invalid lookups
        self.assertEqual(self.helpers.safe_label_lookup(999), 'unknown')
        self.assertEqual(self.helpers.safe_label_lookup(np.nan), 'unknown')
    
    def test_array_validation(self):
        """Test array validation."""
        # Valid array
        valid_array = np.array([1, 2, 3, 4, 5])
        is_valid, issues = self.helpers.validate_array_data(valid_array)
        self.assertTrue(is_valid)
        
        # Invalid array (contains NaN)
        invalid_array = np.array([1, 2, np.nan, 4, 5])
        is_valid, issues = self.helpers.validate_array_data(invalid_array)
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
    
    def test_basic_statistics(self):
        """Test basic statistics calculation."""
        data = np.array([1, 2, 3, 4, 5])
        stats = self.helpers.calculate_basic_statistics(data)
        
        # Check required statistics
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        
        # Check values
        self.assertEqual(stats['mean'], 3.0)
        self.assertEqual(stats['min'], 1.0)
        self.assertEqual(stats['max'], 5.0)
    
    def test_data_io(self):
        """Test data input/output operations."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test data
            test_data = np.array([1, 2, 3, 4, 5])
            
            # Test save/load numpy
            save_success = self.helpers.save_array_data(test_data, temp_path / "test.npy", format='npy')
            self.assertTrue(save_success)
            
            loaded_data = self.helpers.load_array_data(temp_path / "test.npy", format='npy')
            np.testing.assert_array_equal(test_data, loaded_data)


class TestWESADPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary output directory
        self.temp_dir = tempfile.mkdtemp()
        self.config = WESADConfig()
        self.config.output.output_path = self.temp_dir
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = WESADPipeline(
            output_path=self.temp_dir,
            subjects=[2, 3],
            config=self.config,
            log_level="ERROR"  # Suppress logging
        )
        
        # Check that components are initialized
        self.assertIsNotNone(pipeline.data_loader)
        self.assertIsNotNone(pipeline.preprocessor)
        self.assertIsNotNone(pipeline.signal_quality)
        self.assertIsNotNone(pipeline.heart_rate_analyzer)
        self.assertIsNotNone(pipeline.window_analyzer)
    
    def test_pipeline_statistics(self):
        """Test pipeline statistics collection."""
        pipeline = WESADPipeline(
            output_path=self.temp_dir,
            subjects=[2, 3],
            config=self.config,
            log_level="ERROR"
        )
        
        stats = pipeline.get_pipeline_statistics()
        
        # Check stats structure
        self.assertIn('subjects_processed', stats)
        self.assertIn('component_statistics', stats)
        self.assertIn('data_loader', stats['component_statistics'])


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)