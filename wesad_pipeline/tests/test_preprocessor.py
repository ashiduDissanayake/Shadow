"""
Unit Tests for WESAD Preprocessor

Comprehensive tests for the WESADPreprocessor class including
data validation, cleaning, and processing functionality.

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

# Disable logging for tests
logging.getLogger().setLevel(logging.CRITICAL)

class TestWESADPreprocessor(unittest.TestCase):
    """Test WESADPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = WESADConfig()
        self.preprocessor = WESADPreprocessor(self.config)
        
        # Create test data
        self.duration_seconds = 60
        self.sampling_rate = 64
        self.signal_length = self.duration_seconds * self.sampling_rate
        
        # Valid test data
        self.valid_bvp = self._create_test_bvp(self.signal_length)
        self.valid_labels = self._create_test_labels(self.signal_length)
        
    def _create_test_bvp(self, length: int) -> np.ndarray:
        """Create realistic test BVP signal."""
        t = np.linspace(0, length/64, length)
        # Simulate heart rate around 70 BPM
        heart_rate = 70/60  # Hz
        bvp = -np.sin(2 * np.pi * heart_rate * t)
        # Add harmonics and noise
        bvp += -0.3 * np.sin(2 * np.pi * 2 * heart_rate * t)
        bvp += 0.1 * np.random.randn(length)
        return bvp
    
    def _create_test_labels(self, length: int) -> np.ndarray:
        """Create test labels with different conditions."""
        labels = np.ones(length, dtype=int)  # Baseline
        # Add stress condition in middle third
        start_stress = length // 3
        end_stress = 2 * length // 3
        labels[start_stress:end_stress] = 2  # Stress
        # Add amusement in last part
        labels[end_stress:] = 3  # Amusement
        return labels
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsInstance(self.preprocessor, WESADPreprocessor)
        self.assertEqual(self.preprocessor.bvp_rate, 64)
        self.assertEqual(self.preprocessor.resp_rate, 700)
        self.assertIsInstance(self.preprocessor.stats, dict)
        
    def test_validate_signal_data_valid(self):
        """Test validation with valid data."""
        is_valid, issues = self.preprocessor.validate_signal_data(
            self.valid_bvp, self.valid_labels
        )
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
    def test_validate_signal_data_empty(self):
        """Test validation with empty data."""
        empty_bvp = np.array([])
        empty_labels = np.array([])
        
        is_valid, issues = self.preprocessor.validate_signal_data(empty_bvp, empty_labels)
        self.assertFalse(is_valid)
        self.assertIn("BVP signal is empty", issues)
        self.assertIn("Labels array is empty", issues)
        
    def test_validate_signal_data_length_mismatch(self):
        """Test validation with length mismatch."""
        short_labels = self.valid_labels[:100]
        
        is_valid, issues = self.preprocessor.validate_signal_data(
            self.valid_bvp, short_labels
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("Length mismatch" in issue for issue in issues))
        
    def test_validate_signal_data_nan_values(self):
        """Test validation with NaN values."""
        bvp_with_nan = self.valid_bvp.copy()
        bvp_with_nan[100:105] = np.nan
        
        is_valid, issues = self.preprocessor.validate_signal_data(
            bvp_with_nan, self.valid_labels
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("NaN values" in issue for issue in issues))
        
    def test_validate_signal_data_inf_values(self):
        """Test validation with infinite values."""
        bvp_with_inf = self.valid_bvp.copy()
        bvp_with_inf[50:55] = np.inf
        
        is_valid, issues = self.preprocessor.validate_signal_data(
            bvp_with_inf, self.valid_labels
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("infinite values" in issue for issue in issues))
        
    def test_validate_signal_data_label_range(self):
        """Test validation with out-of-range labels."""
        invalid_labels = self.valid_labels.copy()
        invalid_labels[100:110] = 10  # Invalid label
        
        is_valid, issues = self.preprocessor.validate_signal_data(
            self.valid_bvp, invalid_labels
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("out of expected range" in issue for issue in issues))
        
    def test_validate_signal_data_flat_signal(self):
        """Test validation with flat signal."""
        flat_bvp = np.ones(self.signal_length) * 0.5  # Flat signal
        
        is_valid, issues = self.preprocessor.validate_signal_data(
            flat_bvp, self.valid_labels
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("very low variance" in issue for issue in issues))
        
    def test_validate_signal_data_too_short(self):
        """Test validation with too short signal."""
        short_bvp = self.valid_bvp[:100]  # Less than 10 seconds
        short_labels = self.valid_labels[:100]
        
        is_valid, issues = self.preprocessor.validate_signal_data(short_bvp, short_labels)
        self.assertFalse(is_valid)
        self.assertTrue(any("too short" in issue for issue in issues))
        
    def test_clean_signal_data_valid(self):
        """Test cleaning with valid data."""
        clean_bvp, clean_labels = self.preprocessor.clean_signal_data(
            self.valid_bvp, self.valid_labels
        )
        
        self.assertEqual(len(clean_bvp), len(clean_labels))
        self.assertFalse(np.any(np.isnan(clean_bvp)))
        self.assertFalse(np.any(np.isinf(clean_bvp)))
        self.assertEqual(clean_labels.dtype, int)
        
    def test_clean_signal_data_with_nan(self):
        """Test cleaning data with NaN values."""
        bvp_with_nan = self.valid_bvp.copy()
        nan_indices = [100, 101, 102]
        bvp_with_nan[nan_indices] = np.nan
        
        clean_bvp, clean_labels = self.preprocessor.clean_signal_data(
            bvp_with_nan, self.valid_labels
        )
        
        self.assertFalse(np.any(np.isnan(clean_bvp)))
        self.assertEqual(len(clean_bvp), len(clean_labels))
        self.assertGreater(self.preprocessor.stats['nan_values_fixed'], 0)
        
    def test_clean_signal_data_with_inf(self):
        """Test cleaning data with infinite values."""
        bvp_with_inf = self.valid_bvp.copy()
        bvp_with_inf[50:55] = np.inf
        
        clean_bvp, clean_labels = self.preprocessor.clean_signal_data(
            bvp_with_inf, self.valid_labels
        )
        
        self.assertFalse(np.any(np.isinf(clean_bvp)))
        self.assertEqual(len(clean_bvp), len(clean_labels))
        
    def test_clean_signal_data_with_outliers(self):
        """Test cleaning data with extreme outliers."""
        bvp_with_outliers = self.valid_bvp.copy()
        # Add extreme outliers
        outlier_indices = [200, 201, 202]
        bvp_with_outliers[outlier_indices] = 1000  # Extreme values
        
        clean_bvp, clean_labels = self.preprocessor.clean_signal_data(
            bvp_with_outliers, self.valid_labels
        )
        
        self.assertEqual(len(clean_bvp), len(clean_labels))
        # Check that outliers were removed
        self.assertLess(np.max(clean_bvp), 100)  # Should be much smaller now
        
    def test_clean_signal_data_length_mismatch(self):
        """Test cleaning data with different lengths."""
        short_labels = self.valid_labels[:100]
        
        clean_bvp, clean_labels = self.preprocessor.clean_signal_data(
            self.valid_bvp, short_labels
        )
        
        self.assertEqual(len(clean_bvp), len(clean_labels))
        self.assertEqual(len(clean_bvp), 100)  # Should be trimmed to shortest
        
    def test_generate_timestamps(self):
        """Test timestamp generation."""
        timestamps = self.preprocessor.generate_timestamps(self.signal_length)
        
        self.assertEqual(len(timestamps), self.signal_length)
        self.assertEqual(timestamps[0], 0.0)
        self.assertAlmostEqual(timestamps[-1], (self.signal_length - 1) / 64, places=5)
        
        # Test with custom sampling rate
        custom_rate = 100
        timestamps_custom = self.preprocessor.generate_timestamps(1000, custom_rate)
        self.assertEqual(len(timestamps_custom), 1000)
        self.assertAlmostEqual(timestamps_custom[1], 1/custom_rate, places=5)
        
    def test_process_subject_data_complete(self):
        """Test complete subject data processing."""
        subject_data = {
            'bvp': self.valid_bvp,
            'labels': self.valid_labels,
            'subject_id': 2,
            'quality_score': 0.8,
            'metadata': {'test': 'value'}
        }
        
        processed = self.preprocessor.process_subject_data(subject_data)
        
        # Check all required fields
        self.assertIn('bvp', processed)
        self.assertIn('labels', processed)
        self.assertIn('timestamps', processed)
        self.assertIn('sampling_rate', processed)
        self.assertIn('duration_seconds', processed)
        self.assertIn('unique_labels', processed)
        self.assertIn('data_quality', processed)
        
        # Check metadata preservation
        self.assertEqual(processed['subject_id'], 2)
        self.assertEqual(processed['quality_score'], 0.8)
        self.assertEqual(processed['metadata']['test'], 'value')
        
        # Check data quality info
        quality = processed['data_quality']
        self.assertIn('is_valid', quality)
        self.assertIn('issues', quality)
        self.assertIn('signal_variance', quality)
        self.assertIn('signal_mean', quality)
        self.assertIn('signal_std', quality)
        self.assertIn('label_distribution', quality)
        
        # Check dimensions
        self.assertEqual(len(processed['bvp']), len(processed['labels']))
        self.assertEqual(len(processed['bvp']), len(processed['timestamps']))
        
    def test_process_subject_data_minimal(self):
        """Test processing with minimal data."""
        subject_data = {
            'bvp': self.valid_bvp,
            'labels': self.valid_labels
        }
        
        processed = self.preprocessor.process_subject_data(subject_data)
        
        self.assertIn('bvp', processed)
        self.assertIn('labels', processed)
        self.assertIn('timestamps', processed)
        self.assertEqual(processed['sampling_rate'], 64)
        
    def test_process_subject_data_with_issues(self):
        """Test processing data with validation issues."""
        # Create problematic data
        bvp_with_issues = self.valid_bvp.copy()
        bvp_with_issues[100:110] = np.nan
        
        subject_data = {
            'bvp': bvp_with_issues,
            'labels': self.valid_labels
        }
        
        processed = self.preprocessor.process_subject_data(subject_data)
        
        # Should still process but mark issues
        self.assertIn('data_quality', processed)
        self.assertGreater(len(processed['data_quality']['issues']), 0)
        
        # Data should be cleaned
        self.assertFalse(np.any(np.isnan(processed['bvp'])))
        
    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        initial_stats = self.preprocessor.get_processing_stats()
        self.assertIsInstance(initial_stats, dict)
        
        # Process some data
        subject_data = {
            'bvp': self.valid_bvp,
            'labels': self.valid_labels
        }
        self.preprocessor.process_subject_data(subject_data)
        
        updated_stats = self.preprocessor.get_processing_stats()
        self.assertGreater(updated_stats['signals_processed'], 
                          initial_stats['signals_processed'])
        
    def test_reset_stats(self):
        """Test resetting statistics."""
        # Process some data first
        subject_data = {
            'bvp': self.valid_bvp,
            'labels': self.valid_labels
        }
        self.preprocessor.process_subject_data(subject_data)
        
        # Reset stats
        self.preprocessor.reset_stats()
        stats = self.preprocessor.get_processing_stats()
        
        for value in stats.values():
            self.assertEqual(value, 0)

class TestWESADPreprocessorEdgeCases(unittest.TestCase):
    """Test edge cases for WESADPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = WESADConfig()
        self.preprocessor = WESADPreprocessor(self.config)
        
    def test_empty_subject_data(self):
        """Test processing completely empty subject data."""
        subject_data = {}
        
        with self.assertRaises(Exception):
            self.preprocessor.process_subject_data(subject_data)
            
    def test_very_short_signal(self):
        """Test with very short signal."""
        short_bvp = np.random.randn(10)  # Very short
        short_labels = np.ones(10, dtype=int)
        
        is_valid, issues = self.preprocessor.validate_signal_data(short_bvp, short_labels)
        self.assertFalse(is_valid)
        self.assertTrue(any("too short" in issue for issue in issues))
        
    def test_all_nan_signal(self):
        """Test with signal that's all NaN."""
        nan_bvp = np.full(1000, np.nan)
        labels = np.ones(1000, dtype=int)
        
        clean_bvp, clean_labels = self.preprocessor.clean_signal_data(nan_bvp, labels)
        
        # Should be filled with a constant value
        self.assertFalse(np.any(np.isnan(clean_bvp)))
        self.assertTrue(np.all(clean_bvp == clean_bvp[0]))  # All same value
        
    def test_single_label_type(self):
        """Test with signal having only one label type."""
        bvp = np.random.randn(1000)
        single_labels = np.ones(1000, dtype=int)  # All baseline
        
        processed = self.preprocessor.process_subject_data({
            'bvp': bvp,
            'labels': single_labels
        })
        
        self.assertEqual(len(processed['unique_labels']), 1)
        self.assertEqual(processed['unique_labels'][0], 1)

if __name__ == '__main__':
    unittest.main()