"""
Data Preprocessor for WESAD Analysis Pipeline

Handles data validation, cleaning, and preparation for analysis.
Since data is already resampled, skips resampling operations.

Author: Shadow AI Team
License: MIT
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from wesad_pipeline.config import WESADConfig

class WESADPreprocessor:
    """
    Data preprocessor for WESAD analysis pipeline.
    
    Handles data validation, cleaning, and preparation for analysis.
    Note: Assumes BVP and labels are already at the same sampling rate.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the preprocessor.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sampling rates (assuming data is already aligned)
        self.bvp_rate = config.dataset.bvp_sampling_rate
        self.resp_rate = config.dataset.resp_sampling_rate
        
        # Statistics tracking
        self.stats = {
            'signals_processed': 0,
            'samples_cleaned': 0,
            'artifacts_removed': 0,
            'nan_values_fixed': 0,
            'outliers_removed': 0
        }
        
        self.logger.info(f"Preprocessor initialized (data already resampled at {self.bvp_rate}Hz)")
    
    def validate_signal_data(self, bvp: np.ndarray, labels: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate BVP signal and labels data.
        
        Args:
            bvp: BVP signal array
            labels: Labels array
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for empty arrays
        if len(bvp) == 0:
            issues.append("BVP signal is empty")
        if len(labels) == 0:
            issues.append("Labels array is empty")
            
        # Check for length mismatch
        if len(bvp) != len(labels):
            issues.append(f"Length mismatch: BVP={len(bvp)}, Labels={len(labels)}")
            
        # Check for NaN values
        if np.any(np.isnan(bvp)):
            issues.append(f"BVP contains {np.sum(np.isnan(bvp))} NaN values")
            
        # Check for infinite values
        if np.any(np.isinf(bvp)):
            issues.append(f"BVP contains {np.sum(np.isinf(bvp))} infinite values")
            
        # Check label range
        unique_labels = np.unique(labels)
        if len(unique_labels) == 0:
            issues.append("No valid labels found")
        elif np.min(unique_labels) < 0 or np.max(unique_labels) > 7:
            issues.append(f"Labels out of expected range [0-7]: {unique_labels}")
            
        # Check signal variance
        if np.var(bvp) < 1e-10:
            issues.append("BVP signal has very low variance (possibly flat)")
            
        # Check for reasonable signal duration
        duration_seconds = len(bvp) / self.bvp_rate
        if duration_seconds < 10:
            issues.append(f"Signal too short: {duration_seconds:.1f}s")
        elif duration_seconds > 3600:  # 1 hour
            issues.append(f"Signal very long: {duration_seconds:.1f}s")
            
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def clean_signal_data(self, bvp: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean BVP signal and labels data.
        
        Args:
            bvp: Raw BVP signal
            labels: Raw labels
            
        Returns:
            Tuple of (cleaned_bvp, cleaned_labels)
        """
        bvp_clean = bvp.copy()
        labels_clean = labels.copy()
        
        # Handle NaN values in BVP
        nan_mask = np.isnan(bvp_clean)
        if np.any(nan_mask):
            # Interpolate NaN values
            valid_indices = np.where(~nan_mask)[0]
            if len(valid_indices) > 1:
                bvp_clean = np.interp(np.arange(len(bvp_clean)), valid_indices, bvp_clean[valid_indices])
                self.stats['nan_values_fixed'] += np.sum(nan_mask)
            else:
                # If too many NaN values, fill with median
                median_val = np.nanmedian(bvp_clean)
                bvp_clean[nan_mask] = median_val
                
        # Handle infinite values
        inf_mask = np.isinf(bvp_clean)
        if np.any(inf_mask):
            median_val = np.median(bvp_clean[~inf_mask])
            bvp_clean[inf_mask] = median_val
            
        # Remove extreme outliers (beyond 5 standard deviations)
        if len(bvp_clean) > 100:  # Only if we have enough data
            std_val = np.std(bvp_clean)
            mean_val = np.mean(bvp_clean)
            outlier_mask = np.abs(bvp_clean - mean_val) > 5 * std_val
            if np.any(outlier_mask):
                # Replace outliers with median
                median_val = np.median(bvp_clean[~outlier_mask])
                bvp_clean[outlier_mask] = median_val
                self.stats['outliers_removed'] += np.sum(outlier_mask)
                
        # Ensure labels are integers
        labels_clean = labels_clean.astype(int)
        
        # Ensure equal length (trim to shortest)
        min_length = min(len(bvp_clean), len(labels_clean))
        bvp_clean = bvp_clean[:min_length]
        labels_clean = labels_clean[:min_length]
        
        self.stats['samples_cleaned'] += 1
        
        return bvp_clean, labels_clean
    
    def generate_timestamps(self, signal_length: int, sampling_rate: Optional[int] = None) -> np.ndarray:
        """
        Generate timestamps for a signal.
        
        Args:
            signal_length: Length of the signal
            sampling_rate: Sampling rate in Hz (uses BVP rate if None)
            
        Returns:
            Array of timestamps in seconds
        """
        if sampling_rate is None:
            sampling_rate = self.bvp_rate
            
        return np.arange(signal_length) / sampling_rate
    
    def process_subject_data(self, subject_data: Dict) -> Dict:
        """
        Process data for a single subject.
        
        Args:
            subject_data: Dictionary containing BVP, labels, and metadata
            
        Returns:
            Dictionary with processed data
        """
        try:
            # Extract data (assuming already resampled)
            bvp = subject_data.get('bvp', np.array([]))
            labels = subject_data.get('labels', np.array([]))
            
            # Validate data
            is_valid, issues = self.validate_signal_data(bvp, labels)
            if not is_valid:
                self.logger.warning(f"Data validation issues: {'; '.join(issues)}")
                
            # Clean data
            bvp_clean, labels_clean = self.clean_signal_data(bvp, labels)
            
            # Generate timestamps
            timestamps = self.generate_timestamps(len(bvp_clean))
            
            # Calculate basic statistics
            duration = len(bvp_clean) / self.bvp_rate
            unique_labels = np.unique(labels_clean)
            
            processed_data = {
                'bvp': bvp_clean,
                'labels': labels_clean,
                'timestamps': timestamps,
                'sampling_rate': self.bvp_rate,
                'duration_seconds': duration,
                'unique_labels': unique_labels.tolist(),
                'data_quality': {
                    'is_valid': is_valid,
                    'issues': issues,
                    'signal_variance': float(np.var(bvp_clean)),
                    'signal_mean': float(np.mean(bvp_clean)),
                    'signal_std': float(np.std(bvp_clean)),
                    'label_distribution': {int(label): int(np.sum(labels_clean == label)) 
                                        for label in unique_labels}
                }
            }
            
            # Copy metadata if present
            for key in ['subject_id', 'quality_score', 'metadata']:
                if key in subject_data:
                    processed_data[key] = subject_data[key]
                    
            self.stats['signals_processed'] += 1
            self.logger.debug(f"Processed subject data: {duration:.1f}s, {len(unique_labels)} conditions")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Failed to process subject data: {str(e)}")
            raise
    
    def get_processing_stats(self) -> Dict:
        """
        Get preprocessing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        for key in self.stats:
            self.stats[key] = 0
        self.logger.debug("Processing statistics reset")