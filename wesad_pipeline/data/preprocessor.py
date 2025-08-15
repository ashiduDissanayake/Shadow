"""
WESAD Data Preprocessing Module

Handles data alignment, cleaning, and validation for the WESAD analysis pipeline.
Focuses on aligning labels to BVP sampling rate and data validation.

Features:
- Align labels from 700Hz to 64Hz sampling rate
- Data validation and cleaning
- Timestamp generation
- Signal validation and quality checks

Author: Shadow AI Team
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal
import pandas as pd

from wesad_pipeline.config import WESADConfig

logger = logging.getLogger(__name__)

class WESADPreprocessor:
    """
    Data preprocessor for WESAD analysis pipeline.
    
    Handles alignment of labels to BVP sampling rate, data validation,
    and preparation for analysis.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the preprocessor.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Calculate sampling rate ratio
        self.bvp_rate = config.dataset.bvp_sampling_rate
        self.resp_rate = config.dataset.resp_sampling_rate
        self.rate_ratio = self.resp_rate / self.bvp_rate
        
        # Statistics tracking
        self.stats = {
            'signals_processed': 0,
            'labels_aligned': 0,
            'samples_cleaned': 0,
            'artifacts_removed': 0
        }
        
        self.logger.info(f"Preprocessor initialized: {self.resp_rate}Hz → {self.bvp_rate}Hz")
    
    def align_labels_to_bvp(self, labels: np.ndarray, bvp_length: int) -> np.ndarray:
        """
        Align labels from respiratory sampling rate (700Hz) to BVP sampling rate (64Hz).
        
        Args:
            labels: Original labels at 700Hz
            bvp_length: Length of BVP signal at 64Hz
            
        Returns:
            Aligned labels at 64Hz sampling rate
        """
        try:
            if len(labels) == 0:
                raise ValueError("Empty labels array")
            
            # Calculate expected aligned length
            expected_length = int(len(labels) / self.rate_ratio)
            
            # Use actual BVP length if provided and reasonable
            if bvp_length > 0 and abs(bvp_length - expected_length) < expected_length * 0.1:
                target_length = bvp_length
            else:
                target_length = expected_length
                self.logger.warning(f"BVP length mismatch: expected ~{expected_length}, got {bvp_length}")
            
            # Downsample labels using mode (most common value) in each window
            window_size = int(self.rate_ratio)
            aligned_labels = []
            
            for i in range(0, len(labels), window_size):
                window = labels[i:i + window_size]
                if len(window) > 0:
                    # Use the most common label in the window
                    unique_labels, counts = np.unique(window, return_counts=True)
                    most_common_label = unique_labels[np.argmax(counts)]
                    aligned_labels.append(most_common_label)
            
            aligned_labels = np.array(aligned_labels)
            
            # Adjust length to match target
            if len(aligned_labels) > target_length:
                aligned_labels = aligned_labels[:target_length]
            elif len(aligned_labels) < target_length:
                # Pad with the last label
                last_label = aligned_labels[-1] if len(aligned_labels) > 0 else 0
                padding = np.full(target_length - len(aligned_labels), last_label)
                aligned_labels = np.concatenate([aligned_labels, padding])
            
            self.stats['labels_aligned'] += 1
            self.logger.debug(f"Aligned labels: {len(labels)} → {len(aligned_labels)}")
            
            return aligned_labels
            
        except Exception as e:
            self.logger.error(f"Label alignment failed: {str(e)}")
            # Return zeros as fallback
            return np.zeros(bvp_length, dtype=int)
    
    def generate_timestamps(self, signal_length: int, sampling_rate: Optional[int] = None) -> np.ndarray:
        """
        Generate timestamps for a signal.
        
        Args:
            signal_length: Length of the signal
            sampling_rate: Sampling rate (Hz). If None, uses BVP sampling rate.
            
        Returns:
            Array of timestamps in seconds
        """
        if sampling_rate is None:
            sampling_rate = self.config.dataset.bvp_sampling_rate
        
        timestamps = np.arange(signal_length) / sampling_rate
        return timestamps
    
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
        
        # Check if arrays are empty
        if len(bvp) == 0:
            issues.append("BVP signal is empty")
        
        if len(labels) == 0:
            issues.append("Labels array is empty")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(bvp)):
            issues.append("BVP signal contains NaN values")
        
        if np.any(np.isinf(bvp)):
            issues.append("BVP signal contains infinite values")
        
        # Check signal characteristics
        if len(bvp) > 0:
            # Check for constant signal (likely an error)
            if np.std(bvp) < 1e-6:
                issues.append("BVP signal appears constant (very low variance)")
            
            # Check for extreme values
            signal_std = np.std(bvp)
            signal_mean = np.mean(bvp)
            outlier_threshold = 5 * signal_std
            outliers = np.sum(np.abs(bvp - signal_mean) > outlier_threshold)
            
            if outliers > len(bvp) * 0.1:  # More than 10% outliers
                issues.append(f"High number of outliers: {outliers} ({outliers/len(bvp)*100:.1f}%)")
        
        # Check label validity
        if len(labels) > 0:
            unique_labels = np.unique(labels)
            valid_labels = set(self.config.dataset.label_mapping.values())
            
            invalid_labels = [label for label in unique_labels if label not in valid_labels]
            if invalid_labels:
                issues.append(f"Invalid label values found: {invalid_labels}")
        
        # Check length consistency
        if len(bvp) > 0 and len(labels) > 0:
            if abs(len(bvp) - len(labels)) > max(len(bvp), len(labels)) * 0.1:
                issues.append(f"Significant length mismatch: BVP={len(bvp)}, Labels={len(labels)}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def clean_signal(self, bvp: np.ndarray, method: str = 'interpolation') -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean BVP signal by removing artifacts.
        
        Args:
            bvp: Input BVP signal
            method: Cleaning method ('interpolation', 'median_filter', 'none')
            
        Returns:
            Tuple of (cleaned_signal, artifact_mask)
        """
        if method == 'none' or len(bvp) == 0:
            return bvp.copy(), np.zeros(len(bvp), dtype=bool)
        
        try:
            cleaned_signal = bvp.copy()
            artifact_mask = np.zeros(len(bvp), dtype=bool)
            
            # Detect artifacts using multiple methods
            
            # 1. Statistical outliers (Z-score > 4)
            z_scores = np.abs((bvp - np.mean(bvp)) / (np.std(bvp) + 1e-8))
            outlier_mask = z_scores > 4
            artifact_mask |= outlier_mask
            
            # 2. Gradient-based spike detection
            if len(bvp) > 1:
                gradient = np.gradient(bvp)
                gradient_threshold = 4 * np.std(gradient)
                spike_mask = np.abs(gradient) > gradient_threshold
                artifact_mask |= spike_mask
            
            # 3. Moving window variance check
            if len(bvp) > 10:
                window_size = min(10, len(bvp) // 4)
                local_var = np.array([np.var(bvp[max(0, i-window_size//2):i+window_size//2+1]) 
                                    for i in range(len(bvp))])
                var_threshold = np.percentile(local_var, 95)
                var_mask = local_var > var_threshold
                artifact_mask |= var_mask
            
            # Clean the signal based on method
            if method == 'interpolation' and np.any(artifact_mask):
                # Interpolate over artifacts
                clean_indices = ~artifact_mask
                if np.sum(clean_indices) > 2:
                    valid_indices = np.where(clean_indices)[0]
                    invalid_indices = np.where(artifact_mask)[0]
                    
                    cleaned_signal[invalid_indices] = np.interp(
                        invalid_indices, 
                        valid_indices, 
                        cleaned_signal[valid_indices]
                    )
            
            elif method == 'median_filter':
                # Apply median filter
                from scipy.ndimage import median_filter
                window_size = max(3, self.config.dataset.bvp_sampling_rate // 10)  # 100ms window
                cleaned_signal = median_filter(cleaned_signal, size=window_size)
            
            self.stats['artifacts_removed'] += np.sum(artifact_mask)
            self.stats['samples_cleaned'] += 1
            
            self.logger.debug(f"Cleaned signal: {np.sum(artifact_mask)} artifacts removed")
            
            return cleaned_signal, artifact_mask
            
        except Exception as e:
            self.logger.error(f"Signal cleaning failed: {str(e)}")
            return bvp.copy(), np.zeros(len(bvp), dtype=bool)
    
    def process_subject_data(self, subject_data: Dict) -> Dict:
        """
        Process complete subject data including alignment and cleaning.
        
        Args:
            subject_data: Raw subject data dictionary
            
        Returns:
            Processed subject data dictionary
        """
        try:
            self.logger.debug("Processing subject data")
            
            # Extract raw data
            raw_bvp = subject_data.get('bvp', np.array([]))
            raw_labels = subject_data.get('labels', np.array([]))
            
            # Validate input data
            is_valid, issues = self.validate_signal_data(raw_bvp, raw_labels)
            if not is_valid:
                self.logger.warning(f"Data validation issues: {'; '.join(issues)}")
            
            # Clean BVP signal
            cleaned_bvp, artifact_mask = self.clean_signal(raw_bvp, method='interpolation')
            
            # Align labels to BVP sampling rate
            aligned_labels = self.align_labels_to_bvp(raw_labels, len(cleaned_bvp))
            
            # Generate timestamps
            timestamps = self.generate_timestamps(len(cleaned_bvp))
            
            # Create processed data dictionary
            processed_data = {
                'bvp': cleaned_bvp,
                'labels': aligned_labels,
                'timestamps': timestamps,
                'artifact_mask': artifact_mask,
                'processing_info': {
                    'original_length': len(raw_bvp),
                    'processed_length': len(cleaned_bvp),
                    'labels_original_length': len(raw_labels),
                    'labels_aligned_length': len(aligned_labels),
                    'artifacts_detected': np.sum(artifact_mask),
                    'sampling_rate': self.config.dataset.bvp_sampling_rate,
                    'validation_issues': issues
                },
                'quality_score': subject_data.get('quality_score', 0.0)
            }
            
            # Copy over any additional fields
            for key, value in subject_data.items():
                if key not in processed_data:
                    processed_data[key] = value
            
            self.stats['signals_processed'] += 1
            self.logger.debug("Subject data processing completed")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Subject data processing failed: {str(e)}")
            raise
    
    def process_multiple_subjects(self, subjects_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Process data for multiple subjects.
        
        Args:
            subjects_data: Dictionary mapping subject IDs to their raw data
            
        Returns:
            Dictionary mapping subject IDs to their processed data
        """
        processed_data = {}
        
        self.logger.info(f"Processing data for {len(subjects_data)} subjects")
        
        for subject_id, subject_data in subjects_data.items():
            try:
                self.logger.debug(f"Processing subject {subject_id}")
                processed_data[subject_id] = self.process_subject_data(subject_data)
            except Exception as e:
                self.logger.error(f"Failed to process subject {subject_id}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully processed {len(processed_data)} subjects")
        return processed_data
    
    def get_processing_statistics(self) -> Dict:
        """Get preprocessing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset preprocessing statistics."""
        self.stats = {
            'signals_processed': 0,
            'labels_aligned': 0,
            'samples_cleaned': 0,
            'artifacts_removed': 0
        }
        self.logger.debug("Preprocessing statistics reset")