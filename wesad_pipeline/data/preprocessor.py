"""
Data Preprocessor for WESAD Analysis Pipeline

Handles data validation, cleaning, frequency filtering, and preparation for analysis.
Now includes Butterworth bandpass filtering for BVP signals.

Author: Shadow AI Team
License: MIT
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from scipy import signal
import warnings

from wesad_pipeline.config import WESADConfig

class WESADPreprocessor:
    """
    Data preprocessor for WESAD analysis pipeline.
    
    Handles data validation, cleaning, frequency filtering, and preparation for analysis.
    Now includes Butterworth bandpass filtering for optimal BVP signal quality.
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
        
        # Butterworth filter parameters for BVP
        self.filter_order = 4  # 4th order Butterworth (good balance)
        self.low_cutoff = 0.5   # Hz (30 BPM minimum heart rate)
        self.high_cutoff = 4.0  # Hz (240 BPM maximum heart rate)
        
        # Design the Butterworth bandpass filter
        self._design_bvp_filter()
        
        # Statistics tracking
        self.stats = {
            'signals_processed': 0,
            'signals_filtered': 0,
            'samples_cleaned': 0,
            'artifacts_removed': 0,
            'nan_values_fixed': 0,
            'outliers_removed': 0,
            'filter_artifacts_removed': 0
        }
        
        self.logger.info(f"Preprocessor initialized with Butterworth bandpass filter "
                        f"({self.low_cutoff}-{self.high_cutoff} Hz, order {self.filter_order})")
    
    def _design_bvp_filter(self):
        """
        Design Butterworth bandpass filter for BVP signals.
        
        Filter specifications:
        - Type: Butterworth (maximally flat passband)
        - Order: 4 (good balance of performance vs. artifacts)
        - Passband: 0.5-4.0 Hz (30-240 BPM heart rate range)
        """
        try:
            # Calculate normalized frequencies
            nyquist = self.bvp_rate / 2
            low_norm = self.low_cutoff / nyquist
            high_norm = self.high_cutoff / nyquist
            
            # Validate frequency range
            if low_norm >= 1.0 or high_norm >= 1.0:
                raise ValueError(f"Filter cutoff frequencies too high for sampling rate {self.bvp_rate}Hz")
            
            # Design Butterworth bandpass filter
            self.filter_b, self.filter_a = signal.butter(
                self.filter_order, 
                [low_norm, high_norm], 
                btype='band',
                analog=False
            )
            
            self.logger.info(f"Butterworth filter designed: {self.low_cutoff}-{self.high_cutoff} Hz, "
                           f"order {self.filter_order}")
            
        except Exception as e:
            self.logger.error(f"Failed to design Butterworth filter: {str(e)}")
            # Fallback: no filtering
            self.filter_b, self.filter_a = None, None
    
    def apply_bvp_filter(self, bvp_signal: np.ndarray) -> np.ndarray:
        """
        Apply Butterworth bandpass filter to BVP signal.
        
        Args:
            bvp_signal: Raw BVP signal array
            
        Returns:
            Filtered BVP signal array
        """
        try:
            if self.filter_b is None or self.filter_a is None:
                self.logger.warning("No filter available, returning original signal")
                return bvp_signal
            
            if len(bvp_signal) < 3 * self.filter_order:
                self.logger.warning(f"Signal too short for filtering: {len(bvp_signal)} samples")
                return bvp_signal
            
            # Apply zero-phase filtering (filtfilt) to avoid phase distortion
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                filtered_signal = signal.filtfilt(
                    self.filter_b, self.filter_a, bvp_signal, 
                    method='gust'  # Gustafsson's method for better edge handling
                )
            
            # Handle any NaN values that might result from filtering
            if np.any(np.isnan(filtered_signal)):
                self.logger.warning("Filter introduced NaN values, using original signal")
                return bvp_signal
            
            self.stats['signals_filtered'] += 1
            return filtered_signal.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"BVP filtering failed: {str(e)}")
            return bvp_signal
    
    def remove_filter_artifacts(self, filtered_signal: np.ndarray, 
                              artifact_duration_sec: float = 2.0) -> np.ndarray:
        """
        Remove potential filter artifacts from signal edges.
        
        Args:
            filtered_signal: Filtered BVP signal
            artifact_duration_sec: Duration of edge artifacts to remove (seconds)
            
        Returns:
            Signal with edge artifacts removed
        """
        try:
            artifact_samples = int(artifact_duration_sec * self.bvp_rate)
            
            if len(filtered_signal) <= 2 * artifact_samples:
                self.logger.warning("Signal too short for artifact removal")
                return filtered_signal
            
            # Remove edge artifacts
            clean_signal = filtered_signal[artifact_samples:-artifact_samples]
            self.stats['filter_artifacts_removed'] += 1
            
            return clean_signal
            
        except Exception as e:
            self.logger.error(f"Artifact removal failed: {str(e)}")
            return filtered_signal
    
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
            
        # Check minimum length for filtering
        min_samples_for_filter = 3 * self.filter_order
        if len(bvp) < min_samples_for_filter:
            issues.append(f"Signal too short for Butterworth filtering: {len(bvp)} < {min_samples_for_filter}")
            
        is_valid = len(issues) == 0
        return is_valid, issues

    def clean_signal_artifacts(self, bvp: np.ndarray) -> np.ndarray:
        """
        Clean obvious artifacts from BVP signal.
        
        Args:
            bvp: BVP signal array
            
        Returns:
            Cleaned BVP signal
        """
        cleaned_bvp = bvp.copy()
        original_length = len(cleaned_bvp)
        
        # Fix NaN values
        nan_mask = np.isnan(cleaned_bvp)
        if np.any(nan_mask):
            # Simple interpolation for small gaps
            if np.sum(nan_mask) < len(cleaned_bvp) * 0.1:  # Less than 10% NaN
                cleaned_bvp = self._interpolate_nan_values(cleaned_bvp)
                self.stats['nan_values_fixed'] += np.sum(nan_mask)
            else:
                self.logger.warning(f"Too many NaN values: {np.sum(nan_mask)}/{len(cleaned_bvp)}")
        
        # Remove extreme outliers (beyond physiological range)
        # BVP values typically range from -1 to 1 (normalized)
        outlier_threshold = 5 * np.std(cleaned_bvp)
        outlier_mask = np.abs(cleaned_bvp - np.mean(cleaned_bvp)) > outlier_threshold
        
        if np.any(outlier_mask):
            # Clip outliers instead of removing to preserve timing
            median_val = np.median(cleaned_bvp)
            cleaned_bvp[outlier_mask] = median_val
            self.stats['outliers_removed'] += np.sum(outlier_mask)
        
        self.stats['samples_cleaned'] += original_length
        return cleaned_bvp

    def preprocess_subject(self, subject_data: Dict) -> Dict:
        """
        Preprocess data for a single subject with Butterworth filtering.
        
        Args:
            subject_data: Raw subject data dictionary
            
        Returns:
            Dictionary containing preprocessed data
        """
        try:
            self.logger.debug("Starting subject preprocessing with filtering")
            
            # Extract BVP and labels
            bvp_signal = subject_data.get('signal', {}).get('wrist', {}).get('BVP', np.array([]))
            labels = subject_data.get('label', np.array([]))
            
            if len(bvp_signal) == 0 or len(labels) == 0:
                return {'error': 'Missing BVP signal or labels'}
            
            # Validate data
            is_valid, issues = self.validate_signal_data(bvp_signal, labels)
            if not is_valid:
                self.logger.warning(f"Data validation failed: {issues}")
                return {'error': f'Validation failed: {issues}'}
            
            # Step 1: Clean obvious artifacts
            cleaned_bvp = self.clean_signal_artifacts(bvp_signal)
            
            # Step 2: Apply Butterworth bandpass filter
            filtered_bvp = self.apply_bvp_filter(cleaned_bvp)
            
            # Step 3: Remove potential filter edge artifacts
            final_bvp = self.remove_filter_artifacts(filtered_bvp)
            
            # Adjust labels to match filtered signal length
            samples_removed = len(bvp_signal) - len(final_bvp)
            if samples_removed > 0:
                edge_samples = samples_removed // 2
                final_labels = labels[edge_samples:edge_samples + len(final_bvp)]
            else:
                final_labels = labels[:len(final_bvp)]
            
            # Create result
            processed_data = {
                'bvp': final_bvp,
                'labels': final_labels,
                'original_length': len(bvp_signal),
                'processed_length': len(final_bvp),
                'sampling_rate': self.bvp_rate,
                'filter_applied': True,
                'filter_specs': {
                    'type': 'butterworth_bandpass',
                    'order': self.filter_order,
                    'low_cutoff': self.low_cutoff,
                    'high_cutoff': self.high_cutoff,
                    'passband': f"{self.low_cutoff}-{self.high_cutoff} Hz"
                }
            }
            
            # Add other signals if available
            other_signals = subject_data.get('signal', {}).get('wrist', {})
            for signal_name, signal_data in other_signals.items():
                if signal_name != 'BVP' and len(signal_data) > 0:
                    processed_data[signal_name.lower()] = signal_data
            
            self.stats['signals_processed'] += 1
            
            self.logger.debug(f"Subject preprocessing completed: "
                            f"{len(bvp_signal)} → {len(final_bvp)} samples")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Subject preprocessing failed: {str(e)}")
            return {'error': str(e)}

    def _interpolate_nan_values(self, signal: np.ndarray) -> np.ndarray:
        """Interpolate NaN values in signal."""
        mask = ~np.isnan(signal)
        if np.sum(mask) < 2:
            return signal  # Can't interpolate with less than 2 valid points
        
        indices = np.arange(len(signal))
        signal[~mask] = np.interp(indices[~mask], indices[mask], signal[mask])
        return signal

    def get_filter_frequency_response(self, frequencies: Optional[np.ndarray] = None) -> Dict:
        """
        Get frequency response of the Butterworth filter.
        
        Args:
            frequencies: Optional frequency array for response calculation
            
        Returns:
            Dictionary containing frequency response data
        """
        if self.filter_b is None or self.filter_a is None:
            return {}
        
        if frequencies is None:
            frequencies = np.logspace(-1, 1, 1000)  # 0.1 to 10 Hz
        
        # Calculate frequency response
        w, h = signal.freqs(self.filter_b, self.filter_a, worN=frequencies * 2 * np.pi)
        
        return {
            'frequencies': frequencies,
            'magnitude': np.abs(h),
            'phase': np.angle(h),
            'magnitude_db': 20 * np.log10(np.abs(h) + 1e-10),
            'filter_specs': {
                'type': 'butterworth_bandpass',
                'order': self.filter_order,
                'passband': f"{self.low_cutoff}-{self.high_cutoff} Hz"
            }
        }

    def get_preprocessing_stats(self) -> Dict:
        """Get preprocessing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset preprocessing statistics."""
        self.stats = {
            'signals_processed': 0,
            'signals_filtered': 0,
            'samples_cleaned': 0,
            'artifacts_removed': 0,
            'nan_values_fixed': 0,
            'outliers_removed': 0,
            'filter_artifacts_removed': 0
        }