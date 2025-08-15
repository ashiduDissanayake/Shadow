"""
Heart Rate Analysis Module

Provides comprehensive heart rate estimation from BVP signals using peak detection,
interval analysis, and physiological validation.

Features:
- BVP peak detection using scipy.signal.find_peaks
- Heart rate estimation from peak intervals
- Validation of physiological ranges (40-200 BPM)
- Real-time heart rate monitoring
- Heart rate variability analysis

Author: Shadow AI Team
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal
import warnings

from wesad_pipeline.config import WESADConfig

logger = logging.getLogger(__name__)

class HeartRateAnalyzer:
    """
    Heart rate analyzer for BVP signals.
    
    Provides comprehensive heart rate estimation using peak detection algorithms
    and physiological validation.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the heart rate analyzer.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.sampling_rate = config.dataset.bvp_sampling_rate
        self.min_hr = config.analysis.min_heart_rate
        self.max_hr = config.analysis.max_heart_rate
        
        # Peak detection parameters
        self.min_peak_distance = int(60 * self.sampling_rate / self.max_hr)  # Minimum samples between peaks
        self.max_peak_distance = int(60 * self.sampling_rate / self.min_hr)  # Maximum samples between peaks
        
        # Statistics tracking
        self.stats = {
            'signals_analyzed': 0,
            'peaks_detected': 0,
            'valid_hr_estimates': 0,
            'invalid_hr_estimates': 0,
            'avg_heart_rate': 0.0,
            'hr_distribution': {'low': 0, 'normal': 0, 'high': 0}
        }
        
        self.logger.info(f"Heart rate analyzer initialized (range: {self.min_hr}-{self.max_hr} BPM)")
    
    def detect_peaks(self, bvp_signal: np.ndarray, method: str = 'adaptive') -> Tuple[np.ndarray, Dict]:
        """
        Detect peaks in BVP signal.
        
        Args:
            bvp_signal: BVP signal array
            method: Peak detection method ('adaptive', 'fixed', 'template')
            
        Returns:
            Tuple of (peak_indices, detection_info)
        """
        if len(bvp_signal) == 0:
            return np.array([]), {}
        
        try:
            if method == 'adaptive':
                peaks, detection_info = self._detect_peaks_adaptive(bvp_signal)
            elif method == 'fixed':
                peaks, detection_info = self._detect_peaks_fixed(bvp_signal)
            elif method == 'template':
                peaks, detection_info = self._detect_peaks_template(bvp_signal)
            else:
                self.logger.warning(f"Unknown peak detection method: {method}, using adaptive")
                peaks, detection_info = self._detect_peaks_adaptive(bvp_signal)
            
            # Filter peaks based on physiological constraints
            filtered_peaks = self._filter_physiological_peaks(peaks, bvp_signal)
            
            detection_info['original_peaks'] = len(peaks)
            detection_info['filtered_peaks'] = len(filtered_peaks)
            detection_info['filtering_ratio'] = len(filtered_peaks) / max(len(peaks), 1)
            
            self.stats['peaks_detected'] += len(filtered_peaks)
            
            return filtered_peaks, detection_info
            
        except Exception as e:
            self.logger.error(f"Peak detection failed: {str(e)}")
            return np.array([]), {'error': str(e)}
    
    def estimate_heart_rate(self, bvp_signal: np.ndarray, 
                          window_length: Optional[int] = None) -> Dict:
        """
        Estimate heart rate from BVP signal.
        
        Args:
            bvp_signal: BVP signal array
            window_length: Window length for sliding window analysis (samples)
            
        Returns:
            Dictionary containing heart rate estimates and analysis
        """
        try:
            if len(bvp_signal) == 0:
                return self._empty_hr_result()
            
            # Detect peaks
            peaks, peak_info = self.detect_peaks(bvp_signal, method='adaptive')
            
            if len(peaks) < 2:
                self.logger.warning("Insufficient peaks for heart rate estimation")
                return self._empty_hr_result()
            
            # Calculate instantaneous heart rate
            peak_intervals = np.diff(peaks) / self.sampling_rate  # Intervals in seconds
            instantaneous_hr = 60.0 / peak_intervals  # Convert to BPM
            
            # Filter physiologically valid heart rates
            valid_mask = (instantaneous_hr >= self.min_hr) & (instantaneous_hr <= self.max_hr)
            valid_hr = instantaneous_hr[valid_mask]
            
            if len(valid_hr) == 0:
                self.logger.warning("No valid heart rate estimates found")
                return self._empty_hr_result()
            
            # Calculate heart rate statistics
            mean_hr = np.mean(valid_hr)
            median_hr = np.median(valid_hr)
            std_hr = np.std(valid_hr)
            
            # Heart rate variability metrics
            hrv_metrics = self._calculate_hrv_metrics(peak_intervals[valid_mask])
            
            # Sliding window analysis if requested
            windowed_hr = None
            if window_length is not None:
                windowed_hr = self._estimate_windowed_hr(bvp_signal, window_length)
            
            # Heart rate result
            hr_result = {
                'mean_hr': float(mean_hr),
                'median_hr': float(median_hr),
                'std_hr': float(std_hr),
                'min_hr': float(np.min(valid_hr)),
                'max_hr': float(np.max(valid_hr)),
                'valid_estimates': len(valid_hr),
                'total_estimates': len(instantaneous_hr),
                'validity_ratio': len(valid_hr) / len(instantaneous_hr),
                'instantaneous_hr': valid_hr.tolist(),
                'peak_positions': peaks.tolist(),
                'peak_intervals': peak_intervals[valid_mask].tolist(),
                'hrv_metrics': hrv_metrics,
                'windowed_hr': windowed_hr,
                'peak_detection_info': peak_info,
                'signal_length': len(bvp_signal),
                'analysis_duration': len(bvp_signal) / self.sampling_rate
            }
            
            # Update statistics
            self.stats['signals_analyzed'] += 1
            self.stats['valid_hr_estimates'] += len(valid_hr)
            self.stats['invalid_hr_estimates'] += len(instantaneous_hr) - len(valid_hr)
            self._update_hr_distribution(mean_hr)
            
            return hr_result
            
        except Exception as e:
            self.logger.error(f"Heart rate estimation failed: {str(e)}")
            return self._empty_hr_result()
    
    def _detect_peaks_adaptive(self, bvp_signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Adaptive peak detection with signal-dependent parameters."""
        
        # Calculate adaptive threshold based on signal characteristics
        signal_std = np.std(bvp_signal)
        signal_mean = np.mean(bvp_signal)
        
        # Adaptive height threshold
        height_threshold = signal_mean + 0.3 * signal_std
        
        # Adaptive prominence threshold
        prominence_threshold = 0.2 * signal_std
        
        # Find peaks using scipy
        peaks, properties = signal.find_peaks(
            bvp_signal,
            height=height_threshold,
            distance=self.min_peak_distance,
            prominence=prominence_threshold,
            width=2  # Minimum peak width
        )
        
        detection_info = {
            'method': 'adaptive',
            'height_threshold': height_threshold,
            'prominence_threshold': prominence_threshold,
            'min_distance': self.min_peak_distance,
            'signal_std': signal_std,
            'signal_mean': signal_mean,
            'properties': properties
        }
        
        return peaks, detection_info
    
    def _detect_peaks_fixed(self, bvp_signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Fixed threshold peak detection."""
        
        # Fixed parameters
        height_threshold = np.percentile(bvp_signal, 75)  # 75th percentile
        prominence_threshold = np.std(bvp_signal) * 0.3
        
        peaks, properties = signal.find_peaks(
            bvp_signal,
            height=height_threshold,
            distance=self.min_peak_distance,
            prominence=prominence_threshold
        )
        
        detection_info = {
            'method': 'fixed',
            'height_threshold': height_threshold,
            'prominence_threshold': prominence_threshold,
            'min_distance': self.min_peak_distance,
            'properties': properties
        }
        
        return peaks, detection_info
    
    def _detect_peaks_template(self, bvp_signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Template-based peak detection."""
        
        # Create a simple BVP pulse template
        template_length = int(0.5 * self.sampling_rate)  # 0.5 second template
        t = np.linspace(0, 1, template_length)
        template = -np.exp(-((t - 0.3) ** 2) / 0.02)  # Inverted Gaussian (BVP-like)
        
        # Cross-correlation
        correlation = np.correlate(bvp_signal, template, mode='valid')
        
        # Find peaks in correlation
        corr_threshold = np.percentile(correlation, 80)
        peaks, properties = signal.find_peaks(
            correlation,
            height=corr_threshold,
            distance=self.min_peak_distance
        )
        
        # Adjust peak positions (correlation shifts indices)
        peaks = peaks + len(template) // 2
        
        detection_info = {
            'method': 'template',
            'template_length': template_length,
            'correlation_threshold': corr_threshold,
            'max_correlation': np.max(correlation) if len(correlation) > 0 else 0,
            'properties': properties
        }
        
        return peaks, detection_info
    
    def _filter_physiological_peaks(self, peaks: np.ndarray, bvp_signal: np.ndarray) -> np.ndarray:
        """Filter peaks based on physiological constraints."""
        if len(peaks) <= 1:
            return peaks
        
        # Calculate inter-peak intervals
        intervals = np.diff(peaks) / self.sampling_rate
        heart_rates = 60.0 / intervals
        
        # Filter based on physiological heart rate range
        valid_intervals = (heart_rates >= self.min_hr) & (heart_rates <= self.max_hr)
        
        # Keep the first peak and peaks that follow valid intervals
        filtered_peaks = [peaks[0]]
        for i, is_valid in enumerate(valid_intervals):
            if is_valid:
                filtered_peaks.append(peaks[i + 1])
        
        return np.array(filtered_peaks)
    
    def _calculate_hrv_metrics(self, rr_intervals: np.ndarray) -> Dict:
        """Calculate heart rate variability metrics."""
        if len(rr_intervals) < 2:
            return {}
        
        try:
            # Time domain HRV metrics
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))  # Root mean square of successive differences
            sdnn = np.std(rr_intervals)  # Standard deviation of NN intervals
            pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / len(rr_intervals)  # Proportion of NN50
            
            hrv_metrics = {
                'rmssd': float(rmssd),
                'sdnn': float(sdnn),
                'pnn50': float(pnn50),
                'mean_rr': float(np.mean(rr_intervals)),
                'std_rr': float(np.std(rr_intervals)),
                'min_rr': float(np.min(rr_intervals)),
                'max_rr': float(np.max(rr_intervals))
            }
            
            return hrv_metrics
            
        except Exception as e:
            self.logger.warning(f"HRV calculation failed: {str(e)}")
            return {}
    
    def _estimate_windowed_hr(self, bvp_signal: np.ndarray, window_length: int) -> Dict:
        """Estimate heart rate using sliding windows."""
        if len(bvp_signal) < window_length:
            return {}
        
        overlap = int(self.config.analysis.overlap_seconds * self.sampling_rate)
        step_size = window_length - overlap
        
        window_hrs = []
        window_positions = []
        
        for start_idx in range(0, len(bvp_signal) - window_length + 1, step_size):
            end_idx = start_idx + window_length
            window_signal = bvp_signal[start_idx:end_idx]
            
            # Estimate HR for this window
            window_hr_result = self.estimate_heart_rate(window_signal, window_length=None)
            
            if window_hr_result['valid_estimates'] > 0:
                window_hrs.append(window_hr_result['mean_hr'])
            else:
                window_hrs.append(np.nan)
            
            window_positions.append((start_idx, end_idx))
        
        windowed_result = {
            'window_hrs': window_hrs,
            'window_positions': window_positions,
            'valid_windows': sum(1 for hr in window_hrs if not np.isnan(hr)),
            'total_windows': len(window_hrs)
        }
        
        if len(window_hrs) > 0:
            valid_hrs = [hr for hr in window_hrs if not np.isnan(hr)]
            if valid_hrs:
                windowed_result.update({
                    'mean_windowed_hr': np.mean(valid_hrs),
                    'std_windowed_hr': np.std(valid_hrs),
                    'min_windowed_hr': np.min(valid_hrs),
                    'max_windowed_hr': np.max(valid_hrs)
                })
        
        return windowed_result
    
    def _categorize_hr(self, heart_rate: float) -> str:
        """Categorize heart rate into ranges."""
        if heart_rate < 60:
            return 'low'
        elif heart_rate > 100:
            return 'high'
        else:
            return 'normal'
    
    def _update_hr_distribution(self, heart_rate: float) -> None:
        """Update heart rate distribution statistics."""
        category = self._categorize_hr(heart_rate)
        self.stats['hr_distribution'][category] += 1
        
        # Update average heart rate
        total_analyses = self.stats['signals_analyzed']
        current_avg = self.stats['avg_heart_rate']
        self.stats['avg_heart_rate'] = ((current_avg * (total_analyses - 1)) + heart_rate) / total_analyses
    
    def _empty_hr_result(self) -> Dict:
        """Return empty heart rate result for error cases."""
        return {
            'mean_hr': 0.0,
            'median_hr': 0.0,
            'std_hr': 0.0,
            'min_hr': 0.0,
            'max_hr': 0.0,
            'valid_estimates': 0,
            'total_estimates': 0,
            'validity_ratio': 0.0,
            'instantaneous_hr': [],
            'peak_positions': [],
            'peak_intervals': [],
            'hrv_metrics': {},
            'windowed_hr': {},
            'peak_detection_info': {},
            'signal_length': 0,
            'analysis_duration': 0.0
        }
    
    def validate_heart_rate(self, heart_rate: float) -> bool:
        """
        Validate if heart rate is within physiological range.
        
        Args:
            heart_rate: Heart rate in BPM
            
        Returns:
            True if valid, False otherwise
        """
        return self.min_hr <= heart_rate <= self.max_hr
    
    def get_hr_statistics(self) -> Dict:
        """Get heart rate analysis statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset heart rate analysis statistics."""
        self.stats = {
            'signals_analyzed': 0,
            'peaks_detected': 0,
            'valid_hr_estimates': 0,
            'invalid_hr_estimates': 0,
            'avg_heart_rate': 0.0,
            'hr_distribution': {'low': 0, 'normal': 0, 'high': 0}
        }
        self.logger.debug("Heart rate analysis statistics reset")