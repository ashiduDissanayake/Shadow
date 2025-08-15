"""
Signal Quality Assessment Module

Provides comprehensive signal quality assessment for BVP signals including
variance analysis, peak consistency, and sliding window quality assessment.

Features:
- Signal quality computation based on variance and peak consistency
- Sliding window quality assessment
- Quality threshold validation
- Real-time quality monitoring capabilities

Author: Shadow AI Team
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal
from scipy.stats import pearsonr
import warnings

from wesad_pipeline.config import WESADConfig

logger = logging.getLogger(__name__)

class SignalQuality:
    """
    Signal quality analyzer for BVP signals.
    
    Provides comprehensive quality assessment using multiple metrics including
    signal variance, peak consistency, periodicity, and morphological features.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the signal quality analyzer.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Quality assessment parameters
        self.sampling_rate = config.dataset.bvp_sampling_rate
        self.window_size = config.analysis.window_size_seconds * self.sampling_rate
        
        # Heart rate constraints for quality assessment
        self.min_hr = config.analysis.min_heart_rate
        self.max_hr = config.analysis.max_heart_rate
        
        # Statistics tracking
        self.stats = {
            'assessments_performed': 0,
            'segments_assessed': 0,
            'avg_quality_score': 0.0,
            'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        }
        
        self.logger.info(f"Signal quality analyzer initialized (sampling rate: {self.sampling_rate}Hz)")
    
    def assess_signal_quality(self, bvp_signal: np.ndarray) -> Dict:
        """
        Comprehensive signal quality assessment for a BVP signal.
        
        Args:
            bvp_signal: BVP signal array
            
        Returns:
            Dictionary containing quality metrics and overall score
        """
        try:
            if len(bvp_signal) == 0:
                return self._empty_quality_result()
            
            # Individual quality metrics
            variance_score = self._assess_variance_quality(bvp_signal)
            periodicity_score = self._assess_periodicity_quality(bvp_signal)
            morphology_score = self._assess_morphology_quality(bvp_signal)
            amplitude_score = self._assess_amplitude_quality(bvp_signal)
            noise_score = self._assess_noise_quality(bvp_signal)
            
            # Combine metrics into overall quality score
            weights = {
                'variance': 0.2,
                'periodicity': 0.3,
                'morphology': 0.2,
                'amplitude': 0.15,
                'noise': 0.15
            }
            
            overall_score = (
                weights['variance'] * variance_score +
                weights['periodicity'] * periodicity_score +
                weights['morphology'] * morphology_score +
                weights['amplitude'] * amplitude_score +
                weights['noise'] * noise_score
            )
            
            # Ensure score is within bounds
            overall_score = np.clip(overall_score, 0.0, 1.0)
            
            # Quality assessment result
            quality_result = {
                'overall_score': float(overall_score),
                'metrics': {
                    'variance_score': float(variance_score),
                    'periodicity_score': float(periodicity_score),
                    'morphology_score': float(morphology_score),
                    'amplitude_score': float(amplitude_score),
                    'noise_score': float(noise_score)
                },
                'weights': weights,
                'quality_level': self._categorize_quality(overall_score),
                'signal_length': len(bvp_signal),
                'sampling_rate': self.sampling_rate
            }
            
            # Update statistics
            self.stats['assessments_performed'] += 1
            self._update_quality_distribution(overall_score)
            
            return quality_result
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {str(e)}")
            return self._empty_quality_result()
    
    def assess_windowed_quality(self, bvp_signal: np.ndarray, window_length: Optional[int] = None) -> Dict:
        """
        Assess signal quality using sliding windows.
        
        Args:
            bvp_signal: BVP signal array
            window_length: Window length in samples. If None, uses config window size.
            
        Returns:
            Dictionary containing windowed quality assessment
        """
        if window_length is None:
            window_length = int(self.window_size)
        
        if len(bvp_signal) < window_length:
            self.logger.warning(f"Signal too short for windowed analysis: {len(bvp_signal)} < {window_length}")
            return {
                'window_scores': [],
                'window_positions': [],
                'avg_quality': 0.0,
                'min_quality': 0.0,
                'max_quality': 0.0,
                'quality_std': 0.0
            }
        
        overlap = int(self.config.analysis.overlap_seconds * self.sampling_rate)
        step_size = window_length - overlap
        
        window_scores = []
        window_positions = []
        
        # Slide window through signal
        for start_idx in range(0, len(bvp_signal) - window_length + 1, step_size):
            end_idx = start_idx + window_length
            window_signal = bvp_signal[start_idx:end_idx]
            
            # Assess quality for this window
            quality_result = self.assess_signal_quality(window_signal)
            window_scores.append(quality_result['overall_score'])
            window_positions.append((start_idx, end_idx))
            
            self.stats['segments_assessed'] += 1
        
        # Calculate windowed statistics
        window_scores = np.array(window_scores)
        windowed_result = {
            'window_scores': window_scores.tolist(),
            'window_positions': window_positions,
            'avg_quality': float(np.mean(window_scores)) if len(window_scores) > 0 else 0.0,
            'min_quality': float(np.min(window_scores)) if len(window_scores) > 0 else 0.0,
            'max_quality': float(np.max(window_scores)) if len(window_scores) > 0 else 0.0,
            'quality_std': float(np.std(window_scores)) if len(window_scores) > 0 else 0.0,
            'window_length': window_length,
            'step_size': step_size,
            'total_windows': len(window_scores)
        }
        
        return windowed_result
    
    def _assess_variance_quality(self, bvp_signal: np.ndarray) -> float:
        """Assess signal quality based on variance."""
        if len(bvp_signal) == 0:
            return 0.0
        
        signal_var = np.var(bvp_signal)
        signal_mean = np.mean(np.abs(bvp_signal))
        
        # Normalize variance by signal magnitude
        if signal_mean > 0:
            normalized_var = signal_var / (signal_mean ** 2)
            # Good BVP signals typically have normalized variance between 0.01 and 0.1
            variance_score = np.clip(normalized_var / 0.1, 0.0, 1.0)
        else:
            variance_score = 0.0
        
        return variance_score
    
    def _assess_periodicity_quality(self, bvp_signal: np.ndarray) -> float:
        """Assess signal quality based on periodicity (heart rate consistency)."""
        if len(bvp_signal) < self.sampling_rate:  # Need at least 1 second
            return 0.0
        
        try:
            # Calculate autocorrelation
            signal_centered = bvp_signal - np.mean(bvp_signal)
            autocorr = np.correlate(signal_centered, signal_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Look for peaks in the autocorrelation corresponding to heart rate
            min_period = int(60 * self.sampling_rate / self.max_hr)  # Minimum HR period
            max_period = int(60 * self.sampling_rate / self.min_hr)  # Maximum HR period
            
            if max_period < len(autocorr):
                periodicity_score = np.max(autocorr[min_period:max_period])
            else:
                periodicity_score = 0.0
            
            return np.clip(periodicity_score, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _assess_morphology_quality(self, bvp_signal: np.ndarray) -> float:
        """Assess signal quality based on BVP morphology."""
        if len(bvp_signal) < 10:
            return 0.0
        
        try:
            # Calculate gradient smoothness
            gradient = np.gradient(bvp_signal)
            gradient_smoothness = 1.0 / (1.0 + np.std(gradient))
            
            # Calculate signal regularity using template matching
            # Find a representative segment (middle portion)
            mid_start = len(bvp_signal) // 4
            mid_end = 3 * len(bvp_signal) // 4
            template = bvp_signal[mid_start:mid_end]
            
            if len(template) > self.sampling_rate // 2:  # At least 0.5 seconds
                template = template[:self.sampling_rate // 2]  # Use 0.5 second template
                
                # Cross-correlation with the template
                correlation = np.correlate(bvp_signal, template, mode='valid')
                max_correlation = np.max(correlation) / (np.linalg.norm(template) * np.linalg.norm(bvp_signal))
                regularity_score = np.clip(max_correlation, 0.0, 1.0)
            else:
                regularity_score = 0.0
            
            # Combine gradient smoothness and regularity
            morphology_score = 0.6 * gradient_smoothness + 0.4 * regularity_score
            
            return np.clip(morphology_score, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _assess_amplitude_quality(self, bvp_signal: np.ndarray) -> float:
        """Assess signal quality based on amplitude characteristics."""
        if len(bvp_signal) == 0:
            return 0.0
        
        # Dynamic range assessment
        signal_range = np.ptp(bvp_signal)  # Peak-to-peak
        signal_std = np.std(bvp_signal)
        
        if signal_std > 0:
            range_score = np.clip(signal_range / (4 * signal_std), 0.0, 1.0)
        else:
            range_score = 0.0
        
        # Amplitude consistency
        signal_abs = np.abs(bvp_signal)
        amplitude_cv = np.std(signal_abs) / (np.mean(signal_abs) + 1e-8)
        consistency_score = np.clip(1.0 - amplitude_cv, 0.0, 1.0)
        
        # Combine range and consistency
        amplitude_score = 0.7 * range_score + 0.3 * consistency_score
        
        return amplitude_score
    
    def _assess_noise_quality(self, bvp_signal: np.ndarray) -> float:
        """Assess signal quality based on noise characteristics."""
        if len(bvp_signal) < 3:
            return 0.0
        
        try:
            # High-frequency noise assessment using second derivative
            second_derivative = np.diff(bvp_signal, n=2)
            noise_level = np.std(second_derivative)
            signal_level = np.std(bvp_signal)
            
            if signal_level > 0:
                snr_estimate = signal_level / (noise_level + 1e-8)
                noise_score = np.clip(snr_estimate / 10.0, 0.0, 1.0)  # Good SNR ~ 10
            else:
                noise_score = 0.0
            
            return noise_score
            
        except Exception:
            return 0.0
    
    def _categorize_quality(self, quality_score: float) -> str:
        """Categorize quality score into levels."""
        if quality_score >= 0.8:
            return 'excellent'
        elif quality_score >= 0.6:
            return 'good'
        elif quality_score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _update_quality_distribution(self, quality_score: float) -> None:
        """Update quality distribution statistics."""
        category = self._categorize_quality(quality_score)
        self.stats['quality_distribution'][category] += 1
        
        # Update average quality score
        total_assessments = self.stats['assessments_performed']
        current_avg = self.stats['avg_quality_score']
        self.stats['avg_quality_score'] = ((current_avg * (total_assessments - 1)) + quality_score) / total_assessments
    
    def _empty_quality_result(self) -> Dict:
        """Return empty quality result for error cases."""
        return {
            'overall_score': 0.0,
            'metrics': {
                'variance_score': 0.0,
                'periodicity_score': 0.0,
                'morphology_score': 0.0,
                'amplitude_score': 0.0,
                'noise_score': 0.0
            },
            'weights': {},
            'quality_level': 'poor',
            'signal_length': 0,
            'sampling_rate': self.sampling_rate
        }
    
    def validate_quality_threshold(self, bvp_signal: np.ndarray, threshold: Optional[float] = None) -> bool:
        """
        Validate if signal meets quality threshold.
        
        Args:
            bvp_signal: BVP signal array
            threshold: Quality threshold. If None, uses config threshold.
            
        Returns:
            True if signal meets quality threshold, False otherwise
        """
        if threshold is None:
            threshold = self.config.analysis.quality_threshold
        
        quality_result = self.assess_signal_quality(bvp_signal)
        return quality_result['overall_score'] >= threshold
    
    def get_quality_statistics(self) -> Dict:
        """Get quality assessment statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset quality assessment statistics."""
        self.stats = {
            'assessments_performed': 0,
            'segments_assessed': 0,
            'avg_quality_score': 0.0,
            'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        }
        self.logger.debug("Quality assessment statistics reset")