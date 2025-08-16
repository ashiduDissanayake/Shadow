"""
Signal Quality Assessment Module

Provides comprehensive signal quality assessment for individual BVP signals.

Features:
- Multi-metric quality assessment (variance, periodicity, morphology, amplitude, noise)
- Quality threshold validation
- Statistical tracking and reporting

Author: Shadow AI Team
License: MIT
"""

import numpy as np
import logging
from typing import Dict
from scipy import signal
from scipy.stats import pearsonr
import warnings

from wesad_pipeline.config import WESADConfig

logger = logging.getLogger(__name__)

class SignalQuality:
    """
    Signal quality analyzer for individual BVP signals.
    
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
        
        # Heart rate constraints for quality assessment
        self.min_hr = config.analysis.min_heart_rate
        self.max_hr = config.analysis.max_heart_rate
        
        # Statistics tracking
        self.stats = {
            'assessments_performed': 0,
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
    
    def validate_quality_threshold(self, bvp_signal: np.ndarray, threshold: float = None) -> bool:
        """
        Validate if signal meets quality threshold.
        
        Args:
            bvp_signal: BVP signal array
            threshold: Quality threshold (0-1). Uses config default if None.
            
        Returns:
            True if signal quality meets threshold
        """
        if threshold is None:
            threshold = self.config.analysis.quality_threshold
        
        quality_result = self.assess_signal_quality(bvp_signal)
        return quality_result['overall_score'] >= threshold
    
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
                if len(correlation) > 0:
                    template_consistency = np.max(correlation) / (np.linalg.norm(template) * np.linalg.norm(bvp_signal))
                else:
                    template_consistency = 0.0
            else:
                template_consistency = 0.0
            
            # Combine smoothness and consistency
            morphology_score = 0.5 * gradient_smoothness + 0.5 * template_consistency
            
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
    
    def _categorize_quality(self, overall_score: float) -> str:
        """Categorize quality score into levels."""
        if overall_score >= 0.8:
            return 'excellent'
        elif overall_score >= 0.6:
            return 'good'
        elif overall_score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _update_quality_distribution(self, overall_score: float):
        """Update quality distribution statistics."""
        quality_level = self._categorize_quality(overall_score)
        self.stats['quality_distribution'][quality_level] += 1
        
        # Update average
        total_assessments = self.stats['assessments_performed']
        current_avg = self.stats['avg_quality_score']
        self.stats['avg_quality_score'] = ((current_avg * (total_assessments - 1)) + overall_score) / total_assessments
    
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
            'weights': {'variance': 0.2, 'periodicity': 0.3, 'morphology': 0.2, 'amplitude': 0.15, 'noise': 0.15},
            'quality_level': 'poor',
            'signal_length': 0,
            'sampling_rate': self.sampling_rate
        }
    
    def get_quality_stats(self) -> Dict:
        """Get quality assessment statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset quality assessment statistics."""
        self.stats = {
            'assessments_performed': 0,
            'avg_quality_score': 0.0,
            'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        }