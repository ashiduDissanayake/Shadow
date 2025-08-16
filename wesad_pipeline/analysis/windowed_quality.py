"""
Windowed Quality Assessment Module

Provides windowed signal quality assessment for BVP signals using sliding windows.

Features:
- Sliding window quality assessment
- Quality statistics across windows
- Integration with single signal quality module

Author: Shadow AI Team
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Optional
import warnings

from wesad_pipeline.config import WESADConfig
from .signal_quality import SignalQuality

logger = logging.getLogger(__name__)

class WindowedQuality:
    """
    Windowed quality analyzer for BVP signals.
    
    Assesses signal quality using sliding windows and provides
    window-level quality statistics and analysis.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the windowed quality analyzer.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize single signal quality analyzer
        self.signal_quality = SignalQuality(config)
        
        # Windowing parameters
        self.sampling_rate = config.dataset.bvp_sampling_rate
        self.window_size = config.analysis.window_size_seconds * self.sampling_rate
        
        # Statistics tracking
        self.stats = {
            'segments_assessed': 0,
            'avg_window_quality': 0.0,
            'windows_above_threshold': 0,
            'total_windows': 0
        }
        
        self.logger.info(f"Windowed quality analyzer initialized (window size: {config.analysis.window_size_seconds}s)")
    
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
                'quality_std': 0.0,
                'windows_above_threshold': 0,
                'total_windows': 0
            }
        
        overlap = int(self.config.analysis.overlap_seconds * self.sampling_rate)
        step_size = window_length - overlap
        
        window_scores = []
        window_positions = []
        detailed_metrics = []
        
        # Slide window through signal
        for start_idx in range(0, len(bvp_signal) - window_length + 1, step_size):
            end_idx = start_idx + window_length
            window_signal = bvp_signal[start_idx:end_idx]
            
            # Assess quality for this window
            quality_result = self.signal_quality.assess_signal_quality(window_signal)
            window_scores.append(quality_result['overall_score'])
            window_positions.append((start_idx, end_idx))
            detailed_metrics.append(quality_result['metrics'])
            
            self.stats['segments_assessed'] += 1
        
        # Calculate windowed statistics
        window_scores = np.array(window_scores)
        threshold = self.config.analysis.quality_threshold
        windows_above_threshold = np.sum(window_scores >= threshold)
        
        windowed_result = {
            'window_scores': window_scores.tolist(),
            'window_positions': window_positions,
            'detailed_metrics': detailed_metrics,
            'avg_quality': float(np.mean(window_scores)) if len(window_scores) > 0 else 0.0,
            'min_quality': float(np.min(window_scores)) if len(window_scores) > 0 else 0.0,
            'max_quality': float(np.max(window_scores)) if len(window_scores) > 0 else 0.0,
            'quality_std': float(np.std(window_scores)) if len(window_scores) > 0 else 0.0,
            'windows_above_threshold': int(windows_above_threshold),
            'total_windows': len(window_scores),
            'threshold_ratio': float(windows_above_threshold / len(window_scores)) if len(window_scores) > 0 else 0.0,
            'window_length': window_length,
            'step_size': step_size,
            'quality_threshold': threshold
        }
        
        # Update statistics
        self.stats['total_windows'] += len(window_scores)
        self.stats['windows_above_threshold'] += windows_above_threshold
        if len(window_scores) > 0:
            current_avg = self.stats['avg_window_quality']
            total_windows = self.stats['total_windows']
            new_avg = np.mean(window_scores)
            self.stats['avg_window_quality'] = ((current_avg * (total_windows - len(window_scores))) + 
                                               (new_avg * len(window_scores))) / total_windows
        
        return windowed_result
    
    def validate_windowed_quality(self, bvp_signal: np.ndarray, 
                                  min_acceptable_ratio: float = 0.7,
                                  window_length: Optional[int] = None) -> Dict:
        """
        Validate if enough windows meet quality threshold.
        
        Args:
            bvp_signal: BVP signal array
            min_acceptable_ratio: Minimum ratio of windows that must meet threshold
            window_length: Window length in samples
            
        Returns:
            Dictionary with validation results
        """
        windowed_result = self.assess_windowed_quality(bvp_signal, window_length)
        
        threshold_ratio = windowed_result['threshold_ratio']
        is_valid = threshold_ratio >= min_acceptable_ratio
        
        validation_result = {
            'is_valid': is_valid,
            'threshold_ratio': threshold_ratio,
            'min_required_ratio': min_acceptable_ratio,
            'windows_above_threshold': windowed_result['windows_above_threshold'],
            'total_windows': windowed_result['total_windows'],
            'avg_quality': windowed_result['avg_quality'],
            'recommendation': self._get_quality_recommendation(threshold_ratio, min_acceptable_ratio)
        }
        
        return validation_result
    
    def get_quality_segments(self, bvp_signal: np.ndarray, 
                           quality_threshold: Optional[float] = None,
                           window_length: Optional[int] = None) -> Dict:
        """
        Extract high-quality segments from signal.
        
        Args:
            bvp_signal: BVP signal array
            quality_threshold: Quality threshold for segment selection
            window_length: Window length in samples
            
        Returns:
            Dictionary containing high-quality segments
        """
        if quality_threshold is None:
            quality_threshold = self.config.analysis.quality_threshold
        
        windowed_result = self.assess_windowed_quality(bvp_signal, window_length)
        
        window_scores = np.array(windowed_result['window_scores'])
        window_positions = windowed_result['window_positions']
        
        # Find high-quality windows
        high_quality_indices = np.where(window_scores >= quality_threshold)[0]
        
        high_quality_segments = []
        for idx in high_quality_indices:
            start_idx, end_idx = window_positions[idx]
            segment = {
                'signal': bvp_signal[start_idx:end_idx],
                'quality_score': window_scores[idx],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'window_id': idx
            }
            high_quality_segments.append(segment)
        
        segments_result = {
            'segments': high_quality_segments,
            'total_segments': len(high_quality_segments),
            'total_windows': len(window_scores),
            'selection_ratio': len(high_quality_segments) / len(window_scores) if len(window_scores) > 0 else 0.0,
            'avg_quality_selected': float(np.mean(window_scores[high_quality_indices])) if len(high_quality_indices) > 0 else 0.0,
            'quality_threshold': quality_threshold
        }
        
        return segments_result
    
    def _get_quality_recommendation(self, threshold_ratio: float, min_required_ratio: float) -> str:
        """Get recommendation based on quality analysis."""
        if threshold_ratio >= min_required_ratio:
            if threshold_ratio >= 0.9:
                return "Excellent signal quality - proceed with full analysis"
            elif threshold_ratio >= 0.8:
                return "Good signal quality - suitable for most analyses"
            else:
                return "Acceptable signal quality - monitor results carefully"
        else:
            if threshold_ratio >= 0.5:
                return "Marginal signal quality - consider additional filtering"
            elif threshold_ratio >= 0.3:
                return "Poor signal quality - significant preprocessing required"
            else:
                return "Very poor signal quality - consider signal rejection"
    
    def get_windowed_stats(self) -> Dict:
        """Get windowed quality assessment statistics."""
        stats = self.stats.copy()
        if stats['total_windows'] > 0:
            stats['overall_threshold_ratio'] = stats['windows_above_threshold'] / stats['total_windows']
        else:
            stats['overall_threshold_ratio'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset windowed quality statistics."""
        self.stats = {
            'segments_assessed': 0,
            'avg_window_quality': 0.0,
            'windows_above_threshold': 0,
            'total_windows': 0
        }
        self.signal_quality.reset_stats()