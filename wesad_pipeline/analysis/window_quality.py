"""
Window Quality Assessment Module

Provides windowed signal quality assessment for BVP signals using sliding windows.
This module focuses specifically on assessing quality across multiple windows
and uses the SignalQuality class internally for consistent quality metrics.

Features:
- Sliding window quality assessment
- Window-level quality statistics and filtering
- Integration with SignalQuality for consistent metrics
- Quality-based window filtering and statistics

Author: Shadow AI Team
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union

from wesad_pipeline.config import WESADConfig
from .signal_quality import SignalQuality

logger = logging.getLogger(__name__)

class WindowQuality:
    """
    Window quality analyzer for BVP signals.
    
    Provides comprehensive windowed quality assessment using sliding windows
    and leverages the SignalQuality class for consistent quality metrics.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the window quality analyzer.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize the underlying signal quality analyzer
        self.signal_quality = SignalQuality(config)
        
        # Quality assessment parameters
        self.sampling_rate = config.dataset.bvp_sampling_rate
        self.window_size = config.analysis.window_size_seconds * self.sampling_rate
        
        # Statistics tracking for windowed analysis
        self.stats = {
            'windowed_assessments_performed': 0,
            'total_windows_assessed': 0,
            'avg_window_quality': 0.0,
            'windows_above_threshold': 0,
            'windows_below_threshold': 0,
            'window_quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        }
        
        self.logger.info(f"Window quality analyzer initialized (sampling rate: {self.sampling_rate}Hz)")
    
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
                'window_length': window_length,
                'step_size': 0,
                'total_windows': 0
            }
        
        overlap = int(self.config.analysis.overlap_seconds * self.sampling_rate)
        step_size = window_length - overlap
        
        window_scores = []
        window_positions = []
        
        # Slide window through signal
        for start_idx in range(0, len(bvp_signal) - window_length + 1, step_size):
            end_idx = start_idx + window_length
            window_signal = bvp_signal[start_idx:end_idx]
            
            # Assess quality for this window using SignalQuality
            quality_result = self.signal_quality.assess_signal_quality(window_signal)
            window_scores.append(quality_result['overall_score'])
            window_positions.append((start_idx, end_idx))
            
            self.stats['total_windows_assessed'] += 1
        
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
        
        # Update statistics
        self.stats['windowed_assessments_performed'] += 1
        self._update_windowed_statistics(window_scores)
        
        return windowed_result
    
    def filter_windows_by_quality(self, windows_data: List[Dict], quality_threshold: Optional[float] = None) -> Tuple[List[Dict], Dict]:
        """
        Filter windows based on quality threshold.
        
        Args:
            windows_data: List of window dictionaries with 'bvp' data
            quality_threshold: Quality threshold. If None, uses config threshold.
            
        Returns:
            Tuple of (accepted_windows, filter_statistics)
        """
        if quality_threshold is None:
            quality_threshold = self.config.analysis.quality_threshold
        
        accepted_windows = []
        rejected_windows = []
        
        for window_data in windows_data:
            window_bvp = window_data.get('bvp', np.array([]))
            
            if len(window_bvp) == 0:
                rejected_windows.append(window_data)
                continue
            
            # Assess window quality
            quality_result = self.signal_quality.assess_signal_quality(window_bvp)
            window_quality = quality_result['overall_score']
            
            # Add quality information to window data
            enhanced_window = window_data.copy()
            enhanced_window['quality'] = window_quality
            enhanced_window['quality_metrics'] = quality_result['metrics']
            enhanced_window['quality_level'] = quality_result['quality_level']
            
            if window_quality >= quality_threshold:
                accepted_windows.append(enhanced_window)
                self.stats['windows_above_threshold'] += 1
            else:
                rejected_windows.append(enhanced_window)
                self.stats['windows_below_threshold'] += 1
        
        # Calculate filter statistics
        filter_stats = {
            'total_windows': len(windows_data),
            'accepted_windows': len(accepted_windows),
            'rejected_windows': len(rejected_windows),
            'acceptance_rate': len(accepted_windows) / max(len(windows_data), 1),
            'rejection_rate': len(rejected_windows) / max(len(windows_data), 1),
            'quality_threshold': quality_threshold,
            'avg_accepted_quality': np.mean([w['quality'] for w in accepted_windows]) if accepted_windows else 0.0,
            'avg_rejected_quality': np.mean([w['quality'] for w in rejected_windows]) if rejected_windows else 0.0
        }
        
        self.logger.info(f"Window quality filtering: {len(accepted_windows)}/{len(windows_data)} windows accepted "
                        f"(threshold: {quality_threshold:.2f})")
        
        return accepted_windows, filter_stats
    
    def analyze_quality_distribution(self, windows_data: List[Dict]) -> Dict:
        """
        Analyze quality distribution across windows.
        
        Args:
            windows_data: List of window dictionaries with quality information
            
        Returns:
            Dictionary containing quality distribution analysis
        """
        if not windows_data:
            return {}
        
        qualities = []
        quality_levels = []
        
        for window_data in windows_data:
            if 'quality' in window_data:
                qualities.append(window_data['quality'])
                quality_levels.append(window_data.get('quality_level', 'unknown'))
            else:
                # Assess quality if not already present
                window_bvp = window_data.get('bvp', np.array([]))
                if len(window_bvp) > 0:
                    quality_result = self.signal_quality.assess_signal_quality(window_bvp)
                    qualities.append(quality_result['overall_score'])
                    quality_levels.append(quality_result['quality_level'])
        
        if not qualities:
            return {}
        
        # Calculate distribution statistics
        qualities = np.array(qualities)
        level_counts = {level: quality_levels.count(level) for level in set(quality_levels)}
        
        distribution_analysis = {
            'quality_statistics': {
                'mean': float(np.mean(qualities)),
                'std': float(np.std(qualities)),
                'min': float(np.min(qualities)),
                'max': float(np.max(qualities)),
                'median': float(np.median(qualities)),
                'percentile_25': float(np.percentile(qualities, 25)),
                'percentile_75': float(np.percentile(qualities, 75))
            },
            'quality_level_counts': level_counts,
            'quality_level_percentages': {level: count/len(qualities)*100 for level, count in level_counts.items()},
            'total_windows': len(qualities),
            'quality_threshold_analysis': self._analyze_threshold_sensitivity(qualities)
        }
        
        return distribution_analysis
    
    def _update_windowed_statistics(self, window_scores: np.ndarray) -> None:
        """Update windowed quality statistics."""
        if len(window_scores) == 0:
            return
        
        # Update average window quality
        current_avg = self.stats['avg_window_quality']
        total_assessments = self.stats['windowed_assessments_performed']
        new_avg = np.mean(window_scores)
        
        self.stats['avg_window_quality'] = ((current_avg * (total_assessments - 1)) + new_avg) / total_assessments
        
        # Update quality distribution
        for score in window_scores:
            quality_level = self.signal_quality._categorize_quality(score)
            self.stats['window_quality_distribution'][quality_level] += 1
    
    def _analyze_threshold_sensitivity(self, qualities: np.ndarray) -> Dict:
        """Analyze how different thresholds would affect window acceptance."""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_analysis = {}
        
        for threshold in thresholds:
            accepted = np.sum(qualities >= threshold)
            threshold_analysis[threshold] = {
                'accepted_windows': int(accepted),
                'acceptance_rate': float(accepted / len(qualities)),
                'rejected_windows': int(len(qualities) - accepted),
                'rejection_rate': float((len(qualities) - accepted) / len(qualities))
            }
        
        return threshold_analysis
    
    def get_window_quality_statistics(self) -> Dict:
        """Get window quality assessment statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset window quality assessment statistics."""
        self.stats = {
            'windowed_assessments_performed': 0,
            'total_windows_assessed': 0,
            'avg_window_quality': 0.0,
            'windows_above_threshold': 0,
            'windows_below_threshold': 0,
            'window_quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        }
        self.logger.debug("Window quality assessment statistics reset")
    
    def validate_windowed_quality_threshold(self, bvp_signal: np.ndarray, 
                                          threshold: Optional[float] = None,
                                          min_acceptable_windows: int = 1) -> bool:
        """
        Validate if signal has sufficient windows meeting quality threshold.
        
        Args:
            bvp_signal: BVP signal array
            threshold: Quality threshold. If None, uses config threshold.
            min_acceptable_windows: Minimum number of windows that must meet threshold.
            
        Returns:
            True if signal has sufficient high-quality windows, False otherwise
        """
        if threshold is None:
            threshold = self.config.analysis.quality_threshold
        
        windowed_result = self.assess_windowed_quality(bvp_signal)
        window_scores = windowed_result['window_scores']
        
        if len(window_scores) == 0:
            return False
        
        acceptable_windows = sum(1 for score in window_scores if score >= threshold)
        return acceptable_windows >= min_acceptable_windows