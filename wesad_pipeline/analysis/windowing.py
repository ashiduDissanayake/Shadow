"""
Windowing Module

Provides pure windowing functionality for BVP signals including sliding window
creation and label assignment, without quality assessment.

Features:
- Create sliding windows with configurable size and overlap
- Compute window labels (most common label in window)
- Calculate label confidence scores
- Window metadata and statistics

Author: Shadow AI Team
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter
import warnings

from wesad_pipeline.config import WESADConfig

logger = logging.getLogger(__name__)

class WindowAnalyzer:
    """
    Pure window analyzer for BVP signal analysis.
    
    Provides windowing capabilities including window creation and
    label assignment without quality assessment.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the window analyzer.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Window parameters
        self.window_size = config.analysis.window_size_seconds * config.dataset.bvp_sampling_rate
        self.overlap = config.analysis.overlap_seconds * config.dataset.bvp_sampling_rate
        self.step_size = int(self.window_size - self.overlap)
        
        # Analysis parameters
        self.sampling_rate = config.dataset.bvp_sampling_rate
        self.label_mapping = config.dataset.label_mapping
        
        # Statistics tracking
        self.stats = {
            'windows_created': 0,
            'total_signals_processed': 0,
            'label_distribution': {},
            'avg_label_confidence': 0.0
        }
        
        self.logger.info(f"Window analyzer initialized: {config.analysis.window_size_seconds}s windows, "
                        f"{config.analysis.overlap_seconds}s overlap")
    
    def create_windows(self, bvp_signal: np.ndarray, labels: np.ndarray,
                      timestamps: Optional[np.ndarray] = None) -> Dict:
        """
        Create sliding windows from BVP signal and labels.
        
        Args:
            bvp_signal: BVP signal array
            labels: Corresponding labels array
            timestamps: Optional timestamps array
            
        Returns:
            Dictionary containing windowed data and metadata
        """
        try:
            if len(bvp_signal) == 0 or len(labels) == 0:
                return self._empty_windows_result()
            
            # Ensure signal and labels have same length
            min_length = min(len(bvp_signal), len(labels))
            bvp_signal = bvp_signal[:min_length]
            labels = labels[:min_length]
            
            if timestamps is not None:
                timestamps = timestamps[:min_length]
            else:
                timestamps = np.arange(len(bvp_signal)) / self.sampling_rate
            
            # Check if signal is long enough for windowing
            if len(bvp_signal) < self.window_size:
                self.logger.warning(f"Signal too short for windowing: {len(bvp_signal)} < {self.window_size}")
                return self._empty_windows_result()
            
            # Create windows
            windows_data = []
            window_positions = []
            window_labels = []
            window_confidences = []
            window_timestamps = []
            
            # Slide through the signal
            for start_idx in range(0, len(bvp_signal) - int(self.window_size) + 1, self.step_size):
                end_idx = start_idx + int(self.window_size)
                
                # Extract window data
                window_bvp = bvp_signal[start_idx:end_idx]
                window_labels_raw = labels[start_idx:end_idx]
                window_ts = timestamps[start_idx:end_idx]
                
                # Calculate window label and confidence
                window_label, label_confidence = self._calculate_window_label(window_labels_raw)
                
                # Store window data
                window_data = {
                    'bvp': window_bvp,
                    'label': window_label,
                    'confidence': label_confidence,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': window_ts[0],
                    'end_time': window_ts[-1],
                    'window_id': len(windows_data)
                }
                
                windows_data.append(window_data)
                window_positions.append((start_idx, end_idx))
                window_labels.append(window_label)
                window_confidences.append(label_confidence)
                window_timestamps.append((window_ts[0], window_ts[-1]))
                
                self.stats['windows_created'] += 1
                
                # Update label distribution
                label_name = self.config.get_label_name(window_label)
                if label_name not in self.stats['label_distribution']:
                    self.stats['label_distribution'][label_name] = 0
                self.stats['label_distribution'][label_name] += 1
            
            # Calculate summary statistics
            summary_stats = self._calculate_window_statistics(windows_data)
            
            # Update global statistics
            self.stats['total_signals_processed'] += 1
            if len(window_confidences) > 0:
                self.stats['avg_label_confidence'] = np.mean(window_confidences)
            
            # Create result dictionary
            result = {
                'windows': windows_data,
                'window_positions': window_positions,
                'window_labels': np.array(window_labels),
                'window_confidences': np.array(window_confidences),
                'window_timestamps': window_timestamps,
                'summary_stats': summary_stats,
                'metadata': {
                    'total_windows': len(windows_data),
                    'window_size_samples': int(self.window_size),
                    'window_size_seconds': self.config.analysis.window_size_seconds,
                    'overlap_samples': int(self.overlap),
                    'overlap_seconds': self.config.analysis.overlap_seconds,
                    'step_size_samples': self.step_size,
                    'signal_length': len(bvp_signal),
                    'signal_duration': len(bvp_signal) / self.sampling_rate
                }
            }
            
            self.logger.info(f"Created {len(windows_data)} windows from signal")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Window creation failed: {str(e)}")
            return self._empty_windows_result()
    
    def analyze_window_distribution(self, windows: List[Dict]) -> Dict:
        """
        Analyze the distribution of windows across conditions.
        
        Args:
            windows: List of window dictionaries
            
        Returns:
            Dictionary containing distribution analysis
        """
        try:
            if not windows:
                return {}
            
            # Collect statistics
            labels = [w['label'] for w in windows]
            confidences = [w['confidence'] for w in windows]
            
            # Condition analysis
            condition_counts = Counter(labels)
            condition_confidences = {}
            
            # Calculate per-condition statistics
            for condition_id in condition_counts.keys():
                condition_name = self.config.get_label_name(condition_id)
                condition_windows = [w for w in windows if w['label'] == condition_id]
                condition_confidences[condition_name] = np.mean([w['confidence'] for w in condition_windows])
            
            # Calculate temporal distribution
            window_times = [(w['start_time'], w['end_time']) for w in windows]
            total_duration = max(w['end_time'] for w in windows) - min(w['start_time'] for w in windows)
            
            distribution_analysis = {
                'condition_counts': condition_counts,
                'condition_percentages': {k: v/len(windows)*100 for k, v in condition_counts.items()},
                'condition_confidences': condition_confidences,
                'temporal_info': {
                    'total_windows': len(windows),
                    'total_duration': total_duration,
                    'average_window_duration': self.config.analysis.window_size_seconds,
                    'coverage_ratio': len(windows) * self.config.analysis.window_size_seconds / total_duration
                },
                'confidence_stats': {
                    'mean_confidence': np.mean(confidences),
                    'std_confidence': np.std(confidences),
                    'min_confidence': np.min(confidences),
                    'max_confidence': np.max(confidences)
                }
            }
            
            return distribution_analysis
            
        except Exception as e:
            self.logger.error(f"Window distribution analysis failed: {str(e)}")
            return {}
    
    def _calculate_window_label(self, window_labels: np.ndarray) -> Tuple[int, float]:
        """
        Calculate the most common label in the window and its confidence.
        
        Args:
            window_labels: Array of labels within the window
            
        Returns:
            Tuple of (most_common_label, confidence_score)
        """
        if len(window_labels) == 0:
            return 0, 0.0
        
        # Count occurrences of each label
        label_counts = Counter(window_labels)
        most_common_label, most_common_count = label_counts.most_common(1)[0]
        
        # Calculate confidence as ratio of most common label
        confidence = most_common_count / len(window_labels)
        
        return int(most_common_label), float(confidence)
    
    def _calculate_window_statistics(self, windows: List[Dict]) -> Dict:
        """Calculate summary statistics for windows."""
        if not windows:
            return {}
        
        confidences = [w['confidence'] for w in windows]
        labels = [w['label'] for w in windows]
        
        # Label distribution
        label_counts = Counter(labels)
        condition_dist = {}
        for label_id, count in label_counts.items():
            condition_name = self.config.get_label_name(label_id)
            condition_dist[condition_name] = count
        
        stats = {
            'total_windows': len(windows),
            'avg_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'condition_distribution': condition_dist,
            'unique_conditions': len(set(labels))
        }
        
        return stats
    
    def _empty_windows_result(self) -> Dict:
        """Return empty windows result for error cases."""
        return {
            'windows': [],
            'window_positions': [],
            'window_labels': np.array([]),
            'window_confidences': np.array([]),
            'window_timestamps': [],
            'summary_stats': {},
            'metadata': {
                'total_windows': 0,
                'window_size_samples': int(self.window_size),
                'window_size_seconds': self.config.analysis.window_size_seconds,
                'overlap_samples': int(self.overlap),
                'overlap_seconds': self.config.analysis.overlap_seconds,
                'step_size_samples': self.step_size,
                'signal_length': 0,
                'signal_duration': 0.0
            }
        }
    
    def get_windowing_stats(self) -> Dict:
        """Get windowing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset windowing statistics."""
        self.stats = {
            'windows_created': 0,
            'total_signals_processed': 0,
            'label_distribution': {},
            'avg_label_confidence': 0.0
        }