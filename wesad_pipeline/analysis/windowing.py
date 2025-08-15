"""
Windowing Analysis Module

Provides comprehensive windowing analysis for BVP signals including sliding window
creation, label assignment, and window-level statistics computation.

Features:
- Create sliding windows with configurable size and overlap
- Compute window labels (most common label in window)
- Calculate window confidence and quality scores
- Window-level feature extraction
- Batch processing for multiple subjects

Author: Shadow AI Team
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import warnings

from wesad_pipeline.config import WESADConfig

logger = logging.getLogger(__name__)

class WindowAnalyzer:
    """
    Window analyzer for BVP signal analysis.
    
    Provides comprehensive windowing capabilities including window creation,
    label assignment, quality assessment, and feature extraction.
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
        self.min_quality = config.analysis.min_window_quality
        
        # Analysis parameters
        self.sampling_rate = config.dataset.bvp_sampling_rate
        self.label_mapping = config.dataset.label_mapping
        
        # Statistics tracking
        self.stats = {
            'windows_created': 0,
            'windows_accepted': 0,
            'windows_rejected': 0,
            'total_signals_processed': 0,
            'label_distribution': {},
            'avg_window_quality': 0.0,
            'avg_label_confidence': 0.0
        }
        
        self.logger.info(f"Window analyzer initialized: {config.analysis.window_size_seconds}s windows, "
                        f"{config.analysis.overlap_seconds}s overlap")
    
    def create_windows(self, bvp_signal: np.ndarray, labels: np.ndarray,
                      timestamps: Optional[np.ndarray] = None,
                      quality_scores: Optional[np.ndarray] = None) -> Dict:
        """
        Create sliding windows from BVP signal and labels.
        
        Args:
            bvp_signal: BVP signal array
            labels: Corresponding labels array
            timestamps: Optional timestamps array
            quality_scores: Optional per-sample quality scores
            
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
            
            if quality_scores is not None:
                quality_scores = quality_scores[:min_length]
            
            # Check if signal is long enough for windowing
            if len(bvp_signal) < self.window_size:
                self.logger.warning(f"Signal too short for windowing: {len(bvp_signal)} < {self.window_size}")
                return self._empty_windows_result()
            
            # Create windows
            windows_data = []
            window_positions = []
            window_labels = []
            window_confidences = []
            window_qualities = []
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
                
                # Calculate window quality
                if quality_scores is not None:
                    window_quality = np.mean(quality_scores[start_idx:end_idx])
                else:
                    window_quality = self._estimate_window_quality(window_bvp)
                
                # Store window data
                window_data = {
                    'bvp': window_bvp,
                    'label': window_label,
                    'confidence': label_confidence,
                    'quality': window_quality,
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
                window_qualities.append(window_quality)
                window_timestamps.append((window_ts[0], window_ts[-1]))
                
                self.stats['windows_created'] += 1
            
            # Filter windows based on quality threshold
            accepted_windows = []
            for window_data in windows_data:
                if window_data['quality'] >= self.min_quality:
                    accepted_windows.append(window_data)
                    self.stats['windows_accepted'] += 1
                    
                    # Update label distribution
                    label_name = self.config.get_label_name(window_data['label'])
                    if label_name not in self.stats['label_distribution']:
                        self.stats['label_distribution'][label_name] = 0
                    self.stats['label_distribution'][label_name] += 1
                else:
                    self.stats['windows_rejected'] += 1
            
            # Calculate summary statistics
            summary_stats = self._calculate_window_statistics(accepted_windows)
            
            # Update global statistics
            self.stats['total_signals_processed'] += 1
            if len(window_qualities) > 0:
                self.stats['avg_window_quality'] = np.mean(window_qualities)
            if len(window_confidences) > 0:
                self.stats['avg_label_confidence'] = np.mean(window_confidences)
            
            # Create result dictionary
            result = {
                'windows': accepted_windows,
                'all_windows': windows_data,  # Include rejected windows for analysis
                'window_positions': window_positions,
                'window_labels': np.array(window_labels),
                'window_confidences': np.array(window_confidences),
                'window_qualities': np.array(window_qualities),
                'window_timestamps': window_timestamps,
                'summary_stats': summary_stats,
                'metadata': {
                    'total_windows': len(windows_data),
                    'accepted_windows': len(accepted_windows),
                    'rejected_windows': len(windows_data) - len(accepted_windows),
                    'acceptance_rate': len(accepted_windows) / max(len(windows_data), 1),
                    'window_size_samples': int(self.window_size),
                    'window_size_seconds': self.config.analysis.window_size_seconds,
                    'overlap_samples': int(self.overlap),
                    'overlap_seconds': self.config.analysis.overlap_seconds,
                    'step_size_samples': self.step_size,
                    'signal_length': len(bvp_signal),
                    'signal_duration': len(bvp_signal) / self.sampling_rate
                }
            }
            
            self.logger.info(f"Created {len(accepted_windows)}/{len(windows_data)} windows "
                           f"(acceptance rate: {result['metadata']['acceptance_rate']:.2%})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Window creation failed: {str(e)}")
            return self._empty_windows_result()
    
    def analyze_window_distribution(self, windows_result: Dict) -> Dict:
        """
        Analyze the distribution of windows across different conditions.
        
        Args:
            windows_result: Result from create_windows method
            
        Returns:
            Dictionary containing distribution analysis
        """
        try:
            windows = windows_result.get('windows', [])
            if not windows:
                return {}
            
            # Extract labels and confidences
            labels = [w['label'] for w in windows]
            confidences = [w['confidence'] for w in windows]
            qualities = [w['quality'] for w in windows]
            
            # Count distribution by condition
            label_counts = Counter(labels)
            condition_counts = {}
            condition_qualities = {}
            condition_confidences = {}
            
            for label_id, count in label_counts.items():
                condition_name = self.config.get_label_name(label_id)
                condition_counts[condition_name] = count
                
                # Calculate average quality and confidence for this condition
                condition_windows = [w for w in windows if w['label'] == label_id]
                condition_qualities[condition_name] = np.mean([w['quality'] for w in condition_windows])
                condition_confidences[condition_name] = np.mean([w['confidence'] for w in condition_windows])
            
            # Calculate temporal distribution
            window_times = [(w['start_time'], w['end_time']) for w in windows]
            total_duration = max(w['end_time'] for w in windows) - min(w['start_time'] for w in windows)
            
            distribution_analysis = {
                'condition_counts': condition_counts,
                'condition_percentages': {k: v/len(windows)*100 for k, v in condition_counts.items()},
                'condition_qualities': condition_qualities,
                'condition_confidences': condition_confidences,
                'temporal_info': {
                    'total_windows': len(windows),
                    'total_duration': total_duration,
                    'average_window_duration': self.config.analysis.window_size_seconds,
                    'coverage_ratio': len(windows) * self.config.analysis.window_size_seconds / total_duration
                },
                'quality_stats': {
                    'mean_quality': np.mean(qualities),
                    'std_quality': np.std(qualities),
                    'min_quality': np.min(qualities),
                    'max_quality': np.max(qualities)
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
    
    def extract_window_features(self, windows_result: Dict) -> Dict:
        """
        Extract features from windowed data.
        
        Args:
            windows_result: Result from create_windows method
            
        Returns:
            Dictionary containing extracted features
        """
        try:
            windows = windows_result.get('windows', [])
            if not windows:
                return {}
            
            features_list = []
            labels_list = []
            
            for window in windows:
                bvp_window = window['bvp']
                
                # Time domain features
                time_features = self._extract_time_domain_features(bvp_window)
                
                # Frequency domain features (if enabled)
                freq_features = {}
                if self.config.analysis.enable_frequency_domain:
                    freq_features = self._extract_frequency_domain_features(bvp_window)
                
                # Combine features
                window_features = {
                    **time_features,
                    **freq_features,
                    'quality': window['quality'],
                    'confidence': window['confidence']
                }
                
                features_list.append(window_features)
                labels_list.append(window['label'])
            
            # Convert to arrays for easier processing
            feature_names = list(features_list[0].keys()) if features_list else []
            feature_matrix = np.array([[f[name] for name in feature_names] for f in features_list])
            labels_array = np.array(labels_list)
            
            features_result = {
                'features': feature_matrix,
                'labels': labels_array,
                'feature_names': feature_names,
                'n_windows': len(windows),
                'n_features': len(feature_names),
                'feature_stats': self._calculate_feature_statistics(feature_matrix, feature_names)
            }
            
            return features_result
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            return {}
    
    def _calculate_window_label(self, window_labels: np.ndarray) -> Tuple[int, float]:
        """Calculate window label and confidence from constituent labels."""
        if len(window_labels) == 0:
            return 0, 0.0
        
        # Count label occurrences
        unique_labels, counts = np.unique(window_labels, return_counts=True)
        
        # Most common label becomes window label
        most_common_idx = np.argmax(counts)
        window_label = unique_labels[most_common_idx]
        
        # Confidence is the proportion of the most common label
        confidence = counts[most_common_idx] / len(window_labels)
        
        return int(window_label), float(confidence)
    
    def _estimate_window_quality(self, window_bvp: np.ndarray) -> float:
        """Estimate quality score for a BVP window."""
        if len(window_bvp) == 0:
            return 0.0
        
        try:
            # Simple quality estimation based on signal characteristics
            # This is a simplified version - in practice, use the SignalQuality class
            
            # Signal variance (normalized)
            signal_var = np.var(window_bvp)
            signal_mean = np.mean(np.abs(window_bvp))
            var_score = min(1.0, signal_var / (signal_mean + 1e-8))
            
            # Signal smoothness
            if len(window_bvp) > 1:
                gradient = np.gradient(window_bvp)
                smoothness_score = 1.0 / (1.0 + np.std(gradient))
            else:
                smoothness_score = 0.0
            
            # Combined quality score
            quality_score = 0.6 * var_score + 0.4 * smoothness_score
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _extract_time_domain_features(self, bvp_window: np.ndarray) -> Dict:
        """Extract time domain features from BVP window."""
        if len(bvp_window) == 0:
            return {}
        
        features = {
            'mean': np.mean(bvp_window),
            'std': np.std(bvp_window),
            'var': np.var(bvp_window),
            'min': np.min(bvp_window),
            'max': np.max(bvp_window),
            'range': np.ptp(bvp_window),
            'median': np.median(bvp_window),
            'skewness': self._calculate_skewness(bvp_window),
            'kurtosis': self._calculate_kurtosis(bvp_window)
        }
        
        # Additional features
        if len(bvp_window) > 1:
            features.update({
                'mean_abs_deviation': np.mean(np.abs(bvp_window - np.mean(bvp_window))),
                'rms': np.sqrt(np.mean(bvp_window ** 2)),
                'energy': np.sum(bvp_window ** 2)
            })
        
        return features
    
    def _extract_frequency_domain_features(self, bvp_window: np.ndarray) -> Dict:
        """Extract frequency domain features from BVP window."""
        if len(bvp_window) < 4:  # Need minimum samples for FFT
            return {}
        
        try:
            # Compute FFT
            fft = np.fft.fft(bvp_window)
            freqs = np.fft.fftfreq(len(bvp_window), 1/self.sampling_rate)
            
            # Take only positive frequencies
            pos_freqs = freqs[:len(freqs)//2]
            pos_fft = np.abs(fft[:len(fft)//2])
            
            # Frequency domain features
            features = {
                'spectral_centroid': np.sum(pos_freqs * pos_fft) / (np.sum(pos_fft) + 1e-8),
                'spectral_bandwidth': np.sqrt(np.sum(((pos_freqs - np.sum(pos_freqs * pos_fft) / 
                                                     (np.sum(pos_fft) + 1e-8)) ** 2) * pos_fft) / 
                                             (np.sum(pos_fft) + 1e-8)),
                'spectral_rolloff': self._calculate_spectral_rolloff(pos_freqs, pos_fft),
                'dominant_frequency': pos_freqs[np.argmax(pos_fft)] if len(pos_fft) > 0 else 0
            }
            
            return features
            
        except Exception:
            return {}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) == 0:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) == 0:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_spectral_rolloff(self, freqs: np.ndarray, fft_mag: np.ndarray, 
                                   threshold: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        if len(fft_mag) == 0:
            return 0.0
        
        total_energy = np.sum(fft_mag)
        cumulative_energy = np.cumsum(fft_mag)
        
        rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        else:
            return freqs[-1] if len(freqs) > 0 else 0.0
    
    def _calculate_window_statistics(self, windows: List[Dict]) -> Dict:
        """Calculate summary statistics for windows."""
        if not windows:
            return {}
        
        qualities = [w['quality'] for w in windows]
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
            'avg_quality': np.mean(qualities),
            'std_quality': np.std(qualities),
            'avg_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'condition_distribution': condition_dist,
            'unique_conditions': len(set(labels))
        }
        
        return stats
    
    def _calculate_feature_statistics(self, feature_matrix: np.ndarray, 
                                    feature_names: List[str]) -> Dict:
        """Calculate statistics for extracted features."""
        if feature_matrix.size == 0:
            return {}
        
        stats = {}
        for i, name in enumerate(feature_names):
            feature_data = feature_matrix[:, i]
            stats[name] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'median': np.median(feature_data)
            }
        
        return stats
    
    def _empty_windows_result(self) -> Dict:
        """Return empty windows result for error cases."""
        return {
            'windows': [],
            'all_windows': [],
            'window_positions': [],
            'window_labels': np.array([]),
            'window_confidences': np.array([]),
            'window_qualities': np.array([]),
            'window_timestamps': [],
            'summary_stats': {},
            'metadata': {
                'total_windows': 0,
                'accepted_windows': 0,
                'rejected_windows': 0,
                'acceptance_rate': 0.0,
                'window_size_samples': int(self.window_size),
                'window_size_seconds': self.config.analysis.window_size_seconds,
                'overlap_samples': int(self.overlap),
                'overlap_seconds': self.config.analysis.overlap_seconds,
                'step_size_samples': self.step_size,
                'signal_length': 0,
                'signal_duration': 0.0
            }
        }
    
    def get_windowing_statistics(self) -> Dict:
        """Get windowing analysis statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset windowing analysis statistics."""
        self.stats = {
            'windows_created': 0,
            'windows_accepted': 0,
            'windows_rejected': 0,
            'total_signals_processed': 0,
            'label_distribution': {},
            'avg_window_quality': 0.0,
            'avg_label_confidence': 0.0
        }
        self.logger.debug("Windowing analysis statistics reset")