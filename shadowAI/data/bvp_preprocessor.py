"""
BVP Signal Preprocessing Module

This module provides comprehensive preprocessing capabilities for Blood Volume Pulse (BVP) 
signals from wearable devices, specifically optimized for stress detection applications.

Features:
- Advanced signal filtering with artifact removal
- Adaptive segmentation with quality assessment
- Heart Rate Variability (HRV) feature extraction
- Multi-domain feature computation (time, frequency, non-linear)
- Real-time processing capabilities
- Robust noise handling and signal validation

Processing Pipeline:
1. Signal filtering (Butterworth bandpass)
2. Artifact detection and removal
3. Peak detection for heart rate estimation
4. Signal segmentation with overlap
5. HRV feature extraction
6. Quality assessment and validation
7. Normalization and standardization

Author: Shadow AI Team
License: MIT
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration parameters for BVP processing."""
    sampling_rate: int = 64
    window_size_seconds: int = 60
    overlap_seconds: int = 5
    filter_low: float = 0.7
    filter_high: float = 3.7
    filter_order: int = 3
    min_heart_rate: int = 40
    max_heart_rate: int = 200
    quality_threshold: float = 0.6
    enable_artifact_removal: bool = True
    enable_hrv_features: bool = True

class BVPPreprocessor:
    """
    Advanced BVP signal preprocessor for stress detection applications.
    
    This class implements state-of-the-art signal processing techniques
    specifically designed for wearable BVP sensors and stress detection.
    Optimized for real-time processing and deployment on resource-constrained devices.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the BVP preprocessor.
        
        Args:
            config: Processing configuration. Uses default if None.
        """
        self.config = config or ProcessingConfig()
        
        # Derived parameters
        self.window_samples = self.config.window_size_seconds * self.config.sampling_rate
        self.overlap_samples = self.config.overlap_seconds * self.config.sampling_rate
        self.step_samples = self.window_samples - self.overlap_samples
        
        # Design filter once for efficiency
        self._design_filter()
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'segments_created': 0,
            'segments_rejected': 0,
            'avg_quality_score': 0.0
        }
        
        logger.info(f"BVP Preprocessor initialized: "
                   f"Window={self.config.window_size_seconds}s, "
                   f"Overlap={self.config.overlap_seconds}s, "
                   f"Filter={self.config.filter_low}-{self.config.filter_high}Hz")
    
    def process_signal(self, 
                      bvp_signal: np.ndarray,
                      labels: Optional[np.ndarray] = None,
                      subject_id: Optional[int] = None) -> Dict:
        """
        Complete BVP signal processing pipeline.
        
        Args:
            bvp_signal: Raw BVP signal array
            labels: Optional condition labels for each sample
            subject_id: Optional subject identifier for tracking
            
        Returns:
            Dictionary containing processed segments and features:
            {
                'segments': List[np.ndarray],           # Processed signal segments
                'labels': List[int],                    # Segment labels (mode of window)
                'features': List[np.ndarray],           # Extracted HRV features
                'quality_scores': List[float],          # Quality assessment per segment
                'heart_rates': List[float],             # Average heart rate per segment
                'timestamps': List[float],              # Segment start times
                'processing_info': Dict                 # Processing metadata
            }
        """
        logger.info(f"Processing BVP signal: {len(bvp_signal)} samples "
                   f"({len(bvp_signal)/self.config.sampling_rate:.1f} seconds)")
        
        if len(bvp_signal) < self.window_samples:
            logger.warning(f"Signal too short for processing: {len(bvp_signal)} samples "
                         f"(minimum: {self.window_samples})")
            return self._empty_result()
        
        # Step 1: Filter the signal
        filtered_signal = self.filter_signal(bvp_signal)
        
        # Step 2: Artifact detection and removal
        if self.config.enable_artifact_removal:
            clean_signal, artifact_mask = self.remove_artifacts(filtered_signal)
        else:
            clean_signal = filtered_signal
            artifact_mask = np.zeros(len(filtered_signal), dtype=bool)
        
        # Step 3: Segment the signal
        segments_data = self.segment_signal(clean_signal, labels)
        
        # Step 4: Process each segment
        processed_segments = []
        segment_labels = []
        features_list = []
        quality_scores = []
        heart_rates = []
        timestamps = []
        
        for i, segment_info in enumerate(segments_data):
            segment = segment_info['data']
            segment_label = segment_info.get('label', -1)
            start_time = segment_info['start_time']
            
            # Quality assessment
            quality_score = self.assess_segment_quality(segment)
            
            if quality_score >= self.config.quality_threshold:
                # Extract features if enabled
                if self.config.enable_hrv_features:
                    features = self.extract_hrv_features(segment)
                    heart_rate = self.estimate_heart_rate(segment)
                else:
                    features = np.array([])
                    heart_rate = -1.0
                
                processed_segments.append(segment)
                segment_labels.append(segment_label)
                features_list.append(features)
                quality_scores.append(quality_score)
                heart_rates.append(heart_rate)
                timestamps.append(start_time)
                
                self.stats['segments_created'] += 1
            else:
                self.stats['segments_rejected'] += 1
                logger.debug(f"Segment {i} rejected due to low quality: {quality_score:.3f}")
        
        # Update statistics
        self.stats['total_processed'] += 1
        if quality_scores:
            self.stats['avg_quality_score'] = np.mean(quality_scores)
        
        # Processing information
        processing_info = {
            'original_length': len(bvp_signal),
            'filtered_length': len(filtered_signal),
            'total_segments': len(segments_data),
            'accepted_segments': len(processed_segments),
            'rejection_rate': self.stats['segments_rejected'] / max(len(segments_data), 1),
            'avg_quality': self.stats['avg_quality_score'],
            'artifact_percentage': np.mean(artifact_mask) * 100,
            'subject_id': subject_id,
            'config': self.config
        }
        
        logger.info(f"Processing complete: {len(processed_segments)}/{len(segments_data)} segments accepted "
                   f"(avg quality: {self.stats['avg_quality_score']:.3f})")
        
        return {
            'segments': processed_segments,
            'labels': segment_labels,
            'features': features_list,
            'quality_scores': quality_scores,
            'heart_rates': heart_rates,
            'timestamps': timestamps,
            'processing_info': processing_info
        }
    
    def filter_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filtering to remove noise and artifacts.
        
        Args:
            signal_data: Raw BVP signal
            
        Returns:
            Filtered signal
        """
        if len(signal_data) < 3 * self.config.filter_order:
            logger.warning("Signal too short for filtering")
            return signal_data
        
        try:
            # Apply bandpass filter - FIXED: Using signal module function correctly
            filtered = signal.filtfilt(self.filter_b, self.filter_a, signal_data)
            
            # Remove DC component
            filtered = filtered - np.mean(filtered)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Filtering failed: {e}")
            return signal_data
    
    def remove_artifacts(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and remove artifacts using multiple criteria.
        
        Args:
            signal: Filtered BVP signal
            
        Returns:
            Tuple of (clean_signal, artifact_mask)
        """
        # Initialize artifact mask
        artifact_mask = np.zeros(len(signal), dtype=bool)
        
        # 1. Amplitude-based artifact detection
        median_amplitude = np.median(np.abs(signal))
        amplitude_threshold = 5 * median_amplitude
        amplitude_artifacts = np.abs(signal) > amplitude_threshold
        artifact_mask |= amplitude_artifacts
        
        # 2. Gradient-based artifact detection (sudden jumps)
        gradient = np.abs(np.gradient(signal))
        gradient_threshold = np.percentile(gradient, 95)
        gradient_artifacts = gradient > gradient_threshold
        artifact_mask |= gradient_artifacts
        
        # 3. Statistical outlier detection
        z_scores = np.abs((signal - np.mean(signal)) / (np.std(signal) + 1e-8))
        outlier_artifacts = z_scores > 3
        artifact_mask |= outlier_artifacts
        
        # 4. Morphological filtering for isolated spikes
        from scipy.ndimage import median_filter
        window_size = max(3, self.config.sampling_rate // 10)  # 100ms window
        median_filtered = median_filter(signal, size=window_size)
        spike_artifacts = np.abs(signal - median_filtered) > 2 * np.std(signal)
        artifact_mask |= spike_artifacts
        
        # Clean the signal
        clean_signal = signal.copy()
        
        if np.any(artifact_mask):
            # Interpolate over artifacts
            clean_indices = ~artifact_mask
            if np.sum(clean_indices) > 2:
                clean_signal[artifact_mask] = np.interp(
                    np.where(artifact_mask)[0],
                    np.where(clean_indices)[0],
                    signal[clean_indices]
                )
        
        artifact_percentage = np.mean(artifact_mask) * 100
        if artifact_percentage > 20:
            logger.warning(f"High artifact content: {artifact_percentage:.1f}%")
        
        return clean_signal, artifact_mask
    
    def segment_signal(self, signal: np.ndarray, labels: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Segment signal into overlapping windows.
        
        Args:
            signal: Preprocessed BVP signal
            labels: Optional labels for each sample
            
        Returns:
            List of segment dictionaries with data and metadata
        """
        segments = []
        
        # Calculate segment positions
        n_segments = (len(signal) - self.window_samples) // self.step_samples + 1
        
        for i in range(n_segments):
            start_idx = i * self.step_samples
            end_idx = start_idx + self.window_samples
            
            if end_idx <= len(signal):
                segment_data = signal[start_idx:end_idx]
                start_time = start_idx / self.config.sampling_rate
                
                segment_info = {
                    'data': segment_data,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': start_time,
                    'duration': self.config.window_size_seconds,
                    'segment_id': i
                }
                
                # Add label information if available
                if labels is not None and end_idx <= len(labels):
                    segment_labels = labels[start_idx:end_idx]
                    # Use mode of labels in the segment
                    unique_labels, counts = np.unique(segment_labels, return_counts=True)
                    segment_info['label'] = unique_labels[np.argmax(counts)]
                    segment_info['label_confidence'] = np.max(counts) / len(segment_labels)
                
                segments.append(segment_info)
        
        logger.debug(f"Created {len(segments)} segments from signal of {len(signal)} samples")
        return segments
    
    def assess_segment_quality(self, segment: np.ndarray) -> float:
        """
        Assess the quality of a signal segment.
        
        Args:
            segment: Signal segment
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if len(segment) == 0:
            return 0.0
        
        # 1. Signal-to-noise ratio estimate
        signal_power = np.var(segment)
        if signal_power < 1e-10:
            return 0.0
        
        # 2. Periodicity check (BVP should be quasi-periodic)
        try:
            autocorr = np.correlate(segment, segment, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Look for periodic peaks (heart rate range: 40-200 BPM)
            min_period = 60 * self.config.sampling_rate // self.config.max_heart_rate
            max_period = 60 * self.config.sampling_rate // self.config.min_heart_rate
            
            if max_period < len(autocorr):
                periodicity_score = np.max(autocorr[min_period:max_period])
            else:
                periodicity_score = 0.0
        except:
            periodicity_score = 0.0
        
        # 3. Amplitude consistency
        amplitude_std = np.std(segment)
        amplitude_mean = np.mean(np.abs(segment))
        amplitude_score = min(1.0, amplitude_std / (amplitude_mean + 1e-8))
        
        # 4. Gradient smoothness
        gradient = np.gradient(segment)
        gradient_score = 1.0 / (1.0 + np.std(gradient))
        
        # 5. Dynamic range
        dynamic_range = np.ptp(segment)  # Peak-to-peak
        range_score = min(1.0, dynamic_range / (np.std(segment) + 1e-8))
        
        # Combine scores
        scores = [periodicity_score, amplitude_score, gradient_score, range_score]
        quality_score = np.mean([s for s in scores if not np.isnan(s)])
        
        return float(np.clip(quality_score, 0.0, 1.0))
    
    def extract_hrv_features(self, segment: np.ndarray) -> np.ndarray:
        """
        Extract Heart Rate Variability (HRV) features from BVP segment.
        
        Args:
            segment: BVP signal segment
            
        Returns:
            Array of HRV features
        """
        features = []
        
        try:
            # 1. Peak detection for RR intervals
            peaks = self._detect_peaks(segment)
            
            if len(peaks) < 3:
                # Not enough peaks for HRV analysis
                return np.zeros(20)  # Return zero features
            
            # Calculate RR intervals (in milliseconds)
            rr_intervals = np.diff(peaks) / self.config.sampling_rate * 1000
            
            # 2. Time-domain features
            features.extend(self._extract_time_domain_features(rr_intervals))
            
            # 3. Frequency-domain features
            features.extend(self._extract_frequency_domain_features(rr_intervals))
            
            # 4. Non-linear features
            features.extend(self._extract_nonlinear_features(rr_intervals))
            
        except Exception as e:
            logger.debug(f"HRV feature extraction failed: {e}")
            features = [0.0] * 20  # Fallback to zero features
        
        return np.array(features, dtype=np.float32)
    
    def estimate_heart_rate(self, segment: np.ndarray) -> float:
        """
        Estimate heart rate from BVP segment.
        
        Args:
            segment: BVP signal segment
            
        Returns:
            Heart rate in beats per minute
        """
        try:
            peaks = self._detect_peaks(segment)
            
            if len(peaks) < 2:
                return -1.0
            
            # Calculate average RR interval
            rr_intervals = np.diff(peaks) / self.config.sampling_rate
            avg_rr = np.mean(rr_intervals)
            
            # Convert to BPM
            heart_rate = 60.0 / avg_rr
            
            # Validate range
            if self.config.min_heart_rate <= heart_rate <= self.config.max_heart_rate:
                return float(heart_rate)
            else:
                return -1.0
                
        except Exception as e:
            logger.debug(f"Heart rate estimation failed: {e}")
            return -1.0
    
    def normalize_features(self, features_list: List[np.ndarray], method: str = 'zscore') -> List[np.ndarray]:
        """
        Normalize extracted features across segments.
        
        Args:
            features_list: List of feature arrays
            method: Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            List of normalized feature arrays
        """
        if not features_list or len(features_list[0]) == 0:
            return features_list
        
        # Stack features for normalization
        features_matrix = np.vstack(features_list)
        
        if method == 'zscore':
            mean = np.mean(features_matrix, axis=0)
            std = np.std(features_matrix, axis=0) + 1e-8
            normalized_matrix = (features_matrix - mean) / std
            
        elif method == 'minmax':
            min_vals = np.min(features_matrix, axis=0)
            max_vals = np.max(features_matrix, axis=0)
            range_vals = max_vals - min_vals + 1e-8
            normalized_matrix = (features_matrix - min_vals) / range_vals
            
        elif method == 'robust':
            median = np.median(features_matrix, axis=0)
            mad = np.median(np.abs(features_matrix - median), axis=0) + 1e-8
            normalized_matrix = (features_matrix - median) / mad
            
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return features_list
        
        # Convert back to list
        return [normalized_matrix[i] for i in range(len(features_list))]
    
    def get_processing_statistics(self) -> Dict:
        """Get processing statistics and performance metrics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            'total_processed': 0,
            'segments_created': 0,
            'segments_rejected': 0,
            'avg_quality_score': 0.0
        }
    
    def _design_filter(self):
        """Design Butterworth bandpass filter coefficients."""
        nyquist = self.config.sampling_rate / 2
        low = self.config.filter_low / nyquist
        high = self.config.filter_high / nyquist
        
        self.filter_b, self.filter_a = signal.butter(
            self.config.filter_order, [low, high], btype='band'
        )
    
    def _detect_peaks(self, segment: np.ndarray) -> np.ndarray:
        """Detect peaks in BVP signal for heart rate analysis."""
        # Use scipy's find_peaks with appropriate parameters
        min_distance = 60 * self.config.sampling_rate // self.config.max_heart_rate  # Min distance between peaks
        height_threshold = np.std(segment) * 0.5  # Minimum peak height
        
        peaks, _ = signal.find_peaks(
            segment,
            distance=min_distance,
            height=height_threshold,
            prominence=height_threshold * 0.5
        )
        
        return peaks
    
    def _extract_time_domain_features(self, rr_intervals: np.ndarray) -> List[float]:
        """Extract time-domain HRV features."""
        if len(rr_intervals) < 2:
            return [0.0] * 8
        
        # Basic statistics
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        
        # RMSSD (Root Mean Square of Successive Differences)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        
        # SDNN (Standard Deviation of NN intervals)
        sdnn = std_rr
        
        # NN50 (Number of successive RR intervals differing by more than 50ms)
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = nn50 / max(len(rr_intervals) - 1, 1) * 100
        
        # Triangular index approximation
        try:
            hist, _ = np.histogram(rr_intervals, bins=50)
            triangular_index = len(rr_intervals) / max(np.max(hist), 1)
        except:
            triangular_index = 0.0
        
        # Heart rate statistics
        hr_mean = 60000 / mean_rr if mean_rr > 0 else 0
        hr_std = std_rr * 60000 / (mean_rr ** 2) if mean_rr > 0 else 0
        
        return [mean_rr, std_rr, rmssd, sdnn, nn50, pnn50, triangular_index, hr_mean]
    
    def _extract_frequency_domain_features(self, rr_intervals: np.ndarray) -> List[float]:
        """Extract frequency-domain HRV features."""
        if len(rr_intervals) < 10:
            return [0.0] * 8
        
        try:
            # Interpolate RR intervals to regular time grid
            time_original = np.cumsum(rr_intervals) / 1000  # Convert to seconds
            time_regular = np.arange(0, time_original[-1], 1/4)  # 4 Hz sampling
            
            if len(time_regular) < 10:
                return [0.0] * 8
            
            rr_interp = np.interp(time_regular, time_original, rr_intervals)
            
            # Remove trend
            rr_detrended = signal.detrend(rr_interp)
            
            # Calculate PSD using Welch's method
            freqs, psd = signal.welch(rr_detrended, fs=4, nperseg=min(len(rr_detrended), 256))
            
            # Define frequency bands
            vlf_band = (freqs >= 0.003) & (freqs <= 0.04)  # Very Low Frequency
            lf_band = (freqs >= 0.04) & (freqs <= 0.15)    # Low Frequency
            hf_band = (freqs >= 0.15) & (freqs <= 0.4)     # High Frequency
            
            # Calculate power in each band
            vlf_power = np.trapz(psd[vlf_band], freqs[vlf_band])
            lf_power = np.trapz(psd[lf_band], freqs[lf_band])
            hf_power = np.trapz(psd[hf_band], freqs[hf_band])
            
            total_power = vlf_power + lf_power + hf_power
            
            # Normalized powers
            lf_nu = lf_power / (lf_power + hf_power + 1e-8) * 100
            hf_nu = hf_power / (lf_power + hf_power + 1e-8) * 100
            
            # LF/HF ratio
            lf_hf_ratio = lf_power / (hf_power + 1e-8)
            
            # Peak frequency in LF and HF bands
            lf_peak = freqs[lf_band][np.argmax(psd[lf_band])] if np.any(lf_band) else 0
            hf_peak = freqs[hf_band][np.argmax(psd[hf_band])] if np.any(hf_band) else 0
            
            return [vlf_power, lf_power, hf_power, total_power, lf_nu, hf_nu, lf_hf_ratio, lf_peak]
            
        except Exception as e:
            logger.debug(f"Frequency domain analysis failed: {e}")
            return [0.0] * 8
    
    def _extract_nonlinear_features(self, rr_intervals: np.ndarray) -> List[float]:
        """Extract non-linear HRV features."""
        if len(rr_intervals) < 10:
            return [0.0] * 4
        
        try:
            # Sample entropy (approximate)
            def sample_entropy(data, m=2, r=None):
                if r is None:
                    r = 0.2 * np.std(data)
                
                def _maxdist(xi, xj, m):
                    return max([abs(ua - va) for ua, va in zip(xi, xj)])
                
                def _phi(m):
                    patterns = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
                    matches = 0
                    for i in range(len(patterns)):
                        for j in range(i + 1, len(patterns)):
                            if _maxdist(patterns[i], patterns[j], m) <= r:
                                matches += 1
                    return matches
                
                phi_m = _phi(m)
                phi_m_plus_1 = _phi(m + 1)
                
                return -np.log((phi_m_plus_1 + 1e-8) / (phi_m + 1e-8))
            
            # Calculate features
            sampen = sample_entropy(rr_intervals)
            
            # Approximate entropy
            def approximate_entropy(data, m=2, r=None):
                if r is None:
                    r = 0.2 * np.std(data)
                return entropy(np.histogram(data, bins=20)[0] + 1e-8)
            
            apen = approximate_entropy(rr_intervals)
            
            # Detrended fluctuation analysis (simplified)
            def dfa_alpha1(data):
                # Simple implementation of DFA alpha1
                n = len(data)
                y = np.cumsum(data - np.mean(data))
                
                scales = np.logspace(0.7, 1.2, 10).astype(int)
                scales = scales[scales < n//4]
                
                if len(scales) < 3:
                    return 1.0
                
                fluctuations = []
                for scale in scales:
                    n_segments = n // scale
                    if n_segments < 2:
                        continue
                    
                    segments = y[:n_segments * scale].reshape(n_segments, scale)
                    detrended = []
                    
                    for segment in segments:
                        trend = np.polyfit(range(scale), segment, 1)
                        detrended_seg = segment - np.polyval(trend, range(scale))
                        detrended.extend(detrended_seg)
                    
                    fluctuation = np.sqrt(np.mean(np.array(detrended) ** 2))
                    fluctuations.append(fluctuation)
                
                if len(fluctuations) < 3:
                    return 1.0
                
                log_scales = np.log(scales[:len(fluctuations)])
                log_flucts = np.log(fluctuations)
                
                slope, _ = np.polyfit(log_scales, log_flucts, 1)
                return slope
            
            dfa = dfa_alpha1(rr_intervals)
            
            # Correlation dimension (simplified)
            corr_dim = np.log(len(np.unique(rr_intervals)) + 1e-8) / np.log(len(rr_intervals) + 1e-8)
            
            return [sampen, apen, dfa, corr_dim]
            
        except Exception as e:
            logger.debug(f"Non-linear analysis failed: {e}")
            return [0.0] * 4
    
    def _empty_result(self) -> Dict:
        """Return empty result structure for failed processing."""
        return {
            'segments': [],
            'labels': [],
            'features': [],
            'quality_scores': [],
            'heart_rates': [],
            'timestamps': [],
            'processing_info': {
                'original_length': 0,
                'total_segments': 0,
                'accepted_segments': 0,
                'rejection_rate': 1.0,
                'avg_quality': 0.0,
                'artifact_percentage': 0.0
            }
        }