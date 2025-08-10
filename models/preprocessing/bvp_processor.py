"""
BVP Signal Processor

This module handles preprocessing of Blood Volume Pulse (BVP) signals for stress detection.
"""

import numpy as np
from scipy import signal
from scipy.stats import stats
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class BVPProcessor:
    """
    Processor for BVP (Blood Volume Pulse) signals from wearable devices.
    
    Implements filtering, segmentation, and basic preprocessing steps for stress detection.
    """
    
    def __init__(self, 
                 sampling_rate: int = 64,
                 window_size: int = 60,
                 overlap: int = 5,
                 filter_low: float = 0.7,
                 filter_high: float = 3.7,
                 filter_order: int = 3):
        """
        Initialize BVP processor.
        
        Args:
            sampling_rate: Sampling rate of BVP signal (Hz)
            window_size: Window size for segmentation (seconds)
            overlap: Overlap between windows (seconds)
            filter_low: Lower frequency cutoff for bandpass filter (Hz)
            filter_high: Upper frequency cutoff for bandpass filter (Hz)
            filter_order: Order of Butterworth filter
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_order = filter_order
        
        # Calculate window parameters
        self.window_samples = window_size * sampling_rate
        self.overlap_samples = overlap * sampling_rate
        self.step_samples = self.window_samples - self.overlap_samples
        
    def filter_signal(self, bvp_signal: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to BVP signal.
        
        Args:
            bvp_signal: Raw BVP signal
            
        Returns:
            Filtered BVP signal
        """
        # Design Butterworth bandpass filter
        nyquist = self.sampling_rate / 2
        low = self.filter_low / nyquist
        high = self.filter_high / nyquist
        
        # Ensure frequencies are within valid range
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        
        b, a = signal.butter(self.filter_order, [low, high], btype='band')
        
        # Apply filter
        filtered_signal = signal.filtfilt(b, a, bvp_signal)
        
        logger.debug(f"Applied bandpass filter: {self.filter_low}-{self.filter_high} Hz")
        return filtered_signal
    
    def segment_signal(self, bvp_signal: np.ndarray, labels: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Segment BVP signal into windows with corresponding labels.
        
        Args:
            bvp_signal: BVP signal array
            labels: Corresponding labels array
            
        Returns:
            Tuple of (segments, segment_labels)
        """
        segments = []
        segment_labels = []
        
        # Calculate number of segments
        n_segments = max(1, (len(bvp_signal) - self.window_samples) // self.step_samples + 1)
        
        for i in range(n_segments):
            start_idx = i * self.step_samples
            end_idx = start_idx + self.window_samples
            
            if end_idx > len(bvp_signal):
                break
                
            # Extract segment
            segment = bvp_signal[start_idx:end_idx]
            
            # Get corresponding labels (use majority label in segment)
            segment_label_array = labels[start_idx:end_idx]
            segment_label = stats.mode(segment_label_array, keepdims=False)[0]
            
            segments.append(segment)
            segment_labels.append(int(segment_label))
            
        logger.info(f"Created {len(segments)} segments from {len(bvp_signal)} samples")
        return segments, segment_labels
    
    def normalize_segment(self, segment: np.ndarray, method: str = 'z_score') -> np.ndarray:
        """
        Normalize a single BVP segment.
        
        Args:
            segment: BVP segment to normalize
            method: Normalization method ('z_score', 'min_max', 'robust')
            
        Returns:
            Normalized segment
        """
        if method == 'z_score':
            return (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
        elif method == 'min_max':
            return (segment - np.min(segment)) / (np.max(segment) - np.min(segment) + 1e-8)
        elif method == 'robust':
            median = np.median(segment)
            mad = np.median(np.abs(segment - median))
            return (segment - median) / (mad + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def process_signal(self, 
                      bvp_signal: np.ndarray, 
                      labels: np.ndarray,
                      normalize: bool = True,
                      normalize_method: str = 'z_score') -> Tuple[List[np.ndarray], List[int]]:
        """
        Complete BVP signal processing pipeline.
        
        Args:
            bvp_signal: Raw BVP signal
            labels: Corresponding labels
            normalize: Whether to normalize segments
            normalize_method: Normalization method
            
        Returns:
            Tuple of (processed_segments, segment_labels)
        """
        logger.info(f"Processing BVP signal: {len(bvp_signal)} samples")
        
        # Step 1: Filter signal
        filtered_signal = self.filter_signal(bvp_signal)
        
        # Step 2: Segment signal
        segments, segment_labels = self.segment_signal(filtered_signal, labels)
        
        # Step 3: Normalize segments (optional)
        if normalize:
            normalized_segments = []
            for segment in segments:
                normalized_segment = self.normalize_segment(segment, normalize_method)
                normalized_segments.append(normalized_segment)
            segments = normalized_segments
            
        logger.info(f"Processing complete: {len(segments)} segments")
        return segments, segment_labels
    
    def process_batch(self, 
                     data_dict: dict,
                     normalize: bool = True,
                     normalize_method: str = 'z_score') -> dict:
        """
        Process BVP data for multiple subjects.
        
        Args:
            data_dict: Dictionary with subject data {'subject_id': {'bvp': array, 'labels': array}}
            normalize: Whether to normalize segments
            normalize_method: Normalization method
            
        Returns:
            Dictionary with processed data for each subject
        """
        processed_data = {}
        
        for subject_id, subject_data in data_dict.items():
            logger.info(f"Processing subject {subject_id}")
            
            try:
                bvp_signal = subject_data['bvp']
                labels = subject_data['labels']
                
                segments, segment_labels = self.process_signal(
                    bvp_signal, labels, normalize, normalize_method
                )
                
                processed_data[subject_id] = {
                    'segments': segments,
                    'labels': segment_labels,
                    'sampling_rate': self.sampling_rate,
                    'window_size': self.window_size,
                    'overlap': self.overlap
                }
                
            except Exception as e:
                logger.error(f"Error processing subject {subject_id}: {str(e)}")
                
        return processed_data
    
    def get_segment_info(self) -> dict:
        """
        Get information about segment parameters.
        
        Returns:
            Dictionary with segment information
        """
        return {
            'sampling_rate': self.sampling_rate,
            'window_size_seconds': self.window_size,
            'window_size_samples': self.window_samples,
            'overlap_seconds': self.overlap,
            'overlap_samples': self.overlap_samples,
            'step_samples': self.step_samples,
            'filter_low': self.filter_low,
            'filter_high': self.filter_high,
            'filter_order': self.filter_order
        }
