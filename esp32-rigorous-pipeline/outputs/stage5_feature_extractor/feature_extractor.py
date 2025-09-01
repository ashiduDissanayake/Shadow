#!/usr/bin/env python3
"""
Stage 5: Feature Extractor
Bridge from 60-second sensor windows to 30 features for the C model

This module extracts the exact 30 features that the trained model expects:
- BVP features: perm_entropy, l2_iqr, n_sign_changes, l2_peaks
- ACC features: y_perm_entropy, l2_ptp, l2_max, z_peaks, etc.
- EDA features: l2_lineintegral, lineintegral, l2_iqr_5_95, etc.
- TEMP features: min, l2_min, energy, l2_energy, sum

Input: 60-second windows of raw sensor data
Output: 30-feature vector ready for C model
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SensorWindow:
    """Container for 60-second sensor data window"""
    bvp: np.ndarray       # 3840 samples (64 Hz × 60s)
    acc_x: np.ndarray     # 1920 samples (32 Hz × 60s)
    acc_y: np.ndarray     # 1920 samples (32 Hz × 60s)
    acc_z: np.ndarray     # 1920 samples (32 Hz × 60s)
    eda: np.ndarray       # 240 samples (4 Hz × 60s)
    temp: np.ndarray      # 240 samples (4 Hz × 60s)
    timestamp: float      # Window start timestamp

class FeatureExtractor:
    """
    Extract the exact 30 features required by the trained C model
    """
    
    def __init__(self):
        # Expected feature names in exact order
        self.feature_names = [
            "bvp_BVP_perm_entropy",
            "acc_y_perm_entropy", 
            "acc_l2_ptp",
            "acc_l2_max",
            "acc_z_peaks",
            "eda_l2_lineintegral",
            "acc_l2_peaks",
            "acc_z_perm_entropy",
            "acc_y_lineintegral",
            "eda_EDA_lineintegral",
            "temp_TEMP_min",
            "temp_l2_min",
            "acc_z_rms",
            "acc_z_min",
            "acc_z_energy",
            "acc_z_pct_95",
            "acc_z_mean",
            "bvp_l2_iqr",
            "acc_l2_rms",
            "eda_l2_iqr_5_95",
            "acc_y_peaks",
            "bvp_BVP_n_sign_changes",
            "eda_EDA_iqr_5_95",
            "temp_TEMP_energy",
            "temp_l2_energy",
            "acc_l2_min",
            "temp_TEMP_sum",
            "bvp_l2_peaks",
            "eda_l2_min",
            "eda_EDA_max"
        ]
        
        self.sampling_rates = {
            'bvp': 64,
            'acc': 32,
            'eda': 4,
            'temp': 4
        }
        
    def extract_features(self, window: SensorWindow) -> np.ndarray:
        """
        Extract all 30 features from a sensor window
        
        Args:
            window: SensorWindow containing 60 seconds of sensor data
            
        Returns:
            numpy array of 30 features in exact order expected by C model
        """
        features = {}
        
        # Compute derived signals
        acc_l2 = np.sqrt(window.acc_x**2 + window.acc_y**2 + window.acc_z**2)
        bvp_l2 = np.abs(window.bvp)  # L2 norm for single-channel BVP
        eda_l2 = np.abs(window.eda)  # L2 norm for single-channel EDA  
        temp_l2 = np.abs(window.temp)  # L2 norm for single-channel TEMP
        
        # Extract BVP features
        features["bvp_BVP_perm_entropy"] = self._permutation_entropy(window.bvp)
        features["bvp_l2_iqr"] = self._iqr(bvp_l2)
        features["bvp_BVP_n_sign_changes"] = self._n_sign_changes(window.bvp)
        features["bvp_l2_peaks"] = self._count_peaks(bvp_l2)
        
        # Extract ACC features
        features["acc_y_perm_entropy"] = self._permutation_entropy(window.acc_y)
        features["acc_l2_ptp"] = self._peak_to_peak(acc_l2)
        features["acc_l2_max"] = np.max(acc_l2)
        features["acc_z_peaks"] = self._count_peaks(window.acc_z)
        features["acc_l2_peaks"] = self._count_peaks(acc_l2)
        features["acc_z_perm_entropy"] = self._permutation_entropy(window.acc_z)
        features["acc_y_lineintegral"] = self._line_integral(window.acc_y)
        features["acc_z_rms"] = self._rms(window.acc_z)
        features["acc_z_min"] = np.min(window.acc_z)
        features["acc_z_energy"] = self._energy(window.acc_z)
        features["acc_z_pct_95"] = np.percentile(window.acc_z, 95)
        features["acc_z_mean"] = np.mean(window.acc_z)
        features["acc_l2_rms"] = self._rms(acc_l2)
        features["acc_y_peaks"] = self._count_peaks(window.acc_y)
        features["acc_l2_min"] = np.min(acc_l2)
        
        # Extract EDA features
        features["eda_l2_lineintegral"] = self._line_integral(eda_l2)
        features["eda_EDA_lineintegral"] = self._line_integral(window.eda)
        features["eda_l2_iqr_5_95"] = self._iqr_percentile(eda_l2, 5, 95)
        features["eda_EDA_iqr_5_95"] = self._iqr_percentile(window.eda, 5, 95)
        features["eda_l2_min"] = np.min(eda_l2)
        features["eda_EDA_max"] = np.max(window.eda)
        
        # Extract TEMP features
        features["temp_TEMP_min"] = np.min(window.temp)
        features["temp_l2_min"] = np.min(temp_l2)
        features["temp_TEMP_energy"] = self._energy(window.temp)
        features["temp_l2_energy"] = self._energy(temp_l2)
        features["temp_TEMP_sum"] = np.sum(window.temp)
        
        # Return features in exact order
        feature_vector = np.array([features[name] for name in self.feature_names])
        
        # Validate output
        if len(feature_vector) != 30:
            raise ValueError(f"Expected 30 features, got {len(feature_vector)}")
            
        if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
            logger.warning("NaN or Inf values detected in features")
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
            
        return feature_vector.astype(np.float32)
    
    def _permutation_entropy(self, signal: np.ndarray, order: int = 3) -> float:
        """Calculate permutation entropy"""
        try:
            # Create ordinal patterns
            n = len(signal)
            if n < order:
                return 0.0
                
            patterns = []
            for i in range(n - order + 1):
                window = signal[i:i + order]
                pattern = tuple(np.argsort(window))
                patterns.append(pattern)
            
            # Count pattern frequencies
            from collections import Counter
            pattern_counts = Counter(patterns)
            
            # Calculate entropy
            n_patterns = len(patterns)
            entropy = 0.0
            for count in pattern_counts.values():
                p = count / n_patterns
                if p > 0:
                    entropy -= p * np.log2(p)
                    
            return entropy
        except:
            return 0.0
    
    def _iqr(self, signal: np.ndarray) -> float:
        """Calculate interquartile range"""
        return np.percentile(signal, 75) - np.percentile(signal, 25)
    
    def _iqr_percentile(self, signal: np.ndarray, low: float, high: float) -> float:
        """Calculate IQR between custom percentiles"""
        return np.percentile(signal, high) - np.percentile(signal, low)
    
    def _n_sign_changes(self, signal: np.ndarray) -> int:
        """Count number of sign changes"""
        diff = np.diff(signal)
        signs = np.sign(diff)
        sign_changes = np.sum(np.diff(signs) != 0)
        return int(sign_changes)
    
    def _count_peaks(self, signal: np.ndarray) -> int:
        """Count number of peaks"""
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(signal, height=np.mean(signal))
            return len(peaks)
        except:
            # Fallback: simple peak counting
            peaks = 0
            for i in range(1, len(signal) - 1):
                if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                    peaks += 1
            return peaks
    
    def _peak_to_peak(self, signal: np.ndarray) -> float:
        """Calculate peak-to-peak amplitude"""
        return np.max(signal) - np.min(signal)
    
    def _line_integral(self, signal: np.ndarray) -> float:
        """Calculate line integral (total variation)"""
        return np.sum(np.abs(np.diff(signal)))
    
    def _rms(self, signal: np.ndarray) -> float:
        """Calculate root mean square"""
        return np.sqrt(np.mean(signal**2))
    
    def _energy(self, signal: np.ndarray) -> float:
        """Calculate signal energy"""
        return np.sum(signal**2)

def create_sensor_window_from_arrays(
    bvp: np.ndarray,
    acc_x: np.ndarray, 
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    eda: np.ndarray,
    temp: np.ndarray,
    timestamp: float = 0.0
) -> SensorWindow:
    """
    Create a SensorWindow from individual sensor arrays
    
    Args:
        bvp: BVP signal (should be 3840 samples for 60s at 64Hz)
        acc_x, acc_y, acc_z: Accelerometer signals (should be 1920 samples for 60s at 32Hz)
        eda: EDA signal (should be 240 samples for 60s at 4Hz)
        temp: Temperature signal (should be 240 samples for 60s at 4Hz)
        timestamp: Window start timestamp
        
    Returns:
        SensorWindow object
    """
    return SensorWindow(
        bvp=bvp,
        acc_x=acc_x,
        acc_y=acc_y, 
        acc_z=acc_z,
        eda=eda,
        temp=temp,
        timestamp=timestamp
    )

def validate_window_sizes(window: SensorWindow) -> bool:
    """
    Validate that window contains expected number of samples
    
    Returns:
        True if all sizes are correct, False otherwise
    """
    expected_sizes = {
        'bvp': 3840,    # 64 Hz × 60s
        'acc': 1920,    # 32 Hz × 60s  
        'eda': 240,     # 4 Hz × 60s
        'temp': 240     # 4 Hz × 60s
    }
    
    actual_sizes = {
        'bvp': len(window.bvp),
        'acc': len(window.acc_x),  # All acc channels should be same length
        'eda': len(window.eda),
        'temp': len(window.temp)
    }
    
    for sensor, expected in expected_sizes.items():
        actual = actual_sizes[sensor]
        if actual != expected:
            logger.warning(f"{sensor} size mismatch: expected {expected}, got {actual}")
            return False
            
    return True

if __name__ == "__main__":
    # Example usage
    print("🎯 Feature Extractor for 60s → 30 Features")
    print("=" * 50)
    
    # Create example window with correct sizes
    extractor = FeatureExtractor()
    
    # Generate synthetic test data
    window = SensorWindow(
        bvp=np.random.randn(3840),     # 64 Hz × 60s
        acc_x=np.random.randn(1920),   # 32 Hz × 60s
        acc_y=np.random.randn(1920),   # 32 Hz × 60s
        acc_z=np.random.randn(1920),   # 32 Hz × 60s
        eda=np.random.randn(240),      # 4 Hz × 60s
        temp=np.random.randn(240),     # 4 Hz × 60s
        timestamp=0.0
    )
    
    print(f"✅ Window sizes validated: {validate_window_sizes(window)}")
    
    # Extract features
    features = extractor.extract_features(window)
    
    print(f"✅ Extracted {len(features)} features")
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector dtype: {features.dtype}")
    
    # Show first few features
    print(f"\nFirst 5 features:")
    for i in range(5):
        print(f"  {extractor.feature_names[i]}: {features[i]:.6f}")
    
    print(f"\n🚀 Ready to feed into C model!")
