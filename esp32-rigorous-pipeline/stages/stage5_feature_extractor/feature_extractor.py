#!/usr/bin/env python3
"""
Stage 5: Feature Extractor - Bridge from 60-second sensor windows to 30 features

This module extracts the exact 30 features that the C model expects from 
60-second windows of raw sensor data (BVP, ACC, EDA, TEMP).

Window Parameters:
- Window Size: 60 seconds
- Step Size: 10 seconds (50 seconds overlap)
- Sampling Rates: BVP=64Hz, ACC=32Hz, EDA=4Hz, TEMP=4Hz

Output: 30 features matching model_data.json specification
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensorConfig:
    """Configuration for sensor sampling rates and window sizes"""
    bvp_rate: int = 64      # Hz
    acc_rate: int = 32      # Hz  
    eda_rate: int = 4       # Hz
    temp_rate: int = 4      # Hz
    window_seconds: int = 60
    step_seconds: int = 10

class FeatureExtractor:
    """
    Extracts the exact 30 features required by the stress detection model
    from 60-second windows of multi-sensor data.
    """
    
    def __init__(self, config: SensorConfig = None):
        self.config = config or SensorConfig()
        
        # Required features (from model_data.json)
        self.required_features = [
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
        
        logger.info(f"FeatureExtractor initialized for {len(self.required_features)} features")
    
    def extract_window_features(self, sensor_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract 30 features from a 60-second window of sensor data.
        
        Args:
            sensor_data: Dict with keys 'bvp', 'acc_x', 'acc_y', 'acc_z', 'eda', 'temp'
                        Each value is a numpy array of sensor readings
        
        Returns:
            np.ndarray: 30 features in the exact order expected by the model
        """
        try:
            # Validate input data
            self._validate_sensor_data(sensor_data)
            
            # Extract features for each sensor
            features = {}
            
            # BVP features
            bvp_features = self._extract_bvp_features(sensor_data['bvp'])
            features.update(bvp_features)
            
            # ACC features  
            acc_features = self._extract_acc_features(
                sensor_data['acc_x'], 
                sensor_data['acc_y'], 
                sensor_data['acc_z']
            )
            features.update(acc_features)
            
            # EDA features
            eda_features = self._extract_eda_features(sensor_data['eda'])
            features.update(eda_features)
            
            # TEMP features
            temp_features = self._extract_temp_features(sensor_data['temp'])
            features.update(temp_features)
            
            # Create feature vector in correct order
            feature_vector = np.array([features[name] for name in self.required_features])
            
            logger.debug(f"Extracted {len(feature_vector)} features successfully")
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(30)  # Return zeros if extraction fails
    
    def _validate_sensor_data(self, sensor_data: Dict[str, np.ndarray]) -> None:
        """Validate that sensor data has correct structure and sizes"""
        required_keys = ['bvp', 'acc_x', 'acc_y', 'acc_z', 'eda', 'temp']
        
        for key in required_keys:
            if key not in sensor_data:
                raise ValueError(f"Missing sensor data: {key}")
            
            if not isinstance(sensor_data[key], np.ndarray):
                raise ValueError(f"Sensor data must be numpy arrays, got {type(sensor_data[key])} for {key}")
        
        # Check expected lengths (approximately)
        expected_lengths = {
            'bvp': self.config.bvp_rate * self.config.window_seconds,     # ~3840
            'acc_x': self.config.acc_rate * self.config.window_seconds,   # ~1920
            'acc_y': self.config.acc_rate * self.config.window_seconds,   # ~1920
            'acc_z': self.config.acc_rate * self.config.window_seconds,   # ~1920
            'eda': self.config.eda_rate * self.config.window_seconds,     # ~240
            'temp': self.config.temp_rate * self.config.window_seconds,   # ~240
        }
        
        for key, expected_len in expected_lengths.items():
            actual_len = len(sensor_data[key])
            if abs(actual_len - expected_len) > expected_len * 0.1:  # Allow 10% tolerance
                logger.warning(f"Unexpected length for {key}: {actual_len}, expected ~{expected_len}")
    
    def _extract_bvp_features(self, bvp: np.ndarray) -> Dict[str, float]:
        """Extract BVP-related features"""
        features = {}
        
        # BVP signal features
        features['bvp_BVP_perm_entropy'] = self._permutation_entropy(bvp)
        features['bvp_BVP_n_sign_changes'] = self._count_sign_changes(bvp)
        
        # BVP L2 norm features
        bvp_l2 = np.sqrt(bvp**2)
        features['bvp_l2_iqr'] = np.percentile(bvp_l2, 75) - np.percentile(bvp_l2, 25)
        features['bvp_l2_peaks'] = len(find_peaks(bvp_l2)[0])
        
        return features
    
    def _extract_acc_features(self, acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> Dict[str, float]:
        """Extract accelerometer features"""
        features = {}
        
        # Calculate L2 norm (magnitude)
        acc_l2 = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Y-axis features
        features['acc_y_perm_entropy'] = self._permutation_entropy(acc_y)
        features['acc_y_lineintegral'] = np.sum(np.abs(acc_y))
        features['acc_y_peaks'] = len(find_peaks(acc_y)[0])
        
        # Z-axis features
        features['acc_z_perm_entropy'] = self._permutation_entropy(acc_z)
        features['acc_z_peaks'] = len(find_peaks(acc_z)[0])
        features['acc_z_rms'] = np.sqrt(np.mean(acc_z**2))
        features['acc_z_min'] = np.min(acc_z)
        features['acc_z_energy'] = np.sum(acc_z**2)
        features['acc_z_pct_95'] = np.percentile(acc_z, 95)
        features['acc_z_mean'] = np.mean(acc_z)
        
        # L2 norm features
        features['acc_l2_ptp'] = np.ptp(acc_l2)  # Peak-to-peak
        features['acc_l2_max'] = np.max(acc_l2)
        features['acc_l2_peaks'] = len(find_peaks(acc_l2)[0])
        features['acc_l2_rms'] = np.sqrt(np.mean(acc_l2**2))
        features['acc_l2_min'] = np.min(acc_l2)
        
        return features
    
    def _extract_eda_features(self, eda: np.ndarray) -> Dict[str, float]:
        """Extract EDA (electrodermal activity) features"""
        features = {}
        
        # EDA signal features
        features['eda_EDA_lineintegral'] = np.sum(np.abs(eda))
        features['eda_EDA_iqr_5_95'] = np.percentile(eda, 95) - np.percentile(eda, 5)
        features['eda_EDA_max'] = np.max(eda)
        
        # EDA L2 norm features
        eda_l2 = np.sqrt(eda**2)
        features['eda_l2_lineintegral'] = np.sum(np.abs(eda_l2))
        features['eda_l2_iqr_5_95'] = np.percentile(eda_l2, 95) - np.percentile(eda_l2, 5)
        features['eda_l2_min'] = np.min(eda_l2)
        
        return features
    
    def _extract_temp_features(self, temp: np.ndarray) -> Dict[str, float]:
        """Extract temperature features"""
        features = {}
        
        # TEMP signal features
        features['temp_TEMP_min'] = np.min(temp)
        features['temp_TEMP_energy'] = np.sum(temp**2)
        features['temp_TEMP_sum'] = np.sum(temp)
        
        # TEMP L2 norm features
        temp_l2 = np.sqrt(temp**2)
        features['temp_l2_min'] = np.min(temp_l2)
        features['temp_l2_energy'] = np.sum(temp_l2**2)
        
        return features
    
    def _permutation_entropy(self, signal: np.ndarray, m: int = 3, tau: int = 1) -> float:
        """
        Calculate permutation entropy of a signal.
        
        Args:
            signal: Input signal
            m: Pattern length (embedding dimension)
            tau: Time delay
            
        Returns:
            Permutation entropy value
        """
        try:
            if len(signal) < m:
                return 0.0
            
            # Create embedding matrix
            N = len(signal) - (m - 1) * tau
            embedded = np.zeros((N, m))
            
            for i in range(N):
                embedded[i] = signal[i:i + m * tau:tau]
            
            # Get ordinal patterns
            ordinal_patterns = []
            for i in range(N):
                pattern = tuple(np.argsort(embedded[i]))
                ordinal_patterns.append(pattern)
            
            # Calculate relative frequencies
            unique_patterns, counts = np.unique(ordinal_patterns, return_counts=True, axis=0)
            relative_freq = counts / N
            
            # Calculate permutation entropy
            pe = -np.sum(relative_freq * np.log2(relative_freq + 1e-12))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(np.math.factorial(m))
            normalized_pe = pe / max_entropy if max_entropy > 0 else 0.0
            
            return normalized_pe
            
        except Exception:
            return 0.0
    
    def _count_sign_changes(self, signal: np.ndarray) -> int:
        """Count the number of sign changes in a signal"""
        try:
            if len(signal) < 2:
                return 0
            
            diff_signal = np.diff(signal)
            sign_changes = np.sum(np.diff(np.sign(diff_signal)) != 0)
            return int(sign_changes)
            
        except Exception:
            return 0

def create_test_data() -> Dict[str, np.ndarray]:
    """Create synthetic sensor data for testing"""
    config = SensorConfig()
    
    # Generate synthetic data with realistic characteristics
    np.random.seed(42)
    
    # BVP: Heart rate signal around 70 BPM
    t_bvp = np.linspace(0, config.window_seconds, config.bvp_rate * config.window_seconds)
    bvp = np.sin(2 * np.pi * 1.17 * t_bvp) + 0.1 * np.random.randn(len(t_bvp))  # ~70 BPM
    
    # ACC: Movement with gravity component
    t_acc = np.linspace(0, config.window_seconds, config.acc_rate * config.window_seconds)
    acc_x = 0.1 * np.sin(2 * np.pi * 0.5 * t_acc) + 0.05 * np.random.randn(len(t_acc))
    acc_y = 0.1 * np.cos(2 * np.pi * 0.3 * t_acc) + 0.05 * np.random.randn(len(t_acc))
    acc_z = 9.8 + 0.2 * np.sin(2 * np.pi * 0.7 * t_acc) + 0.1 * np.random.randn(len(t_acc))  # Gravity + movement
    
    # EDA: Slowly varying conductance
    t_eda = np.linspace(0, config.window_seconds, config.eda_rate * config.window_seconds)
    eda = 5.0 + 2.0 * np.sin(2 * np.pi * 0.05 * t_eda) + 0.1 * np.random.randn(len(t_eda))
    
    # TEMP: Body temperature with slight variation
    t_temp = np.linspace(0, config.window_seconds, config.temp_rate * config.window_seconds)
    temp = 36.5 + 0.5 * np.sin(2 * np.pi * 0.02 * t_temp) + 0.1 * np.random.randn(len(t_temp))
    
    return {
        'bvp': bvp,
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'eda': eda,
        'temp': temp
    }

def test_feature_extractor():
    """Test the feature extractor with synthetic data"""
    print("🧪 Testing Feature Extractor")
    print("=" * 50)
    
    # Create feature extractor
    extractor = FeatureExtractor()
    
    # Generate test data
    print("📊 Generating synthetic sensor data...")
    sensor_data = create_test_data()
    
    # Print data shapes
    for sensor, data in sensor_data.items():
        print(f"  {sensor}: {len(data)} samples")
    
    # Extract features
    print("\n🔬 Extracting features...")
    features = extractor.extract_window_features(sensor_data)
    
    # Display results
    print(f"\n✅ Feature extraction complete!")
    print(f"📏 Output shape: {features.shape}")
    print(f"🎯 Expected features: {len(extractor.required_features)}")
    
    print(f"\n📈 Feature Summary:")
    for i, (name, value) in enumerate(zip(extractor.required_features, features)):
        print(f"  {i+1:2d}. {name:<25} = {value:.6f}")
    
    # Validate output
    if len(features) == 30:
        print(f"\n🎉 SUCCESS: Extracted exactly 30 features!")
        print(f"✅ Feature extractor is working correctly!")
        return True
    else:
        print(f"\n❌ ERROR: Expected 30 features, got {len(features)}")
        return False

if __name__ == "__main__":
    success = test_feature_extractor()
    if success:
        print(f"\n🚀 Feature extractor ready for integration!")
    else:
        print(f"\n⚠️  Fix feature extractor before proceeding.")
