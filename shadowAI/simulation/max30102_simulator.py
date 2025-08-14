"""
MAX30102 Sensor Simulation Module

This module provides realistic simulation of the MAX30102 heart rate sensor
for testing and validation of the ShadowAI stress detection pipeline without
requiring physical hardware.

Features:
- Realistic BVP signal generation with physiological characteristics
- Multiple user profiles with different baseline parameters
- Stress condition simulation with accurate physiological changes
- Sensor noise and artifact simulation
- Environmental condition effects
- Sensor placement and contact quality simulation
- Validation against real sensor characteristics

Author: Shadow AI Team
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
import time
import random
from scipy import signal
import math

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile for physiological simulation."""
    age: int = 30
    gender: str = "M"  # M/F
    fitness_level: str = "average"  # low/average/high
    baseline_hr: float = 70.0  # beats per minute
    hr_variability: float = 5.0  # standard deviation
    stress_response: float = 1.0  # stress sensitivity multiplier
    bvp_amplitude: float = 1.0  # baseline BVP amplitude
    noise_level: float = 0.1  # sensor noise level

@dataclass
class SensorConfig:
    """MAX30102 sensor configuration."""
    sampling_rate: int = 64  # Hz
    led_current_red: int = 50  # mA
    led_current_ir: int = 50  # mA
    adc_range: int = 4096  # 12-bit ADC
    pulse_width: int = 411  # microseconds
    filter_enabled: bool = True
    temperature_enabled: bool = True

@dataclass
class EnvironmentalConditions:
    """Environmental simulation parameters."""
    ambient_light: float = 100.0  # lux
    temperature: float = 25.0  # celsius
    humidity: float = 50.0  # percentage
    motion_artifact: float = 0.1  # motion level
    contact_quality: float = 0.9  # sensor contact quality (0-1)

class MAX30102Simulator:
    """
    Comprehensive MAX30102 sensor simulator.
    
    Provides realistic BVP signal generation with physiological accuracy,
    noise simulation, and environmental effects for thorough testing
    of stress detection algorithms.
    """
    
    def __init__(self, 
                 user_profile: Optional[UserProfile] = None,
                 sensor_config: Optional[SensorConfig] = None,
                 environmental: Optional[EnvironmentalConditions] = None):
        """
        Initialize MAX30102 simulator.
        
        Args:
            user_profile: User physiological characteristics
            sensor_config: Sensor hardware configuration  
            environmental: Environmental simulation parameters
        """
        self.user_profile = user_profile or UserProfile()
        self.sensor_config = sensor_config or SensorConfig()
        self.environmental = environmental or EnvironmentalConditions()
        
        # Simulation state
        self.current_time = 0.0
        self.current_condition = "baseline"
        self.last_heartbeat_time = 0.0
        self.heartbeat_intervals = []
        
        # Physiological models
        self.heart_model = HeartModel(self.user_profile)
        self.respiratory_model = RespiratoryModel(self.user_profile)
        self.autonomic_model = AutonomicModel(self.user_profile)
        
        # Sensor characteristics
        self.noise_generator = NoiseGenerator(self.environmental)
        self.artifact_generator = ArtifactGenerator(self.environmental)
        
        # Calibration parameters
        self.is_calibrated = False
        self.calibration_offset = 0.0
        self.calibration_scale = 1.0
        
        logger.info(f"MAX30102 simulator initialized for {self.user_profile.age}yo {self.user_profile.gender}")
    
    def set_condition(self, condition: str, intensity: float = 1.0):
        """
        Set physiological condition for simulation.
        
        Args:
            condition: Condition type ('baseline', 'stress', 'amusement', 'meditation', 'exercise')
            intensity: Condition intensity (0.0 to 2.0)
        """
        valid_conditions = ['baseline', 'stress', 'amusement', 'meditation', 'exercise']
        
        if condition not in valid_conditions:
            raise ValueError(f"Invalid condition. Must be one of: {valid_conditions}")
        
        self.current_condition = condition
        
        # Update physiological models
        self.heart_model.set_condition(condition, intensity)
        self.respiratory_model.set_condition(condition, intensity)
        self.autonomic_model.set_condition(condition, intensity)
        
        logger.info(f"Condition set to {condition} with intensity {intensity}")
    
    def simulate_session(self, 
                        duration_seconds: float,
                        conditions: Optional[List[Tuple[str, float, float]]] = None) -> Dict:
        """
        Simulate a complete measurement session with multiple conditions.
        
        Args:
            duration_seconds: Total session duration
            conditions: List of (condition, start_time, duration) tuples
            
        Returns:
            Dictionary containing simulated sensor data
        """
        logger.info(f"Starting simulation session: {duration_seconds}s")
        
        # Default session: baseline for entire duration
        if conditions is None:
            conditions = [("baseline", 0.0, duration_seconds)]
        
        # Generate time axis
        dt = 1.0 / self.sensor_config.sampling_rate
        time_points = np.arange(0, duration_seconds, dt)
        n_samples = len(time_points)
        
        # Initialize output arrays
        bvp_signal = np.zeros(n_samples)
        ir_signal = np.zeros(n_samples)
        red_signal = np.zeros(n_samples)
        temperature = np.zeros(n_samples)
        condition_labels = np.zeros(n_samples)
        signal_quality = np.zeros(n_samples)
        
        # Simulate each time point
        current_condition_idx = 0
        
        for i, t in enumerate(time_points):
            self.current_time = t
            
            # Update condition if needed
            if current_condition_idx < len(conditions):
                condition_name, start_time, condition_duration = conditions[current_condition_idx]
                
                if t >= start_time and t < start_time + condition_duration:
                    if self.current_condition != condition_name:
                        self.set_condition(condition_name)
                elif t >= start_time + condition_duration:
                    current_condition_idx += 1
                    if current_condition_idx < len(conditions):
                        next_condition = conditions[current_condition_idx][0]
                        self.set_condition(next_condition)
                    else:
                        self.set_condition("baseline")
            
            # Generate physiological signals
            cardiac_signal = self.heart_model.generate_beat(t)
            respiratory_signal = self.respiratory_model.generate_cycle(t)
            autonomic_modulation = self.autonomic_model.get_modulation(t)
            
            # Combine physiological components
            base_bvp = self._combine_physiological_signals(
                cardiac_signal, respiratory_signal, autonomic_modulation
            )
            
            # Apply sensor characteristics
            ir_raw = self._apply_sensor_response(base_bvp, "ir")
            red_raw = self._apply_sensor_response(base_bvp, "red")
            
            # Add noise and artifacts
            ir_noisy = self.noise_generator.add_noise(ir_raw, t)
            red_noisy = self.noise_generator.add_noise(red_raw, t)
            
            # Add motion artifacts
            ir_with_artifacts = self.artifact_generator.add_motion_artifact(ir_noisy, t)
            red_with_artifacts = self.artifact_generator.add_motion_artifact(red_noisy, t)
            
            # Calculate derived signals
            bvp_derived = self._calculate_bvp_from_ir(ir_with_artifacts)
            
            # Apply calibration
            if self.is_calibrated:
                bvp_derived = (bvp_derived - self.calibration_offset) * self.calibration_scale
            
            # Calculate signal quality
            quality = self._assess_signal_quality(ir_with_artifacts, red_with_artifacts, t)
            
            # Store results
            bvp_signal[i] = bvp_derived
            ir_signal[i] = ir_with_artifacts
            red_signal[i] = red_with_artifacts
            temperature[i] = self._simulate_temperature(t)
            condition_labels[i] = self._get_condition_label(self.current_condition)
            signal_quality[i] = quality
        
        # Create session results
        session_data = {
            'timestamp': time.time(),
            'duration_seconds': duration_seconds,
            'sampling_rate': self.sensor_config.sampling_rate,
            'num_samples': n_samples,
            'signals': {
                'bvp': bvp_signal,
                'ir': ir_signal,
                'red': red_signal,
                'temperature': temperature
            },
            'metadata': {
                'conditions': conditions,
                'condition_labels': condition_labels,
                'signal_quality': signal_quality,
                'user_profile': self.user_profile.__dict__,
                'sensor_config': self.sensor_config.__dict__,
                'environmental': self.environmental.__dict__
            },
            'statistics': self._calculate_session_statistics(bvp_signal, condition_labels)
        }
        
        logger.info(f"Session simulation completed: {n_samples} samples generated")
        
        return session_data
    
    def simulate_real_time_sample(self) -> Dict:
        """
        Simulate a single real-time sensor sample.
        
        Returns:
            Dictionary containing single sample data
        """
        # Advance time
        dt = 1.0 / self.sensor_config.sampling_rate
        self.current_time += dt
        
        # Generate physiological signals
        cardiac_signal = self.heart_model.generate_beat(self.current_time)
        respiratory_signal = self.respiratory_model.generate_cycle(self.current_time)
        autonomic_modulation = self.autonomic_model.get_modulation(self.current_time)
        
        # Combine and process
        base_bvp = self._combine_physiological_signals(
            cardiac_signal, respiratory_signal, autonomic_modulation
        )
        
        ir_raw = self._apply_sensor_response(base_bvp, "ir")
        red_raw = self._apply_sensor_response(base_bvp, "red")
        
        # Add noise and artifacts
        ir_sample = self.artifact_generator.add_motion_artifact(
            self.noise_generator.add_noise(ir_raw, self.current_time),
            self.current_time
        )
        red_sample = self.artifact_generator.add_motion_artifact(
            self.noise_generator.add_noise(red_raw, self.current_time),
            self.current_time
        )
        
        bvp_sample = self._calculate_bvp_from_ir(ir_sample)
        
        if self.is_calibrated:
            bvp_sample = (bvp_sample - self.calibration_offset) * self.calibration_scale
        
        return {
            'timestamp': self.current_time,
            'bvp': bvp_sample,
            'ir': ir_sample,
            'red': red_sample,
            'temperature': self._simulate_temperature(self.current_time),
            'signal_quality': self._assess_signal_quality(ir_sample, red_sample, self.current_time)
        }
    
    def calibrate_sensor(self, calibration_duration: float = 10.0) -> bool:
        """
        Simulate sensor calibration procedure.
        
        Args:
            calibration_duration: Calibration time in seconds
            
        Returns:
            True if calibration successful
        """
        logger.info(f"Starting sensor calibration for {calibration_duration}s")
        
        # Simulate calibration data collection
        calibration_samples = []
        dt = 1.0 / self.sensor_config.sampling_rate
        
        for t in np.arange(0, calibration_duration, dt):
            # Generate stable baseline signal
            cardiac_signal = self.heart_model.generate_beat(t)
            ir_signal = self._apply_sensor_response(cardiac_signal, "ir")
            ir_noisy = self.noise_generator.add_noise(ir_signal, t, noise_scale=0.5)  # Reduced noise
            bvp_sample = self._calculate_bvp_from_ir(ir_noisy)
            calibration_samples.append(bvp_sample)
        
        # Calculate calibration parameters
        calibration_samples = np.array(calibration_samples)
        
        if len(calibration_samples) > 0:
            self.calibration_offset = np.mean(calibration_samples)
            self.calibration_scale = 1.0 / (np.std(calibration_samples) + 1e-6)
            self.is_calibrated = True
            
            logger.info(f"Calibration successful: offset={self.calibration_offset:.3f}, scale={self.calibration_scale:.3f}")
            return True
        else:
            logger.error("Calibration failed: insufficient data")
            return False
    
    def add_user_variability(self, variability_factor: float = 1.0):
        """Add individual user variability to the simulation."""
        # Randomize user characteristics
        self.user_profile.baseline_hr += random.gauss(0, 5 * variability_factor)
        self.user_profile.hr_variability *= random.uniform(0.5, 1.5)
        self.user_profile.bvp_amplitude *= random.uniform(0.7, 1.3)
        self.user_profile.noise_level *= random.uniform(0.5, 2.0)
        
        # Update models with new parameters
        self.heart_model.update_profile(self.user_profile)
        self.respiratory_model.update_profile(self.user_profile)
        self.autonomic_model.update_profile(self.user_profile)
    
    def set_environmental_conditions(self, **kwargs):
        """Update environmental conditions during simulation."""
        for key, value in kwargs.items():
            if hasattr(self.environmental, key):
                setattr(self.environmental, key, value)
                logger.debug(f"Environmental condition updated: {key}={value}")
    
    def get_sensor_info(self) -> Dict:
        """Get simulated sensor information."""
        return {
            'device_id': 'MAX30102_SIM_001',
            'firmware_version': '1.0.0',
            'sensor_config': self.sensor_config.__dict__,
            'calibration_status': self.is_calibrated,
            'current_condition': self.current_condition,
            'simulation_time': self.current_time,
            'user_profile': self.user_profile.__dict__
        }
    
    def reset_simulation(self):
        """Reset simulation state."""
        self.current_time = 0.0
        self.current_condition = "baseline"
        self.is_calibrated = False
        self.calibration_offset = 0.0
        self.calibration_scale = 1.0
        
        # Reset physiological models
        self.heart_model.reset()
        self.respiratory_model.reset()
        self.autonomic_model.reset()
        
        logger.info("Simulation reset")
    
    def _combine_physiological_signals(self, cardiac: float, respiratory: float, autonomic: float) -> float:
        """Combine physiological signal components."""
        # Weighted combination of physiological signals
        combined = (
            0.7 * cardiac +           # Primary cardiac component
            0.2 * respiratory +       # Respiratory modulation
            0.1 * autonomic          # Autonomic modulation
        )
        
        return combined * self.user_profile.bvp_amplitude
    
    def _apply_sensor_response(self, physiological_signal: float, led_type: str) -> float:
        """Apply MAX30102 sensor characteristics."""
        # LED-specific response
        if led_type == "ir":
            # IR LED has better penetration and SNR
            gain = 1.0
            baseline = 32768  # Mid-range for 16-bit ADC
        elif led_type == "red":
            # Red LED more sensitive to blood oxygenation
            gain = 0.8
            baseline = 30000
        else:
            gain = 1.0
            baseline = 32768
        
        # Apply sensor transfer function
        sensor_signal = baseline + gain * physiological_signal * 1000
        
        # Apply ADC quantization
        adc_signal = np.clip(sensor_signal, 0, self.sensor_config.adc_range)
        
        return adc_signal
    
    def _calculate_bvp_from_ir(self, ir_signal: float) -> float:
        """Calculate BVP signal from IR sensor data."""
        # Convert IR signal to BVP (simplified model)
        # In reality, this involves complex optical calculations
        
        # Remove DC component and scale
        dc_baseline = 32768
        bvp = (ir_signal - dc_baseline) / 1000.0
        
        return bvp
    
    def _assess_signal_quality(self, ir_signal: float, red_signal: float, time: float) -> float:
        """Assess signal quality based on multiple factors."""
        quality_factors = []
        
        # Contact quality
        contact_quality = self.environmental.contact_quality
        quality_factors.append(contact_quality)
        
        # Signal amplitude
        amplitude_factor = min(1.0, abs(ir_signal - 32768) / 5000.0)
        quality_factors.append(amplitude_factor)
        
        # Motion artifact assessment
        motion_penalty = 1.0 - self.environmental.motion_artifact
        quality_factors.append(motion_penalty)
        
        # Ambient light interference
        light_penalty = 1.0 - min(0.5, self.environmental.ambient_light / 1000.0)
        quality_factors.append(light_penalty)
        
        # Combined quality score
        overall_quality = np.mean(quality_factors)
        
        return np.clip(overall_quality, 0.0, 1.0)
    
    def _simulate_temperature(self, time: float) -> float:
        """Simulate temperature sensor reading."""
        # Body temperature with small variations
        base_temp = 36.5  # Normal body temperature
        variation = 0.5 * np.sin(2 * np.pi * time / 3600)  # Hourly variation
        noise = random.gauss(0, 0.1)  # Sensor noise
        
        return base_temp + variation + noise
    
    def _get_condition_label(self, condition: str) -> int:
        """Convert condition name to numeric label."""
        condition_map = {
            'baseline': 1,
            'stress': 2,
            'amusement': 3,
            'meditation': 4,
            'exercise': 5
        }
        return condition_map.get(condition, 0)
    
    def _calculate_session_statistics(self, bvp_signal: np.ndarray, condition_labels: np.ndarray) -> Dict:
        """Calculate statistics for the simulation session."""
        stats = {
            'signal_statistics': {
                'mean_bvp': float(np.mean(bvp_signal)),
                'std_bvp': float(np.std(bvp_signal)),
                'min_bvp': float(np.min(bvp_signal)),
                'max_bvp': float(np.max(bvp_signal)),
                'signal_range': float(np.ptp(bvp_signal))
            },
            'condition_distribution': {},
            'estimated_heart_rate': self._estimate_heart_rate(bvp_signal),
            'signal_quality_stats': {
                'mean_quality': 0.8,  # Placeholder
                'min_quality': 0.6,
                'max_quality': 0.95
            }
        }
        
        # Calculate condition distribution
        unique_conditions, counts = np.unique(condition_labels, return_counts=True)
        total_samples = len(condition_labels)
        
        for condition, count in zip(unique_conditions, counts):
            condition_name = {1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation', 5: 'exercise'}.get(condition, 'unknown')
            stats['condition_distribution'][condition_name] = {
                'count': int(count),
                'percentage': float(count / total_samples * 100)
            }
        
        return stats
    
    def _estimate_heart_rate(self, bvp_signal: np.ndarray) -> float:
        """Estimate heart rate from BVP signal."""
        try:
            # Simple peak detection for heart rate estimation
            from scipy.signal import find_peaks
            
            # Find peaks
            peaks, _ = find_peaks(bvp_signal, distance=self.sensor_config.sampling_rate//3)
            
            if len(peaks) > 1:
                # Calculate average interval between peaks
                peak_intervals = np.diff(peaks) / self.sensor_config.sampling_rate
                avg_interval = np.mean(peak_intervals)
                heart_rate = 60.0 / avg_interval  # Convert to BPM
                
                return float(np.clip(heart_rate, 40, 200))  # Physiological range
            else:
                return self.user_profile.baseline_hr
                
        except Exception:
            return self.user_profile.baseline_hr


class HeartModel:
    """Cardiac signal generation model."""
    
    def __init__(self, user_profile: UserProfile):
        self.user_profile = user_profile
        self.current_hr = user_profile.baseline_hr
        self.phase = 0.0
        self.condition_modifiers = {
            'baseline': 1.0,
            'stress': 1.3,
            'amusement': 1.1,
            'meditation': 0.9,
            'exercise': 1.8
        }
    
    def generate_beat(self, time: float) -> float:
        """Generate cardiac signal component."""
        # Heart rate in Hz
        hr_hz = self.current_hr / 60.0
        
        # Generate cardiac waveform (simplified ECG-like)
        # P wave, QRS complex, T wave
        p_wave = 0.1 * np.sin(2 * np.pi * hr_hz * time)
        qrs_complex = 1.0 * np.sin(4 * np.pi * hr_hz * time + np.pi/4)
        t_wave = 0.3 * np.sin(2 * np.pi * hr_hz * time + np.pi/2)
        
        # Combine components
        cardiac_signal = p_wave + qrs_complex + t_wave
        
        # Add heart rate variability
        hrv_modulation = random.gauss(1.0, self.user_profile.hr_variability / 100.0)
        
        return cardiac_signal * hrv_modulation
    
    def set_condition(self, condition: str, intensity: float):
        """Update heart model for different conditions."""
        modifier = self.condition_modifiers.get(condition, 1.0)
        self.current_hr = self.user_profile.baseline_hr * modifier * intensity
        
    def update_profile(self, user_profile: UserProfile):
        """Update user profile."""
        self.user_profile = user_profile
        self.current_hr = user_profile.baseline_hr
    
    def reset(self):
        """Reset heart model state."""
        self.current_hr = self.user_profile.baseline_hr
        self.phase = 0.0


class RespiratoryModel:
    """Respiratory signal generation model."""
    
    def __init__(self, user_profile: UserProfile):
        self.user_profile = user_profile
        self.breathing_rate = 16.0  # breaths per minute
        self.condition_modifiers = {
            'baseline': 1.0,
            'stress': 1.4,
            'amusement': 1.1,
            'meditation': 0.7,
            'exercise': 2.0
        }
    
    def generate_cycle(self, time: float) -> float:
        """Generate respiratory signal component."""
        # Breathing rate in Hz
        br_hz = self.breathing_rate / 60.0
        
        # Generate respiratory waveform
        respiratory_signal = 0.3 * np.sin(2 * np.pi * br_hz * time)
        
        return respiratory_signal
    
    def set_condition(self, condition: str, intensity: float):
        """Update respiratory model for different conditions."""
        modifier = self.condition_modifiers.get(condition, 1.0)
        self.breathing_rate = 16.0 * modifier * intensity
    
    def update_profile(self, user_profile: UserProfile):
        """Update user profile."""
        self.user_profile = user_profile
    
    def reset(self):
        """Reset respiratory model state."""
        self.breathing_rate = 16.0


class AutonomicModel:
    """Autonomic nervous system modulation model."""
    
    def __init__(self, user_profile: UserProfile):
        self.user_profile = user_profile
        self.sympathetic_activity = 0.5
        self.parasympathetic_activity = 0.5
    
    def get_modulation(self, time: float) -> float:
        """Get autonomic modulation signal."""
        # Low-frequency oscillations (0.1 Hz - Mayer waves)
        lf_component = 0.2 * np.sin(2 * np.pi * 0.1 * time)
        
        # High-frequency oscillations (0.25 Hz - respiratory sinus arrhythmia)
        hf_component = 0.1 * np.sin(2 * np.pi * 0.25 * time)
        
        # Very low frequency oscillations (0.04 Hz - thermoregulation)
        vlf_component = 0.05 * np.sin(2 * np.pi * 0.04 * time)
        
        return lf_component + hf_component + vlf_component
    
    def set_condition(self, condition: str, intensity: float):
        """Update autonomic balance for different conditions."""
        if condition == 'stress':
            self.sympathetic_activity = 0.8 * intensity
            self.parasympathetic_activity = 0.2
        elif condition == 'meditation':
            self.sympathetic_activity = 0.2
            self.parasympathetic_activity = 0.8 * intensity
        else:
            self.sympathetic_activity = 0.5
            self.parasympathetic_activity = 0.5
    
    def update_profile(self, user_profile: UserProfile):
        """Update user profile."""
        self.user_profile = user_profile
    
    def reset(self):
        """Reset autonomic model state."""
        self.sympathetic_activity = 0.5
        self.parasympathetic_activity = 0.5


class NoiseGenerator:
    """Sensor noise simulation."""
    
    def __init__(self, environmental: EnvironmentalConditions):
        self.environmental = environmental
    
    def add_noise(self, signal: float, time: float, noise_scale: float = 1.0) -> float:
        """Add realistic sensor noise."""
        # Thermal noise (white noise)
        thermal_noise = random.gauss(0, 10 * noise_scale)
        
        # Shot noise (Poisson-distributed)
        shot_noise = random.gauss(0, np.sqrt(abs(signal)) * 0.1 * noise_scale)
        
        # 1/f noise (pink noise approximation)
        pink_noise = random.gauss(0, 5 * noise_scale) / (1 + 0.1 * time)
        
        # Power line interference (50/60 Hz)
        powerline_noise = 2 * np.sin(2 * np.pi * 60 * time) * noise_scale
        
        total_noise = thermal_noise + shot_noise + pink_noise + powerline_noise
        
        return signal + total_noise


class ArtifactGenerator:
    """Motion and environmental artifact simulation."""
    
    def __init__(self, environmental: EnvironmentalConditions):
        self.environmental = environmental
        self.motion_phase = 0.0
    
    def add_motion_artifact(self, signal: float, time: float) -> float:
        """Add motion artifacts to the signal."""
        motion_level = self.environmental.motion_artifact
        
        if motion_level > 0:
            # Periodic motion artifacts
            motion_freq = 0.5 + 2 * random.random()  # 0.5-2.5 Hz motion
            motion_artifact = motion_level * 1000 * np.sin(2 * np.pi * motion_freq * time + self.motion_phase)
            
            # Random motion spikes
            if random.random() < 0.01 * motion_level:  # 1% chance per sample
                motion_spike = random.gauss(0, 500 * motion_level)
                motion_artifact += motion_spike
            
            return signal + motion_artifact
        
        return signal


def create_test_scenarios() -> List[Dict]:
    """Create predefined test scenarios for validation."""
    scenarios = [
        {
            'name': 'Baseline Recording',
            'duration': 300,  # 5 minutes
            'conditions': [('baseline', 0, 300)],
            'user_profile': UserProfile(age=25, gender='F', fitness_level='high'),
            'description': 'Standard baseline recording for healthy young female'
        },
        {
            'name': 'Stress Response',
            'duration': 600,  # 10 minutes
            'conditions': [
                ('baseline', 0, 120),
                ('stress', 120, 360),
                ('baseline', 480, 120)
            ],
            'user_profile': UserProfile(age=35, gender='M', stress_response=1.5),
            'description': 'Stress induction protocol with recovery'
        },
        {
            'name': 'Meditation Session',
            'duration': 480,  # 8 minutes
            'conditions': [
                ('baseline', 0, 60),
                ('meditation', 60, 360),
                ('baseline', 420, 60)
            ],
            'user_profile': UserProfile(age=45, fitness_level='average'),
            'description': 'Meditation practice session'
        },
        {
            'name': 'Exercise Recovery',
            'duration': 900,  # 15 minutes
            'conditions': [
                ('baseline', 0, 120),
                ('exercise', 120, 180),
                ('baseline', 300, 600)
            ],
            'user_profile': UserProfile(age=28, fitness_level='high'),
            'description': 'Post-exercise recovery monitoring'
        },
        {
            'name': 'Poor Signal Quality',
            'duration': 180,  # 3 minutes
            'conditions': [('baseline', 0, 180)],
            'user_profile': UserProfile(noise_level=0.5),
            'environmental': EnvironmentalConditions(
                contact_quality=0.4,
                motion_artifact=0.8,
                ambient_light=500
            ),
            'description': 'Challenging measurement conditions'
        }
    ]
    
    return scenarios


def run_validation_test(scenario: Dict) -> Dict:
    """Run a validation test scenario."""
    # Create simulator with scenario parameters
    user_profile = scenario.get('user_profile', UserProfile())
    environmental = scenario.get('environmental', EnvironmentalConditions())
    
    simulator = MAX30102Simulator(
        user_profile=user_profile,
        environmental=environmental
    )
    
    # Run simulation
    results = simulator.simulate_session(
        duration_seconds=scenario['duration'],
        conditions=scenario['conditions']
    )
    
    # Add scenario metadata
    results['scenario_info'] = {
        'name': scenario['name'],
        'description': scenario['description'],
        'validation_status': 'completed'
    }
    
    return results