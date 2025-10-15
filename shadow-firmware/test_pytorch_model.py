#!/usr/bin/env python3
"""
Test script to analyze the PyTorch CNN model (best.pth)
This will help us understand the exact preprocessing and model architecture
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import interpolate
import matplotlib.pyplot as plt
import json

print("=" * 80)
print("Shadow PyTorch Model Analysis")
print("=" * 80)

# ==================== STEP 1: Load the model ====================
print("\n[1] Loading model from best.pth...")

try:
    checkpoint = torch.load('/Users/ashidudissanayake/Dev/Shadow/shadow-firmware/best.pth', 
                           map_location='cpu', weights_only=False)
    print("✓ Model checkpoint loaded successfully")
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    
    # Inspect model architecture
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\n  Model layers:")
        for key, value in state_dict.items():
            print(f"    {key}: {value.shape}")
    
    # Try to load training config if available
    if 'config' in checkpoint:
        print(f"\n  Training config: {checkpoint['config']}")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nPlease ensure:")
    print("  1. best.pth exists in shadow-firmware folder")
    print("  2. The model architecture classes are available")
    exit(1)

# ==================== STEP 2: Generate synthetic sensor data ====================
print("\n[2] Generating synthetic 60-second sensor data...")

# Sampling rates (Hz)
ACC_RATE = 32   # 3-axis accelerometer
BVP_RATE = 64   # Blood volume pulse
EDA_RATE = 4    # Electrodermal activity
TEMP_RATE = 4   # Temperature

WINDOW_DURATION = 60  # seconds

# Generate synthetic data
np.random.seed(42)

# Accelerometer (3 axes) - simulating body movement
acc_samples = ACC_RATE * WINDOW_DURATION  # 1920 samples
acc_x = np.sin(np.linspace(0, 10*np.pi, acc_samples)) * 0.5 + np.random.normal(0, 0.1, acc_samples)
acc_y = np.cos(np.linspace(0, 8*np.pi, acc_samples)) * 0.3 + np.random.normal(0, 0.1, acc_samples)
acc_z = np.ones(acc_samples) * 1.0 + np.random.normal(0, 0.05, acc_samples)  # Gravity + noise

# BVP - simulating heart rate ~70 bpm
bvp_samples = BVP_RATE * WINDOW_DURATION  # 3840 samples
t_bvp = np.linspace(0, WINDOW_DURATION, bvp_samples)
heart_rate = 70 / 60  # Hz
bvp = 50000 + 10000 * np.sin(2*np.pi*heart_rate*t_bvp) + np.random.normal(0, 1000, bvp_samples)
bvp = np.clip(bvp, 0, 262143)  # 18-bit MAX30105 range

# EDA - simulating skin conductance
eda_samples = EDA_RATE * WINDOW_DURATION  # 240 samples
eda = 1.5 + 0.5 * np.sin(np.linspace(0, 2*np.pi, eda_samples)) + np.random.normal(0, 0.1, eda_samples)
eda = np.clip(eda, 0.1, 2.5)  # Realistic EDA voltage range

# Temperature
temp_samples = TEMP_RATE * WINDOW_DURATION  # 240 samples
temp = 36.5 + 0.3 * np.sin(np.linspace(0, np.pi, temp_samples)) + np.random.normal(0, 0.1, temp_samples)

print(f"✓ Synthetic data generated:")
print(f"  ACC:  {acc_samples} samples @ {ACC_RATE}Hz (X: {acc_x.shape}, Y: {acc_y.shape}, Z: {acc_z.shape})")
print(f"  BVP:  {bvp_samples} samples @ {BVP_RATE}Hz")
print(f"  EDA:  {eda_samples} samples @ {EDA_RATE}Hz")
print(f"  TEMP: {temp_samples} samples @ {TEMP_RATE}Hz")

# ==================== STEP 3: Preprocessing Pipeline ====================
print("\n[3] Applying preprocessing pipeline...")

TARGET_RATE = 4  # Resample all to 4Hz
TARGET_SAMPLES = WINDOW_DURATION * TARGET_RATE  # 240 samples

def resample_signal(data, original_rate, target_rate, duration=60):
    """Resample signal using linear interpolation"""
    if original_rate == target_rate:
        return data[:TARGET_SAMPLES]
    
    # Time arrays
    t_old = np.linspace(0, duration, len(data))
    t_new = np.linspace(0, duration, int(duration * target_rate))
    
    # Interpolate
    f = interpolate.interp1d(t_old, data, kind='linear', fill_value='extrapolate')
    resampled = f(t_new)
    
    return resampled[:TARGET_SAMPLES]

def normalize_signal(signal):
    """Z-score normalization"""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-6:
        return signal - mean
    return (signal - mean) / std

# Step 3.1: Compute ACC magnitude
acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
print(f"  ACC magnitude: {acc_mag.shape}")

# Step 3.2: Resample to 4Hz
acc_resampled = resample_signal(acc_mag, ACC_RATE, TARGET_RATE)
bvp_resampled = resample_signal(bvp, BVP_RATE, TARGET_RATE)
eda_resampled = resample_signal(eda, EDA_RATE, TARGET_RATE)
temp_resampled = resample_signal(temp, TEMP_RATE, TARGET_RATE)

print(f"✓ Resampled to {TARGET_RATE}Hz ({TARGET_SAMPLES} samples each):")
print(f"  ACC:  {acc_resampled.shape}")
print(f"  BVP:  {bvp_resampled.shape}")
print(f"  EDA:  {eda_resampled.shape}")
print(f"  TEMP: {temp_resampled.shape}")

# Step 3.3: Normalize each channel
acc_norm = normalize_signal(acc_resampled)
bvp_norm = normalize_signal(bvp_resampled)
eda_norm = normalize_signal(eda_resampled)
temp_norm = normalize_signal(temp_resampled)

print(f"✓ Normalized (z-score):")
print(f"  ACC:  mean={acc_norm.mean():.6f}, std={acc_norm.std():.6f}")
print(f"  BVP:  mean={bvp_norm.mean():.6f}, std={bvp_norm.std():.6f}")
print(f"  EDA:  mean={eda_norm.mean():.6f}, std={eda_norm.std():.6f}")
print(f"  TEMP: mean={temp_norm.mean():.6f}, std={temp_norm.std():.6f}")

# Step 3.4: Stack into (4, 240) tensor
preprocessed = np.stack([acc_norm, bvp_norm, eda_norm, temp_norm], axis=0)
print(f"✓ Stacked tensor shape: {preprocessed.shape} (channels, samples)")

# Step 3.5: Add batch dimension (1, 4, 240)
model_input = np.expand_dims(preprocessed, axis=0)
print(f"✓ Model input shape: {model_input.shape} (batch, channels, samples)")

# ==================== STEP 4: Model Architecture Analysis ====================
print("\n[4] Analyzing model architecture...")

# Try to reconstruct model (you'll need to adapt this based on your actual model)
print("\nNOTE: Model architecture needs to be defined to load weights.")
print("Expected input: (batch, 4, 240)")
print("Expected output: (batch, 1) - stress probability")

# ==================== STEP 5: Save test data for ESP32 validation ====================
print("\n[5] Saving test data for ESP32 validation...")

test_data = {
    'raw_data': {
        'acc_x': acc_x.tolist(),
        'acc_y': acc_y.tolist(),
        'acc_z': acc_z.tolist(),
        'bvp': bvp.tolist(),
        'eda': eda.tolist(),
        'temp': temp.tolist()
    },
    'preprocessed': {
        'acc_resampled': acc_resampled.tolist(),
        'bvp_resampled': bvp_resampled.tolist(),
        'eda_resampled': eda_resampled.tolist(),
        'temp_resampled': temp_resampled.tolist(),
        'acc_normalized': acc_norm.tolist(),
        'bvp_normalized': bvp_norm.tolist(),
        'eda_normalized': eda_norm.tolist(),
        'temp_normalized': temp_norm.tolist()
    },
    'config': {
        'window_duration': WINDOW_DURATION,
        'target_rate': TARGET_RATE,
        'target_samples': TARGET_SAMPLES,
        'sampling_rates': {
            'ACC': ACC_RATE,
            'BVP': BVP_RATE,
            'EDA': EDA_RATE,
            'TEMP': TEMP_RATE
        }
    }
}

with open('test_data_for_esp32.json', 'w') as f:
    json.dump(test_data, f, indent=2)

print("✓ Test data saved to test_data_for_esp32.json")

# ==================== STEP 6: Generate C arrays ====================
print("\n[6] Generating C arrays for ESP32 testing...")

def generate_c_array(data, name, dtype='float'):
    """Generate C array initialization code"""
    lines = [f"const {dtype} {name}[{len(data)}] = {{"]
    for i in range(0, len(data), 8):
        chunk = data[i:i+8]
        values = ', '.join([f'{v:.6f}f' for v in chunk])
        lines.append(f"    {values},")
    lines.append("};")
    return '\n'.join(lines)

c_code = f"""/*
 * Test data for ESP32 preprocessing validation
 * Generated from test_pytorch_model.py
 */

#ifndef TEST_DATA_H
#define TEST_DATA_H

#define TEST_WINDOW_DURATION {WINDOW_DURATION}
#define TEST_TARGET_RATE {TARGET_RATE}
#define TEST_TARGET_SAMPLES {TARGET_SAMPLES}

// Raw sensor data
#define TEST_ACC_SAMPLES {acc_samples}
#define TEST_BVP_SAMPLES {bvp_samples}
#define TEST_EDA_SAMPLES {eda_samples}
#define TEST_TEMP_SAMPLES {temp_samples}

{generate_c_array(acc_x, 'test_acc_x')}

{generate_c_array(acc_y, 'test_acc_y')}

{generate_c_array(acc_z, 'test_acc_z')}

{generate_c_array(bvp, 'test_bvp')}

{generate_c_array(eda, 'test_eda')}

{generate_c_array(temp, 'test_temp')}

// Expected preprocessed output (for validation)
{generate_c_array(acc_norm, 'expected_acc_normalized')}

{generate_c_array(bvp_norm, 'expected_bvp_normalized')}

{generate_c_array(eda_norm, 'expected_eda_normalized')}

{generate_c_array(temp_norm, 'expected_temp_normalized')}

#endif // TEST_DATA_H
"""

with open('test_data.h', 'w') as f:
    f.write(c_code)

print("✓ C header file saved to test_data.h")

# ==================== STEP 7: Plot for visualization ====================
print("\n[7] Creating visualization...")

fig, axes = plt.subplots(4, 2, figsize=(15, 12))
fig.suptitle('Sensor Data Preprocessing Pipeline', fontsize=16)

# ACC
axes[0, 0].plot(acc_mag)
axes[0, 0].set_title(f'ACC Magnitude (Raw {ACC_RATE}Hz)')
axes[0, 0].set_ylabel('g')
axes[0, 1].plot(acc_norm)
axes[0, 1].set_title(f'ACC Normalized ({TARGET_RATE}Hz)')
axes[0, 1].set_ylabel('z-score')

# BVP
axes[1, 0].plot(bvp)
axes[1, 0].set_title(f'BVP (Raw {BVP_RATE}Hz)')
axes[1, 0].set_ylabel('ADC value')
axes[1, 1].plot(bvp_norm)
axes[1, 1].set_title(f'BVP Normalized ({TARGET_RATE}Hz)')
axes[1, 1].set_ylabel('z-score')

# EDA
axes[2, 0].plot(eda)
axes[2, 0].set_title(f'EDA (Raw {EDA_RATE}Hz)')
axes[2, 0].set_ylabel('Voltage (V)')
axes[2, 1].plot(eda_norm)
axes[2, 1].set_title(f'EDA Normalized ({TARGET_RATE}Hz)')
axes[2, 1].set_ylabel('z-score')

# TEMP
axes[3, 0].plot(temp)
axes[3, 0].set_title(f'Temperature (Raw {TEMP_RATE}Hz)')
axes[3, 0].set_ylabel('°C')
axes[3, 0].set_xlabel('Sample')
axes[3, 1].plot(temp_norm)
axes[3, 1].set_title(f'Temperature Normalized ({TARGET_RATE}Hz)')
axes[3, 1].set_ylabel('z-score')
axes[3, 1].set_xlabel('Sample')

plt.tight_layout()
plt.savefig('preprocessing_visualization.png', dpi=150)
print("✓ Visualization saved to preprocessing_visualization.png")

# ==================== SUMMARY ====================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nPreprocessing Steps for ESP32 Implementation:")
print("  1. Collect 60 seconds of raw sensor data:")
print(f"     - ACC (X,Y,Z): {acc_samples} samples @ {ACC_RATE}Hz")
print(f"     - BVP:         {bvp_samples} samples @ {BVP_RATE}Hz")
print(f"     - EDA:         {eda_samples} samples @ {EDA_RATE}Hz")
print(f"     - TEMP:        {temp_samples} samples @ {TEMP_RATE}Hz")
print("\n  2. Compute ACC magnitude: sqrt(x² + y² + z²)")
print(f"\n  3. Resample all signals to {TARGET_RATE}Hz using linear interpolation:")
print(f"     - Target: {TARGET_SAMPLES} samples per channel")
print("\n  4. Apply z-score normalization per channel:")
print("     - normalized = (signal - mean) / std")
print("\n  5. Stack into tensor: (4 channels × 240 samples)")
print("     - Channel order: [ACC, BVP, EDA, TEMP]")
print("\n  6. Pass to CNN model")

print("\nFiles Generated:")
print("  ✓ test_data_for_esp32.json - Full test dataset")
print("  ✓ test_data.h - C arrays for ESP32 validation")
print("  ✓ preprocessing_visualization.png - Visual comparison")

print("\nNext Steps:")
print("  1. Define/load the actual CNN model architecture")
print("  2. Run inference with test data")
print("  3. Convert model to TFLite format")
print("  4. Implement preprocessing in C for ESP32")
print("  5. Validate ESP32 preprocessing matches Python output")

print("\n" + "=" * 80)
